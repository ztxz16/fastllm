"""Independent BF16 operator checks against PyTorch and upstream PixelUMM.

Build operator_bridge.cpp as a separate shared library linked to fastllm_nomas.
Only this reference test imports torch. Pass/fail uses relative L2, not exact
equality or cosine similarity. Inputs are identical for each implementation.

From the repository root, after building build-nomas/libfastllm_nomas.so:
  c++ -std=c++17 -O2 -shared -fPIC -Iinclude \
      test/hy_nomas/operator_bridge.cpp -Lbuild-nomas -lfastllm_nomas \
      -o build-nomas/operator_bridge.so
  LD_LIBRARY_PATH="$PWD/build-nomas:$LD_LIBRARY_PATH" python3 \
      test/hy_nomas/compare_ops.py --library build-nomas/operator_bridge.so \
      --official /path/to/Hy-Nomas-source --report operators.json
"""
import argparse
import ctypes as C
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F
from compare_block import metrics


@torch.no_grad()
def main(args):
    torch.set_num_threads(2)
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    lib = C.CDLL(str(args.library))
    fp = C.POINTER(C.c_float)
    lib.nomas_test_error.restype = C.c_char_p
    lib.nomas_test_create.restype = C.c_void_p
    lib.nomas_test_destroy.argtypes = [C.c_void_p]
    lib.nomas_test_op.argtypes = [C.c_void_p, C.c_char_p, C.POINTER(C.c_int), fp, fp, fp, fp]
    ctx = lib.nomas_test_create()
    if not ctx:
        raise RuntimeError(lib.nomas_test_error().decode())
    report = dict(precision="bf16", relative_l2_limit=args.l2_limit, cases=[])
    rand = lambda *shape: torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

    def run(name, op, shape, x, y, z, expected):
        arrays = [None if t is None else t.float().cpu().contiguous().numpy() for t in (x,y,z)]
        output = np.empty(expected.numel(), dtype=np.float32)
        ptr = lambda a: None if a is None else a.ctypes.data_as(fp)
        rc = lib.nomas_test_op(ctx,op.encode(),(C.c_int*len(shape))(*shape),
                              *map(ptr, arrays),ptr(output))
        if rc:
            raise RuntimeError(lib.nomas_test_error().decode())
        row = dict(name=name, **metrics(output,expected.float().cpu().numpy()))
        row["passed"] = row["finite"] and row["relative_l2"] <= args.l2_limit
        report["cases"].append(row)
        print(json.dumps(row),flush=True)

    try:
        for m,n,k in [(1,12288,2048),(17,2048,3584),(420,8192,2048),(3360,4096,12288)]:
            x,w = rand(m,k),rand(n,k)
            for bias in (None,rand(n)):
                run(f"linear_{m}_{n}_{k}_bias_{bias is not None}","linear",[m,k,n],x,w,bias,F.linear(x,w,bias))
        for width in (128,2048,4096):
            x,weight,bias = rand(37,width),rand(width),rand(width)
            rms=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)).bfloat16()*weight
            run(f"rms_{width}","rms",list(x.shape),x,weight,None,rms)
            run(f"layernorm_{width}","layernorm",list(x.shape),x,weight,bias,F.layer_norm(x,(width,),weight,bias,1e-6))
        x,y = rand(257,2048),rand(257,2048)
        run("swiglu","swiglu",list(x.shape),x,y,None,F.silu(x)*y)
        for op in ("silu","gelu"):
            expected=F.silu(x) if op=="silu" else F.gelu(x,approximate="tanh")
            run(op,op,list(x.shape),x,None,None,expected)
        shift,scale,gate = rand(2048),rand(2048),rand(2048)
        run("modulate","modulate",list(x.shape),x,torch.cat((shift,scale)),None,x*(1+scale)+shift)
        run("residual","residual",list(x.shape),x,y,gate,x+y*gate)
        phase=torch.randn(257,64,device="cuda").repeat_interleave(2,-1)
        cs,sn=phase.cos(),phase.sin()
        rotated=torch.stack((-x[...,1::2],x[...,::2]),dim=-1).flatten(-2)
        expected=(x.float().reshape(257,16,128)*cs[:,None]+rotated.float().reshape(257,16,128)*sn[:,None]).bfloat16()
        run("dit_rope","rope",list(x.shape),x,cs,sn,expected)
        path=args.official/"hymm/models/vae/pixelumm_decoder/model.py"
        spec=importlib.util.spec_from_file_location("pixel_reference_ops",path)
        pixel=importlib.util.module_from_spec(spec);sys.modules[spec.name]=pixel;spec.loader.exec_module(pixel)
        t,h,w=3,5,7; x=rand(t*h*w,4096)
        nt,nh=pixel.RMSNorm(64,1e-6).cuda().bfloat16(),pixel.RMSNorm(64,1e-6).cuda().bfloat16()
        nt.weight.copy_(rand(64));nh.weight.copy_(rand(64))
        cs,sn=pixel.mrope_tables(pixel.QWEN3_8B_GEN,(t,h,w),"cuda")
        normalized=pixel.GenAttention._split_norm(x.reshape(1,-1,32,128).float(),nt,nh)
        expected=pixel.apply_mrope(normalized,cs,sn,pixel.QWEN3_8B_GEN.rope_sections).bfloat16()
        run("pixel_qk_rope","pixel_qk",[t*h*w,4096,t,h,w],x,nt.weight,nh.weight,expected)
        for nq,nk,heads,kv,causal,tile,h,w in [(17,17,16,16,0,0,1,17),(420,2520,16,16,0,0,1,420),(133,133,28,4,1,0,1,133),(420,420,32,8,0,1,5,7),(3360,3360,32,8,0,8,14,24)]:
            q,k,v=rand(nq,heads,128),rand(nk,kv,128),rand(nk,kv,128)
            mask=None
            if tile:
                ids=torch.arange(nq,device="cuda")
                cy=(ids//w%h//tile).clamp(1,max((h+tile-1)//tile-2,1))
                cx=(ids%w//tile).clamp(1,max((w+tile-1)//tile-2,1))
                mask=((cy[:,None]-ids[None,:]//w%h//tile).abs()<=1)&((cx[:,None]-ids[None,:]%w//tile).abs()<=1)
            expected=F.scaled_dot_product_attention(q.transpose(0,1)[None],k.repeat_interleave(heads//kv,1).transpose(0,1)[None],v.repeat_interleave(heads//kv,1).transpose(0,1)[None],attn_mask=mask,is_causal=bool(causal))[0].transpose(0,1)
            run(f"attention_{nq}_{nk}_{heads}_{kv}_causal{causal}_tile{tile}","attention",[nq,heads*128,nk,heads,kv,causal,tile,h,w],q,k,v,expected)
    finally:
        lib.nomas_test_destroy(ctx)
    report["passed"]=all(c["passed"] for c in report["cases"])
    args.report.write_text(json.dumps(report,indent=2)+"\n")
    if not report["passed"]:
        raise SystemExit("BF16 operator L2 comparison failed")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("library","official","report"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--l2-limit",type=float,default=.01)
    main(p.parse_args())
