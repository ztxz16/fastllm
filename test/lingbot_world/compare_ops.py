"""Same-input operator checks; PyTorch is a validation dependency only."""
import argparse
import ctypes
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bridge', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--attention', choices=['flash', 'vllm-fa2'], default='flash')
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(2801)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    lib = ctypes.CDLL(str(args.bridge))
    pointer = ctypes.POINTER(ctypes.c_float)
    lib.lingbot_test_error.restype = ctypes.c_char_p
    lib.lingbot_test_norm.argtypes = [pointer]*4 + [ctypes.c_int]*3
    lib.lingbot_test_activation.argtypes = [pointer]*2 + [ctypes.c_int]*4
    lib.lingbot_test_linear.argtypes = [pointer]*4 + [ctypes.c_int]*4
    lib.lingbot_test_attention.argtypes = [pointer]*4 + [ctypes.c_int]*2
    lib.lingbot_test_patch.argtypes = [pointer]*4 + [ctypes.c_int]*3
    lib.lingbot_test_image.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                     pointer, ctypes.c_int, ctypes.c_int]
    def cpu(x):
        return x.detach().float().contiguous().cpu().numpy()
    def ptr(x):
        return None if x is None else x.ctypes.data_as(pointer)
    def call(fn, *arguments):
        if fn(*arguments):
            raise RuntimeError(lib.lingbot_test_error().decode())
    results = []
    def record(name, actual, expected, gate):
        expected = cpu(expected)
        delta = actual.astype(np.float64)-expected
        l2 = float(np.linalg.norm(delta)/max(np.linalg.norm(expected.astype(np.float64)), 1e-12))
        result = dict(name=name, relative_l2=l2, max_abs=float(np.abs(delta).max()),
                      different=int(np.count_nonzero(delta)), elements=int(delta.size),
                      gate=gate, passed=bool(np.isfinite(actual).all() and l2 < gate))
        results.append(result)
        print(result, flush=True)
    inputs = [('random', torch.randn(48, 1536, device='cuda')),
              ('offset', torch.randn(17, 1536, device='cuda')*.1+4)]
    weight = torch.randn(1536, device='cuda')*.1+1
    bias = torch.randn(1536, device='cuda')*.1
    for name, original in inputs:
        for full in [False, True]:
            x = original.to(torch.float32 if full else torch.bfloat16)
            a, w, b = cpu(x), cpu(weight), cpu(bias)
            out = np.empty_like(a)
            for affine in [False, True]:
                expected = F.layer_norm(x.float(), (1536,), weight if affine else None,
                                        bias if affine else None, 1e-6).to(x.dtype)
                call(lib.lingbot_test_norm, ptr(a), ptr(w) if affine else None,
                     ptr(b) if affine else None, ptr(out), len(a), full, False)
                record(f'{name}/layernorm/full={full}/affine={affine}', out, expected,
                       2e-6 if full else 5e-4)
            if not full:
                expected = (x.float()*torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True)+1e-6)).to(x.dtype)*weight
                call(lib.lingbot_test_norm, ptr(a), ptr(w), None, ptr(out), len(a), False, True)
                record(f'{name}/rms', out, expected, 5e-4)
    for full in [False, True]:
        x = torch.randn(48, 8960, device='cuda').to(torch.float32 if full else torch.bfloat16)*3
        a, out = cpu(x), np.empty(x.shape, dtype=np.float32)
        for gelu in [False, True]:
            expected = F.gelu(x, approximate='tanh') if gelu else F.silu(x)
            call(lib.lingbot_test_activation, ptr(a), ptr(out), *a.shape, full, gelu)
            record(f'activation/full={full}/gelu={gelu}', out, expected, 2e-6 if full else 5e-4)
    for full, rows, inner, outer in [(True, 48, 256, 1536), (True, 48, 1536, 9216),
                                    (False, 48, 1536, 1536), (False, 180, 1536, 8960)]:
        dtype = torch.float32 if full else torch.bfloat16
        x = torch.randn(rows, inner, device='cuda').to(dtype)
        w = (torch.randn(outer, inner, device='cuda')/inner**.5).to(dtype)
        b = torch.randn(outer, device='cuda').to(dtype)
        a, weights, bias_array = cpu(x), cpu(w), cpu(b)
        out = np.empty((rows, outer), dtype=np.float32)
        call(lib.lingbot_test_linear, ptr(a), ptr(weights), ptr(bias_array), ptr(out),
             rows, inner, outer, full)
        record(f'linear/{rows}x{inner}x{outer}/full={full}', out, F.linear(x, w, b),
               2e-6 if full else 5e-4)
    if args.attention == 'vllm-fa2':
        from vllm.vllm_flash_attn import flash_attn_varlen_func
    else:
        from flash_attn import flash_attn_varlen_func
    for rows, count in [(1,1), (17,48), (48,64), (48,65), (129,180), (180,512), (180,1080)]:
        q = torch.randn(rows,12,128,device='cuda').bfloat16()
        k = torch.randn(count,12,128,device='cuda').bfloat16()
        v = torch.randn_like(k)
        expected = flash_attn_varlen_func(q=q,k=k,v=v,
            cu_seqlens_q=torch.tensor([0,rows],device='cuda',dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0,count],device='cuda',dtype=torch.int32),
            max_seqlen_q=rows,max_seqlen_k=count)
        queries, keys, values = map(cpu, [q,k,v])
        out = np.empty_like(queries)
        call(lib.lingbot_test_attention, ptr(queries),ptr(keys),ptr(values),ptr(out),rows,count)
        record(f'attention/{rows}x{count}',out,expected,5e-4)
    for height, width in [(8,8), (12,20), (16,16)]:
        x = torch.randn(3,height,width,36,device='cuda').bfloat16()
        weights = (torch.randn(1536,36,1,2,2,device='cuda')/12).bfloat16()
        bias = torch.randn(1536,device='cuda').bfloat16()
        expected = F.conv3d(x.permute(3,0,1,2)[None].contiguous(),weights,bias,stride=(1,2,2))
        expected = expected.flatten(2).transpose(1,2).reshape(-1,1536)
        a,b,c = map(cpu,[x,weights,bias])
        out = np.empty(expected.shape,dtype=np.float32)
        call(lib.lingbot_test_patch,ptr(a),ptr(b),ptr(c),ptr(out),3,height,width)
        record(f'patch_conv/3x{height}x{width}',out,expected,5e-4)
    # Official image2video.py normalizes on the CPU, then uses FP32 bicubic
    # interpolate with align_corners=False and antialias=False.
    rng = np.random.default_rng(811)
    for sh, sw, h, w in [(96,160,96,160), (137,211,96,160), (29,37,96,160),
                         (512,768,320,576), (1,1,32,48), (7,1,48,32)]:
        rgba = rng.integers(0, 256, (sh, sw, 4), dtype=np.uint8)
        x = torch.from_numpy(rgba[:,:,:3].copy()).permute(2,0,1).float()/255
        expected = F.interpolate(((x-.5)/.5)[None], size=(h,w), mode='bicubic',
                                 align_corners=False)[0].permute(1,2,0)
        out = np.empty((h,w,3), dtype=np.float32)
        call(lib.lingbot_test_image, rgba.ctypes.data_as(ctypes.c_void_p), sh, sw, ptr(out), h, w)
        record(f'image/{sh}x{sw}_to_{h}x{w}', out, expected, 1e-5)
    report = dict(attention=args.attention, passed=all(r['passed'] for r in results), tests=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
