"""Hopper BF16-input/FP16-weight lm_head: production dispatch and FP32 oracle."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import torch

ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, default=ROOT/'build-fastllm/tools/ftllm/libfastllm_tools.so')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    records = []
    with tempfile.TemporaryDirectory() as tmp:
        bridge = Path(tmp)/'bridge.so'
        subprocess.run(['g++', '-std=c++17', '-shared', '-fPIC', '-O2',
            '-I'+str(ROOT/'include'), '-I'+str(ROOT/'third_party/json11'), '-I/usr/local/cuda/include',
            str(ROOT/'test/ops/hopperDecodeTestBridge.cpp'), str(args.library),
            '-Wl,-rpath,'+str(args.library.parent), '-o', str(bridge)], check=True)
        lib = C.CDLL(str(bridge))
        lib.HopperTestBf16Fp16.argtypes = [C.c_void_p]*4+[C.c_int]*3
        lib.HopperTestBf16Fp16.restype = C.c_bool
        lib.HopperTestFp16Fallback.argtypes = [C.c_void_p]*4+[C.c_int]*3+[C.c_bool]
        lib.HopperTestFp16Fallback.restype = C.c_bool
        lib.HopperTestSetExactThreshold.argtypes = [C.c_int]
        lib.HopperTestSetExactThreshold.restype = C.c_int
        lib.Fp8Sm90TestStream.restype = C.c_void_p
        for name, params in [('BeginCapture', []), ('EndCapture', [C.POINTER(C.c_void_p)]),
                ('Instantiate', [C.c_void_p, C.POINTER(C.c_void_p)]), ('Launch', [C.c_void_p]),
                ('Destroy', [C.c_void_p]), ('ExecDestroy', [C.c_void_p])]:
            fn = getattr(lib, 'FastllmCudaGraph'+name)
            fn.argtypes = params
            fn.restype = None if name.endswith('Destroy') else C.c_bool
        with torch.cuda.stream(torch.cuda.ExternalStream(lib.Fp8Sm90TestStream())):
            N, K = 248320, 5120
            weight_storage = torch.randn(N*K+8, device='cuda', dtype=torch.float16)*.02
            cases = [('target', 1, N, K, False, 0, 0, 0, torch.bfloat16)]
            cases += [('batch', m, N, K, False, 0, 0, 0, torch.bfloat16) for m in [2,4,7,8,31,32]]
            cases += [(name,1,n,k,b,t,xoff,woff,torch.bfloat16) for name,n,k,b,t,xoff,woff in [
                ('bias',N,K,True,0,0,0), ('exact',N,K,False,2,0,0),
                ('other_N',65536,K,False,0,0,0), ('other_K',N,4096,False,0,0,0),
                ('input_alignment',N,K,False,0,4,0), ('weight_alignment',N,K,False,0,0,4)]]
            cases += [('fp16',m,N,K,False,0,0,0,torch.float16) for m in [1,2,8,31,32]]
            for name,rows,n,k,with_bias,threshold,xoff,woff,dtype in cases:
                w = weight_storage[woff:woff+n*k].view(n,k)
                x = (torch.randn(rows*k+xoff,device='cuda',dtype=dtype)*.3)[xoff:].view(rows,k)
                bias = torch.randn(n,device='cuda',dtype=dtype)*.02 if with_bias else None
                storage = torch.full((rows+1,n),17.,device='cuda',dtype=dtype)
                y = storage[:rows]
                previous = lib.HopperTestSetExactThreshold(threshold)
                def run():
                    params = [x.data_ptr(),w.data_ptr(),bias.data_ptr() if bias is not None else 0,
                              y.data_ptr(),rows,k,n]
                    return lib.HopperTestBf16Fp16(*params) if dtype==torch.bfloat16 else lib.HopperTestFp16Fallback(*params,False)
                try:
                    # Target must also work in a cold graph, with no tuning or allocation.
                    if name != 'target':
                        assert run()
                        torch.cuda.synchronize()
                    graph, executable = C.c_void_p(), C.c_void_p()
                    assert lib.FastllmCudaGraphBeginCapture()
                    assert run()
                    assert lib.FastllmCudaGraphEndCapture(C.byref(graph))
                    assert lib.FastllmCudaGraphInstantiate(graph,C.byref(executable))
                    lib.FastllmCudaGraphDestroy(graph)
                    modes = ['random','changed','zero']
                    if name == 'target': modes = ['random','changed','wide','tiny','zero']
                    for mode in modes:
                        if mode=='changed': x[0].mul_(-.7)
                        if mode=='wide': x.mul_(1e6)
                        if mode=='tiny': x.mul_(1e-16)
                        if mode=='zero': x.zero_()
                        y.fill_(float('nan'))
                        assert lib.FastllmCudaGraphLaunch(executable)
                        torch.cuda.synchronize()
                        ref = x.float()@w.float().T
                        if bias is not None: ref += bias.float()
                        error = y.float()-ref
                        row_error = error.norm(dim=1)/ref.norm(dim=1).clamp_min(1e-30)
                        max_relative = (error.abs().max()/ref.abs().max().clamp_min(1e-30)).item()
                        tolerance = .003 if dtype==torch.bfloat16 and rows<8 else .006
                        assert torch.isfinite(y).all() and row_error.max()<tolerance and max_relative<.015, (name,rows,mode,row_error,max_relative)
                        assert torch.all(storage[rows:]==17), 'output canary overwritten'
                        record = dict(case=name,rows=rows,mode=mode,nrmse=row_error.max().item(),max_relative=max_relative,
                            sha256=hashlib.sha256(y.view(torch.uint8).cpu().numpy().tobytes()).hexdigest())
                        records.append(record)
                        print(json.dumps(record),flush=True)
                    lib.FastllmCudaGraphExecDestroy(executable)
                    if name=='target':
                        x.normal_(0,.3)
                        assert run()
                        begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        begin.record()
                        for _ in range(20): assert run()
                        end.record();end.synchronize()
                        records.append(dict(case='target_timing',ms=begin.elapsed_time(end)/20))
                finally:
                    lib.HopperTestSetExactThreshold(previous)
    args.output.write_text(json.dumps(records,indent=2))
    print('PASS: mixed lm_head, FP32 oracle, cold graph/replay, changed/zero/wide/tiny input, canary and unchanged fallback cases',flush=True)

if __name__=='__main__': main()
