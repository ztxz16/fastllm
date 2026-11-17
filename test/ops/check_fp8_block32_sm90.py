"""SM90 row-wise FP8: independent FP32 reference, Data dispatch and pooled Graph scratch.

Needs PyTorch/CUDA/g++; no model or Triton package is required.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import subprocess
import tempfile
import torch


def quantized_input(x):
    m, k = x.shape
    xf = x.float().view(m, k//32, 32)
    scale = torch.exp2(torch.ceil(torch.log2(xf.abs().amax(-1, keepdim=True).clamp_min(1.e-4)/448.)))
    return ((xf/scale).to(torch.float8_e4m3fn).float()*scale).view(m, k)


def check_output(y, ref):
    rel = ((y.float()-ref).norm()/ref.norm().clamp_min(1.e-20)).item()
    assert torch.isfinite(y).all() and rel < .004, (y.shape, rel)


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lib', type=Path, required=True)
    parser.add_argument('--cuda-root', type=Path, default=Path('/usr/local/cuda'))
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        print('SKIP: requires SM90')
        return 77
    torch.set_num_threads(4)
    torch.manual_seed(32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_stream(torch.cuda.ExternalStream(2))
    with tempfile.TemporaryDirectory(prefix='fastllm-block32-') as tmp:
        bridge = Path(tmp)/'bridge.so'
        subprocess.run(['g++', '-std=c++17', '-shared', '-fPIC', '-O2', '-DUSE_CUDA',
            '-I'+str(root/'include'), '-I'+str(root/'third_party/json11'), '-I'+str(args.cuda_root/'include'),
            str(Path(__file__).with_name('fp8Block32Sm90TestBridge.cpp')), str(args.lib.resolve()),
            '-Wl,-rpath,'+str(args.lib.resolve().parent), '-o', str(bridge)], check=True)
        lib = C.CDLL(str(bridge)); p = C.c_void_p
        fn = lib.Fp8Block32TestLinear
        fn.argtypes = [p]*5+[C.c_int]*4; fn.restype = C.c_bool
        lib.Fp8Block32TestArena.argtypes = [C.c_bool]
        lib.Fp8Block32TestArena(True)
        for name, types in [('FastllmCudaGraphBeginCapture', []), ('FastllmCudaGraphEndCapture', [C.POINTER(p)]),
                ('FastllmCudaGraphInstantiate', [p, C.POINTER(p)]), ('FastllmCudaGraphLaunch', [p])]:
            getattr(lib, name).argtypes = types; getattr(lib, name).restype = C.c_bool
        for name in ['FastllmCudaGraphDestroy', 'FastllmCudaGraphExecDestroy']:
            getattr(lib, name).argtypes = [p]
        shapes = [(320, 384), (4096, 8192), (14848, 4096)]
        sizes = [1, 3, 8, 9, 16, 17, 32, 33, 64, 65, 128, 4096, 4097]
        if args.smoke: shapes, sizes = shapes[:1], [1, 17, 4097]
        graphs = []
        for n, k in shapes:
            w = torch.randn(n, k, device='cuda').to(torch.float8_e4m3fn)
            # General FP32 scales, not only powers of two.
            scales = torch.rand(n, k//32, device='cuda')*.1+.001
            wf = (w.float().view(n, k//32, 32)*scales[..., None]).view(n, k)
            bias = torch.zeros(2*n, device='cuda')  # FP32 bias and its BF16 cache
            for m in sizes:
                x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
                storage = torch.full((m*n+256,), 17., device='cuda', dtype=torch.bfloat16)
                y = storage[:m*n].view(m, n)
                def run(mode=0, inner=k, channels=n):
                    return fn(x.data_ptr(), w.data_ptr(), scales.data_ptr(), bias.data_ptr(),
                              y.data_ptr(), m, inner, channels, mode)
                assert run()
                graph, executable = p(), p()
                assert lib.FastllmCudaGraphBeginCapture()
                assert run()
                assert lib.FastllmCudaGraphEndCapture(C.byref(graph))
                assert lib.FastllmCudaGraphInstantiate(graph, C.byref(executable))
                lib.FastllmCudaGraphDestroy(graph)
                for repeat in range(2):
                    x.normal_() if repeat == 0 else x.zero_()
                    y.fill_(float('nan'))
                    assert lib.FastllmCudaGraphLaunch(executable)
                    check_output(y, quantized_input(x)@wf.T)
                    assert (storage[m*n:]==17).all()
                # Keep the smallest graph alive while later batches grow the
                # shared pool; replay it after all larger allocations.
                if m == sizes[0]: graphs.append((executable, x, w, scales, bias, storage, y))
                else: lib.FastllmCudaGraphExecDestroy(executable)
                y.fill_(17)
                assert not run(mode=1, inner=k-1)
                assert not run(mode=1, channels=n-1)
                assert (y==17).all()
                print(json.dumps(dict(m=m, n=n, k=k, status='pass')), flush=True)
            # Bias fallback with zero input has an exact, independent answer.
            lib.Fp8Block32TestArena(False)
            x.zero_(); bias[:n].fill_(.5)
            bias[n:].view(torch.bfloat16)[:n].fill_(.5)
            assert run(mode=2) and (y==.5).all()
            lib.Fp8Block32TestArena(True)
        for executable, x, w, scales, bias, storage, y in graphs:
            x.normal_(); y.fill_(float('nan'))
            n, k = w.shape
            wf = (w.float().view(n, k//32, 32)*scales[..., None]).view(n, k)
            ref = quantized_input(x)@wf.T
            torch.cuda.synchronize(); free = torch.cuda.mem_get_info()[0]
            for _ in range(10): assert lib.FastllmCudaGraphLaunch(executable)
            torch.cuda.synchronize()
            assert torch.cuda.mem_get_info()[0] == free
            check_output(y, ref)
            assert (storage[y.numel():]==17).all()
            lib.FastllmCudaGraphExecDestroy(executable)
        lib.Fp8Block32TestArena(False)
        print('PASS: Data dispatch, FP32 scales, tails, fallback, pooled Graph lifetime')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
