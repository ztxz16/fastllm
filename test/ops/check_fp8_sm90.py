"""Independent FP32 reference and CUDA Graph checks; needs PyTorch, CUDA and g++."""
import argparse
import ctypes
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch


def run(lib, m, n, k, dtype, bias_enabled, graph=False, zero=False):
    fn = lib.Fp8Sm90TestLinear
    x = torch.randn(m, k, device="cuda", dtype=dtype) * 0.3
    if zero:
        x.zero_()
    w = torch.randn(n, k, device="cuda").clamp(-3, 3).to(torch.float8_e4m3fn)
    ws = torch.rand(n // 128, k // 128, device="cuda") * 0.1 + 0.002
    bias = torch.randn(n, device="cuda") * 0.02 if bias_enabled else None
    y = torch.full((m, n), float("nan"), device="cuda", dtype=dtype)

    def invoke(rows=m, cols=k, out_cols=n, invalid=0):
        return fn(x.data_ptr(), w.data_ptr(), ws.data_ptr(),
                  bias.data_ptr() if bias is not None else 0, y.data_ptr(),
                  rows, cols, out_cols, dtype == torch.bfloat16, invalid)

    assert invoke(), (m, n, k)
    torch.cuda.synchronize()
    if graph:
        g, executable = ctypes.c_void_p(), ctypes.c_void_p()
        assert lib.FastllmCudaGraphBeginCapture(), lib.FastllmCudaGraphLastError()
        try:
            assert invoke()
            # A new scale cache cannot allocate during capture.
            assert not invoke(invalid=10)
        finally:
            assert lib.FastllmCudaGraphEndCapture(ctypes.byref(g)), lib.FastllmCudaGraphLastError()
        assert lib.FastllmCudaGraphInstantiate(g, ctypes.byref(executable)), lib.FastllmCudaGraphLastError()
        lib.FastllmCudaGraphDestroy(g)
        x.mul_(0.7)
        y.fill_(float("nan"))
        # Grow scratch after capture; the graph must retain valid old pointers.
        gx = torch.ones((257, 512), device="cuda", dtype=dtype)
        gw = torch.ones((128, 512), device="cuda").to(torch.float8_e4m3fn)
        gs = torch.ones((1, 4), device="cuda")
        gy = torch.empty((257, 128), device="cuda", dtype=dtype)
        assert fn(gx.data_ptr(), gw.data_ptr(), gs.data_ptr(), 0, gy.data_ptr(),
                  257, 512, 128, dtype == torch.bfloat16, 0)
        assert lib.FastllmCudaGraphLaunch(executable), lib.FastllmCudaGraphLastError()
        torch.cuda.synchronize()
        lib.FastllmCudaGraphExecDestroy(executable)

    xf = x.float().reshape(m, k // 128, 128)
    xs = (xf.abs().amax(-1, keepdim=True) * (1.0 / 448.0)).clamp_min(1e-10)
    xq = (xf / xs).to(torch.float8_e4m3fn).float() * xs
    wf = w.float() * ws.repeat_interleave(128, 0).repeat_interleave(128, 1)
    reference = (xq.reshape(m, k) @ wf.T).to(dtype).float()
    native_ref = x.float() @ wf.T
    if bias is not None:
        reference = (reference + bias).to(dtype).float()
        native_ref += bias
    diff = y.float() - reference
    nrmse = (diff.square().mean().sqrt() /
             reference.square().mean().sqrt().clamp_min(1e-8)).item()
    max_relative = (diff.abs().max() / reference.abs().max().clamp_min(1e-8)).item()
    quant_nrmse = ((y.float() - native_ref).square().mean().sqrt() /
                   native_ref.square().mean().sqrt().clamp_min(1e-8)).item()
    assert torch.isfinite(y).all(), "nonfinite result or unwritten tail"
    assert nrmse < (0.004 if dtype == torch.bfloat16 else 0.002), nrmse
    assert max_relative < (0.009 if dtype == torch.bfloat16 else 0.006), max_relative
    assert quant_nrmse < 0.04, quant_nrmse

    # Rejected shapes, dtypes and layouts must not write the destination.
    y.fill_(17)
    assert not invoke(rows=1)
    assert not invoke(cols=k - 1)
    assert not invoke(out_cols=n - 1)
    for invalid in range(1, 10):
        if invalid != 7 or bias is not None:
            assert not invoke(invalid=invalid), invalid
    assert torch.all(y == 17)
    print(json.dumps(dict(m=m, n=n, k=k, dtype=str(dtype), bias=bias_enabled,
                          graph=graph, zero=zero, nrmse=nrmse,
                          max_relative=max_relative, quant_nrmse=quant_nrmse)), flush=True)


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib", type=Path,
                        default=root / "build-fastllm/tools/ftllm/libfastllm_tools.so")
    nvcc = shutil.which("nvcc")
    parser.add_argument("--cuda-root", type=Path,
                        default=Path(nvcc).resolve().parents[1] if nvcc else Path("/usr/local/cuda"))
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        print("SKIP: requires SM90")
        return 77
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    with tempfile.TemporaryDirectory(prefix="fastllm-sm90-test-") as tmp:
        bridge = Path(tmp) / "bridge.so"
        subprocess.run([
            "g++", "-std=c++17", "-shared", "-fPIC", "-O2",
            "-I" + str(root / "include"), "-I" + str(args.cuda_root / "include"),
            "-I" + str(root / "third_party/json11"),
            str(Path(__file__).with_name("fp8Sm90TestBridge.cpp")),
            str(args.lib.resolve()), "-Wl,-rpath," + str(args.lib.resolve().parent),
            "-o", str(bridge),
        ], check=True)
        lib = ctypes.CDLL(str(bridge))
        lib.Fp8Sm90TestStream.restype = ctypes.c_void_p
        fn = lib.Fp8Sm90TestLinear
        fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_bool, ctypes.c_int]
        fn.restype = ctypes.c_bool
        # Use FastLLM's capture API: PyTorch's graph helper queries the legacy
        # stream, which CUDA rejects while the per-thread stream is capturing.
        for name, argtypes in (
            ("BeginCapture", []), ("EndCapture", [ctypes.POINTER(ctypes.c_void_p)]),
            ("Instantiate", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]),
            ("Launch", [ctypes.c_void_p]), ("Destroy", [ctypes.c_void_p]),
            ("ExecDestroy", [ctypes.c_void_p]), ("LastError", []),
        ):
            api = getattr(lib, "FastllmCudaGraph" + name)
            api.argtypes = argtypes
            api.restype = (ctypes.c_char_p if name == "LastError" else
                           None if name.endswith("Destroy") else ctypes.c_bool)
        # Match the production operator's per-thread CUDA stream, including capture.
        stream = torch.cuda.ExternalStream(lib.Fp8Sm90TestStream())
        with torch.cuda.stream(stream):
            for dtype in (torch.float16, torch.bfloat16):
                for m, n, k in ((32, 128, 128), (33, 384, 256), (64, 128, 384),
                                (127, 256, 128), (128, 128, 256), (129, 384, 256),
                                (257, 512, 384), (1024, 5120, 5120), (1025, 128, 384)):
                    run(lib, m, n, k, dtype, m % 2 == 1, graph=(m == 129))
                run(lib, 37, 128, 128, dtype, False, zero=True)
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
