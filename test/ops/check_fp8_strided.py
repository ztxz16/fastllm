"""Check the AOT block128 GEMM against a CPU reference, including strided tails.

Requires CUDA, PyTorch and a Python environment with Triton. Example:
    python check_fp8_strided.py --compiler-python /usr/bin/python3
The integer/power-of-two cases require exact equality. Random cases check
normalized error because FP8 Tensor Core accumulation can differ from FP64.
"""
import argparse
import ctypes as ct
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import torch


class Driver:
    def __init__(self):
        self.api = ct.CDLL("libcuda.so.1")
        self.check(self.api.cuInit(0))
        self.modules = []

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f"CUDA driver error {code}")

    def load(self, meta):
        kernel = meta["kernels"]["matmul"]
        module, function = ct.c_void_p(), ct.c_void_p()
        self.check(self.api.cuModuleLoad(ct.byref(module), kernel["cubin"].encode()))
        self.modules.append(module)
        self.check(self.api.cuModuleGetFunction(
            ct.byref(function), module, kernel["kernel"].encode()))
        if kernel["shared"] > 49152:
            self.check(self.api.cuFuncSetAttribute(function, 8, kernel["shared"]))
        return function

    def launcher(self, function, meta, a, b, out, asc, bsc):
        m, k = a.shape
        n = b.shape[0]
        # Match the production AOT ABI, including both hidden scratch pointers.
        values = [ct.c_uint64(x.data_ptr()) for x in (a, b, out, asc, bsc)]
        values += [ct.c_int32(x) for x in (
            m, n, k, 128, 128, a.stride(0), b.stride(0), out.stride(0),
            asc.stride(0), bsc.stride(0))]
        values += [ct.c_uint64(0), ct.c_uint64(0)]
        pointers = (ct.c_void_p * len(values))(
            *[ct.cast(ct.pointer(x), ct.c_void_p) for x in values])
        blocks = math.ceil(m / meta["block_m"]) * math.ceil(n / meta["block_n"])
        kernel = meta["kernels"]["matmul"]

        def run():
            self.check(self.api.cuLaunchKernel(
                function, blocks, 1, 1, kernel["num_warps"] * 32, 1, 1,
                kernel["shared"], ct.c_void_p(torch.cuda.current_stream().cuda_stream),
                pointers, None))

        run.arguments = values
        return run

    def close(self):
        for module in reversed(self.modules):
            self.check(self.api.cuModuleUnload(module))
        self.modules.clear()


def compile_kernel(python, server, payload):
    code = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location('compiler_server', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(json.dumps(module.compile_linear_fp8_block128(json.load(sys.stdin))))
"""
    result = subprocess.run(
        [python, "-c", code, str(server)], input=json.dumps(payload),
        capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return json.loads(result.stdout)


def run_case(driver, function, meta, shape, dtype, pattern, padded):
    m, n, k = shape
    device = torch.cuda.current_device()
    torch.manual_seed(m * 7 + n * 11 + k)
    a = torch.randn((m, k + 128 * padded), device=device) * 4
    b = torch.randn((n, k + 128 * padded), device=device) * 4
    if pattern == "integer":
        a.round_().clamp_(-4, 4)
        b.round_().clamp_(-4, 4)
    elif pattern == "zero":
        a.zero_()
    a = a.to(torch.float8_e4m3fn)[:, :k]
    b = b.to(torch.float8_e4m3fn)[:, :k]
    groups = math.ceil(k / 128)
    asc = torch.exp2(torch.randint(-6, 1, (m, groups + 3 * padded),
                                  device=device).float())[:, :groups]
    bsc = torch.exp2(torch.randint(-6, 1, (math.ceil(n / 128), groups + 5 * padded),
                                  device=device).float())[:, :groups]
    width = math.ceil((n + 16) / 8) * 8
    storage = torch.full((m + 2, width), 1.75, dtype=dtype, device=device)
    out = storage[1:m + 1, :n]
    out.fill_(float("nan"))
    launch = driver.launcher(function, meta, a, b, out, asc, bsc)
    launch()
    if m == 65:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
            launch()
        bsc.mul_(2)
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        graph.reset()
    torch.cuda.synchronize()

    # Independent CPU computation: dequantize whole matrices, then FP64 GEMM.
    aq = a.cpu().double()
    bq = b.cpu().double()
    ascale = asc.cpu().double().repeat_interleave(128, 1)[:, :k]
    bscale = bsc.cpu().double().repeat_interleave(128, 0)[:n]
    bscale = bscale.repeat_interleave(128, 1)[:, :k]
    reference = ((aq * ascale) @ (bq * bscale).T).to(dtype).float()
    actual = out.cpu().float()
    assert torch.isfinite(actual).all(), "nonfinite output or unwritten tail"
    diff = actual - reference
    nrmse = (diff.square().mean().sqrt() /
             reference.square().mean().sqrt().clamp_min(1e-8)).item()
    max_relative = (diff.abs().max() / reference.abs().max().clamp_min(1e-8)).item()
    if pattern in ("integer", "zero"):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    else:
        assert nrmse < (0.004 if dtype == torch.bfloat16 else 0.002), nrmse
        assert max_relative < (0.009 if dtype == torch.bfloat16 else 0.006), max_relative
    assert torch.all(storage[0] == 1.75) and torch.all(storage[-1] == 1.75)
    assert torch.all(storage[1:-1, n:] == 1.75), "output padding was overwritten"
    print(json.dumps(dict(device=device, dtype=str(dtype), shape=shape,
                          pattern=pattern, padded=padded, graph=m == 65,
                          nrmse=nrmse, max_relative=max_relative)), flush=True)


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-python", default=sys.executable)
    parser.add_argument("--devices", nargs="+", type=int)
    parser.add_argument("--tile", nargs=3, type=int, default=(64, 128, 32),
                        metavar=("BLOCK_M", "BLOCK_N", "GROUP_M"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("SKIP: CUDA is unavailable")
        return 77
    torch.set_num_threads(4)
    devices = args.devices if args.devices is not None else range(torch.cuda.device_count())
    tested = False
    with tempfile.TemporaryDirectory(prefix="fastllm-fp8-strided-test-") as cache:
        for device in devices:
            major, minor = torch.cuda.get_device_capability(device)
            if major * 10 + minor < 89:
                continue
            tested = True
            torch.cuda.set_device(device)
            torch.empty(1, device=device)  # Activate the matching CUDA context.
            driver = Driver()
            try:
                for name, dtype in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
                    meta = compile_kernel(args.compiler_python,
                        root / "tools/fastllm_triton_server.py", dict(
                            arch=major * 10 + minor, input_dtype=name, block_m=args.tile[0],
                            block_n=args.tile[1], block_k=128, group_size_m=args.tile[2],
                            quant_num_warps=4, matmul_num_warps=4, num_stages=3,
                            weight_layout="separate", matmul_variant="strided", cache_dir=cache))
                    function = driver.load(meta)
                    for shape in ((1, 1, 16), (7, 129, 144), (33, 257, 384),
                                  (64, 128, 128), (65, 513, 640), (129, 255, 128),
                                  (257, 129, 1792), (2113, 257, 384), (4097, 129, 128)):
                        for pattern in ("random", "integer", "zero"):
                            run_case(driver, function, meta, shape, dtype, pattern,
                                     padded=shape[0] % 2 == 1)
            finally:
                torch.cuda.synchronize()
                driver.close()
    print("PASS" if tested else "SKIP: requires SM89 or later")
    return 0 if tested else 77


if __name__ == "__main__":
    raise SystemExit(main())
