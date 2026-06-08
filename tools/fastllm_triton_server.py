#!/usr/bin/env python3
import argparse
import json
import os
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

_compile_lock = Lock()
_triton_error = None

try:
    import triton
    import triton.language as tl
    from triton.compiler.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
except Exception as exc:
    triton = None
    tl = None
    ASTSource = None
    GPUTarget = None
    _triton_error = exc

if triton is not None:
    @triton.jit
    def _fastllm_fp8e4m3_to_float(x):
        x = x.to(tl.uint32)
        bits = ((x & 0x80) << 8) | ((x & 0x7F) << 7)
        return bits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32) * 256.0


    @triton.jit
    def fastllm_linear_fp8_quant_input_kernel(
        input_ptr,
        q_ptr,
        scale_ptr,
        batch,
        hidden,
        BLOCK_K: tl.constexpr,
    ):
        token = tl.program_id(0)
        group = tl.program_id(1)
        offs = group * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = (token < batch) & (offs < hidden)
        x = tl.load(input_ptr + token * hidden + offs, mask=mask, other=0.0).to(tl.float32)
        absmax = tl.maximum(tl.max(tl.abs(x)), 1.0e-10)
        scale = absmax * (1.0 / 448.0)
        q = tl.clamp(x / scale, -448.0, 448.0).to(q_ptr.dtype.element_ty)
        tl.store(q_ptr + token * hidden + offs, q, mask=mask)
        tl.store(scale_ptr + token * tl.cdiv(hidden, BLOCK_K) + group, scale, mask=token < batch)


    @triton.jit
    def fastllm_linear_fp8_block128_matmul_kernel(
        a_ptr,
        a_scale_ptr,
        b_ptr,
        b_scale_ptr,
        bias_ptr,
        c_ptr,
        M,
        N,
        K,
        PER_ROW,
        SCALE_COLS,
        HAS_BIAS: tl.constexpr,
        PACKED_WEIGHT: tl.constexpr,
        COMPUTE_TYPE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
        group_size_m = tl.maximum(group_size_m, 1)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_scale_cols = tl.cdiv(K, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_idxs = k * BLOCK_K + offs_k
            a = tl.load(
                a_ptr + offs_m[:, None] * K + k_idxs[None, :],
                mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
                other=0.0,
            )
            if PACKED_WEIGHT:
                b = tl.load(
                    b_ptr
                    + offs_n[None, :] * PER_ROW
                    + k * (BLOCK_K + 4)
                    + offs_k[:, None],
                    mask=(offs_n[None, :] < N) & (k_idxs[:, None] < K),
                    other=0.0,
                )
                b_scale_ptrs = (
                    b_ptr + offs_n * PER_ROW + k * (BLOCK_K + 4) + BLOCK_K
                ).to(tl.pointer_type(tl.float32))
                b_scale = tl.load(b_scale_ptrs, mask=offs_n < N, other=0.0)
            else:
                b = tl.load(
                    b_ptr + offs_n[None, :] * K + k_idxs[:, None],
                    mask=(offs_n[None, :] < N) & (k_idxs[:, None] < K),
                    other=0.0,
                )
                b_scale = tl.load(
                    b_scale_ptr + (offs_n // BLOCK_K) * SCALE_COLS + k,
                    mask=offs_n < N,
                    other=0.0,
                )
            a_scale = tl.load(
                a_scale_ptr + offs_m * a_scale_cols + k,
                mask=offs_m < M,
                other=0.0,
            )
            acc += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
            acc += bias[None, :]

        tl.store(
            c_ptr + offs_m[:, None] * N + offs_n[None, :],
            acc.to(COMPUTE_TYPE),
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


def default_cache_dir():
    value = os.environ.get("FASTLLM_CUDA_TRITON_CACHE_DIR")
    if value:
        return Path(value).expanduser()
    value = os.environ.get("XDG_CACHE_HOME")
    if value:
        return Path(value).expanduser() / "fastllm" / "triton"
    value = os.environ.get("HOME")
    if value:
        return Path(value).expanduser() / ".cache" / "fastllm" / "triton"
    return Path("/tmp") / "fastllm-triton"


def require_int(payload, name, fallback=None):
    value = payload.get(name, fallback)
    if value is None:
        raise ValueError(f"missing required field: {name}")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def require_dtype(payload, name):
    value = str(payload.get(name, ""))
    if value not in {"fp16", "bf16", "fp32"}:
        raise ValueError(f"{name} must be fp16, bf16, or fp32")
    return value


LINEAR_FP8_BLOCK128_KERNEL_ORDER = ("quant_input", "matmul")


def linear_fp8_block128_cache_paths(payload):
    arch = require_int(payload, "arch")
    input_dtype = require_dtype(payload, "input_dtype")
    if input_dtype not in {"fp16", "bf16"}:
        raise ValueError("input_dtype must be fp16 or bf16")
    weight_layout = str(payload.get("weight_layout") or "packed")
    if weight_layout not in {"packed", "separate"}:
        raise ValueError("weight_layout must be packed or separate")
    has_bias = 1 if bool(payload.get("has_bias", False)) else 0
    block_m = require_int(payload, "block_m", 16)
    block_n = require_int(payload, "block_n", 128)
    block_k = require_int(payload, "block_k", 128)
    group_size_m = require_int(payload, "group_size_m", 32)
    quant_num_warps = require_int(payload, "quant_num_warps", 4)
    matmul_num_warps = require_int(payload, "matmul_num_warps", 4)
    num_stages = require_int(payload, "num_stages", 3)
    if block_k != 128:
        raise ValueError("FP8 block128 linear requires block_k=128")
    cache_dir = Path(payload.get("cache_dir") or default_cache_dir()).expanduser()
    name = (
        f"linear_fp8_block128_v2_{weight_layout}_{input_dtype}_bias{has_bias}_sm{arch}"
        f"_bm{block_m}_bn{block_n}_bk{block_k}_gsm{group_size_m}"
        f"_qnw{quant_num_warps}_mnw{matmul_num_warps}_ns{num_stages}"
    )
    cubins = {key: cache_dir / f"{name}_{key}.cubin" for key in LINEAR_FP8_BLOCK128_KERNEL_ORDER}
    return cubins, cache_dir / f"{name}.json"


def _compile_cubin(fn, signature, constexprs, arch, num_warps, num_stages, cubin_path):
    attrs = {(i,): [["tt.divisibility", 16]] for i, name in enumerate(fn.arg_names) if str(signature.get(name, "")).startswith("*")}
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs, attrs=attrs)
    target = GPUTarget("cuda", arch, 32)
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options({"num_warps": num_warps, "num_stages": num_stages})
    ccinfo = triton.compile(src, target=target, options=options.__dict__)
    cubin_path.write_bytes(ccinfo.asm[backend.binary_ext])
    return ccinfo


def compile_linear_fp8_block128(payload):
    if triton is None:
        raise RuntimeError(f"failed to import triton: {_triton_error}")
    input_dtype = require_dtype(payload, "input_dtype")
    if input_dtype not in {"fp16", "bf16"}:
        raise ValueError("input_dtype must be fp16 or bf16")
    arch = require_int(payload, "arch")
    block_m = require_int(payload, "block_m", 16)
    block_n = require_int(payload, "block_n", 128)
    block_k = require_int(payload, "block_k", 128)
    group_size_m = require_int(payload, "group_size_m", 32)
    quant_num_warps = require_int(payload, "quant_num_warps", 4)
    matmul_num_warps = require_int(payload, "matmul_num_warps", 4)
    num_stages = require_int(payload, "num_stages", 3)
    has_bias = bool(payload.get("has_bias", False))
    weight_layout = str(payload.get("weight_layout") or "packed")
    if weight_layout not in {"packed", "separate"}:
        raise ValueError("weight_layout must be packed or separate")
    packed_weight = weight_layout == "packed"
    if block_k != 128:
        raise ValueError("FP8 block128 linear requires block_k=128")
    cubin_paths, meta_path = linear_fp8_block128_cache_paths(payload)
    if all(path.exists() for path in cubin_paths.values()) and meta_path.exists():
        return json.loads(meta_path.read_text())
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    compute_type = tl.float16 if input_dtype == "fp16" else tl.bfloat16
    quant_ccinfo = _compile_cubin(
        fastllm_linear_fp8_quant_input_kernel,
        {"input_ptr": f"*{input_dtype}", "q_ptr": "*fp8e4nv", "scale_ptr": "*fp32", "batch": "i32", "hidden": "i32", "BLOCK_K": "constexpr"},
        {"BLOCK_K": block_k}, arch, quant_num_warps, num_stages, cubin_paths["quant_input"])
    matmul_ccinfo = _compile_cubin(
        fastllm_linear_fp8_block128_matmul_kernel,
        {"a_ptr": "*fp8e4nv", "a_scale_ptr": "*fp32", "b_ptr": "*fp8e4nv", "b_scale_ptr": "*fp32", "bias_ptr": "*fp32", "c_ptr": f"*{input_dtype}", "M": "i32", "N": "i32", "K": "i32", "PER_ROW": "i32", "SCALE_COLS": "i32", "HAS_BIAS": "constexpr", "PACKED_WEIGHT": "constexpr", "COMPUTE_TYPE": "constexpr", "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "BLOCK_K": "constexpr", "GROUP_SIZE_M": "constexpr"},
        {"HAS_BIAS": has_bias, "PACKED_WEIGHT": packed_weight, "COMPUTE_TYPE": compute_type, "BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k, "GROUP_SIZE_M": group_size_m},
        arch, matmul_num_warps, num_stages, cubin_paths["matmul"])
    ccinfos = {"quant_input": quant_ccinfo, "matmul": matmul_ccinfo}
    kernels = {}
    for key in LINEAR_FP8_BLOCK128_KERNEL_ORDER:
        ccinfo = ccinfos[key]
        kernels[key] = {"cubin": str(cubin_paths[key]), "kernel": ccinfo.metadata.name, "shared": int(ccinfo.metadata.shared), "num_warps": int(ccinfo.metadata.num_warps)}
    meta = {"ok": True, "op": "linear_fp8_block128", "kernels": kernels, "block_m": block_m, "block_n": block_n, "block_k": block_k, "group_size_m": group_size_m, "quant_num_warps": quant_num_warps, "matmul_num_warps": matmul_num_warps, "num_stages": num_stages, "arch": arch, "input_dtype": input_dtype, "weight_layout": weight_layout, "packed_weight": packed_weight, "has_bias": has_bias}
    meta_path.write_text(json.dumps(meta, sort_keys=True))
    return meta


def handle_compile(payload):
    op = payload.get("op")
    with _compile_lock:
        if op == "linear_fp8_block128":
            return compile_linear_fp8_block128(payload)
        raise ValueError(f"unsupported op: {op}")


class Handler(BaseHTTPRequestHandler):
    server_version = "fastllm-triton/0.1"
    def log_message(self, fmt, *args):
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)
    def write_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        if self.path == "/health":
            self.write_json(200, {"ok": True, "triton": triton is not None})
            return
        self.write_json(404, {"ok": False, "error": "not found"})
    def do_POST(self):
        if self.path != "/compile":
            self.write_json(404, {"ok": False, "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            self.write_json(200, handle_compile(payload))
        except Exception as exc:
            self.write_json(500, {"ok": False, "error": str(exc), "traceback": traceback.format_exc()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=48989)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.verbose = args.verbose
    server.serve_forever()


if __name__ == "__main__":
    main()
