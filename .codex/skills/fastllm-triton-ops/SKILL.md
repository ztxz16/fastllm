---
name: fastllm-triton-ops
description: Guide for adding Triton-backed CUDA operators to FastLLM. Use when modifying FastLLM CUDA op code to add, extend, debug, validate, or benchmark Triton-generated kernels through tools/fastllm_triton_server.py, src/devices/cuda/cudadevice.cpp, src/devices/cuda/fastllm-triton-cuda.cu, include/devices/cuda/fastllm-cuda.cuh, or related CMake wiring.
---

# FastLLM Triton Ops

## Overview

Use the existing Linear Triton prototype as the reference architecture: C++ decides whether an op is eligible, starts a local Python compiler server only on demand, asks it to emit a cached cubin plus metadata, then launches that cubin through the CUDA Driver API. Every Triton path must be environment-gated and must fall back to the original CUDA implementation on unsupported inputs or compile/launch failure.

## Workflow

1. Inspect the existing CUDA op path in `src/devices/cuda/cudadevice.cpp`.
   - Find the op's `Reshape`, `CanRun`, `Run`, and lower-level helper functions.
   - Identify the exact tensor layout, dtype combinations, shape variables, optional bias/scale tensors, and output aliasing behavior.
   - Keep the original implementation as the fallback path.

2. Add or extend the Python compile path in `tools/fastllm_triton_server.py`.
   - Add a `@triton.jit` kernel for the new op.
   - Add `<op>_cache_paths(payload)` with a deterministic filename that includes op name, dtype/layout variants, SM arch, compile-time tile sizes, and feature flags.
   - Add `compile_<op>(payload)` that validates payload fields, compiles with `ASTSource`, writes `.cubin`, writes `.json` metadata, and returns the metadata.
   - Extend `handle_compile(payload)` by dispatching on `payload["op"]`.
   - Keep `/health` and `/compile` stable; do not break existing `"op": "linear"` requests.

3. Add a CUDA launch wrapper in `src/devices/cuda/fastllm-triton-cuda.cu`.
   - Reuse `LoadTritonKernel` for cubin/module/function caching.
   - Add one `extern "C"` wrapper per op, for example `FastllmCudaTriton<Op>(...)`.
   - Use `FastllmCudaPrepareInput`, `FastllmCudaPrepareOutput`, `FastllmCudaFinishInput`, and `FastllmCudaFinishOutput` when passing `Data` buffers.
   - Match the Triton kernel argument order exactly, including Triton's hidden `global_scratch` and `profile_scratch` pointer arguments when needed by AOT metadata.
   - Return `false` on load or launch failure so the C++ caller can fall back.

4. Declare the wrapper in `include/devices/cuda/fastllm-cuda.cuh`.
   - Keep the signature close to the CUDA fallback helper's shape arguments.
   - Pass metadata fields needed for launch, such as `kernelName`, `shared`, `numWarps`, and tile sizes.

5. Wire the op in `src/devices/cuda/cudadevice.cpp`.
   - Add a small metadata struct for the op's `.json` fields.
   - Reuse common helpers: `CudaTritonCacheDir`, `CudaTritonDataTypeName`, `CudaTritonHttpRequest`, and `CudaTritonEnsureServer`.
   - Add `CudaTriton<Op>BaseName`, `CudaTritonRead<Op>Meta`, `CudaTritonRequest<Op>Kernel`, and `TryCudaTriton<Op>`.
   - Gate with global `FASTLLM_CUDA_TRITON`; add an op-specific override such as `FASTLLM_CUDA_TRITON_<OP>=0`.
   - Validate device, pointer presence, dtype, layout, shape, arch, and feature constraints before requesting a kernel.
   - In the original op helper, call `TryCudaTriton<Op>(...)` immediately before the original CUDA implementation.

6. Update build wiring only when needed.
   - `src/devices/cuda/fastllm-triton-cuda.cu` is already in `CMakeLists.txt`.
   - If adding new files, update `CMakeLists.txt` and link dependencies without changing unrelated targets.

## Current Contract

The existing Triton infrastructure uses these environment variables:

- `FASTLLM_CUDA_TRITON=1`: enable Triton-backed CUDA ops globally.
- `FASTLLM_CUDA_TRITON_<OP>=0`: disable one op while keeping the global flag on, for example `FASTLLM_CUDA_TRITON_LINEAR=0`.
- `FASTLLM_CUDA_TRITON_CACHE_DIR`: override cubin/json cache directory.
- `FASTLLM_CUDA_TRITON_SERVER_HOST`, `FASTLLM_CUDA_TRITON_SERVER_PORT`: choose compiler server endpoint.
- `FASTLLM_CUDA_TRITON_PYTHON`: choose Python interpreter.
- `FASTLLM_CUDA_TRITON_SERVER_SCRIPT`: choose server script path.
- `FASTLLM_CUDA_TRITON_SERVER_LOG`: choose compiler server log path.
- `FASTLLM_CUDA_TRITON_SERVER_WAIT_MS`: choose startup wait timeout.
- Per-op tile knobs should use `FASTLLM_CUDA_TRITON_<OP>_<PARAM>`, matching Linear's `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `NUM_WARPS`, and `NUM_STAGES`.

The metadata JSON returned by the compiler server should include at least:

```json
{
  "ok": true,
  "op": "op_name",
  "cubin": "/path/to/kernel.cubin",
  "kernel": "compiled_kernel_name",
  "shared": 0,
  "num_warps": 4
}
```

Add op-specific launch fields, such as tile sizes, only when the C++ launcher needs them.

## C++ Pattern

Keep the C++ control flow shaped like this:

```cpp
if (!CudaEnvFlagEnabled("FASTLLM_CUDA_TRITON")) {
    return false;
}
const char *opEnv = std::getenv("FASTLLM_CUDA_TRITON_MYOP");
if (opEnv != nullptr && opEnv[0] != '\0' && !CudaEnvFlagEnabled("FASTLLM_CUDA_TRITON_MYOP")) {
    return false;
}
if (!inputs_are_supported) {
    return false;
}

Meta meta;
if (!ReadMeta(metaPath, meta)) {
    if (!RequestKernel(..., meta)) {
        return false;
    }
}
return FastllmCudaTritonMyOp(meta.cubinPath.c_str(), meta.kernelName.c_str(), ...);
```

Do not throw or call `ErrorInFastLLM` from the Triton trial path unless the original CUDA path would also fail. Unsupported Triton cases should return `false`.

## Validation

Run validation in layers:

1. Build:

```bash
bash install.sh -DUSE_CUDA=ON
```

2. Unit or op test, with Triton enabled and an isolated cache:

```bash
FASTLLM_CUDA_TRITON=1 \
FASTLLM_CUDA_TRITON_CACHE_DIR=/tmp/fastllm-triton-optest \
../optest --op linear --device cuda:0 --param batch=4 --param in=8 --param out=6
```

Adjust the `optest` command for the op being added.

3. End-to-end server smoke test:

```bash
FASTLLM_CUDA_TRITON=1 \
FASTLLM_CUDA_TRITON_CACHE_DIR=/tmp/fastllm-triton-qwen \
ftllm server ~/hfmodels/Qwen3-8B/ --device cuda:0 --host 127.0.0.1 --port 18080 --tokens 8192 --hide_input
```

Send one non-streaming request to `/v1/chat/completions` and confirm it completes.

4. Benchmark after warmup.
   - Always exclude first compile/server startup from performance numbers.
   - Compare against the same command without `FASTLLM_CUDA_TRITON`.
   - Report token throughput and wall time; state whether the measurement is kernel-only or end-to-end server throughput.

## Guardrails

- Keep Triton optional: default behavior must stay unchanged when `FASTLLM_CUDA_TRITON` is unset.
- Prefer compile keys that are independent of runtime shapes when the kernel supports dynamic dimensions.
- Include every compile-time specialization in the cache filename to prevent stale cubin reuse.
- Serialize compilation in the Python server with the existing lock unless proving concurrent compilation is safe.
- Keep C++ JSON parsing defensive; missing or invalid metadata should fall back.
- Clean up test servers and compiler-server processes after benchmarks.
