# ROCm GGUF MMQ kernels

Derived from [llama.cpp](https://github.com/ggml-org/llama.cpp) commit `74a7c897f049c17e7080423aa2111776eff6ebbf` (MIT; see LICENSE).

The quantization layouts, tile loaders, dot products, WMMA/MFMA code and architecture configurations come from ggml/src/ggml-cuda. `ggml-common.h` contains its packed block definitions and lookup tables. `types.h` keeps the serialized type enum and local assertion helpers. `common.cuh` keeps device helpers, with backend-dependent declarations removed and helper symbols isolated. `mmq.cuh` stops before the upstream backend allocator/launcher.

FastLLM owns activation conversion, temporary allocation, dispatch and stream handling in the adjacent HIP adapter. It uses no llama.cpp runtime or separately installed library.

Runtime dispatch covers the known GCN, CDNA 1–4 and RDNA 1–4 targets in `policy.cuh` (including RDNA 3.5). It reads each device's real ISA and wave size, selects its upstream configuration family and checks shared-memory/thread limits. CDNA uses wave64 MFMA and stream-K with fixup only when needed; RDNA 3/3.5/4 uses wave32 WMMA; older architectures use packed integer dot products (software dot products where the ISA has no dot4 instruction). Unknown targets or unsupported shapes retain the existing fallback.

`test/ops/rocmGgufMmqDispatch.hip` checks host policy across the supported ISAs and compares host/device configuration and wave size on the GPU where it runs. Cross-compilation alone does not establish correctness or speed on another physical GPU. The existing MMQ CPU-reference test can be run on each target.
