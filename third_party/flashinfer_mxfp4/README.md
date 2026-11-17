# SM90 MXFP4 MegaMoE

This directory contains the native CUDA headers needed by FlashInfer v0.7.4's
SM90 MXFP4 MegaMoE, under the upstream Apache-2.0 license. Upstream source:
https://github.com/flashinfer-ai/flashinfer/tree/v0.7.4/include/flashinfer/megamoe
`SOURCE_SHA256.json` records the original files and checksums. The headers are
unmodified. They compile against FastLLM's existing CUTLASS/CuTe headers.
No FlashInfer Python package, Torch, TVM-FFI, CuTe DSL or runtime JIT is needed.

FastLLM's adapter is `src/devices/cuda/moe/fastllm-mxfp4-ep.cu`. It supports H20
SM90/78-SM EP4/EP8, hidden 4096, intermediate 2048, 256 experts and top-8 routing.
`--ep 4` or `--ep 8` selects it automatically for native block-32 UE8M0 MXFP4
expert weights. Unsupported EP hardware, shapes or scale ranges are rejected;
use `--ep 1` for the generic MXFP4 path. Attention TP/DP uses the existing
MiMo scheduler. Decode uses a 128-row arena per rank; larger inputs use a
separate bounded 512-row prefill arena. Unequal/idle sources, masked Graph slots and partial
tails are supported. Both arenas share the same packed expert weights, and
are allocated before serving so switching between them needs no allocation.

The adapter repacks E2M1 nibbles and shifts UE8M0 exponents losslessly during
loading. A per-expert power-of-two multiplier compensates the shift before
SwiGLU. Scales outside the decoder's supported dynamic range are rejected
explicitly. Activations use power-of-two FP8 quantization; this is W4A8 and is
not numerically identical to the generic W4A16 path. SwiGLU is unclamped.

Packed weights, descriptors and symmetric workspaces are prepared before
Graph capture. Inference performs no allocations. Currently the canonical
weights are retained alongside the packed copy. This costs one additional
compact expert-weight copy per rank and should be included in capacity plans.
