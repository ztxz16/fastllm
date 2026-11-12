# Strata CUDA kernel attribution

The CUDA HC implementation in `src/devices/cuda/models/strata-hyper-read.cuh`
is derived from `src/kernels/cuda/fused_gr.cu` and `include/strata/kernels/fused_gr.hpp`
in [Niko1221/Strata](https://github.com/Niko1221/Strata), commit
[`9f7d977d220e9fdf2ec53cf2620cdadac1cfb24e`](https://github.com/Niko1221/Strata/tree/9f7d977d220e9fdf2ec53cf2620cdadac1cfb24e).
Copyright (c) 2026 Niko1221 and the Strata contributors. MIT license, reproduced in LICENSE.

FastLLM adaptations: FP16 packed weights; FP32/FP16/BF16 block outputs and
activated injection scales; FastLLM residual rounding; bounded physical tiles
without a logical row limit; separate outputs and graph-managed scratch;
optional bank-conflict-free shared float4 planes with unchanged arithmetic;
FP32/FP16/BF16 mixed output, fusing the existing post-mix activation cast
while keeping residuals and injection scales FP32.
Only the CUDA operators and necessary types/helpers are incorporated. No Strata
runtime, configuration, environment switches, model loader or Python dependency.
The existing FastLLM implementation remains the fallback for unsupported layouts.

The register-state GDN implementation in
`src/devices/cuda/models/strata-gdn.cuh` is derived from
`src/kernels/cuda/fused_gdn.cu` at the same pinned commit and under the same MIT
license. It retains the register-resident state and row-group reductions;
FastLLM adaptations preserve the head-major state layout, grouped value heads,
Q/K normalization and FP16 output rounding, support arbitrary independent
request batches, and separate state input/output for safe benchmark trials.
The 1/2/4 row-group variants are selected by the common kernel benchmark;
they are physical thread geometries, not request-count limits. Only the CUDA
operator is incorporated; the Strata engine and state manager are not used.

The chunked indexed attention and merge in
`src/devices/cuda/models/qwen4-qsa-batch.cuh` are adapted from
`src/kernels/cuda/qsa_decode_attn.cu` at Strata commit
[`dbb2d168cffa8cf793d9325358809d5263c7c6b0`](https://github.com/Niko1221/Strata/tree/dbb2d168cffa8cf793d9325358809d5263c7c6b0),
under the same copyright and MIT license retained here. FastLLM adaptations
support strided independent KV caches, arbitrary ragged request rows and grouped
heads, FP16 queries and the existing FP16 output gate rounding. The optimized candidates also adapt the two-cell query reuse, head
reduce-scatter, cell-major probability access and grouped value loads from that
source, adding vectorized KV reads with an unaligned scalar fallback. The common
benchmark selects optimized 64/128-token physical attention chunks or the basic
fused 64-token implementation. These tile sizes
do not limit request counts, verifier rows or context length. QSA preparation,
compression, scoring and packed request orchestration use FastLLM storage and
metadata; the existing FastLLM radix selector is shared with the original path.
No Strata configuration, environment switches, scheduler or loader is included.
