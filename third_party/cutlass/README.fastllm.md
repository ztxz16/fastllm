# FastLLM CUTLASS Snapshot

This directory contains the CUTLASS headers used by FastLLM's CUDA FP8
linear kernels.

- Source repository: https://github.com/NVIDIA/cutlass
- CUTLASS version: 4.1.0
- Source tag: v4.1.0
- Source commit: e51efbfe18fe4f4cbb66ab814c55bf4aa0185491

Only the headers and utility headers needed by this integration are vendored.

The SM120 pointer-array block-scaled path includes the narrow-N scale-tile
support from the NVIDIA CUTLASS snapshot distributed with FlashInfer 0.7.4.
The SGLang Inkling-2-Mini backend uses its swap-AB grouped GEMM on
expert-sorted rows. FastLLM's weight-loader hook supplies two read-only 3D TMA
maps for its native packed FP4 codes and block scales. They span all experts
and preserve the weight layout shared with the small-row decode kernel. Four
interleaved FP32 accumulator fragments match decode rounding.
The standard load and accumulator remain the defaults.

- [FlashInfer grouped NVFP4 kernel](https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/gemm/group_gemm_nvfp4_groupwise_sm120.cuh)
- [NVIDIA CUTLASS SM120 block-scaled array mainloop](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp)
