# DeepGEMM headers used by FastLLM

This directory contains a curated header-only subset of
[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM).  The starting point was
the DeepGEMM tree bundled with the local vLLM 0.26.0 DeepSeek-V4 reference
environment.  FastLLM-specific SM120 BF16, MoE, and MQA integrations adapt
parts of that subset, so this directory is not a byte-for-byte upstream
snapshot.

Only the headers needed by FastLLM's optional SM90 and SM120 kernels are kept.
Each integration is compiled behind its own CMake architecture gate.

The SM90 FP8 linear integration adds `impls/sm90_fp8_gemm_1d2d.cuh`,
`mma/sm90.cuh`, and `ptx/wgmma.cuh` from the DeepGEMM headers bundled with
vLLM 0.28.0. Its common dependencies are identical to the existing headers.
The original SM90 kernel header SHA-256 is
`8667552aeb0f56be360bf71bcaeacd1d252cfc0cd12a7c555ae747e68bb64e8d`.
Local changes restrict device code to SM90, add FP16 epilogue stores alongside
BF16, and inline the cluster barrier to avoid unused MoE communication headers.
The FP8/FP32 mainloop is unchanged.

The launcher in `src/devices/cuda/linear/fastllm-linear-fp8-sm90.cu` requires
CUDA >= 12.8, a build architecture list containing 90, and runtime capability
9.0. Its separate object target uses `sm_90a`; other architectures retain their
existing dispatch. It supports row-major E4M3 weights with 128x128 scales and
FP16/BF16 activations, quantized dynamically in 1x128 groups. Unsupported shapes
and layouts fall back. Fresh eligible weights stay row-major for prefill and
decode; already packed Marlin weights keep their existing path.

The persistent grid has 114 CTAs, tuned for H800 PCIe. The scheduler uses the
same grid size independently of the physical SM count. Large M uses a
256x128x128 tile with two-CTA multicast; smaller M uses 64/128x128x128 tiles.
Per-thread, per-device scratch retains old buffers for CUDA Graph replay and
grows geometrically; allocations live for the process lifetime.

Run `python test/ops/check_fp8_sm90.py` after building. It needs PyTorch with
FP8 support, CUDA headers and g++, and builds a temporary test-only adapter.
Use `--lib /path/to/libfastllm_tools.so` for a non-default build directory.
It checks FP16/BF16 arithmetic against an independent FP32 reference, bias,
zero inputs, M tails, graph replay after buffer growth, and rejection of
unsupported shapes/layouts. Model accuracy evaluation is separate.

DeepGEMM is distributed under the MIT license; see `LICENSE`.
