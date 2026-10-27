#pragma once
#include <cstdint>
#include "fastllm-native-prefill-policy.cuh"
#include <cuda_fp16.h>
namespace fastllm_native_prefill {
bool Enabled(const char *name);
bool Supported(int bits);
// Optional W8A8 / W4A4 prefill. These paths change activation precision;
// unsupported shapes, graph capture and unavailable cuBLAS plans return false.
// Modes: 0=Linear, 1=Linear+SwiGLU, 2=Linear+residual. Failures after output
// writes throw instead of falling back and applying residual a second time.
bool Fp8(const half *input, const uint8_t *weight, const float *scales, const half *bias, half *output, int m,
         int n, int k, int mode = 0);
bool Fp4(const half *input, const uint32_t *weight, const uint8_t *scales, const float *global,
         const half *bias, half *output, int m, int n, int packedN, int k, int mode = 0, bool nativeLayout = false);
} // namespace fastllm_native_prefill
