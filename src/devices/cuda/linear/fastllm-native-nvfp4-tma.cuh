#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
namespace fastllm_native_prefill {
// This implementation is linked only in SM120-family builds. The runtime
// image check also rejects GPUs not covered by the dedicated fatbin.
bool Nvfp4TmaCanRun(int m, int n, int k, int mode);
// Input scales use 256-token x 16-group tiles; weight scales retain the
// cuBLASLt block16 layout. Returns false before writing output, or throws on
// a launch failure. Modes are identical to the generic native prefill path.
bool TryNvfp4Tma(const uint8_t *x, const uint8_t *xs, const uint8_t *weight, const uint8_t *weightScales,
                 const float *rowScales, const float *weightGlobal, half *output, int m, int n, int k,
                 int mode);
} // namespace fastllm_native_prefill
