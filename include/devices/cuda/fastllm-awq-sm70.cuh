//
// SM70 (V100) quantized A16 GEMM bridge over ported TurboMind s884 kernels.
//
// This is a thin, torch-free bridge over the TurboMind GEMM library (copied
// into third_party/turbomind). It is only meaningful on compute capability 7.0,
// where the Marlin path is unavailable. It supports FastLLM AWQ INT4_GROUP,
// block-scaled FP8_E4M3, and NVFP4_BLOCK_16 weights.
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace fastllm {
namespace awq_sm70 {

// True only on SM70 devices where the ported TurboMind kernels are registered.
bool Supported();

// Build TurboMind packed weight/scale tensors from raw, unpacked AWQ data.
//   d_qvals_u16 : [K, N] row-major, each element a 4-bit weight value (0..15)
//                 widened to uint16. K is the input dim, N the output dim.
//   d_scales    : [num_groups, N] row-major half, per-group scale.
//   d_zeros     : [num_groups, N] row-major half, per-group integer zero point.
// Dequant convention is w = scale * (q - zero), matching fastllm INT4_GROUP
// (min = -scale * zero).
// Returns an opaque handle (nullptr on failure); device memory is owned by it.
void *Prepare(const uint16_t *d_qvals_u16, const half *d_scales, const half *d_zeros,
              int K, int N, int num_groups, int group_size, cudaStream_t stream);

// out[tokens, N] = in[tokens, K] @ dequant(W). Row-major half in/out.
bool Gemm(void *handle, const half *in, half *out, int tokens, cudaStream_t stream);

void Free(void *handle);

// Dense block-scaled FP8_E4M3 bridge. FastLLM stores the source weight as an
// [N, K] row-major byte matrix and one FP32 scale per
// [outputBlockSize, inputBlockSize] tile. PrepareFp8InPlace converts the
// weight in place to TurboMind's SM70 MMA layout and returns a newly allocated
// packed FP16 scale matrix through packedScales. The caller owns packedScales.
bool Fp8Supported();
bool PrepareFp8InPlace(uint8_t *weight, const float *blockScales,
                       half **packedScales, int K, int N,
                       int inputBlockSize, int outputBlockSize,
                       cudaStream_t stream);
bool GemmFp8(const uint8_t *packedWeight, const half *packedScales,
             const half *in, half *out, int tokens, int K, int N,
             int groupSize, cudaStream_t stream);

// NVFP4_BLOCK_16 bridge.  The source layout contains eight packed E2M1 bytes
// followed by one float scale for each group of sixteen input channels.
// PrepareNvfp4InPlace converts it to [TurboMind weight][TurboMind FP16 scales]
// inside the same allocation; the converted representation is smaller than
// the source representation.  K is the input dimension and N the output
// dimension. K must be divisible by 16 and N by 32, matching the SM70 packed
// operand layout.
bool Nvfp4Supported();
bool PrepareNvfp4InPlace(uint8_t *storage, size_t storageBytes,
                         int K, int N, cudaStream_t stream);
bool GemmNvfp4(const uint8_t *storage, const half *in, half *out,
               int tokens, int K, int N, cudaStream_t stream);

}  // namespace awq_sm70
}  // namespace fastllm
