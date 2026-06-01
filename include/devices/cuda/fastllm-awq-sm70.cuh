//
// SM70 (V100) AWQ W4A16 GEMM, backed by ported TurboMind s884 kernels.
//
// This is a thin, torch-free bridge over the TurboMind GEMM library (copied
// into third_party/turbomind). It is only meaningful on compute capability 7.0,
// where the Marlin path is unavailable.
//
#pragma once

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

}  // namespace awq_sm70
}  // namespace fastllm
