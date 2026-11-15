#pragma once
#include <cuda_runtime.h>
namespace fastllm::minimax_h3_image::detail {
void int8ScaledGemm(void *out, const void *x, const void *w, const float *xs, const float *ws, int m, int n,
                    int k, cudaStream_t stream, bool fp32Output = false);
void nvfp4Gemm(void *out, const void *x, const void *w, const void *xs, const void *ws, const float *alpha,
               int m, int n, int k, cudaStream_t stream, bool fp32Output = false);
} // namespace fastllm::minimax_h3_image::detail
