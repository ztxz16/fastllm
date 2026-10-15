#pragma once
#include "fastllm-cuda.cuh"
#include <cuda_bf16.h>
#include <stdexcept>

inline void FastllmCudaHybridCheck(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("Hybrid prefill CUDA: ") + cudaGetErrorString(status));
}

// False means the caller must recompute the complete output with its fallback.
bool FastllmCudaHybridBf16Gemm(const __nv_bfloat16 *input,
        const __nv_bfloat16 *weight, __nv_bfloat16 *output,
        int rows, int columns, int width, int outputStride, void *handle);
bool FastllmCudaGlm5HybridPrefillAttention(const fastllm::Data &query,
        const fastllm::Data &latent, const fastllm::Data &indices,
        float scale, fastllm::Data &output);

bool FastllmCudaHybridHcPost(const fastllm::Data &input, const fastllm::Data &residual,
        const float *post, const float *comb, fastllm::Data &output,
        int tokens, int hcMult, int dim);
