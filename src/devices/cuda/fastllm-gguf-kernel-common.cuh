#pragma once

// Support for standalone GGUF kernels and their independent reference tests.
// Legacy translation units supply their own GGUF helpers before small-mmvq.cuh.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>
#include <type_traits>
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "gguf.h"
#include "fastllm-gguf-store.cuh"

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif
static __device__ __forceinline__ int get_int_b4(const void *p, const int &i) {
    return static_cast<const int *>(p)[i];
}
static __device__ __forceinline__ int get_int_b2(const void *p, const int &i) {
    const auto *v = static_cast<const uint16_t *>(p);
    return int(uint32_t(v[2 * i]) | (uint32_t(v[2 * i + 1]) << 16));
}
static __device__ __forceinline__ int ggml_cuda_dp4a(int a, int b, int c) { return __dp4a(a, b, c); }
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int d = 16; d; d >>= 1)
        x += __shfl_xor_sync(0xffffffff, x, d);
    return x;
}
