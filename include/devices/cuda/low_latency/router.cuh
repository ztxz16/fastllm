#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
// BF16 input [rows,Hidden], FP32 row-major weight [experts,Hidden], FP32
// sigmoid output [rows,experts]. Kernels never allocate and accept any CUDA
// stream at launch. Hidden is a physical specialization, not a model identity.
// Single/Rows use 256 threads. Rows<=8; grid=(experts,ceil(rows/Rows)).
// Tiled uses 256 threads, grid=(experts/Experts,ceil(rows/Rows)); requires
// experts%Experts==0 and 16-byte aligned input/weights. Logical rows have tails.
namespace fastllm::cuda::low_latency {
using BF16 = __nv_bfloat16;
template<int Hidden, bool WaitForSwa = false>
__global__ void RouterSigmoidSingle(const BF16 *input, const float *weight,
                                  float *output) {
    static_assert(Hidden > 0 && Hidden % 256 == 0);
    int t = threadIdx.x, expert = blockIdx.x;
    __shared__ float sums[8];
    float values[Hidden/256], weights[Hidden/256], partial = 0;
    if constexpr (WaitForSwa) {
        // The SWA epilogue still owns the normalized row. Only weights are ready.
        const float *row = weight + (size_t)expert * Hidden;
        #pragma unroll
        for (int i = 0; i < Hidden/256; ++i) weights[i] = row[t + i * 256];
#if CUDART_VERSION >= 12000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        cudaGridDependencySynchronize();
#endif
    }
    #pragma unroll
    for (int i = 0; i < Hidden/256; ++i) {
        values[i] = (float)input[t + i * 256];
        if constexpr (!WaitForSwa) weights[i] = weight[(size_t)expert * Hidden + t + i * 256];
    }
    #pragma unroll
    for (int i = 0; i < Hidden/256; ++i) partial += values[i] * weights[i];
    #pragma unroll
    for (int stride = 1; stride < 32; stride *= 2)
        partial += __shfl_down_sync(0xffffffffu, partial, stride);
    if (t % 32 == 0) sums[t / 32] = partial;
    __syncthreads();
    if (t < 32) {
        partial = t < 8 ? sums[t] : 0;
        #pragma unroll
        for (int stride = 1; stride < 8; stride *= 2)
            partial += __shfl_down_sync(0xffffffffu, partial, stride);
        if (t == 0) {
            float logit = partial + 0.0f;
            // Match standalone FP32 sigmoid's double-precision division.
            output[expert] = 1.0 / (1.0 + expf(-logit));
        }
    }
}

template<int Rows, int Hidden>
__device__ __forceinline__ void RouterRowsBody(const BF16 *input, const float *weight,
        float *output, int rows, int experts, int t, int first, int expert, float (&sums)[Rows][8]) {
    static_assert(Hidden > 0 && Hidden % 256 == 0);
    static_assert(Rows > 0 && Rows <= 8);
    float weights[Hidden/256], values[Rows][Hidden/256], partial[Rows]{};
    #pragma unroll
    for (int i = 0; i < Hidden/256; ++i) {
        weights[i] = weight[(size_t)expert * Hidden + t + i * 256];
        #pragma unroll
        for (int row = 0; row < Rows; ++row)
            values[row][i] = first + row < rows
                ? (float)input[(size_t)(first + row) * Hidden + t + i * 256] : 0;
    }
    #pragma unroll
    for (int i = 0; i < Hidden/256; ++i) {
        #pragma unroll
        for (int row = 0; row < Rows; ++row) partial[row] += values[row][i] * weights[i];
    }
    #pragma unroll
    for (int row = 0; row < Rows; ++row) {
        #pragma unroll
        for (int stride = 1; stride < 32; stride *= 2)
            partial[row] += __shfl_down_sync(0xffffffffu, partial[row], stride);
        if (t % 32 == 0) sums[row][t / 32] = partial[row];
    }
    __syncthreads();
    // Independent warps finish different rows, including the expensive sigmoid.
    // Keep the reference GEMV's reduction and zero-bias addition exactly.
    const int row = t / 32, lane = t % 32;
    if (row < Rows) {
        float value = lane < 8 ? sums[row][lane] : 0;
        #pragma unroll
        for (int stride = 1; stride < 8; stride *= 2)
            value += __shfl_down_sync(0xffffffffu, value, stride);
        if (lane == 0 && first + row < rows)
            output[(size_t)(first + row) * experts + expert] =
                1.0 / (1.0 + expf(-(value + 0.0f)));
    }
}

template<int Rows, int Hidden>
__global__ void RouterSigmoidRows(const BF16 *input, const float *weight,
        float *output, int rows, int experts) {
    __shared__ float sums[Rows][8];
    RouterRowsBody<Rows, Hidden>(input, weight, output, rows, experts, threadIdx.x,
                                blockIdx.y * Rows, blockIdx.x, sums);
}

template<int Experts, int Rows, int Hidden>
__global__ void RouterSigmoidTiled(const BF16 *input, const float *weight,
        float *output, int rows, int experts) {
    static_assert(Hidden > 0 && Hidden % 256 == 0);
    static_assert(Hidden == 4096, "shared-expert tile currently supports K=4096");
    const int t = threadIdx.x, lane = t % 32, warp = t / 32;
    const int expert = blockIdx.x * Experts, first = blockIdx.y * Rows;
    __shared__ float sums[Experts * Rows][8];
    float4 cached[Experts][4];
#pragma unroll
    for (int e = 0; e < Experts; ++e) {
#pragma unroll
        for (int i = 0; i < 4; ++i)
            cached[e][i] = *reinterpret_cast<const float4 *>(weight +
                size_t(expert + e) * Hidden + t * 8 + (i / 2) * 2048 + (i % 2) * 4);
    }
    float partial[Experts][Rows]{};
#pragma unroll
    for (int c = 0; c < 2; ++c) {
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            const uint4 raw = first + row < rows
                ? *reinterpret_cast<const uint4 *>(input + size_t(first + row) * Hidden + t * 8 + c * 2048)
                : make_uint4(0, 0, 0, 0);
            const auto *src = reinterpret_cast<const __nv_bfloat162 *>(&raw);
            float values[8];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 v = __bfloat1622float2(src[j]);
                values[j * 2] = v.x; values[j * 2 + 1] = v.y;
            }
#pragma unroll
            for (int e = 0; e < Experts; ++e) {
                const float *w = reinterpret_cast<const float *>(&cached[e][c * 2]);
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    partial[e][row] = __fmaf_rn(values[j], w[j], partial[e][row]);
            }
        }
    }
#pragma unroll
    for (int e = 0; e < Experts; ++e) {
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
#pragma unroll
            for (int d = 1; d < 32; d *= 2)
                partial[e][row] += __shfl_down_sync(0xffffffffu, partial[e][row], d);
            if (lane == 0) sums[e * Rows + row][warp] = partial[e][row];
        }
    }
    __syncthreads();
    for (int task = t / 8; task < Experts * Rows; task += 32) {
        float value = sums[task][t % 8];
#pragma unroll
        for (int d = 1; d < 8; d *= 2) value += __shfl_down_sync(0xffffffffu, value, d, 8);
        if (t % 8 == 0 && first + task % Rows < rows)
            output[size_t(first + task % Rows) * experts + expert + task / Rows] =
                1.0 / (1.0 + expf(-(value + 0.f)));
    }
}
} // namespace fastllm::cuda::low_latency
