#pragma once
#include "naivert/attn_dsa/projection.cuh"
namespace fastllm_dsa_layer {
using B = __nv_bfloat16;
constexpr int Threads = 384;
__device__ __forceinline__ float Round(float x) {
    return float(__float2bfloat16_rn(x));
}
template <int Rows> struct ProjectionStorage {
    B input[Rows * 4096];
    float partial[12 * 128 * (Rows / 8)];
};
template <int Rows, int Width> __device__ void LoadProjectionInput(const B *input, int rows, B *output) {
    constexpr int vectorsPerRow = Width / 8;
    for (int i = threadIdx.x; i < Rows * vectorsPerRow; i += Threads) {
        int row = i / vectorsPerRow, col = (i % vectorsPerRow) * 8;
        B *target = output + (row / 8) * 8 * Width + fastllm_dsa_projection::input_offset(row % 8, col);
        if (row < rows) {
            unsigned dst = unsigned(__cvta_generic_to_shared(target));
            asm volatile("cp.async.cg.shared.global [%0],[%1],16;" : : "r"(dst), "l"(input + row * Width + col)
                         : "memory");
        } else {
            *reinterpret_cast<uint4 *>(target) = make_uint4(0, 0, 0, 0);
        }
    }
    asm volatile("cp.async.commit_group;cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
}

__device__ inline void NormalizeIndex(B *x, const float *norm, const float *bias) {
    __shared__ float sums[64], squares[64], mean, inverseStd;
    int t = threadIdx.x;
    if (t < 64) {
        float sum = 0, square = 0;
        for (int j = t; j < 128; j += 64) {
            float v = float(x[j]);
            sum += v;
            square += v * v;
        }
        sums[t] = sum;
        squares[t] = square;
    }
    __syncthreads();
    for (int step = 32; step; step >>= 1) {
        if (t < step) {
            sums[t] += sums[t + step];
            squares[t] += squares[t + step];
        }
        __syncthreads();
    }
    if (!t) {
        mean = sums[0] / 128;
        float variance = squares[0] + mean * mean * 128 - 2 * mean * 128 * mean;
        inverseStd = rsqrtf(variance / 128 + 1e-10f);
    }
    __syncthreads();
    if (t < 128)
        x[t] = __float2bfloat16_rn((float(x[t]) - mean) * inverseStd * norm[t] + bias[t]);
    __syncthreads();
}
template<class Output> __device__ __forceinline__ void RotateWarp(const B *source, Output output, int dim, float cs, float sn) {
    int lane = threadIdx.x % 32;
    float x = float(source[lane]), y = float(source[lane + 32]);
    output[lane] = __float2bfloat16_rn(Round(x * cs) - Round(y * sn));
    output[lane + 32] = __float2bfloat16_rn(Round(y * cs) + Round(x * sn));
    for (int col = 64 + lane; col < dim; col += 32)
        output[col] = source[col];
}
} // namespace fastllm_dsa_layer
