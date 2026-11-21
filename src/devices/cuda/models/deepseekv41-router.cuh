#pragma once
// Reuse each FP32 weight across tokens without changing the original 256-lane
// accumulation chains or pairwise reduction tree. Streaming weight reads on
// SM90 preserve L1 space for the input reused by all expert blocks.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <type_traits>
// Each vector covers adjacent leaves of the reference 256-lane reduction.
// The twenty per-leaf FMAs, followed by ascending pairwise sums, are unchanged.
template<int Threads, int Outputs>
__global__ void V41RouterVector5120(const __nv_bfloat16 *input, const __nv_bfloat16 *weights,
                                     float *scores, int experts, float invTemp) {
    constexpr int width = 256 / Threads;
    using Load = typename std::conditional<width == 4, uint2, unsigned int>::type;
    static_assert(sizeof(Load) == width * sizeof(__nv_bfloat16), "invalid load width");
    const int row = blockIdx.y, first = blockIdx.x * Outputs, tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    float sums[Outputs][width] = {};
#pragma unroll
    for (int part = 0; part < 20; ++part) {
        const int offset = part * Threads + tid;
        Load x = reinterpret_cast<const Load *>(input + size_t(row) * 5120)[offset];
#pragma unroll
        for (int j = 0; j < Outputs; ++j) {
            Load w = reinterpret_cast<const Load *>(weights + size_t(first + j) * 5120)[offset];
#pragma unroll
            for (int i = 0; i < width; ++i)
                sums[j][i] = fmaf(__bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(&x)[i]),
                                   __bfloat162float(reinterpret_cast<const __nv_bfloat16 *>(&w)[i]), sums[j][i]);
        }
    }
    __shared__ float partial[Outputs][Threads / 32];
#pragma unroll
    for (int j = 0; j < Outputs; ++j) {
#pragma unroll
        for (int d = 1; d < width; d *= 2)
#pragma unroll
            for (int i = 0; i < width; i += 2*d) sums[j][i] += sums[j][i+d];
        float value = sums[j][0];
#pragma unroll
        for (int d = 1; d < 32; d *= 2) value += __shfl_down_sync(0xffffffff, value, d);
        if (!lane) partial[j][warp] = value;
    }
    __syncthreads();
    if (!warp) {
#pragma unroll
        for (int j = 0; j < Outputs; ++j) {
            float v = lane < Threads/32 ? partial[j][lane] : 0.f;
#pragma unroll
            for (int d = 1; d < Threads/32; d *= 2) v += __shfl_down_sync(0xffffffff, v, d);
            if (!lane) {
                v = __fmul_rn(__fadd_rn(v, 0.f), invTemp);
                const float sp = v > 20.f ? v : v < -20.f ? expf(v) : log1pf(expf(v));
                scores[size_t(row)*experts+first+j] = sqrtf(sp);
            }
        }
    }
}

template <int Rows, bool Streaming>
__global__ void V41RouterScoresRowsKernel(const __nv_bfloat16 *input, const float *weight, float *scores,
                                          int experts, int dim, float invTemp) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32, expert = blockIdx.x;
    float sums[Rows] = {};
#pragma unroll 4
    for (int k = tid; k < dim; k += 256) {
        float w = Streaming ? __ldcg(weight + (size_t)expert * dim + k) : weight[(size_t)expert * dim + k];
#pragma unroll
        for (int row = 0; row < Rows; ++row)
            sums[row] += __bfloat162float(input[(size_t)row * dim + k]) * w;
    }
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            float other = __shfl_down_sync(0xffffffff, sums[row], offset);
            if ((lane & (2 * offset - 1)) == 0)
                sums[row] += other;
        }
    }
    __shared__ float partial[Rows][8];
    if (lane == 0) {
#pragma unroll
        for (int row = 0; row < Rows; ++row)
            partial[row][warp] = sums[row];
    }
    __syncthreads();
    for (int row = warp; row < Rows; row += 8) {
        float sum = lane < 8 ? partial[row][lane] : 0;
#pragma unroll
        for (int offset = 1; offset < 8; offset *= 2) {
            float other = __shfl_down_sync(0xffffffff, sum, offset);
            if ((lane & (2 * offset - 1)) == 0)
                sum += other;
        }
        if (lane == 0) {
            float value = __fmul_rn(__fadd_rn(sum, 0.f), invTemp);
            float sp = value > 20 ? value : value < -20 ? expf(value) : log1pf(expf(value));
            scores[(size_t)row * experts + expert] = sqrtf(sp);
        }
    }
}
