#pragma once
// Reuse each FP32 weight across tokens without changing the original 256-lane
// accumulation chains or pairwise reduction tree. Streaming weight reads on
// SM90 preserve L1 space for the input reused by all expert blocks.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
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
