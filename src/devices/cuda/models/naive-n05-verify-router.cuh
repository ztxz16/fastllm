#pragma once
#include <cfloat>
#include "devices/cuda/fastllm-cuda-ordered-reduce.cuh"

namespace naive_verify_router {
constexpr int Hidden = 4096, Experts = 256, Top = 8;

// Match RouterTile4096's per-lane FMAs and its two ascending reduction trees.
// Four warps per expert, two independent experts per block.
__device__ __forceinline__ float ExactProbability(const __nv_bfloat16 *input, const float *weight,
                                                  float *warpSums) {
    const int t = threadIdx.x % 128;
    float2 acc{};
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        const int col = 2 * t + 256 * k;
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(input + col));
        const float2 b = *reinterpret_cast<const float2 *>(weight + col);
        acc.x = fmaf(a.x, b.x, acc.x);
        acc.y = fmaf(a.y, b.y, acc.y);
    }
    float sum = acc.x + acc.y;
    #pragma unroll
    for (int d = 1; d < 16; d *= 2) sum += __shfl_down_sync(~0u, sum, d);
    if (t % 16 == 0) warpSums[t / 16] = sum;
    __syncthreads();
    sum = warpSums[t % 8];
    #pragma unroll
    for (int d = 1; d < 8; d *= 2) sum += __shfl_down_sync(~0u, sum, d, 8);
    return 1.0 / (1.0 + expf(-sum));
}

__device__ __forceinline__ float RankKey(float probability, float bias) {
    const float key = probability + bias;
    return isfinite(key) ? key : -FLT_MAX;
}

// One warp selects all eight experts. Identical keys prefer the smaller ID.
// Leave the unselected keys in registers for the eighth/ninth margin test.
__device__ __forceinline__ void Select(const float *keys, float (&remaining)[8], int (&ids)[Top],
                                       float (&best)[Top]) {
    const int first = (threadIdx.x % 32) * 8;
    #pragma unroll
    for (int i = 0; i < 8; ++i) remaining[i] = keys[first + i];
    #pragma unroll
    for (int k = 0; k < Top; ++k) {
        auto winner = fastllm::cuda::WarpArgMax(remaining, first, Experts);
        ids[k] = winner.index;
        best[k] = winner.value;
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            if (first + i == winner.index) remaining[i] = -CUDART_INF_F;
    }
}

__global__ void Finish(const float *products, const __nv_bfloat16 *input, const float *weight,
                       const float *bias, int *indices, float *scores, int rows, bool normalize, float scale) {
    const int row = blockIdx.x, expert = threadIdx.x, group = expert / 128;
    const int plane = rows * Experts;
    __shared__ float probabilities[Experts], keys[Experts], selectedKeys[Top], warpSums[2][8];
    __shared__ int selected[Top], candidates[Experts], numCandidates;
    __shared__ unsigned closeRanks;
    const float *p = products + row * Experts + expert;
    float probability = 1.f / (1.f + expf(-((p[0] + p[plane]) + p[2 * plane])));
    probabilities[expert] = isfinite(probability) ? probability : 0.f;
    keys[expert] = RankKey(probability, bias[expert]);
    __syncthreads();

    // Twice the observed maximum absolute key error on the C512 calibration round.
    // This guard also covers near ties within the returned order.
    constexpr float margin = 1.2e-5f;
    if (expert < 32) {
        float remaining[8], best[Top];
        int ids[Top];
        Select(keys, remaining, ids, best);
        unsigned mask = 0;
        #pragma unroll
        for (int k = 0; k < Top - 1; ++k)
            if (best[k + 1] > -FLT_MAX && best[k] - best[k + 1] <= margin) mask |= 3u << k;
        float runnerUp = -CUDART_INF_F;
        #pragma unroll
        for (int i = 0; i < 8; ++i) runnerUp = fmaxf(runnerUp, remaining[i]);
        if (__any_sync(~0u, best[Top - 1] > -FLT_MAX && runnerUp >= best[Top - 1] - margin))
            mask |= 1u << (Top - 1);
        #pragma unroll
        for (int k = 0; k < Top; ++k)
            if (expert == k) { selected[k] = ids[k]; selectedKeys[k] = best[k]; }
        if (expert == 0) { closeRanks = mask; numCandidates = 0; }
    }
    __syncthreads();
    if (closeRanks) {
        bool close = false;
        #pragma unroll
        for (int k = 0; k < Top; ++k)
            close |= (closeRanks & (1u << k)) && fabsf(keys[expert] - selectedKeys[k]) <= margin;
        if (close) candidates[atomicAdd(&numCandidates, 1)] = expert;
        __syncthreads();
        for (int start = 0; start < numCandidates; start += 2) {
            const int i = start + group, e = candidates[min(i, numCandidates - 1)];
            const float value = ExactProbability(input + size_t(row) * Hidden,
                                                  weight + size_t(e) * Hidden, warpSums[group]);
            if (expert % 128 == 0 && i < numCandidates) {
                probabilities[e] = isfinite(value) ? value : 0.f;
                keys[e] = RankKey(value, bias[e]);
            }
            __syncthreads();
        }
        if (expert < 32) {
            float remaining[8], best[Top];
            int ids[Top];
            Select(keys, remaining, ids, best);
            #pragma unroll
            for (int k = 0; k < Top; ++k) if (expert == k) selected[k] = ids[k];
        }
        __syncthreads();
    }
    if (expert < Top) {
        float sum = 0.f;
        if (normalize) {
            #pragma unroll
            for (int k = 0; k < Top; ++k) sum += probabilities[selected[k]];
        }
        const int e = selected[expert];
        indices[row * Top + expert] = e;
        scores[row * Top + expert] = probabilities[e] / (sum > 1.e-20f ? sum : 1.f) * scale;
    }
}
} // namespace naive_verify_router
