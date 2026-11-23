#pragma once
#include <cstdint>
#include "fastllm-cuda-router-topk.cuh"

namespace fastllm_shared_flex_tp {
// A warp forms eight disjoint groups
// of 32 experts. The weakest group maximum is a lower bound on the eighth
// score. Two groupings tighten the bound without dropping a possible winner.
__device__ __forceinline__ bool SelectShortlist(const float *router, const float *bias,
        int *ids, float *prob, int needNorm, float scale, fastllm::cuda::RouterTopKScratch<8> &scratch) {
    // Bias views only need float alignment. The reference selector reads them scalarly.
    if (reinterpret_cast<std::uintptr_t>(bias) % 16 != 0) return false;
    constexpr unsigned warp = 0xffffffffu;
    const int lane = threadIdx.x % 32;
    float keys[8], values[8], biases[8];
    fastllm::cuda::LoadRouter8(router + lane * 8, values);
    if (bias) fastllm::cuda::LoadRouter8(bias + lane * 8, biases);
    float best = -FLT_MAX;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        float key = values[j] + (bias ? biases[j] : 0.f);
        keys[j] = isfinite(key) ? key : -FLT_MAX;
        best = fmaxf(best, keys[j]);
    }
    float adjacent = fmaxf(best, __shfl_xor_sync(warp, best, 1));
    adjacent = fmaxf(adjacent, __shfl_xor_sync(warp, adjacent, 2));
    float striped = fmaxf(best, __shfl_xor_sync(warp, best, 8));
    striped = fmaxf(striped, __shfl_xor_sync(warp, striped, 16));
#pragma unroll
    for (int d = 4; d < 32; d *= 2) adjacent = fminf(adjacent, __shfl_xor_sync(warp, adjacent, d));
#pragma unroll
    for (int d = 1; d < 8; d *= 2) striped = fminf(striped, __shfl_xor_sync(warp, striped, d));
    const float cutoff = fmaxf(adjacent, striped);
    unsigned mask = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) if (keys[j] >= cutoff) mask |= 1u << j;
    const int count = __popc(mask);
    int prefix = count;
#pragma unroll
    for (int d = 1; d < 32; d *= 2) {
        int other = __shfl_up_sync(warp, prefix, d);
        if (lane >= d) prefix += other;
    }
    const int total = __shfl_sync(warp, prefix, 31);
    if (total > 32) return false;
    int out = prefix - count;
#pragma unroll
    for (int j = 0; j < 8; ++j) if (mask & (1u << j)) {
        scratch.legacyKeys[out][0] = keys[j];
        scratch.legacyIds[out][0] = float(lane * 8 + j);
        ++out;
    }
    __syncwarp();
    float key = lane < total ? scratch.legacyKeys[lane][0] : -CUDART_INF_F;
    int expert = lane < total ? int(scratch.legacyIds[lane][0]) : 0;
    const float original = router[expert];
    unsigned bits = __float_as_uint(key == 0.f ? 0.f : key);
    unsigned ordered = bits ^ (bits >> 31 ? 0xffffffffu : 0x80000000u);
    unsigned ties = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const unsigned top = __reduce_max_sync(warp, ordered);
        const unsigned winners = __ballot_sync(warp, ordered == top);
        ties |= winners & (winners - 1);
        if (ordered == top) {
            if (lane == __ffs(winners)-1) {
                ids[k] = expert;
                prob[k] = isfinite(original) ? original : 0.f;
            }
            ordered = 0;
        }
    }
    __syncwarp();
    if (ties) return false;
    if (lane < 8) {
        float sum = 1.f;
        if (needNorm) {
            sum = 0.f;
#pragma unroll
            for (int k = 0; k < 8; ++k) sum += prob[k];
            if (!isfinite(sum) || fabsf(sum) < 1.e-20f) sum = 1.f;
        }
        prob[lane] = prob[lane] / sum * scale;
    }
    return true;
}
} // namespace fastllm_shared_flex_tp
