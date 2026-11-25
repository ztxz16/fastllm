#pragma once
#include "naive-n05-verify-softmax.cuh"

namespace naive_exact_qk {
// BF16 operands have exact FP32 products for normal finite model values.
// Check both ends of a conservative FP32 reduction error interval. Borderline
// roundings are recomputed using the original six-FMA / warp-tree order.
constexpr float Error16 = 0x1.1p-20f;
__device__ inline bool UncertainBound(float sum, float error) {
    error += 0x1p-112f;
    return !isfinite(sum) || !isfinite(error) ||
           __bfloat16_as_ushort(__float2bfloat16_rn(sum - error)) !=
               __bfloat16_as_ushort(__float2bfloat16_rn(sum + error));
}
struct ScoreTile {
    float values[4]{}, bounds[4]{};
};
template <auto LoadA, auto LoadB, auto Mma>
__device__ __forceinline__ ScoreTile DotTile(const __nv_bfloat16 (*query)[200], const __nv_bfloat16 (*keys)[200],
                                            int w, int l) {
    ScoreTile tile;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    float magnitudes[4]{};
    const int ar = (l & 7) + ((l >> 3) & 1) * 8, ac = (l >> 4) * 8;
#pragma unroll
    for (int d = 0; d < 192; d += 16) {
        uint32_t a[4], b[2];
        LoadA(a, &keys[w * 16 + ar][d + ac]);
        LoadB(b, &query[l & 7][d + ((l >> 3) & 1) * 8]);
        float previous[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
            previous[j] = tile.values[j];
        Mma(tile.values, a, b);
#pragma unroll
        for (int j = 0; j < 4; ++j)
            a[j] &= 0x7fff7fffU;
        b[0] &= 0x7fff7fffU;
        b[1] &= 0x7fff7fffU;
        float segment[4] = {};
        Mma(segment, a, b);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            magnitudes[j] += segment[j];
            tile.bounds[j] += fabsf(previous[j]) + segment[j];
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
        tile.bounds[j] = (tile.bounds[j] + magnitudes[j]) * Error16;
#endif
    return tile;
}
__device__ __forceinline__ float WarpDot(const __nv_bfloat16 *q, const __nv_bfloat16 *k, int lane) {
    float dot = 0;
#pragma unroll
    for (int i = 0; i < 6; ++i)
        dot = fmaf(float(q[lane + i * 32]), float(k[lane + i * 32]), dot);
    return naive_dsa_decode::WarpSum(dot);
}
// Eight GQA heads by 128 keys. Repair only MMA results whose FP32 error
// interval crosses a BF16 rounding boundary.
template <auto LoadA, auto LoadB, auto Mma, class Valid, class Emit>
__device__ __forceinline__ void Scores(const __nv_bfloat16 (*query)[200], const __nv_bfloat16 (*keys)[200], int w,
                                       int l, Valid valid, Emit emit) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    auto tile = DotTile<LoadA, LoadB, Mma>(query, keys, w, l);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int h = (l % 4) * 2 + i % 2, k = w * 16 + l / 4 + (i / 2) * 8;
        unsigned active =
            __ballot_sync(0xffffffff, valid(k) && UncertainBound(tile.values[i], tile.bounds[i]));
        while (active) {
            int owner = __ffs(active) - 1;
            active &= active - 1;
            int hh = __shfl_sync(0xffffffff, h, owner), kk = __shfl_sync(0xffffffff, k, owner);
            float dot = WarpDot(query[hh], keys[kk], l);
            dot = __shfl_sync(0xffffffff, dot, 0);
            if (l == owner)
                tile.values[i] = dot;
        }
        emit(h, k, valid(k) ? naive_dsa_decode::Round(naive_dsa_decode::Round(tile.values[i]) * rsqrtf(192.f)) : -INFINITY);
    }
#endif
}
} // namespace naive_exact_qk
