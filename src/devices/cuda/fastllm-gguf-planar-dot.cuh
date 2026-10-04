#pragma once
#include "fastllm-gguf-kernel-common.cuh"
#include "fastllm-gguf-small-mmvq.cuh"

// Shared arithmetic for separate and fused planar projections. Keep the IQ
// integer rounding and Q2/Q4 minimum correction identical in both paths.
namespace fastllm_gguf_planar {
using namespace fastllm_gguf_small_mmvq;
template <ggml_type Type>
inline constexpr int PlanarTableWords = Type == GGML_TYPE_IQ1_M ? 2048 : CodebookWords<Type>;
__device__ __forceinline__ uint32_t PackIQ1Grid(uint64_t q) {
    // Canonical signed {-1,0,1} -> {0,1,2}. Mask before adding so
    // each byte remains independent, including -1 encoded as 0xff.
    const uint32_t lo = ((uint32_t(q) & 0x03030303u) + 0x01010101u) & 0x03030303u;
    const uint32_t hi = ((uint32_t(q >> 32) & 0x03030303u) + 0x01010101u) & 0x03030303u;
    return lo | (hi << 4);
}
template <ggml_type Type, int Warps> __device__ __forceinline__ void LoadPlanarTable(uint32_t *grid) {
    if constexpr (Type == GGML_TYPE_IQ1_M) {
        for (int i = threadIdx.x; i < 2048; i += Warps * 32)
            grid[i] = PackIQ1Grid(iq1s_grid[i]);
    } else
        LoadCodebook<Type, Warps>(grid);
}
__device__ __forceinline__ int ExpandIQ1Values(uint32_t q, uint32_t offset) {
    // q holds biased values {0,1,2}, one per byte. The biased add
    // remains in [0x77,0x89] and cannot carry across byte boundaries.
    return int(((q << 3) + offset) ^ 0x80808080u);
}
// Mode follows the projection epilogue. Mixed-format gate/up keeps Mode=-1
// to retain its separate register schedule.
template <ggml_type Type, int T, int Mode = -1, bool StreamingWeights = false>
__device__ __forceinline__ void Dot(const void *w, int block, int group, const uint32_t *table,
                                    const int (&a)[T][8], const float (&dx)[T], const int (&loSum)[T],
                                    const int (&hiSum)[T], float (&sum)[T]) {
    int v[8], scale = 0, minimum = 0, sc0 = 0, sc1 = 0;
    float d = 0, dmin = 0;
    if constexpr (Type == GGML_TYPE_IQ4_XS && StreamingWeights) {
        // Gate/up streams two weight matrices. Prefer retaining the reused
        // activations in cache; 32-bit loads preserve the original alignment.
        const auto *b = static_cast<const block_iq4_xs *>(w) + block;
        const auto *header = reinterpret_cast<const uint32_t *>(b);
        const uint32_t dh = __ldcs(header), scales = __ldcs(header + 1);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const auto *q = reinterpret_cast<const uint32_t *>(b->qs) + 4 * group + j;
            const int2 values = LookupIQ4(__ldcs(q));
            v[j] = values.x;
            v[j + 4] = values.y;
        }
        scale = int(((scales >> (4 * group)) & 15) | (((dh >> (16 + 2 * group)) & 3) << 4)) - 32;
        d = __half2float(__ushort_as_half(static_cast<unsigned short>(dh)));
    } else if constexpr (Type == GGML_TYPE_IQ1_M) {
        const auto *b = static_cast<const block_iq1_m *>(w) + block;
        const uint32_t indices = get_int_b4(b->qs, group);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int high = b->qh[2 * group + j / 2] >> (4 * (j % 2));
            const uint32_t g = table[((indices >> (8 * j)) & 255) | ((high & 7) << 8)];
            const uint32_t offset = (high & 8) ? 0x77777777u : 0x79797979u;
            v[2 * j] = ExpandIQ1Values(g & 0x0f0f0f0fu, offset);
            v[2 * j + 1] = ExpandIQ1Values((g >> 4) & 0x0f0f0f0fu, offset);
        }
        const auto *sc = reinterpret_cast<const uint16_t *>(b->scales);
        iq1m_scale_t base;
        base.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0xf0) | ((sc[2] >> 4) & 0xf00) | (sc[3] & 0xf000);
        d = __half2float(base.f16);
        const int factors = sc[group / 2] >> (6 * (group % 2));
        scale = (2 * (factors & 7) + 1) | ((2 * ((factors >> 3) & 7) + 1) << 4);
    } else if constexpr (Type == GGML_TYPE_Q4_K) {
        const auto *b = static_cast<const block_q4_K *>(w) + block;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            v[j] = (get_int_b4(b->qs, (group / 2) * 8 + j) >> (4 * (group % 2))) & 0x0f0f0f0f;
        scale = group < 4 ? (b->scales[group] & 63)
                          : ((b->scales[group + 4] & 15) | ((b->scales[group - 4] >> 6) << 4));
        minimum = group < 4 ? (b->scales[group + 4] & 63)
                            : ((b->scales[group + 4] >> 4) | ((b->scales[group] >> 6) << 4));
        const float2 dm = __half22float2(b->dm);
        d = dm.x;
        dmin = dm.y;
    } else if constexpr (Type == GGML_TYPE_Q2_K) {
        const auto *b = static_cast<const block_q2_K *>(w) + block;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            v[j] = (get_int_b4(b->qs, (group / 4) * 8 + j) >> (2 * (group % 4))) & 0x03030303;
        sc0 = b->scales[2 * group];
        sc1 = b->scales[2 * group + 1];
        const float2 dm = __half22float2(b->dm);
        d = dm.x;
        dmin = dm.y;
    } else
        DecodeBatchWeights<Type, true>(w, block, 2 * group, table, v, d, scale);
    // For s >= 0, trunc((s*x + trunc(x/2))/2) == trunc((2*s+1)*x/4).
    // Hoist the odd multiplier across larger same-format tiles. The int8
    // 32-value dot and 4-bit scale bound its product below 2^24.
    if constexpr (Type == GGML_TYPE_IQ3_XXS && T >= 5 && Mode >= 0)
        scale = 2 * scale + 1;
#pragma unroll
    for (int t = 0; t < T; ++t) {
        if constexpr (Type == GGML_TYPE_IQ2_S || Type == GGML_TYPE_IQ2_XS || Type == GGML_TYPE_IQ1_M ||
                      Type == GGML_TYPE_Q2_K) {
            int lo = 0, hi = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                lo = __dp4a(v[j], a[t][j], lo);
                hi = __dp4a(v[j + 4], a[t][j + 4], hi);
            }
            if constexpr (Type == GGML_TYPE_Q2_K)
                sum[t] += d * (dx[t] * (lo * (sc0 & 15) + hi * (sc1 & 15))) -
                          dmin * (dx[t] * (loSum[t] * (sc0 >> 4) + hiSum[t] * (sc1 >> 4)));
            else if constexpr (Type == GGML_TYPE_IQ1_M) {
                const int dot = lo * (scale & 15) + hi * (scale >> 4);
                sum[t] += (d * dx[t]) * (dot * .125f);
            } else {
                const int dot = (lo * (scale & 15) + hi * (scale >> 4) + (lo + hi) / 2) / 4;
                sum[t] += (d * dx[t]) * dot;
            }
        } else {
            int dot = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                dot = __dp4a(v[j], a[t][j], dot);
            if constexpr (Type == GGML_TYPE_Q4_K)
                sum[t] += d * (dx[t] * (dot * scale)) - dmin * (dx[t] * ((loSum[t] + hiSum[t]) * minimum));
            else {
                if constexpr (Type == GGML_TYPE_IQ3_XXS && T >= 5 && Mode >= 0)
                    dot = (scale * dot) / 4;
                else if constexpr (Type == GGML_TYPE_IQ3_XXS)
                    dot = (scale * dot + dot / 2) / 2;
                else if constexpr (Type == GGML_TYPE_IQ2_XXS)
                    dot = (scale * dot + dot / 2) / 4;
                else
                    dot *= scale;
                sum[t] += (d * dx[t]) * dot;
            }
        }
    }
}

} // namespace fastllm_gguf_planar
