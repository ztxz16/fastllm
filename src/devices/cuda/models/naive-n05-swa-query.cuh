#pragma once
#include "naive-n05-prefill-mma.cuh"
namespace naive_prefill_mma {
template <int Queries> struct SwaQueryStorage {
    BF q[Queries][8][200];
    union {
        BF k[128 + Queries - 1][200];
        BF v[128 + Queries - 1][136];
    } tile;
    BF prob[Queries][8][136];
    int selected[128 + Queries - 1];
};
template <int Width, int Pitch, int Rows, class KV>
__device__ void LoadSwaUnion(BF (*dst)[Pitch], int *selected, KV kv, int first, int keys, int stride,
                             int head) {
    int t = threadIdx.x;
    if (t < Rows) {
        int logical = first + t;
        selected[t] = logical >= 0 && logical < keys ? kv.PhysicalRow(logical) : -1;
    }
    __syncthreads();
    for (int i = t; i < Rows * (Width / 8); i += 256) {
        int row = i / (Width / 8), d = i % (Width / 8) * 8, k = selected[row];
        naive_dsa_decode::Copy<true>(
            &dst[row][d],
            FastllmNaiveKVRow(kv.base, kv.fp8) + ((size_t)max(0, k) * stride + kv.offset + head * Width + d), k >= 0);
    }
    asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
}
template <int Queries>
__device__ __forceinline__ void SwaQueryScores(SwaQueryStorage<Queries> &s, const int *offsets,
        const int *lengths, int count, const float *sink, int head, int w, int l) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace naive_attention_bulk;
    const int ar = (l & 7) + ((l >> 3) & 1) * 8, ac = (l >> 4) * 8;
#pragma unroll
    for (int r = 0; r < Queries; ++r) if (r < count) {
        float scores[4] = {};
#pragma unroll
        for (int d = 0; d < 192; d += 16) {
            uint32_t a[4], b[2];
            ld4(a, &s.tile.k[offsets[r] + w * 16 + ar][d + ac]);
            ld2(b, &s.q[r][l & 7][d + ((l >> 3) & 1) * 8]);
            mma(scores, a, b);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int h = (l % 4) * 2 + i % 2, k = w * 16 + l / 4 + (i / 2) * 8;
            s.prob[r][h][k] = __float2bfloat16(
                k < lengths[r] && s.selected[offsets[r] + k] >= 0 ? naive_dsa_decode::Round(naive_dsa_decode::Round(scores[i]) * rsqrtf(192.f))
                                       : -INFINITY);
        }
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < Queries; ++r) if (r < count)
        Normalize<128, false>(&s.prob[r][0][0], sink, head, lengths[r], w, l);
    __syncthreads();
#endif
}
template <int Queries>
__device__ __forceinline__ void SwaQueryValues(SwaQueryStorage<Queries> &s, const int *offsets,
        BF *out, int row, int count, int heads, int head, int w, int l, const int *lengths = nullptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace naive_attention_bulk;
    float acc[Queries][4] = {};
#pragma unroll
    for (int k = 0; k < 128; k += 16) {
#pragma unroll
        for (int r = 0; r < Queries; ++r) if (r < count) {
            uint32_t a[4], b[2];
            ldt4(a, &s.tile.v[offsets[r] + k + (l & 7) + (l >> 4) * 8][w * 16 + ((l >> 3) & 1) * 8]);
            ld2(b, &s.prob[r][l & 7][k + ((l >> 3) & 1) * 8]);
            mma(acc[r], a, b);
        }
    }
#pragma unroll
    for (int r = 0; r < Queries; ++r)
        if (r < count) {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                out[((size_t)(row + r) * heads + head + (l % 4) * 2 + i % 2) * 128 + w * 16 + l / 4 + (i / 2) * 8] =
                    __float2bfloat16(lengths && !lengths[r] ? 0.f : acc[r][i]);
        }
#endif
}
template <int Queries>
__global__ __launch_bounds__(256) void SwaQueryTile(const BF *q, FastllmNaiveKV key, FastllmNaiveKV value,
                                                    const float *sink, BF *out, int rows, int heads, int kvHeads,
                                                    int stride, int keys, int past) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace naive_attention_bulk;
    extern __shared__ __align__(16) char memory[];
    auto &s = *(SwaQueryStorage<Queries> *)memory;
    int row = blockIdx.x * Queries, head = blockIdx.y * 8, kh = head / (heads / kvHeads);
    int t = threadIdx.x, l = t % 32, w = t / 32;
    for (int i = t; i < Queries * 8 * 24; i += 256) {
        int r = i / (8 * 24), h = i / 24 % 8, d = i % 24 * 8;
        *(uint4 *)&s.q[r][h][d] = row + r < rows
                                      ? *(const uint4 *)(q + ((size_t)(row + r) * heads + head + h) * 192 + d)
                                      : make_uint4(0, 0, 0, 0);
    }
    int offsets[Queries], lengths[Queries];
#pragma unroll
    for (int r = 0; r < Queries; ++r) { offsets[r] = r; lengths[r] = 128; }
    LoadSwaUnion<192, 200, 128 + Queries - 1>(s.tile.k, s.selected, key, past + row + 1 - 128, keys, stride, kh);
    SwaQueryScores(s, offsets, lengths, min(Queries, rows - row), sink, head, w, l);
    LoadSwaUnion<128, 136, 128 + Queries - 1>(s.tile.v, s.selected, value, past + row + 1 - 128, keys, kvHeads * 128,
                                              kh);
    SwaQueryValues(s, offsets, out, row, min(Queries, rows - row), heads, head, w, l);
#endif
}
// Verification keeps committed keys in the ring and candidates after slot128.
// A group contains up to four adjacent queries from the same request; grouping
// never changes the trained draft block or the number of proposed tokens.
struct SwaVerifyGroup { FastllmNaiveDSARow row; int first, count; };
constexpr int SwaVerifyLimit = ((CUDART_VERSION >= 12010 ? 32764 : 4096) - 128) / sizeof(SwaVerifyGroup);
struct SwaVerifyBatch { SwaVerifyGroup groups[SwaVerifyLimit]; };
static_assert(sizeof(SwaVerifyBatch) + 128 <= (CUDART_VERSION >= 12010 ? 32764 : 4096));
struct SwaRingKV {
    const BF *base;
    bool fp8;
    int prefix;
    size_t offset = 0;
    __device__ int PhysicalRow(int logical) const {
        return logical >= prefix ? 128 + logical - prefix : logical & 127;
    }
};
// The large descriptor argument and grid-constant ABI require SM70.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
template <int Queries, bool FP8>
__global__ __launch_bounds__(256) void SwaVerifyQueries(const BF *q,
#if CUDART_VERSION >= 12010
        const __grid_constant__
#endif
        SwaVerifyBatch batch, const float *sink, BF *out, int heads, int kvHeads) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const auto &group = batch.groups[blockIdx.x];
    auto row = FastllmNaiveResolveDSARow(group.row);
    int prefix = -1, end = 0, lengths[Queries]{}, offsets[Queries]{};
    bool compatible = group.count > 1 && row.key && row.value;
#pragma unroll
    for (int r = 0; r < Queries; ++r) if (r < group.count) {
        const int length = row.liveKeys ? row.liveKeys[r] : row.length + r;
        lengths[r] = min(max(length, 0), 128);
        if (length > 0) {
            end = max(end, length);
            const int past = length - row.verifyOffset - r - 1;
            compatible = compatible && past >= 0 && (prefix < 0 || past == prefix);
            prefix = past;
        }
    }
    // Inactive/padded or independently changed dynamic lengths need not form
    // one window. Reuse the unchanged single-query implementation in that case.
    if (!compatible || prefix < 0) {
#pragma unroll
        for (int r = 0; r < Queries; ++r) if (r < group.count) {
            auto single = row;
            if (single.liveKeys) single.liveKeys += r;
            single.length += r; single.verifyOffset += r;
            naive_attention_bulk::DecodeRow<128, true, FP8>(q, single, sink, out,
                heads, kvHeads, group.first + r, blockIdx.y);
            __syncthreads();
        }
        return;
    }
    extern __shared__ __align__(16) char memory[];
    auto &s = *(SwaQueryStorage<Queries> *)memory;
    const int head = blockIdx.y * 8, kh = head / (heads / kvHeads);
    const int t = threadIdx.x, w = t / 32, l = t % 32;
    const int first = max(0, prefix + row.verifyOffset + 1 - 128);
#pragma unroll
    for (int r = 0; r < Queries; ++r)
        offsets[r] = max(0, prefix + row.verifyOffset + r + 1 - 128) - first;
    for (int i = t; i < Queries * 8 * 24; i += 256) {
        const int r = i / (8 * 24), h = i / 24 % 8, d = i % 24 * 8;
        *(uint4 *)&s.q[r][h][d] = r < group.count
            ? *(const uint4 *)(q + ((size_t)(group.first + r) * heads + head + h) * 192 + d)
            : make_uint4(0, 0, 0, 0);
    }
    SwaRingKV key{(const BF *)row.key, FP8, prefix}, value{(const BF *)row.value, FP8, prefix};
    LoadSwaUnion<192, 200, 128 + Queries - 1>(s.tile.k, s.selected, key, first, end, row.keyStride, kh);
    SwaQueryScores(s, offsets, lengths, group.count, sink, head, w, l);
    LoadSwaUnion<128, 136, 128 + Queries - 1>(s.tile.v, s.selected, value, first, end, kvHeads * 128, kh);
    SwaQueryValues(s, offsets, out, group.first, group.count, heads, head, w, l, lengths);
#endif
}
#endif
} // namespace naive_prefill_mma
