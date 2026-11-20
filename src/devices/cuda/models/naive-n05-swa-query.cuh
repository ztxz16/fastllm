#pragma once
#include "naive-n05-prefill-mma.cuh"
#include "devices/cuda/naive-n05-swa-prefill.cuh"
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
template <int Width, int Pitch, int Rows, int Threads = 256, bool Wait = true, class KV>
__device__ void LoadSwaUnion(BF (*dst)[Pitch], int *selected, KV kv, int first, int keys, int stride,
                             int head) {
    int t = threadIdx.x;
    for (int row = t; row < Rows; row += Threads) {
        int logical = first + row;
        selected[row] = logical >= 0 && logical < keys ? kv.PhysicalRow(logical) : -1;
    }
    __syncthreads();
    for (int i = t; i < Rows * (Width / 8); i += Threads) {
        int row = i / (Width / 8), d = i % (Width / 8) * 8, k = selected[row];
        naive_dsa_decode::Copy<true>(
            &dst[row][d],
            FastllmNaiveKVRow(kv.base, kv.fp8) + ((size_t)max(0, k) * stride + kv.offset + head * Width + d), k >= 0);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    if constexpr (Wait) {
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
    }
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
// One warp owns a query. Keep its Q fragments in registers across all key
// tiles and reuse each probability fragment across all output columns. The
// Q/K and P/V MMA order and BF16 boundaries match SwaQueryTile. Once Q is in
// registers, its shared storage becomes K/V storage; prefetch V while the
// unchanged softmax runs. The existing kernel remains the small-shape fallback.
template <int Queries> struct SwaRegisterStorage {
    union {
        BF q[Queries][8][200];
        BF k[128 + Queries - 1][200];
        BF v[128 + Queries - 1][136];
    } tile;
    BF prob[Queries][8][136];
    int selected[128 + Queries - 1];
};
template <int Queries>
__global__ __launch_bounds__(Queries * 32) void SwaRegisterTile(const BF *q, FastllmNaiveKV key,
                                                                FastllmNaiveKV value, const float *sink,
                                                                BF *out, int rows, int heads, int kvHeads,
                                                                int stride, int keys, int past) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace naive_attention_bulk;
    constexpr int Threads = Queries * 32;
    extern __shared__ __align__(16) char memory[];
    auto &s = *(SwaRegisterStorage<Queries> *)memory;
    const int row = blockIdx.x * Queries, head = blockIdx.y * 8, kh = head / (heads / kvHeads), t = threadIdx.x,
              l = t % 32, r = t / 32;
    for (int i = t; i < Queries * 8 * 24; i += Threads) {
        int rr = i / (8 * 24), h = i / 24 % 8, d = i % 24 * 8;
        *(uint4 *)&s.tile.q[rr][h][d] =
            row + rr < rows ? *(const uint4 *)(q + ((size_t)(row + rr) * heads + head + h) * 192 + d)
                            : make_uint4(0, 0, 0, 0);
    }
    __syncthreads();
    uint32_t query[12][2];
#pragma unroll
    for (int d = 0; d < 12; ++d)
        ld2(query[d], &s.tile.q[r][l & 7][d * 16 + ((l >> 3) & 1) * 8]);
    __syncthreads();
    LoadSwaUnion<192, 200, 128 + Queries - 1, Threads, true>(s.tile.k, s.selected, key, past + row + 1 - 128,
                                                             keys, stride, kh);
    const int ar = (l & 7) + ((l >> 3) & 1) * 8, ac = (l >> 4) * 8;
    if (row + r < rows) {
#pragma unroll
        for (int kb = 0; kb < 8; ++kb) {
            float scores[4] = {};
#pragma unroll
            for (int d = 0; d < 12; ++d) {
                uint32_t a[4];
                ld4(a, &s.tile.k[r + kb * 16 + ar][d * 16 + ac]);
                mma(scores, a, query[d]);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int h = (l % 4) * 2 + i % 2, k = kb * 16 + l / 4 + (i / 2) * 8;
                s.prob[r][h][k] = __float2bfloat16(
                    s.selected[r + k] >= 0
                        ? naive_dsa_decode::Round(naive_dsa_decode::Round(scores[i]) * rsqrtf(192.f))
                        : -INFINITY);
            }
        }
    }
    __syncthreads();
    LoadSwaUnion<128, 136, 128 + Queries - 1, Threads, false>(s.tile.v, s.selected, value, past + row + 1 - 128,
                                                              keys, kvHeads * 128, kh);
    for (int h = r; h < 8; h += Queries) {
#pragma unroll
        for (int rr = 0; rr < Queries; ++rr)
            if (row + rr < rows)
                Normalize<128, false>(&s.prob[rr][0][0], sink, head, 128, h, l);
    }
    __syncthreads();
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    if (row + r < rows) {
        float acc[8][4] = {};
#pragma unroll
        for (int k = 0; k < 128; k += 16) {
            uint32_t b[2];
            ld2(b, &s.prob[r][l & 7][k + ((l >> 3) & 1) * 8]);
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                uint32_t a[4];
                ldt4(a, &s.tile.v[r + k + (l & 7) + (l >> 4) * 8][w * 16 + ((l >> 3) & 1) * 8]);
                mma(acc[w], a, b);
            }
        }
#pragma unroll
        for (int w = 0; w < 8; ++w) {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                out[((size_t)(row + r) * heads + head + (l % 4) * 2 + i % 2) * 128 + w * 16 + l / 4 +
                    (i / 2) * 8] = __float2bfloat16(acc[w][i]);
        }
    }
#endif
}
// One warp per query shares the overlapping chronological K/V windows.
template <int R> struct ContigWindow {
    union {
        BF k[127 + R][200];
        BF v[127 + R][136];
    } tile;
    BF prob[R][8][136];
};
template <int R, int Width, int Pitch>
__device__ __forceinline__ void LoadContiguousWindow(BF (*smem)[Pitch], const BF *base, int first, int keys, int stride,
                                                     int kh) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    for (int i = threadIdx.x; i < (127 + R) * (Width / 8); i += R * 32) {
        const int key = first + i / (Width / 8), d = i % (Width / 8) * 8;
        naive_dsa_decode::Copy<true>(&smem[i / (Width / 8)][d],
                                     base + (size_t)max(0, min(key, keys - 1)) * stride + kh * Width + d,
                                     key >= 0 && key < keys);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}
template <int N> struct StagedSwaBatch { FastllmNaiveStagedSwa entries[N]; };
static_assert(sizeof(StagedSwaBatch<32>) + 64 <= 4096);

template <int R, int N>
__global__ __launch_bounds__(R * 32) void SwaContiguous(StagedSwaBatch<N> batch, const float *sink,
                                                        int heads, int kvHeads) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    int entry = 0, firstTile = 0;
    if constexpr (N > 1) {
        while (blockIdx.x >= batch.entries[entry].endTile) firstTile = batch.entries[entry++].endTile;
    }
    const auto &item = batch.entries[entry];
    const auto *q = (const BF *)item.query, *k = (const BF *)item.key, *v = (const BF *)item.value;
    auto *out = (BF *)item.output;
    const int rows = item.rows, keys = item.keys, stride = item.stride, past = item.past;
    using namespace naive_attention_bulk;
    extern __shared__ __align__(16) char buffer[];
    auto &s = *reinterpret_cast<ContigWindow<R> *>(buffer);
    const int row = (blockIdx.x - firstTile) * R, h = blockIdx.y * 8, kh = h / (heads / kvHeads), r = threadIdx.x / 32,
              l = threadIdx.x % 32;
    const int first = past + row - 127;
    LoadContiguousWindow<R, 192, 200>(s.tile.k, k, first, keys, stride, kh);
    const int qrow = min(row + r, rows - 1);
    const BF *src = q + ((size_t)qrow * heads + h + l / 4) * 192 + 2 * (l % 4);
    uint32_t qr[12][2];
#pragma unroll
    for (int d = 0; d < 12; ++d) {
        qr[d][0] = *(const uint32_t *)(src + d * 16);
        qr[d][1] = *(const uint32_t *)(src + d * 16 + 8);
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    const int ar = (l & 7) + ((l >> 3) & 1) * 8, ac = (l >> 4) * 8;
    BF logits[16][2];
    if (row + r < rows) {
#pragma unroll
        for (int kb = 0; kb < 8; ++kb) {
            float dots[4] = {};
#pragma unroll
            for (int d = 0; d < 12; ++d) {
                uint32_t a[4];
                ld4(a, &s.tile.k[r + kb * 16 + ar][d * 16 + ac]);
                mma(dots, a, qr[d]);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int j = kb * 2 + i / 2;
                BF score = __float2bfloat16(naive_dsa_decode::Round(naive_dsa_decode::Round(dots[i]) * rsqrtf(192.f)));
                logits[j][i % 2] = score;
            }
        }
    }
    __syncthreads();
    LoadContiguousWindow<R, 128, 136>(s.tile.v, v, first, keys, kvHeads * 128, kh);
    if (row + r < rows) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int head = (l % 4) * 2 + half;
            float mx = sink ? sink[h + head] : -INFINITY;
#pragma unroll
            for (int j = 0; j < 16; ++j)
                mx = fmaxf(mx, (float)logits[j][half]);
#pragma unroll
            for (int d = 16; d >= 4; d /= 2)
                mx = fmaxf(mx, __shfl_xor_sync(~0u, mx, d));
            float ex[16];
#pragma unroll
            for (int j = 0; j < 16; ++j) ex[j] = expf((float)logits[j][half] - mx);
            float p[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                float left = ex[j];
                if (j == 0 && sink && l / 4 == 0) left += expf(sink[h + head] - mx);
                p[j] = left + ex[j + 8];
            }
            // Spell out the 8-4-2-1 tree. Indexing its live registers with a
            // variable stride makes nvcc put the normalization array in local memory.
            float den = ((p[0] + p[4]) + (p[2] + p[6])) + ((p[1] + p[5]) + (p[3] + p[7]));
#pragma unroll
            for (int d = 16; d >= 4; d /= 2)
                den += __shfl_down_sync(~0u, den, d);
            den = __shfl_sync(~0u, den, l % 4);
#pragma unroll
            for (int j = 0; j < 16; ++j)
                s.prob[r][head][j * 8 + l / 4] = __float2bfloat16(den > 0 ? ex[j] / den : 0.f);
        }
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    if (row + r < rows) {
        float acc[8][4] = {};
#pragma unroll
        for (int j = 0; j < 128; j += 16) {
            uint32_t b[2];
            ld2(b, &s.prob[r][l & 7][j + ((l >> 3) & 1) * 8]);
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                uint32_t a[4];
                ldt4(a, &s.tile.v[r + j + (l & 7) + (l >> 4) * 8][w * 16 + ((l >> 3) & 1) * 8]);
                mma(acc[w], a, b);
            }
        }
#if __CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12000
        // Transpose the MMA accumulator fragments in the now-dead probability
        // tile. Each lane can then write eight adjacent BF16 output channels.
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            uint32_t pairs[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                auto pair = __floats2bfloat162_rn(acc[w][2 * i], acc[w][2 * i + 1]);
                memcpy(&pairs[i], &pair, sizeof(pair));
            }
            const unsigned addr = __cvta_generic_to_shared(&s.prob[r][l & 7][w * 16 + ((l >> 3) & 1) * 8]);
            asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};"
                :: "r"(addr), "r"(pairs[0]), "r"(pairs[1]) : "memory");
        }
        __syncwarp();
#pragma unroll
        for (int slot = l; slot < 8 * 16; slot += 32) {
            auto *dst = out + ((size_t)(row + r) * heads + h + slot / 16) * 128 + (slot % 16) * 8;
            *reinterpret_cast<uint4 *>(dst) = *reinterpret_cast<const uint4 *>(&s.prob[r][slot / 16][(slot % 16) * 8]);
        }
#else
#pragma unroll
        for (int w = 0; w < 8; ++w) {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                out[((size_t)(row + r) * heads + h + (l % 4) * 2 + i % 2) * 128 + w * 16 + l / 4 + (i / 2) * 8] =
                    __float2bfloat16(acc[w][i]);
        }
#endif
    }
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
