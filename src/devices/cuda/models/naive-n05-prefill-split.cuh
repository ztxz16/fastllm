#pragma once
#include "naive-n05-prefill-mma.cuh"
namespace naive_prefill_mma {
// Split prefill stages packed KV to BF16 before launch. Keep page lookup and
// offset semantics while issuing native 16-byte copies in the gather loop.
struct SplitBF16KV : FastllmNaiveKV {
    using FastllmNaiveKV::FastllmNaiveKV;
    __device__ __forceinline__ const BF *RowAtPhysicalIndex(int row, int stride) const {
        return base + (size_t)row * stride + offset;
    }
};

// A selected list can contain gaps, duplicates and unsorted indices. Skip
// only a trailing span with no valid keys, preserving the original PV order.
__device__ inline int SplitActiveCount(const int *indices, int indexRow, int count,
        int keys, int position, bool causal, int *scratch) {
    if (!indices) return min(count, causal ? min(keys, position + 1) : keys);
    // Full sparse lists usually end in a live key: avoid scanning those rows.
    const int tail = indices[(size_t)indexRow * count + count - 1];
    if (tail >= 0 && tail < keys && (!causal || tail <= position)) return count;
    const int t = threadIdx.x, lane = t % 32, warp = t / 32;
    int last = 0;
    for (int slot = t; slot < count; slot += blockDim.x) {
        int key = indices[(size_t)indexRow * count + slot];
        if (key >= 0 && key < keys && (!causal || key <= position)) last = slot + 1;
    }
    for (int step = 16; step; step >>= 1)
        last = max(last, __shfl_down_sync(0xffffffff, last, step));
    if (lane == 0) scratch[warp] = last;
    __syncthreads();
    int active = 0;
    for (int i = 0; i < blockDim.x / 32; ++i) active = max(active, scratch[i]);
    __syncthreads(); // All readers finish before the next KV tile reuses scratch.
    return active;
}
template <int Groups> struct QKStorage {
    static constexpr int Tile = Groups == 2 ? 64 : 128;
    static constexpr int Threads = Tile * 2;
    BF qs[Groups * 8][200];
    BF keys[Tile][200];
    int selected[Tile];
};
template <int Groups, int Max = 2048> struct PVStorage {
    static constexpr int Tile = 64, Threads = 128;
    BF values[Tile][136];
    BF prob[Groups * 8][Tile + 8];
    int selected[Max];
};
// Four adjacent BF16 loads per lane. Two warps normalize one head and four
// heads share a CTA. Keep the 256 logical lane sums and the original FP32
// addition tree before rounding the normalized probabilities to BF16.
struct alignas(8) SplitProbVec { BF value[4]; };

template <int Max> __global__ void SplitNormalize(BF *prob, const float *sink, int heads) {
    static_assert(Max % 256 == 0, "Expected complete probability tiles");
    const int group = threadIdx.x / 64, t = threadIdx.x % 64, lane = t % 32;
    const int head = blockIdx.x * 4 + group;
    auto *p = reinterpret_cast<SplitProbVec *>(prob + ((size_t)blockIdx.y * heads + head) * (Max + 8));
    constexpr int Parts = Max / 256;
    float values[Parts][4], sums[4] = {};
    float mx = sink ? sink[head] : -INFINITY;
    __shared__ float warpMax[4][2], scratch[4][256], denominator[4];
#pragma unroll
    for (int i = 0; i < Parts; ++i) {
        const auto v = p[i * 64 + t];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            values[i][j] = (float)v.value[j];
            mx = fmaxf(mx, values[i][j]);
        }
    }
#pragma unroll
    for (int step = 16; step; step >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, step));
    if (lane == 0) warpMax[group][t / 32] = mx;
    __syncthreads();
    mx = fmaxf(warpMax[group][0], warpMax[group][1]);
#pragma unroll
    for (int i = 0; i < Parts; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            values[i][j] = expf(values[i][j] - mx);
            sums[j] += values[i][j];
        }
    }
    if (t == 0 && sink) sums[0] += expf(sink[head] - mx);
    auto &s = scratch[group];
#pragma unroll
    for (int j = 0; j < 4; ++j) s[t * 4 + j] = sums[j];
    __syncthreads();
    if (t < 32) {
        float a = s[t] + s[t + 128], b = s[t + 64] + s[t + 192];
        float c = s[t + 32] + s[t + 160], d = s[t + 96] + s[t + 224];
        float total = naive_dsa_decode::WarpSum((a + b) + (c + d));
        if (t == 0) denominator[group] = total;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < Parts; ++i) {
        SplitProbVec v;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            v.value[j] = __float2bfloat16(denominator[group] > 0 ? values[i][j] / denominator[group] : 0.f);
        p[i * 64 + t] = v;
    }
}

template <int Max, int Window, int Groups, class KV>
__global__ __launch_bounds__(256) void SplitQK(const BF *q, KV key, const int *indices, BF *out, int heads,
                                               int kvHeads, int stride, int keys, int count, int past, bool causal,
                                               int indexTile = 0, int queryOffset = 0, int queryRows = 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace naive_attention_bulk;
    constexpr int Tile = QKStorage<Groups>::Tile, Threads = QKStorage<Groups>::Threads;
    extern __shared__ __align__(16) char buffer[];
    auto &s = *(QKStorage<Groups> *)buffer;
    int row = blockIdx.x, head = blockIdx.y * (Groups * 8), kh = head / (heads / kvHeads);
    const int globalRow = row + queryOffset;
    const int totalRows = queryRows ? queryRows : gridDim.x;
    int indexRow = globalRow;
    if (indexTile) {
        int rank = min(globalRow, totalRows - 1 - globalRow) / indexTile;
        indexRow = (2 * rank + (globalRow >= totalRows / 2)) * indexTile + globalRow % indexTile;
    }
    int t = threadIdx.x, l = t % 32, w = t / 32;
    const int activeCount = Window == 0
        ? SplitActiveCount(indices, indexRow, count, keys, past + globalRow, causal, s.selected) : count;
    for (int i = t; i < Groups * 8 * 24; i += Threads)
        *(uint4 *)&s.qs[i / 24][i % 24 * 8] =
            *(const uint4 *)(q + ((size_t)globalRow * heads + head + i / 24) * 192 + i % 24 * 8);
    // The last computed tile still masks individual lanes. Initialize only
    // the skipped whole tiles so normalization sees the same BF16 -infinities.
    const int computed = (activeCount + Tile - 1) / Tile * Tile;
    const int tailVectors = (Max - computed) / 8;
    for (int i = t; i < Groups * 8 * tailVectors; i += Threads) {
        int h = i / tailVectors, k = computed + i % tailVectors * 8;
        *(uint4 *)(out + ((size_t)row * heads + head + h) * (Max + 8) + k) =
            make_uint4(0xff80ff80u, 0xff80ff80u, 0xff80ff80u, 0xff80ff80u);
    }
    const int ar = (l & 7) + ((l >> 3) & 1) * 8, ac = (l >> 4) * 8;
    for (int first = 0; first < activeCount; first += Tile) {
        Load<192, 200, Tile, Threads, Window>(s.keys, s.selected, key, indices, globalRow, first, count, keys, past, causal,
                                              stride, kh, indexRow);
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        // Each warp loads its 16 K rows once for every Q-head group.
        float scores[Groups][4] = {};
#pragma unroll
        for (int d = 0; d < 192; d += 16) {
            uint32_t a[4];
            ld4(a, &s.keys[w * 16 + ar][d + ac]);
#pragma unroll
            for (int g = 0; g < Groups; ++g) {
                uint32_t breg[2];
                ld2(breg, &s.qs[g * 8 + (l & 7)][d + ((l >> 3) & 1) * 8]);
                mma(scores[g], a, breg);
            }
        }
#pragma unroll
        for (int g = 0; g < Groups; ++g) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int h = g * 8 + (l % 4) * 2 + i % 2, k = w * 16 + l / 4 + (i / 2) * 8;
                if (first + k < Max)
                    out[((size_t)row * heads + head + h) * (Max + 8) + first + k] = __float2bfloat16(
                        s.selected[k] >= 0 ? naive_dsa_decode::Round(naive_dsa_decode::Round(scores[g][i]) * rsqrtf(192.f))
                                           : -INFINITY);
            }
        }
        __syncthreads();
    }

#endif
}
template <int Max, int Window, int Groups, class KV>
__global__ __launch_bounds__(256) void SplitPV(const BF *probabilities, KV value, const int *indices,
                                               BF *out, int heads, int kvHeads, int keys, int count, int past,
                                               bool causal, int indexTile = 0, int queryOffset = 0, int queryRows = 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace naive_attention_bulk;
    constexpr int Tile = PVStorage<Groups,Max>::Tile, Threads = PVStorage<Groups,Max>::Threads;
    extern __shared__ __align__(16) char buffer[];
    auto &s = *(PVStorage<Groups,Max> *)buffer;
    int row = blockIdx.x, head = blockIdx.y * (Groups * 8), kh = head / (heads / kvHeads);
    const int globalRow = row + queryOffset;
    const int totalRows = queryRows ? queryRows : gridDim.x;
    int indexRow = globalRow;
    if (indexTile) {
        int rank = min(globalRow, totalRows - 1 - globalRow) / indexTile;
        indexRow = (2 * rank + (globalRow >= totalRows / 2)) * indexTile + globalRow % indexTile;
    }
    int t = threadIdx.x, l = t % 32, w = t / 32;
    const int activeCount = Window == 0
        ? SplitActiveCount(indices, indexRow, count, keys, past + globalRow, causal, s.selected) : count;
    // Publish the physical rows once so each tile can start its copy without
    // a separate CTA barrier. Mask also the padding of the last partial tile.
    const int padded = (activeCount + Tile - 1) / Tile * Tile;
    for (int i = t; i < padded; i += Threads)
        s.selected[i] = SelectedRow<Window>(value, indices, indexRow, i, count, keys, past + globalRow, causal);
    __syncthreads();
    float acc[2][Groups][4] = {};
    for (int first = 0; first < activeCount; first += Tile) {
        CopySelected<128,136,Tile,Threads>(s.values, s.selected + first, value, kvHeads * 128, kh);
        // Keep the normalized BF16 values produced by SplitNormalize.
        for (int i = t; i < Groups * 8 * (Tile / 8); i += Threads) {
            int h = i / (Tile / 8), col = i % (Tile / 8) * 8;
            naive_dsa_decode::Copy<true>(
                &s.prob[h][col], probabilities + ((size_t)row * heads + head + h) * (Max + 8) + first + col, true);
        }
        asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
#pragma unroll
        for (int k = 0; k < Tile; k += 16) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                uint32_t a[4];
                ldt4(a, &s.values[k + (l & 7) + (l >> 4) * 8][(w + half * 4) * 16 + ((l >> 3) & 1) * 8]);
#pragma unroll
                for (int g = 0; g < Groups; ++g) {
                    uint32_t breg[2];
                    ld2(breg, &s.prob[g * 8 + (l & 7)][k + ((l >> 3) & 1) * 8]);
                    mma(acc[half][g], a, breg);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int half = 0; half < 2; ++half) {
#pragma unroll
        for (int g = 0; g < Groups; ++g) {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                out[((size_t)globalRow * heads + head + g * 8 + (l % 4) * 2 + i % 2) * 128 + (w + half * 4) * 16 + l / 4 + (i / 2) * 8] =
                    __float2bfloat16(acc[half][g][i]);
        }
    }
#endif
}

} // namespace naive_prefill_mma
