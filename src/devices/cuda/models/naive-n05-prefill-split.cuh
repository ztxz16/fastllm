#pragma once
#include "naive-n05-prefill-mma.cuh"
namespace naive_prefill_mma {
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
    BF qs[Groups * 8][200];
    BF keys[Tile][200];
    int selected[Tile];
};
template <int Groups> struct PVStorage {
    static constexpr int Tile = 128;
    BF values[Tile][136];
    BF prob[Groups * 8][Tile + 8];
    int selected[Tile];
};
// One CTA per head retains the same 256-lane reduction tree and BF16
// probability rounding, with eight logits per thread instead of sixty-four.
template <int Max> __global__ void SplitNormalize(BF *prob, const float *sink, int heads, int count) {
    const int row = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
    BF *p = prob + ((size_t)row * heads + h) * (Max + 8);
    constexpr int Parts = (Max + 255) / 256;
    float values[Parts], mx = sink ? sink[h] : -INFINITY;
    __shared__ float scratch[256], maximum, denominator;
#pragma unroll
    for (int i = 0; i < Parts; ++i) {
        values[i] = t + i * 256 < count ? (float)p[t + i * 256] : -INFINITY;
        mx = fmaxf(mx, values[i]);
    }
    scratch[t] = mx;
    __syncthreads();
    if (t < 32) {
        mx = fmaxf(fmaxf(scratch[t], scratch[t + 128]), fmaxf(scratch[t + 64], scratch[t + 192]));
        mx = fmaxf(mx, fmaxf(fmaxf(scratch[t + 32], scratch[t + 160]), fmaxf(scratch[t + 96], scratch[t + 224])));
#pragma unroll
        for (int step = 16; step; step >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, step));
        if (t == 0) maximum = mx;
    }
    __syncthreads();
    float sum = 0;
#pragma unroll
    for (int i = 0; i < Parts; ++i) {
        values[i] = t + i * 256 < count ? expf(values[i] - maximum) : 0;
        sum += values[i];
    }
    if (t == 0 && sink) sum += expf(sink[h] - maximum);
    scratch[t] = sum;
    __syncthreads();
    if (t < 32) {
        float a = scratch[t] + scratch[t + 128], b = scratch[t + 64] + scratch[t + 192];
        float c = scratch[t + 32] + scratch[t + 160], d = scratch[t + 96] + scratch[t + 224];
        float total = naive_dsa_decode::WarpSum((a + b) + (c + d));
        if (t == 0) denominator = total;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < Parts; ++i)
        if (t + i * 256 < Max)
            p[t + i * 256] = __float2bfloat16(denominator > 0 ? values[i] / denominator : 0);
}

template <int Max, int Window, int Groups>
__global__ __launch_bounds__(256) void SplitQK(const BF *q, FastllmNaiveKV key, const int *indices, BF *out, int heads,
                                               int kvHeads, int stride, int keys, int count, int past, bool causal,
                                               int indexTile = 0, int queryOffset = 0, int queryRows = 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace naive_attention_bulk;
    constexpr int Tile = QKStorage<Groups>::Tile, Threads = 256;
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
        // Split the eight warps between Q-head groups. Each group retains
        // the same loads, MMA accumulation order and BF16 rounding points.
        constexpr int WarpsPerGroup = 8 / Groups;
        const int group = w / WarpsPerGroup, keyWarp = w % WarpsPerGroup;
        float scores[4] = {};
#pragma unroll
        for (int d = 0; d < 192; d += 16) {
            uint32_t a[4], breg[2];
            ld4(a, &s.keys[keyWarp * 16 + ar][d + ac]);
            ld2(breg, &s.qs[group * 8 + (l & 7)][d + ((l >> 3) & 1) * 8]);
            mma(scores, a, breg);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int h = group * 8 + (l % 4) * 2 + i % 2, k = keyWarp * 16 + l / 4 + (i / 2) * 8;
            if (first + k < Max)
                out[((size_t)row * heads + head + h) * (Max + 8) + first + k] = __float2bfloat16(
                    s.selected[k] >= 0 ? naive_dsa_decode::Round(naive_dsa_decode::Round(scores[i]) * rsqrtf(192.f))
                                       : -INFINITY);
        }
        __syncthreads();
    }

#endif
}
template <int Max, int Window, int Groups>
__global__ __launch_bounds__(256) void SplitPV(const BF *probabilities, FastllmNaiveKV value, const int *indices,
                                               BF *out, int heads, int kvHeads, int keys, int count, int past,
                                               bool causal, int indexTile = 0, int queryOffset = 0, int queryRows = 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace naive_attention_bulk;
    constexpr int Tile = PVStorage<Groups>::Tile, Threads = 256;
    extern __shared__ __align__(16) char buffer[];
    auto &s = *(PVStorage<Groups> *)buffer;
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
    float acc[Groups][4] = {};
    for (int first = 0; first < activeCount; first += Tile) {
        Load<128, 136, Tile, Threads, Window>(s.values, s.selected, value, indices, globalRow, first, count, keys, past,
                                              causal, kvHeads * 128, kh, indexRow);
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
            uint32_t a[4];
            ldt4(a, &s.values[k + (l & 7) + (l >> 4) * 8][w * 16 + ((l >> 3) & 1) * 8]);
#pragma unroll
            for (int g = 0; g < Groups; ++g) {
                uint32_t breg[2];
                ld2(breg, &s.prob[g * 8 + (l & 7)][k + ((l >> 3) & 1) * 8]);
                mma(acc[g], a, breg);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int g = 0; g < Groups; ++g) {
#pragma unroll
        for (int i = 0; i < 4; ++i)
            out[((size_t)globalRow * heads + head + g * 8 + (l % 4) * 2 + i % 2) * 128 + w * 16 + l / 4 + (i / 2) * 8] =
                __float2bfloat16(acc[g][i]);
    }
#endif
}

} // namespace naive_prefill_mma
