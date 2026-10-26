#pragma once
#include "naive-n05-prefill-mma.cuh"
namespace naive_prefill_mma {
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
template <int Max> __global__ void SplitNormalize(BF *prob, const float *sink, int heads, int count) {
    int row = blockIdx.x, head = blockIdx.y * 8;
    naive_attention_bulk::Normalize<Max, false>(prob + ((size_t)row * heads + head) * (Max + 8), sink, head, count,
                                                threadIdx.x / 32, threadIdx.x % 32);
}
template <int Max, int Window, int Groups>
__global__ __launch_bounds__(256) void SplitQK(const BF *q, FastllmNaiveKV key, const int *indices, BF *out, int heads,
                                               int kvHeads, int stride, int keys, int count, int past, bool causal,
                                               int indexTile = 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace naive_attention_bulk;
    constexpr int Tile = QKStorage<Groups>::Tile, Threads = 256;
    extern __shared__ __align__(16) char buffer[];
    auto &s = *(QKStorage<Groups> *)buffer;
    int row = blockIdx.x, head = blockIdx.y * (Groups * 8), kh = head / (heads / kvHeads);
    int indexRow = row;
    if (indexTile) {
        int rank = min(row, (int)gridDim.x - 1 - row) / indexTile;
        indexRow = (2 * rank + (row >= gridDim.x / 2)) * indexTile + row % indexTile;
    }
    int t = threadIdx.x, l = t % 32, w = t / 32;
    for (int i = t; i < Groups * 8 * 24; i += Threads)
        *(uint4 *)&s.qs[i / 24][i % 24 * 8] =
            *(const uint4 *)(q + ((size_t)row * heads + head + i / 24) * 192 + i % 24 * 8);
    const int ar = (l & 7) + ((l >> 3) & 1) * 8, ac = (l >> 4) * 8;
    for (int first = 0; first < count; first += Tile) {
        Load<192, 200, Tile, Threads, Window>(s.keys, s.selected, key, indices, row, first, count, keys, past, causal,
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
                                               bool causal, int indexTile = 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace naive_attention_bulk;
    constexpr int Tile = PVStorage<Groups>::Tile, Threads = 256;
    extern __shared__ __align__(16) char buffer[];
    auto &s = *(PVStorage<Groups> *)buffer;
    int row = blockIdx.x, head = blockIdx.y * (Groups * 8), kh = head / (heads / kvHeads);
    int indexRow = row;
    if (indexTile) {
        int rank = min(row, (int)gridDim.x - 1 - row) / indexTile;
        indexRow = (2 * rank + (row >= gridDim.x / 2)) * indexTile + row % indexTile;
    }
    int t = threadIdx.x, l = t % 32, w = t / 32;
    float acc[Groups][4] = {};
    for (int first = 0; first < count; first += Tile) {
        Load<128, 136, Tile, Threads, Window>(s.values, s.selected, value, indices, row, first, count, keys, past,
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
            out[((size_t)row * heads + head + g * 8 + (l % 4) * 2 + i % 2) * 128 + w * 16 + l / 4 + (i / 2) * 8] =
                __float2bfloat16(acc[g][i]);
    }
#endif
}

} // namespace naive_prefill_mma
