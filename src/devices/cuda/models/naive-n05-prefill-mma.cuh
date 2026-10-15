#pragma once
#include "naive-n05-attention-bulk.cuh"
#include "naive-n05-prefill.cuh"

// Fused SM80+ prefill for eight Q heads sharing one KV head. Reuse the native
// decode MMA primitives, but derive each query from blockIdx (no row-descriptor
// packing or per-request launches). Keep the established BF16 rounding points;
// Tensor Core accumulation order may differ from the SIMT fallback.
namespace naive_prefill_mma {
using BF = __nv_bfloat16;
template<int Max> struct Storage {
    static constexpr int Tile = 128;
    BF qs[8][200];
    union {
        BF keys[Tile][200];
        BF values[Tile][136];
    } tile;
    BF prob[8][Max + 8];
    int selected[Tile];
};
template<int Width, int Pitch, int Tile, int Threads>
__device__ inline void Load(BF (*dst)[Pitch], int *selected, FastllmNaiveKV kv,
        const int *indices, int query, int first, int count, int keys,
        int past, bool causal, int stride, int head, int indexRow) {
    int t = threadIdx.x;
    if (t < Tile) {
        int key = first + t < count ? (indices ? indices[(size_t)indexRow * count + first + t] : first + t) : -1;
        if (key >= keys || (causal && key > past + query)) key = -1;
        // Translate each selected row once, then share it across vector loads.
        selected[t] = key >= 0 ? kv.PhysicalRow(key) : -1;
    }
    __syncthreads();
    for (int i = t; i < Tile * (Width / 8); i += Threads) {
        int key = selected[i / (Width / 8)], d = i % (Width / 8) * 8;
        naive_dsa_decode::Copy<true>(&dst[i / (Width / 8)][d],
            FastllmNaiveKVRow(kv.base, kv.fp8) + ((size_t)max(0, key) * stride + kv.offset + head * Width + d), key >= 0);
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}
template<int Max>
__global__ __launch_bounds__(256) void Generic(const BF *q,
        FastllmNaiveKV key, FastllmNaiveKV value, const int *indices,
        const float *sink, BF *out, int heads, int kvHeads, int stride,
        int keys, int count, int past, bool causal, int indexTile = 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace naive_attention_bulk;
    constexpr int Tile = Storage<Max>::Tile, Threads = 256;
    extern __shared__ __align__(16) char buffer[];
    auto &s = *(Storage<Max> *)buffer;
    int row = blockIdx.x, head = blockIdx.y * 8, kh = head / (heads / kvHeads);
    int indexRow = row;
    if (indexTile) {
        int rank = min(row, (int)gridDim.x - 1 - row) / indexTile;
        indexRow = (2 * rank + (row >= gridDim.x / 2)) * indexTile + row % indexTile;
    }
    int t = threadIdx.x, l = t % 32, w = t / 32;
    for (int i = t; i < 8 * 24; i += Threads)
        *(uint4 *)&s.qs[i / 24][i % 24 * 8] =
            *(const uint4 *)(q + ((size_t)row * heads + head + i / 24) * 192 + i % 24 * 8);
    const int ar = (l & 7) + ((l >> 3) & 1) * 8, ac = (l >> 4) * 8;
    for (int first = 0; first < count; first += Tile) {
        Load<192,200,Tile,Threads>(s.tile.keys, s.selected, key, indices,
            row, first, count, keys, past, causal, stride, kh, indexRow);
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        float acc[4] = {};
        #pragma unroll
        for (int d = 0; d < 192; d += 16) {
            uint32_t a[4], breg[2];
            ld4(a, &s.tile.keys[w * 16 + ar][d + ac]);
            ld2(breg, &s.qs[l & 7][d + ((l >> 3) & 1) * 8]);
            mma(acc, a, breg);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int h = (l % 4) * 2 + i % 2, k = w * 16 + l / 4 + (i / 2) * 8;
            if (first + k < Max)
                s.prob[h][first + k] = __float2bfloat16(s.selected[k] >= 0
                    ? naive_dsa_decode::Round(naive_dsa_decode::Round(acc[i]) * rsqrtf(192.f)) : -INFINITY);
        }
        __syncthreads();
    }
    Normalize<Max, false>(&s.prob[0][0], sink, head, count, w, l);
    __syncthreads();
    float acc[4] = {};
    for (int first = 0; first < count; first += Tile) {
        Load<128,136,Tile,Threads>(s.tile.values, s.selected, value, indices,
            row, first, count, keys, past, causal, kvHeads * 128, kh, indexRow);
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < Tile; k += 16) {
            uint32_t breg[2];
            ld2(breg, &s.prob[l & 7][first + k + ((l >> 3) & 1) * 8]);
            uint32_t a[4];
            ldt4(a, &s.tile.values[k + (l & 7) + (l >> 4) * 8][w * 16 + ((l >> 3) & 1) * 8]);
            mma(acc, a, breg);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        out[((size_t)row * heads + head + (l % 4) * 2 + i % 2) * 128 +
            w * 16 + l / 4 + (i / 2) * 8] = __float2bfloat16(acc[i]);
#endif
}
}
