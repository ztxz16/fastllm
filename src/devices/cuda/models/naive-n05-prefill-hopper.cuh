#pragma once
#include "naive-n05-prefill-mma.cuh"
// Hopper path: retain only a KV tile and online softmax state, reducing shared
// memory versus the generic full-probability kernel. Stage V while computing
// softmax. QK keeps the existing BF16 rounding; online normalization rounds the
// unnormalized PV operand instead, so validate against a bounded-error contract.
namespace naive_prefill_mma {
struct HopperStorage {
    static constexpr int Tile = 128;
    BF qs[8][200];
    union { BF keys[Tile][200]; BF values[Tile][136]; } tile;
    BF prob[8][Tile + 8];
    int selected[Tile];
    float alpha[8], denominator[8];
};
__global__ __launch_bounds__(256) void Hopper(const BF *q,
        FastllmNaiveKV key, FastllmNaiveKV value, const int *indices,
        const float *sink, BF *out, int heads, int kvHeads, int stride,
        int keys, int count, int past, bool causal, int indexTile = 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    using namespace naive_attention_bulk;
    constexpr int Tile = HopperStorage::Tile, Threads = 256;
    extern __shared__ __align__(16) char buffer[];
    auto &s = *(HopperStorage *)buffer;
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
    float maximum = sink ? sink[head + w] : -INFINITY;
    float denominator = sink ? 1.f : 0.f;
    float acc[4] = {};
    if (t < 8) s.denominator[t] = sink ? 1.f : 0.f;
    int activeCount = min(count, causal ? min(keys, past + row + 1) : keys);
    if (indices) {
        // Ignore only a trailing span with no valid keys. Indices need not be
        // sorted or unique; keep their original stride and accumulation order.
        int last = 0;
        for (int slot = t; slot < count; slot += Threads) {
            int key = indices[(size_t)indexRow * count + slot];
            if (key >= 0 && key < keys && (!causal || key <= past + row))
                last = slot + 1;
        }
        #pragma unroll
        for (int step = 16; step; step >>= 1)
            last = max(last, __shfl_down_sync(0xffffffff, last, step));
        if (l == 0) s.selected[w] = last;
        __syncthreads();
        activeCount = 0;
        #pragma unroll
        for (int warp = 0; warp < 8; ++warp) activeCount = max(activeCount, s.selected[warp]);
        __syncthreads();
    }
    for (int first = 0; first < activeCount; first += Tile) {
        Load<192,200,Tile,Threads>(s.tile.keys, s.selected, key, indices,
            row, first, count, keys, past, causal, stride, kh, indexRow);
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        float dot[4] = {};
        #pragma unroll
        for (int d = 0; d < 192; d += 16) {
            uint32_t a[4], breg[2];
            ld4(a, &s.tile.keys[w * 16 + ar][d + ac]);
            ld2(breg, &s.qs[l & 7][d + ((l >> 3) & 1) * 8]);
            mma(dot, a, breg);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int h = (l % 4) * 2 + i % 2, k = w * 16 + l / 4 + (i / 2) * 8;
            s.prob[h][k] = __float2bfloat16(s.selected[k] >= 0
                ? naive_dsa_decode::Round(naive_dsa_decode::Round(dot[i]) * rsqrtf(192.f)) : -INFINITY);
        }
        __syncthreads();
        Load<128,136,Tile,Threads>(s.tile.values, s.selected, value, indices,
            row, first, count, keys, past, causal, kvHeads * 128, kh, indexRow);
        float logits[Tile / 32], m = maximum;
        #pragma unroll
        for (int j = 0; j < Tile / 32; ++j) {
            logits[j] = (float)s.prob[w][l + j * 32];
            m = fmaxf(m, logits[j]);
        }
        #pragma unroll
        for (int step = 16; step; step >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, step));
        float alpha = m == -INFINITY ? 1.f : __expf(maximum - m), den = 0;
        #pragma unroll
        for (int j = 0; j < Tile / 32; ++j) {
            float p = m == -INFINITY ? 0.f : __expf(logits[j] - m);
            s.prob[w][l + j * 32] = __float2bfloat16(p);
            den += p;
        }
        den = naive_dsa_decode::WarpSum(den);
        den = __shfl_sync(0xffffffff, den, 0);
        denominator = denominator * alpha + den;
        maximum = m;
        if (l == 0) { s.alpha[w] = alpha; s.denominator[w] = denominator; }
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            acc[i] *= s.alpha[(l % 4) * 2 + i % 2];
        #pragma unroll
        for (int k = 0; k < Tile; k += 16) {
            uint32_t breg[2];
            ld2(breg, &s.prob[l & 7][k + ((l >> 3) & 1) * 8]);
            uint32_t a[4];
            ldt4(a, &s.tile.values[k + (l & 7) + (l >> 4) * 8][w * 16 + ((l >> 3) & 1) * 8]);
            mma(acc, a, breg);
        }
        // The next tile's index-publication barrier also waits for all
        // PV reads before its K copy overwrites the shared tile.
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int h = (l % 4) * 2 + i % 2;
        float den = s.denominator[h];
        out[((size_t)row * heads + head + h) * 128 + w * 16 + l / 4 + (i / 2) * 8] =
            __float2bfloat16(den > 0 ? acc[i] / den : 0.f);
    }
#endif
}
}
