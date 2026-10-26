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
template <int Width, int Pitch, int Rows>
__device__ void LoadSwaUnion(BF (*dst)[Pitch], int *selected, FastllmNaiveKV kv, int first, int keys, int stride,
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
    const int ar = (l & 7) + ((l >> 3) & 1) * 8, ac = (l >> 4) * 8;
    LoadSwaUnion<192, 200, 128 + Queries - 1>(s.tile.k, s.selected, key, past + row + 1 - 128, keys, stride, kh);
#pragma unroll
    for (int r = 0; r < Queries; ++r) {
        float scores[4] = {};
#pragma unroll
        for (int d = 0; d < 192; d += 16) {
            uint32_t a[4], b[2];
            ld4(a, &s.tile.k[r + w * 16 + ar][d + ac]);
            ld2(b, &s.q[r][l & 7][d + ((l >> 3) & 1) * 8]);
            mma(scores, a, b);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int h = (l % 4) * 2 + i % 2, k = w * 16 + l / 4 + (i / 2) * 8;
            s.prob[r][h][k] = __float2bfloat16(
                s.selected[r + k] >= 0 ? naive_dsa_decode::Round(naive_dsa_decode::Round(scores[i]) * rsqrtf(192.f))
                                       : -INFINITY);
        }
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < Queries; ++r)
        Normalize<128, false>(&s.prob[r][0][0], sink, head, 128, w, l);
    __syncthreads();
    LoadSwaUnion<128, 136, 128 + Queries - 1>(s.tile.v, s.selected, value, past + row + 1 - 128, keys, kvHeads * 128,
                                              kh);
    float acc[Queries][4] = {};
#pragma unroll
    for (int k = 0; k < 128; k += 16) {
#pragma unroll
        for (int r = 0; r < Queries; ++r) {
            uint32_t a[4], b[2];
            ldt4(a, &s.tile.v[r + k + (l & 7) + (l >> 4) * 8][w * 16 + ((l >> 3) & 1) * 8]);
            ld2(b, &s.prob[r][l & 7][k + ((l >> 3) & 1) * 8]);
            mma(acc[r], a, b);
        }
    }
#pragma unroll
    for (int r = 0; r < Queries; ++r)
        if (row + r < rows) {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                out[((size_t)(row + r) * heads + head + (l % 4) * 2 + i % 2) * 128 + w * 16 + l / 4 + (i / 2) * 8] =
                    __float2bfloat16(acc[r][i]);
        }
#endif
}
} // namespace naive_prefill_mma
