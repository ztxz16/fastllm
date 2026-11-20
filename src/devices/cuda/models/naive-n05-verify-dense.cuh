#pragma once
#include "naive-n05-attention-bulk.cuh"

namespace naive_attention_bulk {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
// Two queries share a 64-token KV tile. Each retains its own scores, lengths,
// softmax reduction and PV accumulation order.
template<int Max> struct DensePairStorage {
    BF q[2][8][200], prob[2][8][Max + 8];
    union { BF key[64][200]; BF value[64][136]; } tile;
};

template<int Width>
__device__ __forceinline__ void LoadDenseTile(BF *tile, const FastllmNaiveDSARow &r,
        int first, int length, int kh, int kvHeads) {
    constexpr int Pitch = Width + 8;
    const bool value = Width == 128;
    const auto *base = static_cast<const uint8_t *>(value ? r.value : r.key);
    const int stride = value ? kvHeads * Width : r.keyStride;
    for (int i = threadIdx.x; i < 64 * Width / 8; i += 256) {
        int token = first + i / (Width / 8), d = i % (Width / 8) * 8;
        uint4 data{};
        if (token < length) {
            const int physical = FastllmNaivePhysicalRow(r, token, value);
            data = FastllmNaiveKVRow(base + size_t(physical) * stride + kh * Width + d, true).Load8();
        }
        *reinterpret_cast<uint4 *>(tile + (token - first) * Pitch + d) = data;
    }
}

template<int Max>
__global__ __launch_bounds__(256) void DensePair(const BF *q,
#if CUDART_VERSION >= 12010
        const __grid_constant__
#endif
        Batch batch, const float *sink, BF *out, int heads, int kvHeads) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace naive_dsa_decode;
    extern __shared__ __align__(16) char buffer[];
    auto &s = *reinterpret_cast<DensePairStorage<Max> *>(buffer);
    const int firstRow = blockIdx.x * 2, head = blockIdx.y * 8;
    const int t = threadIdx.x, lane = t % 32, warp = t / 32, which = warp / 4, part = warp % 4;
    const int row = firstRow + which, kh = head / (heads / kvHeads);
    const auto r = FastllmNaiveResolveDSARow(batch.rows[firstRow]);
    const int a = Length(batch.rows[firstRow]), b = Length(batch.rows[firstRow + 1]);
    const int length = which ? b : a, total = max(a, b);
    for (int i = t; i < 2 * 8 * 24; i += 256) {
        int j = i / (8 * 24), h = i / 24 % 8, d = i % 24 * 8;
        *reinterpret_cast<uint4 *>(&s.q[j][h][d]) = *reinterpret_cast<const uint4 *>(
            q + (size_t(firstRow + j) * heads + head + h) * 192 + d);
    }
    __syncthreads();
    uint32_t query[12][2];
    #pragma unroll
    for (int d = 0; d < 12; ++d) ld2(query[d], &s.q[which][lane % 8][d * 16 + (lane / 8 % 2) * 8]);
    const int ar = lane % 8 + (lane / 8 % 2) * 8, ac = lane / 16 * 8;
    for (int first = 0; first < total; first += 64) {
        LoadDenseTile<192>(&s.tile.key[0][0], r, first, total, kh, kvHeads);
        __syncthreads();
        float dot[4]{};
        #pragma unroll
        for (int d = 0; d < 12; ++d) {
            uint32_t keys[4];
            ld4(keys, &s.tile.key[part * 16 + ar][d * 16 + ac]);
            mma(dot, keys, query[d]);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int h = lane % 4 * 2 + i % 2, k = first + part * 16 + lane / 4 + i / 2 * 8;
            s.prob[which][h][k] = __float2bfloat16(k < length ? Round(Round(dot[i]) * rsqrtf(192.f)) : -INFINITY);
        }
        __syncthreads();
    }
    Normalize<Max, false>(&s.prob[which][0][0], sink, head, length, part, lane);
    Normalize<Max, false>(&s.prob[which][0][0], sink, head, length, part + 4, lane);
    __syncthreads();
    float sum[2][4]{};
    for (int first = 0; first < total; first += 64) {
        LoadDenseTile<128>(&s.tile.value[0][0], r, first, total, kh, kvHeads);
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < 64; k += 16) {
            uint32_t p[2]; ld2(p, &s.prob[which][lane % 8][first + k + lane / 8 % 2 * 8]);
            #pragma unroll
            for (int block = 0; block < 2; ++block) {
                uint32_t values[4];
                ldt4(values, &s.tile.value[k + lane % 8 + lane / 16 * 8][(part + 4 * block) * 16 + lane / 8 % 2 * 8]);
                // A paired row can end inside this fragment. Zero its masked
                // value lanes before MMA, including NaNs belonging only to its peer.
                if (first + k + 16 > length) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int token = first + k + lane % 4 * 2 + (j / 2) * 8;
                        const unsigned mask = (token < length ? 0xffffu : 0u) |
                                              (token + 1 < length ? 0xffff0000u : 0u);
                        values[j] &= mask;
                    }
                }
                mma(sum[block], values, p);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int block = 0; block < 2; ++block)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int h = head + lane % 4 * 2 + i % 2, d = (part + 4 * block) * 16 + lane / 4 + i / 2 * 8;
            out[(size_t(row) * heads + h) * 128 + d] = __float2bfloat16(sum[block][i]);
        }
#endif
}
#endif
}
