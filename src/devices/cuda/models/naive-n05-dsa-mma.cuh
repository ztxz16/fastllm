#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

// Sparse GQA prefill: sixteen query heads reuse each selected K/V tile.
// Keep materialized logits and the caller's softmax/rounding contract.
namespace naive_dsa_mma {
using BF16 = __nv_bfloat16;
constexpr int kHeads = 16, kKeys = 64, kQkDim = 192, kValueDim = 128;
constexpr int kThreads = 256;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void Load4(uint32_t (&r)[4], const void *p) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void Load2(uint32_t (&r)[2], const void *p) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
        : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void Load2Transpose(uint32_t (&r)[2], const void *p) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
        : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void Mma(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ int SelectedKey(const int *indices, int query, int slot,
                                          int count, int keys, int past, bool causal) {
    if (slot >= count) return -1;
    int key = indices ? indices[(size_t)query * count + slot] : slot;
    return key >= 0 && key < keys && (!causal || key <= past + query) ? key : -1;
}
#endif

__global__ __launch_bounds__(kThreads) void Scores(const BF16 *q, const BF16 *k,
        const int *indices, float *scores, int heads, int kvHeads, int keyStride,
        int keys, int count, int past, bool causal) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(16) BF16 qs[kHeads][kQkDim + 8];
    __shared__ __align__(16) BF16 ks[kKeys][kQkDim + 8];
    __shared__ int valid[kKeys];
    const int query = blockIdx.x, h0 = blockIdx.y * kHeads;
    const int kvHead = h0 / (heads / kvHeads), first = blockIdx.z * kKeys;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int i = threadIdx.x; i < kHeads * (kQkDim / 8); i += kThreads) {
        int h = i / (kQkDim / 8), col = i % (kQkDim / 8) * 8;
        *reinterpret_cast<uint4*>(&qs[h][col]) =
            *reinterpret_cast<const uint4*>(q + ((size_t)query * heads + h0 + h) * kQkDim + col);
    }
    for (int slot = warp; slot < kKeys; slot += kThreads / 32) {
        int key = SelectedKey(indices, query, first + slot, count, keys, past, causal);
        if (lane == 0) valid[slot] = key >= 0;
        if (lane < kQkDim / 8) {
            uint4 values = key >= 0 ? *reinterpret_cast<const uint4*>(
                k + (size_t)key * keyStride + kvHead * kQkDim + lane * 8) : make_uint4(0,0,0,0);
            *reinterpret_cast<uint4*>(&ks[slot][lane * 8]) = values;
        }
    }
    __syncthreads();
    float acc[4] = {};
    const int aRow = (lane & 7) + ((lane >> 3) & 1) * 8, aCol = (lane >> 4) * 8;
    const int bRow = warp * 8 + (lane & 7), bCol = ((lane >> 3) & 1) * 8;
    #pragma unroll
    for (int d = 0; d < kQkDim; d += 16) {
        uint32_t a[4], b[2];
        Load4(a, &qs[aRow][d + aCol]);
        Load2(b, &ks[bRow][d + bCol]);
        Mma(acc, a, b);
    }
    const int row = lane >> 2, col = warp * 8 + (lane & 3) * 2;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int h = row + (i / 2) * 8, slot = col + i % 2;
        if (first + slot < count) {
            float dot = __bfloat162float(__float2bfloat16(acc[i]));
            float scaled = __bfloat162float(__float2bfloat16(dot * rsqrtf(float(kQkDim))));
            scores[((size_t)query * heads + h0 + h) * count + first + slot] = valid[slot] ? scaled : -INFINITY;
        }
    }
#endif
}

__global__ __launch_bounds__(kThreads) void Values(const float *prob, const BF16 *v,
        const int *indices, BF16 *out, int heads, int kvHeads,
        int keys, int count, int past, bool causal) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(16) BF16 ps[kHeads][kKeys + 8];
    __shared__ __align__(16) BF16 vs[kKeys][kValueDim + 8];
    __shared__ int selected[kKeys];
    const int query = blockIdx.x, h0 = blockIdx.y * kHeads;
    const int kvHead = h0 / (heads / kvHeads);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int aRow = (lane & 7) + ((lane >> 3) & 1) * 8, aCol = (lane >> 4) * 8;
    const int bRow = lane & 15;
    float acc[2][4] = {};
    for (int first = 0; first < count; first += kKeys) {
        if (threadIdx.x < kKeys)
            selected[threadIdx.x] = SelectedKey(indices, query, first + threadIdx.x, count, keys, past, causal);
        __syncthreads();
        for (int i = threadIdx.x; i < kHeads * kKeys; i += kThreads) {
            int h = i / kKeys, slot = i % kKeys;
            ps[h][slot] = __float2bfloat16(selected[slot] >= 0
                ? prob[((size_t)query * heads + h0 + h) * count + first + slot] : 0.f);
        }
        for (int i = threadIdx.x; i < kKeys * (kValueDim / 8); i += kThreads) {
            int slot = i / (kValueDim / 8), col = i % (kValueDim / 8) * 8;
            int key = selected[slot];
            uint4 values = key >= 0 ? *reinterpret_cast<const uint4*>(
                v + ((size_t)key * kvHeads + kvHead) * kValueDim + col) : make_uint4(0,0,0,0);
            *reinterpret_cast<uint4*>(&vs[slot][col]) = values;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < kKeys; k += 16) {
            uint32_t a[4];
            Load4(a, &ps[aRow][k + aCol]);
            #pragma unroll
            for (int n = 0; n < 2; ++n) {
                uint32_t b[2];
                Load2Transpose(b, &vs[k + bRow][warp * 16 + n * 8]);
                Mma(acc[n], a, b);
            }
        }
        __syncthreads();
    }
    const int row = lane >> 2, col = warp * 16 + (lane & 3) * 2;
    #pragma unroll
    for (int n = 0; n < 2; ++n) {
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            out[((size_t)query * heads + h0 + row + (i / 2) * 8) * kValueDim + col + n * 8 + i % 2] =
                __float2bfloat16(acc[n][i]);
    }
#endif
}

// Store the original E4M3 quantization result exactly in BF16, with its FP32
// block scale kept separately. Do not round dequantized FP32 values to BF16.
__global__ void QuantizeIndexer(const BF16 *input, BF16 *values, float *scales,
                                int stride, int offset) {
    __shared__ float maximum[128];
    const int d = threadIdx.x, row = blockIdx.x;
    float x = (float)input[(size_t)row * stride + offset + d];
    maximum[d] = fabsf(x);
    __syncthreads();
    for (int step = 64; step; step >>= 1) {
        if (d < step) maximum[d] = fmaxf(maximum[d], maximum[d + step]);
        __syncthreads();
    }
    float scale = fmaxf(maximum[0], 1e-4f) / 448.0f;
    float value = (float)__nv_fp8_e4m3(fmaxf(-448.0f, fminf(448.0f, x / scale)));
    values[(size_t)row * 128 + d] = __float2bfloat16(value);
    if (d == 0) scales[row] = scale;
}

// 64 queries share each 64-key tile across all sixteen indexer heads. The
// FP8 quantizer is unchanged; MMA reduction and scale factoring change FP32 rounding.
// ReLU and weighted head accumulation remain in ascending head order.
__global__ __launch_bounds__(256) void IndexerScores(const BF16 *q, const BF16 *k,
        const float *qscale, const float *kscale, const BF16 *weights,
        float *scores, int queries, int keys, int past) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(16) BF16 qs[64][136], ks[64][136];
    __shared__ float qw[64], qsc[64], ksc[64];
    const int firstQ = blockIdx.y * 64, firstK = blockIdx.x * 64;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    if (firstK > past + min(queries - 1, firstQ + 63)) {
        for (int i = tid; i < 64 * 64; i += 256) {
            int row = firstQ + i / 64, col = firstK + i % 64;
            if (row < queries && col < keys) scores[(size_t)row * keys + col] = -INFINITY;
        }
        return;
    }
    for (int i = tid; i < 64 * 16; i += 256) {
        int row = i / 16, d = (i % 16) * 8;
        *reinterpret_cast<uint4*>(&ks[row][d]) = firstK + row < keys
            ? *reinterpret_cast<const uint4*>(k + (size_t)(firstK + row) * 128 + d)
            : make_uint4(0, 0, 0, 0);
    }
    if (tid < 64) ksc[tid] = firstK + tid < keys ? kscale[firstK + tid] : 0.f;
    const int mTile = warp & 3, nHalf = warp >> 2;
    const int aRow = mTile * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
    const int bRow = nHalf * 32 + (lane & 7) + ((lane >> 3) & 1) * 8;
    const int tileCol = (lane >> 4) * 8;
    float acc[2][2][4] = {};
    for (int h = 0; h < 16; ++h) {
        __syncthreads();
        for (int i = tid; i < 64 * 16; i += 256) {
            int row = i / 16, d = (i % 16) * 8;
            *reinterpret_cast<uint4*>(&qs[row][d]) = firstQ + row < queries
                ? *reinterpret_cast<const uint4*>(q + ((size_t)(firstQ + row) * 16 + h) * 128 + d)
                : make_uint4(0, 0, 0, 0);
        }
        if (tid < 64) {
            qw[tid] = firstQ + tid < queries ? (float)weights[(firstQ + tid) * 16 + h] : 0.f;
            qsc[tid] = firstQ + tid < queries ? qscale[(firstQ + tid) * 16 + h] : 0.f;
        }
        __syncthreads();
        float dot[2][2][4] = {};
        #pragma unroll
        for (int d = 0; d < 128; d += 16) {
            uint32_t a[4]; Load4(a, &qs[aRow][d + tileCol]);
            #pragma unroll
            for (int n = 0; n < 2; ++n) {
                uint32_t b[4]; Load4(b, &ks[bRow + n * 16][d + tileCol]);
                uint32_t b0[2] = {b[0], b[2]}, b1[2] = {b[1], b[3]};
                Mma(dot[n][0], a, b0); Mma(dot[n][1], a, b1);
            }
        }
        #pragma unroll
        for (int n = 0; n < 2; ++n) {
            #pragma unroll
            for (int u = 0; u < 2; ++u) {
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    int row = mTile * 16 + (lane >> 2) + (e / 2) * 8;
                    int col = nHalf * 32 + n * 16 + u * 8 + (lane & 3) * 2 + e % 2;
                    float scaled = (dot[n][u][e] * qsc[row]) * ksc[col];
                    acc[n][u][e] += fmaxf(scaled, 0.f) * qw[row];
                }
            }
        }
    }
    #pragma unroll
    for (int n = 0; n < 2; ++n) {
        #pragma unroll
        for (int u = 0; u < 2; ++u) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                int row = firstQ + mTile * 16 + (lane >> 2) + (e / 2) * 8;
                int col = firstK + nHalf * 32 + n * 16 + u * 8 + (lane & 3) * 2 + e % 2;
                if (row < queries && col < keys)
                    scores[(size_t)row * keys + col] = col <= past + row ? acc[n][u][e] : -INFINITY;
            }
        }
    }
#endif
}

} // namespace naive_dsa_mma
