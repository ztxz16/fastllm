#pragma once
#include <cuda_bf16.h>
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
} // namespace naive_dsa_mma
