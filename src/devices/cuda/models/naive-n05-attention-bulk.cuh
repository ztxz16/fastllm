#pragma once
// Batched single-query attention, QK=192/V=128. Eight GQA heads share each
// KV tile. This CUDA Tensor Core path requires SM80 and 16-byte aligned
// caches and a GQA group divisible by eight. Unsupported shapes retain the
// existing attention kernels.
// No global score or partial-output workspace is required.
#include "devices/cuda/naive-n05-dsa.h"
#include "naive-n05-dsa-decode.cuh"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
namespace naive_attention_bulk {
using BF = __nv_bfloat16;
#if CUDART_VERSION >= 12010
constexpr int BatchLimit = 256;
#else
constexpr int BatchLimit = (4096 - 128) / sizeof(FastllmNaiveDSARow);
#endif
struct Batch {
    FastllmNaiveDSARow rows[BatchLimit];
};
static_assert(sizeof(Batch) + 128 <= (CUDART_VERSION >= 12010 ? 32764 : 4096),
              "Attention arguments exceed CUDA parameter space");
__device__ inline int index(const FastllmNaiveDSARow &r, int i, int length, bool ring) {
    if (ring)
        return i < min(length, 128) ? ((max(0, length - 128) + i) & 127) : -1;
    int k = i < (r.indices ? r.selectedCount : length) ? (r.indices ? r.indices[i] : i) : -1;
    return k >= 0 && k < length ? k : -1;
}
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ inline void ld4(uint32_t (&r)[4], const void *p) {
    unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}
__device__ inline void ldt4(uint32_t (&r)[4], const void *p) {
    unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
}
__device__ inline void ld2(uint32_t (&r)[2], const void *p) {
    unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(a));
}
__device__ inline void mma(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
#endif
// A warp emulates the previous 256-lane softmax tree. Keep logits and
// normalized probabilities in BF16; MMA changes FP32 accumulation order.
template <int Max, bool FastMath>
__device__ inline void Normalize(BF *prob, const float *sink, int head, int n, int w, int l) {
    using naive_dsa_decode::WarpSum;
    constexpr int Parts = (Max + 255) / 256;
    float e[8][Parts];
    float mx = sink ? sink[head + w] : -INFINITY;
#pragma unroll
    for (int j = 0; j < 8; j++) {
#pragma unroll
        for (int i = 0; i < Parts; i++) {
            int slot = l + j * 32 + i * 256;
            e[j][i] = slot < n ? (float)prob[w * (Max + 8) + slot] : -INFINITY;
            mx = fmaxf(mx, e[j][i]);
        }
    }
#pragma unroll
    for (int d = 16; d; d >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, d));
    float sums[8] = {};
#pragma unroll
    for (int j = 0; j < 8; j++) {
#pragma unroll
        for (int i = 0; i < Parts; i++) {
            e[j][i] = l + j * 32 + i * 256 < n
                          ? (FastMath ? __expf(e[j][i] - mx) : expf(e[j][i] - mx))
                          : 0;
            sums[j] += e[j][i];
        }
    }
    if (l == 0 && sink)
        sums[0] += (FastMath ? __expf(sink[head + w] - mx) : expf(sink[head + w] - mx));
    float den = WarpSum(((sums[0] + sums[4]) + (sums[2] + sums[6])) +
                        ((sums[1] + sums[5]) + (sums[3] + sums[7])));
    den = __shfl_sync(0xffffffff, den, 0);
    const float inv = den > 0 ? __fdividef(1.f, den) : 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) {
#pragma unroll
        for (int i = 0; i < Parts; i++) {
            int slot = l + j * 32 + i * 256;
            if (slot < Max)
                prob[w * (Max + 8) + slot] =
                    __float2bfloat16(den > 0 ? (FastMath ? e[j][i] * inv : e[j][i] / den) : 0);
        }
    }
}
template<bool FP8> struct PackedStorage {};
template<> struct PackedStorage<true> { __align__(16) unsigned char packed[128][192]; };
template <int Max, bool FP8 = false> struct Storage : PackedStorage<FP8> {
    BF qs[8][200];
    union {
        BF keys[128][200];
        BF values[128][136];
    } tile;
    BF prob[8][Max + 8];
    int selected[128];
};
template <int Max, bool Ring, bool FP8, bool AsyncFP8 = true>
__global__ __launch_bounds__(256) void Decode(const BF *q, Batch batch, const float *sink, BF *out,
                                              int heads, int kvHeads) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace naive_dsa_decode;
    extern __shared__ __align__(16) char buffer[];
    auto &s = *(Storage<Max, FP8 && AsyncFP8> *)buffer;
    const int row = blockIdx.x, group = blockIdx.y, head = group * 8, t = threadIdx.x, l = t % 32,
              w = t / 32, kh = head / (heads / kvHeads);
    auto r = FastllmNaiveResolveDSARow(batch.rows[row]);
    if (ZeroInactive(r, out + ((size_t)row * heads + head) * 128, 8 * 128)) return;
    r.fp8 = FP8;
    int len = Ring ? (r.liveKeys ? *r.liveKeys : r.length) : Length(r);
    int n = Ring ? min(len, 128) : (r.indices ? r.selectedCount : len);
    for (int i = t; i < 8 * 24; i += 256) {
        int h = i / 24, d = i % 24 * 8;
        *(uint4 *)&s.qs[h][d] = *(const uint4 *)(q + ((size_t)row * heads + head + h) * 192 + d);
    }
    const int ar = (l & 7) + ((l >> 3) & 1) * 8, ac = (l >> 4) * 8;
    for (int first = 0; first < n; first += 128) {
        if (t < 128) {
            int k = index(r, first + t, len, Ring);
            if constexpr (!Ring)
                if (k >= 0) k = FastllmNaivePhysicalRow(r, k);
            s.selected[t] = k;
        }
        __syncthreads();
        if constexpr (FP8 && AsyncFP8) {
            // Stage packed bytes asynchronously; expand only this shared tile.
            for (int i = t; i < 128 * 12; i += 256) {
                int k = s.selected[i / 12], d = i % 12 * 16;
                Copy<true>(&s.packed[i / 12][d],
                    (const unsigned char *)r.key + (size_t)max(k, 0) * r.keyStride + kh * 192 + d, k >= 0);
            }
            asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
            __syncthreads();
            for (int i = t; i < 128 * 24; i += 256) {
                int row = i / 24, d = i % 24 * 8;
                *(uint4 *)&s.tile.keys[row][d] = FastllmNaiveKVRow(&s.packed[row][d], true).Load8();
            }
        } else {
            for (int i = t; i < 128 * 24; i += 256) {
                int k = s.selected[i / 24], d = i % 24 * 8;
                Copy<true>(&s.tile.keys[i / 24][d],
                           FastllmNaiveKVRow(r.key, r.fp8) + (size_t)max(k, 0) * r.keyStride + kh * 192 + d, k >= 0);
            }
            asm volatile("cp.async.commit_group; cp.async.wait_group 0;");
        }
        __syncthreads();
        float acc[4] = {};
#pragma unroll
        for (int d = 0; d < 192; d += 16) {
            uint32_t a[4], b[2];
            ld4(a, &s.tile.keys[w * 16 + ar][d + ac]);
            ld2(b, &s.qs[l & 7][d + ((l >> 3) & 1) * 8]);
            mma(acc, a, b);
        }
#pragma unroll
        for (int i = 0; i < 4; i++) {
            int h = (l % 4) * 2 + i % 2, k = w * 16 + l / 4 + (i / 2) * 8;
            if (first + k < Max)
                s.prob[h][first + k] = __float2bfloat16(
                    s.selected[k] >= 0 ? Round(Round(acc[i]) * rsqrtf(192.f)) : -INFINITY);
        }
        __syncthreads();
    }
    // One warp owns a head; reproduce the virtual 256-lane softmax tree.
    Normalize<Max, false>(&s.prob[0][0], sink, head, n, w, l);
    __syncthreads();
    float acc[4] = {};
    for (int first = 0; first < n; first += 128) {
        if (t < 128) {
            int k = index(r, first + t, len, Ring);
            if constexpr (!Ring)
                if (k >= 0) k = FastllmNaivePhysicalRow(r, k, true);
            s.selected[t] = k;
        }
        __syncthreads();
        if constexpr (FP8 && AsyncFP8) {
            // Stage packed bytes asynchronously; expand only this shared tile.
            for (int i = t; i < 128 * 8; i += 256) {
                int k = s.selected[i / 8], d = i % 8 * 16;
                Copy<true>(&s.packed[i / 8][d],
                    (const unsigned char *)r.value + ((size_t)max(k, 0) * kvHeads + kh) * 128 + d, k >= 0);
            }
            asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
            __syncthreads();
            for (int i = t; i < 128 * 16; i += 256) {
                int row = i / 16, d = i % 16 * 8;
                *(uint4 *)&s.tile.values[row][d] = FastllmNaiveKVRow(&s.packed[row][d], true).Load8();
            }
        } else {
            for (int i = t; i < 128 * 16; i += 256) {
                int k = s.selected[i / 16], d = i % 16 * 8;
                Copy<true>(&s.tile.values[i / 16][d],
                           FastllmNaiveKVRow(r.value, r.fp8) + ((size_t)max(k, 0) * kvHeads + kh) * 128 + d, k >= 0);
            }
            asm volatile("cp.async.commit_group; cp.async.wait_group 0;");
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < 128; k += 16) {
            uint32_t a[4], b[2];
            ldt4(a, &s.tile.values[k + (l & 7) + (l >> 4) * 8][w * 16 + ((l >> 3) & 1) * 8]);
            ld2(b, &s.prob[l & 7][first + k + ((l >> 3) & 1) * 8]);
            mma(acc, a, b);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
        out[((size_t)row * heads + head + (l % 4) * 2 + i % 2) * 128 + w * 16 + l / 4 +
            (i / 2) * 8] = __float2bfloat16(acc[i]);

#endif
}
} // namespace naive_attention_bulk
