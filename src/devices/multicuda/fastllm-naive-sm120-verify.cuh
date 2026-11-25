#pragma once

// Included beside the TP8 collective primitives. The grid size is bounded by the
// actual residency on each GPU. No producer spins on an unscheduled block.
namespace sm120_verify {
using BF = __nv_bfloat16;
constexpr int Threads = 512, Heads = 8, QDim = 192, VDim = 128;
struct Args {
    FastllmNaiveDSABatch batch;
    const BF *input;
    const void *qkvWeight, *outWeight;
    const float *qkvScales, *outScales, *pos, *sink, *gamma;
    BF *query, *attn, *projection, *hidden, *normed;
    FastllmMxfp8Activation quant, prepared;
    float *scores, *partials;
    const uint8_t *swaScales;
    const __nv_bfloat162 *rotary;
    CustomArRankSignals signals;
    CustomArRankScratch exchange;
    const BF *iqWeight, *ikWeight, *iwWeight;
    const float *indexNorm, *indexBias;
    BF *indexRaw;
    uint8_t *iqBytes;
    float *iqScale;
    unsigned *indexScores;
    unsigned long long *lists;
    int *selected;
    int rows, rank, stride, shardWidth;
    bool scaledIndex;
    float theta, valueScale, eps;
};

__device__ void Pack(const BF *input, int width, const Args &a) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < a.rows * width / 8; i += gridDim.x * blockDim.x)
        fastllm_mxfp8::Store8(a.quant, i / (width / 8), i * 8 % width, reinterpret_cast<const uint4 *>(input)[i]);
}
__device__ __forceinline__ void StoreOutput(const Args &a, int i, float x) {
    const BF rounded[] = {__float2bfloat16_rn(x)};
    a.attn[i] = rounded[0];
    auto packed = a.quant;
    packed.width = 1024;
    fastllm_mxfp8::StoreGroup(packed, i / 1024, i % 1024, rounded);
}
__device__ inline void NativeStep(float (&d)[4], const uint8_t *lo, const uint8_t *hi, const uint8_t *in, int sfa,
                                  int sfb) {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200 &&                                                            \
    (defined(__CUDA_ARCH_SPECIFIC__) || defined(__CUDA_ARCH_FAMILY_SPECIFIC__))
    int q = threadIdx.x % 4;
    auto A = (const unsigned *)(lo + 4 * q), A1 = (const unsigned *)(hi + 4 * q),
         B = (const unsigned *)(in + 4 * q);
    unsigned short z = 0;
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32."
                 "ue8m0 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%11},{%12},{%11,%11};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(A[0]), "r"(A1[0]), "r"(A[4]), "r"(A1[4]), "r"(B[0]), "r"(B[4]), "r"(sfa), "h"(z), "r"(sfb));
#else
    asm("trap;");
#endif
}

template <bool Ring>
__device__ __forceinline__ void CommitQkv(const Args &a, int tile, const float *values) {
    const int t = threadIdx.x;
    if (t < 128) {
        const int r = t % 8, oc = t / 8;
        const int col = fastllm_swa_mma::QkvColumn(tile, oc);
        float val = values[t];
        auto desc = FastllmNaiveResolveDSARow(a.batch.rows[r]);
        const int live = *desc.liveKeys;
        if (tile < 36) {
            const int d = tile % 4 * 8 + oc % 8;
            float c, s;
            if constexpr (Ring) {
                const float2 cs = __bfloat1622float2(a.rotary[r * 32 + d]);
                c = cs.x;
                s = cs.y;
            } else {
                const float angle = a.pos[r] * powf(a.theta, -2.f * d / 64.f);
                c = CustomArNormRound(cosf(angle));
                s = CustomArNormRound(sinf(angle));
            }
            const float other = values[t ^ 64];
            val = CustomArNormRound(val * c) + (oc < 8 ? -1.f : 1.f) * CustomArNormRound(other * s);
        }
        if (col < Heads * QDim)
            a.query[r * Heads * QDim + col] = __float2bfloat16_rn(live > 0 ? val : 0.f);
        else if (live > 0 && live <= desc.capacity) {
            const bool value = col >= 1728;
            const int dst =
                Ring ? 128 + desc.verifyOffset : FastllmNaivePhysicalRow(desc, live - 1, value);
            FastllmNaiveKVWriteRow p{const_cast<void *>(value ? desc.value : desc.key), desc.fp8};
            if (value)
                p[dst * VDim + col - 1728] = __float2bfloat16_rn(val * a.valueScale);
            else
                p[dst * desc.keyStride + col - 1536] = __float2bfloat16_rn(val);
        }
    }
}

template <bool Ring>
__device__ __forceinline__ void FinishVerifyQkv(const Args &a, int tile, const float (&acc)[4],
                                                  float (&part)[16][128]) {
    const int t = threadIdx.x, warp = t / 32, g = t % 32 / 4, lane = t % 4;
#pragma unroll
    for (int i = 0; i < 4; ++i)
        part[warp][(g + i / 2 * 8) * 8 + lane * 2 + i % 2] = acc[i];
    __syncthreads();
    if (t < 128) {
        const int column = fastllm_swa_mma::QkvColumn(tile, t / 8);
        const int rot = (column / 16) % 2;
        float sum = 0;
#pragma unroll
        for (int s = 0; s < 4; ++s)
            sum += (part[3 * s + rot][t] + part[3 * s + (rot + 1) % 3][t]) + part[3 * s + (rot + 2) % 3][t];
        part[0][t] = CustomArNormRound(sum);
    }
    __syncthreads();
    if constexpr (Ring) {
        if (t < 128)
            a.projection[(t % 8) * 1856 + fastllm_swa_mma::QkvColumn(tile, t / 8)] = __float2bfloat16_rn(part[0][t]);
    } else {
        CommitQkv<false>(a, tile, part[0]);
    }
}

template <bool Ring> __device__ int Token(const FastllmNaiveDSARow &r, int slot, int live) {
    if constexpr (Ring) {
        const int logical = max(0, live - 128) + slot, prefix = live - r.verifyOffset - 1;
        return logical >= prefix ? 128 + logical - prefix : logical % 128;
    }
    const int token = r.indices ? r.indices[slot] : slot;
    return token >= 0 && token < live ? token : -1;
}

template <bool Ring> __device__ int Count(const FastllmNaiveDSARow &r, int live) {
    if (live <= 0)
        return 0;
    return Ring ? min(128, live) : (r.indices ? r.selectedCount : live);
}

// The CTA stages a key tile once for eight heads. Preserve the reference warp dot order.
__device__ void Scores(const Args &a, BF *keys) {
    const int t = threadIdx.x, lane = t % 32, warp = t / 32, head = warp % 8;
    __shared__ int tokens[32];
    for (int tile = blockIdx.x; tile < a.rows * 64; tile += gridDim.x) {
        int row = tile / 64, first = tile % 64 * 32;
        auto r = FastllmNaiveResolveDSARow(a.batch.rows[row]);
        int live = max(0, min(r.capacity, *r.liveKeys));
        if (t < 32) {
            int k = r.indices[first + t];
            tokens[t] = k >= 0 && k < live ? k : -1;
        }
        __syncthreads();
        FastllmNaiveKV cache(r.key, r.keyPages, r.pageLen, true);
        for (int i = t; i < 32 * 24; i += Threads) {
            int slot = i / 24, d = i % 24 * 8, token = tokens[slot];
            *reinterpret_cast<uint4 *>(keys + slot * 200 + d) =
                token < 0 ? make_uint4(0, 0, 0, 0) : (cache.Row(token, r.keyStride) + d).Load8();
        }
        float q[6];
#pragma unroll
        for (int j = 0; j < 6; ++j)
            q[j] = float(a.query[(row * Heads + head) * 192 + lane + j * 32]);
        __syncthreads();
        for (int off = warp / 8 * 16; off < (warp / 8 + 1) * 16; ++off) {
            float dot = 0;
#pragma unroll
            for (int j = 0; j < 6; ++j)
                dot = fmaf(q[j], float(keys[off * 200 + lane + j * 32]), dot);
            dot = CustomArNormWarpSum(dot);
            if (!lane)
                a.scores[(row * Heads + head) * a.stride + first + off] =
                    tokens[off] < 0 ? -INFINITY : CustomArNormRound(CustomArNormRound(dot) * rsqrtf(192.f));
        }
        __syncthreads();
    }
}
__device__ void ValueTiles(const Args &a, BF *values, float *probabilities) {
    constexpr int Tile = 64;
    const int parts = (a.stride + Tile - 1) / Tile, t = threadIdx.x;
    for (int task = blockIdx.x; task < a.rows * parts; task += gridDim.x) {
        int row = task / parts, p = task % parts, first = p * Tile;
        auto r = FastllmNaiveResolveDSARow(a.batch.rows[row]);
        int live = max(0, min(r.capacity, *r.liveKeys)), count = Count<false>(r, live);
        FastllmNaiveKV v(r.value, r.valuePages, r.pageLen, r.fp8, false);
        for (int i = t; i < Tile * 16; i += Threads) {
            int slot = first + i / 16, col = i % 16 * 8;
            int k = slot < count ? Token<false>(r, slot, live) : -1;
            reinterpret_cast<uint4 *>(values)[i] = k < 0 ? make_uint4(0, 0, 0, 0) : (v.Row(k, 128) + col).Load8();
        }
        const int h = t / 64, slot = first + t % 64;
        probabilities[t] = slot < count ? a.scores[(row * 8 + h) * a.stride + slot] : 0.f;
        __syncthreads();
        const int head = t / 128, col = t % 128;
        float lo = 0.f, hi = 0.f;
#pragma unroll 8
        for (int i = 0; i < Tile; ++i) {
            float val = float(values[i * 128 + col]);
            lo = fmaf(probabilities[head * Tile + i], val, lo);
            hi = fmaf(probabilities[(head + 4) * Tile + i], val, hi);
        }
        a.partials[(task * Heads + head) * VDim + col] = lo;
        a.partials[(task * Heads + head + 4) * VDim + col] = hi;
        __syncthreads();
    }
}

} // namespace sm120_verify
#include "fastllm-naive-sm120-indexer.cuh"
#include "fastllm-naive-sm120-projection.cuh"
#include "fastllm-naive-sm120-swa.cuh"
#include "fastllm-naive-sm120-dsa.cuh"
