#pragma once
// Small-row GQA decode: share decoded KV across heads and use BF16 MMA
// for QK. Recompute ambiguous BF16 roundings in the existing scalar order.
// Sparse PV retains the original 64-token FMA groups and ordered merge.
#include "naive-n05-attention-bulk.cuh"
#include "naive-n05-exact-qk.cuh"
namespace naive_attention_small {
using namespace naive_attention_bulk;
using naive_dsa_decode::Round;
template <int Groups> struct ScoreStorage {
    BF query[Groups][200], keys[128][200];
    int selected[128];
};

// Redistribute the rare scalar repairs across all warps in the CTA.
template <int Groups> struct SparseScoreStorage : ScoreStorage<Groups> {
    int workCount, work[Groups * 128];
};

template <int Groups>
__global__ __launch_bounds__(Groups * 32) void Scores(const BF *query, FastllmNaiveDSABatch batch,
                                                      float *scores, int heads, int kvHeads, int stride) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const int row = blockIdx.z, head = blockIdx.x * Groups, first = blockIdx.y * 128;
    const int t = threadIdx.x, l = t % 32, w = t / 32 % 8, g = t / 256, kh = head / (heads / kvHeads);
    const auto r = FastllmNaiveResolveDSARow(batch.rows[row]);
    const int length = naive_dsa_decode::Length(r), n = r.indices ? r.selectedCount : length;
    if (!length)
        return;
    extern __shared__ __align__(16) unsigned char bytes[];
    auto &s = *(SparseScoreStorage<Groups> *)bytes;
    if (t == 0)
        s.workCount = 0;
    if (t < 128) {
        int k = first + t < n ? (r.indices ? r.indices[first + t] : first + t) : -1;
        s.selected[t] = k >= 0 && k < length ? k : -1;
    }
    for (int i = t; i < Groups * 24; i += Groups * 32) {
        int h = i / 24, d = i % 24 * 8;
        *(uint4 *)&s.query[h][d] = *(const uint4 *)(query + ((size_t)row * heads + head + h) * 192 + d);
    }
    __syncthreads();
    FastllmNaiveKV key(r.key, r.keyPages, r.pageLen, r.fp8, r.fp4);
    for (int i = t; i < 128 * 24; i += Groups * 32) {
        int k = s.selected[i / 24], d = i % 24 * 8;
        naive_dsa_decode::Copy<false>(&s.keys[i / 24][d], key.Row(max(k, 0), r.keyStride) + kh * 192 + d,
                                      k >= 0);
    }
    __syncthreads();
    const auto tile = naive_exact_qk::DotTile<ld4, ld2, mma>(s.query + g * 8, s.keys, w, l);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int h = g * 8 + (l % 4) * 2 + i % 2, k = w * 16 + l / 4 + (i / 2) * 8;
        if (first + k < n)
            scores[((size_t)row * heads + head + h) * stride + first + k] =
                s.selected[k] >= 0 ? Round(Round(tile.values[i]) * rsqrtf(192.f)) : -INFINITY;
        bool bad = s.selected[k] >= 0 && naive_exact_qk::UncertainBound(tile.values[i], tile.bounds[i]);
        unsigned active = __ballot_sync(0xffffffff, bad);
        if (active) {
            int start = 0;
            if (l == 0)
                start = atomicAdd(&s.workCount, __popc(active));
            start = __shfl_sync(0xffffffff, start, 0);
            if (bad)
                s.work[start + __popc(active & ((1u << l) - 1u))] = h * 128 + k;
        }
    }
    __syncthreads();
    for (int task = t / 32; task < s.workCount; task += Groups) {
        const int item = s.work[task], h = item / 128, k = item % 128;
        const float dot = naive_exact_qk::WarpDot(s.query[h], s.keys[k], l);
        if (l == 0)
            scores[((size_t)row * heads + head + h) * stride + first + k] = Round(Round(dot) * rsqrtf(192.f));
    }
#endif
}
struct ValueStorage {
    BF values[64][136], prob[16][72];
    int selected[64];
};
__global__ __launch_bounds__(256) void Values(const float *prob, FastllmNaiveDSABatch batch, float *parts,
                                              int heads, int kvHeads) {
    int row = blockIdx.z, head = blockIdx.x * 16, part = blockIdx.y, t = threadIdx.x,
        kh = head / (heads / kvHeads);
    auto r = FastllmNaiveResolveDSARow(batch.rows[row]);
    int length = naive_dsa_decode::Length(r);
    if (!length) {
        for (int i = t; i < 16 * 128; i += 256) {
            size_t at = (((size_t)row * heads + head + i / 128) * 32 + part) * 128 + i % 128;
            parts[at] = 0;
        }
        return;
    }
    extern __shared__ __align__(16) unsigned char bytes[];
    auto &s = *(ValueStorage *)bytes;
    if (t < 64) {
        int k = r.indices[part * 64 + t];
        s.selected[t] = k >= 0 && k < length ? k : -1;
    }
    for (int i = t; i < 16 * 64; i += 256)
        s.prob[i / 64][i % 64] =
            __float2bfloat16(prob[((size_t)row * heads + head + i / 64) * 2048 + part * 64 + i % 64]);
    __syncthreads();
    FastllmNaiveKV v(r.value, r.valuePages, r.pageLen, r.fp8, r.fp4);
    for (int i = t; i < 64 * 16; i += 256) {
        int k = s.selected[i / 16], d = i % 16 * 8;
        naive_dsa_decode::Copy<false>(&s.values[i / 16][d], v.Row(max(k, 0), kvHeads * 128) + kh * 128 + d,
                                      k >= 0);
    }
    __syncthreads();
    float sums[8] = {};
    int h = t / 16, c = t % 16 * 8;
#pragma unroll
    for (int k = 0; k < 64; ++k) {
        float p = float(s.prob[h][k]);
        uint4 packed = *(const uint4 *)&s.values[k][c];
        const BF *values = (const BF *)&packed;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            sums[i] = fmaf(p, float(values[i]), sums[i]);
    }
    size_t dst = (((size_t)row * heads + head + h) * 32 + part) * 128 + c;
    *(float4 *)(parts + dst) = make_float4(sums[0], sums[1], sums[2], sums[3]);
    *(float4 *)(parts + dst + 4) = make_float4(sums[4], sums[5], sums[6], sums[7]);
}
// Keep the established sum of 32 consecutive 64-token partials.
__global__ void MergeValues(const float *parts, FastllmNaiveDSABatch batch, BF *out, int heads) {
    const int head = blockIdx.x, row = blockIdx.y, column = threadIdx.x;
    const auto r = FastllmNaiveResolveDSARow(batch.rows[row]);
    if (naive_dsa_decode::ZeroInactive(r, out + ((size_t)row * heads + head) * 128, 128))
        return;
    float sum = 0;
#pragma unroll
    for (int part = 0; part < 32; ++part)
        sum += parts[(((size_t)row * heads + head) * 32 + part) * 128 + column];
    out[((size_t)row * heads + head) * 128 + column] = __float2bfloat16(sum);
}
struct SwaStorage : ScoreStorage<8> {
    BF values[128][136], prob[8][136];
};
// Use the same rounding rule for independent and packed candidate rows.
__device__ __forceinline__ void SwaScores(const BF (*query)[200], const BF (*keys)[200], BF (*prob)[136],
                                          const float *sink, int head, int n, int w, int l) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    naive_exact_qk::Scores<ld4, ld2, mma>(query, keys, w, l,
        [=](int k) { return k < n; },
        [&](int h, int k, float value) { prob[h][k] = __float2bfloat16(value); });
    __syncthreads();
    Normalize<128, false>(&prob[0][0], sink, head, n, w, l);
    __syncthreads();
#endif
}
__device__ __forceinline__ void SwaValues(const BF (*values)[136], const BF (*prob)[136], BF *out,
                                          int n, int w, int l) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    float sums[4] = {}, errors[4] = {};
#pragma unroll
    for (int k = 0; k < 128; k += 16) {
        uint32_t a[4], b[2];
        ldt4(a, &values[k + (l & 7) + (l >> 4) * 8][w * 16 + ((l >> 3) & 1) * 8]);
        ld2(b, &prob[l & 7][k + ((l >> 3) & 1) * 8]);
        float previous[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
            previous[j] = sums[j];
        mma(sums, a, b);
        {
#pragma unroll
            for (int j = 0; j < 4; ++j)
                a[j] &= 0x7fff7fffU;
            b[0] &= 0x7fff7fffU;
            b[1] &= 0x7fff7fffU;
            float segment[4] = {};
            mma(segment, a, b);
#pragma unroll
            for (int j = 0; j < 4; ++j)
                errors[j] = errors[j] * (1.f + naive_exact_qk::Error16) + 2.f * naive_exact_qk::Error16 * (fabsf(previous[j]) + segment[j]);
        }
    }
    {
        bool repair = false;
#pragma unroll
        for (int i = 0; i < 4; ++i)
            repair = repair || naive_exact_qk::UncertainBound(sums[i], errors[i]);
        if (repair) {
            float exact[4] = {};
            int h = (l % 4) * 2, c = w * 16 + l / 4;
            for (int k = 0; k < n; ++k) {
                float p0 = float(prob[h][k]), p1 = float(prob[h + 1][k]), v0 = float(values[k][c]),
                      v1 = float(values[k][c + 8]);
                exact[0] = fmaf(p0, v0, exact[0]);
                exact[1] = fmaf(p1, v0, exact[1]);
                exact[2] = fmaf(p0, v1, exact[2]);
                exact[3] = fmaf(p1, v1, exact[3]);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
                sums[i] = exact[i];
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int h = (l % 4) * 2 + i % 2, c = w * 16 + l / 4 + (i / 2) * 8;
        out[h * 128 + c] = __float2bfloat16(sums[i]);
    }

#endif
}
__global__ __launch_bounds__(256) void Swa(const BF *query, FastllmNaiveDSABatch batch, const float *sink,
                                           BF *out, int heads, int kvHeads) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const int row = blockIdx.x, head = blockIdx.y * 8, t = threadIdx.x, l = t % 32, w = t / 32,
              kh = head / (heads / kvHeads);
    const auto r = FastllmNaiveResolveDSARow(batch.rows[row]);
    const int length = r.liveKeys ? *r.liveKeys : r.length, n = min(length, 128);
    if (n <= 0) {
        for (int i = t; i < 8 * 128; i += 256)
            out[((size_t)row * heads + head) * 128 + i] = __float2bfloat16(0.f);
        return;
    }
    extern __shared__ __align__(16) unsigned char bytes[];
    auto &s = *(SwaStorage *)bytes;
    if (t < 128)
        s.selected[t] = index(r, t, length, true);
    for (int i = t; i < 8 * 24; i += 256) {
        int h = i / 24, d = i % 24 * 8;
        *(uint4 *)&s.query[h][d] = *(const uint4 *)(query + ((size_t)row * heads + head + h) * 192 + d);
    }
    __syncthreads();
    FastllmNaiveKV key(r.key, nullptr, 0, r.fp8), value(r.value, nullptr, 0, r.fp8);
    for (int i = t; i < 128 * 24; i += 256) {
        int k = s.selected[i / 24], d = i % 24 * 8;
        naive_dsa_decode::Copy<false>(&s.keys[i / 24][d], key.Row(max(k, 0), r.keyStride) + kh * 192 + d,
                                      k >= 0);
    }
    for (int i = t; i < 128 * 16; i += 256) {
        int k = s.selected[i / 16], d = i % 16 * 8;
        naive_dsa_decode::Copy<false>(&s.values[i / 16][d],
                                      value.Row(max(k, 0), kvHeads * 128) + kh * 128 + d, k >= 0);
    }
    __syncthreads();
    SwaScores(s.query, s.keys, s.prob, sink, head, n, w, l);
    SwaValues(s.values, s.prob, out + (size_t(row) * heads + head) * 128, n, w, l);

#endif
}
} // namespace naive_attention_small
