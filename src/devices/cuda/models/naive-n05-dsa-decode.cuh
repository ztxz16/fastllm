#pragma once
#include "devices/cuda/naive-n05-kv.cuh"
#include "devices/cuda/naive-n05-dsa.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace naive_dsa_decode {
using BF16 = __nv_bfloat16;
__device__ __forceinline__ int Length(const FastllmNaiveDSARow &r) {
    return max(0, min(r.capacity, r.liveKeys ? *r.liveKeys : r.length));
}
// A whole CTA owns this output slice. Test before page translation, index
// reads or barriers: padded rows have no KV allocation or valid tensor map.
__device__ __forceinline__ bool ZeroInactive(const FastllmNaiveDSARow &r, BF16 *out, int count) {
    if (!r.liveKeys || *r.liveKeys > 0) return false;
    for (int i = threadIdx.x; i < count; i += blockDim.x) out[i] = __float2bfloat16(0.f);
    return true;
}
__device__ __forceinline__ float Round(float x) {
    return __bfloat162float(__float2bfloat16(x));
}
__device__ __forceinline__ float WarpSum(float x) {
    for (int s = 16; s; s >>= 1)
        x += __shfl_down_sync(0xffffffffu, x, s);
    return x;
}
template <bool Async> __device__ __forceinline__ void Copy(void *dst, const void *src, bool valid) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if constexpr (Async) {
        unsigned a = (unsigned)__cvta_generic_to_shared(dst);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(a), "l"(src), "r"(valid ? 16 : 0));
        return;
    }
#endif
    *reinterpret_cast<uint4 *>(dst) = valid ? *reinterpret_cast<const uint4 *>(src) : make_uint4(0, 0, 0, 0);
}
template <bool Async> __device__ __forceinline__ void Copy(void *dst, FastllmNaiveKVRow src, bool valid) {
    if (src.fp8) *(uint4 *)dst = valid ? src.Load8() : make_uint4(0, 0, 0, 0);
    else Copy<Async>(dst, src.data, valid);
}
template <int FixedDim = 0>
__global__ void Scores(const BF16 *q, FastllmNaiveDSABatch batch, float *scores, int heads, int kvHeads,
                       int dim, int stride) {
    if constexpr (FixedDim)
        dim = FixedDim;
    const int h = blockIdx.x, b = blockIdx.z, first = blockIdx.y * 64, t = threadIdx.x, lane = t & 31,
              warp = t >> 5;
    auto r = FastllmNaiveResolveDSARow(batch.rows[b]);
    if (Length(r) == 0) return;
    int length = Length(r), count = r.indices ? r.selectedCount : length, kh = h / (heads / kvHeads);
    __shared__ __align__(16) BF16 keys[64][264];
    __shared__ int selected[64];
    if (t < 64) {
        int i = first + t, index = i < count ? (r.indices ? r.indices[i] : i) : -1;
        selected[t] = index >= 0 && index < length ? index : -1;
    }
    __syncthreads();
    FastllmNaiveKV key((const BF16 *)r.key, r.keyPages, r.pageLen, r.fp8);
    if (dim % 8 == 0 && r.keyStride % 8 == 0 && (size_t)key.base % 16 == 0) {
        for (int i = t; i < 64 * (dim / 8); i += 256) {
            int row = i / (dim / 8), d = i % (dim / 8) * 8, index = selected[row];
            Copy<false>(&keys[row][d], key.Row(max(0, index), r.keyStride) + kh * dim + d, index >= 0);
        }
    } else {
        for (int i = t; i < 64 * dim; i += 256) {
            int row = i / dim, d = i % dim, index = selected[row];
            keys[row][d] =
                index >= 0 ? key.Row(index, r.keyStride)[kh * dim + d] : __float2bfloat16(0.f);
        }
    }
    __syncthreads();
    float qreg[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
        qreg[i] =
            lane + i * 32 < dim ? __bfloat162float(q[((size_t)b * heads + h) * dim + lane + i * 32]) : 0.f;
    float keyReg[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int d = 0; d < 8; ++d)
            keyReg[i][d] = lane + d * 32 < dim ? __bfloat162float(keys[warp + i * 8][lane + d * 32]) : 0.f;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int slot = warp + i * 8;
        float dot = 0;
#pragma unroll
        for (int d = 0; d < 8; ++d)
            if (lane + d * 32 < dim)
                dot += qreg[d] * keyReg[i][d];
        dot = WarpSum(dot);
        if (!lane && first + slot < count)
            scores[((size_t)b * heads + h) * stride + first + slot] =
                selected[slot] >= 0 ? Round(Round(dot) * rsqrtf(float(dim))) : -INFINITY;
    }
}

template <int RowsPerWarp, int Warps>
__global__ void DirectScores(const BF16 *q, FastllmNaiveDSABatch batch, float *scores, int heads, int kvHeads,
                             int stride) {
    const int h = blockIdx.x, b = blockIdx.z, t = threadIdx.x, lane = t & 31, warp = t >> 5,
              first = blockIdx.y * RowsPerWarp * Warps;
    const auto r = FastllmNaiveResolveDSARow(batch.rows[b]);
    if (Length(r) == 0) return;
    int length = Length(r), count = r.indices ? r.selectedCount : length, kh = h / (heads / kvHeads);
    FastllmNaiveKV key((const BF16 *)r.key, r.keyPages, r.pageLen, r.fp8);
    float qr[6];
#pragma unroll
    for (int d = 0; d < 6; ++d)
        qr[d] = __bfloat162float(q[((size_t)b * heads + h) * 192 + lane + d * 32]);
#pragma unroll
    for (int i = 0; i < RowsPerWarp; ++i) {
        int slot = first + warp + i * Warps, index = slot < count ? (r.indices ? r.indices[slot] : slot) : -1;
        bool valid = index >= 0 && index < length;
        // Resolve the logical row once; all six vector components share it.
        const auto base = valid ? key.Row(index, r.keyStride) + kh * 192 + lane : FastllmNaiveKVRow{};
        float kr[6];
#pragma unroll
        for (int d = 0; d < 6; ++d)
            kr[d] = valid ? __bfloat162float(base[d * 32]) : 0;
        float dot = 0;
#pragma unroll
        for (int d = 0; d < 6; ++d)
            dot = fmaf(qr[d], kr[d], dot);
        dot = WarpSum(dot);
        if (!lane && slot < count)
            scores[((size_t)b * heads + h) * stride + slot] =
                valid ? Round(Round(dot) * rsqrtf(192.f)) : -INFINITY;
    }
}
// Preserve the original 256-lane reduction tree and BF16 probability rounding.
__device__ __forceinline__ void Normalize(const float *input, float *output, const float *sink, int head,
                                          int count, float *scratch, float &maximum, float &denominator,
                                          int t) {
    float logits[8];
    float mx = sink ? sink[head] : -INFINITY;
    if (t < 256) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            logits[i] = t + i * 256 < count ? input[t + i * 256] : -INFINITY;
            mx = fmaxf(mx, logits[i]);
        }
        scratch[t] = mx;
    }
    __syncthreads();
    if (t < 32) {
        float a = fmaxf(scratch[t], scratch[t + 128]), b = fmaxf(scratch[t + 64], scratch[t + 192]);
        float c = fmaxf(scratch[t + 32], scratch[t + 160]), d = fmaxf(scratch[t + 96], scratch[t + 224]);
        float top = fmaxf(fmaxf(a, b), fmaxf(c, d));
        for (int s = 16; s; s >>= 1)
            top = fmaxf(top, __shfl_down_sync(0xffffffffu, top, s));
        if (!t)
            maximum = top;
    }
    __syncthreads();
    if (t < 256) {
        float sum = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            logits[i] = t + i * 256 < count ? expf(logits[i] - maximum) : 0;
            sum += logits[i];
        }
        if (t == 0 && sink)
            sum += expf(sink[head] - maximum);
        scratch[t] = sum;
    }
    __syncthreads();
    if (t < 32) {
        float a = scratch[t] + scratch[t + 128], b = scratch[t + 64] + scratch[t + 192],
              c = scratch[t + 32] + scratch[t + 160], d = scratch[t + 96] + scratch[t + 224];
        float total = WarpSum((a + b) + (c + d));
        if (!t)
            denominator = total;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 8; ++i)
        if (t < 256 && t + i * 256 < count)
            output[t + i * 256] = denominator > 0 ? Round(logits[i] / denominator) : 0;
}
__global__ void Softmax(float *scores, const float *sink, FastllmNaiveDSABatch batch, int heads, int stride) {
    const int row = blockIdx.x;
    auto r = batch.rows[row / heads];
    if (Length(r) == 0) return;
    int count = r.indices ? r.selectedCount : Length(r);
    __shared__ float scratch[256], maximum, denominator;
    Normalize(scores + (size_t)row * stride, scores + (size_t)row * stride, sink, row % heads, count, scratch,
              maximum, denominator, threadIdx.x);
}
__device__ __forceinline__ void Accumulate8(const float *prob, const BF16 *values, int first, int rows,
                                            int column, float &sum) {
    float p[8], v[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
        if (first + i < rows) {
            p[i] = prob[first + i];
            v[i] = __bfloat162float(values[(first + i) * 8 + column]);
        }
#pragma unroll
    for (int i = 0; i < 8; ++i)
        if (first + i < rows)
            sum = fmaf(p[i], v[i], sum);
}
template <int ValueDim>
__global__ void Values(const float *scores, FastllmNaiveDSABatch batch, BF16 *out, int heads, int kvHeads,
                       int stride) {
    constexpr int Tile = 256, Columns = 8;
    const int h = blockIdx.x, b = blockIdx.z, column = blockIdx.y * Columns, t = threadIdx.x;
    auto r = FastllmNaiveResolveDSARow(batch.rows[b]);
    if (ZeroInactive(r, out + ((size_t)b * heads + h) * ValueDim + column, Columns)) return;
    int length = Length(r), count = r.indices ? r.selectedCount : length, kh = h / (heads / kvHeads);
    FastllmNaiveKV value((const BF16 *)r.value, r.valuePages, r.pageLen, r.fp8);
    const float *p = scores + ((size_t)b * heads + h) * stride;
    __shared__ __align__(16) BF16 values[Tile][Columns];
    __shared__ float prob[Tile];
    float sum = 0;
    for (int first = 0; first < count; first += Tile) {
        int rows = min(Tile, count - first);
        for (int i = t; i < rows; i += 128) {
            int index = r.indices ? r.indices[first + i] : first + i;
            bool valid = index >= 0 && index < length;
            prob[i] = p[first + i];
            Copy<false>(&values[i][0], value.Row(valid ? index : 0, kvHeads * ValueDim) + kh * ValueDim + column,
                        valid);
        }
        __syncthreads();
        if (t < Columns) {
            if (r.indices && heads >= 4 && ValueDim == 128) {
                for (int start = 0; start < rows; start += 64) {
                    float part = 0;
                    for (int i = start; i < min(rows, start + 64); i += 8)
                        Accumulate8(prob, &values[0][0], i, rows, t, part);
                    sum += part;
                }
            } else {
                for (int i = 0; i < rows; i += 8)
                    Accumulate8(prob, &values[0][0], i, rows, t, sum);
            }
        }
        __syncthreads();
    }
    if (t < Columns)
        out[((size_t)b * heads + h) * ValueDim + column + t] = __float2bfloat16(sum);
}
template <int MaxKeys, int Columns> struct TransposedStorage {
    __align__(16) BF16 values[Columns][MaxKeys + 8];
    float scores[MaxKeys], temp[256], maximum, denominator;
};
template <int MaxKeys, int Columns>
__global__ void SoftmaxTransposedValues(const float *scores, const float *sink, FastllmNaiveDSABatch batch,
                                        BF16 *out, int heads, int kvHeads, int stride) {
    const int h = blockIdx.x, b = blockIdx.z, t = threadIdx.x, column = blockIdx.y * Columns;
    const auto r = FastllmNaiveResolveDSARow(batch.rows[b]);
    if (ZeroInactive(r, out + ((size_t)b * heads + h) * 128 + column, Columns)) return;
    const int length = Length(r), count = r.indices ? r.selectedCount : length, kh = h / (heads / kvHeads);
    extern __shared__ __align__(16) unsigned char buffer[];
    auto &st = *(TransposedStorage<MaxKeys, Columns> *)buffer;
    auto &values = st.values;
    auto &prob = st.scores;
    auto &scratch = st.temp;
    auto &maximum = st.maximum;
    auto &denominator = st.denominator;
    FastllmNaiveKV v((const BF16 *)r.value, r.valuePages, r.pageLen, r.fp8);
    const float *src = scores + ((size_t)b * heads + h) * stride;
    for (int i = t; i < count * (Columns / 8); i += 256) {
        int row = i / (Columns / 8), d = i % (Columns / 8) * 8, index = r.indices ? r.indices[row] : row;
        bool valid = index >= 0 && index < length;
        uint4 packed = valid ? (v.Row(valid ? index : 0, kvHeads * 128) + kh * 128 + column + d).Load8()
                             : make_uint4(0, 0, 0, 0);
        BF16 *bf = (BF16 *)&packed;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            values[d + j][row] = bf[j];
    }
    Normalize(src, prob, sink, h, count, scratch, maximum, denominator, threadIdx.x);
    __syncthreads();
    const int c = t;
    const bool active = t < Columns;
    if (active) {
        float total = 0;
        const bool split = r.indices && heads >= 4;
        for (int first = 0; first < count; first += 64) {
            float part = split ? 0 : total;
            int end = min(count, first + 64), start = first;
            for (; start + 16 <= end; start += 16) {
                float ps[16], vs[16];
#pragma unroll
                for (int i = 0; i < 16; i += 8) {
                    *(float4 *)(ps + i) = *(const float4 *)(prob + start + i);
                    *(float4 *)(ps + i + 4) = *(const float4 *)(prob + start + i + 4);
                    uint4 packed = *(const uint4 *)(&values[c][start + i]);
                    BF16 *bf = (BF16 *)&packed;
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        vs[i + j] = __bfloat162float(bf[j]);
                }

#pragma unroll
                for (int i = 0; i < 16; ++i)
                    part = fmaf(ps[i], vs[i], part);
            }
            for (; start < end; ++start)
                part = fmaf(prob[start], __bfloat162float(values[c][start]), part);
            if (split)
                total += part;
            else
                total = part;
        }
        out[((size_t)b * heads + h) * 128 + column + c] = __float2bfloat16(total);
    }
}
template <int MaxKeys, int Columns> struct SparseStorage {
    __align__(16) BF16 values[MaxKeys][Columns];
    float scores[MaxKeys], temp[256], maximum, denominator, partials[32][Columns];
};
template <int MaxKeys, int Columns>
__global__ void SoftmaxSparseValues(const float *scores, const float *sink, FastllmNaiveDSABatch batch,
                                    BF16 *out, int heads, int kvHeads, int stride) {
    const int h = blockIdx.x, b = blockIdx.z, t = threadIdx.x, column = blockIdx.y * Columns;
    const auto r = FastllmNaiveResolveDSARow(batch.rows[b]);
    if (ZeroInactive(r, out + ((size_t)b * heads + h) * 128 + column, Columns)) return;
    const int length = Length(r), count = r.indices ? r.selectedCount : length, kh = h / (heads / kvHeads);
    extern __shared__ __align__(16) unsigned char buffer[];
    auto &st = *(SparseStorage<MaxKeys, Columns> *)buffer;
    auto &values = st.values;
    auto &prob = st.scores;
    auto &scratch = st.temp;
    auto &maximum = st.maximum;
    auto &denominator = st.denominator;
    FastllmNaiveKV v((const BF16 *)r.value, r.valuePages, r.pageLen, r.fp8);
    const float *src = scores + ((size_t)b * heads + h) * stride;
    for (int i = t; i < count * (Columns / 8); i += 32 * Columns) {
        int row = i / (Columns / 8), d = i % (Columns / 8) * 8, index = r.indices ? r.indices[row] : row;
        bool valid = index >= 0 && index < length;
        Copy<true>(&values[row][d], v.Row(valid ? index : 0, kvHeads * 128) + kh * 128 + column + d, valid);
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;" ::);
#endif
    Normalize(src, prob, sink, h, count, scratch, maximum, denominator, threadIdx.x);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;" ::);
#endif
    __syncthreads();
    const int c = t % Columns, part = t / Columns;
    float total = 0;
    if (part < 32) {
        const int start = part * 64;
#pragma unroll
        for (int chunk = 0; chunk < 64; chunk += 16) {
            float ps[16], vs[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                ps[i] = prob[start + chunk + i];
                vs[i] = __bfloat162float(values[start + chunk + i][c]);
            }
#pragma unroll
            for (int i = 0; i < 16; ++i)
                total = fmaf(ps[i], vs[i], total);
        }
        st.partials[part][c] = total;
    }
    __syncthreads();
    if (t < Columns) {
        float result = 0;
#pragma unroll
        for (int p = 0; p < 32; ++p)
            result += st.partials[p][t];
        out[((size_t)b * heads + h) * 128 + column + t] = __float2bfloat16(result);
    }
}

// Share V across two/four query heads. A 256-thread group per head preserves
// softmax's reduction tree; output warps keep the original PV summation order.
template <int MaxKeys, int Groups, int Columns> struct GroupStorage {
    __align__(16) BF16 values[Columns][MaxKeys + 8];
    float probabilities[Groups][MaxKeys], scratch[Groups][256], maximum[Groups], denominator[Groups],
        partials[32][Groups * Columns];
};
template <int MaxKeys, int Groups, int Columns, bool Sparse>
__global__ void GroupedValues(const float *scores, const float *sink, FastllmNaiveDSABatch batch, BF16 *out,
                              int heads, int kvHeads, int stride) {
    const int t = threadIdx.x, b = blockIdx.z, firstHead = blockIdx.x * Groups, column = blockIdx.y * Columns;
    const auto r = FastllmNaiveResolveDSARow(batch.rows[b]);
    if (r.liveKeys && *r.liveKeys <= 0) {
        for (int i = t; i < Groups * Columns; i += blockDim.x)
            out[((size_t)b * heads + firstHead + i / Columns) * 128 + column + i % Columns] = __float2bfloat16(0.f);
        return;
    }
    int length = Length(r), count = r.indices ? r.selectedCount : length, kh = firstHead / (heads / kvHeads);
    extern __shared__ __align__(16) unsigned char memory[];
    auto &s = *(GroupStorage<MaxKeys, Groups, Columns> *)memory;
    FastllmNaiveKV v((const BF16 *)r.value, r.valuePages, r.pageLen, r.fp8);
    for (int i = t; i < count * (Columns / 8); i += Groups * 256) {
        int row = i / (Columns / 8), col = i % (Columns / 8) * 8, index = r.indices ? r.indices[row] : row;
        bool valid = index >= 0 && index < length;
        uint4 packed = valid
                           ? (v.Row(valid ? index : 0, kvHeads * 128) + kh * 128 + column + col).Load8()
                           : make_uint4(0, 0, 0, 0);
        BF16 *bf = (BF16 *)&packed;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            s.values[col + j][row] = bf[j];
    }
    int group = t / 256;
    Normalize(scores + ((size_t)b * heads + firstHead + group) * stride, s.probabilities[group], sink,
              firstHead + group, count, s.scratch[group], s.maximum[group], s.denominator[group], t % 256);
    __syncthreads();
    if constexpr (Sparse) {
        const int item = t % (Groups * Columns), part = t / (Groups * Columns), h = item / Columns,
                  c = item % Columns;
        if (part < 32) {
            float sum = 0;
#pragma unroll
            for (int chunk = 0; chunk < 64; chunk += 16) {
                float ps[16], vs[16];
#pragma unroll
                for (int i = 0; i < 16; i += 8) {
                    *(float4 *)(ps + i) = *(float4 *)(s.probabilities[h] + part * 64 + chunk + i);
                    *(float4 *)(ps + i + 4) = *(float4 *)(s.probabilities[h] + part * 64 + chunk + i + 4);
                    uint4 packed = *(uint4 *)(&s.values[c][part * 64 + chunk + i]);
                    BF16 *bf = (BF16 *)&packed;
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        vs[i + j] = __bfloat162float(bf[j]);
                }
#pragma unroll
                for (int i = 0; i < 16; ++i)
                    sum = fmaf(ps[i], vs[i], sum);
            }
            s.partials[part][item] = sum;
        }
        __syncthreads();
        if (t < Groups * Columns) {
            float sum = 0;
#pragma unroll
            for (int p = 0; p < 32; ++p)
                sum += s.partials[p][t];
            out[((size_t)b * heads + firstHead + t / Columns) * 128 + column + t % Columns] =
                __float2bfloat16(sum);
        }
    } else {
        const int h = t / 32, c = t % 32;
        if (h < Groups && c < Columns) {
            float sum = 0;
            int start = 0;
            for (; start + 16 <= count; start += 16) {
                float ps[16], vs[16];
#pragma unroll
                for (int i = 0; i < 16; i += 8) {
                    *(float4 *)(ps + i) = *(float4 *)(s.probabilities[h] + start + i);
                    *(float4 *)(ps + i + 4) = *(float4 *)(s.probabilities[h] + start + i + 4);
                    uint4 packed = *(uint4 *)(&s.values[c][start + i]);
                    BF16 *bf = (BF16 *)&packed;
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        vs[i + j] = __bfloat162float(bf[j]);
                }
#pragma unroll
                for (int i = 0; i < 16; ++i)
                    sum = fmaf(ps[i], vs[i], sum);
            }
            for (; start < count; ++start)
                sum = fmaf(s.probabilities[h][start], __bfloat162float(s.values[c][start]), sum);
            out[((size_t)b * heads + firstHead + h) * 128 + column + c] = __float2bfloat16(sum);
        }
    }
}

} // namespace naive_dsa_decode
