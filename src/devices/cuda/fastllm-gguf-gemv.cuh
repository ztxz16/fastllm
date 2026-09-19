#pragma once

// Shared GGUF block decoders for matrix dequantization and direct GEMV.
// The includer supplies GGML_COMMON_DECL_CUDA / GGML_COMMON_IMPL_CUDA.
// A pointer output materializes weights; a dot-product output consumes each
// decoded value in registers. This keeps both paths on the same block layout.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "gguf.h"

static constexpr __device__ int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
template<typename dst_t>
struct DequantizeCast;

template<>
struct DequantizeCast<float> {
    static __device__ __forceinline__ float cast(float v) {
        return v;
    }
};

template<>
struct DequantizeCast<half> {
    static __device__ __forceinline__ half cast(float v) {
        return __float2half_rn(v);
    }
};

template<>
struct DequantizeCast<__nv_bfloat16> {
    static __device__ __forceinline__ __nv_bfloat16 cast(float v) {
        return __float2bfloat16_rn(v);
    }
};

static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

template<typename dst_t, typename Output>
static __device__ __forceinline__ void dequantize_block_q4_0_impl(const void * __restrict__ vx,
                                              Output yy,
                                              int64_t blockCount, int64_t blockIndex, int threadIndex) {
    const int64_t group = blockIndex;
    const int64_t il = threadIndex / 8;
    const int64_t ir = threadIndex % 8;
    const int64_t block = 8 * group + ir;
    if (block >= blockCount) {
        return;
    }

    const block_q4_0 *x = (const block_q4_0 *)vx + block;
    auto y = yy + 256 * group + 32 * ir + 4 * il;
    const float d = __half2float(x->d);
    const float minimum = -8.0f * d;
    const uint8_t *q = x->qs + 4 * il;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        y[j] = DequantizeCast<dst_t>::cast(
            d * (q[j] & 0x0f) + minimum);
        y[j + 16] = DequantizeCast<dst_t>::cast(
            d * (q[j] >> 4) + minimum);
    }
}
template<typename dst_t, typename Output>
static __device__ __forceinline__ void dequantize_block_q4_1_impl(const void * __restrict__ vx,
                                              Output yy,
                                              int64_t blockCount, int64_t blockIndex, int threadIndex) {
    const int64_t group = blockIndex;
    const int64_t il = threadIndex / 8;
    const int64_t ir = threadIndex % 8;
    const int64_t block = 8 * group + ir;
    if (block >= blockCount) {
        return;
    }

    const block_q4_1 *x = (const block_q4_1 *)vx + block;
    auto y = yy + 256 * group + 32 * ir + 4 * il;
    const float2 dm = __half22float2(x->dm);
    const uint8_t *q = x->qs + 4 * il;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        y[j] = DequantizeCast<dst_t>::cast(
            dm.x * (q[j] & 0x0f) + dm.y);
        y[j + 16] = DequantizeCast<dst_t>::cast(
            dm.x * (q[j] >> 4) + dm.y);
    }
}
template<typename dst_t, typename Output>
static __device__ __forceinline__ void dequantize_block_iq2_xxs_impl(const void * __restrict__ vx,
                                                 Output yy, int64_t blockIndex, int threadIndex) {
    const int64_t block = blockIndex;
    const block_iq2_xxs *x = (const block_iq2_xxs *)vx + block;
    const int il = threadIndex / 8;
    const int ib = threadIndex % 8;
    auto y = yy + block * QK_K + 32 * ib + 8 * il;

    const uint16_t *q2 = x->qs + 4 * ib;
    const uint32_t low = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
    const uint32_t aux = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
    const unsigned gridIndex = (low >> (8u * (unsigned)il)) & 0xffu;
    const uint64_t grid = iq2xxs_grid[gridIndex];
    const uint8_t signs = ksigns_iq2xs[(aux >> (7 * il)) & 0x7f];
    const float d = __half2float(x->d) * (0.5f + (aux >> 28)) * 0.25f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int q = (int)((grid >> (8u * (unsigned)j)) & 0xffu);
        const float sign = (signs & (1u << j)) ? -1.0f : 1.0f;
        y[j] = DequantizeCast<dst_t>::cast(d * q * sign);
    }
}
template<typename dst_t, typename Output>
static __device__ __forceinline__ void dequantize_block_iq2_xs_impl(const void * __restrict__ vx,
                                                Output yy, int64_t blockIndex, int threadIndex) {
    const int64_t block = blockIndex;
    const block_iq2_xs *x = (const block_iq2_xs *)vx + block;
    const int il = threadIndex / 8;
    const int ib = threadIndex % 8;
    auto y = yy + block * QK_K + 32 * ib + 8 * il;

    const uint16_t q2 = x->qs[4 * ib + il];
    const uint64_t grid = iq2xs_grid[q2 & 0x1ff];
    const uint8_t signs = ksigns_iq2xs[q2 >> 9];
    const int scale = (x->scales[ib] >> (4 * (il / 2))) & 0x0f;
    const float d = __half2float(x->d) * (0.5f + scale) * 0.25f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int q = (int)((grid >> (8u * (unsigned)j)) & 0xffu);
        const float sign = (signs & (1u << j)) ? -1.0f : 1.0f;
        y[j] = DequantizeCast<dst_t>::cast(d * q * sign);
    }
}
template<typename dst_t, typename Output>
static __device__ __forceinline__ void dequantize_block_iq2_s_impl(const void * __restrict__ vx,
                                               Output yy, int64_t blockIndex, int threadIndex) {
    const int64_t block = blockIndex;
    const block_iq2_s *x = (const block_iq2_s *)vx + block;
    const int il = threadIndex / 8;
    const int ib = threadIndex % 8;
    auto y = yy + block * QK_K + 32 * ib + 8 * il;

    const unsigned gridIndex =
        x->qs[4 * ib + il] |
        (((unsigned)x->qh[ib] << (8 - 2 * il)) & 0x300u);
    const uint64_t grid = iq2s_grid[gridIndex];
    const uint8_t signs = x->qs[QK_K / 8 + 4 * ib + il];
    const int scale = (x->scales[ib] >> (4 * (il / 2))) & 0x0f;
    const float d = __half2float(x->d) * (0.5f + scale) * 0.25f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int q = (int)((grid >> (8u * (unsigned)j)) & 0xffu);
        const float sign = (signs & (1u << j)) ? -1.0f : 1.0f;
        y[j] = DequantizeCast<dst_t>::cast(d * q * sign);
    }
}
template<typename dst_t, typename Output>
static __device__ __forceinline__ void dequantize_block_iq3_xxs_impl(const void * __restrict__ vx,
                                                 Output yy, int64_t blockIndex, int threadIndex) {
    const int64_t block = blockIndex;
    const block_iq3_xxs *x = (const block_iq3_xxs *)vx + block;
    const int il = threadIndex / 8;
    const int ib = threadIndex % 8;
    auto y = yy + block * QK_K + 32 * ib + 8 * il;

    const uint8_t *qs = x->qs + 8 * ib;
    const uint8_t *gas = x->qs + QK_K / 4 + 4 * ib;
    const uint32_t aux =
        (uint32_t)gas[0] | ((uint32_t)gas[1] << 8) |
        ((uint32_t)gas[2] << 16) | ((uint32_t)gas[3] << 24);
    const uint32_t grid0 = iq3xxs_grid[qs[2 * il + 0]];
    const uint32_t grid1 = iq3xxs_grid[qs[2 * il + 1]];
    const uint8_t signs = ksigns_iq2xs[(aux >> (7 * il)) & 0x7f];
    const float d = __half2float(x->d) * (0.5f + (aux >> 28)) * 0.5f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q0 = (int)((grid0 >> (8u * (unsigned)j)) & 0xffu);
        const int q1 = (int)((grid1 >> (8u * (unsigned)j)) & 0xffu);
        const float sign0 = (signs & (1u << j)) ? -1.0f : 1.0f;
        const float sign1 = (signs & (1u << (j + 4))) ? -1.0f : 1.0f;
        y[j] = DequantizeCast<dst_t>::cast(d * q0 * sign0);
        y[j + 4] = DequantizeCast<dst_t>::cast(d * q1 * sign1);
    }
}
template<typename dst_t, typename Output>
static __device__ __forceinline__ void dequantize_block_iq1_s_impl(const void * __restrict__ vx,
                                               Output yy, int64_t blockIndex, int threadIndex) {
    const int64_t block = blockIndex;
    const block_iq1_s *x = (const block_iq1_s *)vx + block;
    const int il = threadIndex / 8;
    const int ib = threadIndex % 8;
    auto y = yy + block * QK_K + 32 * ib + 8 * il;

    const uint16_t qh = x->qh[ib];
    const unsigned gridIndex =
        x->qs[4 * ib + il] | (((qh >> (3 * il)) & 0x07u) << 8);
    const uint64_t grid = iq1s_grid[gridIndex];
    const float delta =
        (qh & 0x8000u) ? -IQ1S_DELTA : IQ1S_DELTA;
    const float d = __half2float(x->d) * (2 * ((qh >> 12) & 7) + 1);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int q = (int)(int8_t)((grid >> (8u * (unsigned)j)) & 0xffu);
        y[j] = DequantizeCast<dst_t>::cast(d * (q + delta));
    }
}
template<typename dst_t, typename Output>
static __device__ __forceinline__ void dequantize_block_iq1_m_impl(const void * __restrict__ vx,
                                               Output yy, int64_t blockIndex, int threadIndex) {
    const int64_t block = blockIndex;
    const block_iq1_m *x = (const block_iq1_m *)vx + block;
    const int il = threadIndex / 8;
    const int ib = threadIndex % 8;
    auto y = yy + block * QK_K + 32 * ib + 8 * il;

    const uint16_t *sc = (const uint16_t *)x->scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int ib16 = 2 * ib + il / 2;
    const float d = __half2float(scale.f16) *
        (2 * ((sc[ib16 / 4] >> (3 * (ib16 % 4))) & 0x07) + 1);
    const uint8_t qh = x->qh[2 * ib + il / 2] >> (4 * (il % 2));
    const unsigned gridIndex = x->qs[4 * ib + il] | ((qh & 0x07u) << 8);
    const uint64_t grid = iq1s_grid[gridIndex];
    const float delta =
        (qh & 0x08u) ? -IQ1M_DELTA : IQ1M_DELTA;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int q = (int)(int8_t)((grid >> (8u * (unsigned)j)) & 0xffu);
        y[j] = DequantizeCast<dst_t>::cast(d * (q + delta));
    }
}
template<typename dst_t, typename Output>
static __device__ __forceinline__ void dequantize_block_iq4_nl_impl(const void * __restrict__ vx,
                                                Output y,
                                                const int64_t k, int64_t blockIndex, int threadIndex) {
    const int64_t group = blockIndex;
    const int64_t groupOffset = group * QK_K;
    if (groupOffset >= k) {
        return;
    }

    const block_iq4_nl *x = (const block_iq4_nl *)vx +
                            group * (QK_K / QK4_NL);
    const int il = threadIndex / 8;
    const int ib = threadIndex % 8;
    if (groupOffset + 32 * ib >= k) return;
    const int64_t outputOffset = groupOffset + 32 * ib + 4 * il;
    const uint8_t *q4 = x[ib].qs + 4 * il;
    const float d = __half2float(x[ib].d);

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        if (outputOffset + j < k) {
            y[outputOffset + j] = DequantizeCast<dst_t>::cast(
                d * kvalues_iq4nl[q4[j] & 0x0f]);
        }
        if (outputOffset + j + 16 < k) {
            y[outputOffset + j + 16] = DequantizeCast<dst_t>::cast(
                d * kvalues_iq4nl[q4[j] >> 4]);
        }
    }
}
template<typename dst_t, typename Output>
static __device__ __forceinline__ void dequantize_block_iq4_xs_impl(const void * __restrict__ vx,
                                                Output y,
                                                const int64_t k, int64_t blockIndex, int threadIndex) {
    const int64_t block = blockIndex;
    const int64_t blockOffset = block * QK_K;
    if (blockOffset >= k) {
        return;
    }

    const block_iq4_xs *x = (const block_iq4_xs *)vx + block;
    const int il = threadIndex / 8;
    const int ib = threadIndex % 8;
    const int64_t outputOffset = blockOffset + 32 * ib + 4 * il;
    const uint8_t *q4 = x->qs + 16 * ib + 4 * il;
    const int scale =
        ((x->scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0f) |
        (((x->scales_h >> (2 * ib)) & 0x03) << 4);
    const float d = __half2float(x->d) * (scale - 32);

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        if (outputOffset + j < k) {
            y[outputOffset + j] = DequantizeCast<dst_t>::cast(
                d * kvalues_iq4nl[q4[j] & 0x0f]);
        }
        if (outputOffset + j + 16 < k) {
            y[outputOffset + j + 16] = DequantizeCast<dst_t>::cast(
                d * kvalues_iq4nl[q4[j] >> 4]);
        }
    }
}
template<typename dst_t, typename Output>
static __device__ __forceinline__ void dequantize_block_q2_K_impl(const void * __restrict__ vx, Output yy, int64_t blockIndex, int threadIndex, int workerCount) {
    const block_q2_K * x = (const block_q2_K *) vx;

    const int64_t i = blockIndex;
    const int tid = threadIndex;

    const float d = __low2float(x[i].dm);
    const float dmin = __high2float(x[i].dm);
    auto y = yy + i * QK_K;

    for (int idx = tid; idx < QK_K; idx += workerCount) {
        const int ib128 = idx / 128;
        const int i128 = idx - ib128 * 128;
        const int scale_idx = ib128 * 8 + i128 / 16;
        const int shift = 2 * (i128 / 32);
        const int qidx = ib128 * 32 + (i128 & 15) + ((i128 & 31) >= 16 ? 16 : 0);

        const uint8_t sc = x[i].scales[scale_idx];
        const float dl = d * (sc & 0xF);
        const float ml = dmin * (sc >> 4);
        const int q = (x[i].qs[qidx] >> shift) & 0x3;

        y[idx] = DequantizeCast<dst_t>::cast(dl * q - ml);
    }
}
template<typename dst_t, typename Output>
static __device__ __forceinline__ void dequantize_block_q3_K_impl(const void * __restrict__ vx,
                                              Output yy, int64_t blockIndex, int threadIndex) {
    const int64_t block = blockIndex;
    const block_q3_K *x = (const block_q3_K *)vx;

    const int64_t r = threadIndex / 4;
    const int64_t tid = r / 2;
    const int64_t is0 = r % 2;
    const int64_t l0 = 16 * is0 + 4 * (threadIndex % 4);
    const int64_t n = tid / 4;
    const int64_t j = tid - 4 * n;

    const uint8_t mask = 1 << (4 * n + j);
    const int64_t is = 8 * n + 2 * j + is0;
    const int shift = 2 * j;

    const int8_t packedScale =
        is < 4 ? (x[block].scales[is] & 0x0f) |
                       (((x[block].scales[is + 8] >> 0) & 0x03) << 4) :
        is < 8 ? (x[block].scales[is] & 0x0f) |
                       (((x[block].scales[is + 4] >> 2) & 0x03) << 4) :
        is < 12 ? (x[block].scales[is - 8] >> 4) |
                        (((x[block].scales[is] >> 4) & 0x03) << 4) :
                  (x[block].scales[is - 8] >> 4) |
                        (((x[block].scales[is - 4] >> 6) & 0x03) << 4);
    const float scale = __half2float(x[block].d) * (packedScale - 32);

    auto y = yy + block * QK_K + 128 * n + 32 * j;
    const uint8_t *q = x[block].qs + 32 * n;
    const uint8_t *highMask = x[block].hmask;
#pragma unroll
    for (int l = l0; l < l0 + 4; ++l) {
        const int value = (int)((q[l] >> shift) & 0x03) -
                          ((highMask[l] & mask) ? 0 : 4);
        y[l] = DequantizeCast<dst_t>::cast(scale * value);
    }
}
template<typename dst_t, typename Output>
static __device__ __forceinline__ void dequantize_block_q4_K_impl(const void * __restrict__ vx, Output yy, int64_t blockIndex, int threadIndex) {
    const block_q4_K * x = (const block_q4_K *) vx;

    const int64_t i = blockIndex;

    // assume 32 threads
    const int64_t tid = threadIndex;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t is  = 2*il;
    const int64_t n   = 4;

    auto y = yy + i*QK_K + 64*il + n*ir;

    const float dall = __low2half(x[i].dm);
    const float dmin = __high2half(x[i].dm);

    const uint8_t * q = x[i].qs + 32*il + n*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = DequantizeCast<dst_t>::cast(d1 * (q[l] & 0xF) - m1);
        y[l +32] = DequantizeCast<dst_t>::cast(d2 * (q[l] >>  4) - m2);
    }
}
template<typename dst_t, typename Output>
static __device__ __forceinline__ void dequantize_block_q5_K_impl(const void * __restrict__ vx, Output yy, int64_t blockIndex, int threadIndex) {
    const block_q5_K * x = (const block_q5_K *) vx;

    const int64_t i = blockIndex;

    // assume 64 threads - this is very slightly better than the one below
    const int64_t tid = threadIndex;
    const int64_t il  = tid/16;   // il is in 0...3
    const int64_t ir  = tid%16;   // ir is in 0...15
    const int64_t is  = 2*il;     // is is in 0...6

    auto y = yy + i*QK_K + 64*il + 2*ir;

    const float dall = __low2half(x[i].dm);
    const float dmin = __high2half(x[i].dm);

    const uint8_t * ql = x[i].qs + 32*il + 2*ir;
    const uint8_t * qh = x[i].qh + 2*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;

    uint8_t   hm  = 1 << (2*il);
    y[ 0] = DequantizeCast<dst_t>::cast(d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1);
    y[ 1] = DequantizeCast<dst_t>::cast(d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1);
    hm <<= 1;
    y[32] = DequantizeCast<dst_t>::cast(d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2);
    y[33] = DequantizeCast<dst_t>::cast(d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2);
}
template<typename dst_t, typename Output>
static __device__ __forceinline__ void dequantize_block_q6_K_impl(const void * __restrict__ vx, Output yy, int64_t blockIndex, int threadIndex) {
    const block_q6_K * x = (const block_q6_K *) vx;

    const int64_t i = blockIndex;

    // assume 64 threads - this is very slightly better than the one below
    const int64_t tid = threadIndex;
    const int64_t ip  = tid/32;   // ip is 0 or 1
    const int64_t il  = tid - 32*ip; // 0...32
    const int64_t is  = 8*ip + il/16;

    auto y = yy + i*QK_K + 128*ip + il;

    const float d = x[i].d;

    const uint8_t * ql = x[i].ql + 64*ip + il;
    const uint8_t   qh = x[i].qh[32*ip + il];
    const int8_t  * sc = x[i].scales + is;

    y[ 0] = DequantizeCast<dst_t>::cast(d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32));
    y[32] = DequantizeCast<dst_t>::cast(d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32));
    y[64] = DequantizeCast<dst_t>::cast(d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32));
    y[96] = DequantizeCast<dst_t>::cast(d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32));
}
template<typename dst_t, typename Output>
static __device__ __forceinline__ void dequantize_block_iq3_s_impl(const void * __restrict__ vx,
                                               Output yy, int64_t blockIndex, int threadIndex) {
    const int64_t block = blockIndex;
    const block_iq3_s *x = (const block_iq3_s *)vx;

    const int64_t tid = threadIndex;
    const int64_t il = tid / 8;
    const int64_t ib = tid % 8;
    auto y = yy + block * QK_K + 32 * ib + 8 * il;
    const uint8_t *qs = x[block].qs + 8 * ib;
    const uint8_t *grid1 = (const uint8_t *)(
        iq3s_grid +
        (qs[2 * il] | ((x[block].qh[ib] << (8 - 2 * il)) & 0x100)));
    const uint8_t *grid2 = (const uint8_t *)(
        iq3s_grid +
        (qs[2 * il + 1] |
         ((x[block].qh[ib] << (7 - 2 * il)) & 0x100)));
    const float scale = __half2float(x[block].d) *
        (1 + 2 * ((x[block].scales[ib / 2] >> (4 * (ib % 2))) & 0x0f));
    const uint8_t signs = x[block].signs[4 * ib + il];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float sign0 = (signs & kmask_iq2xs[j]) ? -1.0f : 1.0f;
        const float sign1 = (signs & kmask_iq2xs[j + 4]) ? -1.0f : 1.0f;
        y[j] = DequantizeCast<dst_t>::cast(scale * grid1[j] * sign0);
        y[j + 4] = DequantizeCast<dst_t>::cast(scale * grid2[j] * sign1);
    }
}

// No global-memory weight buffer and no activation requantization. Round each
// decoded weight as the corresponding safe path does, then accumulate in FP32.
template<typename T>
struct FastllmGgufGemvDotOutput {
    const T *input;
    float *sum;
    int64_t offset;
    __device__ __forceinline__ FastllmGgufGemvDotOutput operator+(int64_t n) const {
        return {input, sum, offset + n};
    }
    struct Element {
        const T *input;
        float *sum;
        __device__ __forceinline__ void operator=(T weight) const {
            *sum = fmaf(static_cast<float>(weight), static_cast<float>(*input), *sum);
        }
    };
    __device__ __forceinline__ Element operator[](int64_t n) const {
        return {input + offset + n, sum};
    }
};

template<ggml_type type, typename T>
static __device__ __forceinline__ void FastllmGgufGemvBlock(
        const void *weight, FastllmGgufGemvDotOutput<T> output,
        int block, int lane, int columns) {
    if constexpr (type == GGML_TYPE_Q4_0) {
        dequantize_block_q4_0_impl<T>(weight, output, columns / QK4_0, block, lane);
    }
    else if constexpr (type == GGML_TYPE_Q4_1) {
        dequantize_block_q4_1_impl<T>(weight, output, columns / QK4_1, block, lane);
    }
    else if constexpr (type == GGML_TYPE_IQ2_XXS) {
        dequantize_block_iq2_xxs_impl<T>(weight, output, block, lane);
    }
    else if constexpr (type == GGML_TYPE_IQ2_XS) {
        dequantize_block_iq2_xs_impl<T>(weight, output, block, lane);
    }
    else if constexpr (type == GGML_TYPE_IQ2_S) {
        dequantize_block_iq2_s_impl<T>(weight, output, block, lane);
    }
    else if constexpr (type == GGML_TYPE_IQ3_XXS) {
        dequantize_block_iq3_xxs_impl<T>(weight, output, block, lane);
    }
    else if constexpr (type == GGML_TYPE_IQ1_S) {
        dequantize_block_iq1_s_impl<T>(weight, output, block, lane);
    }
    else if constexpr (type == GGML_TYPE_IQ1_M) {
        dequantize_block_iq1_m_impl<T>(weight, output, block, lane);
    }
    else if constexpr (type == GGML_TYPE_IQ4_NL) {
        dequantize_block_iq4_nl_impl<T>(weight, output, columns, block, lane);
    }
    else if constexpr (type == GGML_TYPE_IQ4_XS) {
        dequantize_block_iq4_xs_impl<T>(weight, output, columns, block, lane);
    }
    else if constexpr (type == GGML_TYPE_Q2_K) {
        dequantize_block_q2_K_impl<T>(weight, output, block, lane, 32);
    }
    else if constexpr (type == GGML_TYPE_Q3_K) {
        dequantize_block_q3_K_impl<T>(weight, output, block, lane);
        dequantize_block_q3_K_impl<T>(weight, output, block, lane + 32);
    }
    else if constexpr (type == GGML_TYPE_Q4_K) {
        dequantize_block_q4_K_impl<T>(weight, output, block, lane);
    }
    else if constexpr (type == GGML_TYPE_Q5_K) {
        dequantize_block_q5_K_impl<T>(weight, output, block, lane);
        dequantize_block_q5_K_impl<T>(weight, output, block, lane + 32);
    }
    else if constexpr (type == GGML_TYPE_Q6_K) {
        dequantize_block_q6_K_impl<T>(weight, output, block, lane);
        dequantize_block_q6_K_impl<T>(weight, output, block, lane + 32);
    }
    else if constexpr (type == GGML_TYPE_IQ3_S) {
        dequantize_block_iq3_s_impl<T>(weight, output, block, lane);
    }
    else if constexpr (type == GGML_TYPE_Q8_0) {
        const auto *blocks = static_cast<const block_q8_0 *>(weight);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int column = block * QK_K + j * 32 + lane;
            if (column < columns) {
                const auto &q = blocks[column / QK8_0];
                output[column] = DequantizeCast<T>::cast(__half2float(q.d) * q.qs[lane]);
            }
        }
    }
}

template<ggml_type type, typename T, int warps = 4>
static __global__ void FastllmGgufDirectGemvKernel(
        const T *__restrict__ input, const char *__restrict__ weight,
        T *__restrict__ output, int columns, size_t rowBytes) {
    const int row = blockIdx.x;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    float sum = 0.0f;
    FastllmGgufGemvDotOutput<T> dot{input, &sum, 0};
    const void *rowWeight = weight + static_cast<size_t>(row) * rowBytes;
    for (int b = warp; b < (columns + QK_K - 1) / QK_K; b += warps) {
        FastllmGgufGemvBlock<type>(rowWeight, dot, b, lane, columns);
    }
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, mask);
    }
    __shared__ float partial[warps];
    if (lane == 0) partial[warp] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float result = 0.0f;
#pragma unroll
        for (int w = 0; w < warps; ++w) result += partial[w];
        output[row] = DequantizeCast<T>::cast(result);
    }
}

// Only ordinary GGUF layouts are accepted. Repacked R4 layouts retain their
// existing dispatch; they must never be interpreted as an ordinary block.
template<typename T>
static bool FastllmGgufDirectGemv(
        const T *input, const void *weight, T *output, ggml_type type,
        int columns, int rows, cudaStream_t stream) {
    if (columns <= 0 || rows <= 0 || columns % ggml_blck_size(type) != 0) return false;
    const size_t rowBytes = ggml_row_size(type, columns);
    switch (type) {
#define FASTLLM_GGUF_DIRECT_GEMV_CASE(name) \
        case GGML_TYPE_##name: \
            FastllmGgufDirectGemvKernel<GGML_TYPE_##name><<<rows, 128, 0, stream>>>( \
                input, static_cast<const char *>(weight), output, columns, rowBytes); \
            return true;
        FASTLLM_GGUF_DIRECT_GEMV_CASE(Q4_0)
        FASTLLM_GGUF_DIRECT_GEMV_CASE(Q4_1)
        FASTLLM_GGUF_DIRECT_GEMV_CASE(IQ2_XXS)
        FASTLLM_GGUF_DIRECT_GEMV_CASE(IQ2_XS)
        FASTLLM_GGUF_DIRECT_GEMV_CASE(IQ2_S)
        FASTLLM_GGUF_DIRECT_GEMV_CASE(IQ3_XXS)
        FASTLLM_GGUF_DIRECT_GEMV_CASE(IQ1_S)
        FASTLLM_GGUF_DIRECT_GEMV_CASE(IQ1_M)
        FASTLLM_GGUF_DIRECT_GEMV_CASE(IQ4_NL)
        FASTLLM_GGUF_DIRECT_GEMV_CASE(IQ4_XS)
        FASTLLM_GGUF_DIRECT_GEMV_CASE(Q2_K)
        FASTLLM_GGUF_DIRECT_GEMV_CASE(Q3_K)
        FASTLLM_GGUF_DIRECT_GEMV_CASE(Q4_K)
        FASTLLM_GGUF_DIRECT_GEMV_CASE(Q5_K)
        FASTLLM_GGUF_DIRECT_GEMV_CASE(Q6_K)
        FASTLLM_GGUF_DIRECT_GEMV_CASE(IQ3_S)
        FASTLLM_GGUF_DIRECT_GEMV_CASE(Q8_0)
#undef FASTLLM_GGUF_DIRECT_GEMV_CASE
        default: return false;
    }
}
