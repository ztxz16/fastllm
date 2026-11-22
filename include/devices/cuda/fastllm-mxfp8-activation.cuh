#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include "fastllm-mxfp8-projection.h"

namespace fastllm_mxfp8 {

__device__ __forceinline__ uint32_t ScaleExponent(float maximum) {
    const uint32_t bits = __float_as_uint(maximum / 448.f);
    return min(254u, max(1u, (bits >> 23) + ((bits & 0x7fffffu) != 0)));
}

// Each warp supplies contiguous, already rounded BF16 output to one or more
// K32 groups. The scale address is identical to the standalone pack kernel.
template <int N>
__device__ __forceinline__ void StoreGroup(FastllmMxfp8Activation dst, int row, int col,
                                           const __nv_bfloat16 (&values)[N]) {
    static_assert(N == 1 || N == 4);
    float maximum = 0.f;
#pragma unroll
    for (int i = 0; i < N; ++i)
        maximum = fmaxf(maximum, fabsf(__bfloat162float(values[i])));
#pragma unroll
    for (int d = 16 / N; d; d >>= 1)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, d, 32 / N));
    const unsigned exponent = ScaleExponent(maximum);
    const float inv = __uint_as_float(exponent == 254 ? 0x00400000u : (254u - exponent) << 23);
    uint32_t packed = 0;
#pragma unroll
    for (int i = 0; i < N; ++i)
        packed |= uint32_t(__nv_fp8_e4m3(__bfloat162float(values[i]) * inv).__x) << (8 * i);
    uint8_t *out = dst.values + size_t(row) * dst.width + col;
    if constexpr (N == 4) *reinterpret_cast<uint32_t *>(out) = packed;
    else *out = packed;
    if (col % 32 == 0)
        dst.scales[(size_t(col / 128) * dst.pitch + row) * 4 + (col / 32) % 4] = exponent;
}

// Four consecutive lanes share one K32 exponent. Each lane packs eight BF16 values.
__device__ __forceinline__ uint2 PackActivation(const __nv_bfloat162 *pairs, uint32_t &exponent, unsigned mask) {
    __nv_bfloat162 maxPair = __habs2(pairs[0]);
#pragma unroll
    for (int i = 1; i < 4; ++i)
        maxPair = __hmax2(maxPair, __habs2(pairs[i]));
    const float2 pair = __bfloat1622float2(maxPair);
    float maximum = fmaxf(0.f, fmaxf(pair.x, pair.y));
    maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, 1, 4));
    maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, 2, 4));
    exponent = ScaleExponent(maximum);
    const float inv = __uint_as_float(exponent == 254 ? 0x00400000u : (254u - exponent) << 23);
    union {
        uint2 word;
        __nv_fp8x2_storage_t pair[4];
    } encoded;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 f = __bfloat1622float2(pairs[i]);
        f.x *= inv;
        f.y *= inv;
        encoded.pair[i] = __nv_cvt_float2_to_fp8x2(f, __NV_SATFINITE, __NV_E4M3);
    }
    return encoded.word;
}

// Four adjacent lanes own one K32 block of already rounded BF16 values.
__device__ __forceinline__ void Store8(FastllmMxfp8Activation dst, int row, int col, uint4 values) {
    uint32_t exponent;
    const auto bytes = PackActivation(reinterpret_cast<const __nv_bfloat162 *>(&values), exponent, 0xffffffffu);
    *reinterpret_cast<uint2 *>(dst.values + size_t(row) * dst.width + col) = bytes;
    if (col % 32 == 0)
        dst.scales[(size_t(col / 128) * dst.pitch + row) * 4 + col / 32 % 4] = exponent;
}

__device__ __forceinline__ void StoreZero8(FastllmMxfp8Activation dst, int row, int col) {
    *reinterpret_cast<uint2 *>(dst.values + size_t(row) * dst.width + col) = make_uint2(0, 0);
    if (col % 32 == 0)
        dst.scales[(size_t(col / 128) * dst.pitch + row) * 4 + col / 32 % 4] = 1;
}

} // namespace fastllm_mxfp8
