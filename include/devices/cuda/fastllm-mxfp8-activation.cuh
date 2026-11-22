#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include "fastllm-mxfp8-projection.h"

namespace fastllm_mxfp8 {

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
    const uint32_t bits = __float_as_uint(maximum / 448.f);
    exponent = min(254u, max(1u, (bits >> 23) + ((bits & 0x7fffffu) != 0)));
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

} // namespace fastllm_mxfp8
