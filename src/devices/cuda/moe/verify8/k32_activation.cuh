#pragma once
#include <cuda_fp8.h>

namespace fastllm_rows8_k32 {
// Each warp owns one K128 tile. Eight consecutive lanes own each logical K32.
__device__ __forceinline__ unsigned quantize(float4 values, unsigned &packedScales) {
    float amax = fmaxf(fmaxf(fabsf(values.x), fabsf(values.y)), fmaxf(fabsf(values.z), fabsf(values.w)));
#pragma unroll
    for (int d = 4; d; d >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d, 8));
    // Same UE8M0 rounding threshold and minimum scale as FastLLM's K32 producer.
    const unsigned e = max((__float_as_uint(amax) + 0x1fffffu) >> 23, 113u) - 8u;
    const float inv = __uint_as_float((254u - e) << 23);
    unsigned pair = e | (__shfl_down_sync(0xffffffffu, e, 8) << 8);
    packedScales = pair | (__shfl_down_sync(0xffffffffu, pair, 16) << 16);
    return __nv_fp8x4_e4m3(make_float4(values.x * inv, values.y * inv, values.z * inv, values.w * inv)).__x;
}

__device__ __forceinline__ float scale(unsigned packed, int sub) {
    return __uint_as_float(((packed >> (8 * sub)) & 255u) << 23);
}
} // namespace fastllm_rows8_k32
