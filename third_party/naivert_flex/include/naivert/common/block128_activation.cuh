#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "naivert/common/expert_activation.hpp"

namespace naivert {
// Runtime W8A8 recipe: E4M3 RN payload and a power-of-two scale per K128.
// Kept separate from weight scales, which preserve the checkpoint FP32 values.
__device__ __forceinline__ float block128_activation_scale(float amax) {
  const unsigned raw = __float_as_uint(amax);
  if ((raw & 0x7fffffffu) == 0u) return 1.0f;
  // At the minimum subnormal scale, the first FP32 amax above 448*2^-127
  // still rounds down to 2^-127 when divided. Include that one extra value.
  if (raw <= 0x04600001u) return __uint_as_float(0x00400000u);
  // Preserve the original fmax/divide behavior for Inf, NaN and negative input.
  if (raw >= 0x7f800000u)
    return __uint_as_float(raw == 0x7f800000u ? raw : 0x00400000u);
  // 448 = 1.75 * 2^8. The next FP32 value above each threshold divides
  // to more than half an ULP above the power of two, so RN division cannot
  // round it back down across this comparison boundary.
  const int exponent = int(raw >> 23) - 8 + int((raw & 0x007fffffu) > 0x00600000u);
  return __uint_as_float(exponent <= 0 ? 0x00400000u : unsigned(exponent) << 23);
}

// The quantizer's scales are powers of two in [2^-127, 2^120], or 1
// for a zero block. Their reciprocals are exactly representable in FP32,
// including the minimum subnormal scale. RN multiplication therefore has
// the same result as RN division, without a general floating-point divide.
__device__ __forceinline__ float block128_inverse_scale(float scale) {
  const unsigned bits = __float_as_uint(scale);
  return __uint_as_float(bits == 0x00400000u ? 0x7f000000u : 0x7f000000u - bits);
}

// Values are nonnegative maxima of absolute values. Map NaN below zero;
// a finite lane beats NaN, while an all-NaN warp stays NaN, as with fmaxf.
__device__ __forceinline__ float block128_warp_amax(float value) {
  const int bits = __float_as_int(value);
  const int key = bits > 0x7f800000 ? -1 : bits;
  return __int_as_float(__reduce_max_sync(0xffffffffu, key));
}

// Every participating warp owns complete K128 blocks. No inter-warp barrier
// is needed; the caller publishes all activations after its existing sync.
__device__ __forceinline__ void quantize_block128_row(
    const __nv_bfloat16* source, __nv_fp8_e4m3* destination, float* scales,
    int hidden, int worker, int workers) {
  const int lane = worker & 31;
  for (int block = worker >> 5; block < hidden / 128; block += workers / 32) {
    const int k = block * 128 + lane * 4;
    const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(source + k);
    const float2 a = __bfloat1622float2(pairs[0]);
    const float2 b = __bfloat1622float2(pairs[1]);
    float amax = fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(b.x), fabsf(b.y)));
    amax = block128_warp_amax(amax);
    const float scale = block128_activation_scale(amax);
    const float inverse = block128_inverse_scale(scale);
    const __nv_fp8x4_e4m3 q(make_float4(__fmul_rn(a.x, inverse), __fmul_rn(a.y, inverse),
                                        __fmul_rn(b.x, inverse), __fmul_rn(b.y, inverse)));
    *reinterpret_cast<unsigned*>(destination + k) = q.__x;
    if (lane == 0) scales[block] = scale;
  }
}
}  // namespace naivert
