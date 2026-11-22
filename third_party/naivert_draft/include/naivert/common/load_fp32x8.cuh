#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace naivert {

// Load eight FP32 values without conversion. The caller must provide a
// 32-byte-aligned pointer. Keep default caching for these reused gamma values.
__device__ __forceinline__ void load_fp32x8_aligned32(
    const float* ptr, float4& lo, float4& hi) {
  std::uint32_t bits[8];
  asm volatile("ld.global.v8.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=r"(bits[0]), "=r"(bits[1]), "=r"(bits[2]), "=r"(bits[3]),
        "=r"(bits[4]), "=r"(bits[5]), "=r"(bits[6]), "=r"(bits[7])
      : "l"(ptr));
  lo = make_float4(__uint_as_float(bits[0]), __uint_as_float(bits[1]),
                    __uint_as_float(bits[2]), __uint_as_float(bits[3]));
  hi = make_float4(__uint_as_float(bits[4]), __uint_as_float(bits[5]),
                    __uint_as_float(bits[6]), __uint_as_float(bits[7]));
}

}  // namespace naivert
