#pragma once

#include <cuda_bf16.h>

namespace naivert {

// SM100+ mixed-precision FMA: exactly the FP32 round-to-nearest operation on
// widened BF16 operands, without materializing both FP32 conversions. Keep the
// caller's accumulation order; this is not a Tensor Core reduction.
__device__ __forceinline__ float bf16_fma(__nv_bfloat16 a, __nv_bfloat16 b, float c) {
  float d;
  asm("fma.rn.f32.bf16 %0, %1, %2, %3;" : "=f"(d)
      : "h"(__bfloat16_as_ushort(a)), "h"(__bfloat16_as_ushort(b)), "f"(c));
  return d;
}

}  // namespace naivert
