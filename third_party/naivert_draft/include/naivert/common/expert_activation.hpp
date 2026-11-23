#pragma once

#include <cuda_bf16.h>

namespace naivert {

// One shared storage contract for the routed SwiGLU producer and down consumer.
// Native W8A8 quantizes the FP32 epilogue directly, without an intermediate
// BF16 round. Other model-op builds retain their existing BF16 activation ABI.
#if defined(NAIVERT_EXPERT_DOWN_W8A8) && NAIVERT_EXPERT_DOWN_W8A8
using ExpertActivation = float;
using RouterWeight = float;
using ExpertWeightScale = float;
inline constexpr bool kExpertActivationFp32 = true;
#else
using ExpertActivation = __nv_bfloat16;
using RouterWeight = __nv_bfloat16;
using ExpertWeightScale = __nv_bfloat16;
inline constexpr bool kExpertActivationFp32 = false;
#endif

#if defined(__CUDACC__)
__device__ __forceinline__ float expert_weight_scale(ExpertWeightScale value) {
#if defined(NAIVERT_EXPERT_DOWN_W8A8) && NAIVERT_EXPERT_DOWN_W8A8
  return value;
#else
  return __bfloat162float(value);
#endif
}

// A router pair has the same logical coordinates in both storage ABIs.
__device__ __forceinline__ float2 router_weight_pair(const void* base, int pair) {
#if defined(NAIVERT_EXPERT_DOWN_W8A8) && NAIVERT_EXPERT_DOWN_W8A8
  return reinterpret_cast<const float2*>(base)[pair];
#else
  return __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(base)[pair]);
#endif
}

__device__ __forceinline__ ExpertActivation expert_activation_from_float(float value) {
#if defined(NAIVERT_EXPERT_DOWN_W8A8) && NAIVERT_EXPERT_DOWN_W8A8
  return value;
#else
  // Torch builds disable BF16 converting constructors; retain the original
  // explicit round-to-nearest conversion for the BF16 storage ABI.
  return __float2bfloat16_rn(value);
#endif
}
#endif

}  // namespace naivert
