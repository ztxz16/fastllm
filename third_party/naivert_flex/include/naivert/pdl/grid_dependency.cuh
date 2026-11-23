#pragma once

#include <cuda_device_runtime_api.h>

namespace naivert::pdl {

template <bool Enabled>
__device__ __forceinline__ void wait_for_primary() {
  if constexpr (Enabled) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize();
#endif
  }
}

template <bool Enabled>
__device__ __forceinline__ void trigger_dependents() {
  if constexpr (Enabled) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaTriggerProgrammaticLaunchCompletion();
#endif
  }
}

}  // namespace naivert::pdl
