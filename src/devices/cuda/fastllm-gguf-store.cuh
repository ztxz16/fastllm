#pragma once

#include <cuda_fp16.h>
#include <type_traits>

// CUDA half exp is available from SM53. Older devices keep the same FP16
// materialization boundary through a scalar FP32 calculation.
static __device__ __forceinline__ half FastllmGgufHalfExp(half x) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
    return __float2half_rn(expf(__half2float(x)));
#else
    return hexp(x);
#endif
}

// Keep Linear's FP16 materialization boundary even when its temporary
// tensor disappears. This matches Linear followed by AddTo(alpha=1).
// 0: overwrite; 1: residual add; 2: SiLU(existing gate) * projected up.
template <int StoreMode, typename Output>
static __device__ __forceinline__ void FastllmGgufStore(Output *output, float value) {
    if constexpr (StoreMode == 1) {
        static_assert(std::is_same<Output, half>::value, "GGUF residual requires FP16");
        const half projected = __float2half_rn(value);
        *output = __hadd(*output, projected);
    } else if constexpr (StoreMode == 2) {
        static_assert(std::is_same<Output, half>::value, "GGUF gate product requires FP16");
        const half gate = *output;
        const half activated = __hdiv(gate, __hadd(__float2half(1.0f), FastllmGgufHalfExp(-gate)));
        *output = __hmul(activated, __float2half_rn(value));
    } else {
        static_assert(StoreMode == 0, "Unknown GGUF store mode");
        *output = static_cast<Output>(value);
    }
}
