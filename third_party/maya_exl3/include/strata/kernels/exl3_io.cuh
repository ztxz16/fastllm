// Fastllm extension to the MIT-licensed Maya kernels; see NOTICE.txt.
// Storage conversion only: arithmetic and BF16 rounding boundaries stay unchanged.
#pragma once
#include <cuda_fp16.h>
#include <cstdint>
namespace strata::kernels::exl3 {
__device__ __forceinline__ float io_load16(uint16_t bits, bool bf16) {
    return bf16 ? __uint_as_float(unsigned(bits) << 16) : __half2float(__ushort_as_half(bits));
}
__device__ __forceinline__ uint16_t io_store16(float value, bool bf16) {
    if (!bf16) return __half_as_ushort(__float2half_rn(value));
    unsigned u = __float_as_uint(value);
    if ((u & 0x7fffffffU) > 0x7f800000U) u |= 0x00400000U;
    else u += 0x7fffU + ((u >> 16) & 1U);
    return uint16_t(u >> 16);
}
} // namespace strata::kernels::exl3
