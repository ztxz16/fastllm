#pragma once
// OCP MX shared scales: one ue8m0 byte per (row, 32-wide K block), value 2^(byte - 127).
// Byte 0 (2^-127) is below the fp32 normal range; the quantizer never emits it (blocks of zeros
// carry byte 127, scale 1), so the decode is one shift into the exponent field.
#include <cstdint>

#if defined(__CUDACC__)
namespace naivert::mx {

__device__ __forceinline__ float ue8m0_to_float(std::uint8_t code) {
  return __uint_as_float(std::uint32_t(code) << 23);
}

}  // namespace naivert::mx
#endif
