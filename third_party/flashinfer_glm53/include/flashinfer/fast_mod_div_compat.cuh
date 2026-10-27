//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
//
// This is a self-contained compatibility copy of the 32-bit subset of
// cuda::fast_mod_div used by FlashInfer.  CUDA 12.9 ships an older CCCL that
// does not provide this API.  Prefer CCCL's implementation whenever the
// corresponding public header is available.
// Source: https://github.com/NVIDIA/cccl/blob/main/libcudacxx/include/cuda/__cmath/fast_modulo_division.h

#ifndef FLASHINFER_FAST_MOD_DIV_COMPAT_CUH_
#define FLASHINFER_FAST_MOD_DIV_COMPAT_CUH_

#if defined(__has_include)
#if __has_include(<cuda/__cmath/fast_modulo_division.h>)
#define FLASHINFER_HAS_CCCL_FAST_MOD_DIV 1
#endif
#endif

#if defined(FLASHINFER_HAS_CCCL_FAST_MOD_DIV)

#include <cuda/cmath>

#else

#include <cassert>
#include <cstdint>
#include <type_traits>

namespace cuda {
namespace fast_mod_div_compat_detail {

template <typename UInt>
__host__ __device__ __forceinline__ UInt mul_hi(UInt lhs, UInt rhs) noexcept {
  static_assert(std::is_unsigned<UInt>::value && sizeof(UInt) == sizeof(uint32_t),
                "cuda::fast_mod_div compatibility implementation only supports 32-bit integers");
#if defined(__CUDA_ARCH__)
  return static_cast<UInt>(__umulhi(static_cast<uint32_t>(lhs), static_cast<uint32_t>(rhs)));
#else
  return static_cast<UInt>((static_cast<uint64_t>(lhs) * static_cast<uint64_t>(rhs)) >> 32);
#endif
}

template <typename UInt>
__host__ __device__ __forceinline__ int ilog2(UInt value) noexcept {
  int result = 0;
  while (value > 1) {
    value >>= 1;
    ++result;
  }
  return result;
}

template <typename UInt>
__host__ __device__ __forceinline__ bool is_power_of_two(UInt value) noexcept {
  return value != 0 && (value & (value - 1)) == 0;
}

template <typename UInt>
__host__ __device__ __forceinline__ int ceil_ilog2(UInt value) noexcept {
  return ilog2(value) + !is_power_of_two(value);
}

template <typename UInt>
struct divmod_result {
  UInt quotient;
  UInt remainder;
};

template <typename UInt>
__host__ __device__ __forceinline__ divmod_result<UInt> divmod_pow2(int power,
                                                                   UInt divisor) noexcept {
  constexpr int num_bits = sizeof(UInt) * 8;
  UInt quotient = 0;
  UInt remainder = 0;
  for (int bit = power; bit >= 0; --bit) {
    const bool carry = (remainder >> (num_bits - 1)) != 0;
    remainder <<= 1;
    remainder |= static_cast<UInt>(bit == power);
    const bool quotient_bit = carry || remainder >= divisor;
    quotient <<= 1;
    quotient |= static_cast<UInt>(quotient_bit);
    if (quotient_bit) {
      remainder -= divisor;
    }
  }
  return {quotient, remainder};
}

}  // namespace fast_mod_div_compat_detail

// Backport of CCCL's cuda::fast_mod_div for the 32-bit signed and unsigned
// integer types used by FlashInfer and its TRT-LLM kernels.  Keep the member
// order in sync with CCCL: KernelParams containing this type are ABI-sensitive.
template <typename T, bool DivisorIsNeverOne = false>
class fast_mod_div {
  static_assert(std::is_integral<T>::value && !std::is_same<T, bool>::value &&
                    sizeof(T) == sizeof(uint32_t),
                "cuda::fast_mod_div compatibility implementation only supports 32-bit integers");

  using unsigned_t = typename std::make_unsigned<T>::type;

 public:
  fast_mod_div() = delete;

  __host__ __device__ explicit fast_mod_div(T divisor) noexcept : divisor_(divisor) {
    assert(divisor > 0);
    assert(!DivisorIsNeverOne || divisor != 1);
    const auto unsigned_divisor = static_cast<unsigned_t>(divisor);
    constexpr int num_bits = sizeof(T) * 8;

    if constexpr (std::is_signed<T>::value) {
      shift_ = fast_mod_div_compat_detail::ceil_ilog2(unsigned_divisor) - 1;
      const auto result =
          fast_mod_div_compat_detail::divmod_pow2(num_bits + shift_, unsigned_divisor);
      multiplier_ = result.quotient + static_cast<unsigned_t>(result.remainder != 0);
    } else {
      shift_ = fast_mod_div_compat_detail::ilog2(unsigned_divisor);
      if (fast_mod_div_compat_detail::is_power_of_two(unsigned_divisor)) {
        multiplier_ = 0;
        return;
      }
      const auto result =
          fast_mod_div_compat_detail::divmod_pow2(num_bits + shift_, unsigned_divisor);
      const auto threshold = unsigned_divisor - (unsigned_t{1} << shift_);
      multiplier_ = result.quotient + static_cast<unsigned_t>(result.remainder >= threshold);
      add_ = static_cast<unsigned>(result.remainder < threshold);
    }
  }

  template <typename Lhs>
  [[nodiscard]] __host__ __device__ __forceinline__ friend
      typename std::common_type<T, Lhs>::type
      operator/(Lhs dividend, fast_mod_div divisor) noexcept {
    static_assert(std::is_integral<Lhs>::value && !std::is_same<Lhs, bool>::value,
                  "cuda::fast_mod_div dividend must be an integer");
    static_assert(sizeof(Lhs) <= sizeof(T),
                  "cuda::fast_mod_div dividend type must not be wider than divisor type");
    static_assert(sizeof(Lhs) < sizeof(T) || std::is_signed<Lhs>::value ||
                      std::is_unsigned<T>::value,
                  "cuda::fast_mod_div dividend maximum must fit in divisor type");

    if constexpr (std::is_signed<Lhs>::value) {
      assert(dividend >= 0);
    }

    using common_t = typename std::common_type<T, Lhs>::type;
    using unsigned_common_t = typename std::make_unsigned<common_t>::type;
    using unsigned_lhs_t = typename std::make_unsigned<Lhs>::type;
    auto unsigned_dividend = static_cast<unsigned_lhs_t>(dividend);

    if constexpr (std::is_unsigned<T>::value) {
      if (divisor.multiplier_ == 0) {
        return static_cast<common_t>(static_cast<unsigned_common_t>(unsigned_dividend) >>
                                     divisor.shift_);
      }
      if (std::is_signed<Lhs>::value ||
          unsigned_dividend != static_cast<unsigned_lhs_t>(~unsigned_lhs_t{0})) {
        unsigned_dividend += static_cast<unsigned_lhs_t>(divisor.add_);
      }
    } else if (!DivisorIsNeverOne && divisor.divisor_ == 1) {
      return static_cast<common_t>(dividend);
    }

    const auto high_bits = fast_mod_div_compat_detail::mul_hi(
        static_cast<unsigned_common_t>(unsigned_dividend),
        static_cast<unsigned_common_t>(divisor.multiplier_));
    return static_cast<common_t>(high_bits >> divisor.shift_);
  }

  template <typename Lhs>
  [[nodiscard]] __host__ __device__ __forceinline__ friend
      typename std::common_type<T, Lhs>::type
      operator%(Lhs dividend, fast_mod_div divisor) noexcept {
    return dividend - (dividend / divisor) * divisor.divisor_;
  }

  [[nodiscard]] __host__ __device__ __forceinline__ operator T() const noexcept {
    return divisor_;
  }

 private:
  T divisor_ = 1;
  unsigned_t multiplier_ = 0;
  unsigned add_ = 0;
  int shift_ = 0;
};

static_assert(sizeof(fast_mod_div<uint32_t>) == 16,
              "cuda::fast_mod_div<uint32_t> must match CCCL's ABI");
static_assert(sizeof(fast_mod_div<int32_t>) == 16,
              "cuda::fast_mod_div<int32_t> must match CCCL's ABI");

}  // namespace cuda

#endif  // FLASHINFER_HAS_CCCL_FAST_MOD_DIV

#endif  // FLASHINFER_FAST_MOD_DIV_COMPAT_CUH_
