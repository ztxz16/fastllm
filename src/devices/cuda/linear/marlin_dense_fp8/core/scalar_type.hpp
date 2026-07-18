#pragma once

// Torch-free ScalarType for FastLLM dense Marlin (adapted from vLLM).
#include <cstdint>
#include <string>

#define TORCH_CHECK(cond, ...) ((void)0)

namespace vllm {

class ScalarType {
 public:
  enum NanRepr : uint8_t {
    NAN_NONE = 0,
    NAN_IEEE_754 = 1,
    NAN_EXTD_RANGE_MAX_MIN = 2,
    NAN_REPR_ID_MAX
  };

  constexpr ScalarType(uint8_t exponent, uint8_t mantissa, bool signed_,
                       int32_t bias, bool finite_values_only = false,
                       NanRepr nan_repr = NAN_IEEE_754)
      : exponent(exponent),
        mantissa(mantissa),
        signed_(signed_),
        bias(bias),
        finite_values_only(finite_values_only),
        nan_repr(nan_repr) {}

  static constexpr ScalarType int_(uint8_t size_bits, int32_t bias = 0) {
    return ScalarType(0, size_bits - 1, true, bias);
  }

  static constexpr ScalarType uint(uint8_t size_bits, int32_t bias = 0) {
    return ScalarType(0, size_bits, false, bias);
  }

  static constexpr ScalarType float_IEEE754(uint8_t exponent,
                                            uint8_t mantissa) {
    return ScalarType(exponent, mantissa, true, 0, false, NAN_IEEE_754);
  }

  static constexpr ScalarType float_(uint8_t exponent, uint8_t mantissa,
                                     bool finite_values_only, NanRepr nan_repr) {
    return ScalarType(exponent, mantissa, true, 0, finite_values_only, nan_repr);
  }

  using Id = int64_t;

 private:
  static constexpr auto id_exponent_size = 8;
  static constexpr auto id_mantissa_size = 8;
  static constexpr auto id_signed_size = 1;
  static constexpr auto id_bias_size = 32;
  static constexpr auto id_finite_values_only_size = 1;
  static constexpr auto id_nan_repr_size = 8;

  static constexpr Id id_size_bits() {
    return id_exponent_size + id_mantissa_size + id_signed_size + id_bias_size +
           id_finite_values_only_size + id_nan_repr_size;
  }

  template <typename T, int offset, int bits>
  static constexpr T extract_bits(Id id) {
    return static_cast<T>((id >> offset) & ((Id(1) << bits) - 1));
  }

 public:
  constexpr Id id() const {
    Id id = 0;
    int offset = 0;
    auto pack = [&](Id v, int bits) {
      id |= (v & ((Id(1) << bits) - 1)) << offset;
      offset += bits;
    };
    pack(exponent, id_exponent_size);
    pack(mantissa, id_mantissa_size);
    pack(signed_, id_signed_size);
    pack(static_cast<uint32_t>(bias), id_bias_size);
    pack(finite_values_only, id_finite_values_only_size);
    pack(nan_repr, id_nan_repr_size);
    return id;
  }

  static constexpr ScalarType from_id(Id id) {
    int offset = 0;
    auto unpack = [&](int bits) -> Id {
      Id v = (id >> offset) & ((Id(1) << bits) - 1);
      offset += bits;
      return v;
    };
    uint8_t exponent = static_cast<uint8_t>(unpack(id_exponent_size));
    uint8_t mantissa = static_cast<uint8_t>(unpack(id_mantissa_size));
    bool signed_ = unpack(id_signed_size) != 0;
    int32_t bias = static_cast<int32_t>(unpack(id_bias_size));
    bool finite_values_only = unpack(id_finite_values_only_size) != 0;
    NanRepr nan_repr = static_cast<NanRepr>(unpack(id_nan_repr_size));
    return ScalarType(exponent, mantissa, signed_, bias, finite_values_only,
                      nan_repr);
  }

  constexpr int64_t size_bits() const {
    return static_cast<int64_t>(exponent) + static_cast<int64_t>(mantissa) +
           (signed_ ? 1 : 0);
  }

  constexpr bool is_signed() const { return signed_; }
  constexpr bool is_floating_point() const { return exponent != 0; }

  constexpr bool operator==(const ScalarType& other) const {
    return mantissa == other.mantissa && exponent == other.exponent &&
           bias == other.bias && signed_ == other.signed_ &&
           finite_values_only == other.finite_values_only &&
           nan_repr == other.nan_repr;
  }

  constexpr bool operator!=(const ScalarType& other) const {
    return !(*this == other);
  }

  std::string str() const { return "ScalarType"; }

  uint8_t const exponent;
  uint8_t const mantissa;
  bool const signed_;
  int32_t const bias;
  bool const finite_values_only;
  NanRepr const nan_repr;
};

using ScalarTypeId = ScalarType::Id;

static inline constexpr auto kS4 = ScalarType::int_(4);
static inline constexpr auto kU4 = ScalarType::uint(4);
static inline constexpr auto kU4B8 = ScalarType::uint(4, 8);
static inline constexpr auto kS8 = ScalarType::int_(8);
static inline constexpr auto kU8 = ScalarType::uint(8);
static inline constexpr auto kU8B128 = ScalarType::uint(8, 128);
static inline constexpr auto kFE2M1f =
    ScalarType::float_(2, 1, true, ScalarType::NAN_NONE);
static inline constexpr auto kFE4M3fn =
    ScalarType::float_(4, 3, true, ScalarType::NAN_EXTD_RANGE_MAX_MIN);
static inline constexpr auto kFE8M0fnu =
    ScalarType(8, 0, false, 0, true, ScalarType::NAN_EXTD_RANGE_MAX_MIN);
static inline constexpr auto kFE5M2 = ScalarType::float_IEEE754(5, 2);
static inline constexpr auto kFE8M7 = ScalarType::float_IEEE754(8, 7);
static inline constexpr auto kFE5M10 = ScalarType::float_IEEE754(5, 10);
static inline constexpr auto kFloat16 = kFE5M10;
static inline constexpr auto kBFloat16 = kFE8M7;
static inline constexpr auto kHalf = kFloat16;

}  // namespace vllm
