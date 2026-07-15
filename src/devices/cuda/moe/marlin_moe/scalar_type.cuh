#pragma once

#include <cstdint>

// Minimal compile-time scalar descriptors required by the vendored Marlin
// kernel.  These IDs are private to FastLLM's CUDA translation unit; they are
// not the vLLM ABI and deliberately carry no Torch dependency.
namespace fastllm_marlin_moe_types {

class ScalarType {
 public:
    using Id = int64_t;

    constexpr explicit ScalarType(Id value) : value_(value) {}

    constexpr Id id() const {
        return value_;
    }

    constexpr int size_bits() const {
        return static_cast<int>(value_ & 0xff);
    }

    static constexpr ScalarType from_id(Id value) {
        return ScalarType(value);
    }

    constexpr bool operator==(const ScalarType &other) const {
        return value_ == other.value_;
    }

    constexpr bool operator!=(const ScalarType &other) const {
        return !(*this == other);
    }

 private:
    Id value_;
};

using ScalarTypeId = ScalarType::Id;

constexpr ScalarType MakeScalarType(int tag, int bits) {
    return ScalarType((static_cast<ScalarTypeId>(tag) << 8) | bits);
}

static constexpr ScalarType kS4 = MakeScalarType(1, 4);
static constexpr ScalarType kU4 = MakeScalarType(2, 4);
static constexpr ScalarType kU4B8 = MakeScalarType(3, 4);
static constexpr ScalarType kS8 = MakeScalarType(4, 8);
static constexpr ScalarType kU8 = MakeScalarType(5, 8);
static constexpr ScalarType kU8B128 = MakeScalarType(6, 8);
static constexpr ScalarType kFE2M1f = MakeScalarType(7, 4);
static constexpr ScalarType kFE4M3fn = MakeScalarType(8, 8);
static constexpr ScalarType kFE8M0fnu = MakeScalarType(9, 8);
static constexpr ScalarType kFloat16 = MakeScalarType(10, 16);
static constexpr ScalarType kBFloat16 = MakeScalarType(11, 16);

}  // namespace fastllm_marlin_moe_types
