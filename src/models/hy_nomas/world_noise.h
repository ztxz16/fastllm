#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace fastllm { namespace nomas { namespace world_noise {

constexpr int horizon = 720;
constexpr int channels = 32;

inline uint16_t pack(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

inline float unpack(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

// The released BF16 worker draws low-24-bit uniforms in groups of 16.
// Pair the two halves of each group before rounding the Gaussian samples.
inline std::vector<uint16_t> draw(std::mt19937 &rng, size_t tokens) {
    std::vector<uint16_t> values(channels * tokens);
    for (size_t base = 0; base < values.size(); base += 16) {
        std::array<float, 16> uniform;
        for (float &u : uniform)
            u = static_cast<float>(rng() & 0xffffffu) * 0x1p-24f;
        for (int j = 0; j < 8; ++j) {
            const float radius = std::sqrt(-2.f * std::log(1.f - uniform[j]));
            const float angle = 6.2831853071795864769f * uniform[j + 8];
            values[base + j] = pack(radius * std::cos(angle));
            values[base + j + 8] = pack(radius * std::sin(angle));
        }
    }
    return values;
}

// Official noise is contiguous [C,T,H,W]. The DiT consumes [T,H,W,C].
inline void read(const std::vector<uint16_t> &source, size_t offset, std::vector<float> &out) {
    const size_t stride = source.size() / channels;
    for (size_t i = 0; i < out.size() / channels; ++i)
        for (int c = 0; c < channels; ++c)
            out[i * channels + c] = unpack(source[c * stride + offset + i]);
}

}}} // namespace fastllm::nomas::world_noise
