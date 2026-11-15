#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace fastllm {
// One host definition for CPU execution and CUDA's immutable device copy.
// Keep FP32 SiLU until multiplication by up and the route score: rounding
// SiLU itself to BF16 would move the expert's quantization boundary.
inline const std::array<float, 65536> &GetBFloat16SiluLookup() {
    static const std::array<float, 65536> values = [] {
        std::array<float, 65536> result{};
        for (uint32_t i = 0; i < result.size(); ++i) {
            uint32_t bits = i << 16;
            float gate;
            std::memcpy(&gate, &bits, sizeof(gate));
            result[i] = gate / (1.0f + std::exp(-gate));
        }
        return result;
    }();
    return values;
}
}
