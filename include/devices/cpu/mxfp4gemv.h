#pragma once

#include "fastllm.h"

namespace fastllm {
    // FP32 expansion of the model-quantized BF16 row in the order used by the FP4 nibbles.
    struct Mxfp4GemvInput {
        std::vector<float, alignedAllocator<float, 64>> values;
    };

    bool Mxfp4Avx2GemvAvailable();
    // Both projections consist of complete 32-element blocks.
    void PrepareMxfp4GemvInput(const uint16_t *bf16, int columns, Mxfp4GemvInput &input);

    void RunMxfp4Avx2Gemv(const uint16_t *bf16, const Mxfp4GemvInput &input,
                          const uint8_t *weights, float *output, int columns, int rows);
}
