#pragma once
#include <cstdint>

namespace fastllm { class Data; }

// Producer/consumer layout: row-major E4M3 values and one UE8M0 per K32.
// Four consecutive exponents form a word at [column / 128, row] with stride pitch.
// Quantize the rounded BF16 values with ceil(log2(max(abs(x)) / 448));
// clamp the stored biased exponent to [1, 254], including zero groups.
struct FastllmMxfp8Activation {
    uint8_t *values = nullptr, *scales = nullptr;
    int rows = 0, width = 0, pitch = 0;
    bool Matches(int actualRows, int actualWidth) const {
        return values && scales && rows > 0 && rows == actualRows && width == actualWidth &&
               width > 0 && width % 128 == 0 && pitch == (rows + 3) / 4 * 4;
    }
};

// Each CUDA backend opts in to the shared activation layout independently of
// its weight layout. prepareWeight runs on the final shard before capture.
struct FastllmMxfp8ProjectionBackend {
    void (*prepareWeight)(fastllm::Data &weight);
    // Zero declines this weight/phase. A positive limit bounds the shared arena.
    int (*maxRows)(const fastllm::Data &weight, bool verifying);
    // Called on the shard's device. False must leave the output untouched.
    bool (*project)(FastllmMxfp8Activation input, const fastllm::Data &weight, void *output);

    FastllmMxfp8Activation Bind(const fastllm::Data &weight, int rows, bool verifying,
                                int maxWidth, fastllm::Data &storage) const;
    // Verification may alias graph-owned scratch only between its producer and this projection.
    FastllmMxfp8Activation Borrow(const fastllm::Data &weight, int rows, const fastllm::Data &scratch) const;
    bool TryProject(FastllmMxfp8Activation prepared, const fastllm::Data &input,
                    const fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output) const;
};
// Null when no compiled consumer supports the current device. Producers keep
// their BF16 output so the existing Linear is always available.
const FastllmMxfp8ProjectionBackend *FastllmCudaGetMxfp8ProjectionBackend();
