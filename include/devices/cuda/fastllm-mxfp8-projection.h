#pragma once
#include <cstdint>
#include <cstddef>

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

// Each request writes at the parent matrix pitch. Only producers that enqueue
// all their packed rows contribute to the count. A BF16 fallback leaves it unchanged.
struct FastllmMxfp8PrefillOutput {
    FastllmMxfp8Activation slice{};
    int *completedRows = nullptr;

    static FastllmMxfp8PrefillOutput ForRows(FastllmMxfp8Activation matrix, int first, int count, int &completed) {
        if (!matrix.values || !matrix.scales || first < 0 || count <= 0 || first > matrix.rows - count) return {};
        matrix.values += std::size_t(first) * matrix.width;
        matrix.scales += std::size_t(first) * 4;
        matrix.rows = count;
        return {matrix, &completed};
    }
    bool Accepts(int rows, int width) const {
        return completedRows && slice.rows == rows && slice.width == width;
    }
    void Complete() const { *completedRows += slice.rows; }
};

// Each CUDA backend opts in to the shared activation layout independently of
// its weight layout. prepareWeight runs on the final shard before capture.
struct FastllmMxfp8ProjectionBackend {
    void (*prepareWeight)(fastllm::Data &weight);
    // Zero declines this weight/phase. A positive limit bounds the shared arena.
    int (*maxRows)(const fastllm::Data &weight, bool verifying);
    // Called on the shard's device. False must leave the output untouched.
    bool (*project)(FastllmMxfp8Activation input, const fastllm::Data &weight, void *output);

    // Prefill producers can outlive a single request and borrow a much larger
    // arena. A backend must admit its full matrix before any request packs rows.
    bool (*canPackPrefill)(const fastllm::Data &weight, int rows) = nullptr;

    FastllmMxfp8Activation Bind(const fastllm::Data &weight, int rows, bool verifying,
                                int maxWidth, fastllm::Data &storage) const;
    // Borrow existing storage only between the producer and its projection.
    FastllmMxfp8Activation Borrow(const fastllm::Data &weight, int rows, const fastllm::Data &scratch,
                                  bool prefill = false) const;
    bool TryProject(FastllmMxfp8Activation prepared, const fastllm::Data &input,
                    const fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output) const;
};
// Null when no compiled consumer supports the current device. Producers keep
// their BF16 output so the existing Linear is always available.
const FastllmMxfp8ProjectionBackend *FastllmCudaGetMxfp8ProjectionBackend();
