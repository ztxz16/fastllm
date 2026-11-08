#ifndef FASTLLM_QWEN35_BATCH_ROWS_H
#define FASTLLM_QWEN35_BATCH_ROWS_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace fastllm {

struct ResponseContext;

// Proposed pure metadata contract for one row of a multimodal native forward.
// It carries identity and layout only; it does not own KV, linear, MTP, DFlash,
// image, or cache storage.
struct Qwen35MultimodalBatchRow {
    ResponseContext *context = nullptr;
    uint64_t nativeHandle = 0;
    uint64_t nativeGeneration = 0;
    int seqLen = 0;
    std::size_t tokenOffset = 0;
    std::size_t positionOffset = 0;
    std::size_t maskOffset = 0;
    int positionAxes = 3;
    int positionTokens = 0;
    int cacheLen = 0;
    int embeddingCursor = 0;
    int embeddingTokens = 0;
    int maskRows = 0;
    int maskColumns = 0;
    std::size_t maskCount = 0;
    bool multimodal = true;
};

namespace qwen35_batch_rows {

struct CpuPositionPlan {
    int axes = 3;
    int stride = 0;
    std::vector<float> values; // axis-major: [axis * stride + token]
};

struct CpuMaskView {
    const float *values = nullptr;
    std::size_t count = 0;
    int rows = 0;
    int columns = 0;
};

inline bool Fail(std::string *error, const char *message) {
    if (error != nullptr) *error = message;
    return false;
}

inline bool BuildRowOffsets(const std::vector<int> &lengths,
                            std::vector<std::size_t> &offsets,
                            std::string *error = nullptr) {
    offsets.clear();
    if (lengths.empty()) return Fail(error, "empty multimodal batch");
    offsets.push_back(0);
    for (int length : lengths) {
        if (length <= 0 || static_cast<std::size_t>(length) >
                              std::numeric_limits<std::size_t>::max() - offsets.back()) {
            offsets.clear();
            return Fail(error, "invalid row length or token-offset overflow");
        }
        offsets.push_back(offsets.back() + static_cast<std::size_t>(length));
    }
    return true;
}

// CPU-only equivalent of the candidate SplitMultimodalPositionIdsCPU contract.
// It performs no model/cache mutation and preserves the three-axis row layout.
inline bool SplitMultimodalPositionIdsCPU(const CpuPositionPlan &positions,
                                          int start, int end,
                                          CpuPositionPlan &chunk,
                                          std::string *error = nullptr) {
    if (positions.axes != 3 || positions.stride <= 0 || start < 0 || end <= start ||
        end > positions.stride || positions.values.size() <
            static_cast<std::size_t>(positions.axes) * static_cast<std::size_t>(positions.stride)) {
        return Fail(error, "invalid CPU multimodal position plan");
    }
    chunk.axes = positions.axes;
    chunk.stride = end - start;
    chunk.values.assign(static_cast<std::size_t>(chunk.axes) * chunk.stride, 0.0f);
    for (int axis = 0; axis < chunk.axes; ++axis) {
        std::copy(positions.values.begin() +
                      static_cast<std::size_t>(axis) * positions.stride + start,
                  positions.values.begin() +
                      static_cast<std::size_t>(axis) * positions.stride + end,
                  chunk.values.begin() + static_cast<std::size_t>(axis) * chunk.stride);
    }
    return true;
}

inline bool ValidateCausalMask(const CpuMaskView &mask,
                               int queryLength, int cacheLength,
                               std::string *error = nullptr) {
    if (queryLength <= 0 || cacheLength < 0 ||
        cacheLength > std::numeric_limits<int>::max() - queryLength) {
        return Fail(error, "invalid causal-mask lengths");
    }
    if (mask.values == nullptr && mask.count == 0 && mask.rows == 0 && mask.columns == 0)
        return true;
    const int columns = cacheLength + queryLength;
    if (mask.values == nullptr || mask.rows != queryLength || mask.columns != columns ||
        mask.count != static_cast<std::size_t>(mask.rows) * mask.columns) {
        return Fail(error, "causal-mask shape mismatch");
    }
    for (int query = 0; query < queryLength; ++query) {
        for (int key = 0; key < columns; ++key) {
            const float expected = key > cacheLength + query ? 1.0f : 0.0f;
            if (mask.values[static_cast<std::size_t>(query) * columns + key] != expected)
                return Fail(error, "noncausal multimodal mask");
        }
    }
    return true;
}

// Read-only validation. Call this before allocating/writing KV, linear, MTP,
// DFlash, image, or prefix-cache state. A false return leaves all caller-owned
// storage untouched.
inline bool ValidateRows(const std::vector<Qwen35MultimodalBatchRow> &rows,
                         const std::vector<CpuMaskView> &masks,
                         std::size_t expectedTotalTokens,
                         std::string *error = nullptr) {
    if (rows.empty()) return Fail(error, "empty multimodal row set");
    if (!masks.empty() && masks.size() != rows.size())
        return Fail(error, "row/mask count mismatch");

    std::set<std::pair<uint64_t, uint64_t>> identities;
    std::vector<int> lengths;
    lengths.reserve(rows.size());
    for (const Qwen35MultimodalBatchRow &row : rows) {
        if (row.nativeHandle == 0 || row.nativeGeneration == 0 || row.seqLen <= 0)
            return Fail(error, "missing row identity or length");
        if (!identities.insert({row.nativeHandle, row.nativeGeneration}).second)
            return Fail(error, "duplicate request generation");
        // Text rows share the batch (B mixes them with media rows) and still use
        // three-axis MRoPE positions; only media rows carry prepared embeddings.
        if (row.positionAxes != 3 || row.positionTokens < row.seqLen)
            return Fail(error, "row is not a valid three-axis row");
        if (!row.multimodal && (row.embeddingCursor != 0 || row.embeddingTokens != 0))
            return Fail(error, "text row carries media embeddings");
        if (row.embeddingCursor < 0 || row.embeddingTokens < 0 ||
            (row.embeddingTokens > 0 &&
             row.seqLen > row.embeddingTokens - row.embeddingCursor))
            return Fail(error, "embedding cursor exceeds prepared media suffix");
        if (!masks.empty()) {
            const CpuMaskView &mask = masks[&row - rows.data()];
            if (!ValidateCausalMask(mask, row.seqLen, row.cacheLen, error)) return false;
        }
        lengths.push_back(row.seqLen);
    }

    std::vector<std::size_t> offsets;
    if (!BuildRowOffsets(lengths, offsets, error)) return false;
    if (offsets.back() != expectedTotalTokens)
        return Fail(error, "packed token total mismatch");
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (rows[i].tokenOffset != offsets[i])
            return Fail(error, "non-prefix-sum token offset");
    }
    return true;
}

inline bool ValidateRows(const std::vector<Qwen35MultimodalBatchRow> &rows,
                         std::size_t expectedTotalTokens,
                         std::string *error = nullptr) {
    return ValidateRows(rows, std::vector<CpuMaskView>(), expectedTotalTokens, error);
}

} // namespace qwen35_batch_rows
} // namespace fastllm

#endif // FASTLLM_QWEN35_BATCH_ROWS_H
