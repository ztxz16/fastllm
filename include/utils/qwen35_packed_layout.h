#ifndef FASTLLM_QWEN35_PACKED_LAYOUT_H
#define FASTLLM_QWEN35_PACKED_LAYOUT_H

#include <cstddef>
#include <climits>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace fastllm {
namespace qwen35_packed {

inline bool Fail(std::string *error, const char *message) {
    if (error != nullptr) *error = message;
    return false;
}

// Divide the existing round budget among remaining ready rows. A single row
// keeps its normal chunk cap at the caller; no workspace or page pool is grown.
inline int PrefillTokenShare(int remainingBudget, int remainingRows) {
    if (remainingBudget <= 0 || remainingRows <= 0) return 0;
    return remainingBudget < remainingRows ? 1 : remainingBudget / remainingRows;
}

inline bool BuildRowOffsets(const std::vector<int> &lengths,
                            std::vector<int> &offsets, std::string *error = nullptr) {
    offsets.clear();
    if (lengths.empty()) return Fail(error, "empty packed batch");
    offsets.push_back(0);
    for (int length : lengths) {
        if (length <= 0 || length > INT_MAX - offsets.back()) {
            offsets.clear();
            return Fail(error, "invalid packed sequence length or total overflow");
        }
        offsets.push_back(offsets.back() + length);
    }
    return true;
}

struct PositionRow {
    const float *values;
    size_t count;
    int axes;
    int stride;
};

// Position values already contain their row's cache base/MRoPE delta. Never
// add a packed offset to them: only the destination index uses packed offsets.
inline bool PackPositionRows(const std::vector<PositionRow> &rows,
                             const std::vector<int> &lengths, bool all1,
                             std::vector<float> &packed, int &axes,
                             std::string *error = nullptr) {
    packed.clear(); axes = 0;
    std::vector<int> offsets;
    if (!BuildRowOffsets(lengths, offsets, error)) return false;
    if (rows.size() != lengths.size()) return Fail(error, "position row count mismatch");
    axes = rows[0].axes;
    if (axes != 1 && axes != 3) return Fail(error, "invalid position axis count");
    for (size_t row = 0; row < rows.size(); ++row) {
        const PositionRow &source = rows[row];
        if (source.axes != axes || source.values == nullptr ||
            source.stride < lengths[row] || (all1 && lengths[row] != 1) ||
            source.count < size_t(source.axes - 1) * size_t(source.stride) + size_t(lengths[row]))
            return Fail(error, "invalid position row layout or mixed axes");
    }
    const size_t total = size_t(offsets.back());
    if (total > packed.max_size() / size_t(axes))
        return Fail(error, "position allocation size overflow");
    packed.resize(total * size_t(axes));
    for (int axis = 0; axis < axes; ++axis) {
        for (size_t row = 0; row < rows.size(); ++row) {
            std::memcpy(packed.data() + size_t(axis) * total + size_t(offsets[row]),
                        rows[row].values + size_t(axis) * size_t(rows[row].stride),
                        size_t(lengths[row]) * sizeof(float));
        }
    }
    return true;
}

inline bool ValidateMediaRows(const std::vector<bool> &media,
                              std::string *error = nullptr) {
    if (media.empty()) return Fail(error, "empty media batch");
    for (bool hasMedia : media)
        if (!hasMedia) return Fail(error, "mixed text and multimodal rows");
    return true;
}

// Empty masks mean the target's implicit causal mask. Explicit masks must be
// exactly that same 0/1 mask; an arbitrary mask cannot be silently discarded.
inline bool ValidateCausalMask(const float *values, size_t count,
                               int queryLength, int cacheLength,
                               int maskRows, int maskColumns,
                               std::string *error = nullptr) {
    if (queryLength <= 0 || cacheLength < 0 || cacheLength > INT_MAX - queryLength)
        return Fail(error, "invalid causal mask lengths");
    if (values == nullptr && count == 0 && maskRows == 0 && maskColumns == 0)
        return true;
    const int columns = cacheLength + queryLength;
    if (values == nullptr || maskRows != queryLength || maskColumns != columns ||
        size_t(maskRows) > std::numeric_limits<size_t>::max() / size_t(columns) ||
        count != size_t(maskRows) * size_t(columns))
        return Fail(error, "causal mask shape mismatch");
    for (int query = 0; query < queryLength; ++query)
        for (int key = 0; key < columns; ++key)
            if (values[size_t(query) * size_t(columns) + size_t(key)] !=
                (key > cacheLength + query ? 1.0f : 0.0f))
                return Fail(error, "noncausal packed mask is unsupported");
    return true;
}

struct HiddenRow {
    const void *values;
    size_t bytes;
    int tokens;
    int width;
    int type;
};

inline bool CopyPackedHiddenRows(const std::vector<HiddenRow> &rows,
                                 const std::vector<int> &lengths,
                                 size_t tokenBytes, void *destination,
                                 size_t destinationBytes,
                                 std::string *error = nullptr) {
    std::vector<int> offsets;
    if (!BuildRowOffsets(lengths, offsets, error)) return false;
    if (rows.size() != lengths.size() || destination == nullptr || tokenBytes == 0 ||
        size_t(offsets.back()) > std::numeric_limits<size_t>::max() / tokenBytes ||
        destinationBytes != size_t(offsets.back()) * tokenBytes)
        return Fail(error, "packed hidden output size mismatch");
    for (size_t row = 0; row < rows.size(); ++row) {
        if (rows[row].values == nullptr || rows[row].tokens != lengths[row] ||
            rows[row].width <= 0 || rows[row].width != rows[0].width ||
            rows[row].type != rows[0].type ||
            rows[row].bytes != size_t(lengths[row]) * tokenBytes)
            return Fail(error, "packed hidden row size or type mismatch");
    }
    // Validate every row before writing any bytes, so rejection leaves output intact.
    for (size_t row = 0; row < rows.size(); ++row)
        std::memcpy(static_cast<unsigned char*>(destination) + size_t(offsets[row]) * tokenBytes,
                    rows[row].values, rows[row].bytes);
    return true;
}

} // namespace qwen35_packed
} // namespace fastllm
#endif
