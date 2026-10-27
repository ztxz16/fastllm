#pragma once

#include <algorithm>
#include <climits>
#include <cstdint>
#include <vector>

namespace fastllm {

// Graph shapes and request counts are deliberately separate. The scheduler,
// cache ownership and sampling use actualRows; captured kernels use graphRows.
// Model adapters must supply device metadata marking the tail inactive and
// must not read/write request state for those rows. Selection never captures.
struct DecodeGraphBatch {
    int actualRows = 0;
    int graphRows = 0;

    explicit operator bool() const { return actualRows > 0 && graphRows >= actualRows; }

    static DecodeGraphBatch Select(int actualRows, const std::vector<int> &sortedBuckets) {
        if (actualRows <= 0) return {};
        auto it = std::lower_bound(sortedBuckets.begin(), sortedBuckets.end(), actualRows);
        return it == sortedBuckets.end() ? DecodeGraphBatch{} : DecodeGraphBatch{actualRows, *it};
    }
};

// Align the final compute bucket too: an unaligned endpoint can disable a
// backend's sharded execution. Admission limits remain the scheduler's job.
inline std::vector<int> DecodeGraphBatchBuckets(int maxRows, int alignment = 8, int exactRows = 4) {
    if (maxRows <= 0 || alignment <= 0) return {};
    const int64_t last = maxRows <= exactRows ? maxRows :
        ((int64_t)maxRows + alignment - 1) / alignment * alignment;
    if (last > INT_MAX) return {};
    std::vector<int> result;
    for (int n = 1; n <= std::min(maxRows, exactRows); ++n) result.push_back(n);
    for (int64_t n = alignment; n <= last; n += alignment) result.push_back((int)n);
    result.push_back((int)last);
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}
}
