#pragma once

#include <algorithm>
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

// Ascending, unique buckets; the endpoint is always included. Alignment is a
// backend choice (e.g. the EP group size), not a requirement of the scheduler.
inline std::vector<int> DecodeGraphBatchBuckets(int maxRows, int alignment = 8, int exactRows = 4) {
    if (maxRows <= 0 || alignment <= 0) return {};
    std::vector<int> result;
    for (int n = 1; n <= std::min(maxRows, exactRows); ++n) result.push_back(n);
    for (int n = alignment; n <= maxRows; n += alignment) result.push_back(n);
    result.push_back(maxRows);
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}
}
