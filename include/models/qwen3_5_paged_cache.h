#ifndef FASTLLM_QWEN3_5_PAGED_CACHE_H
#define FASTLLM_QWEN3_5_PAGED_CACHE_H

#include <cstddef>
#include <unordered_set>
#include <vector>

namespace fastllm {
    // Return old pages absent from the restored view, preserving their order.
    // Speculative rollback normally keeps the whole page list, or a prefix.
    // Do not assume sorted or unique page IDs: shared/reordered views must
    // retain the same set-difference semantics as the general rollback path.
    inline std::vector<int> Qwen35UnreferencedPages(
            const std::vector<int> &current, const std::vector<int> &retained) {
        std::size_t prefix = 0;
        while (prefix < current.size() && prefix < retained.size() &&
               current[prefix] == retained[prefix]) {
            ++prefix;
        }
        if (prefix == current.size()) {
            return {};
        }
        std::unordered_set<int> keep(retained.begin(), retained.end());
        std::vector<int> released;
        released.reserve(current.size() - prefix);
        for (std::size_t i = prefix; i < current.size(); ++i) {
            if (keep.count(current[i]) == 0) {
                released.push_back(current[i]);
            }
        }
        return released;
    }
}

#endif
