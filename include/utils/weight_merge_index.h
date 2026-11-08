#pragma once

#include <cstddef>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace fastllm {

using WeightMergeIndex = std::unordered_map<std::string_view, std::vector<std::size_t>>;

// Keys borrow allInputs' strings. The caller must keep rules unchanged and alive;
// destroy/clear the index before changing rules, then rebuild it before lookup.
template <class RuleRange>
WeightMergeIndex BuildWeightMergeIndex(const RuleRange &rules) {
    WeightMergeIndex index;
    std::size_t inputs = 0;
    for (const auto &rule : rules) inputs += rule.allInputs.size();
    index.reserve(inputs);
    std::size_t ruleId = 0;
    for (const auto &rule : rules) {
        for (const auto &name : rule.allInputs) {
            index[std::string_view(name)].push_back(ruleId);
        }
        ++ruleId;
    }
    return index;
}

template <class RuleRange>
WeightMergeIndex BuildWeightMergeIndex(const RuleRange &&rules) = delete;

inline const std::vector<std::size_t> &FindWeightMergeRules(
        const WeightMergeIndex &index, std::string_view name) {
    static const std::vector<std::size_t> empty;
    const auto found = index.find(name);
    return found == index.end() ? empty : found->second;
}

} // namespace fastllm
