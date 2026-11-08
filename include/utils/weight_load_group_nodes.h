#pragma once

#include "weight_merge_index.h"
#include <set>

namespace fastllm {

// Call before starting workers; preserves existing nodes and their payloads.
template <class WeightMap, class Rules, class Names>
std::size_t PrepareMergeOutputNodes(WeightMap &weights, const Rules &rules,
        const WeightMergeIndex &index, const Names &names) {
    std::set<std::size_t> ruleIds;
    for (const auto &name : names) {
        const auto found = index.find(std::string_view(name));
        if (found == index.end()) continue;
        ruleIds.insert(found->second.begin(), found->second.end());
    }
    std::size_t created = 0;
    for (const std::size_t ruleId : ruleIds) {
        const auto &rule = rules.at(ruleId);
        for (const auto &single : rule.rules) {
            auto inserted = weights.emplace(single.output, typename WeightMap::mapped_type{});
            if (inserted.second) ++created;
        }
    }
    return created;
}

// Caller holds the existing loader lock; metadata remains until workers join.
template <class WeightMap, class Inputs, class Pending>
void ReleaseMergedInputs(WeightMap &weights, const Inputs &inputs, Pending &pending) {
    for (const auto &name : inputs) {
        const auto found = weights.find(name);
        if (found == weights.end()) continue;
        found->second.FreeSpace();
        pending.insert(name);
    }
}

// Call after every group worker joins and before the group's finish callback.
template <class WeightMap, class Pending>
void EraseConsumedWeightNodes(WeightMap &weights, Pending &pending) {
    for (const auto &name : pending) weights.erase(name);
    pending.clear();
}

} // namespace fastllm
