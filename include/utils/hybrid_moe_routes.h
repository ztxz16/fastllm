#pragma once

#include "devices/moe_decode_scheduler.h"
#include "devices/moe_frequency_policy.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace fastllm {
// Preserve route order within each expert. Charge a missed expert one transfer,
// even when multiple requests or verification rows reuse it.
struct HybridMoeRoutes {
    std::vector<std::vector<int>> grouped;
    std::vector<int> hits, misses;
    int residentRows = 0, stagedRows = 0, staged = 0;

    HybridMoeRoutes(const std::vector<int> &ids, int experts, int rows, int keyBase, bool prefill,
                    MoeFrequencyPolicy *frequency, const MoeDecodeOverlapScheduler &timing, int stagingSlots)
        : grouped(experts) {
        for (int r = 0; r < int(ids.size()); ++r) {
            if (ids[r] < 0 || ids[r] >= experts) throw std::out_of_range("Hybrid MoE expert index");
            grouped[ids[r]].push_back(r);
        }
        std::vector<int> active;
        if (frequency && prefill) prefillCounts.resize(experts);
        for (int e = 0; e < experts; ++e) {
            if (grouped[e].empty()) continue;
            if (frequency) {
                if (prefill) prefillCounts[e] = grouped[e].size();
                else active.push_back(e);
            }
            if (frequency && frequency->Slot(keyBase + e) >= 0) {
                hits.push_back(e);
                residentRows += grouped[e].size();
            } else {
                misses.push_back(e);
            }
        }
        if (!frequency) return;
        if (prefill) {
            frequency->ObservePrefill(keyBase, prefillCounts, rows, true);
        } else {
            frequency->Observe(keyBase, active.data(), active.size());
        }
        std::stable_sort(misses.begin(), misses.end(), [&](int a, int b) {
            return grouped[a].size() > grouped[b].size();
        });
        std::vector<int> reuse;
        reuse.reserve(misses.size());
        for (int e : misses) reuse.push_back(grouped[e].size());
        staged = std::min(stagingSlots, timing.SelectMisses(misses.size(), residentRows, reuse.data()));
        stagedRows = std::accumulate(reuse.begin(), reuse.begin() + staged, 0);
    }

    MoeFrequencyPolicy::Admission selectPrefill(const MoeFrequencyPolicy &frequency, int keyBase, int expert) const {
        return frequency.SelectPrefill(keyBase + expert, keyBase, prefillCounts, frequency.HasDecodeHistory());
    }
private:
    // The eviction policy indexes this by expert ID, including inactive experts.
    std::vector<int> prefillCounts;
};
} // namespace fastllm
