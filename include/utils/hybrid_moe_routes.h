#pragma once

#include "devices/moe_decode_scheduler.h"
#include "devices/moe_frequency_policy.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace fastllm {
// Observe early replays and periodically refresh costs after warmup.
inline bool SampleHybridMoeReplay(uint64_t replays) {
    return replays < 8 || (replays % 32) == 0;
}

// Preserve route order within each expert. Charge a missed expert one transfer,
// even when multiple requests or verification rows reuse it.
struct HybridMoeRoutes {
    std::vector<std::vector<int>> grouped;
    std::vector<int> hits, misses;
    int residentRows = 0, stagedRows = 0, staged = 0;

    HybridMoeRoutes(const std::vector<int> &ids, int experts, int rows, int keyBase, bool prefill,
                    MoeFrequencyPolicy *frequency, const MoeDecodeOverlapScheduler &timing, int stagingSlots, bool probeUpload = false)
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
        const bool decode = !prefill && rows == 1 && hits.size() + misses.size() == ids.size();
        const int selected = decode ? timing.SelectDecodeMisses(misses.size(), residentRows)
                                    : timing.SelectMisses(misses.size(), residentRows, reuse.data());
        // Match an infrequently used transfer to a timed replay. Use one of the
        // actual misses, preserving the cache capacity and the uniqueness order.
        staged = std::min(stagingSlots, std::max(selected, int(probeUpload && !misses.empty())));
        stagedRows = std::accumulate(reuse.begin(), reuse.begin() + staged, 0);
    }

    static void ObserveCpu(MoeDecodeOverlapScheduler &timing, int rows, int routes, double us) {
        if (!routes) return;
        if (rows == 1 && routes < int(timing.decodeCpu.size())) timing.ObserveDecodeCpu(routes, us);
        else timing.cpuExpert.Observe(us / routes);
    }

    MoeFrequencyPolicy::Admission selectPrefill(const MoeFrequencyPolicy &frequency, int keyBase, int expert) const {
        return frequency.SelectPrefill(keyBase + expert, keyBase, prefillCounts, frequency.HasDecodeHistory());
    }
private:
    // The eviction policy indexes this by expert ID, including inactive experts.
    std::vector<int> prefillCounts;
};
} // namespace fastllm
