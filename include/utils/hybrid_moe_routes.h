#pragma once

#include "devices/moe_decode_scheduler.h"
#include "devices/moe_frequency_policy.h"
#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace fastllm {
// Observe early replays and periodically refresh costs after warmup.
inline bool SampleHybridMoeReplay(uint64_t replays) {
    return replays < 8 || (replays % 32) == 0;
}

// Temporary slots of a captured graph are refilled together before its staged
// projection starts. The resident and NUMA branches can run during that refill.
struct HybridMoeGraphSlots {
    std::vector<int> keys, selected;
    int Find(int expert) const {
        auto it = std::find(keys.begin(), keys.end(), expert);
        return it == keys.end() ? -1 : int(it - keys.begin());
    }

    void Select(std::vector<int> &misses, const std::vector<std::vector<int>> &grouped,
                int warm, int rows, int totalRoutes, int residentRows,
                const MoeDecodeOverlapScheduler &timing, bool probe) {
        selected.clear();
        const int capacity = std::min<int>(keys.size(), totalRoutes);
        if (!capacity || misses.empty()) return;
        const auto cold = misses.begin() + warm;
        const int fresh = int(misses.size()) - warm;
        int takeWarm = warm, takeFresh = 0;
        if (timing.cpuExpert.initialized && timing.stagedExpert.initialized) {
            const double residentUs = residentRows * timing.residentExpert.us;
            int coldRows = 0;
            double best = std::numeric_limits<double>::infinity();
            // Each new record has the same byte size within this layer. A slot
            // hit costs no transfer, regardless of its position in this batch.
            for (int n = 0; n <= std::min(fresh, capacity); ++n) {
                if (n && !timing.copiedExpert.initialized) break;
                if (n) coldRows += grouped[cold[n - 1]].size();
                int gpuRows = coldRows;
                for (int h = 0; h <= std::min(warm, capacity - n); ++h) {
                    if (h) gpuRows += grouped[misses[h - 1]].size();
                    const int cpuRows = totalRoutes - residentRows - gpuRows;
                    const double cpuUs = rows == 1 ? timing.DecodeCpuUs(cpuRows) : cpuRows * timing.cpuExpert.us;
                    const double stagedUs = n * timing.copiedExpert.us + gpuRows * timing.stagedExpert.us;
                    const double finish = std::max({cpuUs, residentUs, stagedUs});
                    // Equal finish times favor fewer uploads, then more existing slot hits.
                    if (finish < best || (finish == best && n == takeFresh)) {
                        best = finish;
                        takeFresh = n;
                        takeWarm = h;
                    }
                }
            }
        }
        if (probe && fresh && !takeFresh && takeWarm < capacity) takeFresh = 1;
        // Keep all selected slot hits protected before assigning destinations.
        for (int i = 0; i < takeWarm; ++i) selected.push_back(Find(misses[i]));
        for (int i = 0; i < takeFresh; ++i) {
            next %= capacity;
            while (std::find(selected.begin(), selected.end(), next) != selected.end())
                next = (next + 1) % capacity;
            selected.push_back(next++);
        }
        std::rotate(misses.begin() + takeWarm, cold, cold + takeFresh);
    }
private:
    int next = 0;
};

// Preserve route order within each expert. Charge a missed expert one transfer,
// even when multiple requests or verification rows reuse it.
struct HybridMoeRoutes {
    std::vector<std::vector<int>> grouped;
    std::vector<int> hits, misses;
    int residentRows = 0, stagedRows = 0, staged = 0;

    HybridMoeRoutes() = default;
    HybridMoeRoutes(const std::vector<int> &ids, int experts, int rows, int keyBase, bool prefill,
                    MoeFrequencyPolicy *frequency, const MoeDecodeOverlapScheduler &timing, int stagingSlots,
                    bool probeUpload = false) {
        Group(ids.data(), ids.size(), experts, rows, keyBase, prefill, frequency);
        if (!frequency) return;
        std::vector<int> reuse;
        reuse.reserve(misses.size());
        for (int e : misses) reuse.push_back(grouped[e].size());
        const bool decode = !prefill && rows == 1 && int(hits.size() + misses.size()) == int(ids.size());
        const int selected = decode ? timing.SelectDecodeMisses(misses.size(), residentRows)
                                    : timing.SelectMisses(misses.size(), residentRows, reuse.data());
        // Match an infrequently used transfer to a timed replay. Use one of the
        // actual misses, preserving the cache capacity and the uniqueness order.
        staged = std::min(stagingSlots, std::max(selected, int(probeUpload && !misses.empty())));
        stagedRows = std::accumulate(reuse.begin(), reuse.begin() + staged, 0);
    }

    void BuildGraph(const int *ids, int count, int experts, int rows, int keyBase,
                    MoeFrequencyPolicy *frequency, const MoeDecodeOverlapScheduler &timing,
                    HybridMoeGraphSlots &slots, bool probeUpload) {
        const int warm = Group(ids, count, experts, rows, keyBase, false, frequency, &slots);
        if (!frequency) {
            slots.selected.clear();
            return;
        }
        slots.Select(misses, grouped, warm, rows, count, residentRows, timing, probeUpload);
        staged = slots.selected.size();
        for (int i = 0; i < staged; ++i) stagedRows += grouped[misses[i]].size();
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
    // Reuse the grouping vectors across serialized graph visits. Warm and cold
    // misses are each ordered by reuse, then by expert ID for deterministic ties.
    int Group(const int *ids, int count, int experts, int rows, int keyBase, bool prefill,
              MoeFrequencyPolicy *frequency, const HybridMoeGraphSlots *slots = nullptr) {
        grouped.resize(experts);
        for (auto &g : grouped) g.clear();
        hits.clear();
        misses.clear();
        residentRows = stagedRows = staged = 0;
        for (int r = 0; r < count; ++r) {
            if (ids[r] < 0 || ids[r] >= experts) throw std::out_of_range("Hybrid MoE expert index");
            grouped[ids[r]].push_back(r);
        }
        if (frequency && prefill) prefillCounts.assign(experts, 0);
        for (int e = 0; e < experts; ++e) {
            if (grouped[e].empty()) continue;
            if (frequency) {
                if (prefill) prefillCounts[e] = grouped[e].size();
                else frequency->Observe(keyBase, &e, 1);
            }
            if (frequency && frequency->Slot(keyBase + e) >= 0) {
                hits.push_back(e);
                residentRows += grouped[e].size();
            } else {
                misses.push_back(e);
            }
        }
        if (!frequency) return 0;
        if (prefill) frequency->ObservePrefill(keyBase, prefillCounts, rows, true);
        auto cold = misses.begin();
        if (slots && !slots->keys.empty())
            cold = std::partition(misses.begin(), misses.end(), [&](int e) { return slots->Find(e) >= 0; });
        auto byReuse = [&](int a, int b) {
            if (grouped[a].size() != grouped[b].size()) return grouped[a].size() > grouped[b].size();
            return a < b;
        };
        std::sort(misses.begin(), cold, byReuse);
        std::sort(cold, misses.end(), byReuse);
        return int(cold - misses.begin());
    }

    // The eviction policy indexes this by expert ID, including inactive experts.
    std::vector<int> prefillCounts;
};
} // namespace fastllm
