#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace fastllm {
struct MoePrefillExpertCost {
    int routes;
    bool pinned;
};
struct MoePrefillSplit {
    int limit = 0;
    double cpuMs = 0, gpuMs = 0;
};
// A valid all-GPU prediction must not permanently stop CPU feedback. Probe
// actual expert IDs: a threshold could move an unbounded group of tied experts.
// Callers serialize this state with their timing samples and skip explicit
// placement overrides and formats whose bootstrap requires synthetic profiling.
class MoePrefillCpuProbe {
    int cooldown = 0;
    int interval = 1;
public:
    void ObserveCpu(double sample, double previousEstimate) {
        // Cold first-touch costs decay over several samples. Continue promptly
        // while measurements improve; genuinely slow CPU paths back off.
        if (std::isfinite(sample) && sample > 0 && previousEstimate > 0 &&
                sample < previousEstimate * .75) {
            cooldown = 0;
            interval = 1;
        }
    }

    std::vector<int> Select(const std::vector<MoePrefillExpertCost> &experts, int limit) {
        if (limit > 1) {
            cooldown = 8; // Ordinary CPU work is already refreshing the model.
            interval = 1;
            return {};
        }
        if (cooldown > 0) { --cooldown; return {}; }
        std::vector<int> order;
        int active = 0;
        int64_t totalRoutes = 0;
        for (int e = 0; e < int(experts.size()); ++e) if (experts[e].routes > 0) {
            ++active;
            totalRoutes += experts[e].routes;
            if (!experts[e].pinned) order.push_back(e);
        }
        if (order.empty() || active < 2) return {};
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return experts[a].routes != experts[b].routes
                ? experts[a].routes < experts[b].routes : a < b;
        });
        // At most two experts and 1/16 of routes, allowing one indivisible
        // expert for small layers. Always leave work on GPU for paired samples.
        const int64_t budget = std::max<int64_t>(totalRoutes / 16, experts[order[0]].routes);
        int64_t routes = 0;
        size_t count = 0;
        while (count < order.size() && count < 2 && int(count) < active - 1 &&
                routes + experts[order[count]].routes <= budget)
            routes += experts[order[count++]].routes;
        order.resize(count);
        cooldown = interval - 1;
        interval = std::min(interval * 2, 64);
        return order;
    }
};
// Single-device makespan search. Equal thresholds move as one group, and
// strict improvement retains the original lowest-threshold tie break.
inline MoePrefillSplit PredictSingleGpuPrefillSplit(
        const std::vector<MoePrefillExpertCost> &experts,
        double cpuMsPerRoute, double gpuMsPerExpert) {
    std::vector<int> movable;
    size_t active = 0;
    for (const auto &e : experts) if (e.routes > 0) {
        ++active;
        if (!e.pinned) movable.push_back(e.routes);
    }
    if (!active) return {};
    std::sort(movable.begin(), movable.end());
    // Preserve the exhaustive solver's floating-point accumulation exactly.
    std::vector<double> gpuCosts(active + 1, 0);
    for (size_t i = 1; i <= active; ++i) gpuCosts[i] = gpuCosts[i - 1] + gpuMsPerExpert;
    MoePrefillSplit best{1, 0, gpuCosts[active]};
    double routes = 0, bestCost = best.gpuMs;
    for (size_t i = 0; i < movable.size();) {
        const int count = movable[i];
        do { routes += movable[i++]; } while (i < movable.size() && movable[i] == count);
        const double cpuMs = routes * cpuMsPerRoute, gpuMs = gpuCosts[active - i];
        const double cost = std::max(cpuMs, gpuMs);
        if (cost < bestCost) { best = {count + 1, cpuMs, gpuMs}; bestCost = cost; }
    }
    return best;
}
} // namespace fastllm
