#pragma once
#include <algorithm>
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
