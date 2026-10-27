#ifndef FASTLLM_MOE_EXPERT_PARTITION_H
#define FASTLLM_MOE_EXPERT_PARTITION_H

#include <algorithm>
#include <functional>
#include <limits>
#include <set>
#include <vector>

namespace fastllm {
namespace detail {

struct MoeExpertCost {
    int routes;
    double cpuUs;
    double gpuUs;
};

inline int SelectMoeExpertLimit(const std::vector<MoeExpertCost> &experts,
                                int defaultLimit, int gpuCount) {
    // A partition changes only when the threshold crosses an active expert's
    // route count. Include all-CPU execution when GPU transfers are costly.
    // The default is a fallback, not an upper bound on the measured optimum.
    std::set<int> candidates = {1};
    for (const auto &expert : experts) {
        if (expert.routes > 0) candidates.insert(expert.routes + 1);
    }
    if (candidates.size() == 1) return defaultLimit;

    int bestLimit = defaultLimit;
    double bestTime = std::numeric_limits<double>::max();
    for (int limit : candidates) {
        double cpuTime = 0.0;
        std::vector<double> gpuJobs;
        for (const auto &expert : experts) {
            if (expert.routes <= 0) continue;
            if (expert.routes < limit) cpuTime += expert.cpuUs;
            else gpuJobs.push_back(expert.gpuUs);
        }
        std::sort(gpuJobs.begin(), gpuJobs.end(), std::greater<double>());
        std::vector<double> gpuLoads(std::max(1, gpuCount), 0.0);
        for (double job : gpuJobs) {
            *std::min_element(gpuLoads.begin(), gpuLoads.end()) += job;
        }
        const double elapsed = std::max(cpuTime,
            *std::max_element(gpuLoads.begin(), gpuLoads.end()));
        if (elapsed < bestTime) {
            bestTime = elapsed;
            bestLimit = limit;
        }
    }
    return bestLimit;
}

} // namespace detail
} // namespace fastllm

#endif
