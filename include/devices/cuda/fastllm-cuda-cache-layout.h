#pragma once
#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace fastllm { namespace cuda {
struct CacheSlotSpan { int begin = 0, count = 0; };
struct CacheSlotPlan {
    std::vector<uint64_t> offsets;
    std::vector<CacheSlotSpan> layers;
    uint64_t bytes = 0;
};

// Equal residency fractions across record sizes. Layers of the same size
// share an LRU partition; their quantization format need not be identical.
// For tiny budgets, retain the uniform-slot cache so every layer can run.
inline CacheSlotPlan PlanCacheSlots(const std::vector<size_t> &strides,
                                    int experts, uint64_t budget, int topk) {
    CacheSlotPlan plan;
    if (strides.empty() || experts <= 0 || topk <= 0 ||
        strides.size() > size_t(INT_MAX / experts)) return plan;
    std::map<size_t, int> records;
    uint64_t full = 0, minimum = 0;
    for (auto stride : strides) {
        if (!stride || stride % 16 || uint64_t(experts) > (UINT64_MAX - full) / stride) return plan;
        full += uint64_t(experts) * stride;
        records[stride] += experts;
    }
    for (auto item : records) minimum += uint64_t(std::min(topk, item.second)) * item.first;
    budget = std::min(budget, full);
    if (budget < minimum) {
        const size_t stride = records.rbegin()->first;
        const int count = int(std::min(budget / stride, uint64_t(strides.size()) * experts));
        if (count < topk) return plan;
        for (int i = 0; i < count; ++i) plan.offsets.push_back(uint64_t(i) * stride);
        plan.bytes = uint64_t(count) * stride;
        plan.layers.assign(strides.size(), {0, count});
        return plan;
    }
    auto countAt = [topk](int records, double fraction) {
        return std::min(records, std::max(std::min(topk, records), int(records * fraction)));
    };
    double lo = 0, hi = 1;
    for (int step = 0; step < 60; ++step) {
        const double mid = (lo + hi) / 2;
        uint64_t bytes = 0;
        for (auto item : records) bytes += uint64_t(countAt(item.second, mid)) * item.first;
        if (bytes <= budget) lo = mid; else hi = mid;
    }
    std::map<size_t, CacheSlotSpan> spans;
    for (auto item : records) {
        const int count = countAt(item.second, lo);
        spans[item.first] = {int(plan.offsets.size()), count};
        for (int i = 0; i < count; ++i) {
            plan.offsets.push_back(plan.bytes);
            plan.bytes += item.first;
        }
    }
    for (auto stride : strides) plan.layers.push_back(spans.at(stride));
    return plan;
}
} } // namespace fastllm::cuda
