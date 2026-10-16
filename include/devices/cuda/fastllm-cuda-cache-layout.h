#pragma once
#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>
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

// Keep the original layer/table indices while excluding other PP stages from
// the capacity calculation. Unrestricted layers (e.g. a shared draft) remain.
inline CacheSlotPlan PlanDeviceCacheSlots(const std::vector<size_t> &strides,
        const std::vector<int> &owners, int device, int experts,
        uint64_t budget, int topk) {
    if (owners.empty()) return PlanCacheSlots(strides, experts, budget, topk);
    if (owners.size() != strides.size()) return {};
    std::vector<size_t> local;
    for (size_t i = 0; i < strides.size(); ++i)
        if (owners[i] < 0 || owners[i] == device) local.push_back(strides[i]);
    auto plan = PlanCacheSlots(local, experts, budget, topk);
    auto spans = std::move(plan.layers);
    // A distinct empty partition cannot borrow slots from the first layer.
    plan.layers.assign(strides.size(), {int(plan.offsets.size()), 0});
    if (!spans.empty()) {
        size_t next = 0;
        for (size_t i = 0; i < strides.size(); ++i)
            if (owners[i] < 0 || owners[i] == device) plan.layers[i] = spans[next++];
    }
    return plan;
}
} } // namespace fastllm::cuda
