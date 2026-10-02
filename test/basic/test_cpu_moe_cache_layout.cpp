#include "devices/cuda/fastllm-cuda-cache-layout.h"
#include <cstdio>
#include <random>
#include <stdexcept>

static void Check(bool ok) { if (!ok) throw std::runtime_error("invalid cache slot plan"); }
int main() {
    using fastllm::cuda::PlanCacheSlots;
    try {
        std::mt19937 random(38094);
        for (int test = 0; test < 1000; ++test) {
            const int experts = 16 + random() % 500, layers = 1 + random() % 48;
            std::vector<size_t> sizes(layers);
            uint64_t full = 0;
            size_t maximum = 0;
            for (auto &size : sizes) { size = 128 * (1 + random() % 8); full += size * experts; maximum = std::max(size, maximum); }
            const uint64_t budget = test % 3 ? random() % (full + 1) : full;
            const auto plan = PlanCacheSlots(sizes, experts, budget, 16);
            Check(plan.bytes <= budget);
            if (plan.offsets.empty()) { Check(budget < 16 * maximum); continue; }
            Check(plan.layers.size() == sizes.size());
            Check(plan.offsets.front() == 0);
            Check(plan.offsets.size() <= size_t(layers * experts));
            for (int layer = 0; layer < layers; ++layer) {
                const auto span = plan.layers[layer];
                Check(span.begin >= 0 && span.count >= 16 && size_t(span.begin + span.count) <= plan.offsets.size());
                for (int slot = span.begin; slot < span.begin + span.count; ++slot) {
                    const uint64_t end = slot + 1 < int(plan.offsets.size()) ? plan.offsets[slot + 1] : plan.bytes;
                    Check(plan.offsets[slot] % 16 == 0 && plan.offsets[slot] + sizes[layer] <= end);
                }
            }
            if (budget == full) Check(plan.offsets.size() == size_t(layers * experts) && plan.bytes == full);
        }
        Check(PlanCacheSlots({128,256}, 24, 16*256, 16).offsets.size() == 16);
        Check(PlanCacheSlots({128,256}, 24, 32*256, 16).offsets.size() > 32);
        Check(PlanCacheSlots({0}, 24, 8192, 16).offsets.empty());
        Check(PlanCacheSlots({17}, 24, 8192, 16).offsets.empty());
        Check(PlanCacheSlots({128}, -1, 8192, 16).offsets.empty());
        std::puts("PASS: cache size classes, budgets, fallback and non-overlapping records"); return 0;
    } catch (const std::exception &e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
