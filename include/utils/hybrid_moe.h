#pragma once

#include "fastllm.h"
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace fastllm {
// One complete target token/verification step owns one policy scope. Cache
// admission must finish outside captured GPU segments, including on exceptions.
// Weight-table pointers remain cache keys; this class never copies the tables.
class HybridMoeDecodeStep {
public:
    HybridMoeDecodeStep() = default;
    HybridMoeDecodeStep(const HybridMoeDecodeStep &) = delete;
    HybridMoeDecodeStep &operator=(const HybridMoeDecodeStep &) = delete;
    HybridMoeDecodeStep(HybridMoeDecodeStep &&other) noexcept
        : policy(std::exchange(other.policy, nullptr)) {}
    ~HybridMoeDecodeStep();
    void Begin(Data **weights, int count, int topk,
               const std::vector<int> *devices = nullptr);
    void Finish();
    explicit operator bool() const { return policy != nullptr; }
private:
    void *policy = nullptr;
};

// One policy per cache-owning device, started lazily when its first MoE layer
// is reached. Failed/ineligible starts are not retained; later layers may retry.
class HybridMoeDecodeGroup {
public:
    ~HybridMoeDecodeGroup();
    bool Contains(int device) const { return steps.count(device) != 0; }
    bool Begin(int device, Data **weights, int count, int topk,
               const std::vector<int> *devices = nullptr);
private:
    std::map<int, HybridMoeDecodeStep> steps;
};

// Device/shape eligibility is centralized; model code only supplies placement,
// the requested layer and whether it is running a pipelined prefill.
void HybridMoePrefetch(std::vector<std::vector<Data *>> &weights,
                       int layer, int endLayer, bool pipelinedPrefill,
                       int rows, const std::string &expertDevice);

// Shared hybrid backend: its format adapter retains cache eviction, admission,
// dynamic CPU/GPU split, uploads and CPU overlap. Unsupported layouts return
// false so the caller can use its established host backend.
bool RunHybridMoe(const Data &input, const Data &index, const Data &score,
                  Data &output, Data **weights, int count, int layer,
                  const std::function<void()> &launchParallel = {});

// Tensor-parallel host experts can use multiple GPU cache/assist devices while
// retaining their existing format-specific reduction and shared-expert overlap.
bool RunHybridMoeOnDevices(const Data &input, const Data &index, const Data &score,
                          Data &output, Data **weights, int count, int layer,
                          const std::vector<int> &devices,
                          const std::function<void()> &launchParallel = {});
}
