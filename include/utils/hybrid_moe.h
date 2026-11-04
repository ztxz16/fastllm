#pragma once

#include "fastllm.h"
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace fastllm {
// Numerical semantics belong to the expert layout, not to the storage tier.
// Keep score placement, projection rounding and activation quantization distinct.
enum class HybridMoeMath { Standard, Glm5, DeepSeekV41 };

// Both placements use the same kernels and cache policy in NUMA-enabled builds.
bool IsSharedHostMoeDevice(const std::string &device);

// Full prefill timings are meaningful for these host-expert layouts. Keep
// this capability independent of model names and cache capacity.
bool HybridMoeMeasuredNativePrefill(const Data *gate, const Data *down,
                                   HybridMoeMath math);

// A short-lived preparation plan borrows the original expert tables. GPU and
// disk-resident layers stay outside the host cache. CPU tables register as one
// unbound shard; NUMA tables use the supplied placement-aware registration.
class HybridMoeCachePlan {
public:
    explicit HybridMoeCachePlan(HybridMoeMath math = HybridMoeMath::Standard,
                               float swigluLimit = 0.0f, double budgetFraction = 1.0)
        : math(math), swigluLimit(swigluLimit), budgetFraction(budgetFraction) {}
    void Add(const std::vector<Data *> &weights, const std::string &device,
             int cacheDevice = -1);
    bool Prepare(const std::function<void()> &registerNumaWeights = {},
                 bool allowStreaming = false) const;
    bool Prepare(const std::function<void()> &registerNumaWeights, bool allowStreaming,
                 const std::map<int, uint64_t> &deviceBudgets, bool fullyResident) const;
private:
    struct Layer { Data *const *weights; int count; int cacheDevice; };
    std::vector<Layer> layers;
    HybridMoeMath math;
    float swigluLimit;
    double budgetFraction; // Model subgroups share one per-device cache budget.
    bool allNuma = true, allCpu = true;
};

// Packed native BF16 experts share the NUMA workers and generic cache /
// whole-expert scheduling policies. The caller retains immutable CPU weights
// and serializes Run calls. Construction registers host shards before requests.
class HybridMoeBF16 {
public:
    HybridMoeBF16(const std::vector<std::pair<Data *, Data *>> &layers, int device);
    ~HybridMoeBF16();
    struct Stats { uint64_t cpuRoutes = 0, gpuRoutes = 0, cacheHits = 0, uploads = 0; };
    Stats GetStats() const;
    // Set before a serialized forward; prefill does not create decode heat.
    void SetPrefill(bool prefill);
    // The callback may submit GPU work while NUMA workers are active; it must
    // not submit CPU work to the shared host MoE pool.
    void Run(const Data &input, const Data &index, const Data &score,
             Data &output, int layer, const std::function<void()> &launchParallel = {});
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

// One complete target token/verification step owns one policy scope. Cache
// admission must finish outside captured GPU segments, including on exceptions.
// Weight-table pointers remain cache keys; this class never copies the tables.
class HybridMoeDecodeStep {
public:
    // Concurrent TP ranks own disjoint caches and finish only their owner.
    explicit HybridMoeDecodeStep(bool deviceLocal = false) : deviceLocal(deviceLocal) {}
    HybridMoeDecodeStep(const HybridMoeDecodeStep &) = delete;
    HybridMoeDecodeStep &operator=(const HybridMoeDecodeStep &) = delete;
    HybridMoeDecodeStep(HybridMoeDecodeStep &&other) noexcept
        : policy(std::exchange(other.policy, nullptr)), deviceLocal(other.deviceLocal) {}
    ~HybridMoeDecodeStep();
    void Begin(Data **weights, int count, int topk,
               const std::vector<int> *devices = nullptr);
    // Join deferred CPU work without ending cache observation/admission.
    void WaitCpu();
    void Finish();
    explicit operator bool() const { return policy != nullptr; }
private:
    void *policy = nullptr;
    bool deviceLocal = false;
};

// One policy per cache-owning device, started lazily when its first MoE layer
// is reached. Failed/ineligible starts are not retained; later layers may retry.
class HybridMoeDecodeGroup {
public:
    ~HybridMoeDecodeGroup();
    bool Contains(int device) const { return steps.count(device) != 0; }
    bool Begin(int device, Data **weights, int count, int topk,
               const std::vector<int> *devices = nullptr);
    // Submit admissions after this device's last expert use in the forward.
    // The caller must rule out later stages/peers reading its cache. Retain the
    // completed entry so it cannot start a second policy step in this group.
    void Finish(int device);
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

// Host experts can borrow multiple GPUs independently of dense PP/TP placement.
// The caller hands off their streams; format-specific expert math and ordered
// reduction stay in the backend. Residency is independent of temporary work.
bool RunHybridMoeOnDevices(const Data &input, const Data &index, const Data &score,
                          Data &output, Data **weights, int count, int layer,
                          const std::vector<int> &devices,
                          const std::function<void()> &launchParallel = {});
}
