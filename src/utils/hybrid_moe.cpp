#include "utils/hybrid_moe.h"
#ifdef USE_NUMAS
#include "devices/numas/numasdevice.h"
#endif
#if defined(USE_CUDA) && !defined(USE_ROCM)
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-moe-policy.h"
#include "devices/cuda/fastllm-moe-prefetch.h"
#endif

namespace fastllm {
bool IsSharedHostMoeDevice(const std::string &device) {
    if (device == "numa" || device.rfind("numa:", 0) == 0) return true;
#ifdef USE_NUMAS
    return device == "cpu";
#else
    return false;
#endif
}
bool HybridMoeMeasuredNativePrefill(const Data *gate, const Data *down, HybridMoeMath math) {
    if (!gate || !down || gate->dataType != down->dataType) return false;
    switch (gate->dataType) {
    case FP8_E4M3_BLOCK_128:
        return true;
    case FP8_E4M3: case FP8_E4M3_PERCHANNEL:
        return math == HybridMoeMath::Standard;
    case NVFP4_BLOCK_16: case NVFP4_BLOCK_16_E4M3:
        return math == HybridMoeMath::Standard;
    case NVFP4_BLOCK_16_E4M3_PACKED:
        return math == HybridMoeMath::Standard || math == HybridMoeMath::Glm5;
    case NVFP4_BLOCK_32_E8M0:
        return math == HybridMoeMath::DeepSeekV41;
    default:
        return false;
    }
}

void HybridMoeCachePlan::Add(const std::vector<Data *> &weights,
                            const std::string &device, int cacheDevice) {
    const bool numa = device == "numa" || device.rfind("numa:", 0) == 0;
    if (device != "cpu" && !numa) return;
    layers.push_back({weights.data(), static_cast<int>(weights.size()), cacheDevice});
    allNuma = allNuma && numa;
    allCpu = allCpu && device == "cpu";
}
bool HybridMoeCachePlan::Prepare(const std::function<void()> &registerNumaWeights,
                                 bool allowStreaming) const {
#if defined(USE_CUDA) && !defined(USE_ROCM)
    if (layers.empty()) return false;
#ifdef USE_NUMAS
    HostMoePlacementScope placement(allCpu);
#endif
    std::function<void()> registerWeights = allNuma ? registerNumaWeights : std::function<void()>{};
#ifdef USE_NUMAS
    if (allCpu) {
        // Registration belongs to the host backend, regardless of the model's
        // math profile. Keep the cache backend's snapshot/packing order intact.
        for (const auto &layer : layers)
            for (int i = 0; i < layer.count; ++i) {
                const auto *w = layer.weights[i];
                if (w && (w->isDiskWeight || !IsNumasLinearWeightSupported(w) ||
                    (!w->numasData.empty() && !w->hostMoeUnbound))) return false;
            }
        registerWeights = [&] {
            if (registerNumaWeights) registerNumaWeights();
            std::vector<std::pair<Data *, std::string>> pending;
            for (const auto &layer : layers)
                for (int i = 0; i < layer.count; ++i)
                    if (layer.weights[i] && !IsNumasLinearWeightRegistered(layer.weights[i]))
                        pending.push_back({layer.weights[i], i % 2 ? "linear" : "linearSwiglu"});
            if (!pending.empty()) RegisterHostMoeWeightBatch(pending, true);
        };
    }
#endif
    std::vector<FastllmCudaMoeCacheLayer> cacheLayers;
    cacheLayers.reserve(layers.size());
    for (const auto &layer : layers) {
        cacheLayers.push_back({layer.weights, layer.count,
            math == HybridMoeMath::DeepSeekV41, swigluLimit, math == HybridMoeMath::Glm5,
            layer.cacheDevice});
    }
    // The backend retains format-dependent snapshot/registration order and
    // publishes a cache only after validation. Failure leaves host fallback intact.
    if (!FastllmCudaPrepareMoeCache(cacheLayers.data(), static_cast<int>(cacheLayers.size()),
            registerWeights, allowStreaming)) return false;
    return FastllmCudaSetMoeCacheBudgetFraction(const_cast<Data **>(layers.front().weights),
        layers.front().count, budgetFraction);
#else
    return false;
#endif
}

HybridMoeDecodeStep::~HybridMoeDecodeStep() { Finish(); }
void HybridMoeDecodeStep::Begin(Data **weights, int count, int topk,
                               const std::vector<int> *devices) {
    Finish();
#if defined(USE_CUDA) && !defined(USE_ROCM)
    policy = FastllmCudaBeginMoeDecode(weights, count, topk, devices);
#endif
}
void HybridMoeDecodeStep::WaitCpu() {
#if defined(USE_CUDA) && !defined(USE_ROCM)
    FastllmCudaWaitMoeDecodeCpu(policy);
#endif
}
void HybridMoeDecodeStep::Finish() {
    if (!policy) return;
#if defined(USE_CUDA) && !defined(USE_ROCM)
    void *current = policy;
    policy = nullptr;
    FastllmCudaEndMoeDecode(current);
#endif
}
HybridMoeDecodeGroup::~HybridMoeDecodeGroup() {
    // Preserve deterministic device order. Individual destructors become no-ops.
    for (auto &entry : steps) entry.second.Finish();
}
bool HybridMoeDecodeGroup::Begin(int device, Data **weights, int count, int topk,
                                 const std::vector<int> *devices) {
    const auto found = steps.find(device);
    if (found != steps.end()) return bool(found->second);
    HybridMoeDecodeStep step;
    step.Begin(weights, count, topk, devices);
    if (!step) return false;
    steps.emplace(device, std::move(step));
    return true;
}
void HybridMoeDecodeGroup::Finish(int device) {
    const auto found = steps.find(device);
    if (found != steps.end()) found->second.Finish();
}
void HybridMoePrefetch(std::vector<std::vector<Data *>> &weights,
                       int layer, int endLayer, bool pipelinedPrefill,
                       int rows, const std::string &expertDevice) {
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM)
    if (!pipelinedPrefill || rows < 1024 || layer < 0 || layer >= endLayer ||
        layer >= int(weights.size()) ||
        !IsSharedHostMoeDevice(expertDevice)) return;
    auto &table = weights[layer];
    if (table.size() <= 2) return;
    HostMoePlacementScope placement(table[2]);
    FastllmCudaPrefetchMoeGGUFHost(table.data(), int(table.size()) / 2 - 1);
#endif
}
bool RunHybridMoe(const Data &input, const Data &index, const Data &score,
                  Data &output, Data **weights, int count, int layer,
                  const std::function<void()> &launchParallel) {
#if defined(USE_CUDA) && !defined(USE_ROCM)
#ifdef USE_NUMAS
    HostMoePlacementScope placement(weights && count >= 4 ? weights[2] : nullptr);
#endif
    return FastllmCudaMergeMOEHybrid(input, index, score, output, weights, count,
                                   layer, launchParallel);
#else
    return false;
#endif
}
bool RunHybridMoeOnDevices(const Data &input, const Data &index, const Data &score,
                          Data &output, Data **weights, int count, int layer,
                          const std::vector<int> &devices,
                          const std::function<void()> &launchParallel) {
#if defined(USE_CUDA) && !defined(USE_ROCM)
#ifdef USE_NUMAS
    HostMoePlacementScope placement(weights && count >= 4 ? weights[2] : nullptr);
#endif
    return FastllmCudaMergeMOEHybridOnDevices(input, index, score, output,
        weights, count, layer, devices, launchParallel);
#else
    return false;
#endif
}
}
