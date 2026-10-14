#include "utils/hybrid_moe.h"
#if defined(USE_CUDA) && !defined(USE_ROCM)
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-moe-policy.h"
#include "devices/cuda/fastllm-moe-prefetch.h"
#endif

namespace fastllm {
HybridMoeDecodeStep::~HybridMoeDecodeStep() { Finish(); }
void HybridMoeDecodeStep::Begin(Data **weights, int count, int topk,
                               const std::vector<int> *devices) {
    Finish();
#if defined(USE_CUDA) && !defined(USE_ROCM)
    policy = FastllmCudaBeginMoeDecode(weights, count, topk, devices);
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
    if (Contains(device)) return true;
    HybridMoeDecodeStep step;
    step.Begin(weights, count, topk, devices);
    if (!step) return false;
    steps.emplace(device, std::move(step));
    return true;
}
void HybridMoePrefetch(std::vector<std::vector<Data *>> &weights,
                       int layer, int endLayer, bool pipelinedPrefill,
                       int rows, const std::string &expertDevice) {
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM)
    if (!pipelinedPrefill || rows < 1024 || layer < 0 || layer >= endLayer ||
        layer >= int(weights.size()) ||
        (expertDevice != "numa" && expertDevice.rfind("numa:", 0) != 0)) return;
    auto &table = weights[layer];
    if (table.size() <= 2) return;
    FastllmCudaPrefetchMoeGGUFHost(table.data(), int(table.size()) / 2 - 1);
#endif
}
bool RunHybridMoe(const Data &input, const Data &index, const Data &score,
                  Data &output, Data **weights, int count, int layer,
                  const std::function<void()> &launchParallel) {
#if defined(USE_CUDA) && !defined(USE_ROCM)
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
    return FastllmCudaMergeMOEHybridOnDevices(input, index, score, output,
        weights, count, layer, devices, launchParallel);
#else
    return false;
#endif
}
}
