#include "utils/hybrid_moe.h"
#include "utils.h"
#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-graph-workspace.h"
#include "devices/multicuda/fastllm-multicuda.cuh"
#endif
#ifdef USE_NUMAS
#include "devices/numas/numasdevice.h"
#endif
#if defined(USE_CUDA) && !defined(USE_ROCM)
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-moe-policy.h"
#include "devices/cuda/fastllm-moe-prefetch.h"
#endif

namespace fastllm {
#ifdef USE_CUDA
HybridMoeGraph::~HybridMoeGraph() { Reset(); }
void HybridMoeGraph::Reset() {
    const int previous = FastllmCudaGetDevice();
    if (device >= 0) FastllmCudaSetDevice(device);
    if (replay) FastllmCudaGraphExecDestroy(replay);
    if (exec) FastllmCudaGraphExecDestroy(exec);
    if (graph) FastllmCudaGraphDestroy(graph);
    for (void *handle : handles) FastllmCudaReleaseNativeMoeFullGraph(handle);
    handles.clear();
    graph = exec = replay = nullptr;
    launches = 0;
    device = -1;
    FastllmCudaSetDevice(previous);
}
bool HybridMoeGraph::Compose(const std::vector<const void *> &children,
                            std::vector<void *> ownedHandles) {
    Reset();
    device = FastllmCudaGetDevice();
    handles = std::move(ownedHandles);
    bool ok = !children.empty() &&
        FastllmCudaGraphCompose(children.data(), children.size(), &graph, &exec);
    if (ok && !handles.empty())
        ok = FastllmCudaCreateNativeMoeReplayGraph(graph, handles.data(), handles.size(), &replay);
    if (!ok) Reset();
    return ok;
}
bool HybridMoeGraph::Launch() {
    if (!exec) return false;
    const bool timed = launches < 8 || launches % 64 == 0;
    const bool ok = handles.empty() ? FastllmCudaGraphLaunch(exec) :
        FastllmCudaLaunchNativeMoeFullGraph(timed ? exec : replay,
            handles.data(), handles.size(), timed);
    if (ok) ++launches;
    return ok;
}

HybridMoeRemoteWorkspace::~HybridMoeRemoteWorkspace() {
    const int original = FastllmCudaGetDevice();
    if (source >= 0) FastllmCudaSyncDevice(source);
    if (device >= 0) FastllmCudaSyncDevice(device);
    if (ready) FastllmCudaEventDestroy(ready);
    if (done) FastllmCudaEventDestroy(done);
    FastllmCudaSetDevice(original);
}

void RunRemoteHybridMoe(HybridMoeRemoteWorkspace &remote,
    const Data &input, const Data &index, const Data &score, Data &output, int device,
    const std::function<void(HybridMoeRemoteWorkspace &)> &compute) {
    AssertInFastLLM(input.dataDevice == CUDA && input.dataDeviceIds.size() == 1 &&
        index.dataDevice == CUDA && score.dataDevice == CUDA &&
        index.dataDeviceIds == input.dataDeviceIds && score.dataDeviceIds == input.dataDeviceIds &&
        input.cudaData && index.cudaData && score.cudaData && device >= 0 &&
        device != input.dataDeviceIds.front(), "Invalid remote resident MoE placement.");
    const int source = input.dataDeviceIds.front();
    FastllmCudaSetDevice(source);
    const bool capturing = FastllmCudaGraphIsCapturingFast();
    if (!remote.ready) {
        AssertInFastLLM(!capturing, "Remote MoE must warm up before graph capture.");
        remote.source = source; remote.device = device;
        remote.ready = FastllmCudaEventCreate();
        FastllmCudaSetDevice(device);
        remote.done = FastllmCudaEventCreate();
        AssertInFastLLM(FastllmCudaGraphPrepareCaptureDevice(), "Remote MoE capture preparation failed.");
    }
    AssertInFastLLM(remote.source == source && remote.device == device,
        "Remote MoE graph owner changed.");
    auto allocate = [](Data &dst, DataType type, const std::vector<int> &dims, int owner) {
        if (dst.dataDevice != CUDA || dst.dataDeviceIds != std::vector<int>{owner} || dst.dataType != type) {
            dst.FreeSpace(); dst.dataType = type; dst.UpdateUnitSize();
            dst.dataDevice = CUDA; dst.dataDeviceIds = {owner};
        }
        dst.Resize(dims); dst.Allocate(false);
    };
    auto copy = [](Data &dst, const Data &src) {
        AssertInFastLLM(FastllmCudaCopyFromDeviceToDeviceAsyncCurrentThread(
            dst.cudaData, src.cudaData, src.GetBytes()), "Remote MoE async transfer failed.");
    };
    FastllmCudaSetDevice(device);
    allocate(remote.input, input.dataType, input.dims, device);
    allocate(remote.index, index.dataType, index.dims, device);
    allocate(remote.score, score.dataType, score.dims, device);
    FastllmCudaSetDevice(source);
    copy(remote.input, input); copy(remote.index, index); copy(remote.score, score);
    FastllmCudaEventRecordCurrentThread(remote.ready);
    FastllmCudaSetDevice(device);
    FastllmCudaCurrentThreadStreamWaitEvent(remote.ready);
    if (capturing) AssertInFastLLM(FastllmCudaGraphJoinCapture(), "Remote MoE capture join failed.");
    std::vector<size_t> allocations;
    {
        FastllmCudaAllocationRecorder recorder(capturing ? nullptr : &allocations);
        compute(remote);
    }
    FastllmCudaEventRecordCurrentThread(remote.done);
    if (!capturing) {
        FastllmCudaEventSynchronize(remote.done);
        AssertInFastLLM(FastllmCudaReserveGraphWorkspace(allocations), "Remote MoE scratch reservation failed.");
    }
    FastllmCudaSetDevice(source);
    FastllmCudaCurrentThreadStreamWaitEvent(remote.done);
    allocate(output, remote.output.dataType, remote.output.dims, source);
    copy(output, remote.output);
}
#endif

bool IsSharedHostMoeDevice(const std::string &device) {
    if (device == "numa" || device.rfind("numa:", 0) == 0) return true;
#ifdef USE_NUMAS
    return device == "cpu";
#else
    return false;
#endif
}
bool MoeHasNativeFp32Output(const Data *weight, HybridMoeMath math) {
    return weight && weight->dataType == EXL3 && math == HybridMoeMath::DeepSeekV41;
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
    return Prepare(registerNumaWeights, allowStreaming, {}, false);
}
bool HybridMoeCachePlan::Prepare(const std::function<void()> &registerNumaWeights,
                                 bool allowStreaming, const std::map<int, uint64_t> &deviceBudgets,
                                 bool fullyResident) const {
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
            layer.cacheDevice, math == HybridMoeMath::Qwen38FlashNext});
    }
    // The backend retains format-dependent snapshot/registration order and
    // publishes a cache only after validation. Failure leaves host fallback intact.
    if (!FastllmCudaPrepareMoeCacheWithBudgets(cacheLayers.data(), static_cast<int>(cacheLayers.size()),
            registerWeights, allowStreaming, deviceBudgets, fullyResident)) return false;
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
    if (deviceLocal) FastllmCudaEndMoeDecodeOnDevice(current);
    else FastllmCudaEndMoeDecode(current);
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
