// Shared PP host-expert cooperation. Numerical adapters remain in the CUDA backend.
#include "utils/hybrid_parallel.h"
#include "utils.h"
#if defined(USE_CUDA) && !defined(USE_ROCM)
#include "fastllm-multicuda.cuh"
#include "devices/cuda/fastllm-cuda-moe-policy.h"
namespace fastllm {
namespace {
struct HybridCudaDeviceGuard {
    int previous = FastllmCudaGetDevice();
    ~HybridCudaDeviceGuard() { FastllmCudaSetDevice(previous); }
};
}
#if defined(USE_NUMAS) && defined(__linux__)
HybridMoePeerRuntime::CallerAffinity::CallerAffinity(HybridMoePeerRuntime *state) {
    // Create the peer before narrowing the caller mask, so it does not inherit
    // only the caller's cores. Restore the serving thread after this request.
    if (state && state->workerCpus.size() == 2 && state->workers.HasWorkers() &&
        sched_getaffinity(0, sizeof(original), &original) == 0)
        restore = BindNumasWorkerCpuSet(state->workerCpus[0]);
}
HybridMoePeerRuntime::CallerAffinity::~CallerAffinity() {
    if (restore) sched_setaffinity(0, sizeof(original), &original);
}
#endif
HybridMoePeerRuntime::HybridMoePeerRuntime(const std::vector<CudaPrefillStage> &stages) {
    if (stages.size() != 2)
        return;
    HybridCudaDeviceGuard guard;
    for (const auto &stage : stages)
        devices.push_back(stage.device);
#ifdef USE_NUMAS
    workerCpus = GetNumasCudaWorkerCpuSets(devices);
#endif
    // Peer copies otherwise silently stage through host memory, even with
    // NVLink present. Failure leaves the existing staged-copy fallback.
    peerAccess = FastllmCudaPeerAccessInit(devices);
    experts = FastllmCudaCreateMoeExpertParallel(2);
    for (int rank = 0; rank < 2; ++rank) {
        FastllmCudaSetDevice(devices[rank]);
        ready[rank] = FastllmCudaEventCreate();
        done[rank] = FastllmCudaEventCreate();
    }
    AssertInFastLLM(!FastllmCudaGetThreadError(), "PP MoE event initialization failed.");
}
HybridMoePeerRuntime::~HybridMoePeerRuntime() {
    workers.Stop();
    HybridCudaDeviceGuard guard;
    experts.reset();
    for (int rank = 0; rank < int(devices.size()); ++rank) {
        FastllmCudaSetDevice(devices[rank]);
        FastllmCudaSyncCurrentThreadStream();
        peerInput[rank].FreeSpace();
        peerOutput[rank].FreeSpace();
        sum[rank].FreeSpace();
        if (ready[rank])
            FastllmCudaEventDestroy(ready[rank]);
        if (done[rank])
            FastllmCudaEventDestroy(done[rank]);
    }
}

bool HybridMoePeerRuntime::Readback(const Data &input, const Data &index, const Data &score) {
#ifdef USE_NUMAS
    const auto found = std::find(devices.begin(), devices.end(), FastllmCudaGetDevice());
    return found != devices.end() &&
        FastllmCudaMoeExpertParallelReadback(*experts, found - devices.begin(), input, index, score);
#else
    return false;
#endif
}
bool HybridMoePeerRuntime::Run(int layer, const Data &input, const Data &index,
                               const Data &score, Data &output,
                               Data **moeWeights, int weightsCount,
                               const std::function<void()> &sharedExpert,
                               Data **deferredPeer, bool preparedReadback) {
#ifdef USE_NUMAS
    if (input.dataDevice != CUDA ||
        input.dims.size() != 2 || input.dims[0] < 1 ||
        input.dims[0] > FASTLLM_CUDA_MOE_CACHE_MAX_BATCH || !FastllmCudaMoeCacheRequested() ||
        FastllmCudaGraphIsCapturingFast())
        return false;
    if (devices.size() != 2)
        return false;
    const int device = FastllmCudaGetDevice();
    const auto found = std::find(devices.begin(), devices.end(), device);
    if (found == devices.end())
        return false;
    const int owner = found - devices.begin(), peer = 1 - owner;
    HybridCudaDeviceGuard guard;
    auto prepare = [&](Data &data, int target, DataType type) {
        FastllmCudaSetDevice(target);
        data.dataType = type;
        data.Resize({FASTLLM_CUDA_MOE_CACHE_MAX_BATCH, input.dims[1]});
        data.ToDevice(CUDA, std::vector<int>{target}, false);
        data.Allocate(false);
        data.Resize(input.dims);
    };
    FastllmCudaEventRecordCurrentThread(ready[owner]);
    const bool borrowInput =
        peerAccess && input.dataDeviceIds == std::vector<int>{device};
    bool accepted[2] = {false, false};
    std::vector<std::exception_ptr> errors(2);
    // Keep physical rank/expert ownership stable as the PP layer owner moves.
    // Only the narrow normalized activation and the reduced MoE result travel.
    workers.RunWithCaller(
        {0, 1},
        [&](int worker) {
            const auto thread = std::this_thread::get_id();
            if (worker != 0 && boundPeerThread != thread) {
                BindNumasWorkerCpuSet(workerCpus[worker]);
                boundPeerThread = thread;
            }
            const int rank = worker == 0 ? owner : peer;
            const int target = devices[rank];
            void *previous = GetExecutor();
            struct RestoreExecutor {
                void *previous;
                bool previousCaptureMode;
                ~RestoreExecutor() {
                    SetCurrentThreadExecutor(previous);
                    FastllmCudaGraphSetManagedCaptureOnly(previousCaptureMode);
                }
            } restore{previous, FastllmCudaGraphSetManagedCaptureOnly(true)};
            FastllmCudaSetDevice(target);
            if (worker != 0) {
                // A stream wait cannot protect storage freed by host-side
                // reallocation. Fence only calls that may replace peer buffers.
                const uint64_t required = uint64_t(FASTLLM_CUDA_MOE_CACHE_MAX_BATCH) * input.dims[1];
                auto needsStorage = [&](const Data &data, DataType type) {
                    return !data.cudaData || data.dataDevice != CUDA ||
                        data.dataDeviceIds.size() != 1 || data.dataDeviceIds[0] != target ||
                        data.dataType != type || data.expansionSize < required;
                };
                if (needsStorage(peerOutput[peer], FLOAT32) ||
                    (!borrowInput && needsStorage(peerInput[peer], input.dataType)))
                    FastllmCudaEventSynchronize(ready[owner]);
                FastllmCudaCurrentThreadStreamWaitEvent(ready[owner]);
                // Dense graph variants retain this peer output across changing
                // verifier lengths. Its allocation must not grow on a later row.
                prepare(peerOutput[peer], target, FLOAT32);
                // Allocate on the thread that owns the peer stream, avoiding a
                // switch to the peer device and back on every owner submission.
                if (!borrowInput)
                    prepare(peerInput[peer], target, input.dataType);
                static thread_local std::unique_ptr<Executor> executor(new Executor());
                executor->SetFirstDevice("cuda:" + std::to_string(target));
                SetCurrentThreadExecutor(executor.get());
                if (!borrowInput && !FastllmCudaMemcpyPeerAsyncCurrentThread(
                                        target, peerInput[peer].cudaData, device,
                                        input.cudaData, input.GetBytes())) {
                    FastllmCudaEventSynchronize(ready[owner]);
                    FastllmCudaMemcpyBetweenDevices(target, peerInput[peer].cudaData,
                                                    device, input.cudaData, input.GetBytes());
                }
            }
            accepted[rank] = FastllmCudaMergeMOEExpertParallel(
                *experts, rank,
                worker == 0 || borrowInput ? input : peerInput[peer], index, score,
                worker == 0 ? output : peerOutput[peer], moeWeights,
                weightsCount, layer,
                worker == 0 ? sharedExpert : std::function<void()>(), owner,
                preparedReadback && worker == 0);
            if (worker != 0)
                FastllmCudaEventRecordCurrentThread(done[rank]);
            // The persistent peer keeps its stream alive. The owner's event wait
            // below joins GPU completion; do not also block its CPU worker here.
        },
        errors);
    FastllmCudaSetDevice(device);
    for (const auto &error : errors)
        if (error)
            std::rethrow_exception(error);
    AssertInFastLLM(accepted[0] == accepted[1], "PP MoE collective rejection mismatch.");
    if (!accepted[owner])
        return false;
    FastllmCudaCurrentThreadStreamWaitEvent(done[peer]);
    if (peerAccess && output.dataType == FLOAT32 &&
        peerOutput[peer].dataType == FLOAT32 &&
        output.dims == peerOutput[peer].dims) {
        // Both allocations are peer-accessible. The event protects the remote
        // read; the next input-ready event protects it from peer reuse.
        if (deferredPeer) {
            // The per-rank output stays at a stable address. Consume it in
            // the next dense graph, after this stream's peer-completion wait.
            *deferredPeer = &peerOutput[peer];
            return true;
        }
        return FastllmCudaAddTo(output, peerOutput[peer], 1.0f);
    }
    prepare(sum[owner], device, FLOAT32);
    if (!FastllmCudaMemcpyPeerAsyncCurrentThread(
            device, sum[owner].cudaData, devices[peer],
            peerOutput[peer].cudaData, sum[owner].GetBytes())) {
        FastllmCudaEventSynchronize(done[peer]);
        FastllmCudaMemcpyBetweenDevices(device, sum[owner].cudaData, devices[peer],
                                        peerOutput[peer].cudaData,
                                        sum[owner].GetBytes());
    }
    AddTo(output, sum[owner]);
    return true;
#else
    return false;
#endif
}
} // namespace fastllm
#endif
