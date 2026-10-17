#pragma once

#include "fastllm.h"
#include "utils.h"
#include <mutex>
#include <optional>
#ifdef USE_CUDA
#include "utils/cuda_chunked_prefill.h"
#ifndef USE_ROCM
#include "devices/cuda/fastllm-moe-prefetch.h"
#endif
#endif
#ifdef USE_NUMAS
#include "devices/numas/numasdevice.h"
#ifdef __linux__
#include <sched.h>
#endif
#endif

namespace fastllm {
// The model owns layer/request state; the hybrid runtime owns host-worker
// exclusion, per-stage NUMA staging and submission-thread placement.
struct HybridPrefillContext {
    int firstLayer;
    std::mutex *hostExperts;
    std::vector<int> devices;
};

class HybridMoePipelineGuard {
public:
    HybridMoePipelineGuard(const Data &input, const std::string &expertDevice,
                           std::mutex *mutex, int workspace) {
        const bool numa = expertDevice == "numa" || expertDevice.rfind("numa:", 0) == 0;
        if (mutex && (numa || expertDevice == "cpu")) {
            lock = std::unique_lock<std::mutex>(*mutex, std::defer_lock);
            // Large NUMA batches release/reacquire the pool around GPU work.
            // Generic CPU work and decode keep the guard throughout the call.
            if (input.dataDevice != DataDevice::CUDA || input.dims[0] < 1024) lock.lock();
        }
#if defined(USE_CUDA) && defined(USE_NUMAS)
        pipeline.emplace(workspace, mutex && numa ? &lock : nullptr);
#endif
    }
    void LockCpu() { if (lock.mutex() && !lock.owns_lock()) lock.lock(); }
    bool HoldsCpuPool() const { return lock.owns_lock(); }
private:
    std::unique_lock<std::mutex> lock;
#if defined(USE_CUDA) && defined(USE_NUMAS)
    std::optional<NumasMoePipelineScope> pipeline;
#endif
};

#ifdef USE_CUDA
class HybridPrefillScope {
public:
    HybridPrefillScope(const std::vector<int> &devices, bool numa)
        : devices(devices), workerCpus(devices.size()) {
#ifdef USE_NUMAS
        if (numa) workerCpus = GetNumasCudaWorkerCpuSets(devices);
#endif
    }
    void BindWorker(int rank) {
#ifdef USE_NUMAS
        BindNumasWorkerCpuSet(workerCpus.at(rank));
#endif
    }
    ~HybridPrefillScope() {
#ifndef USE_ROCM
        const int previous = FastllmCudaGetDevice();
        for (int device : devices) {
            FastllmCudaSetDevice(device);
            FastllmCudaClearMoeGGUFHostPrefetch();
        }
        FastllmCudaSetDevice(previous);
#endif
    }
    std::mutex hostExperts;
private:
    std::vector<int> devices;
    std::vector<std::vector<int>> workerCpus;
};
#endif
#if defined(USE_CUDA) && !defined(USE_ROCM)
// Two-device expert cooperation for layouts supported by the parallel backend.
// Stable buffers/events survive decode and verify shape changes. The result is
// reduced on the layer owner; a deferred peer result must be consumed before
// the next Run. Rejection leaves the model's ordinary host fallback available.
class HybridMoePeerRuntime {
public:
    explicit HybridMoePeerRuntime(const std::vector<CudaPrefillStage> &stages);
    ~HybridMoePeerRuntime();
    bool Readback(const Data &input, const Data &index, const Data &score);
    bool Run(int layer, const Data &input, const Data &index, const Data &score,
             Data &output, Data **weights, int count,
             const std::function<void()> &sharedExpert, Data **deferredPeer = nullptr,
             bool preparedReadback = false);
#if defined(USE_NUMAS) && defined(__linux__)
    struct CallerAffinity {
        cpu_set_t original;
        bool restore = false;
        explicit CallerAffinity(HybridMoePeerRuntime *state);
        ~CallerAffinity();
    };
#endif
private:
    std::vector<int> devices;
    std::vector<std::vector<int>> workerCpus;
    std::thread::id boundPeerThread;
    bool peerAccess = false;
    PersistentWorkerGroup workers;
    std::shared_ptr<FastllmCudaMoeExpertParallel> experts;
    Data peerInput[2], peerOutput[2], sum[2];
    void *ready[2] = {nullptr, nullptr}, *done[2] = {nullptr, nullptr};
};
#endif

}
