// Cross-device host expert cooperation; independent of layer prefill pipelining.
#include "qwen4_exp_execution.h"
#include "qwen4_exp_hybrid.h"
#include "qwen4_exp_tp.h"
#include "models/qwen4_paged_cache.h"
#include "executor.h"
#include "utils.h"
#ifdef USE_CUDA
#include "fastllm-multicuda.cuh"
#ifndef USE_ROCM
#include "devices/cuda/fastllm-cuda-moe-policy.h"
#include "devices/cuda/fastllm-moe-prefetch.h"
#endif
#endif
#ifdef USE_NUMAS
#include "devices/numas/numasdevice.h"
#endif
namespace fastllm {
    Qwen4ExpModel::HybridPrefillScope::HybridPrefillScope(Qwen4ExpModel &model,
                                                          const std::vector<int> &devices)
        : devices(devices) {
        workerCpus = model.HybridWorkerCpuSets(devices);
    }
    void Qwen4ExpModel::HybridPrefillScope::BindWorker(int rank) {
#ifdef USE_NUMAS
        // Rebind at the first chunk: the pipeline's workers may be reused.
        BindNumasWorkerCpuSet(workerCpus[rank]);
#endif
    }
    Qwen4ExpModel::HybridPrefillScope::~HybridPrefillScope() {
#if defined(USE_CUDA) && !defined(USE_ROCM)
        Qwen4CudaDeviceGuard guard;
        for (int device : devices) {
            FastllmCudaSetDevice(device);
            FastllmCudaClearMoeGGUFHostPrefetch();
        }
#endif
    }
    void Qwen4ExpModel::HybridTpState::Init(Qwen4ExpModel &model, const std::vector<int> &devices) {
        hostMoeLayers.assign(model.block_cnt, false);
        for (int layer = 0; layer < model.block_cnt; ++layer) {
            const auto device = model.SelectMoeDeviceForLayer(layer);
            const bool host = device == "cpu" || device == "numa" || device.rfind("numa:", 0) == 0;
            AssertInFastLLM(host || device.empty() || device == "cuda" ||
                                device.rfind("cuda:", 0) == 0,
                            "Qwen4 TP requires CUDA or CPU/NUMA experts in every layer.");
            hostMoeLayers[layer] = host;
        }
        hostMoe =
            std::any_of(hostMoeLayers.begin(), hostMoeLayers.end(), [](bool host) { return host; });
#if defined(USE_CUDA) && !defined(USE_ROCM)
        if (hostMoe && model.MoeCudaCacheRequested())
            expertParallel = FastllmCudaCreateMoeExpertParallel(devices.size());
#endif
    }
    std::vector<std::vector<int>>
    Qwen4ExpModel::HybridWorkerCpuSets(const std::vector<int> &devices) const {
        std::vector<std::vector<int>> workerCpus(devices.size());
#ifdef USE_NUMAS
        for (int layer = 0; layer < block_cnt; ++layer) {
            const auto device = SelectMoeDeviceForLayer(layer);
            if (device == "numa" || device.rfind("numa:", 0) == 0) {
                workerCpus = GetNumasCudaWorkerCpuSets(devices);
                break;
            }
        }
#endif
        return workerCpus;
    }
    void Qwen4ExpModel::HybridTpState::BindWorker(int rank) {
#ifdef USE_NUMAS
        // The persistent worker belongs to this TP group for its lifetime.
        static thread_local bool placementReady = false;
        if (!placementReady) {
            BindNumasWorkerCpuSet(workerCpus[rank]);
            placementReady = true;
        }
#endif
    }

#if defined(USE_CUDA) && !defined(USE_ROCM)

#if defined(USE_NUMAS) && defined(__linux__)
    struct Qwen4ExpModel::HybridPeerState::CallerAffinity {
        cpu_set_t original;
        bool restore = false;
        explicit CallerAffinity(HybridPeerState *state) {
            // The persistent peer must already exist before narrowing the
            // caller's mask, otherwise it would inherit the caller's cores.
            if (state && state->workerCpus.size() == 2 && state->workers.HasWorkers() &&
                sched_getaffinity(0, sizeof(original), &original) == 0) {
                restore = BindNumasWorkerCpuSet(state->workerCpus[0]);
            }
        }
        ~CallerAffinity() {
            if (restore)
                sched_setaffinity(0, sizeof(original), &original);
        }
    };
#endif
    Qwen4ExpModel::HybridPeerState::HybridPeerState(const std::vector<CudaPrefillStage> &stages) {
        if (stages.size() != 2)
            return;
        Qwen4CudaDeviceGuard guard;
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
    Qwen4ExpModel::HybridPeerState::~HybridPeerState() {
        workers.Stop();
        Qwen4CudaDeviceGuard guard;
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

    bool Qwen4ExpModel::TryPipelineMoeReadback(const Data &input, const Data &index,
                                               const Data &score) {
#ifdef USE_NUMAS
        if (threadTpRank >= 0) {
            return threadTpRank == 0 && threadTpOwner && threadTpOwner->hybrid.expertParallel &&
                   FastllmCudaMoeExpertParallelReadback(*threadTpOwner->hybrid.expertParallel,
                                                        threadTpRank, input, index, score);
        }
        if (threadTpState || !hybridPeer)
            return false;
        const auto &devices = hybridPeer->devices;
        const auto found = std::find(devices.begin(), devices.end(), FastllmCudaGetDevice());
        return found != devices.end() &&
               FastllmCudaMoeExpertParallelReadback(*hybridPeer->experts, found - devices.begin(),
                                                    input, index, score);
#else
        return false;
#endif
    }

    bool Qwen4ExpModel::TryRunPipelineMoe(int layer, const Data &input, const Data &index,
                                          const Data &score, Data &output,
                                          std::vector<Data *> &moeWeights,
                                          const std::function<void()> &sharedExpert,
                                          Data **deferredPeer, bool preparedReadback) {
#ifdef USE_NUMAS
        if (threadTpState || threadTpRank >= 0 || input.dataDevice != CUDA ||
            input.dims.size() != 2 || input.dims[0] < 1 ||
            input.dims[0] > FASTLLM_CUDA_MOE_CACHE_MAX_BATCH || !MoeCudaCacheRequested() ||
            FastllmCudaGraphIsCapturingFast())
            return false;
        if (!hybridPeer) {
            std::vector<CudaPrefillStage> stages;
            if (!BuildCudaPrefillStages(deviceMap, block_cnt, stages))
                stages.clear();
            hybridPeer.reset(new HybridPeerState(stages));
        }
        auto &state = *hybridPeer;
        if (state.devices.size() != 2)
            return false;
        const int device = FastllmCudaGetDevice();
        const auto found = std::find(state.devices.begin(), state.devices.end(), device);
        if (found == state.devices.end())
            return false;
        const int owner = found - state.devices.begin(), peer = 1 - owner;
        Qwen4CudaDeviceGuard guard;
        auto prepare = [&](Data &data, int target, DataType type) {
            FastllmCudaSetDevice(target);
            data.dataType = type;
            data.Resize({FASTLLM_CUDA_MOE_CACHE_MAX_BATCH, input.dims[1]});
            data.ToDevice(CUDA, std::vector<int>{target}, false);
            data.Allocate(false);
            data.Resize(input.dims);
        };
        FastllmCudaEventRecordCurrentThread(state.ready[owner]);
        const bool borrowInput =
            state.peerAccess && input.dataDeviceIds == std::vector<int>{device};
        bool accepted[2] = {false, false};
        std::vector<std::exception_ptr> errors(2);
        // Keep physical rank/expert ownership stable as the PP layer owner moves.
        // Only the narrow normalized activation and the reduced MoE result travel.
        state.workers.RunWithCaller(
            {0, 1},
            [&](int worker) {
                const auto thread = std::this_thread::get_id();
                if (worker != 0 && state.boundPeerThread != thread) {
                    BindNumasWorkerCpuSet(state.workerCpus[worker]);
                    state.boundPeerThread = thread;
                }
                const int rank = worker == 0 ? owner : peer;
                const int target = state.devices[rank];
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
                    // Dense graph variants retain this peer output across changing
                    // verifier lengths. Its allocation must not grow on a later row.
                    prepare(state.peerOutput[peer], target, FLOAT32);
                    // Allocate on the thread that owns the peer stream, avoiding a
                    // switch to the peer device and back on every owner submission.
                    if (!borrowInput)
                        prepare(state.peerInput[peer], target, input.dataType);
                    static thread_local std::unique_ptr<Executor> executor(new Executor());
                    executor->SetFirstDevice("cuda:" + std::to_string(target));
                    SetCurrentThreadExecutor(executor.get());
                    FastllmCudaCurrentThreadStreamWaitEvent(state.ready[owner]);
                    if (!borrowInput && !FastllmCudaMemcpyPeerAsyncCurrentThread(
                                            target, state.peerInput[peer].cudaData, device,
                                            input.cudaData, input.GetBytes())) {
                        FastllmCudaEventSynchronize(state.ready[owner]);
                        FastllmCudaMemcpyBetweenDevices(target, state.peerInput[peer].cudaData,
                                                        device, input.cudaData, input.GetBytes());
                    }
                }
                accepted[rank] = FastllmCudaMergeMOEExpertParallel(
                    *state.experts, rank,
                    worker == 0 || borrowInput ? input : state.peerInput[peer], index, score,
                    worker == 0 ? output : state.peerOutput[peer], moeWeights.data(),
                    int(moeWeights.size()), layer,
                    worker == 0 ? sharedExpert : std::function<void()>(), owner,
                    preparedReadback && worker == 0);
                if (worker != 0)
                    FastllmCudaEventRecordCurrentThread(state.done[rank]);
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
        FastllmCudaCurrentThreadStreamWaitEvent(state.done[peer]);
        if (state.peerAccess && output.dataType == FLOAT32 &&
            state.peerOutput[peer].dataType == FLOAT32 &&
            output.dims == state.peerOutput[peer].dims) {
            // Both allocations are peer-accessible. The event protects the remote
            // read; the next input-ready event protects it from peer reuse.
            if (deferredPeer) {
                // The per-rank output stays at a stable address. Consume it in
                // the next dense graph, after this stream's peer-completion wait.
                *deferredPeer = &state.peerOutput[peer];
                return true;
            }
            return FastllmCudaAddTo(output, state.peerOutput[peer], 1.0f);
        }
        prepare(state.sum[owner], device, FLOAT32);
        if (!FastllmCudaMemcpyPeerAsyncCurrentThread(
                device, state.sum[owner].cudaData, state.devices[peer],
                state.peerOutput[peer].cudaData, state.sum[owner].GetBytes())) {
            FastllmCudaEventSynchronize(state.done[peer]);
            FastllmCudaMemcpyBetweenDevices(device, state.sum[owner].cudaData, state.devices[peer],
                                            state.peerOutput[peer].cudaData,
                                            state.sum[owner].GetBytes());
        }
        AddTo(output, state.sum[owner]);
        return true;
#else
        return false;
#endif
    }
#endif

    Qwen4ExpModel::HybridExecutionScope::HybridExecutionScope(Qwen4ExpModel &model,
                                                              const Data &inputIds,
                                                              bool pipelinedPrefill,
                                                              bool mtpIteration) {
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM) && defined(__linux__)
        const bool bind =
            mtpIteration
                ? model.threadTpRank < 0 && inputIds.dims[1] == 1 && Qwen4MtpDraftsPerStep() > 0
                : !pipelinedPrefill && !model.threadTpState && model.threadTpRank < 0 &&
                      inputIds.dims[1] == 1 && Qwen4MtpDraftsPerStep() == 0;
        affinity.reset(
            new HybridPeerState::CallerAffinity(bind ? model.hybridPeer.get() : nullptr));
#endif
        if (!mtpIteration && model.executionMode == ExecutionMode::Hybrid &&
            model.threadTpRank < 0 && Qwen4MtpDraftsPerStep() == 0 && inputIds.dims[1] == 1 &&
            Qwen4CudaOnlyDeviceMap(model.deviceMap) && !model.weights.empty() &&
            !model.weights[0].empty()) {
            decodeStep.Begin(model.weights[0].data(), model.weights[0].size(),
                             model.num_experts_per_tok);
        }
    }
    Qwen4ExpModel::HybridExecutionScope::~HybridExecutionScope() = default;

} // namespace fastllm
