// Cross-device host expert cooperation; independent of layer prefill pipelining.
#include "qwen4_exp_execution.h"
#include "qwen4_exp_hybrid.h"
#include "qwen4_exp_tp.h"
#include "models/qwen4_paged_cache.h"
#include "utils.h"
#ifdef USE_NUMAS
#include "devices/numas/numasdevice.h"
#endif
namespace fastllm {
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

    bool Qwen4ExpModel::TryPipelineMoeReadback(const Data &input, const Data &index,
                                               const Data &score) {
#ifdef USE_NUMAS
        if (hybridOwnedPp) return false;
        if (threadTpRank >= 0) {
            return threadTpRank == 0 && threadTpOwner && threadTpOwner->hybrid.expertParallel &&
                   FastllmCudaMoeExpertParallelReadback(*threadTpOwner->hybrid.expertParallel,
                                                        threadTpRank, input, index, score);
        }
        if (threadTpState || !hybridPeer)
            return false;
        return hybridPeer->Readback(input, index, score);
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
        if (hybridOwnedPp) return false;
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
        return hybridPeer->Run(layer, input, index, score, output,
            moeWeights.data(), int(moeWeights.size()), sharedExpert, deferredPeer, preparedReadback);
#else
        return false;
#endif
    }
#endif

    Qwen4ExpModel::HybridExecutionScope::HybridExecutionScope(Qwen4ExpModel &model,
                                                              const Data &inputIds,
                                                              bool pipelinedPrefill,
                                                              bool mtpIteration, bool targetVerifier) {
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM) && defined(__linux__)
        const bool bind =
            mtpIteration
                ? model.threadTpRank < 0 && inputIds.dims[1] == 1 && Qwen4MtpDraftsPerStep() > 0
                : !pipelinedPrefill && !model.threadTpState && model.threadTpRank < 0 &&
                      inputIds.dims[1] == 1 && Qwen4MtpDraftsPerStep() == 0;
        affinity.reset(
            new HybridPeerState::CallerAffinity(bind ? model.hybridPeer.get() : nullptr));
#endif
        const char *fullGraph = std::getenv("FASTLLM_QWEN38_HYBRID_GRAPH");
        const bool verifierPolicy = targetVerifier && fullGraph && std::string(fullGraph) == "1";
        // One scope spans the complete target forward, including all PLE and
        // device ranges. Draft proposals and small prefill chunks are not
        // verifier steps and must not start an independent admission cycle.
        if (!mtpIteration && model.executionMode == ExecutionMode::Hybrid &&
            model.threadTpRank < 0 && (inputIds.dims[1] == 1 || verifierPolicy) &&
            Qwen4CudaOnlyDeviceMap(model.deviceMap) && !model.weights.empty()) {
#if defined(USE_CUDA) && !defined(USE_ROCM)
            if (model.hybridOwnedPp) {
                // Each immutable PP owner has one complete-forward policy.
                // No peer reads these caches, so their admission lifetimes
                // can be managed by the common per-device scope.
                std::vector<CudaPrefillStage> stages;
                if (BuildCudaPrefillStages(model.deviceMap, model.block_cnt, stages)) {
                    const int previous = FastllmCudaGetDevice();
                    for (const auto &stage : stages) {
                        FastllmCudaSetDevice(stage.device);
                        for (int layer = stage.firstLayer; layer < stage.endLayer; ++layer)
                            if (deviceSteps.Begin(stage.device, model.weights[layer].data(),
                                    model.weights[layer].size(), model.num_experts_per_tok)) break;
                    }
                    FastllmCudaSetDevice(previous);
                    return;
                }
            }
#endif
            // A resident prefix is not a host-cache key. Find the first target
            // layer that owns the shared cache so later NUMA layers still use
            // complete-forward observation and admission.
            for (int layer = 0; layer < model.block_cnt && layer < int(model.weights.size()); ++layer) {
                auto &weights = model.weights[layer];
                if (weights.empty()) continue;
                decodeStep.Begin(weights.data(), weights.size(), model.num_experts_per_tok);
                if (decodeStep) break;
            }
        }
    }
    Qwen4ExpModel::HybridExecutionScope::~HybridExecutionScope() = default;

} // namespace fastllm
