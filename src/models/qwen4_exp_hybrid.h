#pragma once
#include "qwen4_exp.h"
#include "utils/hybrid_moe.h"
#include "utils/persistent_worker_group.h"
#include "utils/cuda_chunked_prefill.h"
#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif
namespace fastllm {
    struct Qwen4ExpModel::HybridTpState {
        bool hostMoe = false;
        std::vector<bool> hostMoeLayers;
        std::vector<std::vector<int>> workerCpus;
#if defined(USE_CUDA) && !defined(USE_ROCM)
        std::shared_ptr<FastllmCudaMoeExpertParallel> expertParallel;
#endif
        void Init(Qwen4ExpModel &model, const std::vector<int> &devices);
        void BindWorker(int rank);
    };

#if defined(USE_CUDA) && !defined(USE_ROCM)
    struct Qwen4ExpModel::HybridPeerState {
        std::vector<int> devices;
        std::vector<std::vector<int>> workerCpus;
        std::thread::id boundPeerThread;
        bool peerAccess = false;
        PersistentWorkerGroup workers;
        std::shared_ptr<FastllmCudaMoeExpertParallel> experts;
        Data peerInput[2], peerOutput[2], sum[2];
        void *ready[2] = {nullptr, nullptr}, *done[2] = {nullptr, nullptr};

#if defined(USE_NUMAS) && defined(__linux__)
        struct CallerAffinity;
#endif

        explicit HybridPeerState(const std::vector<CudaPrefillStage> &stages);
        ~HybridPeerState();
    };
#endif
    struct Qwen4ExpModel::HybridPrefillScope {
        const std::vector<int> &devices;
        std::vector<std::vector<int>> workerCpus;
        HybridPrefillScope(Qwen4ExpModel &model, const std::vector<int> &devices);
        void BindWorker(int rank);
        ~HybridPrefillScope();
    };
    struct Qwen4ExpModel::HybridExecutionScope {
        HybridMoeDecodeStep decodeStep;
#if defined(USE_CUDA) && defined(USE_NUMAS) && !defined(USE_ROCM) && defined(__linux__)
        std::unique_ptr<HybridPeerState::CallerAffinity> affinity;
#endif
        HybridExecutionScope(Qwen4ExpModel &model, const Data &inputIds, bool pipelinedPrefill,
                             bool mtpIteration = false);
        ~HybridExecutionScope();
    };
} // namespace fastllm
