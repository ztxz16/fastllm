#pragma once
#include "qwen4_exp.h"
#include "utils/hybrid_moe.h"
#include "utils/hybrid_parallel.h"
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
