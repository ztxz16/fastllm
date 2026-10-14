#pragma once
#include "qwen4_exp.h"
#include "qwen4_exp_hybrid.h"
#include "utils/persistent_worker_group.h"
#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif
namespace fastllm {
    struct Qwen4ExpModel::ThreadTpState {
        std::vector<int> devices;
        HybridTpState hybrid;
        int loadingExpertLayer = -1;
        std::vector<std::string> loadingReplicas;
        std::vector<int> votes;
        int vocabSize = 0;
        std::vector<std::vector<std::pair<int, float>>> topCandidates;
        std::vector<std::pair<float, float>> topProbabilities;
        std::vector<const Data *> shardLogits;
        Data gatheredLogits;
        bool TrySampleLogits(int rank, Data &logits, const GenerationConfig &config,
                             int cacheLength, int &token, std::vector<int> *tokens = nullptr,
                             float *topProbability = nullptr);
        std::vector<std::unique_ptr<Qwen4ExpModel>> ranks;
        using Cache = std::vector<std::pair<Data, Data>>;
        struct RequestCache {
            std::vector<Cache> ranks;
            bool ownsRankZero = false;
            bool failed = false;
            bool mtpDisabled = false;
            bool startupReserved = false;
        };
        std::map<const Data *, std::unique_ptr<RequestCache>> caches;
        // Keep at most one completed request. Graphs and all buffers
        // whose addresses they capture share the same lifetime. Active
        // requests never share mutable storage with this idle slot.
        std::unique_ptr<RequestCache> idleCache;
        std::mutex forwardMutex, barrierMutex;
        std::condition_variable barrierCv;
        std::atomic<unsigned> generation{0};
        int arrived = 0;
        // Destroy workers before the rank models and synchronization state.
        PersistentWorkerGroup workers;

        void ReleaseCache(const Data *key, std::unique_ptr<RequestCache> &cache);

        ~ThreadTpState();

        void Barrier();
    };
} // namespace fastllm
