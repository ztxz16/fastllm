#pragma once
#include "devices/cuda/naive-n05-cuda.cuh"
#include <functional>

namespace fastllm {
namespace parallel { class CudaRuntime; }
struct NaiveAttentionCapabilities {
    bool fusedQkvFrontend, swaQkvFrontend, batchedPrefillIndexer, coldSwaKV, compactIndexer;
};
NaiveAttentionCapabilities FastllmCudaNaiveAttentionCapabilities();
bool FastllmCudaStablePrefillReduction(const std::vector<int> &architectures);
bool FastllmCudaStablePrefillReduction(parallel::CudaRuntime &runtime);

struct NaiveVerifyAttentionPlan {
    bool packed = false, swa = false, dsa = false, replicaLocalDsa = false;
    int replicaArch = 0;
    bool exportsRingFeature = false, packedIndexer = false;
    bool hopperRing = false, hopperReplicated = false, hopperContext = false;
    struct Layer {
        bool packed, swa, dsa, replica;
        explicit operator bool() const { return packed || swa || dsa || replica; }
    };
    Layer ForLayer(bool sliding, bool residual, bool localIndexer, int cacheRequests,
                   bool ringCache, bool contextIndex) const;
};
bool FastllmCudaVerifyReduceNormSupported(int rows, const std::vector<int> &devices);
NaiveVerifyAttentionPlan FastllmCudaPlanVerifyAttention(int device, int rows, int tpRanks,
        const std::vector<int> *requests, bool ringCache, bool integratedExperts, bool localDsa,
        int capacity, int indexHeads, int indexDim, int topK, int localAttentionRanks = 0);
using IndexerGather = std::function<void(const void *, void *, size_t)>;
void RunContextVerifyIndexer(const Data &query, const Data &weights,
        const std::vector<FastllmNaiveVerifyIndexerItem> &items,
        FastllmNaiveDecodeScratch &scratch, Data &live, Data &exchange,
        int rank, int ranks, int page, int gpu, bool fp8Query, const std::vector<int> &devices, const IndexerGather &gather = {});
void RunContextDecodeIndexer(const Data &query, const Data &weights, const Data &live,
        const std::vector<FastllmNaiveIndexerRow> &rows, Data &exchange,
        int rank, int ranks, int page, int gpu, bool fp8Query, const IndexerGather &gather = {});
}
