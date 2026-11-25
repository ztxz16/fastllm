#pragma once
#include "devices/cuda/naive-n05-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"

// Caller-owned buffers and live descriptors; no packed weights or global request state.
struct FastllmNaiveDSALayerInput {
    const fastllm::Data *branch = nullptr;
    fastllm::Data *hidden = nullptr, *feature = nullptr;
    const fastllm::Data *inputNorm = nullptr, *qkvBias = nullptr;
    fastllm::Data *qkvWeight = nullptr, *outputWeight = nullptr;
    const fastllm::Data *indexKeyWeight = nullptr, *indexQueryWeight = nullptr, *indexWeight = nullptr;
    const fastllm::Data *indexNorm = nullptr, *indexBias = nullptr;
    const fastllm::Data *postNorm = nullptr, *sink = nullptr, *positions = nullptr, *live = nullptr;
    const fastllm::Data *routerWeight = nullptr;
    fastllm::Data *query = nullptr, *attention = nullptr, *projected = nullptr, *normed = nullptr;
    fastllm::Data *router = nullptr;
    FastllmNaiveDecodeScratch *scratch = nullptr;
    std::vector<FastllmNaiveRopeCacheRow> caches;
    std::vector<int> requests;
    int capacity = 0;
    bool fp8Query = true;
    float eps = 1e-5f, theta = 10000.f, valueScale = 1.f;
};

enum class FastllmVerifyOutput { Unsupported, Projected, Normalized, Router, DeferredNorm };

struct FastllmNaiveVerifyAttentionInput : FastllmNaiveDSALayerInput {
    FastllmSwaLayerConfig config{};
    FastllmMxfp8Activation qkvActivation{};
    fastllm::Data *indices = nullptr;
    int attentionRank = 0, topK = 0, indexHeads = 0, indexDim = 0;
    fastllm::Data *qkv = nullptr;
    const fastllm::Data *projectionScratch = nullptr;
    bool replica = false, fp8IndexerCache = false;
    bool sliding = false, packedTp = false, afterExpertReduce = false, deferPostNorm = false;
};

// A decline leaves residual/KV untouched. A successful backend reports the
// state handed to the FFN; the model owns the lifetime of all referenced data.
FastllmVerifyOutput FastllmCudaVerifyAttention(const FastllmNaiveVerifyAttentionInput &input,
        int device, const FastllmMoePreparedInput *prepared);
bool FastllmCudaPackedVerifySupported(int rows, int ranks);

bool FastllmCudaLocalDsaLayerSupported(const std::vector<int> &requests, int rows, int capacity,
                                      int heads, int dim, int topK);
bool FastllmCudaLocalDsaIndexerSupported(const std::vector<FastllmNaiveVerifyIndexerItem> &items,
                                        int heads, int kvHeads, int dim, int valueDim);
bool FastllmCudaVerifyLocalDsaIndexer(const fastllm::Data &query, const fastllm::Data &indexQuery,
                                    const fastllm::Data &weights, const fastllm::Data &sink,
                                    const std::vector<FastllmNaiveVerifyIndexerItem> &items,
                                    const std::vector<FastllmNaiveDSARow> &rows, bool fp8Query, int heads,
                                    int kvHeads, int dim, int valueDim, FastllmNaiveDecodeScratch &scratch,
                                    fastllm::Data &output);
