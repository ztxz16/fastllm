#pragma once
#include "devices/cuda/naive-n05-cuda.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Caller-owned buffers and live descriptors; no packed weights or global request state.
struct FastllmNaiveDSALayerInput {
    const fastllm::Data *branch = nullptr;
    fastllm::Data *hidden = nullptr, *feature = nullptr;
    const fastllm::Data *inputNorm = nullptr, *qkvWeight = nullptr, *qkvBias = nullptr;
    const fastllm::Data *indexKeyWeight = nullptr, *indexQueryWeight = nullptr, *indexWeight = nullptr;
    const fastllm::Data *indexNorm = nullptr, *indexBias = nullptr, *outputWeight = nullptr;
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

// Internal launch ABI shared by the SM103 object and the TP rendezvous adapter.
// No request pointers are retained by the adapter after it returns.
struct FastllmNaiveDSALayerArgs {
    using B = __nv_bfloat16;
    const B *branch, *qkvWeight, *indexKeyWeight, *indexQueryWeight, *indexWeight, *outputWeight;
    B *hidden, *feature, *normed, *query, *attention, *projected;
    const float *inputNorm, *indexNorm, *indexBias, *positions, *sink, *qkvScales, *outputScales;
    unsigned char *workspace, *coreWorkspace;
    FastllmNaiveDSARow bindings[16];
    int requests[16], requestCount, rows, rank;
    char *peers[8];
    uint64_t *epochs;
    float eps, theta, valueScale;
    bool fp8Key;
    // Internal publication-only diagnostic; production Prepare always clears it.
    // Retain this launch ABI/control flow: removing it changes SM103 code generation.
    int phaseStop;
};
struct FastllmNaiveDSALayerLayout {
    static constexpr size_t Qkv = 0, IndexKey = Qkv + 16 * 1856 * 2;
    static constexpr size_t IndexQuery = IndexKey + 16 * 128 * 2;
    static constexpr size_t IndexWeight = IndexQuery + 16 * 2048 * 2;
    static constexpr size_t Indices = IndexWeight + 16 * 16 * 2;
    static constexpr size_t Bytes = Indices + 16 * 2048 * 4;
};
bool FastllmCudaNaiveDSALayerPrepare(const FastllmNaiveDSALayerInput &input, int device,
                                     FastllmNaiveDSALayerArgs &args);
cudaError_t FastllmCudaNaiveDSALayerLaunch(FastllmNaiveDSALayerArgs args);
bool FastllmCudaNaiveDSALayerSupported();
struct FastllmMoePreparedInput;
bool FastllmCudaNaiveDSALayer(const FastllmNaiveDSALayerInput &input, int device,
                              const FastllmMoePreparedInput *prepared = nullptr);

// A collectively admitted FFN can own post-attention normalization.
bool FastllmCudaNaiveDSALayer(const FastllmNaiveDSALayerInput &input,int device,
    const FastllmMoePreparedInput *prepared,bool deferPostNorm);
