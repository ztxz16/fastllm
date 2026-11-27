#pragma once
#include "devices/cuda/naive-n05-cuda.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "devices/multicuda/fastllm-naive-verify.h"

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
    bool fp8Key, mxfp8Gemv;
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
bool FastllmCudaNaiveDSALayerPrepare(const FastllmNaiveDSALayerInput &input, int device,
                                     FastllmNaiveDSALayerArgs &args, std::vector<FastllmNaiveDSALayerArgs> *tiles);
cudaError_t FastllmCudaNaiveDSALayerLaunch(FastllmNaiveDSALayerArgs args);
bool FastllmCudaNaiveDSALayerSupported();
struct FastllmMoePreparedInput;
bool FastllmCudaNaiveDSALayer(const FastllmNaiveDSALayerInput &input, int device,
                              const FastllmMoePreparedInput *prepared = nullptr);

// A collectively admitted FFN can own post-attention normalization.
bool FastllmCudaNaiveDSALayer(const FastllmNaiveDSALayerInput &input,int device,
    const FastllmMoePreparedInput *prepared,bool deferPostNorm);
