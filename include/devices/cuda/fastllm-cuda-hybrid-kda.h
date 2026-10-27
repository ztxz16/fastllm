#pragma once
#include "fastllm.h"

// Single-token KDA preparation for the hybrid backend. Unsupported layouts
// return false without changing either cache. The caller owns the mode guard.
bool FastllmCudaHybridKdaDecode(
    const fastllm::Data &q, const fastllm::Data &k, const fastllm::Data &v,
    const fastllm::Data &f, const fastllm::Data &g, const fastllm::Data &beta,
    const fastllm::Data &fWeight, const fastllm::Data &gWeight,
    const fastllm::Data &qConv, const fastllm::Data &kConv, const fastllm::Data &vConv,
    const fastllm::Data &aLog, const fastllm::Data &dtBias, const fastllm::Data &norm,
    fastllm::Data &convCache, fastllm::Data &state, int heads, int dimension,
    int kernelSize, float lowerBound, float eps, fastllm::Data &output);

// Three bias-free BF16 gate projections share one launch. The input is a
// single token; unsupported layouts leave all outputs untouched.
bool FastllmCudaHybridKdaInputProjections(const fastllm::Data &input,
    const fastllm::Data &fWeight, const fastllm::Data &betaWeight, const fastllm::Data &gWeight,
    fastllm::Data &f, fastllm::Data &beta, fastllm::Data &g);
