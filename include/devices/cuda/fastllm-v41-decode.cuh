#pragma once

#include "fastllm.h"
#include <vector>

struct FastllmCudaV41DecodeTransform {
    fastllm::Data *input;
    fastllm::Data *window;
    int startPos;
    int quantMode;
    bool inverse;
    int quantBlock = 32;
};

// Single-token BF16 rows, with independent request positions and ring buffers.
// Rejection does not mutate inputs; callers retain the ordinary CUDA fallback.
extern "C" bool
FastllmCudaV41DecodeTransformsBatch(const std::vector<FastllmCudaV41DecodeTransform> &items,
                                    int ropeDim, float ropeBase, int originalSeqLen,
                                    float ropeFactor, int betaFast, int betaSlow, int windowSize);

struct FastllmCudaV41DecodeIndexTask {
    fastllm::Data *query, *weights, *keys, *candidates, *topk;
    int startPos;
    bool writeCandidates, useCandidates;
};

extern "C" bool
FastllmCudaV41DecodeIndexerBatch(const std::vector<FastllmCudaV41DecodeIndexTask> &items, int ratio,
                                 int topK, int blockSize, int candidateTopK, int ropeDim,
                                 float ropeBase, int originalSeqLen, float ropeFactor, int betaFast,
                                 int betaSlow, bool quantizeQuery);

// Finish the current mix while normalizing with the previous sublayer coefficients.
extern "C" bool FastllmCudaDeepSeekV41HcMixPreNorm(const fastllm::Data &x, const fastllm::Data &previousPre,
    fastllm::Data &fn, fastllm::Data &scale, fastllm::Data &base, fastllm::Data &normWeight,
    int sinkhornIters, float mixEps, float normEps, fastllm::Data &pre, fastllm::Data &post,
    fastllm::Data &comb, fastllm::Data &output);

// Full-K output-channel slice; no partial dot products or reduction across GPUs.
extern "C" bool FastllmCudaBFloat16FP8ReplicatedRowSlice(const fastllm::Data &input,
    fastllm::Data &weight, fastllm::Data &output, int rowStart, int rowCount);
