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
