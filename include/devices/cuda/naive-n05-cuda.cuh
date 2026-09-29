#pragma once
#include "fastllm.h"

// Model-specific CUDA operations use BF16 activations. Keys/values are packed
// as [1, tokens, heads * dim (+ indexDim for DSA keys)].
void FastllmCudaNaiveRope(fastllm::Data &input, const fastllm::Data &positions,
                         int heads, int dim, int rotaryDim, float theta);
void FastllmCudaNaiveIndexer(const fastllm::Data &query,
                            const fastllm::Data &weights,
                            const fastllm::Data &packedKeys,
                            int heads, int dim, int queryStart, int topK,
                            bool fp8, fastllm::Data &indices);
void FastllmCudaNaiveAttention(const fastllm::Data &query,
                              const fastllm::Data &key,
                              const fastllm::Data &value,
                              const fastllm::Data &indices,
                              const fastllm::Data &sink,
                              int heads, int kvHeads, int dim, int valueDim,
                              int pastLength, int window,
                              fastllm::Data &output, bool causal = true);

// NUMA FP8 weights are row-packed [128 E4M3 bytes, FP32 scale]. Gate/up
// output rows are interleaved. Route ids index the original [token, top-k].
struct FastllmNaiveFP8ExpertTask {
    const uint8_t *gateWeight, *downWeight;
    std::vector<int> routes;
};
bool FastllmCudaNaiveExpertPrefill(int device, const uint16_t *input,
    const float *scores, int tokens, int topk, int hidden, int intermediate,
    const std::vector<FastllmNaiveFP8ExpertTask> &tasks,
    const float *siluLookup, float *perRouteOutput);
void FastllmCudaNaiveClearExpertPrefill();
