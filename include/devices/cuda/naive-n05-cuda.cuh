#pragma once
#include "fastllm.h"

// Model-specific CUDA operations use BF16 activations. Keys/values are packed
// as [1, tokens, heads * dim (+ indexDim for DSA keys)].
void FastllmCudaNaiveRope(fastllm::Data &input, const fastllm::Data &positions,
                         int heads, int dim, int rotaryDim, float theta);
// In-place Q/K RoPE and V scaling, preserving eager BF16 rounding.
void FastllmCudaNaiveRopeQKScaleV(fastllm::Data &q, fastllm::Data &k,
    fastllm::Data &v, const fastllm::Data &positions,
    int heads, int kvHeads, int dim, int valueDim,
    int rotaryDim, float theta, float valueScale);
// Keep the allocation and logical row order when retaining a sliding suffix.
void FastllmCudaNaiveTrimCache(fastllm::Data &key, fastllm::Data &value, int keep);
// Exact descending score / ascending position order for each query row.
// Row r considers only keys [0, queryStart + r]; missing slots are -1.
void FastllmCudaNaiveTopK(const fastllm::Data &scores, int queryStart, int topK,
                         fastllm::Data &indices);
// Reuse the row quantizer for Indexer Q/K. Values are exact E4M3 numbers in
// BF16 storage, with a separate FP32 scale. roundScale selects UE8M0 scales.
bool FastllmCudaNaiveQuantizeIndexer(const fastllm::Data &input,
    fastllm::Data &values, fastllm::Data &scales, bool roundScale);
// Only Q is optionally E4M3-rounded; Indexer K always retains its BF16 input.
void FastllmCudaNaiveIndexer(const fastllm::Data &query,
                            const fastllm::Data &weights,
                            const fastllm::Data &packedKeys,
                            int heads, int dim, int queryStart, int topK,
                            bool fp8Query, fastllm::Data &indices);
void FastllmCudaNaiveAttention(const fastllm::Data &query,
                              const fastllm::Data &key,
                              const fastllm::Data &value,
                              const fastllm::Data &indices,
                              const fastllm::Data &sink,
                              int heads, int kvHeads, int dim, int valueDim,
                              int pastLength, int window,
                              fastllm::Data &output, bool causal = true);

// Whole-step decode graphs own these buffers until the executable is destroyed.
// liveKeys is an INT32 device scalar (past length + 1), updated before replay.
struct FastllmNaiveDecodeScratch {
    fastllm::Data indexQuery, indexScale, indexScores, topk;
    fastllm::Data attentionScores, attentionPartial;
};
bool FastllmCudaNaiveDecodeGraphSupported();
// One BF16 decode row; update the residual and preserve RMSNorm rounding.
void FastllmCudaNaiveAddDecodeRMSNorm(fastllm::Data &hidden,
    const fastllm::Data &branch, const fastllm::Data &weight,
    float eps, fastllm::Data &output);

void FastllmCudaNaiveAppendDecodeCache(fastllm::Data &key, fastllm::Data &value,
    const fastllm::Data &newKey, const fastllm::Data &newValue,
    const fastllm::Data &liveKeys, int window);
void FastllmCudaNaiveTrimDecodeCache(fastllm::Data &key, fastllm::Data &value,
    const fastllm::Data &liveKeys, int window);
void FastllmCudaNaiveDecodeIndexer(const fastllm::Data &query,
    const fastllm::Data &weights, const fastllm::Data &packedKeys,
    const fastllm::Data &liveKeys, int capacity, bool fp8Query,
    FastllmNaiveDecodeScratch &scratch, fastllm::Data &indices);
void FastllmCudaNaiveDecodeAttention(const fastllm::Data &query,
    const fastllm::Data &key, const fastllm::Data &value,
    const fastllm::Data &indices, const fastllm::Data &sink,
    const fastllm::Data &liveKeys, int capacity, int heads, int kvHeads,
    int dim, int valueDim, int window,
    FastllmNaiveDecodeScratch &scratch, fastllm::Data &output);

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
