#pragma once
#include "fastllm.h"

// Match the full-vocabulary CUDA Top1's 256-lane reduction order on ties.
// Global IDs are essential: shards need not start at a multiple of 256.
#ifdef __CUDACC__
__host__ __device__
#endif
inline bool FastllmNaiveTop1Better(float score, int id, float best, int bestId) {
    if (score != best) return score > best;
#ifdef __CUDA_ARCH__
    unsigned rank = __brev((unsigned)id & 255u), bestRank = __brev((unsigned)bestId & 255u);
#else
    auto reverseLane = [](unsigned x) {
        x = ((x & 0x55u) << 1) | ((x >> 1) & 0x55u);
        x = ((x & 0x33u) << 2) | ((x >> 2) & 0x33u);
        return ((x & 0x0fu) << 4) | ((x >> 4) & 0x0fu);
    };
    unsigned rank = reverseLane((unsigned)id & 255u), bestRank = reverseLane((unsigned)bestId & 255u);
#endif
    return rank < bestRank || (rank == bestRank && id < bestId);
}

// These limits come from the selection workspace and FP32 token-ID format,
// independent of the GPU architecture, model vocabulary or TP rank count.
constexpr int kNaiveLogitsMaxTopK = 64;
constexpr int kNaiveLogitsMaxShard = 256 * 1024;
constexpr int kNaiveLogitsMaxVocab = 1 << 24;
inline bool FastllmNaiveCanSelectLogits(int vocab, int offset, int count, bool greedy) {
    return vocab > 0 && offset >= 0 && (int64_t)offset + vocab <= kNaiveLogitsMaxVocab &&
        (greedy ? count == 1 : count > 0 && count <= kNaiveLogitsMaxTopK &&
                               vocab <= kNaiveLogitsMaxShard);
}

// Contiguous FP32 vocabulary shards. Output [rows, count * 2] holds global
// ID/score pairs. Missing candidates use ID -1; greedy ignores NaN/-infinity.
// Greedy retains the legacy Top1 tie order. Sampling returns score-descending,
// ID-ascending candidates, with scaling applied before ranking.
void FastllmCudaNaiveLogitsSelect(const fastllm::Data &logits, int vocabOffset,
    int count, bool greedy, float invTemperature,
    fastllm::Data &partial, fastllm::Data &output);

// Model-specific CUDA operations use BF16 activations. Keys/values are packed
// as [1, tokens, heads * dim (+ indexDim for DSA keys)].
// raw is [1, rows, layers * 2 * heads * dim], in K0,V0,K1,V1 order.
// Returns false on unsupported layouts without changing cache contents/metadata.
bool FastllmCudaNaiveDraftKV(const fastllm::Data &raw, const fastllm::Data &norm,
    int start, std::vector<std::pair<fastllm::Data, fastllm::Data>> &kv,
    int heads, int dim, int window, int reserve, float eps, float theta);

void FastllmCudaNaiveRope(fastllm::Data &input, const fastllm::Data &positions,
                         int heads, int dim, int rotaryDim, float theta);
// Q/K RoPE and V scaling, preserving eager BF16 rounding. With packedQkv,
// read [Q, K, V] within each row and write contiguous Q/K/V outputs; otherwise
// operate in place. Q/K and V may have different head dimensions.
void FastllmCudaNaiveRopeQKScaleV(fastllm::Data &q, fastllm::Data &k,
    fastllm::Data &v, const fastllm::Data &positions,
    int heads, int kvHeads, int dim, int valueDim,
    int rotaryDim, float theta, float valueScale, const fastllm::Data *packedQkv = nullptr);
// Decode/verify: rotate Q in place and write rotated K/indexer K and scaled V
// directly to reserved caches. Inputs K/V remain unchanged; no metadata update.
// An empty indexKey denotes SWA. Unsupported layouts return false before writes.
// An empty liveKeys appends at the cache's host length, using existing capacity.
// packedQkv reads Q/K/V from a single projection; Q is written contiguously.
bool FastllmCudaNaiveRopeAppendCache(fastllm::Data &q, const fastllm::Data &k,
    const fastllm::Data &v, const fastllm::Data &indexKey,
    const fastllm::Data &positions, fastllm::Data &key, fastllm::Data &value,
    const fastllm::Data &liveKeys, int heads, int kvHeads, int dim, int valueDim,
    int rotaryDim, float theta, float valueScale, int window,
    const fastllm::Data *packedQkv = nullptr);
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

// Small speculative blocks use the same per-position arithmetic as decode.
// The temporary KV suffix is visible only through each row's causal prefix.
void FastllmCudaNaiveVerifyIndexer(const fastllm::Data &query,
    const fastllm::Data &weights, const fastllm::Data &packedKeys,
    int heads, int dim, int queryStart, int topK, bool fp8Query,
    fastllm::Data &indices);
void FastllmCudaNaiveVerifyAttention(const fastllm::Data &query,
    const fastllm::Data &key, const fastllm::Data &value,
    const fastllm::Data &indices, const fastllm::Data &sink,
    int heads, int kvHeads, int dim, int valueDim, int pastLength,
    int window, fastllm::Data &output);

// Whole-step decode graphs own these buffers until the executable is destroyed.
// liveKeys is an INT32 device scalar (past length + 1), updated before replay.
struct FastllmNaiveDecodeScratch {
    fastllm::Data indexQuery, indexScale, indexScores, topk;
    fastllm::Data attentionScores, attentionPartial, windowKey, windowValue;
};
bool FastllmCudaNaiveDecodeGraphSupported();
// BF16 rows; update each residual and preserve RMSNorm rounding.
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

// Fixed-shape verification graphs keep one live prefix length per query.
void FastllmCudaNaiveAppendVerifyCache(fastllm::Data &key, fastllm::Data &value,
    const fastllm::Data &newKey, const fastllm::Data &newValue,
    const fastllm::Data &liveKeys, int window);
void FastllmCudaNaiveGraphVerifyIndexer(const fastllm::Data &query,
    const fastllm::Data &weights, const fastllm::Data &packedKeys,
    const fastllm::Data &liveKeys, int capacity, bool fp8Query,
    FastllmNaiveDecodeScratch &scratch, fastllm::Data &indices);
void FastllmCudaNaiveGraphVerifyAttention(const fastllm::Data &query,
    const fastllm::Data &key, const fastllm::Data &value,
    const fastllm::Data &indices, const fastllm::Data &sink,
    const fastllm::Data &liveKeys, int capacity, int heads, int kvHeads,
    int dim, int valueDim, int window,
    FastllmNaiveDecodeScratch &scratch, fastllm::Data &output);

// Fixed draft inputs and attention read the live absolute prefix on the GPU.
void FastllmCudaNaiveDraftInput(const fastllm::Data &id, const fastllm::Data &embedding,
    const fastllm::Data &mask, const fastllm::Data &liveKeys, int rows,
    fastllm::Data &hidden, fastllm::Data &positions);
void FastllmCudaNaiveDraftAttention(const fastllm::Data &query,
    const fastllm::Data &key, const fastllm::Data &value,
    const fastllm::Data &liveKeys, int heads, int kvHeads, int dim, int window,
    bool shortAttention, fastllm::Data &scores, fastllm::Data &output);

// Greedy Markov proposals retain token ids on the GPU between steps. The
// argmax preserves the existing TopK(..., 1) tie order and BF16 addition.
void FastllmCudaNaiveDraftEmbedding(const fastllm::Data &ids, int step,
    const fastllm::Data &weight, fastllm::Data &latent);
void FastllmCudaNaiveDraftArgmax(const fastllm::Data &base,
    const fastllm::Data &bias, int step, fastllm::Data &partial,
    fastllm::Data &ids);
// Join leading rows of dense, equal-width BF16 features on the current GPU.
// Unsupported layouts return false before modifying output.
bool FastllmCudaNaiveDraftConcat(const std::vector<const fastllm::Data *> &inputs,
    int rows, fastllm::Data &output);

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
