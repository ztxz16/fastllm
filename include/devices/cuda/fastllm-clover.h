#ifndef FASTLLM_CUDA_CLOVER_H
#define FASTLLM_CUDA_CLOVER_H
#include <cstddef>

// Typed entry points default to BF16; fp16=true selects FP16 storage.
// Both retain FP32 recurrent state and projection/reduction accumulation.

// Host-side pointer table copied into kernel arguments. Each request retains
// its own cache allocation and [T,...] snapshots; no state gather/scatter is
// needed. Null snapshot entries disable snapshots for that request.
struct CloverCudaGdnCaches {
    // 2 KiB of pointers leaves room for the other arguments under the 4 KiB
    // kernel-parameter limit of older CUDA architectures.
    static constexpr int Capacity = 64;
    void *conv[Capacity] = {};
    float *state[Capacity] = {};
    void *convSnapshots[Capacity] = {};
    float *stateSnapshots[Capacity] = {};
};

// Separate NHD allocations and committed lengths for each request. Preparation
// writes only the appended block; preview leaves the committed prefix intact.
struct CloverCudaAttentionCaches {
    static constexpr int Capacity = CloverCudaGdnCaches::Capacity;
    void *key[Capacity] = {}, *value[Capacity] = {};
    int position[Capacity] = {};
};

// Clover preview reads committed history without modifying it. Activations and
// convolution weights use the selected 16-bit type; recurrent states, gates and head scalars are FP32.
// The optional stream defaults to cudaStreamPerThread. Negative cache slots and
// false valid entries are inactive; null slots/valid mean contiguous/all active.
bool FastllmCudaCloverPreviewConv(
    const void *input, const void *history, const void *weight, const void *rightWeight,
    const int *slots, const unsigned char *valid, void *output,
    int batch, int tokens, int channels, void *stream = nullptr,
    const CloverCudaGdnCaches *caches = nullptr, bool fp16 = false);

// Q/K: [B,T,KH,128], V/output: [B,T,VH,128], state: [slot,VH,128,128]
// (value-major), g: [B,T,VH], scalars: [VH]. Block width is a divisor of 32.
// BF16 SM80+ blocks of 16/32 use BF16 normalized Q/K, state read operands and local
// probabilities with FP32 accumulation, matching SGLang preview precision.
// The persistent FP32 state remains read-only; other dispatches retain FP32.
// Optional qkvRowStride is in 16-bit elements, allowing Q/K/V views into a
// fused projection. Zero selects the contiguous stride of each input tensor.
bool FastllmCudaCloverPreviewGdn(
    const void *query, const void *key, const void *value, const float *g,
    const float *state, const float *logitScale, const float *historyBias,
    const float *historyScale, const int *slots, const unsigned char *valid,
    void *output, int batch, int tokens, int keyHeads, int valueHeads,
    void *stream = nullptr, int qkvRowStride = 0, const CloverCudaGdnCaches *caches = nullptr, bool fp16 = false);

// Causal replay commits 16-bit convolution history and FP32 recurrent state.
// Optional snapshots hold the state after each token, laid out [T,B,...], for
// accepting any prefix of a speculative block without recomputing the model.
bool FastllmCudaCloverCausalConv(const void *input, void *history, const void *weight,
    void *output, void *snapshots, int batch, int tokens, int channels,
    void *stream = nullptr, const CloverCudaGdnCaches *caches = nullptr, bool fp16 = false);
// Optional FP32 workspace has B*T*(2*KH*128+VH) elements. For T>=16 it
// reuses normalized Q/K and decay without changing the recurrent arithmetic.
// qkvRowStride has the same meaning as in preview; outputs remain contiguous.
bool FastllmCudaCloverCausalGdn(const void *query, const void *key, const void *value,
    const float *g, const void *beta, float *state, void *output, float *snapshots,
    int batch, int tokens, int keyHeads, int valueHeads, void *stream = nullptr,
    float *workspace = nullptr, int qkvRowStride = 0, const CloverCudaGdnCaches *caches = nullptr, bool fp16 = false);

// Block prefill uses BF16 Tensor Core products, FP32 accumulation/state and
// FP64 decay prefixes. It commits the entire input; calls requiring accepted-
// prefix snapshots must use CausalGdn. The query returns zero for unsupported
// geometry or loaded CUDA architectures. Workspace is caller-owned, aligned
// to 128 bytes, and must not overlap inputs, output or recurrent state.
// State/output require 8/4-byte alignment; other inputs use element alignment.
size_t FastllmCudaCloverPrefillGdnWorkspaceBytes(int batch, int tokens,
    int keyHeads, int valueHeads);
bool FastllmCudaCloverPrefillGdn(const void *query, const void *key, const void *value,
    const float *g, const void *beta, float *state, void *output,
    int batch, int tokens, int keyHeads, int valueHeads,
    void *workspace, size_t workspaceBytes, void *stream = nullptr, int qkvRowStride = 0);

// Unpack fused [QKV, Z, A, B] projections and compute FP32 decay / 16-bit beta.
// beta may be null for previews. QKV channels and Z [heads,128] stay 16-bit.
bool FastllmCudaCloverUnpackGdn(const void *input, const float *aLog, const float *bias,
    void *qkv, void *z, float *g, void *beta, int rows, int channels, int heads,
    void *stream = nullptr, bool fp16 = false);
// In-place partial rotary on [T,H,D], with separately rounded 16-bit products.
bool FastllmCudaCloverRotary(void *input, int tokens, int heads, int dim,
    int rotaryDim, int position, float theta, void *stream = nullptr, bool fp16 = false);
// Fixed 256-wide attention heads. Split [Q,gate] per head, zero-centered
// Q/K RMSNorm, then partial RoPE, preserving the separate operators' rounding.
// Q/gate outputs are [T,QH,256], key output is [T,KH,256]. Outputs must be distinct.
// A nonzero stride reads fused [Qgate,K,V] rows and also unpacks V into value.
bool FastllmCudaCloverPrepareAttention(const void *qgate, const void *key, const float *qWeight,
    const float *kWeight, void *query, void *normalizedKey, void *gate, int tokens,
    int queryHeads, int keyHeads, int rotaryDim, int position, float theta, float eps,
    void *stream = nullptr, int qkvStride = 0, void *value = nullptr, bool fp16 = false);
bool FastllmCudaCloverPrepareAttentionBatch(const void *qkv, const float *qWeight,
    const float *kWeight, void *query, void *gate, int batch, int tokens,
    int queryHeads, int keyHeads, int rotaryDim, float theta, float eps,
    const CloverCudaAttentionCaches &caches, void *stream = nullptr, bool fp16 = false);

// Stable expert-id tie breaking. Scratch: [B,T,K] INT32 ids and FP32 scores,
// plus [B,E] byte mask; only needed for preview coreset filtering.
// logitsStride defaults to E; a larger stride supports a fused gate projection.
bool FastllmCudaCloverRoute(const void *logits, int *indices, float *scores,
    int *scratchIds, float *scratchScores, unsigned char *allowed,
    int batch, int tokens, int experts, int topk, float coresetTopP,
    void *stream = nullptr, int logitsStride = 0, bool fp16 = false);
// Response-only penalties apply in FP32 before sampling. rowBias stores
// [stop-token bias, thinking-end bias] per row; counts never include the prompt.
struct CloverCudaPenalty {
    float repetition, presence, frequency;
    int maskToken, thinkToken, firstPosition, minTokens;
};
struct CloverCudaTokenStats {
    int token, error;
    float confidence, probability, entropy;
};
bool FastllmCudaCloverPenalize(float *logits, const int *counts, const int *stops,
    int numStops, const float *rowBias, int rows, int vocab,
    CloverCudaPenalty penalty, void *stream = nullptr);
// Argmax always uses the smallest token id for ties. Optional probabilities
// have temperature 1, independently of the subsequent bonus-token sampler.
// Optional device workspace spreads large vocabularies across multiple CTAs.
// Allocate the returned byte count, or pass null to use the single-CTA path.
std::size_t FastllmCudaCloverTokenStatsWorkspaceBytes(int rows, int vocab);
bool FastllmCudaCloverTokenStats(const float *logits, const int *candidates,
    CloverCudaTokenStats *output, int rows, int vocab, bool confidence,
    bool entropy, void *stream = nullptr, void *workspace = nullptr);

// BF16/FP16 NHD attention, fixed checkpoint head dimension 256. Optional split-KV
// scratch: ceil((cachedTokens+tokens)/256)*tokens*heads*(256*2+4) bytes.
bool FastllmCudaCloverAttention(const void *query, const void *key,
    const void *value, void *output, int tokens, int cachedTokens,
    int heads, int kvHeads, bool preview, void *stream = nullptr,
    void *workspace = nullptr, bool fp16 = false);
std::size_t FastllmCudaCloverAttentionBatchWorkspaceBytes(int batch, int tokens,
    int heads, int kvHeads, const CloverCudaAttentionCaches &caches);
bool FastllmCudaCloverAttentionBatch(const void *query, void *output, int batch,
    int tokens, int heads, int kvHeads, const CloverCudaAttentionCaches &caches,
    bool preview, void *workspace, void *stream = nullptr, bool fp16 = false);

#endif
