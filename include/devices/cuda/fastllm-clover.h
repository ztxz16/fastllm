#ifndef FASTLLM_CUDA_CLOVER_H
#define FASTLLM_CUDA_CLOVER_H
#include <cstddef>

// Clover preview reads committed history without modifying it. Activations and
// convolution weights are BF16; recurrent states, gates and head scalars are FP32.
// The optional stream defaults to cudaStreamPerThread. Negative cache slots and
// false valid entries are inactive; null slots/valid mean contiguous/all active.
bool FastllmCudaCloverPreviewConv(
    const void *input, const void *history, const void *weight, const void *rightWeight,
    const int *slots, const unsigned char *valid, void *output,
    int batch, int tokens, int channels, void *stream = nullptr);

// Q/K: [B,T,KH,128], V/output: [B,T,VH,128], state: [slot,VH,128,128]
// (value-major), g: [B,T,VH], scalars: [VH]. Block width is a divisor of 32.
// Optional qkvRowStride is in BF16 elements, allowing Q/K/V views into a
// fused projection. Zero selects the contiguous stride of each input tensor.
bool FastllmCudaCloverPreviewGdn(
    const void *query, const void *key, const void *value, const float *g,
    const float *state, const float *logitScale, const float *historyBias,
    const float *historyScale, const int *slots, const unsigned char *valid,
    void *output, int batch, int tokens, int keyHeads, int valueHeads,
    void *stream = nullptr, int qkvRowStride = 0);

// Causal replay commits BF16 convolution history and FP32 recurrent state.
// Optional snapshots hold the state after each token, laid out [T,B,...], for
// accepting any prefix of a speculative block without recomputing the model.
bool FastllmCudaCloverCausalConv(const void *input, void *history, const void *weight,
    void *output, void *snapshots, int batch, int tokens, int channels,
    void *stream = nullptr);
// Optional FP32 workspace has B*T*(2*KH*128+VH) elements. For T>=16 it
// reuses normalized Q/K and decay without changing the recurrent arithmetic.
// qkvRowStride has the same meaning as in preview; outputs remain contiguous.
bool FastllmCudaCloverCausalGdn(const void *query, const void *key, const void *value,
    const float *g, const void *beta, float *state, void *output, float *snapshots,
    int batch, int tokens, int keyHeads, int valueHeads, void *stream = nullptr,
    float *workspace = nullptr, int qkvRowStride = 0);

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

// BF16 input/output, FP32 norm weights. Optional BF16 gate is SiLU-gated
// before the single output cast, as required by the GDN output norm.
bool FastllmCudaCloverNorm(const void *input, const float *weight, const void *gate,
    void *output, int rows, int width, float eps, bool zeroCentered,
    void *stream = nullptr);
// Update BF16 residual += input before normalization, including the BF16
// residual rounding. Output must not overlap either input allocation.
bool FastllmCudaCloverAddNorm(void *residual, const void *input, const float *weight,
    void *output, int rows, int width, float eps, bool zeroCentered,
    void *stream = nullptr);
// Unpack fused [QKV, Z, A, B] projections and compute FP32 decay / BF16 beta.
// beta may be null for previews. QKV channels and Z [heads,128] stay BF16.
bool FastllmCudaCloverUnpackGdn(const void *input, const float *aLog, const float *bias,
    void *qkv, void *z, float *g, void *beta, int rows, int channels, int heads,
    void *stream = nullptr);
// In-place partial rotary on [T,H,D], with the reference's BF16 products.
bool FastllmCudaCloverRotary(void *input, int tokens, int heads, int dim,
    int rotaryDim, int position, float theta, void *stream = nullptr);
// Fixed 256-wide attention heads. Split [Q,gate] per head, zero-centered
// Q/K RMSNorm, then partial RoPE, preserving the separate operators' rounding.
// Q/gate outputs are [T,QH,256], key output is [T,KH,256]. Outputs must be distinct.
bool FastllmCudaCloverPrepareAttention(const void *qgate, const void *key, const float *qWeight,
    const float *kWeight, void *query, void *normalizedKey, void *gate, int tokens,
    int queryHeads, int keyHeads, int rotaryDim, int position, float theta, float eps,
    void *stream = nullptr);

// Stable expert-id tie breaking. Scratch: [B,T,K] INT32 ids and FP32 scores,
// plus [B,E] byte mask; only needed for preview coreset filtering.
bool FastllmCudaCloverRoute(const void *logits, int *indices, float *scores,
    int *scratchIds, float *scratchScores, unsigned char *allowed,
    int batch, int tokens, int experts, int topk, float coresetTopP,
    void *stream = nullptr);
// Resident packed BF16 [E,2I,H] / [E,H,I] experts. Scratch is BF16
// [M,K,I] and [M,K,H]. Each down contribution rounds AFTER route weighting.
// Optional INT32 workspace: E*(M+1) + E + ceil(M*K/16) + 1 elements.
// It enables grouped Tensor Core products for blocks with at least four rows.
// Optional 128-byte-aligned packed scratch enables native SM90 TMA for
// aligned blocks. The caller may omit it for small eager blocks; zero means
// the shape/compiled architecture requires the generic path.
std::size_t FastllmCudaCloverExpertsPackedBytes(int rows, int hidden, int intermediate, int experts, int topk);
// Fused shared expert epilogue. shared is ungated BF16 [rows,hidden],
// sharedGate is BF16 [rows]. Both must be null or both supplied. Retain
// the BF16 sigmoid, product, routed-sum and final-add rounding boundaries.
bool FastllmCudaCloverExperts(const void *input, const void *gateUp, const void *down, const int *indices,
    const float *scores, void *middle, void *parts, void *output, int rows, int hidden,
    int intermediate, int experts, int topk, void *stream = nullptr, int *workspace = nullptr,
    void *packed = nullptr, const void *shared = nullptr, const void *sharedGate = nullptr);

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

// BF16 NHD attention, fixed checkpoint head dimension 256. Optional split-KV
// scratch: ceil((cachedTokens+tokens)/256)*tokens*heads*(256*2+4) bytes.
bool FastllmCudaCloverAttention(const void *query, const void *key,
    const void *value, void *output, int tokens, int cachedTokens,
    int heads, int kvHeads, bool preview, void *stream = nullptr,
    void *workspace = nullptr);

#endif
