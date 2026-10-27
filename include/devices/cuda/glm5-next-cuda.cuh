#pragma once
#include "fastllm.h"

// GLM53 no-RoPE sparse prefill. Contiguous Q/output [1,Q,64,512], latent
// [1,T,512] in BF16 and causal indices [1,Q,K] in INT32 (-1 for padding).
// Reuses FlashInfer FP8 math on SM120 for Q >= 64. Returns false before
// modifying output when unsupported; CUDA errors fail. Backend selection is
// handled by the caller.
bool FastllmCudaGlm5NextDsaPrefill(const fastllm::Data &query,
    const fastllm::Data &latent, const fastllm::Data &indices,
    float scale, fastllm::Data &output);

// Reuse DeepSeek-V4's fused HC kernels with GLM/Kimi's BF16 rounding before
// the RMSNorm weight and its original FP32 reduction order.
bool FastllmCudaGlm5NextHcPreNorm(
    const fastllm::Data &x, fastllm::Data &fn, fastllm::Data &scale,
    fastllm::Data &base, fastllm::Data &norm, int hcMult, int iters,
    float eps, float normEps, fastllm::Data &output,
    fastllm::Data &post, fastllm::Data &comb);

// SM75 hybrid prefill; retain the ordinary 256-lane HC reduction order.
bool FastllmCudaGlm5NextHybridHcPreNorm(
    const fastllm::Data &x, fastllm::Data &fn, fastllm::Data &scale,
    fastllm::Data &base, fastllm::Data &norm, int hcMult, int iters,
    float eps, float normEps, fastllm::Data &output,
    fastllm::Data &post, fastllm::Data &comb);

// FP32 sparse attention for SM75 hybrid prefill/decode/verification.
// BF16 authoritative KV and queries; no additional KV quantization.
bool FastllmCudaGlm5NextHybridSparseAttention(const fastllm::Data &query,
    const fastllm::Data &latent, const fastllm::Data &indices,
    float scale, fastllm::Data &output);

// Batched FP32 weighted-ReLU index scoring on SM75 hybrid prefill.
bool FastllmCudaGlm5NextHybridIndexerScore(const fastllm::Data &query,
    const fastllm::Data &weights, const fastllm::Data &keys, fastllm::Data &output);
