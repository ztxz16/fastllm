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
