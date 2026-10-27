#pragma once
#include "fastllm.h"
namespace fastllm {
// Fixed chunk=64, K/V head dimension=128; all batch/head/chunk counts are dynamic.
// The optional native path requires dense CUDA FP16 tensors and an SM80+ image.
bool FastllmCudaGdnPrepareWyCanRun(const Data &attn, const Data &vBeta, const Data &kBeta, const Data &g);
// Returns false before any tensor value is written. Caller retains the complete
// TransferAttn + Exp + MatMul + MulTo + MatMul fallback. Compact vBeta/kOutput
// aliasing is supported; launch errors after dispatch are fatal, not fallback.
bool FastllmCudaTryGdnPrepareWy(const Data &attn, const Data &vBeta, const Data &kBeta, const Data &g,
                                Data &vOutput, Data &kOutput);
} // namespace fastllm
namespace fastllm {
// Also fuse KKT, the sequential FP16 prefix sum, and the decay/causal mask.
// Requires unmapped dense keys with the same head count as kBeta. Updates g
// in place only after admission; all failure fallbacks occur before dispatch.
bool FastllmCudaGdnPrepareFromKeyCanRun(const Data &key, const Data &vBeta, const Data &kBeta, const Data &g);
bool FastllmCudaTryGdnPrepareFromKey(const Data &key, const Data &vBeta, const Data &kBeta, Data &g,
                                     Data &decayMask, Data &vOutput, Data &kOutput);
} // namespace fastllm
