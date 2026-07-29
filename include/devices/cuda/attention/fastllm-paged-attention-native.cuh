#pragma once

#include "fastllm.h"

// V100 等无 FlashInfer 时的原生分页注意力实现（gather + cublas / graph-capturable kernel）。
bool FastllmCudaHalfPagedAttentionFastllmFallback(
    fastllm::Data &q, fastllm::Data &k, fastllm::Data &v, fastllm::Data &output, int group, float scale);

bool FastllmCudaHalfPagedAttentionBatchFastllmFallback(
    fastllm::Data &q, fastllm::Data &kCaches, fastllm::Data &vCaches,
    fastllm::Data &qSizes, fastllm::Data &pageSizes, fastllm::Data &pageIndexs,
    fastllm::Data &lastPageLens, fastllm::Data &output, int group, float scale);

// V100 decode specialization for Qwen3.5's FP16 page128 Q24/KV4 D256 GQA6 layout.
// Enabled by default for the exact shape; FASTLLM_CUDA_SM70_PAGED_XQA=0 disables it.
bool FastllmCudaTrySm70PagedAttentionDecode(
    fastllm::Data &q, fastllm::Data &kCaches, fastllm::Data &vCaches,
    fastllm::Data &qSizes, fastllm::Data &pageSizes, fastllm::Data &pageIndexs,
    fastllm::Data &lastPageLens, fastllm::Data &output,
    int group, float scale, int attentionType);

// V100 causal prefill specialization transplanted from the SM70 Volta BM32 path.
bool FastllmCudaTrySm70FlashAttentionPrefill(
    fastllm::Data &q, fastllm::Data &kCaches, fastllm::Data &vCaches,
    fastllm::Data &qSizes, fastllm::Data &pageSizes,
    fastllm::Data &pageIndexs, fastllm::Data &lastPageLens,
    fastllm::Data &output, int group, float scale, int attentionType);
