#pragma once

#include "fastllm.h"

// V100 等无 FlashInfer 时的原生分页注意力实现（gather + cublas / graph-capturable kernel）。
bool FastllmCudaHalfPagedAttentionFastllmFallback(
    fastllm::Data &q, fastllm::Data &k, fastllm::Data &v, fastllm::Data &output, int group, float scale);

bool FastllmCudaHalfPagedAttentionBatchFastllmFallback(
    fastllm::Data &q, fastllm::Data &kCaches, fastllm::Data &vCaches,
    fastllm::Data &qSizes, fastllm::Data &pageSizes, fastllm::Data &pageIndexs,
    fastllm::Data &lastPageLens, fastllm::Data &output, int group, float scale);
