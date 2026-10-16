#pragma once
#include "fastllm.h"

// Hybrid output helpers. Unsupported descriptors leave output intact.
bool FastllmCudaHybridHcMeanNorm(const fastllm::Data &input,
    const fastllm::Data &weight, float eps, fastllm::Data &output);
bool FastllmCudaHybridTop1(const fastllm::Data &input, fastllm::Data &output);
// Preserve the BF16 expert sum and the original FP64 mHC residual update.
bool FastllmCudaHybridMoeHcPost(const fastllm::Data &shared, const fastllm::Data &routed,
    const fastllm::Data &residual, const fastllm::Data &post,
    const fastllm::Data &comb, fastllm::Data &output);
