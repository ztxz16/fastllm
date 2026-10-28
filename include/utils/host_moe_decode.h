#pragma once
#include "fastllm.h"
#include <functional>
namespace fastllm {
// Ordinary host expert execution. Storage/affinity is independent of routing.
bool CanRunHostMoeDecodeExperts(Data *const *weights, int count);
void PrepareHostMoeDecodeWeights(Data *const *weights, int count);
void HostMoeDecodeExperts(const float *input, float *output, Data **weights,
    const int32_t *indices, const int32_t *gpuIndices, int topk,
    const std::function<void()> &submitGpu, double *cpuElapsedUs);
}
