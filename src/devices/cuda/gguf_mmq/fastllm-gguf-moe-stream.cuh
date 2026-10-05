#pragma once

#include "fastllm-cuda.cuh"
#include <cuda_runtime_api.h>

namespace fastllm_gguf_mmq {
// Streamed experts can upload down weights while gate/up is computing.
// A null event means all weights are already ready on the calling stream.
bool RunGrouped(const fastllm::Data &input, fastllm::Data &gate,
    fastllm::Data &output, const void *weightPointers, const int32_t *indices,
    const float *scores, void *workspace, int gateType, int downType,
    int hidden, int inter, int experts, int topk, bool deepSeekV41,
    float swigluLimit, cudaEvent_t downWeightsReady);
}
