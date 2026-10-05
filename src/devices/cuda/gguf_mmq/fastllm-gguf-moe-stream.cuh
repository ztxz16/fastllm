#pragma once

#include "fastllm-cuda.cuh"
#include <cuda_runtime_api.h>

namespace fastllm_gguf_mmq {
struct StreamedMoeBatch {
    int experts = 0, rows = 0;
    const uint8_t *const *weights = nullptr;
    const int *counts = nullptr, *offsets = nullptr, *tileExperts = nullptr, *routes = nullptr;
};
enum class StreamedMoePhase { Prepare, Compute, Finish };
size_t StreamedMoeWorkspaceBytes(int rows, int hidden, int inter, int topk, int capacity);
bool RunStreamedMoe(StreamedMoePhase phase, const fastllm::Data &input,
    fastllm::Data &gate, fastllm::Data &output, void *workspace, int capacity,
    int hidden, int inter, int topk, int gateType, int downType,
    const StreamedMoeBatch &batch, const float *scores);

// Streamed experts can upload down weights while gate/up is computing.
// A null event means all weights are already ready on the calling stream.
bool RunGrouped(const fastllm::Data &input, fastllm::Data &gate,
    fastllm::Data &output, const void *weightPointers, const int32_t *indices,
    const float *scores, void *workspace, int gateType, int downType,
    int hidden, int inter, int experts, int topk, bool deepSeekV41,
    float swigluLimit, cudaEvent_t downWeightsReady);
}
