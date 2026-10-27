#pragma once
#include "fastllm.h"

namespace fastllm_gguf_mmq {
// Internal GLM Q8_K grouped prefill. Zero workspace size leaves unsupported
// types/devices on the resident GEMV path; no persistent weight conversion.
size_t Glm5GroupedWorkspaceBytes(int gateType, int downType, int rows,
    int hidden, int inter, int experts, int topk);
bool RunGlm5Grouped(const fastllm::Data &input, fastllm::Data &gate,
    fastllm::Data &output, const void *weightPointers, const int32_t *indices,
    const float *scores, void *workspace, int gateType, int downType,
    int hidden, int inter, int experts, int topk, float swigluLimit);
}
