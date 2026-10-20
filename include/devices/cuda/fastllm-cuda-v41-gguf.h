#pragma once
#include "fastllm.h"

// Resident native V4.1 GGUF MoE. GPU routes and immutable weight pointers;
// indexed GEMV for <=32 rows, grouped MMQ for supported larger batches.
// Both preserve block-32 activation quantization and FP32 expert accumulation.
bool FastllmCudaMergeMOEV41GGUFResident(
    const fastllm::Data &input, fastllm::Data &gate, fastllm::Data &activation,
    fastllm::Data &workspace, fastllm::Data &output, fastllm::Data **weights,
    int weightsBatch, const fastllm::Data &indices, const fastllm::Data &scores,
    float sharedScale, float swigluLimit);

// Internal launcher; caller validates layouts and owns all scratch tensors.
bool FastllmCudaRunV41GGUF(
    const fastllm::Data &input, fastllm::Data &gate, fastllm::Data &activation,
    fastllm::Data &workspace, fastllm::Data &output, const void *weightPointers,
    const int32_t *indices, const float *scores, int gateType, int downType,
    int hidden, int inter, int experts, int topk, float swigluLimit);

// Native CUDA activation semantics (Q8_1), distinct from NUMA's Q8_K path.
size_t FastllmCudaV41GGUFGroupedWorkspaceBytes(
    int gateType, int downType, int rows, int hidden, int inter, int experts, int topk);
bool FastllmCudaV41GGUFGroupedProjection(
    bool gate, int type, const void *inputQ8, void *outputBF16,
    const void *weightPointers, const int32_t *indices, void *workspace,
    int rows, int hidden, int inter, int experts, int topk);
