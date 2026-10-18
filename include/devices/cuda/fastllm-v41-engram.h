#pragma once

#include "fastllm.h"
#include <cstdint>
#include <vector>

// Compact FP8/E8M0 rows are partitioned over the selected peer-accessible GPUs.
// Creation is optional: unsupported topology or insufficient memory returns null.
// The caller owns the handle and must keep it alive through all gather consumers.
void *FastllmCudaV41CreateEngram(const std::vector<int> &devices, const uint8_t *weights,
                                 const uint8_t *scales, int64_t rows, int dim, int scaleBlock);
void FastllmCudaV41FreeEngram(void *handle);
bool FastllmCudaV41GatherEngram(void *handle, const std::vector<int64_t> &rows,
                                fastllm::Data &output);

// Graph-safe gather. The owner validates row IDs before their upload and keeps
// the device row buffer and output alive for the captured graph's lifetime.
bool FastllmCudaV41GatherEngramDevice(void *handle, const int64_t *rows, size_t count,
                                      fastllm::Data &output);
