#pragma once
#include "devices/cuda/naive-n05-dsa.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <vector>

bool FastllmSm90DsaSelectConfig(int device, int rows, int capacity, int &grid, size_t &bytes);
cudaError_t FastllmSm90DsaSelectAttention(const std::vector<FastllmNaiveDSARow> &rows, int grid, unsigned *scores,
                                          const __nv_bfloat16 *query, const float *sink, __nv_bfloat16 *out,
                                          float *scratch, int *selected, const __nv_bfloat16 *index,
                                          const float *scales, uint8_t *packed, float *outScales);
