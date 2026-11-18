#pragma once
#include "devices/cuda/naive-n05-kv.cuh"
#include <cuda_runtime.h>

namespace naive_indexer_sm100 {
// A request keeps its own live length and optional graph-rebound KV pointers.
// The scorer consumes the existing quantized BF16 Q scratch and FP32 scales.
struct Row {
    FastllmNaiveKV key;
    const int *live;
    const __nv_bfloat16 *query;
    const float *scale;
    unsigned *scores;
    int sequence, stride;
    const void *const *cache;
    __device__ int Begin() const { return 0; }
    __device__ int Count() const { return *live; }
};
constexpr int BatchLimit = 128;
} // namespace naive_indexer_sm100
bool FastllmCudaNaiveIndexerSm100(const naive_indexer_sm100::Row *rows, int count, int capacity,
                                  const __nv_bfloat16 *weights, cudaStream_t stream);
