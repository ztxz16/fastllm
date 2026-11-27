#pragma once
#include "devices/cuda/naive-n05-kv.cuh"
#include <cuda_runtime.h>

namespace naive_indexer_sm100 {
// A request keeps its own live length and optional graph-rebound KV pointers.
// Verification packs FP8 queries into the existing Q scratch. Single-query scoring uses BF16.
struct Row {
    FastllmNaiveKV key;
    const int *live;
    __nv_bfloat16 *query;
    float *scale;
    unsigned *scores;
    int sequence, stride;
    const void *const *cache;
    __device__ int Begin() const { return 0; }
    __device__ int Count() const { return *live; }
};
constexpr int BatchLimit = 128;
// Each verification group shares a KV view and its replay binding. Lengths, Q and outputs are per row.
constexpr int VerifyQueries = 4;
constexpr int VerifyLimit = 512;
struct VerifyBatch { Row rows[VerifyLimit]; };
} // namespace naive_indexer_sm100
bool FastllmCudaNaiveIndexerSm100(const naive_indexer_sm100::Row *rows, int count, int capacity,
                                  const __nv_bfloat16 *weights, cudaStream_t stream);

bool FastllmCudaNaiveVerifyIndexerSm100(const naive_indexer_sm100::VerifyBatch &batch, int count, int capacity,
        const __nv_bfloat16 *query, const __nv_bfloat16 *weights, void *scratch, size_t scratchBytes,
        cudaStream_t stream);

bool FastllmCudaNaivePrefillIndexerSm100(const __nv_bfloat16 *query, const __nv_bfloat16 *weights,
        FastllmNaiveKV cache, int stride, int rows, int keys, int past, void *scratch, size_t scratchBytes,
        float *queryScales, float *scores, cudaStream_t stream);
