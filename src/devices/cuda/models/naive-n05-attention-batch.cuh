#pragma once
#include "devices/cuda/naive-n05-dsa.h"
#include <cuda_runtime.h>

namespace naive_attention_bulk {
#if CUDART_VERSION >= 12010
constexpr int BatchLimit = 256;
#else
constexpr int BatchLimit = (4096 - 128) / sizeof(FastllmNaiveDSARow);
#endif
struct Batch {
    FastllmNaiveDSARow rows[BatchLimit];
};
static_assert(sizeof(Batch) + 128 <= (CUDART_VERSION >= 12010 ? 32764 : 4096),
              "Attention arguments exceed CUDA parameter space");
__device__ inline FastllmNaiveDSARow BatchRow(const Batch &batch, int i) { return batch.rows[i]; }
#if CUDART_VERSION >= 12010
// Verification queries may share KV bindings while using independent lengths
// and selection lists. Common geometry leaves room for larger argument batches.
struct CompactRow {
    const void *const *cache;
    const int *liveKeys, *indices;
};
constexpr int CompactLimit = 1024;
struct CompactBatch {
    FastllmNaiveDSARow geometry;
    CompactRow rows[CompactLimit];
};
static_assert(sizeof(CompactBatch) + 128 <= 32764);
inline void FillCompactBatch(CompactBatch &batch, const FastllmNaiveDSARow *rows, int count) {
    batch.geometry = rows[0];
    for (int i = 0; i < count; ++i)
        batch.rows[i] = {rows[i].cache, rows[i].liveKeys, rows[i].indices};
}
__device__ inline FastllmNaiveDSARow BatchRow(const CompactBatch &batch, int i) {
    auto row = batch.geometry;
    row.cache = batch.rows[i].cache;
    row.liveKeys = batch.rows[i].liveKeys;
    row.indices = batch.rows[i].indices;
    return row;
}
#endif
} // namespace naive_attention_bulk

namespace naive_dsa_decode {
__device__ __forceinline__ int Length(const FastllmNaiveDSARow &r) {
    return max(0, min(r.capacity, r.liveKeys ? *r.liveKeys : r.length));
}
} // namespace naive_dsa_decode
