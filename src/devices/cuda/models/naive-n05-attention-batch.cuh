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
} // namespace naive_attention_bulk

namespace naive_dsa_decode {
__device__ __forceinline__ int Length(const FastllmNaiveDSARow &r) {
    return max(0, min(r.capacity, r.liveKeys ? *r.liveKeys : r.length));
}
} // namespace naive_dsa_decode
