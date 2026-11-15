#pragma once
#include <cuda_bf16.h>

// Stable per-device storage. Returns nullptr on a cold graph capture; callers
// must prepare the table during ordinary warmup before recording kernels.
const float *FastllmCudaBFloat16SiluTable();

#ifdef __CUDACC__
__device__ __forceinline__ float FastllmCudaBFloat16Silu(float gate, const float *table) {
    const unsigned bits = __float_as_uint(gate);
    // An arbitrary clipping limit can produce a non-BF16 gate. Preserve the
    // floating-point fallback instead of silently rounding that limit.
    if (table && !(bits & 0xffffu)) return table[bits >> 16];
    return gate / (1.0f + expf(-gate));
}
#endif
