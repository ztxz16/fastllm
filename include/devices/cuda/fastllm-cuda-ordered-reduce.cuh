#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>

namespace fastllm {
namespace cuda {

struct OrderedSum {
    __device__ __forceinline__ float operator()(float a, float b) const { return a + b; }
};
struct OrderedMax {
    __device__ __forceinline__ float operator()(float a, float b) const { return max(a, b); }
};

struct WarpArgMaxResult { float value; int index; };

// Lane-contiguous finite keys, with the lowest index winning equal values.
// Separating value and index reductions avoids carrying a comparison tuple
// through every local comparison. Callers retain their own NaN/tie policies.
template<int Items>
__device__ __forceinline__ WarpArgMaxResult WarpArgMax(
        const float (&keys)[Items], int firstIndex, int size) {
    float best = -CUDART_INF_F;
    #pragma unroll
    for (int i = 0; i < Items; ++i)
        if (firstIndex + i < size) best = fmaxf(best, keys[i]);
    #pragma unroll
    for (int delta = 16; delta; delta >>= 1)
        best = fmaxf(best, __shfl_xor_sync(0xffffffffu, best, delta));
    int index = 0x7fffffff;
    #pragma unroll
    for (int i = 0; i < Items; ++i)
        if (firstIndex + i < size && keys[i] == best) index = min(index, firstIndex + i);
    #pragma unroll
    for (int delta = 16; delta; delta >>= 1)
        index = min(index, __shfl_xor_sync(0xffffffffu, index, delta));
    return {best, index};
}

// Reproduce a descending-stride shared-memory tree, including FP32 addition
// order, while completing its final five levels inside warp 0. All block
// threads participate; only the first LogicalThreads provide values. This can
// serve different row sizes and input types after conversion to FP32 by the
// caller. No architecture-specific instructions or SM dispatch are required.
// scratch has LogicalThreads + 1 floats. The separate result cell lets the
// caller immediately reuse scratch for another reduction without a read/write
// race on the previous result. LogicalThreads is a power of two, at least 32.
template<int LogicalThreads, class Operation>
__device__ __forceinline__ float OrderedBlockReduce(float value, float *scratch,
                                                   Operation op) {
    static_assert(LogicalThreads >= 32 && (LogicalThreads & (LogicalThreads - 1)) == 0,
                  "ordered reduction requires a power-of-two thread count");
    int tid = threadIdx.x;
    if (tid < LogicalThreads) scratch[tid] = value;
    __syncthreads();
    #pragma unroll
    for (int stride = LogicalThreads / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) {
            value = op(value, scratch[tid + stride]);
            scratch[tid] = value;
        }
        __syncthreads();
    }
    if (tid < 32) {
        #pragma unroll
        for (int stride = 16; stride; stride >>= 1) {
            float other = __shfl_down_sync(0xffffffffu, value, stride);
            if (tid < stride) value = op(value, other);
        }
        if (tid == 0) scratch[LogicalThreads] = value;
    }
    __syncthreads();
    return scratch[LogicalThreads];
}
} // namespace cuda
} // namespace fastllm
