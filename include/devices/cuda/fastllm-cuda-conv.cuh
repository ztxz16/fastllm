#pragma once
#include <cstddef>

namespace fastllm { namespace cuda_conv {
// Precision is independent of state layout: some checkpoints round products
// to the activation dtype, while others accumulate full FP32 products.
template<class Activation, bool RoundProduct>
__device__ __forceinline__ float Accumulate(float sum, float x, float weight) {
    if constexpr (RoundProduct) return sum + float(Activation(x * weight));
    else return sum + x * weight;
}
template<class T, class Weight, bool RoundProduct>
__device__ __forceinline__ float Dot(const T *x, const Weight *weight, int kernel) {
    float sum = 0.f;
    #pragma unroll
    for (int tap = 0; tap < kernel; ++tap)
        sum = Accumulate<T, RoundProduct>(sum, float(x[tap]), float(weight[tap]));
    return sum;
}
// channelInput points to token zero. Histories contain the last historyLength
// samples, with historyLength >= kernel-1. Neither input nor history is mutated.
template<class T, class Weight, class State, bool RoundProduct>
__device__ __forceinline__ float At(const T *channelInput, const State *history,
        const Weight *weight, int token, int inputStride, int kernel, int historyLength) {
    float sum = 0.f;
    for (int tap = 0; tap < kernel; ++tap) {
        int source = token + tap - kernel + 1;
        float x = source < 0 ? float(history[historyLength + source])
                            : float(channelInput[(size_t)source * inputStride]);
        sum = Accumulate<T, RoundProduct>(sum, x, float(weight[tap]));
    }
    return sum;
}
}} // namespace fastllm::cuda_conv
