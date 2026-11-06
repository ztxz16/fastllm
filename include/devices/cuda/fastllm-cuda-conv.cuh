#pragma once
#include <cstddef>

namespace fastllm { namespace cuda_conv {
// Preserve float(x / (1.0 + expf(-x))) including its promoted denominator.
// Refine a reciprocal, then check the residual and float-rounding boundary.
// Uncertain rounding, subnormals and exceptional values use the original divide.
__device__ __forceinline__ float SiluPromoted(float x) {
    const float e = expf(-x);
    const double denominator = 1.0 + double(e);
#if defined(USE_ROCM)
    return float(double(x) / denominator);
#else
    double reciprocal;
    asm("rcp.approx.ftz.f64 %0, %1;" : "=d"(reciprocal) : "d"(denominator));
    const double seedError = __fma_rn(-denominator, reciprocal, 1.0);
    // A checked 16-bit seed gives 32 bits after the reciprocal correction
    // and full double precision after correcting the quotient. Reuse this
    // residual rather than paying for a separate final division check.
    if (fabs(seedError) > 0x1p-16) return float(double(x) / denominator);
    reciprocal = __fma_rn(reciprocal, seedError, reciprocal);
    double quotient = double(x) * reciprocal;
    quotient = __fma_rn(__fma_rn(-quotient, denominator, double(x)), reciprocal, quotient);
    const float rounded = float(quotient);
    // Stay 32 double ulps away from a float midpoint, covering the refined
    // quotient's rounding error and the original division's double rounding.
    const auto tail = static_cast<unsigned long long>(__double_as_longlong(quotient)) & 0x1fffffffULL;
    if (!isfinite(rounded) || fabsf(rounded) < 0x1p-126f ||
        (tail >= 0x10000000ULL - 32 && tail <= 0x10000000ULL + 32))
        return float(double(x) / denominator);
    return rounded;
#endif
}
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
