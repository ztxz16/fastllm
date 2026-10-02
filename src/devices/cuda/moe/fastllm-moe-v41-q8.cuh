#pragma once

namespace fastllm::cuda::v41_gguf {
// One 256-thread CTA encodes one GGML Q8_K block. Both cache and prefill
// preserve its signed maximum (first index breaks ties) and FP32 scale.
// Callers apply the model's FP8 boundary before entering this helper.
__device__ __forceinline__ void QuantizeQ8K(float x, block_q8_K &q) {
    const int c = threadIdx.x;
    float maximum = fabsf(x);
    int first = c;
    for (int mask = 16; mask; mask >>= 1) {
        const float other = __shfl_xor_sync(0xffffffff, maximum, mask);
        const int index = __shfl_xor_sync(0xffffffff, first, mask);
        if (other > maximum || (other == maximum && index < first)) { maximum = other; first = index; }
    }
    __shared__ float maxima[8], values[256], inverse;
    __shared__ int indices[8];
    values[c] = x;
    if (c % 32 == 0) { maxima[c / 32] = maximum; indices[c / 32] = first; }
    __syncthreads();
    if (c == 0) {
        maximum = maxima[0]; first = indices[0];
        for (int i = 1; i < 8; ++i)
            if (maxima[i] > maximum || (maxima[i] == maximum && indices[i] < first)) {
                maximum = maxima[i]; first = indices[i];
            }
        inverse = maximum == 0 ? 0 : __fdiv_rn(-127.f, values[first]);
        q.d = maximum == 0 ? 0 : __fdiv_rn(1.f, inverse);
        q.sum = 0;
    }
    __syncthreads();
    q.qs[c] = inverse == 0 ? 0 : min(127, __float2int_rn(__fmul_rn(inverse, x)));
    // These consumers form exact integer sums in their dot kernels.
    if (c < 16) q.bsums[c] = 0;
}
} // namespace fastllm::cuda::v41_gguf
