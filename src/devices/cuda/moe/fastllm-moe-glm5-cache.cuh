#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace fastllm {
namespace cuda {
namespace glm5_cache {

__device__ inline float BFloat(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ inline float FP4(unsigned value) {
    const unsigned magnitude = value & 7;
    const unsigned bits = magnitude >= 2 ? (magnitude + 252) << 22 : magnitude * 0x3f000000u;
    return __uint_as_float(bits | ((value & 8) << 28));
}

// Eight lanes consume one block-16 at a time. Scales remain E4M3 in the
// cache, with the gate/up/down global multiplier in each row header.
__device__ inline float Dot(const __nv_bfloat16 *input,
                           const uint8_t *weight, int columns) {
    const int lane = threadIdx.x & 7;
    const float global = *reinterpret_cast<const float *>(weight);
    float sum = 0;
    for (int block = 0; block < columns / 16; ++block) {
        const uint8_t *w = weight + 4 + block * 9;
        __nv_fp8_e4m3 scale;
        scale.__x = w[8];
        const float effective = __fmul_rn(float(scale), global);
        const unsigned packed = w[lane];
        const int col = block * 16 + lane * 2;
        const float pair = __fadd_rn(
            __fmul_rn(__bfloat162float(input[col]), FP4(packed & 15)),
            __fmul_rn(__bfloat162float(input[col + 1]), FP4(packed >> 4)));
        sum = __fmaf_rn(pair, effective, sum);
    }
    for (int delta = 4; delta; delta >>= 1)
        sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, delta, 8));
    return sum;
}

// One complete block-128 of down activations per CTA. Keep the same BF16
// boundaries, asymmetric clamp, score placement and power-of-two FP8 scale
// as GLM's NUMA path; the incoming activation is not quantized to FP8.
static __global__ void Gate(const __nv_bfloat16 *input, const int32_t *slots,
        const uint8_t *records, const float *scores, __nv_bfloat16 *activation,
        int hidden, int inter, size_t stride, float limit,
        int topk = 0, const int32_t *routeMap = nullptr) {
    const int selected = blockIdx.y, slot = slots ? slots[selected] : 0;
    if (slot < 0) return;
    const int route = routeMap ? routeMap[selected] : selected;
    if (topk) input += size_t(route / topk) * hidden;
    const int column = blockIdx.x * 128 + threadIdx.x / 8;
    const size_t pitch = 4 + (hidden / 16) * 9;
    const uint8_t *record = records + size_t(slot) * stride;
    // Resident and streamed records retain NUMA's adjacent gate/up rows.
    float gate = BFloat(Dot(input, record + (2 * column) * pitch, hidden));
    float up = BFloat(Dot(input, record + (2 * column + 1) * pitch, hidden));
    __shared__ float values[128], maxima[4];
    if ((threadIdx.x & 7) == 0) {
        if (limit > 0) {
            gate = fminf(gate, limit);
            up = fmaxf(-limit, fminf(up, limit));
        }
        const float value = __fmul_rn(gate / (1.0f + expf(-gate)), up);
        values[threadIdx.x / 8] = BFloat(__fmul_rn(value, scores[route]));
    }
    __syncthreads();
    if (threadIdx.x < 128) {
        float amax = fmaxf(1e-4f, fabsf(values[threadIdx.x]));
        for (int delta = 16; delta; delta >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, delta));
        if ((threadIdx.x & 31) == 0) maxima[threadIdx.x / 32] = amax;
    }
    __syncthreads();
    if (threadIdx.x < 128) {
        const float amax = fmaxf(fmaxf(maxima[0], maxima[1]), fmaxf(maxima[2], maxima[3]));
        const unsigned bits = __float_as_uint(amax / 448.0f);
        const int exponent = int((bits >> 23) & 255) - 127 + ((bits & 0x7fffff) != 0);
        const float scale = exp2f(float(exponent));
        const float q = fmaxf(-448.0f, fminf(448.0f, values[threadIdx.x] / scale));
        activation[route * inter + blockIdx.x * 128 + threadIdx.x] =
            __float2bfloat16_rn(float(__nv_fp8_e4m3(q)) * scale);
    }
}

static __global__ void Down(const __nv_bfloat16 *activation, const int32_t *slots,
        const uint8_t *records, float *output, int hidden, int inter,
        size_t stride, size_t downOffset, const int32_t *routeMap = nullptr) {
    const int selected = blockIdx.y, slot = slots ? slots[selected] : 0;
    if (slot < 0) return;
    const int route = routeMap ? routeMap[selected] : selected;
    const int column = blockIdx.x * 16 + threadIdx.x / 8;
    const uint8_t *weight = records + size_t(slot) * stride + downOffset +
        size_t(column) * (4 + (inter / 16) * 9);
    const float value = Dot(activation + route * inter, weight, inter);
    if ((threadIdx.x & 7) == 0) output[route * hidden + column] = BFloat(value);
}

} // namespace glm5_cache
} // namespace cuda
} // namespace fastllm
