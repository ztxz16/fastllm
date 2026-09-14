#pragma once

// Compact NUMA block-32 E2M1/UE8M0 weights. Cache copies restore the original
// gate/up halves. The caller quantizes the BF16 input to FP8 in groups of 32.
// All routes include their score before down-input quantization and round the
// down projection to BF16 before the ordered FP32 expert sum.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace fastllm {
namespace cuda {
namespace dsv41_cache {
__device__ inline float FP4(unsigned value, unsigned exponent) {
    // Normal E2M1 magnitudes 2..7 map to consecutive FP32 bit
    // patterns spaced by 1 << 22. Handle zero and the .5 subnormal
    // explicitly, preserving signed zero and the UE8M0 edge scales.
    const unsigned magnitude = value & 7;
    const unsigned bits = magnitude >= 2 ? (magnitude + 252) << 22 : magnitude * 0x3f000000u;
    const float decoded = __uint_as_float(bits | ((value & 8) << 28));
    const float scale = exponent == 0     ? __int_as_float(0x00400000)
                        : exponent == 255 ? __int_as_float(0x7fc00000)
                                          : __int_as_float(exponent << 23);
    return __bfloat162float(__float2bfloat16_rn(__fmul_rn(decoded, scale)));
}
__device__ inline float BFloat(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}
__device__ inline float Dot(const __nv_bfloat16 *input, const uint8_t *weight, int columns) {
    const int lane = threadIdx.x & 15;
    float sum = 0;
    for (int block = 0; block < columns / 32; ++block) {
        const uint8_t *w = weight + block * 17;
        const unsigned packed = w[lane], exponent = w[16];
        // Match BF16 dot semantics: accumulate each adjacent product pair.
        const float pair =
            __fadd_rn(__fmul_rn(__bfloat162float(input[block * 32 + lane * 2]), FP4(packed & 15, exponent)),
                      __fmul_rn(__bfloat162float(input[block * 32 + lane * 2 + 1]), FP4(packed >> 4, exponent)));
        sum = __fadd_rn(sum, pair);
    }
    for (int delta = 8; delta; delta >>= 1)
        sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, delta, 16));
    return sum;
}
template <bool Batched = false>
__global__ void Gate(const __nv_bfloat16 *input, const int32_t *slots, const uint8_t *records, const float *scores,
                     __nv_bfloat16 *activation, int hidden, int inter, size_t stride, float limit, int topk = 1) {
    const int route = blockIdx.y, slot = slots[route];
    if constexpr (Batched)
        input += (size_t)(route / topk) * hidden;
    if (slot < 0)
        return;
    const int column = blockIdx.x * 32 + threadIdx.x / 16;
    const size_t pitch = size_t(hidden / 32) * 17;
    const uint8_t *record = records + size_t(slot) * stride;
    float gate = BFloat(Dot(input, record + column * pitch, hidden));
    float up = BFloat(Dot(input, record + (column + inter) * pitch, hidden));
    __shared__ float values[32];
    if ((threadIdx.x & 15) == 0) {
        if (limit > 0) {
            gate = fminf(gate, limit);
            up = fmaxf(-limit, fminf(up, limit));
        }
        float value = __fmul_rn(gate / (1.0f + expf(-gate)), up);
        values[threadIdx.x / 16] = BFloat(__fmul_rn(value, scores[route]));
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        float value = values[threadIdx.x];
        float amax = fmaxf(1e-4f, fabsf(value));
        for (int delta = 16; delta; delta >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, delta));
        const unsigned bits = __float_as_uint(amax / 448.0f);
        const int exponent = int((bits >> 23) & 255) - 127 + ((bits & 0x7fffff) != 0);
        const float scale = exp2f(float(exponent));
        const float q = fmaxf(-448.0f, fminf(448.0f, value / scale));
        activation[route * inter + blockIdx.x * 32 + threadIdx.x] =
            __float2bfloat16_rn(float(__nv_fp8_e4m3(q)) * scale);
    }
}
static __global__ void Down(const __nv_bfloat16 *activation, const int32_t *slots, const uint8_t *records,
                            float *output, int hidden, int inter, size_t stride, size_t downOffset) {
    const int route = blockIdx.y, slot = slots[route];
    if (slot < 0)
        return;
    const int column = blockIdx.x * 8 + threadIdx.x / 16;
    const uint8_t *weight = records + size_t(slot) * stride + downOffset + size_t(column) * (inter / 32) * 17;
    const float value = Dot(activation + route * inter, weight, inter);
    if ((threadIdx.x & 15) == 0)
        output[route * hidden + column] = BFloat(value);
}
static __global__ void Reduce(const float *cpu, const float *gpu, const int32_t *gpuIndices, const int32_t *indices,
                              __nv_bfloat16 *output, int hidden, int topk) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= hidden)
        return;
    const int row = blockIdx.y;
    cpu += (size_t)row * topk * hidden;
    gpu += (size_t)row * topk * hidden;
    gpuIndices += row * topk;
    indices += row * topk;
    output += (size_t)row * hidden;
    // CPU V4/V4.1 accumulates in ascending expert-id order (stable on ties).
    int order[16];
    for (int r = 0; r < topk; ++r) {
        int pos = r;
        while (pos && indices[order[pos - 1]] > indices[r]) {
            order[pos] = order[pos - 1];
            --pos;
        }
        order[pos] = r;
    }
    float sum = 0;
    for (int r = 0; r < topk; ++r) {
        const int route = order[r];
        sum = __fadd_rn(sum, (gpuIndices[route] < 0 ? cpu : gpu)[route * hidden + column]);
    }
    output[column] = __float2bfloat16_rn(sum);
}
}
}
}
