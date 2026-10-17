#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace fastllm { namespace cuda { namespace fp8_cache {

// Packed FP8 rows retain their 128 E4M3 bytes followed by the FP32 scale.
// This reader also works directly on native NUMA gate/up row pairs; cache
// admission never expands the weights or makes another host weight copy.
__device__ inline float Dot(const __nv_bfloat16 *input,
                            const uint8_t *weight, int columns) {
    const int lane = threadIdx.x & 7;
    float sum = 0;
    for (int block = 0; block < columns / 128; ++block) {
        const uint8_t *w = weight + block * 132;
        const float scale = *reinterpret_cast<const float *>(w + 128);
        float partial = 0;
        for (int c = lane * 4; c < 128; c += 32) {
            const uint32_t bytes = *reinterpret_cast<const uint32_t *>(w + c);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                __nv_fp8_e4m3 value;
                value.__x = uint8_t(bytes >> (8 * j));
                partial = __fmaf_rn(__bfloat162float(input[block * 128 + c + j]),
                                    float(value), partial);
            }
        }
        sum = __fmaf_rn(partial, scale, sum);
    }
    for (int delta = 4; delta; delta >>= 1)
        sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, delta, 8));
    return sum;
}

__device__ inline float BFloat(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

// Scored BF16 arithmetic is separate from the weight encoding: GLM uses
// block-128 down activations, V4.1 block-32. The caller supplies any required
// input quantization. Apply scores before down quantization, exactly once.
template<int QuantBlock>
static __global__ void GateScored(const __nv_bfloat16 *input, const int32_t *slots,
        const uint8_t *records, const float *scores, __nv_bfloat16 *activation,
        int hidden, int inter, size_t stride, float limit, bool interleaved,
        int topk, const int32_t *routeMap = nullptr) {
    const int selected = blockIdx.y, slot = slots ? slots[selected] : 0;
    if (slot < 0) return;
    const int route = routeMap ? routeMap[selected] : selected;
    input += size_t(route / topk) * hidden;
    const int column = blockIdx.x * QuantBlock + threadIdx.x / 8;
    const size_t pitch = (hidden / 128) * 132;
    const uint8_t *record = records + size_t(slot) * stride;
    const int gateRow = interleaved ? 2 * column : column;
    const int upRow = interleaved ? 2 * column + 1 : column + inter;
    float gate = BFloat(Dot(input, record + size_t(gateRow) * pitch, hidden));
    float up = BFloat(Dot(input, record + size_t(upRow) * pitch, hidden));
    __shared__ float values[QuantBlock], maxima[QuantBlock / 32];
    if ((threadIdx.x & 7) == 0) {
        if (limit > 0) {
            gate = fminf(gate, limit);
            up = fmaxf(-limit, fminf(up, limit));
        }
        values[threadIdx.x / 8] = BFloat(__fmul_rn(
            __fmul_rn(gate / (1.0f + expf(-gate)), up), scores[route]));
    }
    __syncthreads();
    if (threadIdx.x < QuantBlock) {
        float amax = fmaxf(1e-4f, fabsf(values[threadIdx.x]));
        for (int delta = 16; delta; delta >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, delta));
        if ((threadIdx.x & 31) == 0) maxima[threadIdx.x / 32] = amax;
    }
    __syncthreads();
    if (threadIdx.x < QuantBlock) {
        float amax = maxima[0];
#pragma unroll
        for (int i = 1; i < QuantBlock / 32; ++i) amax = fmaxf(amax, maxima[i]);
        const unsigned bits = __float_as_uint(amax / 448.0f);
        const int exponent = int((bits >> 23) & 255) - 127 + ((bits & 0x7fffff) != 0);
        const float scale = exp2f(float(exponent));
        const float q = fmaxf(-448.0f, fminf(448.0f, values[threadIdx.x] / scale));
        activation[size_t(route) * inter + blockIdx.x * QuantBlock + threadIdx.x] =
            __float2bfloat16_rn(float(__nv_fp8_e4m3(q)) * scale);
    }
}

static __global__ void DownScored(const __nv_bfloat16 *activation, const int32_t *slots,
        const uint8_t *records, float *output, int hidden, int inter,
        size_t stride, size_t downOffset, const int32_t *routeMap = nullptr) {
    const int selected = blockIdx.y, slot = slots ? slots[selected] : 0;
    if (slot < 0) return;
    const int route = routeMap ? routeMap[selected] : selected;
    const int column = blockIdx.x * 16 + threadIdx.x / 8;
    const uint8_t *weight = records + size_t(slot) * stride + downOffset +
        size_t(column) * (inter / 128) * 132;
    const float value = Dot(activation + size_t(route) * inter, weight, inter);
    if ((threadIdx.x & 7) == 0) output[size_t(route) * hidden + column] = BFloat(value);
}

} } } // namespace fastllm::cuda::fp8_cache
