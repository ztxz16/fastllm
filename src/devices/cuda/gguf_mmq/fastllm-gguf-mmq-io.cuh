#pragma once
// Included inside fastllm_gguf_mmq. Shared scalar conversions and Q8 activation packing.

template <typename T>
struct mmq_io;

template <>
struct mmq_io<half> {
    static __device__ __forceinline__ float to_float(half value) {
        return __half2float(value);
    }
    static __device__ __forceinline__ half from_float(float value) {
        return __float2half_rn(value);
    }
};

template <>
struct mmq_io<__nv_bfloat16> {
    static __device__ __forceinline__ float to_float(__nv_bfloat16 value) {
        return __bfloat162float(value);
    }
    static __device__ __forceinline__ __nv_bfloat16 from_float(float value) {
        return __float2bfloat16_rn(value);
    }
};

template <>
struct mmq_io<float> {
    static __device__ __forceinline__ float to_float(float value) {
        return value;
    }
    static __device__ __forceinline__ float from_float(float value) {
        return value;
    }
};

template <typename InputType>
static __global__ void quantize_mmvq_q8_1(
        const InputType *__restrict__ input,
        block_q8_1 *__restrict__ quantized, int cols) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) {
        return;
    }
    const int row = blockIdx.y;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int block = row * (cols / QK8_1) + col / QK8_1;
    const float value = mmq_io<InputType>::to_float(
        input[static_cast<size_t>(row) * cols + col]);

    float abs_max = warp_reduce_max(fabsf(value));
    float sum = warp_reduce_sum(value);
    const float scale = abs_max / 127.0f;
    quantized[block].qs[lane] = static_cast<int8_t>(
        abs_max == 0.0f ? 0 : roundf(value / scale));
    if (lane == 0) {
        quantized[block].ds = make_half2(
            __float2half(scale), __float2half(sum));
    }
}
