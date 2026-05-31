#pragma once

#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

template <typename T>
__device__ __forceinline__ float FastllmAttentionValueToFloat(T value);

template <>
__device__ __forceinline__ float FastllmAttentionValueToFloat<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ float FastllmAttentionValueToFloat<half>(half value) {
    return __half2float(value);
}

template <>
__device__ __forceinline__ float FastllmAttentionValueToFloat<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <>
__device__ __forceinline__ float FastllmAttentionValueToFloat<__nv_fp8_e4m3>(__nv_fp8_e4m3 value) {
    return (float)value;
}

template <typename T>
__device__ __forceinline__ T FastllmAttentionFloatToValue(float value);

template <>
__device__ __forceinline__ float FastllmAttentionFloatToValue<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ half FastllmAttentionFloatToValue<half>(float value) {
    return __float2half(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 FastllmAttentionFloatToValue<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

template <>
__device__ __forceinline__ __nv_fp8_e4m3 FastllmAttentionFloatToValue<__nv_fp8_e4m3>(float value) {
    return __nv_fp8_e4m3(value);
}
