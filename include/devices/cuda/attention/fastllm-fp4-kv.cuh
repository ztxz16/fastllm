#ifndef FASTLLM_FP4_KV_CUH
#define FASTLLM_FP4_KV_CUH

#include "fastllm-attention-dtype.cuh"
#include <cstdint>
#include <type_traits>

// NVFP4 KV uses an E4M3 scale per 16 E2M1 values. Each physical page
// contains [packed values][scales], so page copies include both arrays.
// The tensor-level scale is 1; attention receives the same block scales.
__device__ __forceinline__ uint8_t FastllmFP4KVEncode(float value) {
    const float a = fabsf(value);
    const uint8_t code = a <= 0.25f ? 0 : a < 0.75f ? 1 :
        a <= 1.25f ? 2 : a < 1.75f ? 3 : a <= 2.5f ? 4 :
        a < 3.5f ? 5 : a <= 5.0f ? 6 : 7;
    return code | (signbit(value) ? 8 : 0);
}

template <typename T>
__device__ __forceinline__ void FastllmWriteFP4KVBlock(
        uint8_t *cache, const T *input, size_t logicalOffset, size_t pageElements) {
    // Caller provides one lane for each aligned block of 16 contiguous values.
    const size_t page = logicalOffset / pageElements;
    const size_t offset = logicalOffset % pageElements;
    uint8_t *pageData = cache + page * (pageElements / 16 * 9);
    uint8_t *packed = pageData + offset / 2;
    uint8_t *sf = pageData + pageElements / 2 + offset / 16;
    float values[16];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        values[i] = float(input[i]);
        amax = fmaxf(amax, fabsf(values[i]));
    }
    const __nv_fp8_e4m3 scale(amax / 6.0f);
    *sf = scale.__x;
    const float roundedScale = float(scale);
    const float invScale = roundedScale > 0.0f ? 1.0f / roundedScale : 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        packed[i] = FastllmFP4KVEncode(values[2 * i] * invScale) |
            (FastllmFP4KVEncode(values[2 * i + 1] * invScale) << 4);
    }
}

template <typename DstT, typename SrcT>
__device__ __forceinline__ void FastllmWritePagedKV(
        DstT *cache, const SrcT *input, size_t logicalOffset, size_t pageElements) {
    if constexpr (std::is_same_v<DstT, uint8_t>) {
        if (logicalOffset % 16 == 0) {
            FastllmWriteFP4KVBlock(cache, input, logicalOffset, pageElements);
        }
    } else {
        cache[logicalOffset] = FastllmAttentionFloatToValue<DstT>(
            FastllmAttentionValueToFloat(*input));
    }
}

#endif
