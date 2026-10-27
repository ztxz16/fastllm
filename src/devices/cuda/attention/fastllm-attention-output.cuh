#pragma once
#include "attention/cascade.cuh"

namespace flashinfer {
// KV format does not affect output merging. Share the four combinations used
// by FastLLM; other head dimensions and ID types keep generic instantiation.
extern template cudaError_t VariableLengthMergeStatesDispatched<128, half, half, uint32_t>(
    half *, float *, uint32_t *, half *, float *, uint32_t, uint32_t *,
    uint32_t, bool, cudaStream_t);
extern template cudaError_t VariableLengthMergeStatesDispatched<128, __nv_bfloat16, __nv_bfloat16, uint32_t>(
    __nv_bfloat16 *, float *, uint32_t *, __nv_bfloat16 *, float *, uint32_t, uint32_t *,
    uint32_t, bool, cudaStream_t);
extern template cudaError_t VariableLengthMergeStatesDispatched<256, half, half, uint32_t>(
    half *, float *, uint32_t *, half *, float *, uint32_t, uint32_t *,
    uint32_t, bool, cudaStream_t);
extern template cudaError_t VariableLengthMergeStatesDispatched<256, __nv_bfloat16, __nv_bfloat16, uint32_t>(
    __nv_bfloat16 *, float *, uint32_t *, __nv_bfloat16 *, float *, uint32_t, uint32_t *,
    uint32_t, bool, cudaStream_t);
} // namespace flashinfer
