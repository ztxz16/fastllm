#include "fastllm-attention-common.cuh"

#ifdef FASTLLM_ENABLE_FLASHINFER
namespace flashinfer {
template cudaError_t VariableLengthMergeStatesDispatched<128, half, half, uint32_t>(
    half *, float *, uint32_t *, half *, float *, uint32_t, uint32_t *,
    uint32_t, bool, cudaStream_t);
template cudaError_t VariableLengthMergeStatesDispatched<128, __nv_bfloat16, __nv_bfloat16, uint32_t>(
    __nv_bfloat16 *, float *, uint32_t *, __nv_bfloat16 *, float *, uint32_t, uint32_t *,
    uint32_t, bool, cudaStream_t);
template cudaError_t VariableLengthMergeStatesDispatched<256, half, half, uint32_t>(
    half *, float *, uint32_t *, half *, float *, uint32_t, uint32_t *,
    uint32_t, bool, cudaStream_t);
template cudaError_t VariableLengthMergeStatesDispatched<256, __nv_bfloat16, __nv_bfloat16, uint32_t>(
    __nv_bfloat16 *, float *, uint32_t *, __nv_bfloat16 *, float *, uint32_t, uint32_t *,
    uint32_t, bool, cudaStream_t);
} // namespace flashinfer
#endif
