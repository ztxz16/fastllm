#include "fastllm-paged-attention-kernels.cuh"

#if defined(FASTLLM_ENABLE_FLASHINFER)
FASTLLM_INSTANTIATE_PAGED_ATTENTION(half, half)
FASTLLM_INSTANTIATE_PAGED_ATTENTION(__nv_bfloat16, __nv_bfloat16)
#endif
