#include "fastllm-paged-attention-kernels.cuh"

// FP4 KV storage; the query precision selects independent kernel instances.
#if defined(FASTLLM_ENABLE_FLASHINFER) && CUDA_VERSION >= 12080
FASTLLM_INSTANTIATE_PAGED_ATTENTION(half, __nv_fp4x2_e2m1)
#endif
