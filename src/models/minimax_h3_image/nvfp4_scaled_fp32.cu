#define FASTLLM_NVFP4_OUTPUT float
#define FASTLLM_NVFP4_ENTRY nvfp4GemmFloat
#include "nvfp4_gemm.cuh"
#undef FASTLLM_NVFP4_OUTPUT
#undef FASTLLM_NVFP4_ENTRY
