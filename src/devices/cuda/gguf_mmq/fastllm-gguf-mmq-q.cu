#include "fastllm-gguf-mmq-kernels.cuh"

namespace fastllm_gguf_mmq {
#include "../moe/fastllm-moe-gguf-q8.cuh"
#include "fastllm-gguf-mmq-q2.cuh"
FASTLLM_INSTANTIATE_MMQ(GGML_TYPE_Q2_0)
FASTLLM_INSTANTIATE_MMQ(GGML_TYPE_Q4_0)
FASTLLM_INSTANTIATE_MMQ(GGML_TYPE_Q4_1)
FASTLLM_INSTANTIATE_MMQ(GGML_TYPE_Q5_0)
FASTLLM_INSTANTIATE_MMQ(GGML_TYPE_Q5_1)
FASTLLM_INSTANTIATE_MMQ(GGML_TYPE_Q8_0)
} // namespace fastllm_gguf_mmq
