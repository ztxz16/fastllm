#include "fastllm-gguf-mmq-kernels.cuh"

namespace fastllm_gguf_mmq {
FASTLLM_INSTANTIATE_MMQ(GGML_TYPE_IQ2_XXS)
FASTLLM_INSTANTIATE_MMQ(GGML_TYPE_IQ2_XS)
FASTLLM_INSTANTIATE_MMQ(GGML_TYPE_IQ2_S)
} // namespace fastllm_gguf_mmq
