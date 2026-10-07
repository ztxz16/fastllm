#include "fastllm-gguf-t8-extended-kernels.cuh"

namespace fastllm_gguf_small_mmvq {
FASTLLM_INSTANTIATE_EXTENDED_T8(GGML_TYPE_IQ2_S)
FASTLLM_INSTANTIATE_EXTENDED_T8(GGML_TYPE_IQ2_XS)
FASTLLM_INSTANTIATE_EXTENDED_T8(GGML_TYPE_IQ2_XXS)
} // namespace fastllm_gguf_small_mmvq
