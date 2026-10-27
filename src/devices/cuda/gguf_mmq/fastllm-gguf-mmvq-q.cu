#include "fastllm-gguf-mmvq-kernels.cuh"

namespace fastllm_gguf_mmq {
FASTLLM_INSTANTIATE_EXTENDED_MMVQ(GGML_TYPE_Q4_0)
FASTLLM_INSTANTIATE_EXTENDED_MMVQ(GGML_TYPE_Q4_1)
} // namespace fastllm_gguf_mmq
