#include "fastllm-gguf-mmq-kernels.cuh"

namespace fastllm_gguf_mmq {
#include "fastllm-gguf-mmq-iq1.cuh"
FASTLLM_INSTANTIATE_MMQ(GGML_TYPE_IQ1_S)
FASTLLM_INSTANTIATE_MMQ(GGML_TYPE_IQ1_M)
} // namespace fastllm_gguf_mmq
