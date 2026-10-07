#include "fastllm-gguf-mmvq-kernels.cuh"

namespace fastllm_gguf_mmq {
#include "fastllm-gguf-mmq-iq1.cuh"

void ensure_extended_iq1s_grid(cudaStream_t stream) {
    ensure_iq1s_grid(stream);
}

FASTLLM_INSTANTIATE_EXTENDED_MMVQ(GGML_TYPE_IQ1_S)
FASTLLM_INSTANTIATE_EXTENDED_MMVQ(GGML_TYPE_IQ1_M)
FASTLLM_INSTANTIATE_EXTENDED_MMVQ_STORES(GGML_TYPE_IQ1_M)
} // namespace fastllm_gguf_mmq
