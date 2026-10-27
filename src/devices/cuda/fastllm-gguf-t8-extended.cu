#include "fastllm-gguf-t8-extended-dispatch.cuh"

namespace fastllm_gguf_small_mmvq {

bool DispatchExtendedT8(ggml_type type, int outputKind, int storeMode, const void *weights,
                        const block_q8_1 *input, void *output, int columns, int rows, int inputStride,
                        int outputStride, cudaStream_t stream) {
    if (rows < 4096)
        return false;
#define FASTLLM_EXTENDED_TYPE(T)                                                                             \
    case T:                                                                                                  \
        return DispatchOutput<T>(outputKind, storeMode, weights, input, output, columns, rows, inputStride,  \
                                 outputStride, stream)
    switch (type) {
        FASTLLM_EXTENDED_TYPE(GGML_TYPE_IQ2_S);
        FASTLLM_EXTENDED_TYPE(GGML_TYPE_IQ2_XS);
        FASTLLM_EXTENDED_TYPE(GGML_TYPE_IQ2_XXS);
        FASTLLM_EXTENDED_TYPE(GGML_TYPE_IQ3_S);
        FASTLLM_EXTENDED_TYPE(GGML_TYPE_Q2_K);
    default:
        return false;
    }
#undef FASTLLM_EXTENDED_TYPE
}
} // namespace fastllm_gguf_small_mmvq

// The MMQ-backed MMVQ units include small-mmvq.cuh inside their own namespace.
// Forward that entry point to the same instantiations instead of duplicating
// all extended kernels in the legacy MMQ compilation unit.
namespace fastllm_gguf_mmq::fastllm_gguf_small_mmvq {
bool DispatchExtendedT8(ggml_type type, int outputKind, int storeMode, const void *weights,
                        const block_q8_1 *input, void *output, int columns, int rows, int inputStride,
                        int outputStride, cudaStream_t stream) {
    return ::fastllm_gguf_small_mmvq::DispatchExtendedT8(type, outputKind, storeMode, weights, input, output,
                                                         columns, rows, inputStride, outputStride, stream);
}
} // namespace fastllm_gguf_mmq::fastllm_gguf_small_mmvq
