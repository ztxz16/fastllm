#include "fastllm-gguf-mmvq-kernels.cuh"

FASTLLM_INSTANTIATE_MMVQ_OUTPUTS(GGML_TYPE_Q2_K)
FASTLLM_INSTANTIATE_MMVQ_OUTPUTS(GGML_TYPE_Q3_K)
FASTLLM_INSTANTIATE_MMVQ_OUTPUTS(GGML_TYPE_Q4_K)
FASTLLM_INSTANTIATE_MMVQ_OUTPUTS(GGML_TYPE_Q5_K)
FASTLLM_INSTANTIATE_MMVQ_OUTPUTS(GGML_TYPE_Q6_K)

FASTLLM_INSTANTIATE_MMVQ_STORES(GGML_TYPE_Q2_K)
FASTLLM_INSTANTIATE_MMVQ_STORES(GGML_TYPE_Q4_K)

// Independent WoA groups retain the ordinary Linear reduction tree for
// each batch size. Write token-major output directly with a strided row pitch.
void FastllmGgufQ5KGroupedGemv(const void *weights, const void *input,
        void *output, fastllm::DataType dtype, int groups, int columns, int rows,
        int padded, int tokens, cudaStream_t stream) {
    const uint64_t ws = size_t(rows) * ggml_row_size(GGML_TYPE_Q5_K, columns);
    const uint64_t xs = size_t(tokens) * padded / QK8_1 * sizeof(block_q8_1);
#define GROUPED(T, WARPS) mul_mat_vec_q_cuda_T<GGML_TYPE_Q5_K, WARPS, T>(weights, input, \
    static_cast<T *>(output), nullptr, columns, rows, padded, tokens, groups * rows, \
    groups, ws, xs, size_t(rows) * sizeof(T), 0, stream)
#define DISPATCH(T) if (tokens <= 4 || tokens < fastllm::FastllmCudaGetLinearExactBatchThreshold()) { GROUPED(T, 4); } else { GROUPED(T, 1); }
    if (dtype == fastllm::FLOAT32) { DISPATCH(float); }
    else if (dtype == fastllm::FLOAT16) { DISPATCH(half); }
    else { DISPATCH(__nv_bfloat16); }
#undef DISPATCH
#undef GROUPED
}
