#include "fastllm-gguf-mmvq-kernels.cuh"

FASTLLM_INSTANTIATE_MMVQ_OUTPUTS(GGML_TYPE_Q2_K)
FASTLLM_INSTANTIATE_MMVQ_OUTPUTS(GGML_TYPE_Q3_K)
FASTLLM_INSTANTIATE_MMVQ_OUTPUTS(GGML_TYPE_Q4_K)
FASTLLM_INSTANTIATE_MMVQ_OUTPUTS(GGML_TYPE_Q5_K)
FASTLLM_INSTANTIATE_MMVQ_OUTPUTS(GGML_TYPE_Q6_K)

FASTLLM_INSTANTIATE_MMVQ_STORES(GGML_TYPE_Q2_K)
FASTLLM_INSTANTIATE_MMVQ_STORES(GGML_TYPE_Q4_K)

// Independent WoA groups retain the single-GEMV four-warp reduction order.
// The generic ne2 dispatch uses one warp for expert batches instead.
void FastllmGgufQ5KGroupedGemv(const void *weights, const void *input,
        void *output, fastllm::DataType dtype, int groups, int columns, int rows,
        int padded, int tokens, cudaStream_t stream) {
    const uint64_t ws = size_t(rows) * ggml_row_size(GGML_TYPE_Q5_K, columns);
    const uint64_t xs = size_t(padded) / QK8_1 * sizeof(block_q8_1);
#define GROUPED(T) mul_mat_vec_q_cuda_T<GGML_TYPE_Q5_K, 4, T>(weights, input, \
    static_cast<T *>(output), nullptr, columns, rows, groups * padded, tokens, groups * rows, \
    groups, ws, xs, size_t(rows) * sizeof(T), 0, stream)
    if (dtype == fastllm::FLOAT32) { GROUPED(float); }
    else if (dtype == fastllm::FLOAT16) { GROUPED(half); }
    else { GROUPED(__nv_bfloat16); }
#undef GROUPED
}
