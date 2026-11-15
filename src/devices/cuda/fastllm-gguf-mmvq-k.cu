#include "fastllm-gguf-mmvq-kernels.cuh"

FASTLLM_INSTANTIATE_MMVQ_OUTPUTS(GGML_TYPE_Q2_K)
FASTLLM_INSTANTIATE_MMVQ_OUTPUTS(GGML_TYPE_Q3_K)
FASTLLM_INSTANTIATE_MMVQ_OUTPUTS(GGML_TYPE_Q4_K)
FASTLLM_INSTANTIATE_MMVQ_OUTPUTS(GGML_TYPE_Q5_K)
FASTLLM_INSTANTIATE_MMVQ_OUTPUTS(GGML_TYPE_Q6_K)

FASTLLM_INSTANTIATE_MMVQ_STORES(GGML_TYPE_Q2_K)
FASTLLM_INSTANTIATE_MMVQ_STORES(GGML_TYPE_Q4_K)

// Each CTA handles one group and a bounded token tile. The dot-product
// implementation and reduction order are shared with ordinary MMVQ.
template <typename Output, int Warps>
__global__ void FastllmGgufGroupedMmvqKernel(const char *weights, const block_q8_1 *input,
        Output *output, int groups, int columns, int rows, int padded, int tokens, size_t weightStride) {
    const int group = blockIdx.y, first = blockIdx.z * MMVQ_MAX_BATCH_SIZE;
    const int count = min(MMVQ_MAX_BATCH_SIZE, tokens - first);
    const auto *w = weights + size_t(group) * weightStride;
    const auto *x = input + (size_t(group) * tokens + first) * (padded / QK8_1);
    auto *y = output + (size_t(first) * groups + group) * rows;
#define GROUP_TILE(N) case N: \
    if constexpr (N <= 4) { \
        mul_mat_vec_q<GGML_TYPE_Q5_K, N, Warps, Output>(w,x,y,columns,rows,padded,groups*rows,2*blockIdx.x); \
        if (2*blockIdx.x+1<rows) mul_mat_vec_q<GGML_TYPE_Q5_K,N,Warps,Output>(w,x,y,columns,rows,padded,groups*rows,2*blockIdx.x+1); \
    } else mul_mat_vec_q<GGML_TYPE_Q5_K,N,Warps,Output>(w,x,y,columns,rows,padded,groups*rows); \
    break
    switch (count) {
        GROUP_TILE(1); GROUP_TILE(2); GROUP_TILE(3); GROUP_TILE(4);
        GROUP_TILE(5); GROUP_TILE(6); GROUP_TILE(7); GROUP_TILE(8);
    }
#undef GROUP_TILE
}

// Independent WoA groups retain the ordinary Linear reduction tree for
// each batch size. Write token-major output directly with a strided row pitch.
void FastllmGgufQ5KGroupedGemv(const void *weights, const void *input,
        void *output, fastllm::DataType dtype, int groups, int columns, int rows,
        int padded, int tokens, cudaStream_t stream) {
    const uint64_t ws = size_t(rows) * ggml_row_size(GGML_TYPE_Q5_K, columns);
    if (tokens > MMVQ_MAX_BATCH_SIZE) {
        const dim3 grid((rows + 1) / 2, groups, (tokens + MMVQ_MAX_BATCH_SIZE - 1) / MMVQ_MAX_BATCH_SIZE);
#define BATCH_GROUP(T, W) FastllmGgufGroupedMmvqKernel<T,W><<<grid,dim3(WARP_SIZE,W),0,stream>>>( \
        (const char*)weights,(const block_q8_1*)input,(T*)output,groups,columns,rows,padded,tokens,ws)
#define BATCH_TYPE(T) if(tokens < fastllm::FastllmCudaGetLinearExactBatchThreshold()) { BATCH_GROUP(T,4); } else { BATCH_GROUP(T,1); }
        if(dtype==fastllm::FLOAT32) { BATCH_TYPE(float); }
        else if(dtype==fastllm::FLOAT16) { BATCH_TYPE(half); }
        else { BATCH_TYPE(__nv_bfloat16); }
#undef BATCH_TYPE
#undef BATCH_GROUP
        return;
    }
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

// Shared-input projections reuse this translation unit's GGUF primitives.
FASTLLM_INSTANTIATE_SHARED_MMVQ(GGML_TYPE_Q5_K)
FASTLLM_INSTANTIATE_SHARED_MMVQ(GGML_TYPE_Q6_K)
