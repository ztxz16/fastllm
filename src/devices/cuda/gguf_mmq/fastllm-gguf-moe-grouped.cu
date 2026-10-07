#include "fastllm-cuda-gguf-projections.h"
#include "fastllm-gguf-mmq-common.cuh"
#include "fastllm-gguf-moe-stream.cuh"
#include "fastllm-gguf-moe-glm5.h"
#include "../moe/fastllm-moe-v41-q8.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <type_traits>

#include <mutex>

namespace fastllm_gguf_mmq {
#include "mmq.cuh"
#include "../moe/fastllm-moe-gguf-q8.cuh"
#include "fastllm-gguf-mmq-io.cuh"
#include "fastllm-gguf-moe-grouped.cuh"
} // namespace fastllm_gguf_mmq

size_t fastllm_gguf_mmq::Glm5GroupedWorkspaceBytes(int gateType, int downType,
        int rows, int hidden, int inter, int experts, int topk) {
    using namespace fastllm_gguf_mmq;
    // IQ4_XS down uses a BF16 dot in GLM. Quantizing it to Q8 would change
    // the model's arithmetic, so retain GEMV for that type pair.
    if ((gateType != GGML_TYPE_IQ2_XXS && gateType != GGML_TYPE_IQ2_S) ||
        downType != GGML_TYPE_IQ3_XXS || rows <= 0 || rows > 4096 ||
        hidden <= 0 || hidden > 24576 || hidden%256 ||
        inter <= 0 || inter > 24576 || inter%256 ||
        experts <= 0 || experts > 1024 || topk <= 0 || topk > 16 ||
        rows*topk + experts*(grouped_moe::kTile-1) + grouped_moe::kTile-1 > 65535 ||
        !int8_mma_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) return 0;
    return grouped_moe::Workspace(nullptr, rows, hidden, inter, experts, topk).bytes;
}

bool fastllm_gguf_mmq::RunGlm5Grouped(const fastllm::Data &input, fastllm::Data &gate,
        fastllm::Data &output, const void *weightPointers, const int32_t *indices,
        const float *scores, void *workspace, int gateType, int downType,
        int hidden, int inter, int experts, int topk, float swigluLimit) {
    if (input.dataType != fastllm::BFLOAT16 || input.dataDevice != fastllm::CUDA ||
        input.dims.size() != 2 || input.dims[1] != hidden || !input.cudaData ||
        !gate.cudaData || !output.cudaData || !weightPointers || !indices || !scores || !workspace ||
        !Glm5GroupedWorkspaceBytes(gateType, downType, input.dims[0], hidden, inter, experts, topk)) return false;
    const int rows = input.dims[0];
    grouped_moe::Workspace w(workspace, rows, hidden, inter, experts, topk);
    const auto *weights = static_cast<const uint8_t *const *>(weightPointers);
    grouped_moe::PrepareRoutes(weights, indices, w, rows*topk, experts, cudaStreamPerThread);
    grouped_moe::RunGlm5(static_cast<const __nv_bfloat16 *>(input.cudaData),
        static_cast<__nv_bfloat16 *>(gate.cudaData), static_cast<__nv_bfloat16 *>(output.cudaData),
        weights, indices, scores, w, gateType, rows, hidden, inter, experts, topk, swigluLimit);
    return cudaGetLastError() == cudaSuccess;
}

size_t FastllmCudaMoeGGUFGroupedWorkspaceBytes(int gateType, int downType,
        int rows, int hidden, int inter, int experts, int topk, bool deepSeekV41) {
    using namespace fastllm_gguf_mmq;
    const bool typesSupported = deepSeekV41
        ? gateType == GGML_TYPE_Q2_K && downType == GGML_TYPE_Q4_K &&
          hidden%256 == 0 && inter%256 == 0
        : (grouped_moe::MatrixType(gateType, hidden) ||
          (gateType == GGML_TYPE_IQ1_M && hidden%256 == 0)) &&
          grouped_moe::MatrixType(downType, inter);
    if (rows <= 32 || rows > 4096 || experts <= 0 || experts > 1024 || topk <= 0 || topk > 32 ||
        rows*topk+experts*(grouped_moe::kTile-1)+grouped_moe::kTile-1 > 65535 ||
        hidden <= 0 || hidden > 24576 || inter <= 0 || inter > 24576 ||
        !typesSupported) return 0;
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!int8_mma_available(cc)) return 0;
    return grouped_moe::Workspace(nullptr, rows, hidden, inter, experts, topk).bytes;
}

bool fastllm_gguf_mmq::RunGrouped(const fastllm::Data &input, fastllm::Data &gate,
        fastllm::Data &output, const void *weightPointers, const int32_t *indices,
        const float *scores, void *workspace, int gateType, int downType,
        int hidden, int inter, int experts, int topk, bool deepSeekV41, float swigluLimit,
        cudaEvent_t downWeightsReady) {
    using namespace fastllm_gguf_mmq;
    if (deepSeekV41 && input.dataType != fastllm::BFLOAT16) return false;
    if (!FastllmCudaMoeGGUFGroupedWorkspaceBytes(gateType, downType,
        input.dims[0], hidden, inter, experts, topk, deepSeekV41)) return false;
    const auto *weights = static_cast<const uint8_t *const *>(weightPointers);
    switch (input.dataType) {
#define GROUPED_RUN(DType, T) case fastllm::DType: return grouped_moe::Run( \
            static_cast<const T *>(input.cudaData), static_cast<T *>(gate.cudaData), \
            static_cast<T *>(output.cudaData), weights, indices, scores, workspace, \
            gateType, downType, input.dims[0], hidden, inter, experts, topk, deepSeekV41, swigluLimit, downWeightsReady);
        GROUPED_RUN(FLOAT32, float)
        GROUPED_RUN(FLOAT16, half)
        GROUPED_RUN(BFLOAT16, __nv_bfloat16)
#undef GROUPED_RUN
        default: return false;
    }
}

bool FastllmCudaMoeGGUFGrouped(const fastllm::Data &input, fastllm::Data &gate,
        fastllm::Data &output, const void *weightPointers, const int32_t *indices,
        const float *scores, void *workspace, int gateType, int downType,
        int hidden, int inter, int experts, int topk, bool deepSeekV41, float swigluLimit) {
    return fastllm_gguf_mmq::RunGrouped(input, gate, output, weightPointers, indices,
        scores, workspace, gateType, downType, hidden, inter, experts, topk,
        deepSeekV41, swigluLimit, nullptr);
}

size_t fastllm_gguf_mmq::StreamedMoeWorkspaceBytes(int rows, int hidden, int inter, int topk, int capacity) {
    return grouped_moe::StreamedWorkspace(nullptr, rows, hidden, inter, topk, capacity).bytes;
}

bool fastllm_gguf_mmq::RunStreamedMoe(StreamedMoePhase phase, const fastllm::Data &input,
        fastllm::Data &gate, fastllm::Data &output, void *workspace, int capacity,
        int hidden, int inter, int topk, int gateType, int downType,
        const StreamedMoeBatch &batch, const float *scores) {
    switch (input.dataType) {
#define STREAMED_RUN(DType, T) case fastllm::DType: return grouped_moe::RunStreamed( \
        phase, static_cast<const T *>(input.cudaData), static_cast<T *>(gate.cudaData), \
        static_cast<T *>(output.cudaData), workspace, capacity, input.dims[0], hidden, inter, topk, \
        gateType, downType, batch, scores);
        STREAMED_RUN(FLOAT32, float)
        STREAMED_RUN(FLOAT16, half)
        STREAMED_RUN(BFLOAT16, __nv_bfloat16)
#undef STREAMED_RUN
        default: return false;
    }
}
