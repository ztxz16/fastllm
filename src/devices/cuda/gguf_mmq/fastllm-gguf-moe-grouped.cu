#include "fastllm-cuda-gguf-projections.h"
#include "fastllm-cuda-v41-gguf.h"
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
#include "fastllm-gguf-mmq-iq1.cuh"
#include "../moe/fastllm-moe-gguf-q8.cuh"
#include "fastllm-gguf-mmq-io.cuh"
#include "fastllm-gguf-moe-grouped.cuh"
#include "fastllm-gguf-moe-v41-native.cuh"
} // namespace fastllm_gguf_mmq

size_t fastllm_gguf_mmq::Glm5GroupedWorkspaceBytes(int gateType, int downType,
        int rows, int hidden, int inter, int experts, int topk) {
    using namespace fastllm_gguf_mmq;
    // Keep IQ4_XS on the existing GEMV fallback until its grouped prefill
    // path has independent numerical coverage.
    const auto kType = [](int type) {
        return type == GGML_TYPE_Q2_K || type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q4_K;
    };
    if ((gateType != GGML_TYPE_IQ2_XXS && gateType != GGML_TYPE_IQ2_S && !kType(gateType)) ||
        (downType != GGML_TYPE_IQ3_XXS && !kType(downType)) || rows <= 0 || rows > 4096 ||
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
        weights, indices, scores, w, gateType, downType, rows, hidden, inter, experts, topk, swigluLimit);
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
        const StreamedMoeBatch &batch, const float *scores, bool glm5,
        float swigluLimit, const int32_t *indices) {
    if (glm5 && (input.dataType != fastllm::BFLOAT16 || !indices)) return false;
    switch (input.dataType) {
#define STREAMED_RUN(DType, T) case fastllm::DType: return grouped_moe::RunStreamed( \
        phase, static_cast<const T *>(input.cudaData), static_cast<T *>(gate.cudaData), \
        static_cast<T *>(output.cudaData), workspace, capacity, input.dims[0], hidden, inter, topk, \
        gateType, downType, batch, scores, glm5, swigluLimit, indices);
        STREAMED_RUN(FLOAT32, float)
        STREAMED_RUN(FLOAT16, half)
        STREAMED_RUN(BFLOAT16, __nv_bfloat16)
#undef STREAMED_RUN
        default: return false;
    }
}

size_t FastllmCudaV41GGUFGroupedWorkspaceBytes(int gateType, int downType,
        int rows, int hidden, int inter, int experts, int topk) {
    using namespace fastllm_gguf_mmq;
    const auto supported = [](int type) {
        return type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M ||
            type == GGML_TYPE_Q2_0 || type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q4_K;
    };
    if (rows <= 32 || rows > 4096 || !supported(gateType) || !supported(downType) ||
        hidden <= 0 || hidden > 24576 || hidden%256 ||
        inter <= 0 || inter > 24576 || inter%256 ||
        experts <= 0 || experts > 1024 || topk <= 0 || topk > 32 ||
        rows*topk + experts*(grouped_moe::kTile-1) + grouped_moe::kTile-1 > 65535 ||
        !int8_mma_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) return 0;
    return grouped_moe::Workspace(nullptr, rows, hidden, inter, experts, topk).bytes;
}

bool FastllmCudaV41GGUFGroupedProjection(bool gate, int type, const void *inputQ8,
        void *outputBF16, const void *weightPointers, const int32_t *indices,
        void *workspace, int rows, int hidden, int inter, int experts, int topk) {
    using namespace fastllm_gguf_mmq;
    grouped_moe::Workspace w(workspace, rows, hidden, inter, experts, topk);
    const auto *weights = static_cast<const uint8_t *const *>(weightPointers);
    if (gate) grouped_moe::PrepareRoutes(weights, indices, w, rows*topk, experts, cudaStreamPerThread);
    const int columns = gate ? hidden : inter, width = gate ? 2*inter : hidden;
    // IQ1_S consumes the original FP16 scale AND sum. K-format bias
    // corrections use integer sums; all other formats consume D4 scales.
    const bool ds4 = type == GGML_TYPE_IQ1_S;
    const dim3 gather((w.activeRows+7)/8, columns/128);
    if (ds4) grouped_moe::GatherNativeV41<true><<<gather, 256, 0, cudaStreamPerThread>>>(
        static_cast<const block_q8_1 *>(inputQ8), w.quantized, w.groupRoutes,
        w.offsets+experts, columns, w.capacity, gate ? topk : 1);
    else grouped_moe::GatherNativeV41<false><<<gather, 256, 0, cudaStreamPerThread>>>(
        static_cast<const block_q8_1 *>(inputQ8), w.quantized, w.groupRoutes,
        w.offsets+experts, columns, w.capacity, gate ? topk : 1);
    if (type == GGML_TYPE_IQ1_S) ensure_iq1s_grid(cudaStreamPerThread);
    switch (type) {
#define NATIVE_MATRIX(T) case GGML_TYPE_##T: \
        if (rows >= 1024) grouped_moe::LaunchMatrix<GGML_TYPE_##T, 64, false, true>( \
            weights, gate ? 0 : 1, w, experts, columns, width, cudaStreamPerThread); \
        else grouped_moe::LaunchMatrix<GGML_TYPE_##T, 16, false, true>( \
            weights, gate ? 0 : 1, w, experts, columns, width, cudaStreamPerThread); break;
        NATIVE_MATRIX(IQ1_S) NATIVE_MATRIX(IQ1_M) NATIVE_MATRIX(Q2_0)
        NATIVE_MATRIX(Q3_K) NATIVE_MATRIX(Q4_K)
#undef NATIVE_MATRIX
        default: return false;
    }
    grouped_moe::ScatterNativeV41<<<dim3((width+255)/256, rows*topk), 256, 0, cudaStreamPerThread>>>(
        w.products, static_cast<__nv_bfloat16 *>(outputBF16), w.routeGroups, width);
    return cudaGetLastError() == cudaSuccess;
}
