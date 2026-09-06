#pragma once

#include "fastllm-cuda.cuh"
#include "gguf.h"

#include <cuda_bf16.h>
#include <algorithm>
#include <stdexcept>
#include <type_traits>

template<typename T>
using to_t_cuda_t = void (*)(const void * __restrict__ x, T * __restrict__ y, int64_t nrows, int64_t n_per_row, cudaStream_t stream);

typedef to_t_cuda_t<float> to_fp32_cuda_t;
typedef to_t_cuda_t<half> to_fp16_cuda_t;
typedef to_t_cuda_t<__nv_bfloat16> to_bf16_cuda_t;

to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type);
to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type);
to_bf16_cuda_t ggml_get_to_bf16_cuda(ggml_type type);

inline bool FastllmGGUFIsR4Type(ggml_type type) {
    return type == GGML_TYPE_Q2_K_R4 || type == GGML_TYPE_Q4_K_R4 ||
           type == GGML_TYPE_Q5_K_R4 || type == GGML_TYPE_Q6_K_R4;
}

inline size_t FastllmGGUFAlignBytes(size_t bytes) {
    const size_t align = 256;
    return ((bytes + align - 1) / align) * align;
}

inline const char *FastllmGGUFWeightDisplayName(const fastllm::Data &weight) {
    return weight.name.empty() ? "(unnamed)" : weight.name.c_str();
}

inline int FastllmGGUFDequantRowGroup(ggml_type type) {
    return FastllmGGUFIsR4Type(type) ? 4 : 1;
}

inline int FastllmGGUFCalcChunkRows(size_t workspaceBytes, int m, int k,
                                    size_t bytesPerElement, size_t extraBytesPerElement,
                                    int rowGroup,
                                    const fastllm::Data &weight,
                                    const char *context) {
    if (m <= 0 || k <= 0) {
        return 0;
    }
    if (rowGroup > 1 && k % rowGroup != 0) {
        fastllm::ErrorInFastLLM(
                "Fastllm GGUF CUDA " + std::string(context) +
                " requires output rows aligned to " + std::to_string(rowGroup) +
                " for R4 dequant, got rows = " + std::to_string(k) +
                ", weight = " + FastllmGGUFWeightDisplayName(weight) + ".\n");
    }

    size_t bytesPerRow = (size_t)m * (bytesPerElement + extraBytesPerElement);
    int rows = bytesPerRow == 0 ? 0 : (int)std::min<size_t>((size_t)k, workspaceBytes / bytesPerRow);
    if (rowGroup > 1) {
        rows = (rows / rowGroup) * rowGroup;
    }

    while (rows > 0) {
        size_t mainBytes = (size_t)rows * m * bytesPerElement;
        size_t totalBytes = FastllmGGUFAlignBytes(mainBytes) +
                            (size_t)rows * m * extraBytesPerElement;
        if (totalBytes <= workspaceBytes) {
            return rows;
        }
        rows -= rowGroup;
    }

    size_t minRows = rowGroup;
    size_t minMainBytes = minRows * (size_t)m * bytesPerElement;
    size_t minBytes = FastllmGGUFAlignBytes(minMainBytes) +
                      minRows * (size_t)m * extraBytesPerElement;
    fastllm::ErrorInFastLLM(
            "Fastllm GGUF CUDA " + std::string(context) +
            " dequant workspace is too small, need at least " +
            std::to_string(minBytes) + " bytes, got " +
            std::to_string(workspaceBytes) + " bytes, weight = " +
            FastllmGGUFWeightDisplayName(weight) + ".\n");
    return 0;
}

// The caller owns the workspace and serializes its use on this stream.
// Keeping it explicit also lets regression tests exercise small chunk sizes.
template<typename T>
void FastllmGGUFDequantGemm(const T *input, const fastllm::Data &weight,
                          T *output, int n, int m, int k,
                          void *workspace, size_t workspaceBytes,
                          to_t_cuda_t<T> dequant, cublasHandle_t handle,
                          cudaStream_t stream) {
    static_assert(std::is_same<T, half>::value ||
                  std::is_same<T, __nv_bfloat16>::value,
                  "GGUF dequant GEMM requires FP16 or BF16");
    constexpr bool fp16 = std::is_same<T, half>::value;
    using Scalar = typename std::conditional<fp16, half, float>::type;
    const Scalar alpha = static_cast<Scalar>(1.0f);
    const Scalar beta = static_cast<Scalar>(0.0f);
    constexpr cudaDataType_t dataType = fp16 ? CUDA_R_16F : CUDA_R_16BF;
    // Preserve the existing accumulation type; chunking can still change
    // cuBLAS reduction order and therefore is not guaranteed bitwise equal.
    constexpr cudaDataType_t computeType = fp16 ? CUDA_R_16F : CUDA_R_32F;
    const char *context = fp16 ? "FP16 GEMM" : "BF16 GEMM";
    const ggml_type type = static_cast<ggml_type>(weight.ggmlType);
    const int chunkRows = FastllmGGUFCalcChunkRows(
            workspaceBytes, m, k, sizeof(T), 0,
            FastllmGGUFDequantRowGroup(type), weight, context);
    const size_t srcRowBytes = ggml_row_size(type, m);
    T *dequantized = static_cast<T *>(workspace);

    for (int offset = 0; offset < k; offset += chunkRows) {
        const int rows = std::min(chunkRows, k - offset);
        dequant(static_cast<const char *>(weight.cudaData) + (size_t)offset * srcRowBytes,
                dequantized, rows, m, stream);
        // Each token retains the full k-column output stride across chunks.
        const cublasStatus_t status = cublasGemmEx(
                handle, CUBLAS_OP_T, CUBLAS_OP_N, rows, n, m,
                &alpha, dequantized, dataType, m, input, dataType, m,
                &beta, output + offset, dataType, k, computeType,
                CUBLAS_GEMM_DEFAULT);
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error(
                    "Fastllm GGUF CUDA " + std::string(context) +
                    " failed, cuBLAS status = " + std::to_string((int)status) +
                    ", weight = " + FastllmGGUFWeightDisplayName(weight) + ".\n");
        }
    }
}
