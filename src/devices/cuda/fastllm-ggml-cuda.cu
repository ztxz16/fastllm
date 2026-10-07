#include "fastllm-cuda-gguf-projections.h"
#include "fastllm-cuda-gguf-planar.h"
#include "fastllm-cuda-gguf-linear-add.h"
#include "fastllm-gguf-store.cuh"
//
// Created by huangyuyang on 8/6/25.
//
// This is the code compatible with GGUF in Fastllm
// Most code copy from
// https://github.com/ggml-org/llama.cpp
// https://github.com/ikawrakow/ik_llama.cpp

#include <cassert>
#include <set>
#include <cstdlib>
#include <string>

#include "fastllm-cuda.cuh"
#include "fastllm.h"
#include <cuda_bf16.h>

#include "fastllm-gguf-mmvq-common.cuh"
#include "fastllm-gguf-mmvq-dispatch.cuh"
#include "fastllm-gguf-dequant.cuh"
#if !defined(USE_ROCM)
#include "fastllm-gguf-iq3-gemv.cuh"
#endif

#define CUDA_QUANTIZE_BLOCK_SIZE     256

template <typename T, bool Permuted = false>
static __global__ void quantize_q8_1(const T * __restrict__ x, void * __restrict__ vy, const int64_t kx, const int64_t kx0_padded, int keyHeads = 0, int groups = 0, int headDim = 0) {
    const int64_t ix0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (ix0 >= kx0_padded) {
        return;
    }

    const int64_t ix1 = blockIdx.y;

    const int64_t i_padded = ix1*kx0_padded + ix0;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int64_t ib = i_padded / QK8_1; // block index
    const int64_t iqs = i_padded % QK8_1; // quant index

    int64_t source = ix0;
    if constexpr (Permuted) {
        const int head = ix0 / headDim;
        source = ((head % keyHeads) * groups + head / keyHeads) * headDim + ix0 % headDim;
    }
    const float xi = ix0 < kx ? (float)x[ix1*kx + source] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max(amax);
    sum = warp_reduce_sum(sum);

    const float d = amax / 127;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    reinterpret_cast<half&>(y[ib].ds.x) = d;
    reinterpret_cast<half&>(y[ib].ds.y) = sum;
}

template <typename T>
void quantize_row_q8_1_cuda(
    const T * x, void * vy, const int64_t kx0, const int64_t kx1, const int64_t channels,
    const int64_t kx0_padded, const ggml_type type_x, cudaStream_t stream) {

    assert(kx0_padded % QK8_1 == 0);

    const int64_t block_num_x = (kx0_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, kx1*channels, 1);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(x, vy, kx0, kx0_padded);
}

bool get_has_vec_dot_q_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q2_K   : return true;
        case GGML_TYPE_Q3_K   : return true;
        case GGML_TYPE_IQ3_XXS: return true;
        case GGML_TYPE_IQ3_S  : return true;
        case GGML_TYPE_Q4_K   : return true;
        case GGML_TYPE_IQ4_NL : return true;
        case GGML_TYPE_IQ4_XS : return true;
        case GGML_TYPE_Q5_0   : return true;
        case GGML_TYPE_Q5_1   : return true;
        case GGML_TYPE_Q5_K   : return true;
        case GGML_TYPE_Q6_K   : return true;
        case GGML_TYPE_Q8_0   : return true;
        default               : return false;
    }
}

// Only ordinary, block-aligned 2..8-row verification may bypass the model's
// dequant safety flag. Extended MMVQ must still succeed before fallback is
// skipped; unsupported types/layouts and larger prefill retain that fallback.
static bool FastllmGGUFSmallMmvqShape(ggml_type type, int n, int m, int k) {
    const int blockSize = ggml_blck_size(type);
    return n >= 2 && n <= MMVQ_MAX_BATCH_SIZE && m > 0 && k > 0 &&
           blockSize > 0 && m % blockSize == 0 && m % QK8_1 == 0;
}

struct ggml_backend_cuda_context {

};

template <typename OType>
static void ggml_cuda_op_mul_mat_vec_q_impl(ggml_backend_cuda_context & ctx, ggml_type type,
        const int64_t ne00, const int64_t ne0, const int64_t ne2,
        const int64_t nb02, const int64_t nb12, const int64_t nb2, const int64_t ids_nb0,
        const char * src0_dd_i, const char * src1_ddq_i, OType * dst_dd_i, const char * ids_data,
        const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
        const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t row_diff = row_high - row_low;

    /*
    int id = ggml_cuda_get_device();
    const int64_t nrows_dst = id == ctx.device ? ne0 : row_diff;
    */
    const int64_t nrows_dst = true ? ne0 : row_diff;

    switch (type) {
        case GGML_TYPE_Q2_K:
            mul_mat_vec_q_cuda<GGML_TYPE_Q2_K, OType>(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_vec_q_cuda<GGML_TYPE_Q3_K, OType>(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_vec_q_cuda<GGML_TYPE_IQ3_XXS, OType>(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_vec_q_cuda<GGML_TYPE_IQ3_S, OType>(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_vec_q_cuda<GGML_TYPE_Q4_K, OType>(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_vec_q_cuda<GGML_TYPE_IQ4_NL, OType>(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_vec_q_cuda<GGML_TYPE_IQ4_XS, OType>(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_vec_q_cuda<GGML_TYPE_Q5_0, OType>(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_vec_q_cuda<GGML_TYPE_Q5_1, OType>(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_vec_q_cuda<GGML_TYPE_Q5_K, OType>(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_vec_q_cuda<GGML_TYPE_Q6_K, OType>(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q_cuda<GGML_TYPE_Q8_0, OType>(src0_dd_i, src1_ddq_i, dst_dd_i, ids_data, ne00, row_diff, src1_padded_row_size, src1_ncols, nrows_dst, ne2, nb02, nb12, nb2, ids_nb0, stream);
            break;
        default:
            printf("Error: unsupport cuda linear type %s\n", ggml_type_name(type));
            exit(0);
            break;
    }
}

/*
void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];
    assert(ne10 % QK8_1 == 0);

    const int64_t ne0 = dst->ne[0];

    ggml_cuda_op_mul_mat_vec_q_impl(ctx, src0->type,
        ne00, ne0, 1, 0, 0, 0, 0,
        src0_dd_i, src1_ddq_i, dst_dd_i, nullptr,
        row_low, row_high, src1_ncols,
        src1_padded_row_size, stream);

    GGML_UNUSED(src1_ddf_i);
}
*/

static void *FastllmGGUFGetDequantWorkspace(size_t *workspaceBytes,
                                            const fastllm::Data &weight,
                                            const char *context) {
    void *workspace = FastllmCudaGetFlashInferFloatWorkspace(workspaceBytes);
    if (workspace == nullptr || workspaceBytes == nullptr || *workspaceBytes == 0) {
        fastllm::ErrorInFastLLM(
                "Fastllm GGUF CUDA " + std::string(context) +
                " failed to get dequant workspace, weight = " +
                FastllmGGUFWeightDisplayName(weight) + ".\n");
    }
    return workspace;
}

bool FastllmCudaGGUFPrefillSupported(fastllm::DataType inputType, int weightType) {
    const auto type = static_cast<ggml_type>(weightType);
    if (inputType == fastllm::BFLOAT16) return ggml_get_to_bf16_cuda(type) != nullptr;
    if (inputType == fastllm::FLOAT16 || inputType == fastllm::FLOAT32)
        return ggml_get_to_fp16_cuda(type) != nullptr;
    return false;
}

bool FastllmCudaMatMulFloatGGUF(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    if ((ggml_type)weight.ggmlType == GGML_TYPE_BF16) {
        return FastllmCudaMatMulBFloat16(
            input, weight, bias, output, n, m, k);
    }
    if (weight.cudaData == nullptr || weight.extraCudaData.size() == 0) {
        float *cudaBiasData;
        auto state = cudaMalloc(&cudaBiasData, k * sizeof(float));
        if (bias.dims.size() > 0) {
            state = cudaMemcpy(cudaBiasData, (uint8_t*)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(float));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaData.push_back((void*)cudaBiasData);
    }

    float *cudaBiasData = (float*)weight.extraCudaData[0];

    float *cudaInput = (float*)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float*)FastllmCudaPrepareOutput(output);
    block_q8_1 * q8Input = nullptr;

    ggml_backend_cuda_context ctx;

    ggml_type ggufType = (ggml_type)weight.ggmlType;
    const bool allowSmallMmvq = weight.forceGGUFFp32Dequant &&
        FastllmGGUFSmallMmvqShape(ggufType, n, m, k);
    const bool forceFp32Dequant = n != 1 && weight.forceGGUFFp32Dequant &&
        !(allowSmallMmvq && get_has_vec_dot_q_cuda(ggufType));
    auto dequantFp32 = weight.forceGGUFFp32Dequant ? ggml_get_to_fp32_cuda(ggufType) : nullptr;
    auto dequantFp16 = ggml_get_to_fp16_cuda(ggufType);
    auto has_vec_dot = get_has_vec_dot_q_cuda((ggml_type)weight.ggmlType);
    cudaStream_t stream = cudaStreamPerThread;
    const bool usedMmq = !forceFp32Dequant && !allowSmallMmvq &&
        FastllmCudaFloatMatMulGGUFMMQ(
            cudaInput, weight.cudaData, cudaOutput, weight.ggmlType,
            n, m, k, stream);
    // has_vec_dot covers only the legacy dispatcher. Try extended MMVQ
    // before the direct fallback so IQ2/IQ1 and Q4_0/Q4_1 are not shadowed.
    const bool usedExtendedMmvq = !usedMmq && (!forceFp32Dequant || allowSmallMmvq) &&
        FastllmCudaFloatMatMulGGUFMMVQ(
            cudaInput, weight.cudaData, cudaOutput, weight.ggmlType,
            n, m, k, stream);
    const bool usedDirectGemv = n == 1 && !usedMmq && !usedExtendedMmvq && !has_vec_dot &&
        FastllmGgufDirectGemv(cudaInput, weight.cudaData, cudaOutput,
                             ggufType, m, k, stream);

    if (!usedDirectGemv && !usedMmq && !usedExtendedMmvq &&
        (forceFp32Dequant || n > MMVQ_MAX_BATCH_SIZE || !has_vec_dot) &&
        dequantFp32 != nullptr) {
        auto fastllmCublasHandle = getFastllmCublasHandle();

        size_t wsBytes = 0;
        float *cudaFp32Weight = (float *) FastllmGGUFGetDequantWorkspace(
                &wsBytes, weight, "FP32 SGEMM");
        int rowGroup = FastllmGGUFDequantRowGroup(ggufType);
        int maxRowsPerChunk = FastllmGGUFCalcChunkRows(
                wsBytes, m, k, sizeof(float), 0, rowGroup, weight, "FP32 SGEMM");
        size_t srcRowBytes = ggml_row_size(ggufType, m);

        cublasStatus_t status;

        float h_alpha = 1.0f, h_beta = 0.0f;
        for (int kOff = 0; kOff < k; kOff += maxRowsPerChunk) {
            int kc = std::min(maxRowsPerChunk, k - kOff);
            dequantFp32((const char *)weight.cudaData + (size_t)kOff * srcRowBytes,
                        cudaFp32Weight, kc, m, stream);

            status = cublasSgemm(
                    fastllmCublasHandle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    kc, n, m,
                    &h_alpha, cudaFp32Weight,
                    m, cudaInput,
                    m, &h_beta,
                    cudaOutput + kOff,
                    k);
            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("Error: cublas error.\n");
                throw("cublas error");
            }
        }
    } else if (!usedDirectGemv && !usedMmq && !usedExtendedMmvq &&
               (n > MMVQ_MAX_BATCH_SIZE || !has_vec_dot) &&
               dequantFp16 != nullptr) {
        auto fastllmCublasHandle = getFastllmCublasHandle();

        size_t wsBytes = 0;
        uint8_t *workspace = (uint8_t *) FastllmGGUFGetDequantWorkspace(
                &wsBytes, weight, "FP16-to-FP32 SGEMM");
        int rowGroup = FastllmGGUFDequantRowGroup(ggufType);
        int maxRowsPerChunk = FastllmGGUFCalcChunkRows(
                wsBytes, m, k, sizeof(half), sizeof(float), rowGroup,
                weight, "FP16-to-FP32 SGEMM");
        size_t srcRowBytes = ggml_row_size(ggufType, m);

        cublasStatus_t status;

        float h_alpha = 1.0f, h_beta = 0.0f;
        for (int kOff = 0; kOff < k; kOff += maxRowsPerChunk) {
            int kc = std::min(maxRowsPerChunk, k - kOff);
            size_t fp16Bytes = (size_t)kc * m * sizeof(half);
            half *cudaFp16Weight = (half *)workspace;
            float *cudaFp32Weight = (float *)(workspace + FastllmGGUFAlignBytes(fp16Bytes));

            int len = kc * m;
            int threadPerBlock = std::min(256, len);
            dequantFp16((const char *)weight.cudaData + (size_t)kOff * srcRowBytes,
                        cudaFp16Weight, kc, m, stream);
            FastllmCudaHalf2FloatKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock, 0, stream >>>(
                    cudaFp16Weight, cudaFp32Weight, len);

            status = cublasSgemm(
                    fastllmCublasHandle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    kc, n, m,
                    &h_alpha, cudaFp32Weight,
                    m, cudaInput,
                    m, &h_beta,
                    cudaOutput + kOff,
                    k);
            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("Error: cublas error.\n");
                throw("cublas error");
            }
        }
    } else if (!usedDirectGemv && !usedMmq && !usedExtendedMmvq) {
        q8Input = (block_q8_1*)FastllmCudaMalloc(n * m * sizeof(half));
        quantize_row_q8_1_cuda (
            cudaInput, q8Input, m, n, 1, m, GGML_TYPE_Q8_1, stream
        );
    }

    if (q8Input != nullptr && n > 1) {
        int i = 0;
        for (; i + MMVQ_MAX_BATCH_SIZE - 1 < n; i += MMVQ_MAX_BATCH_SIZE) {
            ggml_cuda_op_mul_mat_vec_q_impl (
                    ctx, (ggml_type)weight.ggmlType, m, k, 1,
                    0, 0, 0, 0,
                    (char*)weight.cudaData,
                    (char*)(q8Input + i * (m / QK8_1)),
                    cudaOutput + i * k,
                    nullptr,
                    0, k, MMVQ_MAX_BATCH_SIZE, m, stream
            );
        }

        if (n - i > 0) {
            ggml_cuda_op_mul_mat_vec_q_impl (
                    ctx, (ggml_type)weight.ggmlType, m, k, 1,
                    0, 0, 0, 0,
                    (char*)weight.cudaData,
                    (char*)(q8Input + i * (m / QK8_1)),
                    cudaOutput + i * k,
                    nullptr,
                    0, k, n - i, m, stream
            );
        }
    } else if (q8Input != nullptr) {
        ggml_cuda_op_mul_mat_vec_q_impl (
                ctx, (ggml_type)weight.ggmlType, m, k, 1,
                0, 0, 0, 0,
                (char*)weight.cudaData, (char*)q8Input, cudaOutput, nullptr,
                0, k, n, m, stream
        );
    }
    if (bias.dims.size() > 0) {
        FastllmCudaBiasKernel <<< n, 256, 0, stream >>> (cudaOutput, cudaBiasData, k);
    }

    if (q8Input != nullptr) {
        FastllmCudaFree(q8Input);
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

namespace {
static bool GgufDenseLocal(const fastllm::Data &data) {
    using namespace fastllm;
    const int device = FastllmCudaGetDevice();
    if (data.dataDevice != DataDevice::CUDA || !data.cudaData || data.multiDeviceData ||
        data.dims.empty() || data.strides.size() != data.dims.size() ||
        (!data.dataDeviceIds.empty() &&
         (data.dataDeviceIds.size() != 1 || data.dataDeviceIds[0] != device))) return false;
    uint64_t stride = 1;
    for (int i = int(data.dims.size())-1; i >= 0; --i) {
        if (data.dims[i] <= 0 || data.strides[i] != stride) return false;
        stride *= data.dims[i];
    }
    return true;
}
static bool GgufOverlap(const fastllm::Data &a, const fastllm::Data &b) {
    const auto x = reinterpret_cast<uintptr_t>(a.cudaData);
    const auto y = reinterpret_cast<uintptr_t>(b.cudaData);
    return x < y+b.GetBytes() && y < x+a.GetBytes();
}
static bool GgufSharedProjectionCanRun(const fastllm::Data &input,
        const fastllm::Data &weight, const fastllm::Data &output, bool allowBfloat16 = false) {
    using namespace fastllm;
    const bool bfloat16 = allowBfloat16 && input.dataType == BFLOAT16;
    if ((!bfloat16 && input.dataType != FLOAT16) || output.dataType != input.dataType ||
        weight.dataType != DATA_GGUF_FORMAT || weight.dims.size() != 2 ||
        !GgufDenseLocal(input) || !GgufDenseLocal(output) ||
        weight.dataDevice != DataDevice::CUDA || !weight.cudaData || weight.multiDeviceData ||
        (!weight.dataDeviceIds.empty() && (weight.dataDeviceIds.size() != 1 ||
            weight.dataDeviceIds[0] != FastllmCudaGetDevice()))) return false;
    const int columns = input.dims.back(), rows = input.Count(0)/columns, outputs = weight.dims[0];
    if (columns % 256 || rows < 1 || rows > 8 || outputs < 1 || weight.dims[1] != columns ||
        output.dims.back() != outputs || output.Count(0) != size_t(rows)*outputs ||
        GgufOverlap(input, output) || GgufOverlap(weight, output)) return false;
    const auto type = static_cast<ggml_type>(weight.ggmlType);
    if (bfloat16) {
        // Match the ordinary BF16 MMVQ dispatch, including its reduction
        // order. Extended IQ kernels and large-batch MMQ keep their fallback.
        if (!get_has_vec_dot_q_cuda(type)) return false;
    } else {
        switch (type) {
            case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ4_XS:
            case GGML_TYPE_Q4_K: case GGML_TYPE_Q2_K:
            case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S:
            case GGML_TYPE_IQ1_M: break;
            default: return false;
        }
    }
    const auto *tensor = static_cast<const ggml_tensor *>(weight.ggmlTensor);
    if (!tensor || tensor->type != type || tensor->ne[0] != columns || tensor->ne[1] != outputs ||
        tensor->nb[0] != ggml_type_size(type) || tensor->nb[1] != ggml_row_size(type, columns)) return false;
    if (rows == 8 && !weight.forceGGUFFp32Dequant) {
        int major = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, FastllmCudaGetDevice()) != cudaSuccess ||
            major >= 10) return false;
    }
    return true;
}

} // namespace

static bool GgufPlanarProjectionShape(int type, int tokens, int columns, int outputs) {
#if defined(USE_ROCM)
    return false;
#else
    // Use one arithmetic path for plain and fused projections at 1..8 rows.
    // Larger batches keep the established MMQ selection.
    return tokens >= 1 && tokens <= 8 &&
        FastllmGgufPlanarSupported(type, tokens, columns, outputs);
#endif
}

static bool GgufTryPlanarProjection(const void *input, const void *weight, void *output,
        int type, int tokens, int columns, int outputs, int mode,
        int keyHeads = 0, int groups = 0, int headDim = 0) {
#if defined(USE_ROCM)
    return false;
#else
    if (!GgufPlanarProjectionShape(type, tokens, columns, outputs)) return false;
    void *workspace = nullptr;
    if (FastllmCudaTryMalloc(&workspace, FastllmGgufPlanarBytes(tokens, columns)) !=
            FASTLLM_CUDA_TRY_MALLOC_SUCCESS) return false;
    const auto stream = cudaStreamPerThread;
    const bool quantized = FastllmGgufQuantizePlanar(input, 1, workspace, tokens, columns,
        stream, keyHeads, groups, headDim);
    const bool projected = quantized && FastllmGgufProjectPlanar(type, mode, weight, nullptr,
        workspace, output, tokens, columns, outputs, outputs, stream);
    FastllmCudaFree(workspace);
    return projected;
#endif
}

static bool FastllmGGUFLinearAddImpl(const fastllm::Data &input, fastllm::Data &weight,
        const fastllm::Data &bias, fastllm::Data &output, int keyHeads, int valueHeads, int headDim) {
    using namespace fastllm;
    if (!bias.dims.empty() || !GgufSharedProjectionCanRun(input, weight, output)) return false;
    const int m = input.dims.back(), k = weight.dims[0], n = input.Count(0) / m;
    const bool permuted = keyHeads != 0;
    if (permuted && (keyHeads < 1 || valueHeads < keyHeads || valueHeads % keyHeads ||
        headDim < 32 || headDim % 32 || int64_t(valueHeads) * headDim != m)) return false;
    const ggml_type type = static_cast<ggml_type>(weight.ggmlType);
    switch (type) {
        case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q2_K:
        case GGML_TYPE_IQ2_S: case GGML_TYPE_IQ2_XS: break;
        default: return false;
    }
    cudaStream_t stream = cudaStreamPerThread;
    if (GgufTryPlanarProjection(input.cudaData, weight.cudaData, output.cudaData,
            type, n, m, k, 1, keyHeads, permuted ? valueHeads/keyHeads : 0, headDim)) return true;
    if (!permuted && (type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ2_XS)) {
        return FastllmCudaHalfMatMulGGUFMMVQAddTo(input.cudaData, weight.cudaData,
            output.cudaData, weight.ggmlType, n, m, k, stream);
    }
    block_q8_1 *quantized = nullptr;
    if (FastllmCudaTryMalloc(reinterpret_cast<void **>(&quantized),
                            size_t(n)*(m/QK8_1)*sizeof(block_q8_1)) !=
        FASTLLM_CUDA_TRY_MALLOC_SUCCESS) return false;
    if (permuted) {
        quantize_q8_1<half, true><<<dim3((m+255)/256, n), 256, 0, stream>>>(
            static_cast<const half *>(input.cudaData), quantized, m, m,
            keyHeads, valueHeads/keyHeads, headDim);
    } else {
        quantize_row_q8_1_cuda(static_cast<const half *>(input.cudaData), quantized,
                              m, n, 1, m, GGML_TYPE_Q8_1, stream);
    }
#define FASTLLM_GGUF_ADD_CASE(TYPE) \
    case TYPE: mul_mat_vec_q_cuda<TYPE, half, true>(weight.cudaData, quantized, \
        static_cast<half *>(output.cudaData), nullptr, m, k, m, n, k, \
        1, 0, 0, 0, 0, stream); break
    switch (type) {
        FASTLLM_GGUF_ADD_CASE(GGML_TYPE_IQ3_S);
        FASTLLM_GGUF_ADD_CASE(GGML_TYPE_IQ3_XXS);
        FASTLLM_GGUF_ADD_CASE(GGML_TYPE_IQ4_XS);
        FASTLLM_GGUF_ADD_CASE(GGML_TYPE_Q4_K);
        FASTLLM_GGUF_ADD_CASE(GGML_TYPE_Q2_K);
        case GGML_TYPE_IQ2_S: case GGML_TYPE_IQ2_XS:
            FastllmCudaGGUFExtendedFromQ8(quantized, weight.cudaData, output.cudaData,
                type, n, m, k, 1, stream); break;
        default: break; // checked before allocating or writing
    }
#undef FASTLLM_GGUF_ADD_CASE
    FastllmCudaFree(quantized);
    return true;
}

bool FastllmCudaGGUFLinearAdd(const fastllm::Data &input, fastllm::Data &weight,
        const fastllm::Data &bias, fastllm::Data &output) {
    return FastllmGGUFLinearAddImpl(input, weight, bias, output, 0, 0, 0);
}

bool FastllmCudaGGUFLinearAddPermuted(const fastllm::Data &input,
        fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output,
        int keyHeads, int valueHeads, int headDim) {
    if (keyHeads <= 0) return false;
    return FastllmGGUFLinearAddImpl(input, weight, bias, output, keyHeads, valueHeads, headDim);
}

bool FastllmCudaHalfMatMulGGUF(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    if ((ggml_type)weight.ggmlType == GGML_TYPE_BF16) {
        return FastllmCudaHalfMatMulBFloat16(
            input, weight, bias, output, n, m, k);
    }
    if (weight.cudaData == nullptr ||
        (weight.extraCudaHalfData.size() == 0 && bias.dims.size() > 0)) {
        half *cudaBiasData;
        cudaError_t state = cudaSuccess;
        state = cudaMalloc(&cudaBiasData, k * sizeof(half));
        if (bias.dims.size() > 0) {
            float *tempBiasData;
            state = cudaMalloc(&tempBiasData, k * sizeof(float));
            state = cudaMemcpy(tempBiasData, (uint8_t *) bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
            int threadPerBlock = std::min(256, k);
            FastllmCudaFloat2HalfKernel <<< (k - 1) / threadPerBlock + 1, threadPerBlock>>>(tempBiasData, cudaBiasData, k);
            state = cudaFree(tempBiasData);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(half));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaHalfData.push_back((void *) cudaBiasData);
    }

    // float *cudaBiasData = (float*)weight.extraCudaData[0];
    half *cudaBiasData = bias.dims.size() == 0 ? nullptr : (half *) weight.extraCudaHalfData[0];
    half *cudaInput = (half*)FastllmCudaPrepareInput(input);
    half *cudaOutput = (half*)FastllmCudaPrepareOutput(output);
    block_q8_1 * q8Input = nullptr;

    ggml_backend_cuda_context ctx;

    const ggml_type ggufType = (ggml_type)weight.ggmlType;
    const bool allowSmallMmvq = weight.forceGGUFFp32Dequant &&
        FastllmGGUFSmallMmvqShape(ggufType, n, m, k);
    const bool forceDequant = n != 1 && weight.forceGGUFFp32Dequant &&
        !(allowSmallMmvq && get_has_vec_dot_q_cuda(ggufType));
    auto dequant = ggml_get_to_fp16_cuda(ggufType);
    auto has_vec_dot = get_has_vec_dot_q_cuda(ggufType);
    cudaStream_t stream = cudaStreamPerThread;
    const bool usedMmq = !forceDequant && !allowSmallMmvq &&
        FastllmCudaHalfMatMulGGUFMMQ(
            cudaInput, weight.cudaData, cudaOutput, weight.ggmlType,
            n, m, k, stream);
    const bool usedPlanar = (!forceDequant || allowSmallMmvq) && !usedMmq &&
        GgufTryPlanarProjection(cudaInput, weight.cudaData, cudaOutput,
            weight.ggmlType, n, m, k, 0);
    const bool usedExtendedMmvq = (!forceDequant || allowSmallMmvq) && !usedMmq && !usedPlanar &&
        FastllmCudaHalfMatMulGGUFMMVQ(
            cudaInput, weight.cudaData, cudaOutput, weight.ggmlType,
            n, m, k, stream);

    const bool usedDirectGemv = n == 1 && !usedMmq && !usedPlanar && !usedExtendedMmvq && !has_vec_dot &&
        FastllmGgufDirectGemv(cudaInput, weight.cudaData, cudaOutput,
                             ggufType, m, k, stream);

    if (!usedDirectGemv && !usedMmq && !usedPlanar && !usedExtendedMmvq &&
        (forceDequant || n > MMVQ_MAX_BATCH_SIZE || !has_vec_dot) &&
        dequant != nullptr) {
        auto handle = getFastllmCublasHandle();
        size_t workspaceBytes = 0;
        void *workspace = FastllmGGUFGetDequantWorkspace(
                &workspaceBytes, weight, "FP16 GEMM");
        FastllmGGUFDequantGemm(
                cudaInput, weight, cudaOutput, n, m, k,
                workspace, workspaceBytes, dequant, handle, stream);
    } else if (!usedDirectGemv && !usedMmq && !usedPlanar && !usedExtendedMmvq) {
        q8Input = (block_q8_1*)FastllmCudaMalloc(n * m * sizeof(half));
        quantize_row_q8_1_cuda (
            cudaInput, q8Input, m, n, 1, m, GGML_TYPE_Q8_1, stream
        );
    }

    if (q8Input != nullptr && n > 1) {
        int i = 0;
        for (; i + MMVQ_MAX_BATCH_SIZE - 1 < n; i += MMVQ_MAX_BATCH_SIZE) {
            ggml_cuda_op_mul_mat_vec_q_impl (
                    ctx, (ggml_type)weight.ggmlType, m, k, 1,
                    0, 0, 0, 0,
                    (char*)weight.cudaData,
                    (char*)(q8Input + i * (m / QK8_1)),
                    cudaOutput + i * k,
                    nullptr,
                    0, k, MMVQ_MAX_BATCH_SIZE, m, stream
            );
        }

        if (n - i > 0) {
            ggml_cuda_op_mul_mat_vec_q_impl (
                    ctx, (ggml_type)weight.ggmlType, m, k, 1,
                    0, 0, 0, 0,
                    (char*)weight.cudaData,
                    (char*)(q8Input + i * (m / QK8_1)),
                    cudaOutput + i * k,
                    nullptr,
                    0, k, n - i, m, stream
            );
        }
    } else if (q8Input != nullptr) {
        ggml_cuda_op_mul_mat_vec_q_impl (
                ctx, (ggml_type)weight.ggmlType, m, k, 1,
                0, 0, 0, 0,
                (char*)weight.cudaData, (char*)q8Input, cudaOutput, nullptr,
                0, k, n, m, stream
        );
    }
    if (bias.dims.size() > 0) {
        FastllmCudaBiasKernel <<< n, 256, 0, stream >>> (cudaOutput, cudaBiasData, k);
    }

    if (q8Input != nullptr) {
        FastllmCudaFree(q8Input);
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);

    return true;
}

static __global__ void FastllmGgufHalfSiluMulKernel(
        half *gate, const half *up, int len) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) {
        return;
    }
    gate[index] = FastllmGgufHalfSiluMulValue(gate[index], up[index]);
}

template <ggml_type type, int nwarps>
#if !defined(USE_ROCM)
__launch_bounds__(nwarps * WARP_SIZE, 1)
#endif
static __global__ void FastllmGgufFusedGateUpMmvqKernel(
        const void * __restrict__ gateWeight,
        const void * __restrict__ upWeight,
        const block_q8_1 * __restrict__ input,
        half * __restrict__ output,
        int inputColumns, int outputRows) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int qi = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = get_vdr_mmvq(type);
    constexpr vec_dot_q_cuda_t vecDot = get_vec_dot_q_cuda(type);
    constexpr int blocksPerIteration =
        vdr * nwarps * WARP_SIZE / qi;

    const int row = blockIdx.x;
    const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
    const int weightBlocksPerRow = inputColumns / qk;
    float gateSum = 0.0f;
    float upSum = 0.0f;

    for (int weightBlock = tid / (qi / vdr);
         weightBlock < weightBlocksPerRow;
         weightBlock += blocksPerIteration) {
        const int inputBlock = weightBlock * (qk / QK8_1);
        const int quantIndex = vdr * (tid % (qi / vdr));
        gateSum += vecDot(
            gateWeight, &input[inputBlock],
            row * weightBlocksPerRow + weightBlock, quantIndex);
        upSum += vecDot(
            upWeight, &input[inputBlock],
            row * weightBlocksPerRow + weightBlock, quantIndex);
    }

    __shared__ float gateShared[
        nwarps - 1 > 0 ? nwarps - 1 : 1][WARP_SIZE];
    __shared__ float upShared[
        nwarps - 1 > 0 ? nwarps - 1 : 1][WARP_SIZE];
    if (threadIdx.y > 0) {
        gateShared[threadIdx.y - 1][threadIdx.x] = gateSum;
        upShared[threadIdx.y - 1][threadIdx.x] = upSum;
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

#pragma unroll
    for (int warp = 0; warp < nwarps - 1; ++warp) {
        gateSum += gateShared[warp][threadIdx.x];
        upSum += upShared[warp][threadIdx.x];
    }
    gateSum = warp_reduce_sum(gateSum);
    upSum = warp_reduce_sum(upSum);
    if (threadIdx.x == 0 && row < outputRows) {
        output[row] = FastllmGgufHalfSiluMulValue(
            (half)gateSum, (half)upSum);
    }
}

template <ggml_type type>
static void FastllmLaunchGgufFusedGateUpMmvq(
        const void *gateWeight, const void *upWeight,
        const block_q8_1 *input, half *output,
        int inputColumns, int outputRows, cudaStream_t stream) {
#if !defined(USE_ROCM)
    if constexpr (type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ3_XXS) {
        if (fastllm_gguf_iq3::Supports(input, inputColumns, outputRows)) {
            fastllm_gguf_iq3::Launch<type, true>(
                gateWeight, upWeight, input, output, inputColumns, outputRows, stream);
            return;
        }
    }
#endif
    constexpr int nwarps = 4;
    FastllmGgufFusedGateUpMmvqKernel<type, nwarps><<<
        outputRows, dim3(WARP_SIZE, nwarps, 1), 0, stream>>>(
            gateWeight, upWeight, input, output,
            inputColumns, outputRows);
}

static bool FastllmDispatchGgufFusedGateUpMmvq(
        ggml_type type,
        const void *gateWeight, const void *upWeight,
        const block_q8_1 *input, half *output,
        int inputColumns, int outputRows, cudaStream_t stream) {
    switch (type) {
        case GGML_TYPE_Q2_K:
            FastllmLaunchGgufFusedGateUpMmvq<GGML_TYPE_Q2_K>(
                gateWeight, upWeight, input, output,
                inputColumns, outputRows, stream);
            return true;
        case GGML_TYPE_Q3_K:
            FastllmLaunchGgufFusedGateUpMmvq<GGML_TYPE_Q3_K>(
                gateWeight, upWeight, input, output,
                inputColumns, outputRows, stream);
            return true;
        case GGML_TYPE_IQ3_XXS:
            FastllmLaunchGgufFusedGateUpMmvq<GGML_TYPE_IQ3_XXS>(
                gateWeight, upWeight, input, output,
                inputColumns, outputRows, stream);
            return true;
        case GGML_TYPE_IQ3_S:
            FastllmLaunchGgufFusedGateUpMmvq<GGML_TYPE_IQ3_S>(
                gateWeight, upWeight, input, output,
                inputColumns, outputRows, stream);
            return true;
        case GGML_TYPE_Q4_K:
            FastllmLaunchGgufFusedGateUpMmvq<GGML_TYPE_Q4_K>(
                gateWeight, upWeight, input, output,
                inputColumns, outputRows, stream);
            return true;
        case GGML_TYPE_IQ4_NL:
            FastllmLaunchGgufFusedGateUpMmvq<GGML_TYPE_IQ4_NL>(
                gateWeight, upWeight, input, output,
                inputColumns, outputRows, stream);
            return true;
        case GGML_TYPE_IQ4_XS:
            FastllmLaunchGgufFusedGateUpMmvq<GGML_TYPE_IQ4_XS>(
                gateWeight, upWeight, input, output,
                inputColumns, outputRows, stream);
            return true;
        case GGML_TYPE_Q5_0:
            FastllmLaunchGgufFusedGateUpMmvq<GGML_TYPE_Q5_0>(
                gateWeight, upWeight, input, output,
                inputColumns, outputRows, stream);
            return true;
        case GGML_TYPE_Q5_1:
            FastllmLaunchGgufFusedGateUpMmvq<GGML_TYPE_Q5_1>(
                gateWeight, upWeight, input, output,
                inputColumns, outputRows, stream);
            return true;
        case GGML_TYPE_Q5_K:
            FastllmLaunchGgufFusedGateUpMmvq<GGML_TYPE_Q5_K>(
                gateWeight, upWeight, input, output,
                inputColumns, outputRows, stream);
            return true;
        case GGML_TYPE_Q6_K:
            FastllmLaunchGgufFusedGateUpMmvq<GGML_TYPE_Q6_K>(
                gateWeight, upWeight, input, output,
                inputColumns, outputRows, stream);
            return true;
        case GGML_TYPE_Q8_0:
            FastllmLaunchGgufFusedGateUpMmvq<GGML_TYPE_Q8_0>(
                gateWeight, upWeight, input, output,
                inputColumns, outputRows, stream);
            return true;
        default:
            return false;
    }
}

namespace {
template<int StoreMode>
static void GgufProjectFromQ8(const block_q8_1 *input, const fastllm::Data &weight,
        half *output, int rows, int columns, cudaStream_t stream) {
    const int outputs = weight.dims[0];
#define PROJECT_CASE(TYPE) \
    case TYPE: mul_mat_vec_q_cuda<TYPE, half, StoreMode>(weight.cudaData, input, \
        output, nullptr, columns, outputs, columns, rows, outputs, \
        1, 0, 0, 0, 0, stream); break
    switch (static_cast<ggml_type>(weight.ggmlType)) {
        PROJECT_CASE(GGML_TYPE_IQ3_S); PROJECT_CASE(GGML_TYPE_IQ3_XXS);
        PROJECT_CASE(GGML_TYPE_IQ4_XS); PROJECT_CASE(GGML_TYPE_Q4_K); PROJECT_CASE(GGML_TYPE_Q2_K);
        case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S: case GGML_TYPE_IQ1_M:
            FastllmCudaGGUFExtendedFromQ8(input, weight.cudaData, output,
                weight.ggmlType, rows, columns, outputs, StoreMode, stream); break;
        default: break; // validated before allocating or launching
    }
#undef PROJECT_CASE
}
}

bool FastllmCudaGGUFLinearShared(const fastllm::Data &input,
        fastllm::Data *const *weights, fastllm::Data *const *outputs, int count) {
    if (count < 2 || !weights || !outputs) return false;
    for (int i = 0; i < count; ++i) {
        if (!weights[i] || !outputs[i] || !GgufSharedProjectionCanRun(input, *weights[i], *outputs[i], true)) return false;
        for (int j = 0; j < count; ++j) {
            if (!weights[j] || GgufOverlap(*weights[j], *outputs[i])) return false;
            if (j < i && GgufOverlap(*outputs[i], *outputs[j])) return false;
        }
    }
    const int columns = input.dims.back(), rows = input.Count(0)/columns;
#if !defined(USE_ROCM)
    bool planar = input.dataType == fastllm::FLOAT16;
    for (int i = 0; i < count; ++i)
        planar = planar && GgufPlanarProjectionShape(weights[i]->ggmlType, rows, columns, weights[i]->dims[0]);
    if (planar) {
        void *workspace = nullptr;
        if (FastllmCudaTryMalloc(&workspace, FastllmGgufPlanarBytes(rows, columns)) ==
                FASTLLM_CUDA_TRY_MALLOC_SUCCESS) {
            const auto stream = cudaStreamPerThread;
            FastllmGgufQuantizePlanar(input.cudaData, 1, workspace, rows, columns, stream);
            for (int i = 0; i < count; ++i)
                FastllmGgufProjectPlanar(weights[i]->ggmlType, 0, weights[i]->cudaData, nullptr,
                    workspace, outputs[i]->cudaData, rows, columns, weights[i]->dims[0], weights[i]->dims[0], stream);
            FastllmCudaFree(workspace);
            return true;
        }
    }
#endif
    block_q8_1 *q8 = nullptr;
    if (FastllmCudaTryMalloc(reinterpret_cast<void **>(&q8),
            size_t(rows)*(columns/QK8_1)*sizeof(block_q8_1)) != FASTLLM_CUDA_TRY_MALLOC_SUCCESS) return false;
    const auto stream = cudaStreamPerThread;
    if (input.dataType == fastllm::BFLOAT16) {
        quantize_row_q8_1_cuda(static_cast<const __nv_bfloat16 *>(input.cudaData), q8,
            columns, rows, 1, columns, GGML_TYPE_Q8_1, stream);
        ggml_backend_cuda_context ctx;
        for (int i = 0; i < count; ++i) {
            const int k = weights[i]->dims[0];
            ggml_cuda_op_mul_mat_vec_q_impl(ctx, (ggml_type)weights[i]->ggmlType,
                columns, k, 1, 0, 0, 0, 0,
                (const char *)weights[i]->cudaData, (const char *)q8,
                static_cast<__nv_bfloat16 *>(outputs[i]->cudaData), nullptr,
                0, k, rows, columns, stream);
        }
    } else {
        quantize_row_q8_1_cuda(static_cast<const half *>(input.cudaData), q8,
            columns, rows, 1, columns, GGML_TYPE_Q8_1, stream);
        for (int i = 0; i < count; ++i)
            GgufProjectFromQ8<0>(q8, *weights[i], static_cast<half *>(outputs[i]->cudaData), rows, columns, stream);
    }
    FastllmCudaFree(q8);
    return true;
}

// Share one quantization and one projection launch, including mixed formats.
static bool GgufTryPlanarGateUp(const void *input, const void *gate, const void *up,
        void *output, int gateType, int upType, int rows, int columns, int outputs) {
#if defined(USE_ROCM)
    return false;
#else
    if (!GgufPlanarProjectionShape(gateType, rows, columns, outputs) ||
        !GgufPlanarProjectionShape(upType, rows, columns, outputs)) return false;
    void *workspace = nullptr;
    if (FastllmCudaTryMalloc(&workspace, FastllmGgufPlanarBytes(rows, columns)) !=
        FASTLLM_CUDA_TRY_MALLOC_SUCCESS) return false;
    const auto stream = cudaStreamPerThread;
    const bool quantized = FastllmGgufQuantizePlanar(input, 1, workspace, rows, columns, stream);
    const bool projected = quantized && FastllmGgufGateUpPlanar(gateType, upType, gate, up, workspace,
        output, rows, columns, outputs, outputs, stream);
    FastllmCudaFree(workspace);
    return projected;
#endif
}

bool FastllmCudaGGUFMixedGateUp(const fastllm::Data &input,
        fastllm::Data &gate, fastllm::Data &up, fastllm::Data &output) {
    if (gate.dims != up.dims ||
        !GgufSharedProjectionCanRun(input, gate, output) ||
        !GgufSharedProjectionCanRun(input, up, output)) return false;
    const int columns = input.dims.back(), rows = input.Count(0)/columns;
    if (GgufTryPlanarGateUp(input.cudaData, gate.cudaData, up.cudaData,
            output.cudaData, gate.ggmlType, up.ggmlType, rows, columns, gate.dims[0])) return true;
    block_q8_1 *q8 = nullptr;
    if (FastllmCudaTryMalloc(reinterpret_cast<void **>(&q8),
            size_t(rows)*(columns/QK8_1)*sizeof(block_q8_1)) != FASTLLM_CUDA_TRY_MALLOC_SUCCESS) return false;
    const auto stream = cudaStreamPerThread;
    quantize_row_q8_1_cuda(static_cast<const half *>(input.cudaData), q8,
        columns, rows, 1, columns, GGML_TYPE_Q8_1, stream);
    GgufProjectFromQ8<0>(q8, gate, static_cast<half *>(output.cudaData), rows, columns, stream);
    // The up projection reads the rounded FP16 gate already stored here.
    // Its epilogue preserves both the SiLU and projection rounding boundaries.
    GgufProjectFromQ8<2>(q8, up, static_cast<half *>(output.cudaData), rows, columns, stream);
    FastllmCudaFree(q8);
    return true;
}

bool FastllmCudaHalfGgufGateUpSiluMul(
        const fastllm::Data &input,
        fastllm::Data &gateWeight,
        fastllm::Data &upWeight,
        fastllm::Data &output,
        int n, int m, int k) {
    if (n <= 0 || n > MMVQ_MAX_BATCH_SIZE || m <= 0 || k <= 0 ||
        m % QK8_1 != 0 || input.dataType != fastllm::DataType::FLOAT16 ||
        gateWeight.dataType != fastllm::DataType::DATA_GGUF_FORMAT ||
        upWeight.dataType != fastllm::DataType::DATA_GGUF_FORMAT ||
        gateWeight.cudaData == nullptr || upWeight.cudaData == nullptr) {
        return false;
    }

    half *cudaInput = (half*)FastllmCudaPrepareInput(input);
    half *cudaOutput = (half*)FastllmCudaPrepareOutput(output);
    if (cudaInput == nullptr || cudaOutput == nullptr) {
        FastllmCudaFinishInput(input, cudaInput);
        FastllmCudaFinishOutput(output, cudaOutput);
        return false;
    }

    cudaStream_t stream = cudaStreamPerThread;

    // The batch-8 verifier is faster through the tensor-core MMQ path.  Keep
    // this at the fused gate/up boundary so both projections avoid falling
    // back to the high-register-pressure multi-row MMVQ kernel.
    if (n == 8) {
        half *upOutput = (half*)FastllmCudaMalloc(
            (size_t)n * k * sizeof(half));
        if (upOutput != nullptr) {
            const bool gateMmq = FastllmCudaHalfMatMulGGUFMMQ(
                cudaInput, gateWeight.cudaData, cudaOutput,
                gateWeight.ggmlType, n, m, k, stream);
            const bool upMmq = gateMmq && FastllmCudaHalfMatMulGGUFMMQ(
                cudaInput, upWeight.cudaData, upOutput,
                upWeight.ggmlType, n, m, k, stream);
            if (gateMmq && upMmq) {
                const int elements = n * k;
                constexpr int threads = 256;
                FastllmGgufHalfSiluMulKernel<<<
                    (elements + threads - 1) / threads,
                    threads, 0, stream>>>(cudaOutput, upOutput, elements);
                FastllmCudaFree(upOutput);
                FastllmCudaFinishInput(input, cudaInput);
                FastllmCudaFinishOutput(output, cudaOutput);
                return true;
            }
            FastllmCudaFree(upOutput);
        }
    }

    if (gateWeight.ggmlType != upWeight.ggmlType &&
        FastllmCudaGGUFMixedGateUp(input, gateWeight, upWeight, output)) {
        FastllmCudaFinishInput(input, cudaInput);
        FastllmCudaFinishOutput(output, cudaOutput);
        return true;
    }

    if (gateWeight.ggmlType == upWeight.ggmlType &&
        FastllmCudaHalfGgufGateUpSiluMulMMVQ(
            cudaInput, gateWeight.cudaData, upWeight.cudaData, cudaOutput,
            gateWeight.ggmlType, n, m, k, stream)) {
        FastllmCudaFinishInput(input, cudaInput);
        FastllmCudaFinishOutput(output, cudaOutput);
        return true;
    }

    if (!get_has_vec_dot_q_cuda((ggml_type)gateWeight.ggmlType) ||
        !get_has_vec_dot_q_cuda((ggml_type)upWeight.ggmlType)) {
        FastllmCudaFinishInput(input, cudaInput);
        FastllmCudaFinishOutput(output, cudaOutput);
        return false;
    }

    block_q8_1 *q8Input = (block_q8_1*)FastllmCudaMalloc(
        (size_t)n * (m / QK8_1) * sizeof(block_q8_1));
    if (q8Input == nullptr) {
        FastllmCudaFinishInput(input, cudaInput);
        FastllmCudaFinishOutput(output, cudaOutput);
        return false;
    }

    quantize_row_q8_1_cuda(
        cudaInput, q8Input, m, n, 1, m, GGML_TYPE_Q8_1, stream);

    const bool fusedMmvq = n == 1 &&
        gateWeight.ggmlType == upWeight.ggmlType &&
        FastllmDispatchGgufFusedGateUpMmvq(
            (ggml_type)gateWeight.ggmlType,
            gateWeight.cudaData, upWeight.cudaData,
            q8Input, cudaOutput, m, k, stream);

    if (!fusedMmvq) {
        half *upOutput = (half*)FastllmCudaMalloc(
            (size_t)n * k * sizeof(half));
        if (upOutput == nullptr) {
            FastllmCudaFree(q8Input);
            FastllmCudaFinishInput(input, cudaInput);
            FastllmCudaFinishOutput(output, cudaOutput);
            return false;
        }

        ggml_backend_cuda_context ctx;
        ggml_cuda_op_mul_mat_vec_q_impl(
            ctx, (ggml_type)gateWeight.ggmlType, m, k, 1,
            0, 0, 0, 0,
            (char*)gateWeight.cudaData, (char*)q8Input, cudaOutput, nullptr,
            0, k, n, m, stream);
        ggml_cuda_op_mul_mat_vec_q_impl(
            ctx, (ggml_type)upWeight.ggmlType, m, k, 1,
            0, 0, 0, 0,
            (char*)upWeight.cudaData, (char*)q8Input, upOutput, nullptr,
            0, k, n, m, stream);

        const int elements = n * k;
        constexpr int threads = 256;
        FastllmGgufHalfSiluMulKernel<<<
            (elements + threads - 1) / threads,
            threads, 0, stream>>>(cudaOutput, upOutput, elements);
        FastllmCudaFree(upOutput);
    }

    FastllmCudaFree(q8Input);
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaHalfGgufMergedGateUpSiluMul(
        const fastllm::Data &input,
        fastllm::Data &weight,
        fastllm::Data &output,
        int n, int m, int k) {
    const ggml_type type = (ggml_type)weight.ggmlType;
    if (GgufPlanarProjectionShape(type, n, m, k) &&
        input.dataType == fastllm::FLOAT16 && output.dataType == fastllm::FLOAT16 &&
        weight.dataType == fastllm::DATA_GGUF_FORMAT &&
        GgufDenseLocal(input) && GgufDenseLocal(output) &&
        input.dims.back() == m && input.Count(0) == size_t(n)*m &&
        output.dims.back() == k && output.Count(0) == size_t(n)*k &&
        weight.dims == std::vector<int>({2*k,m}) &&
        weight.dataDevice == fastllm::DataDevice::CUDA && weight.cudaData && !weight.multiDeviceData &&
        (weight.dataDeviceIds.empty() || (weight.dataDeviceIds.size() == 1 &&
            weight.dataDeviceIds[0] == FastllmCudaGetDevice())) &&
        !GgufOverlap(input,output) && !GgufOverlap(weight,output)) {
        const auto *tensor = static_cast<const ggml_tensor *>(weight.ggmlTensor);
        int major = 0;
        if (tensor && tensor->type == type &&
            tensor->ne[0] == m && tensor->ne[1] == 2*k &&
            tensor->nb[0] == ggml_type_size(type) && tensor->nb[1] == ggml_row_size(type,m) &&
            (n != 8 || weight.forceGGUFFp32Dequant ||
                (cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,FastllmCudaGetDevice()) == cudaSuccess && major < 10))) {
            const char *up = static_cast<const char *>(weight.cudaData)+size_t(k)*ggml_row_size(type,m);
            if (GgufTryPlanarGateUp(input.cudaData,weight.cudaData,up,output.cudaData,
                    type,type,n,m,k)) return true;
        }
    }
    if (n != 1 || m <= 0 || k <= 0 || m % QK8_1 != 0 ||
        input.dataType != fastllm::DataType::FLOAT16 ||
        weight.dataType != fastllm::DataType::DATA_GGUF_FORMAT ||
        weight.cudaData == nullptr) {
        return false;
    }

    half *cudaInput = (half*)FastllmCudaPrepareInput(input);
    half *cudaOutput = (half*)FastllmCudaPrepareOutput(output);
    if (cudaInput == nullptr || cudaOutput == nullptr) {
        FastllmCudaFinishInput(input, cudaInput);
        FastllmCudaFinishOutput(output, cudaOutput);
        return false;
    }

    cudaStream_t stream = cudaStreamPerThread;
    const size_t rowBytes = ggml_row_size(type, m);
    const char *gateWeight = (const char*)weight.cudaData;
    const char *upWeight = gateWeight + (size_t)k * rowBytes;
    if (FastllmCudaHalfGgufGateUpSiluMulMMVQ(
            cudaInput, gateWeight, upWeight, cudaOutput,
            weight.ggmlType, n, m, k, stream)) {
        FastllmCudaFinishInput(input, cudaInput);
        FastllmCudaFinishOutput(output, cudaOutput);
        return true;
    }
    if (!get_has_vec_dot_q_cuda(type)) {
        FastllmCudaFinishInput(input, cudaInput);
        FastllmCudaFinishOutput(output, cudaOutput);
        return false;
    }

    block_q8_1 *q8Input = (block_q8_1*)FastllmCudaMalloc(
        (size_t)(m / QK8_1) * sizeof(block_q8_1));
    if (q8Input == nullptr) {
        FastllmCudaFinishInput(input, cudaInput);
        FastllmCudaFinishOutput(output, cudaOutput);
        return false;
    }
    quantize_row_q8_1_cuda(
        cudaInput, q8Input, m, 1, 1, m, GGML_TYPE_Q8_1, stream);
    const bool launched = FastllmDispatchGgufFusedGateUpMmvq(
        type, gateWeight, upWeight, q8Input, cudaOutput,
        m, k, stream);

    FastllmCudaFree(q8Input);
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return launched;
}

bool FastllmCudaBFloat16MatMulGGUF(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    if ((ggml_type)weight.ggmlType == GGML_TYPE_BF16) {
        return FastllmCudaBFloat16MatMulBFloat16(
            input, weight, bias, output, n, m, k);
    }
    if (weight.cudaData == nullptr ||
        (weight.extraCudaHalfData.size() == 0 && bias.dims.size() > 0)) {
        __nv_bfloat16 *cudaBiasData;
        cudaError_t state = cudaSuccess;
        state = cudaMalloc(&cudaBiasData, k * sizeof(__nv_bfloat16));
        if (bias.dims.size() > 0) {
            float *tempBiasData;
            state = cudaMalloc(&tempBiasData, k * sizeof(float));
            state = cudaMemcpy(tempBiasData, (uint8_t *)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
            int threadPerBlock = std::min(256, k);
            FastllmCudaFloat2Bf16Kernel <<<(k - 1) / threadPerBlock + 1, threadPerBlock>>>(tempBiasData, cudaBiasData, k);
            state = cudaFree(tempBiasData);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(__nv_bfloat16));
        }
        checkCudaErrors("Error: CUDA error when moving bias (bf16) to device!", state);
        weight.extraCudaHalfData.push_back((void *)cudaBiasData);
    }

    __nv_bfloat16 *cudaBiasData = bias.dims.size() == 0 ? nullptr : (__nv_bfloat16 *)weight.extraCudaHalfData[0];
    __nv_bfloat16 *cudaInput = (__nv_bfloat16 *)FastllmCudaPrepareInput(input);
    __nv_bfloat16 *cudaOutput = (__nv_bfloat16 *)FastllmCudaPrepareOutput(output);
    block_q8_1 *q8Input = nullptr;

    ggml_backend_cuda_context ctx;

    const ggml_type ggufType = (ggml_type)weight.ggmlType;
    const bool allowSmallMmvq = weight.forceGGUFFp32Dequant &&
        FastllmGGUFSmallMmvqShape(ggufType, n, m, k);
    const bool forceDequant = n != 1 && weight.forceGGUFFp32Dequant &&
        !(allowSmallMmvq && get_has_vec_dot_q_cuda(ggufType));
    auto dequant = ggml_get_to_bf16_cuda(ggufType);
    auto has_vec_dot = get_has_vec_dot_q_cuda(ggufType);
    cudaStream_t stream = cudaStreamPerThread;
    const bool usedMmq = !forceDequant && !allowSmallMmvq &&
        FastllmCudaBFloat16MatMulGGUFMMQ(
            cudaInput, weight.cudaData, cudaOutput, weight.ggmlType,
            n, m, k, stream);
    const bool usedExtendedMmvq = (!forceDequant || allowSmallMmvq) && !usedMmq &&
        FastllmCudaBFloat16MatMulGGUFMMVQ(
            cudaInput, weight.cudaData, cudaOutput, weight.ggmlType,
            n, m, k, stream);

    const bool usedDirectGemv = n == 1 && !usedMmq && !usedExtendedMmvq && !has_vec_dot &&
        FastllmGgufDirectGemv(cudaInput, weight.cudaData, cudaOutput,
                             ggufType, m, k, stream);

    if (!usedDirectGemv && !usedMmq && !usedExtendedMmvq &&
        (forceDequant || n > MMVQ_MAX_BATCH_SIZE || !has_vec_dot) &&
        dequant != nullptr) {
        auto handle = getFastllmCublasHandle();
        size_t workspaceBytes = 0;
        void *workspace = FastllmGGUFGetDequantWorkspace(
                &workspaceBytes, weight, "BF16 GEMM");
        FastllmGGUFDequantGemm(
                cudaInput, weight, cudaOutput, n, m, k,
                workspace, workspaceBytes, dequant, handle, stream);
    } else if (!usedDirectGemv && !usedMmq && !usedExtendedMmvq) {
        q8Input = (block_q8_1 *)FastllmCudaMalloc(n * m * sizeof(__nv_bfloat16));
        quantize_row_q8_1_cuda(
            cudaInput, q8Input, m, n, 1, m, GGML_TYPE_Q8_1, stream
        );
    }

    if (q8Input != nullptr && n > 1) {
        int i = 0;
        for (; i + MMVQ_MAX_BATCH_SIZE - 1 < n; i += MMVQ_MAX_BATCH_SIZE) {
            ggml_cuda_op_mul_mat_vec_q_impl(
                ctx, (ggml_type)weight.ggmlType, m, k, 1,
                0, 0, 0, 0,
                (char *)weight.cudaData,
                (char *)(q8Input + i * (m / QK8_1)),
                cudaOutput + i * k,
                nullptr,
                0, k, MMVQ_MAX_BATCH_SIZE, m, stream
            );
        }

        if (n - i > 0) {
            ggml_cuda_op_mul_mat_vec_q_impl(
                ctx, (ggml_type)weight.ggmlType, m, k, 1,
                0, 0, 0, 0,
                (char *)weight.cudaData,
                (char *)(q8Input + i * (m / QK8_1)),
                cudaOutput + i * k,
                nullptr,
                0, k, n - i, m, stream
            );
        }
    } else if (q8Input != nullptr) {
        ggml_cuda_op_mul_mat_vec_q_impl(
            ctx, (ggml_type)weight.ggmlType, m, k, 1,
            0, 0, 0, 0,
            (char *)weight.cudaData, (char *)q8Input, cudaOutput, nullptr,
            0, k, n, m, stream
        );
    }

    if (bias.dims.size() > 0) {
        FastllmCudaBiasKernel <<<n, 256, 0, stream>>>(cudaOutput, cudaBiasData, k);
    }

    if (q8Input != nullptr) {
        FastllmCudaFree(q8Input);
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);

    return true;
}

template <typename T>
struct FastllmMoeGGUFTraits;

template <>
struct FastllmMoeGGUFTraits<half> {
    static constexpr fastllm::DataType dataType = fastllm::DataType::FLOAT16;
    __device__ __forceinline__ static float toFloat(half value) {
        return __half2float(value);
    }
    __device__ __forceinline__ static half fromFloat(float value) {
        return __float2half_rn(value);
    }
};

template <>
struct FastllmMoeGGUFTraits<__nv_bfloat16> {
    static constexpr fastllm::DataType dataType = fastllm::DataType::BFLOAT16;
    __device__ __forceinline__ static float toFloat(__nv_bfloat16 value) {
        return __bfloat162float(value);
    }
    __device__ __forceinline__ static __nv_bfloat16 fromFloat(float value) {
        return __float2bfloat16_rn(value);
    }
};

template <>
struct FastllmMoeGGUFTraits<float> {
    static constexpr fastllm::DataType dataType = fastllm::DataType::FLOAT32;
    __device__ __forceinline__ static float toFloat(float value) {
        return value;
    }
    __device__ __forceinline__ static float fromFloat(float value) {
        return value;
    }
};

template <typename T>
static __global__ void FastllmMoeGGUFReduceKernel(const T *partOutput, T *output, const float *scores,
                                                  int topk, int hidden) {
    int st = blockIdx.x * blockDim.x + threadIdx.x;
    if (st >= hidden) {
        return;
    }

    float value = 0.0f;
    for (int j = 0; j < topk; j++) {
        value += FastllmMoeGGUFTraits<T>::toFloat(partOutput[(size_t)j * hidden + st]) * scores[j];
    }
    output[st] = FastllmMoeGGUFTraits<T>::fromFloat(value);
}

static size_t FastllmMoeGGUFQ8Bytes(int rows, int cols) {
    return (size_t)rows * (cols / QK8_1) * sizeof(block_q8_1);
}

template <typename T>
static void FastllmMoeGGUFMatMulQ8(ggml_type type, const void *weight, const block_q8_1 *q8Input,
                                   T *output, int m, int k) {
    ggml_backend_cuda_context ctx;
    ggml_cuda_op_mul_mat_vec_q_impl(
        ctx, type, m, k, 1,
        0, 0, 0, 0,
        (const char*)weight, (const char*)q8Input, output, nullptr,
        0, k, 1, m, nullptr
    );
}

static bool FastllmMoeGGUFCanRunWeight(const fastllm::Data *weight, int m, int k) {
    if (weight == nullptr || weight->dataType != fastllm::DataType::DATA_GGUF_FORMAT ||
        weight->ggmlType < 0 || weight->cudaData == nullptr ||
        weight->dims.size() != 2 || weight->dims[0] != k || weight->dims[1] != m) {
        return false;
    }
    ggml_type type = (ggml_type)weight->ggmlType;
    return get_has_vec_dot_q_cuda(type) && m % ggml_blck_size(type) == 0;
}

template <typename T>
static bool FastllmCudaTypedMergeMOEGGUFBatch1(const fastllm::Data &input, fastllm::Data &w1,
                                               fastllm::Data &output, fastllm::Data **gateups,
                                               fastllm::Data **downs, const float *scores,
                                               bool scoresOnCuda, int topk, int hidden, int inter) {
    if (topk <= 0 || hidden <= 0 || inter <= 0 || scores == nullptr ||
        gateups == nullptr || downs == nullptr ||
        input.dataType != FastllmMoeGGUFTraits<T>::dataType ||
        input.dataDevice != fastllm::DataDevice::CUDA ||
        hidden % QK8_1 != 0 || inter % QK8_1 != 0) {
        return false;
    }

    for (int j = 0; j < topk; j++) {
        if (!FastllmMoeGGUFCanRunWeight(gateups[j], hidden, inter * 2) ||
            !FastllmMoeGGUFCanRunWeight(downs[j], inter, hidden)) {
            return false;
        }
    }

    fastllm::Data gateOutput;
    gateOutput.dataDevice = input.dataDevice;
    gateOutput.dataDeviceIds = input.dataDeviceIds;
    gateOutput.dataType = FastllmMoeGGUFTraits<T>::dataType;
    gateOutput.Resize({topk, inter * 2});
    gateOutput.Allocate(false);

    w1.dataDevice = input.dataDevice;
    w1.dataDeviceIds = input.dataDeviceIds;
    w1.dataType = FastllmMoeGGUFTraits<T>::dataType;
    w1.Resize({topk, inter});
    w1.Allocate(false);

    fastllm::Data downOutput;
    downOutput.dataDevice = input.dataDevice;
    downOutput.dataDeviceIds = input.dataDeviceIds;
    downOutput.dataType = FastllmMoeGGUFTraits<T>::dataType;
    downOutput.Resize({topk, hidden});
    downOutput.Allocate(false);

    output.dataDevice = input.dataDevice;
    output.dataDeviceIds = input.dataDeviceIds;
    output.dataType = FastllmMoeGGUFTraits<T>::dataType;
    output.Resize({1, hidden});
    output.Allocate(false);

    T *cudaInput = (T*)FastllmCudaPrepareInput(input);
    T *cudaGate = (T*)FastllmCudaPrepareOutput(gateOutput);
    T *cudaW1 = (T*)FastllmCudaPrepareOutput(w1);
    T *cudaDown = (T*)FastllmCudaPrepareOutput(downOutput);
    T *cudaOutput = (T*)FastllmCudaPrepareOutput(output);

    block_q8_1 *q8Input = (block_q8_1*)FastllmCudaMalloc(FastllmMoeGGUFQ8Bytes(1, hidden));
    quantize_row_q8_1_cuda(cudaInput, q8Input, hidden, 1, 1, hidden, GGML_TYPE_Q8_1, nullptr);

    for (int j = 0; j < topk; j++) {
        FastllmMoeGGUFMatMulQ8((ggml_type)gateups[j]->ggmlType, gateups[j]->cudaData, q8Input,
                               cudaGate + (size_t)j * inter * 2, hidden, inter * 2);
    }

    FastllmCudaSwiglu(gateOutput, w1);

    block_q8_1 *q8W1 = (block_q8_1*)FastllmCudaMalloc(FastllmMoeGGUFQ8Bytes(topk, inter));
    quantize_row_q8_1_cuda(cudaW1, q8W1, inter, topk, 1, inter, GGML_TYPE_Q8_1, nullptr);
    for (int j = 0; j < topk; j++) {
        FastllmMoeGGUFMatMulQ8((ggml_type)downs[j]->ggmlType, downs[j]->cudaData,
                               q8W1 + (size_t)j * (inter / QK8_1),
                               cudaDown + (size_t)j * hidden, inter, hidden);
    }

    const float *cudaScores = scores;
    float *ownedCudaScores = nullptr;
    if (!scoresOnCuda) {
        ownedCudaScores = (float*)FastllmCudaMalloc((size_t)topk * sizeof(float));
        FastllmCudaCopyFromHostToDevice(ownedCudaScores, (void*)scores, (size_t)topk * sizeof(float));
        cudaScores = ownedCudaScores;
    }

    int threadPerBlock = 256;
    FastllmMoeGGUFReduceKernel<T><<<(hidden - 1) / threadPerBlock + 1, threadPerBlock>>>(
        cudaDown, cudaOutput, cudaScores, topk, hidden);

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);

    if (ownedCudaScores != nullptr) {
        FastllmCudaFree(ownedCudaScores);
    }
    FastllmCudaFree(q8W1);
    FastllmCudaFree(q8Input);
    return true;
}

bool FastllmCudaFloatMergeMOEGGUFBatch1(const fastllm::Data &input, fastllm::Data &w1,
                                        fastllm::Data &output, fastllm::Data **gateups,
                                        fastllm::Data **downs, const float *scores,
                                        bool scoresOnCuda, int topk, int hidden, int inter) {
    return FastllmCudaTypedMergeMOEGGUFBatch1<float>(
        input, w1, output, gateups, downs, scores, scoresOnCuda, topk, hidden, inter);
}

bool FastllmCudaHalfMergeMOEGGUFBatch1(const fastllm::Data &input, fastllm::Data &w1,
                                       fastllm::Data &output, fastllm::Data **gateups,
                                       fastllm::Data **downs, const float *scores,
                                       bool scoresOnCuda, int topk, int hidden, int inter) {
    return FastllmCudaTypedMergeMOEGGUFBatch1<half>(
        input, w1, output, gateups, downs, scores, scoresOnCuda, topk, hidden, inter);
}

bool FastllmCudaBFloat16MergeMOEGGUFBatch1(const fastllm::Data &input, fastllm::Data &w1,
                                           fastllm::Data &output, fastllm::Data **gateups,
                                           fastllm::Data **downs, const float *scores,
                                           bool scoresOnCuda, int topk, int hidden, int inter) {
    return FastllmCudaTypedMergeMOEGGUFBatch1<__nv_bfloat16>(
        input, w1, output, gateups, downs, scores, scoresOnCuda, topk, hidden, inter);
}
