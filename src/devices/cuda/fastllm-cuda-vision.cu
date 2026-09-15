#include "devices/cuda/fastllm-cuda-vision.h"
#include "devices/cuda/fastllm-cuda.cuh"

#include <algorithm>
#include <climits>
#include <cstdio>

bool FastllmCudaVisionLinearFloat32(const fastllm::Data &input,
                                    const fastllm::Data &weight,
                                    fastllm::Data &output) {
    using namespace fastllm;
    if (input.dataDevice != DataDevice::CUDA || weight.dataDevice != DataDevice::CUDA ||
        output.dataDevice != DataDevice::CUDA || input.dataType != weight.dataType ||
        input.dataDeviceIds != weight.dataDeviceIds || input.dataDeviceIds != output.dataDeviceIds ||
        output.dataType != DataType::FLOAT32 || input.dims.empty() || weight.dims.size() != 2 ||
        input.dims.back() != weight.dims[1] || input.dims.back() <= 0 ||
        input.Count(0) / input.dims.back() > INT_MAX ||
        input.cudaData == nullptr || weight.cudaData == nullptr || output.cudaData == nullptr) {
        return false;
    }
    cudaDataType_t operandType;
    if (input.dataType == DataType::FLOAT16) operandType = CUDA_R_16F;
    else if (input.dataType == DataType::BFLOAT16) operandType = CUDA_R_16BF;
    else if (input.dataType == DataType::FLOAT32) operandType = CUDA_R_32F;
    else return false;
    const int m = input.dims.back(), k = weight.dims[0];
    const int n = input.Count(0) / m;
    if (n <= 0 || k <= 0 || output.Count(0) != (uint64_t)n * k) return false;
    const float alpha = 1.0f, beta = 0.0f;
    const cublasStatus_t status = cublasGemmEx(
        getFastllmCublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N, k, n, m,
        &alpha, weight.cudaData, operandType, m,
        input.cudaData, operandType, m, &beta,
        output.cudaData, CUDA_R_32F, k, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[Vision] FP32 partial GEMM failed: status=%d, n=%d, m=%d, k=%d.\n",
                (int)status, n, m, k);
        return false;
    }
    return true;
}

namespace {
    __global__ void VisionFloat32ToHalfKernel(const float *input, half *output, uint64_t count) {
        for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
             i < count; i += (uint64_t)blockDim.x * gridDim.x) {
            output[i] = __float2half_rn(input[i]);
        }
    }
}

bool FastllmCudaVisionFloat32ToHalf(const fastllm::Data &input,
                                  fastllm::Data &output) {
    using namespace fastllm;
    if (input.dataDevice != DataDevice::CUDA || output.dataDevice != DataDevice::CUDA ||
        input.dataDeviceIds != output.dataDeviceIds ||
        input.dataType != DataType::FLOAT32 || output.dataType != DataType::FLOAT16 ||
        input.dims != output.dims || input.dims.empty() ||
        input.cudaData == nullptr || output.cudaData == nullptr ||
        input.cudaData == output.cudaData) {
        return false;
    }
    const uint64_t count = input.Count(0);
    if (count == 0) return true;
    const int blocks = (int)std::min<uint64_t>((count + 255) / 256, 4096);
    VisionFloat32ToHalfKernel<<<blocks, 256>>>(
        (const float*)input.cudaData, (half*)output.cudaData, count);
    return cudaPeekAtLastError() == cudaSuccess;
}
