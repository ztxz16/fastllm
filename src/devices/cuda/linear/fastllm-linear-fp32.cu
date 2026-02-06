//
// Created by huangyuyang on 2/6/26.
//

// fp32的乘法计算，早期验证使用，几乎不会被用到

#include "fastllm-cuda.cuh"
#include "fastllm.h"

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvFp32Fp32Kernel2(float *A, float *B, float *C, float *bias, int m, int k) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    // 1. 计算
    int st = blockIdx.x * PART;
    int end = st + PART;
    for (int p = st; p < end; p++) {
        sdata[tid] = 0;
        for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
            sdata[tid] += A[i] * B[p * m + i];
        }
        __syncthreads();
        for (unsigned int s = 1; s < THREAD_PER_BLOCK; s *= 2) {
            if ((tid & (2 * s - 1)) == 0) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            C[p] = sdata[0] + bias[p];
        }
        __syncthreads();
    }
}

static void FastllmCudaFloat32EnsureBiasOnDevice(fastllm::Data &weight, const fastllm::Data &bias, int k) {
    if (weight.cudaData == nullptr || weight.extraCudaData.size() == 0) {
        cudaError_t state = cudaSuccess;
        float *cudaBiasData;
        state = cudaMalloc(&cudaBiasData, k * sizeof(float));
        if (bias.dims.size() > 0) {
            state = cudaMemcpy(cudaBiasData, (uint8_t*)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(float));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaData.push_back((void*)cudaBiasData);
    }
}

bool FastllmCudaMatMulFloat32(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    FastllmCudaFloat32EnsureBiasOnDevice(weight, bias, k);

    float *cudaBiasData = (float*)weight.extraCudaData[0];
    float *cudaInput = (float*)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float*)FastllmCudaPrepareOutput(output);

    if (n > 1) {
        float h_alpha = 1.0, h_beta = 0.0;
        auto fastllmCublasHandle = getFastllmCublasHandle();
        //cudaDeviceSynchronize();
        cudaDataType_t AType = CUDA_R_32F, BType = CUDA_R_32F, CType = CUDA_R_32F, ComputeType = CUDA_R_32F;
        cublasStatus_t status;

        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              k, n, m,
                              &h_alpha, weight.cudaData, AType,
                              m, cudaInput, BType,
                              m, &h_beta,
                              cudaOutput, CType,
                              k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw("cublas error");
            FastllmCudaFinishInput(input, cudaInput);
            FastllmCudaFinishOutput(output, cudaOutput);
            exit(0);
        }

        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, (float*)weight.extraCudaData[0], k);
        }
    } else {
        FastllmGemvFp32Fp32Kernel2<256, 1> <<< k, 256 >>>(cudaInput, (float *) weight.cudaData, cudaOutput, cudaBiasData, m, k);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaHalfMatMulFloat32(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    FastllmCudaFloat32EnsureBiasOnDevice(weight, bias, k);

    float *cudaBiasData = (float*)weight.extraCudaData[0];
    float *cudaInput = (float*)FastllmCudaMalloc(input.Count(0) * sizeof(float));
    float *cudaOutput = (float*)FastllmCudaMalloc(output.Count(0) * sizeof(float));
    int inputLen = input.Count(0);
    FastllmCudaHalf2FloatKernel <<< (inputLen - 1) / 256 + 1, 256 >>>((half*)input.cudaData, cudaInput, inputLen);

    if (n > 1) {
        float h_alpha = 1.0, h_beta = 0.0;
        auto fastllmCublasHandle = getFastllmCublasHandle();
        //cudaDeviceSynchronize();
        cudaDataType_t AType = CUDA_R_32F, BType = CUDA_R_32F, CType = CUDA_R_32F, ComputeType = CUDA_R_32F;
        cublasStatus_t status;

        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              k, n, m,
                              &h_alpha, weight.cudaData, AType,
                              m, cudaInput, BType,
                              m, &h_beta,
                              cudaOutput, CType,
                              k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw("cublas error");
            FastllmCudaFinishInput(input, cudaInput);
            FastllmCudaFinishOutput(output, cudaOutput);
            exit(0);
        }

        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, (float*)weight.extraCudaData[0], k);
        }
    } else {
        FastllmGemvFp32Fp32Kernel2<256, 1> <<< k, 256 >>>(cudaInput, (float *) weight.cudaData, cudaOutput, cudaBiasData, m, k);
    }
    
    int outputLen = output.Count(0);
    FastllmCudaFloat2HalfKernel <<< (outputLen - 1) / 256 + 1, 256>>>(cudaOutput, (half*)output.cudaData, outputLen);
    DeviceSync();
    return true;
}
