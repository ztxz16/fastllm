#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "fastllm-cuda.h"
#include "fastllm.h"

static cublasHandle_t fastllmCublasHandle = nullptr;

void FastllmMatMulInt8(int8_t *A, int8_t *B, int32_t *C, int n, int m, int k) {
    int32_t i_alpha = 1, i_beta = 0;
    if (fastllmCublasHandle == nullptr) {
        cublasCreate(&fastllmCublasHandle);
    }

    cudaDeviceSynchronize();
    cudaDataType_t AType = CUDA_R_8I, BType = CUDA_R_8I, CType = CUDA_R_32I, ComputeType = CUDA_R_32I;
    cublasStatus_t status;
    status = cublasGemmEx(fastllmCublasHandle,
                          CUBLAS_OP_T,
                          CUBLAS_OP_N,
                          n,
                          k,
                          m,
                          &i_alpha,
                          A,
                          AType,
                          m,
                          B,
                          BType,
                          m,
                          &i_beta,
                          C,
                          CType,
                          n,
                          ComputeType,
                          static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("Error: cublas error.\n");
        exit(0);
    }
    cudaDeviceSynchronize();
}


__global__ void MatMulFloatInt8Kernel(float *A, uint8_t *B, float *C, float *bias, float *scales, uint8_t *zeros,
                                      int n, int m, int k) {
    int idx = blockIdx.x;
    int idy = threadIdx.x;
    int curId = idx * 64 + idy;
    int per = n * k / 4096;

    int st = curId * per, end = st + per;
    if (curId == 4095) {
        end = n * k;
    }

    for (int id = st; id < end; id++) {
        int i = id / k;
        int j = id % k;
        float now = 0.0f;
        int l = 0;
        for (; l < m; l++) {
            now += A[i * m + l] * (B[j * m + l] - zeros[j]);
        }

        now = now * scales[j];
        now += bias[j];
        C[i * k + j] = now;
    }
}


bool FastllmMatMulFloatInt8(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    float *inputData = (float *) input.cpuData;
    uint8_t *weightData = (uint8_t *) weight.cpuData;
    float *outputData = (float *) output.cpuData;
    float *biasData = bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr;

    float *cudaScales;
    uint8_t *cudaZeropoints;
    float *cudaBiasData;

    cudaMalloc(&cudaScales, k * sizeof(float));
    cudaMalloc(&cudaZeropoints, k);
    cudaMalloc(&cudaBiasData, k * sizeof(float));

    float *scales = new float[k];
    uint8_t *zeropoints = new uint8_t[k];
    float *biass = new float[k];
    for (int i = 0; i < k; i++) {
        zeropoints[i] = weight.perChannelsConfigs[i].zeroPoint;
        scales[i] = weight.perChannelsConfigs[i].scale;
        biass[i] = (biasData ? biasData[i] : 0.0f);
    }

    cudaMemcpy(cudaScales, scales, k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaZeropoints, zeropoints, k, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaBiasData, biass, k * sizeof(float), cudaMemcpyHostToDevice);
/*
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            float now = 0.0f;
            int l = 0;
            for (; l < m; l++) {
                now += inputData[i * m + l] * (weightData[j * m + l] - zeropoints[j]);
            }

            now = now * scales[j];
            now += biass[j];
            outputData[i * k + j] = now;
        }
    }
*/
    float *cudaOutput, *cudaInput;
    cudaMalloc(&cudaInput, n * m * sizeof(float));
    cudaMalloc(&cudaOutput, n * k * sizeof(float));
    cudaMemcpy(cudaInput, inputData, n * m * sizeof(float), cudaMemcpyHostToDevice);

    MatMulFloatInt8Kernel <<< 64, 64 >>> (cudaInput, (uint8_t*)weight.cudaData, cudaOutput,
                                                     cudaBiasData, cudaScales, cudaZeropoints, n, m, k);

    cudaMemcpy(outputData, cudaOutput, n * k * sizeof(float), cudaMemcpyDeviceToHost);

    delete[] zeropoints;
    delete[] scales;
    delete[] biass;
    cudaFree(cudaZeropoints);
    cudaFree(cudaScales);
    cudaFree(cudaBiasData);
    cudaFree(cudaOutput);

    return true;
}

void * FastllmCudaMalloc(size_t size) {
    void * ret;
    cudaMalloc(&ret, size);
    return ret;
}

void FastllmCudaCopyFromHostToDevice(void *dst, void *src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

void FastllmCudaCopyFromDeviceToHost(void *dst, void *src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

void FastllmCudaFree(void *ret) {
    cudaFree(ret);
}