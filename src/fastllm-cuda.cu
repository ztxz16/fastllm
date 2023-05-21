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