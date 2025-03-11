//
// Created by huangyuyang on 8/2/24.
//

#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <chrono>

#include "fastllm-cuda.cuh"
#include "fastllm-multicuda.cuh"
#include "fastllm.h"
#include "utils.h"

#include "devices/cpu/alivethreadpool.h"
#include "devices/cpu/cpudevice.h"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 // support tensor core
#include "mma.h"
using namespace nvcuda;
#endif

#define checkCudaErrors(message, val) showError(val, message, __FILE__, __LINE__)

extern __global__ void FastllmSwigluKernel(half* __restrict__ a, half* __restrict__ b, int len, int spatial, int mid);
extern __global__ void FastllmSwigluKernel(float* __restrict__ a, float* __restrict__ b, int len, int spatial, int mid);

extern __global__ void FastllmCudaBiasKernel(float *a, float *bias, int k);
extern __global__ void FastllmCudaBiasKernel(half *a, half *bias, int k);
extern __global__ void FastllmCudaFloat2HalfKernel(float* a, half *b, int len);
extern __global__ void FastllmCudaHalf2FloatKernel(half* a, float *b, int len);
extern __global__ void FastllmCudaInt42HalfKernel(uint8_t* a, float *scales, float *mins, half *b, int len, int per);
extern __global__ void FastllmCudaInt4Group2HalfKernel(uint8_t* a, float *scales, float *mins, half *b, int len, int per, int group, int groupCnt);
extern __global__ void FastllmCudaInt4Group2HalfKernel(uint8_t* a, half *scales, half *mins, half *b, int len, int per, int group, int groupCnt);
extern __global__ void FastllmCudaInt82HalfKernel(uint8_t* a, float *scales, uint8_t *zeros, half *b, int len, int per);

extern double GetSpan(std::chrono::system_clock::time_point time1, std::chrono::system_clock::time_point time2);
extern void showError(cudaError_t result, char const* const message, const char* const file, int const line);
extern cublasHandle_t getFastllmCublasHandle();
extern void *FastllmCudaPrepareInput(const fastllm::Data &input);
extern void *FastllmCudaPrepareOutput(fastllm::Data &output);
extern void FastllmCudaFinishInput(const fastllm::Data &input, void *data);
extern void FastllmCudaFinishOutput(fastllm::Data &output, void *data);

extern void LaunchFastllmGemmFp16Fp16(half *input, half *weight, half *output, half *bias, int n, int m, int k);
extern void LaunchFastllmGemmFp16Int8(half *input, uint8_t *weight, half *output, half *bias, float *scales, uint8_t *zeros, int n, int m, int k);
extern void LaunchFastllmGemmFp16Int4NoZero(half *input, uint8_t *weight, half *output, half *bias, float *scales, float *mins, int n, int m, int k);
extern void LaunchFastllmGemmFp16Int4Group(half *input, uint8_t *weight, half *output, half *bias, half *scales, half *mins, int n, int m, int k, int group, int groupCnt);

extern void LaunchFastllmGemmFp32Fp16(float *input, half *weight, float *output, float *bias, int n, int m, int k);
extern void LaunchFastllmGemmFp32Int8(float *input, uint8_t *weight, float *output, float *bias, float *scales, uint8_t *zeros, int n, int m, int k);
extern void LaunchFastllmGemmFp32Int4NoZero(float *input, uint8_t *weight, float *output, float *bias, float *scales, float *mins, int n, int m, int k);
extern void LaunchFastllmGemmFp32Int4Group(float *input, uint8_t *weight, float *output, float *bias, half *scales, half *mins, int n, int m, int k, int group, int groupCnt);

extern __global__ void FastllmReduceKernel(float *output, float* input, int len, int threadNum);
extern __global__ void FastllmReduceKernel(half *output, half* input, int len, int threadNum);

std::map <int, std::string> specialDeviceIds = {
    {99999, "cpu"}
};

void SwitchDeviceAndGetInfos(int deviceId, std::string &specialId, int &mallocType) {
    specialId = "";
    if (specialDeviceIds.find(deviceId) == specialDeviceIds.end()) {
        cudaSetDevice(deviceId);
    } else {
        specialId = specialDeviceIds[deviceId];
    }
    mallocType = 1;
    if (specialId == "cpu") {
        mallocType = 0;
    }
}

/*
type: device type (0 for cpu, 1 for cuda)
*/
void *AutoMalloc(size_t size, int type) {
    if (type == 0) {
        return (void*)(new uint8_t[size]);
    } else {
        return (void*)FastllmCudaMalloc(size);
    }
}

cudaError_t AutoMemset(void *a, int value, size_t size, int type) {
    if (type == 0) {
        memset(a, value, size);
        return cudaSuccess;
    } else {
        return cudaMemset(a, value, size);
    }
}

cudaMemcpyKind GetCudaMemcpyType(int dstType, int srcType) {
    if (srcType == 0) {
        if (dstType == 0) {
            return cudaMemcpyHostToHost;
        } else {
            return cudaMemcpyHostToDevice;
        }
    } else {
        if (dstType == 0) {
            return cudaMemcpyDeviceToHost;
        } else {
            return cudaMemcpyDeviceToDevice;
        }
    }
}

std::vector <int> multiCudaCurrentDevices;
std::map <int, int> multiCudaCurrentRatios;

void FastllmMultiCudaSetDevice(std::vector <int> ids) {
    multiCudaCurrentDevices = ids;
}

void FastllmMultiCudaSetDeviceRatio(std::map <int, int> &deviceRatio) {
    multiCudaCurrentRatios = deviceRatio;
}

namespace fastllm {
    extern FP16ToFP32Manager fp16tofp32;
    extern void Float16ToFloat32(uint16_t *float16, float *float32, int len);
    extern void Float32ToFloat16(float *float32, uint16_t *float16, int len);

    template <typename T>
    void RunMatmul(void *weight, DataType weightDataType, T *bias, 
                        int n, int m, int k, bool hasBias, float *scales, float *mins, uint8_t *zeros, int group, int groupCnt, 
                        T *curInput, T *curOutput);

    template <typename T>
    void CpuRunMatmul(Data *oriWeight, void *weight, DataType weightDataType, T *bias, 
                        int n, int m, int k, bool hasBias, float *scales, float *mins, uint8_t *zeros, int group, int groupCnt, 
                        T *curInput, T *curOutput, 
                        AliveThreadPool *pool, int startTid, int threadNum) {
        if (typeid(T) == typeid(float)) {
            float *inputData = (float*)curInput;
            float *outputData = (float*)curOutput;
            float *biasData = (float*)bias;
            if (weightDataType == DataType::FLOAT16) {
                uint16_t *halfWeight = (uint16_t*)weight;
                int per = k / threadNum;
                int cur = 0;
                std::vector<fastllm::MultiThreadFloat16LinearOp*> ops;
                for (int i = 0; i < threadNum; i++) {
                    int end = cur + per + (cur + per * (threadNum - i) < k);
                    if (i == threadNum - 1) {
                        end = k;
                    }
                    ops.push_back(new MultiThreadFloat16LinearOp(inputData, halfWeight, biasData, outputData,
                                                   n, m, k, cur, end));
                    cur = end;
                }
                for (int i = 0; i < threadNum; i++) {
                    pool->PushOp(startTid + i, ops[i]);
                }
                for (int i = 0; i < threadNum; i++) {
                    pool->Wait(startTid + i);
                    delete ops[i];
                }
            } else if (weightDataType == DataType::INT8) {
                printf("Error: CpuRunMatmul unsupport type: %d.\n", weightDataType);;
                exit(0);
            } else if (weightDataType == DataType::INT4_NOZERO) {
                printf("Error: CpuRunMatmul unsupport type: %d.\n", weightDataType);;
                exit(0);
            } else if (weightDataType == DataType::INT4_GROUP) {
// auto st = std::chrono::system_clock::now();
                uint8_t *weightData = (uint8_t *) weight;
                if (oriWeight->weightSum.size() == 0) {
                    auto &weightSum = oriWeight->weightSum;
                    weightSum.resize(k * group);
                    for (int i = 0; i < k; i++) {
                        for (int g = 0; g < group; g++) {
                            int gid = i * group + g;
                            int st = g * groupCnt;
                            int end = std::min(m, (g + 1) * groupCnt);
                            int j = st;
                            for (; j + 1 < end; j += 2) {
                                int id = (i * m + j) / 2;
                                weightSum[gid] += (weightData[id] & 0xF) + (weightData[id] >> 4);
                            }
                            for (; j < end; j++) {
                                int id = (i * m + j) / 2;
                                if ((i * m + j) % 2) {
                                    weightSum[gid] += (weightData[id] & 0xF);
                                } else {
                                    weightSum[gid] += (weightData[id] >> 4);
                                }
                            }
                        }
                    }

                    oriWeight->mins.resize(k * group);
                    oriWeight->scales.resize(k * group);
                    Float16ToFloat32((uint16_t*)mins, oriWeight->mins.data(), k * group);
                    Float16ToFloat32((uint16_t*)scales, oriWeight->scales.data(), k * group);
                }
                std::vector<LowBitConfig> inputConfigs;
                inputConfigs.resize(n * group);
                std::vector<uint8_t> uinput;
                uinput.resize(n * m);
                std::vector <float> inputSums;
                inputSums.resize(n * group);
                std::vector <float> iscales, izeros;
                iscales.resize(n * group);
                izeros.resize(n * group);
                MultiThreadOnlineQuantizationOp(inputData, uinput.data(), inputConfigs.data(), n, m, group, groupCnt, inputSums.data(), iscales.data(), izeros.data()).Run();
                int per = k / threadNum;
                int cur = 0;
                std::vector<fastllm::MultiThreadLinearInt4GroupOp*> ops;
                for (int i = 0; i < threadNum; i++) {
                    int end = (i == threadNum - 1 ? k : cur + per + (cur + per * (threadNum - i) < k));
                    ops.push_back(new MultiThreadLinearInt4GroupOp(uinput.data(), weightData + cur * m / 2, (int32_t*)outputData + cur, n, m, end - cur, k,
                                                    oriWeight->weightSum.data() + cur * group, oriWeight->mins.data() + cur * group, oriWeight->scales.data() + cur * group,
                                                    (bias == nullptr ? (float *) nullptr : (float*)bias + cur), iscales.data(), izeros.data(),
                                                    inputSums.data(), group, groupCnt));
                    cur = end;
                }
                for (int i = 0; i < threadNum; i++) {
                    pool->PushOp(startTid + i, ops[i]);
                }
                for (int i = 0; i < threadNum; i++) {
                    pool->Wait(startTid + i);
                    delete ops[i];
                }
/*
                int per = k / threadNum;
                int cur = 0;
                std::vector<fastllm::MultiThreadInt4GroupLinearOp*> ops;
                for (int i = 0; i < threadNum; i++) {
                    int end = cur + per + (cur + per * (threadNum - i) < k);
                    if (i == threadNum - 1) {
                        end = k;
                    }
                    ops.push_back(new MultiThreadInt4GroupLinearOp(inputData, (uint8_t*)weight, biasData, outputData, (uint16_t*)mins, (uint16_t*)scales,
                                                   n, m, k, cur, end, group, groupCnt));
                    cur = end;
                }
                for (int i = 0; i < threadNum; i++) {
                    pool->PushOp(startTid + i, ops[i]);
                }
                for (int i = 0; i < threadNum; i++) {
                    pool->Wait(startTid + i);
                    delete ops[i];
                }
*/
// float spend = GetSpan(st, std::chrono::system_clock::now());
// float gops = (float)n * m * k / spend / 1e9;
// printf("n = %d, m = %d, k = %d, spend %f s, gops = %f\n", n, m, k, spend, gops);
            } else {
                printf("Error: CpuRunMatmul unsupport type: %d.\n", weightDataType);;
                exit(0);
            }
        } else if (typeid(T) == typeid(half)) {
            uint16_t *inputData = (uint16_t*)curInput;
            uint16_t *outputData = (uint16_t*)curOutput;
            std::vector <float> floatBias;
            floatBias.resize(k);
            Float16ToFloat32((uint16_t*)bias, floatBias.data(), k);
            if (weightDataType == DataType::FLOAT16) {
                uint16_t *halfWeight = (uint16_t*)weight;
                int per = k / threadNum;
                int cur = 0;
                std::vector<fastllm::MultiThreadFloat16Float16LinearOp*> ops;
                for (int i = 0; i < threadNum; i++) {
                    int end = cur + per + (cur + per * (threadNum - i) < k);
                    if (i == threadNum - 1) {
                        end = k;
                    }
                    ops.push_back(new MultiThreadFloat16Float16LinearOp(inputData, halfWeight, floatBias.data(), outputData,
                                                   n, m, k, cur, end));
                    cur = end;
                }
                for (int i = 0; i < threadNum; i++) {
                    pool->PushOp(startTid + i, ops[i]);
                }
                for (int i = 0; i < threadNum; i++) {
                    pool->Wait(startTid + i);
                    delete ops[i];
                }
            } else if (weightDataType == DataType::INT8) {
                printf("Error: CpuRunMatmul unsupport type: %d.\n", weightDataType);;
                exit(0);
            } else if (weightDataType == DataType::INT4_NOZERO) {
                printf("Error: CpuRunMatmul unsupport type: %d.\n", weightDataType);;
                exit(0);
            } else if (weightDataType == DataType::INT4_GROUP) {
                std::vector <float> floatInput, floatOutput;
                floatInput.resize(n * m);
                floatOutput.resize(n * k);
                Float16ToFloat32((uint16_t*)inputData, floatInput.data(), n * m);

                uint8_t *weightData = (uint8_t *) weight;
                if (oriWeight->weightSum.size() == 0) {
                    auto &weightSum = oriWeight->weightSum;
                    weightSum.resize(k * group);
                    for (int i = 0; i < k; i++) {
                        for (int g = 0; g < group; g++) {
                            int gid = i * group + g;
                            int st = g * groupCnt;
                            int end = std::min(m, (g + 1) * groupCnt);
                            int j = st;
                            for (; j + 1 < end; j += 2) {
                                int id = (i * m + j) / 2;
                                weightSum[gid] += (weightData[id] & 0xF) + (weightData[id] >> 4);
                            }
                            for (; j < end; j++) {
                                int id = (i * m + j) / 2;
                                if ((i * m + j) % 2) {
                                    weightSum[gid] += (weightData[id] & 0xF);
                                } else {
                                    weightSum[gid] += (weightData[id] >> 4);
                                }
                            }
                        }
                    }

                    oriWeight->mins.resize(k * group);
                    oriWeight->scales.resize(k * group);
                    Float16ToFloat32((uint16_t*)mins, oriWeight->mins.data(), k * group);
                    Float16ToFloat32((uint16_t*)scales, oriWeight->scales.data(), k * group);
                }
                std::vector<LowBitConfig> inputConfigs;
                inputConfigs.resize(n * group);
                std::vector<uint8_t> uinput;
                uinput.resize(n * m);
                std::vector <float> inputSums;
                inputSums.resize(n * group);
                std::vector <float> iscales, izeros;
                iscales.resize(n * group);
                izeros.resize(n * group);
                MultiThreadOnlineQuantizationOp(floatInput.data(), uinput.data(), inputConfigs.data(), n, m, group, groupCnt, inputSums.data(), iscales.data(), izeros.data()).Run();
                int per = k / threadNum;
                int cur = 0;
                std::vector<fastllm::MultiThreadLinearInt4GroupOp*> ops;
                for (int i = 0; i < threadNum; i++) {
                    int end = (i == threadNum - 1 ? k : cur + per + (cur + per * (threadNum - i) < k));
                    ops.push_back(new MultiThreadLinearInt4GroupOp(uinput.data(), weightData + cur * m / 2, (int32_t*)floatOutput.data() + cur, n, m, end - cur, k,
                                                    oriWeight->weightSum.data() + cur * group, oriWeight->mins.data() + cur * group, oriWeight->scales.data() + cur * group,
                                                    floatBias.data() + cur, iscales.data(), izeros.data(),
                                                    inputSums.data(), group, groupCnt));
                    cur = end;
                }
                for (int i = 0; i < threadNum; i++) {
                    pool->PushOp(startTid + i, ops[i]);
                }
                for (int i = 0; i < threadNum; i++) {
                    pool->Wait(startTid + i);
                    delete ops[i];
                }

                Float32ToFloat16(floatOutput.data(), (uint16_t*)outputData, n * k);
            } else {
                printf("Error: CpuRunMatmul unsupport type: %d.\n", weightDataType);;
                exit(0);
            }
        } else {
            printf("CpuRunMatmul: Unsuppoert type.\n");
            exit(0);
        }
    }

    template <typename T>
    void RunMatmul(void *weight, DataType weightDataType, T *bias, 
                        int n, int m, int k, bool hasBias, float *scales, float *mins, uint8_t *zeros, int group, int groupCnt, 
                        T *curInput, T *curOutput) {
        if (typeid(T) == typeid(float)) {
            if (weightDataType == DataType::FLOAT16 && n < 8) {
                LaunchFastllmGemmFp32Fp16((float*)curInput, (half*)weight, (float*)curOutput, (float*)bias, n, m, k);
            } else if (weightDataType == DataType::INT8 && n < 8) {
                LaunchFastllmGemmFp32Int8((float*)curInput, (uint8_t*)weight, (float*)curOutput, (float*)bias, scales, zeros, n, m, k);
            } else if (weightDataType == DataType::INT4_NOZERO && n < 8) {
                LaunchFastllmGemmFp32Int4NoZero((float*)curInput, (uint8_t*)weight, (float*)curOutput, (float*)bias, scales, mins, n, m, k);
            } else if (weightDataType == DataType::INT4_GROUP && n < 8) {
               LaunchFastllmGemmFp32Int4Group((float*)curInput, (uint8_t*)weight, (float*)curOutput, (float*)bias, (half*)scales, (half*)mins, n, m, k, group, groupCnt);
            } else {
                auto fastllmCublasHandle = getFastllmCublasHandle();
                half *cudaFp16Input, *cudaFp16Output;
                cudaFp16Input = (half *) FastllmCudaMalloc(n * m * sizeof(half));
                cudaFp16Output = (half *) FastllmCudaMalloc(n * k * sizeof(half));

                __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
                cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
                cublasStatus_t status;

                int threadPerBlock = std::min(256, n * m);
                FastllmCudaFloat2HalfKernel <<< (n * m - 1) / threadPerBlock + 1, threadPerBlock>>>((float*)curInput, cudaFp16Input, n * m);

                half *fp16Weight = (half*)weight;
                bool isQuant = false;
                if (weightDataType == DataType::INT4_NOZERO) {
                    int threadPerBlock = std::min(256, k * m);
                    isQuant = true;
                    fp16Weight = (half*)FastllmCudaMalloc(k * m * sizeof(half));
                    FastllmCudaInt42HalfKernel <<< (k * m - 1) / threadPerBlock + 1, threadPerBlock>>>((uint8_t*)weight, scales, mins, fp16Weight, k * m, m);
                } else if (weightDataType == DataType::INT4_GROUP) {
                    int threadPerBlock = std::min(256, k * m);
                    isQuant = true;
                    fp16Weight = (half*)FastllmCudaMalloc(k * m * sizeof(half));
                    FastllmCudaInt4Group2HalfKernel <<< (k * m - 1) / threadPerBlock + 1, threadPerBlock>>>((uint8_t*)weight, (half*)scales, (half*)mins, fp16Weight, k * m, m, group, groupCnt);
                } else if (weightDataType == DataType::INT8) {
                    int threadPerBlock = std::min(256, k * m);
                    isQuant = true;
                    fp16Weight = (half*)FastllmCudaMalloc(k * m * sizeof(half));
                    FastllmCudaInt82HalfKernel <<< (k * m - 1) / threadPerBlock + 1, threadPerBlock>>>((uint8_t*)weight, scales, zeros, fp16Weight, k * m, m);
                }

                status = cublasGemmEx(fastllmCublasHandle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    k, n, m,
                                    &h_alpha, fp16Weight, AType,
                                    m, cudaFp16Input, BType,
                                    m, &h_beta,
                                    cudaFp16Output, CType,
                                    k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
                if (status != CUBLAS_STATUS_SUCCESS) {
                    printf("Error: cublas error.\n");
                    throw("cublas error");
                    exit(0);
                }

                FastllmCudaHalf2FloatKernel <<< (n * k - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaFp16Output, (float*)curOutput, n * k);
                if (hasBias) {
                    FastllmCudaBiasKernel <<< n, 256 >>> ((float*)curOutput, (float*)bias, k);
                }

                FastllmCudaFree(cudaFp16Input);
                FastllmCudaFree(cudaFp16Output);

                if (isQuant) {
                    FastllmCudaFree(fp16Weight);
                }
            }
        } else if (typeid(T) == typeid(half)) {
            if (weightDataType == DataType::FLOAT16 && n < 8) {
                LaunchFastllmGemmFp16Fp16((half*)curInput, (half*)weight, (half*)curOutput, (half*)bias, n, m, k);
            } else if (weightDataType == DataType::INT8 && n < 8) {
                LaunchFastllmGemmFp16Int8((half*)curInput, (uint8_t*)weight, (half*)curOutput, (half*)bias, scales, zeros, n, m, k);
            } else if (weightDataType == DataType::INT4_NOZERO && n < 8) {
                LaunchFastllmGemmFp16Int4NoZero((half*)curInput, (uint8_t*)weight, (half*)curOutput, (half*)bias, scales, mins, n, m, k);
            } else if (weightDataType == DataType::INT4_GROUP && n < 8) {
                LaunchFastllmGemmFp16Int4Group((half*)curInput, (uint8_t*)weight, (half*)curOutput, (half*)bias, (half*)scales, (half*)mins, n, m, k, group, groupCnt);
            } else {
                __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
                auto fastllmCublasHandle = getFastllmCublasHandle();
                cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
                cublasStatus_t status;

                half *fp16Weight = (half*)weight;
                bool isQuant = false;
                if (weightDataType == DataType::INT4_NOZERO) {
                    int threadPerBlock = std::min(256, k * m);
                    isQuant = true;
                    fp16Weight = (half*)FastllmCudaMalloc(k * m * sizeof(half));
                    FastllmCudaInt42HalfKernel <<< (k * m - 1) / threadPerBlock + 1, threadPerBlock>>>((uint8_t*)weight, scales, mins, fp16Weight, k * m, m);
                } else if (weightDataType == DataType::INT4_GROUP) {
                    int threadPerBlock = std::min(256, k * m);
                    isQuant = true;
                    fp16Weight = (half*)FastllmCudaMalloc(k * m * sizeof(half));
                    FastllmCudaInt4Group2HalfKernel <<< (k * m - 1) / threadPerBlock + 1, threadPerBlock>>>((uint8_t*)weight, (half*)scales, (half*)mins, fp16Weight, k * m, m, group, groupCnt);
                } else if (weightDataType == DataType::INT8) {
                    int threadPerBlock = std::min(256, k * m);
                    isQuant = true;
                    fp16Weight = (half*)FastllmCudaMalloc(k * m * sizeof(half));
                    FastllmCudaInt82HalfKernel <<< (k * m - 1) / threadPerBlock + 1, threadPerBlock>>>((uint8_t*)weight, scales, zeros, fp16Weight, k * m, m);
                }

                status = cublasGemmEx(fastllmCublasHandle,
                                        CUBLAS_OP_T, CUBLAS_OP_N,
                                        k, n, m, &h_alpha, fp16Weight, AType, 
                                        m, curInput, BType,
                                        m, &h_beta,
                                        curOutput, CType,
                                        k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
                if (status != CUBLAS_STATUS_SUCCESS) {
                    printf("Error: cublas error.\n");
                    throw ("cublas error");
                    exit(0);
                }            

                if (hasBias) {
                    FastllmCudaBiasKernel <<< n, 256 >>>((half*)curOutput, (half*)bias, k);
                }

                if (isQuant) {
                    FastllmCudaFree(fp16Weight);
                }
            }
        } else {
            printf("RunMatMul: Unsuppoert type.\n");
            exit(0);
        }
    }

    template <typename T>
    struct MultiCudaMatMulSingleOp : MultiThreadBaseOp {
        int deviceId;
        void *weight;
        DataType weightDataType;
        T *cpuInput, *cudaInput, *cudaOutput, *bias;
        int n, m, k, start, len;
        bool hasBias;
        float *scales, *mins;
        uint8_t *zeros;
        int group, groupCnt;

        MultiCudaMatMulSingleOp(int deviceId, void *weight, DataType weightDataType, T *cpuInput, T *cudaInput, T *cudaOutput, T *bias, 
                                    int n, int m, int k, int start, int len, bool hasBias, float *scales, float *mins, uint8_t *zeros, int group, int groupCnt)
            : deviceId(deviceId), weight(weight), weightDataType(weightDataType), cpuInput(cpuInput), cudaInput(cudaInput), cudaOutput(cudaOutput), bias(bias), 
            n(n), m(m), k(k), start(start), len(len), hasBias(hasBias), scales(scales), mins(mins), zeros(zeros), group(group), groupCnt(groupCnt) {}

        void Run() {
            cudaSetDevice(deviceId);
            T *curInput = cudaInput; 
            T *curOutput = cudaOutput; 

            if (deviceId != 0) {
                curInput = (T*)FastllmCudaMalloc(n * m * sizeof(T));                
                cudaMemcpy(curInput, cpuInput, n * m * sizeof(T), cudaMemcpyHostToDevice);
            }
            if (deviceId != 0 || n > 1) {
                curOutput = (T*)FastllmCudaMalloc(n * len * sizeof(T));
            }

            RunMatmul(weight, weightDataType, bias, n, m, len, hasBias, scales, mins, zeros, group, groupCnt, curInput, curOutput);
            if (deviceId != 0) {
                FastllmCudaFree(curInput);
            }
            if (deviceId != 0 || n > 1) {
                cudaMemcpy2D(cudaOutput + start, k * sizeof(T), curOutput, len * sizeof(T), len * sizeof(T), n, cudaMemcpyDeviceToDevice);
                FastllmCudaFree(curOutput);
            }
        }
    };

    template <typename T>
    struct MultiCudaCpuMatMulSingleOp : MultiThreadBaseOp {
        Data *oriWeight;
        void *weight;
        DataType weightDataType;
        T *cpuInput, *cudaInput, *cudaOutput, *bias;
        int n, m, k, start, len;
        bool hasBias;
        float *scales, *mins;
        uint8_t *zeros;
        int group, groupCnt;
        AliveThreadPool *pool;
        int curStartTid, curThreadNum;

        MultiCudaCpuMatMulSingleOp(Data *oriWeight, void *weight, DataType weightDataType, T *cpuInput, T *cudaInput, T *cudaOutput, T *bias, 
                                    int n, int m, int k, int start, int len, bool hasBias, float *scales, float *mins, uint8_t *zeros, int group, int groupCnt,
                                    AliveThreadPool *pool, int curStartTid, int curThreadNum)
            : oriWeight(oriWeight), weight(weight), weightDataType(weightDataType), cpuInput(cpuInput), cudaInput(cudaInput), cudaOutput(cudaOutput), bias(bias), 
            n(n), m(m), k(k), start(start), len(len), hasBias(hasBias), scales(scales), mins(mins), zeros(zeros), group(group), groupCnt(groupCnt),
            pool(pool), curStartTid(curStartTid), curThreadNum(curThreadNum) {}

        void Run() {
            T *curOutput = new T[n * len];
            CpuRunMatmul(oriWeight, weight, weightDataType, bias, n, m, len, hasBias, scales, mins, zeros, group, groupCnt, cpuInput, curOutput, pool, curStartTid, curThreadNum);
            cudaMemcpy2D(cudaOutput + start, k * sizeof(T), curOutput, len * sizeof(T), len * sizeof(T), n, cudaMemcpyHostToDevice);
            delete[] curOutput;
        }
    };

    template <typename T>
    struct MultiCudaMLPSingleOp : MultiThreadBaseOp {
        int deviceId;
        Data *weight0, *weight1;
        int n, m, k1, k2;
        T *cudaInput, *cudaOutput;
        T **curInput, **curOutput;
        T *cpuInput, *partOutput;
        int threadNum, tid;

        MultiCudaMLPSingleOp(int deviceId, Data *weight0, Data *weight1, 
                            int n, int m, int k1, int k2, T *cudaInput, T *cudaOutput, T **curInput, T **curOutput, 
                            T *cpuInput, T *partOutput, 
                            int threadNum, int tid)
            : deviceId(deviceId), weight0(weight0), weight1(weight1), 
            n(n), m(m), k1(k1), k2(k2), cudaInput(cudaInput), cudaOutput(cudaOutput), curInput(curInput), curOutput(curOutput), 
            cpuInput(cpuInput), partOutput(partOutput),
            threadNum(threadNum), tid(tid) {}

        void Run() {
            cudaSetDevice(deviceId);
            *curInput = cudaInput; 
            *curOutput = cudaOutput; 
            if (deviceId != 0) {
                *curInput = (T*)FastllmCudaMalloc(n * m * sizeof(T));                
                *curOutput = (T*)FastllmCudaMalloc(n * k2 * sizeof(T));
                // cudaMemcpy(*curInput, cudaInput, n * m * sizeof(T), cudaMemcpyDeviceToDevice);
                cudaMemcpy(*curInput, cpuInput, n * m * sizeof(T), cudaMemcpyHostToDevice);
            } else if (threadNum > 1) {
                *curOutput = partOutput;
            }

            T *mid0 = (T*)FastllmCudaMalloc(n * k1 * sizeof(T));
            T *mid1 = (T*)FastllmCudaMalloc(n * k1 / 2 * sizeof(T));
            bool isQuantWeight = weight0->mins.size() > 0;
            std::vector <void*> datas0 = weight0->extraCudaData, datas1 = weight1->extraCudaData;
            if (typeid(T) == typeid(half)) {
                datas0 = weight0->extraCudaHalfData;
                datas1 = weight1->extraCudaHalfData;
            }

            RunMatmul(datas0[tid * 2], weight0->dataType,
                            (T *) datas0[tid * 2 + 1],
                            n, m, k1, false,
                            (float*)(isQuantWeight ? datas0[threadNum * 2 + tid * 3] : nullptr),
                            (float*)(isQuantWeight ? datas0[threadNum * 2 + tid * 3 + 1] : nullptr),
                            (uint8_t*)(isQuantWeight ? datas0[threadNum * 2 + tid * 3 + 2] : nullptr),
                            weight0->group, weight0->groupCnt, *curInput, mid0);
            int threadPerBlock = std::min(1024, n * k1);
            FastllmSwigluKernel <<< (n * k1 - 1) / threadPerBlock + 1, threadPerBlock>>>(mid0, mid1, n * k1 / 2, k1, k1 / 2);
            RunMatmul(datas1[tid * 2], weight1->dataType,
                            (T *) datas1[tid * 2 + 1],
                            n, k1 / 2, k2, false,
                            (float*)(isQuantWeight ? datas1[threadNum * 2 + tid * 3] : nullptr),
                            (float*)(isQuantWeight ? datas1[threadNum * 2 + tid * 3 + 1] : nullptr),
                            (uint8_t*)(isQuantWeight ? datas1[threadNum * 2 + tid * 3 + 2] : nullptr),
                            weight1->group, weight1->groupCnt, mid1, *curOutput);

            if (threadNum > 1 && deviceId > 0) {
                cudaMemcpy(partOutput, *curOutput, n * k2 * sizeof(T), cudaMemcpyDeviceToDevice);
            }

            FastllmCudaFree(mid0);
            FastllmCudaFree(mid1);
        }
    };

    template <typename T>
    struct MultiCudaCpuMLPSingleOp : MultiThreadBaseOp {
        int deviceId;
        Data *weight0, *weight1;
        int n, m, k1, k2;
        T *cudaInput, *cudaOutput;
        T **curInput, **curOutput;
        T *cpuInput, *partOutput;
        int threadNum, tid;
        AliveThreadPool *pool;
        int curStartTid, curThreadNum;

        MultiCudaCpuMLPSingleOp(int deviceId, Data *weight0, Data *weight1, 
                            int n, int m, int k1, int k2, T *cudaInput, T *cudaOutput, T **curInput, T **curOutput, 
                            T *cpuInput, T *partOutput, 
                            int threadNum, int tid, 
                            AliveThreadPool *pool, int curStartTid, int curThreadNum)
            : deviceId(deviceId), weight0(weight0), weight1(weight1), 
            n(n), m(m), k1(k1), k2(k2), cudaInput(cudaInput), cudaOutput(cudaOutput), curInput(curInput), curOutput(curOutput), 
            cpuInput(cpuInput), partOutput(partOutput),
            threadNum(threadNum), tid(tid), pool(pool), curStartTid(curStartTid), curThreadNum(curThreadNum) {}

        void Run() {
            T *mid0 = (T*)AutoMalloc(n * k1 * sizeof(T), 0);
            T *mid1 = (T*)AutoMalloc(n * k1 / 2 * sizeof(T), 0);
            *curOutput = (T*)AutoMalloc(n * k2 * sizeof(T), 0);

            bool isQuantWeight = weight0->mins.size() > 0;
            std::vector <void*> datas0 = weight0->extraCudaData, datas1 = weight1->extraCudaData;
            if (typeid(T) == typeid(half)) {
                datas0 = weight0->extraCudaHalfData;
                datas1 = weight1->extraCudaHalfData;
            }
            CpuRunMatmul(weight0, datas0[tid * 2], weight0->dataType,
                            (T *) datas0[tid * 2 + 1],
                            n, m, k1, false,
                            (float*)(isQuantWeight ? datas0[threadNum * 2 + tid * 3] : nullptr),
                            (float*)(isQuantWeight ? datas0[threadNum * 2 + tid * 3 + 1] : nullptr),
                            (uint8_t*)(isQuantWeight ? datas0[threadNum * 2 + tid * 3 + 2] : nullptr),
                            weight0->group, weight0->groupCnt, cpuInput, mid0, 
                            pool, curStartTid, curThreadNum);
            if (typeid(T) == typeid(float)) {
                (MultiThreadSwigluOp((float*)mid0, k1 / 2, k1 / 2, (float*)mid1, n, k1, k1 / 2)).Run();
            } else if (typeid(T) == typeid(half) || typeid(T) == typeid(uint16_t)) {
                (MultiThreadSwigluFloat16Op((uint16_t*)mid0, k1 / 2, k1 / 2, (uint16_t*)mid1, n, k1, k1 / 2)).Run();
            } else {
                printf("Unsupport swiglu type.");
            }
            CpuRunMatmul(weight1, datas1[tid * 2], weight1->dataType,
                            (T *) datas1[tid * 2 + 1],
                            n, k1 / 2, k2, false,
                            (float*)(isQuantWeight ? datas1[threadNum * 2 + tid * 3] : nullptr),
                            (float*)(isQuantWeight ? datas1[threadNum * 2 + tid * 3 + 1] : nullptr),
                            (uint8_t*)(isQuantWeight ? datas1[threadNum * 2 + tid * 3 + 2] : nullptr),
                            weight1->group, weight1->groupCnt, mid1, *curOutput,
                            pool, curStartTid, curThreadNum);
            cudaMemcpy(partOutput, *curOutput, n * k2 * sizeof(T), cudaMemcpyHostToDevice);
            delete[] *curOutput;
            delete[] mid0;
            delete[] mid1;

            *curOutput = nullptr;
        }
    };
}

// 将total个计算任务切分
// 若当前有x个设备，返回一个长度为(x + 1)的vector，第i个设备执行任务[ret[i], ret[i + 1])
std::vector <int> FastllmMultiCudaGetSplitPoints(int total, int unit = 1) {
    int deviceNum = multiCudaCurrentDevices.size();
    int nodes = total / unit;
    int totalRatio = 0;
    if (multiCudaCurrentRatios.size() > 0) {
        for (auto &it : multiCudaCurrentRatios) {
            totalRatio += it.second;
        }
    } else {
        totalRatio = deviceNum;
    }
    std::vector <int> ret;
    int cur = 0;
    for (int i = 0; i < deviceNum; i++) {
        int curRatio = 1;
        if (multiCudaCurrentRatios.find(multiCudaCurrentDevices[i]) != multiCudaCurrentRatios.end()) {
            curRatio = multiCudaCurrentRatios[i];
        }
        int now = std::max(1, nodes * curRatio / totalRatio) * unit;
        int end = (i == deviceNum - 1 ? total : cur + now);
        ret.push_back(cur);
        if (i == deviceNum - 1) {
            ret.push_back(end);
        }
        cur = end;
    }
    return ret;
}

// deviceId -> [[l0, r0), [l1, r1), ...]
using DivisionScheme = std::map <int, std::vector <std::pair <int, int> > >;

bool PrepareMultiCudaWeight(fastllm::Data &weight, const fastllm::Data &bias, DivisionScheme divisionScheme, int splitAxis) {
    int k = weight.dims[0], m = weight.dims[1];
    cudaError_t state = cudaSuccess;
    float *cudaBiasData = (float*)FastllmCudaMalloc(k * sizeof(float));
    if (bias.dims.size() > 0) {
        state = cudaMemcpy(cudaBiasData, (uint8_t *) bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        state = cudaMemset(cudaBiasData, 0, k * sizeof(float));
    }
        
    for (int i = 0; i < multiCudaCurrentDevices.size(); i++) {
        int deviceId = multiCudaCurrentDevices[i], mallocType = 0;
        std::string specialId = "";
        SwitchDeviceAndGetInfos(deviceId, specialId, mallocType);

        auto &div = divisionScheme[deviceId];
        int len = 0;
        for (auto &it : div) {
            len += it.second - it.first;
        }

        void *deviceWeightData;
        float *deviceBiasData;
        cudaError_t state = cudaSuccess;
        if (splitAxis == 0) {
            deviceWeightData = (void*)AutoMalloc(len * m * weight.unitSize / weight.unitSizeDiv, mallocType);
            deviceBiasData = (float*)AutoMalloc(len * sizeof(float), mallocType);
            int curLen = 0;
            for (auto &it : div) {
                state = cudaMemcpy((uint8_t*)deviceWeightData + curLen * m * weight.unitSize / weight.unitSizeDiv, 
                                    (uint8_t*)weight.cudaData + it.first * m * weight.unitSize / weight.unitSizeDiv, 
                                    (it.second - it.first) * m * weight.unitSize / weight.unitSizeDiv, GetCudaMemcpyType(mallocType, 1));
                state = cudaMemcpy(deviceBiasData + curLen, cudaBiasData + it.first, (it.second - it.first) * sizeof(float), GetCudaMemcpyType(mallocType, 1));
                curLen += (it.second - it.first);
            }
        } else {
            deviceWeightData = (void*)AutoMalloc(k * len * weight.unitSize / weight.unitSizeDiv, mallocType);
            deviceBiasData = (float*)AutoMalloc(k * sizeof(float), mallocType);
            int curLen = 0;
            for (auto &it : div) {
                state = cudaMemcpy2D((uint8_t*)deviceWeightData + curLen * weight.unitSize / weight.unitSizeDiv,
                                    (it.second - it.first) * weight.unitSize / weight.unitSizeDiv,
                                    (uint8_t*)weight.cudaData + it.first * weight.unitSize / weight.unitSizeDiv, 
                                    m * weight.unitSize / weight.unitSizeDiv, 
                                    (it.second - it.first) * weight.unitSize / weight.unitSizeDiv,
                                    k, GetCudaMemcpyType(mallocType, 1));
                curLen += (it.second - it.first);
            }
            if (i == 0) {
                state = cudaMemcpy(deviceBiasData, cudaBiasData, k * sizeof(float), GetCudaMemcpyType(mallocType, 1));
            } else {
                state = AutoMemset(deviceBiasData, 0, k * sizeof(float), mallocType);
            }
        }

        weight.extraCudaData.push_back((void *) deviceWeightData);
        weight.extraCudaData.push_back((void *) deviceBiasData);

        if (cudaSuccess != state) {
            checkCudaErrors("Error: CUDA error when split weight!", state);
            return false;
        }
    }

    // 1. mins, scales
    if (weight.mins.size() > 0) {
        int weightGroup = weight.group < 0 ? 1 : weight.group;
        std::vector <uint8_t> zeropoints = std::vector <uint8_t> (k * weightGroup, 0);
        for (int i = 0; i < k * weightGroup; i++) {
            zeropoints[i] = weight.perChannelsConfigs[i].zeroPoint;
        }
        for (int i = 0; i < multiCudaCurrentDevices.size(); i++) {
           int deviceId = multiCudaCurrentDevices[i], mallocType = 0;
            std::string specialId = "";
            SwitchDeviceAndGetInfos(deviceId, specialId, mallocType);
            
            auto &div = divisionScheme[deviceId];
            int len = 0;
            for (auto &it : div) {
                len += it.second - it.first;
            }

            float *cudaScales;
            float *cudaMins;
            uint8_t *cudaZeropoints;
            if (splitAxis == 0) {
                if (weight.dataType == fastllm::DataType::INT4_GROUP) {
                    cudaScales = (float*)AutoMalloc(len * weightGroup * sizeof(half), mallocType);
                    cudaMins = (float*)AutoMalloc(len * weightGroup * sizeof(half), mallocType);
                    std::vector <half> halfScales, halfMins;
                    for (int i = 0; i < weight.scales.size(); i++) {
                        halfScales.push_back(__float2half(weight.scales[i]));
                    }
                    for (int i = 0; i < weight.mins.size(); i++) {
                        halfMins.push_back(__float2half(weight.mins[i]));
                    }
                    int curLen = 0;
                    for (auto &it : div) {
                        state = cudaMemcpy(((half*)cudaScales) + curLen * weightGroup, halfScales.data() + it.first * weightGroup, (it.second - it.first) * weightGroup * sizeof(half), GetCudaMemcpyType(mallocType, 0));
                        state = cudaMemcpy(((half*)cudaMins) + curLen * weightGroup, halfMins.data() + it.first * weightGroup, (it.second - it.first) * weightGroup * sizeof(half), GetCudaMemcpyType(mallocType, 0));
                        curLen += (it.second - it.first);
                    }
                } else {
                    cudaScales = (float*)AutoMalloc(len * weightGroup * sizeof(float), mallocType);
                    cudaMins = (float*)AutoMalloc(len * weightGroup * sizeof(float), mallocType);
                    cudaZeropoints = (uint8_t*)AutoMalloc(len * weightGroup, mallocType);

                    int curLen = 0;
                    for (auto &it : div) {
                        state = cudaMemcpy(cudaScales + curLen * weightGroup, weight.scales.data() + it.first * weightGroup, (it.second - it.first) * weightGroup * sizeof(float), GetCudaMemcpyType(mallocType, 0));
                        state = cudaMemcpy(cudaMins + curLen * weightGroup, weight.mins.data() + it.first * weightGroup, (it.second - it.first) * weightGroup * sizeof(float), GetCudaMemcpyType(mallocType, 0));
                        state = cudaMemcpy(cudaZeropoints + curLen * weightGroup, zeropoints.data() + it.first * weightGroup, (it.second - it.first) * weightGroup, GetCudaMemcpyType(mallocType, 0));
                        curLen += (it.second - it.first);
                    }
                }
            } else {
                if (weight.dataType == fastllm::DataType::INT4_GROUP) {
                    cudaScales = (float*)AutoMalloc(k * weightGroup * sizeof(half), mallocType);
                    cudaMins = (float*)AutoMalloc(k * weightGroup * sizeof(half), mallocType);
                    int base = div[0].first / weight.groupCnt;
                    std::vector <half> halfScales, halfMins;
                    for (int i = 0; i < weight.scales.size(); i++) {
                        halfScales.push_back(__float2half(i + base < weight.scales.size() ? weight.scales[i + base] : 0.0f));
                    }
                    for (int i = 0; i < weight.mins.size(); i++) {
                        halfMins.push_back(__float2half(i + base < weight.mins.size() ? weight.mins[i + base] : 0.0f));
                    }
                    state = cudaMemcpy(cudaScales, halfScales.data(), k * weightGroup * sizeof(half), GetCudaMemcpyType(mallocType, 0));
                    state = cudaMemcpy(cudaMins, halfMins.data(), k * weightGroup * sizeof(half), GetCudaMemcpyType(mallocType, 0));
                } else {
                    cudaScales = (float*)AutoMalloc(k * weightGroup * sizeof(float), mallocType);
                    cudaMins = (float*)AutoMalloc(k * weightGroup * sizeof(float), mallocType);
                    cudaZeropoints = (uint8_t*)AutoMalloc(k * weightGroup, mallocType);
                    state = cudaMemcpy(cudaScales, weight.scales.data(), k * weightGroup * sizeof(float), GetCudaMemcpyType(mallocType, 0));
                    state = cudaMemcpy(cudaMins, weight.mins.data(), k * weightGroup * sizeof(float), GetCudaMemcpyType(mallocType, 0));
                    state = cudaMemcpy(cudaZeropoints, zeropoints.data(), k * weightGroup, GetCudaMemcpyType(mallocType, 0));
                }
            }
                
            weight.extraCudaData.push_back((void*)cudaScales);
            weight.extraCudaData.push_back((void*)cudaMins);
            weight.extraCudaData.push_back((void*)cudaZeropoints);
        }
    }

    if (cudaSuccess != state) {
        checkCudaErrors("Error: CUDA error when split weight!", state);
        return false;
    }

    cudaSetDevice(0);
    FastllmCudaFree(weight.cudaData);
    FastllmCudaFree(cudaBiasData);
    weight.cudaData = nullptr;
    weight.weightSum.clear();
    return true;
}

bool PrepareMultiCudaHalfWeight(fastllm::Data &weight, const fastllm::Data &bias, DivisionScheme divisionScheme, int splitAxis) {
    int k = weight.dims[0], m = weight.dims[1];
    cudaError_t state = cudaSuccess;
    half *cudaBiasData = (half*)FastllmCudaMalloc(k * sizeof(half));
    if (bias.dims.size() > 0) {
        float *tempBiasData = (float*)FastllmCudaMalloc(k * sizeof(float));            
        state = cudaMemcpy(tempBiasData, (uint8_t *) bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
        int threadPerBlock = std::min(256, k);
        FastllmCudaFloat2HalfKernel <<< (k - 1) / threadPerBlock + 1, threadPerBlock>>>(tempBiasData, cudaBiasData, k);
        FastllmCudaFree(tempBiasData);
    } else {
        state = cudaMemset(cudaBiasData, 0, k * sizeof(half));
    }
        
    for (int i = 0; i < multiCudaCurrentDevices.size(); i++) {
        int deviceId = multiCudaCurrentDevices[i], mallocType = 0;
        std::string specialId = "";
        SwitchDeviceAndGetInfos(deviceId, specialId, mallocType);

        auto &div = divisionScheme[deviceId];
        int len = 0;
        for (auto &it : div) {
            len += it.second - it.first;
        }

        half *deviceBiasData;
        cudaError_t state = cudaSuccess;

        if (splitAxis == 0) {
            deviceBiasData = (half*)AutoMalloc(len * sizeof(half), mallocType);
            int curLen = 0;
            for (auto &it : div) {
                state = cudaMemcpy(deviceBiasData + curLen, cudaBiasData + it.first, (it.second - it.first) * sizeof(half), GetCudaMemcpyType(mallocType, 1));
                curLen += (it.second - it.first);
            }
        } else {
            deviceBiasData = (half*)AutoMalloc(k * sizeof(half), mallocType);
            if (i == 0) {
                state = cudaMemcpy(deviceBiasData, cudaBiasData, k * sizeof(half), GetCudaMemcpyType(mallocType, 1));
            } else {
                state = AutoMemset(deviceBiasData, 0, k * sizeof(half), mallocType);
            }
        }

        weight.extraCudaHalfData.push_back((void *) weight.extraCudaData[i * 2]);
        weight.extraCudaHalfData.push_back((void *) deviceBiasData);

        if (cudaSuccess != state) {
            checkCudaErrors("Error: CUDA error when split weight!", state);
            return false;
        }
    }

    // 1. mins, scales
    if (weight.mins.size() > 0) {
        for (int i = 0; i < multiCudaCurrentDevices.size(); i++) {
            weight.extraCudaHalfData.push_back(weight.extraCudaData[multiCudaCurrentDevices.size() * 2 + i * 3 + 0]);
            weight.extraCudaHalfData.push_back(weight.extraCudaData[multiCudaCurrentDevices.size() * 2 + i * 3 + 1]);
            weight.extraCudaHalfData.push_back(weight.extraCudaData[multiCudaCurrentDevices.size() * 2 + i * 3 + 2]);
        }
    }
        
    if (cudaSuccess != state) {
        checkCudaErrors("Error: CUDA error when split weight!", state);
        return false;
    }

    cudaSetDevice(0);
    FastllmCudaFree(weight.cudaData);
    FastllmCudaFree(cudaBiasData);
    weight.cudaData = nullptr;
    return true;
}

template <typename T>
bool FastllmMultiCudaMatMulInner(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    std::vector <int> points = FastllmMultiCudaGetSplitPoints(k, weight.groupCnt <= 0 ? 1 : weight.groupCnt);
    DivisionScheme divisionScheme;
    for (int i = 0; i < multiCudaCurrentDevices.size(); i++) {
        int deviceId = multiCudaCurrentDevices[i];
        divisionScheme[deviceId].push_back(std::make_pair(points[i], points[i + 1]));
    }
    if ((weight.extraCudaData.size() == 0)) {
        if (!PrepareMultiCudaWeight(weight, bias, divisionScheme, 0)) {
            return false;
        }
    }
    if (input.dataType == fastllm::DataType::FLOAT16 && weight.extraCudaHalfData.size() == 0) {
        if (!PrepareMultiCudaHalfWeight(weight, bias, divisionScheme, 0)) {
            return false;
        }
    }

    T *cudaInput = (T *) FastllmCudaPrepareInput(input);
    T *cudaOutput = (T *) FastllmCudaPrepareOutput(output);

    auto *pool = fastllm::GetAlivePool();
    std::vector<fastllm::MultiThreadBaseOp*> ops;

    int isQuantWeight = (weight.mins.size() > 0);
    int threadNum = multiCudaCurrentDevices.size();

    T *cpuInput = new T[input.Count(0)];
    cudaMemcpy(cpuInput, cudaInput, input.GetBytes(), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (int i = 0; i < threadNum; i++) {
        int deviceId = multiCudaCurrentDevices[i], mallocType = 0;
        std::string specialId = "";
        SwitchDeviceAndGetInfos(deviceId, specialId, mallocType);
        if (specialId != "cpu") {
            continue;
        }
        int start = points[i], len = points[i + 1] - points[i];
        std::vector <void*> datas = weight.extraCudaData;
        if (typeid(T) == typeid(half)) {
            datas = weight.extraCudaHalfData;
        }
        ops.push_back(new fastllm::MultiCudaCpuMatMulSingleOp <T> (
            &weight, datas[i * 2], weight.dataType,
            cpuInput, cudaInput, cudaOutput, 
            (T*) datas[i * 2 + 1],
            n, m, k, start, len, bias.dims.size() > 0,
            (float*)(isQuantWeight ? datas[threadNum * 2 + i * 3] : nullptr),
            (float*)(isQuantWeight ? datas[threadNum * 2 + i * 3 + 1] : nullptr),
            (uint8_t*)(isQuantWeight ? datas[threadNum * 2 + i * 3 + 2] : nullptr),
            weight.group, weight.groupCnt, pool, threadNum, pool->threads.size() - threadNum
        ));
    }
    for (int i = 0; i < threadNum; i++) {
        int deviceId = multiCudaCurrentDevices[i], mallocType = 0;
        std::string specialId = "";
        SwitchDeviceAndGetInfos(deviceId, specialId, mallocType);
        if (specialId != "") {
            continue;
        }
        int start = points[i], len = points[i + 1] - points[i];
        std::vector <void*> datas = weight.extraCudaData;
        if (typeid(T) == typeid(half)) {
            datas = weight.extraCudaHalfData;
        }
        ops.push_back(new fastllm::MultiCudaMatMulSingleOp <T> (
            deviceId, datas[i * 2], weight.dataType,
            cpuInput, cudaInput, cudaOutput, 
            (T*) datas[i * 2 + 1],
            n, m, k, start, len, bias.dims.size() > 0,
            (float*)(isQuantWeight ? datas[threadNum * 2 + i * 3] : nullptr),
            (float*)(isQuantWeight ? datas[threadNum * 2 + i * 3 + 1] : nullptr),
            (uint8_t*)(isQuantWeight ? datas[threadNum * 2 + i * 3 + 2] : nullptr),
            weight.group, weight.groupCnt
        ));
    }
    for (int i = 0; i < threadNum; i++) {
        pool->PushOp(i, ops[i]);
    }
    for (int i = 0; i < threadNum; i++) {
        pool->Wait(i);
        delete ops[i];
    }

    delete[] cpuInput;
    cudaSetDevice(0);
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmMultiCudaMatMul(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    if (input.dataType == fastllm::DataType::FLOAT32) {
        return FastllmMultiCudaMatMulInner <float> (input, weight, bias, output, n, m, k);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        return FastllmMultiCudaMatMulInner <half> (input, weight, bias, output, n, m, k);
    } else {
        printf("Fastllm Multicuda MatMul Error: input's type error.\n");
        exit(0);
    }
    return false;
}

std::vector <bool> streamInits = std::vector <bool> (4, 0);
cudaStream_t streams[4];

cudaStream_t *GetFastllmStream(int id) {
    if (!streamInits[id]) {
        streamInits[id] = true;
        cudaSetDevice(id);
        cudaStreamCreate(&streams[id]);
        cudaSetDevice(0);
    }
    return &streams[id];
}

template <typename T>
bool FastllmMultiCudaMLPInner(const fastllm::Data &input, fastllm::Data &weight0, fastllm::Data &weight1, fastllm::Data &output) {
    int deviceNum = multiCudaCurrentDevices.size();
    for (int i = 0; i < deviceNum; i++) {
        if (specialDeviceIds.find(multiCudaCurrentDevices[i]) == specialDeviceIds.end()) {
            cudaSetDevice(multiCudaCurrentDevices[i]);
        } 
        for (int j = 0; j < deviceNum; j++) {
            if (i != j) {
                if (specialDeviceIds.find(multiCudaCurrentDevices[j]) == specialDeviceIds.end()) {
                    cudaDeviceEnablePeerAccess(multiCudaCurrentDevices[j], 0);
                }
            }
        }
    }
    cudaSetDevice(0);

    std::vector <int> points = FastllmMultiCudaGetSplitPoints(weight0.dims[0] / 2, weight0.groupCnt <= 0 ? 1 : weight0.groupCnt);
    if ((weight0.extraCudaData.size() == 0)) {
        int mid = weight0.dims[0] / 2;
        DivisionScheme divisionScheme;
        for (int i = 0; i < multiCudaCurrentDevices.size(); i++) {
            int deviceId = multiCudaCurrentDevices[i];
            divisionScheme[deviceId].push_back(std::make_pair(points[i], points[i + 1]));
        }
        if (!PrepareMultiCudaWeight(weight1, fastllm::Data(), divisionScheme, 1)) {
            return false;
        }
        for (int i = 0; i < multiCudaCurrentDevices.size(); i++) {
            int deviceId = multiCudaCurrentDevices[i];
            divisionScheme[deviceId].push_back(std::make_pair(mid + points[i], mid + points[i + 1]));
        }
        if (!PrepareMultiCudaWeight(weight0, fastllm::Data(), divisionScheme, 0)) {
            return false;
        }
    }

    if (input.dataType == fastllm::DataType::FLOAT16 && (weight0.extraCudaHalfData.size() == 0)) {
        int mid = weight0.dims[0] / 2;
        DivisionScheme divisionScheme;
        for (int i = 0; i < multiCudaCurrentDevices.size(); i++) {
            int deviceId = multiCudaCurrentDevices[i];
            divisionScheme[deviceId].push_back(std::make_pair(points[i], points[i + 1]));
        }
        if (!PrepareMultiCudaHalfWeight(weight1, fastllm::Data(), divisionScheme, 1)) {
            return false;
        }
        for (int i = 0; i < multiCudaCurrentDevices.size(); i++) {
            int deviceId = multiCudaCurrentDevices[i];
            divisionScheme[deviceId].push_back(std::make_pair(mid + points[i], mid + points[i + 1]));
        }
        if (!PrepareMultiCudaHalfWeight(weight0, fastllm::Data(), divisionScheme, 1)) {
            return false;
        }
    }

    {
        // fused MLP
        T *cudaInput = (T *) FastllmCudaPrepareInput(input);
        T *cudaOutput = (T *) FastllmCudaPrepareOutput(output);

        std::vector <uint8_t*> curInputs, curOutputs;
        auto *pool = fastllm::GetAlivePool();
        std::vector<fastllm::MultiThreadBaseOp*> ops;

        int threadNum = multiCudaCurrentDevices.size();
        curInputs.resize(threadNum);
        curOutputs.resize(threadNum);

        T *cpuInput = new T[input.Count(0)];
        if (threadNum > 1) {
            cudaMemcpy(cpuInput, cudaInput, input.GetBytes(), cudaMemcpyDeviceToHost);
            // cudaDeviceSynchronize();
        }

        T *partOutput = (T*)FastllmCudaMalloc(output.GetBytes() * threadNum);

        std::vector <int> points = FastllmMultiCudaGetSplitPoints(weight0.dims[0] / 2, weight0.groupCnt <= 0 ? 1 : weight0.groupCnt); // 因为要swiglu，所以先/2
        for (int i = 0; i < threadNum; i++) {
            int deviceId = multiCudaCurrentDevices[i], mallocType = 0;
            std::string specialId = "";
            SwitchDeviceAndGetInfos(deviceId, specialId, mallocType);
            if (specialId != "cpu") {
                continue;
            }
            int start = points[i], len = points[i + 1] - points[i];
            ops.push_back(new fastllm::MultiCudaCpuMLPSingleOp <T>  (
                deviceId, &weight0, &weight1,
                (int)input.Count(0) / input.dims.back(), input.dims.back(), len * 2, weight1.dims[0], 
                cudaInput, cudaOutput, (T**)&curInputs[i], (T**)&curOutputs[i], 
                cpuInput, partOutput + output.Count(0) * i, threadNum, i, pool, threadNum, pool->threads.size() - threadNum
            ));
        }
        for (int i = 0; i < threadNum; i++) {
            int deviceId = multiCudaCurrentDevices[i], mallocType = 0;
            std::string specialId = "";
            SwitchDeviceAndGetInfos(deviceId, specialId, mallocType);
            if (specialId != "") {
                continue;
            }
            int start = points[i], len = points[i + 1] - points[i];
            ops.push_back(new fastllm::MultiCudaMLPSingleOp <T>  (
                deviceId, &weight0, &weight1,
                (int)input.Count(0) / input.dims.back(), input.dims.back(), len * 2, weight1.dims[0], 
                cudaInput, cudaOutput, (T**)&curInputs[i], (T**)&curOutputs[i], 
                cpuInput, partOutput + output.Count(0) * i, threadNum, i
            ));
        }
        for (int i = 1; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        ops[0]->Run();
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }

        delete[] cpuInput;
        cudaSetDevice(0);
        if (threadNum > 1) {
            int len = output.Count(0);
            int threadPerBlock = std::min(256, len);
            FastllmReduceKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaOutput, partOutput, len, threadNum);
        }

        std::set <void*> releaseSpaces;
        for (int t = 0; t < threadNum; t++) {
            if ((long long)curOutputs[t] != (long long)cudaOutput) {
                releaseSpaces.insert(curOutputs[t]);
            }
            if ((long long)curInputs[t] != (long long)cudaInput) {
                releaseSpaces.insert(curInputs[t]);
            }
        }
        releaseSpaces.insert(partOutput);
        for (auto it : releaseSpaces) {
            FastllmCudaFree(it);
        }

        FastllmCudaFinishInput(input, cudaInput);
        FastllmCudaFinishOutput(output, cudaOutput);
    }
    return true;
}

bool FastllmMultiCudaMLP(const fastllm::Data &input, fastllm::Data &weight0, fastllm::Data &weight1, fastllm::Data &output) {
    if (input.dataType == fastllm::DataType::FLOAT32) {
        return FastllmMultiCudaMLPInner <float> (input, weight0, weight1, output);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        return FastllmMultiCudaMLPInner <half> (input, weight0, weight1, output);
    } else {
        printf("Fastllm Multicuda Mlp Error: input's type error.\n");
        exit(0);
    }
    return false;
}