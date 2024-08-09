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

#include "devices/cpu/alivethreadpool.h"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 // support tensor core
#include "mma.h"
using namespace nvcuda;
#endif

#define checkCudaErrors(message, val) showError(val, message, __FILE__, __LINE__)

extern __global__ void FastllmCudaBiasKernel(float *a, float *bias, int k);
extern __global__ void FastllmCudaBiasKernel(half *a, half *bias, int k);
extern __global__ void FastllmCudaFloat2HalfKernel(float* a, half *b, int len);
extern __global__ void FastllmCudaHalf2FloatKernel(half* a, float *b, int len);
extern __global__ void FastllmCudaInt42HalfKernel(uint8_t* a, float *scales, float *mins, half *b, int len, int per);
extern __global__ void FastllmCudaInt4Group2HalfKernel(uint8_t* a, float *scales, float *mins, half *b, int len, int per, int group, int groupCnt);
extern __global__ void FastllmCudaInt82HalfKernel(uint8_t* a, float *scales, uint8_t *zeros, half *b, int len, int per);

extern void showError(cudaError_t result, char const* const message, const char* const file, int const line);
extern cublasHandle_t getFastllmCublasHandle();
extern void *FastllmCudaPrepareInput(const fastllm::Data &input);
extern void *FastllmCudaPrepareOutput(fastllm::Data &output);
extern void FastllmCudaFinishInput(const fastllm::Data &input, void *data);
extern void FastllmCudaFinishOutput(fastllm::Data &output, void *data);

extern void LaunchFastllmGemmFp16Fp16(half *input, half *weight, half *output, half *bias, int n, int m, int k);
extern void LaunchFastllmGemmFp16Int8(half *input, uint8_t *weight, half *output, half *bias, float *scales, uint8_t *zeros, int n, int m, int k);
extern void LaunchFastllmGemmFp16Int4NoZero(half *input, uint8_t *weight, half *output, half *bias, float *scales, float *mins, int n, int m, int k);
extern void LaunchFastllmGemmFp16Int4Group(half *input, uint8_t *weight, half *output, half *bias, float *scales, float *mins, int n, int m, int k, int group, int groupCnt);

extern void LaunchFastllmGemmFp32Fp16(float *input, half *weight, float *output, float *bias, int n, int m, int k);
extern void LaunchFastllmGemmFp32Int8(float *input, uint8_t *weight, float *output, float *bias, float *scales, uint8_t *zeros, int n, int m, int k);
extern void LaunchFastllmGemmFp32Int4NoZero(float *input, uint8_t *weight, float *output, float *bias, float *scales, float *mins, int n, int m, int k);
extern void LaunchFastllmGemmFp32Int4Group(float *input, uint8_t *weight, float *output, float *bias, float *scales, float *mins, int n, int m, int k, int group, int groupCnt);

std::vector <int> multiCudaCurrentDevices;

void FastllmMultiCudaSetDevice(std::vector <int> ids) {
    multiCudaCurrentDevices = ids;
}

namespace fastllm {
    struct MultiCudaMatMulFloat16SingleOp : MultiThreadBaseOp {
        int deviceId;
        void *weight;
        DataType weightDataType;
        half *cudaInput, *cudaOutput, *bias;
        int n, m, k, start, len;
        bool hasBias;
        float *scales, *mins;
        uint8_t *zeros;
        int group, groupCnt;

        MultiCudaMatMulFloat16SingleOp(int deviceId, void *weight, DataType weightDataType, half *cudaInput, half *cudaOutput, half *bias, 
                                    int n, int m, int k, int start, int len, bool hasBias, float *scales, float *mins, uint8_t *zeros, int group, int groupCnt)
            : deviceId(deviceId), weight(weight), weightDataType(weightDataType), cudaInput(cudaInput), cudaOutput(cudaOutput), bias(bias), 
            n(n), m(m), k(k), start(start), len(len), hasBias(hasBias), scales(scales), mins(mins), zeros(zeros), group(group), groupCnt(groupCnt) {}

        void Run() {
            cudaSetDevice(deviceId);
            half *curInput = cudaInput; 
            half *curOutput = cudaOutput; 

            if (deviceId != 0 || n > 1) {
                curInput = (half*)FastllmCudaMalloc(n * m * sizeof(half));                
                curOutput = (half*)FastllmCudaMalloc(n * len * sizeof(half));
                
                cudaMemcpy(curInput, cudaInput, n * m * sizeof(half), cudaMemcpyDeviceToDevice);
            }

            if (weightDataType == DataType::FLOAT16 && n < 8 && false) {
                LaunchFastllmGemmFp16Fp16(curInput, (half*)weight, curOutput, bias, n, m, len);
            } else if (weightDataType == DataType::INT8 && n < 8) {
                LaunchFastllmGemmFp16Int8(curInput, (uint8_t*)weight, curOutput, bias, scales, zeros, n, m, len);
            } else if (weightDataType == DataType::INT4_NOZERO && n < 8) {
                LaunchFastllmGemmFp16Int4NoZero(curInput, (uint8_t*)weight, curOutput, bias, scales, mins, n, m, len);
            } else if (weightDataType == DataType::INT4_GROUP && n < 8) {
                LaunchFastllmGemmFp16Int4Group(curInput, (uint8_t*)weight, curOutput, bias, scales, mins, n, m, len, group, groupCnt);
            } else {
                __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
                auto fastllmCublasHandle = getFastllmCublasHandle();
                cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
                cublasStatus_t status;

                half *fp16Weight = (half*)weight;
                bool isQuant = false;
                if (weightDataType == DataType::INT4_NOZERO) {
                    int threadPerBlock = std::min(256, len * m);
                    isQuant = true;
                    fp16Weight = (half*)FastllmCudaMalloc(len * m * sizeof(half));
                    FastllmCudaInt42HalfKernel <<< (len * m - 1) / threadPerBlock + 1, threadPerBlock>>>((uint8_t*)weight, scales, mins, fp16Weight, len * m, m);
                } else if (weightDataType == DataType::INT4_GROUP) {
                    int threadPerBlock = std::min(256, len * m);
                    isQuant = true;
                    fp16Weight = (half*)FastllmCudaMalloc(len * m * sizeof(half));
                    FastllmCudaInt4Group2HalfKernel <<< (len * m - 1) / threadPerBlock + 1, threadPerBlock>>>((uint8_t*)weight, scales, mins, fp16Weight, len * m, m, group, groupCnt);
                } else if (weightDataType == DataType::INT8) {
                    int threadPerBlock = std::min(256, len * m);
                    isQuant = true;
                    fp16Weight = (half*)FastllmCudaMalloc(len * m * sizeof(half));
                    FastllmCudaInt82HalfKernel <<< (len * m - 1) / threadPerBlock + 1, threadPerBlock>>>((uint8_t*)weight, scales, zeros, fp16Weight, len * m, m);
                }

                status = cublasGemmEx(fastllmCublasHandle,
                                        CUBLAS_OP_T, CUBLAS_OP_N,
                                        len, n, m, &h_alpha, fp16Weight, AType, 
                                        m, curInput, BType,
                                        m, &h_beta,
                                        curOutput, CType,
                                        len, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
                if (status != CUBLAS_STATUS_SUCCESS) {
                    printf("Error: cublas error.\n");
                    throw ("cublas error");
                    exit(0);
                }            

                if (hasBias) {
                    FastllmCudaBiasKernel <<< n, 256 >>>(curOutput, bias, len);
                }

                if (isQuant) {
                    FastllmCudaFree(fp16Weight);
                }
            }

            if (deviceId != 0 || n > 1) {
                cudaMemcpy2D(cudaOutput + start, k * sizeof(half), curOutput, len * sizeof(half), len * sizeof(half), n, cudaMemcpyDeviceToDevice);
                FastllmCudaFree(curInput);
                FastllmCudaFree(curOutput);
            }
        }
    };

    struct MultiCudaMatMulFloat32SingleOp : MultiThreadBaseOp {
        int deviceId;
        void *weight;
        DataType weightDataType;
        float *cudaInput, *cudaOutput, *bias;
        int n, m, k, start, len;
        bool hasBias;
        float *scales, *mins;
        uint8_t *zeros;
        int group, groupCnt;

        MultiCudaMatMulFloat32SingleOp(int deviceId, void *weight, DataType weightDataType, float *cudaInput, float *cudaOutput, float *bias, 
                                    int n, int m, int k, int start, int len, bool hasBias, float *scales, float *mins, uint8_t *zeros, int group, int groupCnt)
            : deviceId(deviceId), weight(weight), weightDataType(weightDataType), cudaInput(cudaInput), cudaOutput(cudaOutput), bias(bias), 
            n(n), m(m), k(k), start(start), len(len), hasBias(hasBias), scales(scales), mins(mins), zeros(zeros), group(group), groupCnt(groupCnt) {}

        void Run() {
            cudaSetDevice(deviceId);
            float *curInput = cudaInput; 
            float *curOutput = cudaOutput; 

            if (deviceId != 0 || n > 1) {
                curInput = (float*)FastllmCudaMalloc(n * m * sizeof(float));                
                curOutput = (float*)FastllmCudaMalloc(n * len * sizeof(float));
                
                cudaMemcpy(curInput, cudaInput, n * m * sizeof(float), cudaMemcpyDeviceToDevice);
            }
            
            if (weightDataType == DataType::FLOAT16 && n < 8 && false) {
                LaunchFastllmGemmFp32Fp16(curInput, (half*)weight, curOutput, bias, n, m, len);
            } else if (weightDataType == DataType::INT8 && n < 8) {
                LaunchFastllmGemmFp32Int8(curInput, (uint8_t*)weight, curOutput, bias, scales, zeros, n, m, len);
            } else if (weightDataType == DataType::INT4_NOZERO && n < 8) {
                LaunchFastllmGemmFp32Int4NoZero(curInput, (uint8_t*)weight, curOutput, bias, scales, mins, n, m, len);
            } else if (weightDataType == DataType::INT4_GROUP && n < 8) {
                LaunchFastllmGemmFp32Int4Group(curInput, (uint8_t*)weight, curOutput, bias, scales, mins, n, m, len, group, groupCnt);
            } else {
                auto fastllmCublasHandle = getFastllmCublasHandle();
                half *cudaFp16Input, *cudaFp16Output;
                cudaFp16Input = (half *) FastllmCudaMalloc(n * m * sizeof(half));
                cudaFp16Output = (half *) FastllmCudaMalloc(n * len * sizeof(half));

                __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
                cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
                cublasStatus_t status;

                int threadPerBlock = std::min(256, n * m);
                FastllmCudaFloat2HalfKernel <<< (n * m - 1) / threadPerBlock + 1, threadPerBlock>>>(curInput, cudaFp16Input, n * m);

                half *fp16Weight = (half*)weight;
                bool isQuant = false;
                if (weightDataType == DataType::INT4_NOZERO) {
                    int threadPerBlock = std::min(256, len * m);
                    isQuant = true;
                    fp16Weight = (half*)FastllmCudaMalloc(len * m * sizeof(half));
                    FastllmCudaInt42HalfKernel <<< (len * m - 1) / threadPerBlock + 1, threadPerBlock>>>((uint8_t*)weight, scales, mins, fp16Weight, len * m, m);
                } else if (weightDataType == DataType::INT4_GROUP) {
                    int threadPerBlock = std::min(256, len * m);
                    isQuant = true;
                    fp16Weight = (half*)FastllmCudaMalloc(len * m * sizeof(half));
                    FastllmCudaInt4Group2HalfKernel <<< (len * m - 1) / threadPerBlock + 1, threadPerBlock>>>((uint8_t*)weight, scales, mins, fp16Weight, len * m, m, group, groupCnt);
                } else if (weightDataType == DataType::INT8) {
                    int threadPerBlock = std::min(256, len * m);
                    isQuant = true;
                    fp16Weight = (half*)FastllmCudaMalloc(len * m * sizeof(half));
                    FastllmCudaInt82HalfKernel <<< (len * m - 1) / threadPerBlock + 1, threadPerBlock>>>((uint8_t*)weight, scales, zeros, fp16Weight, len * m, m);
                }

                status = cublasGemmEx(fastllmCublasHandle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    len, n, m,
                                    &h_alpha, fp16Weight, AType,
                                    m, cudaFp16Input, BType,
                                    m, &h_beta,
                                    cudaFp16Output, CType,
                                    len, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
                if (status != CUBLAS_STATUS_SUCCESS) {
                    printf("Error: cublas error.\n");
                    throw("cublas error");
                    exit(0);
                }

                FastllmCudaHalf2FloatKernel <<< (n * len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaFp16Output, curOutput, n * len);
                if (hasBias) {
                    FastllmCudaBiasKernel <<< n, 256 >>> (curOutput, bias, len);
                }

                FastllmCudaFree(cudaFp16Input);
                FastllmCudaFree(cudaFp16Output);

                if (isQuant) {
                    FastllmCudaFree(fp16Weight);
                }
            }
            if (deviceId != 0 || n > 1) {
                cudaMemcpy2D(cudaOutput + start, k * sizeof(float), curOutput, len * sizeof(float), len * sizeof(float), n, cudaMemcpyDeviceToDevice);
                FastllmCudaFree(curInput);
                FastllmCudaFree(curOutput);
            }
        }
    };
}

// 将total个计算任务切分
// 若当前有x个设备，返回一个长度为(x + 1)的vector，第i个设备执行任务[ret[i], ret[i + 1])
std::vector <int> FastllmMultiCudaGetSplitPoints(int total) {
    std::vector <int> ret;
    int deviceNum = multiCudaCurrentDevices.size();
    int cur = 0, per = total / deviceNum;
    for (int i = 0; i < deviceNum; i++) {
        int end = (i == deviceNum - 1 ? total : cur + per + (cur + per * (deviceNum - i) < total));
        ret.push_back(cur);
        if (i == deviceNum - 1) {
            ret.push_back(end);
        }
        cur = end;
    }

    return ret;
}

bool FastllmMultiCudaMatMul(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    std::vector <int> points = FastllmMultiCudaGetSplitPoints(k);
    if ((weight.extraCudaData.size() == 0)) {
        cudaError_t state = cudaSuccess;
        float *cudaBiasData = (float*)FastllmCudaMalloc(k * sizeof(float));
        if (bias.dims.size() > 0) {
            state = cudaMemcpy(cudaBiasData, (uint8_t *) bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(float));
        }
        
        for (int i = 0; i < multiCudaCurrentDevices.size(); i++) {
            int deviceId = multiCudaCurrentDevices[i];
            int start = points[i], len = points[i + 1] - points[i];
            cudaSetDevice(deviceId);

            void *deviceWeightData;
            float *deviceBiasData;
            cudaError_t state = cudaSuccess;
            deviceWeightData = (void*)FastllmCudaMalloc(len * m * weight.unitSize / weight.unitSizeDiv);
            deviceBiasData = (float*)FastllmCudaMalloc(len * sizeof(float));
            state = cudaMemcpy(deviceWeightData, (uint8_t*)weight.cudaData + start * m * weight.unitSize / weight.unitSizeDiv, 
                            len * m * weight.unitSize / weight.unitSizeDiv, cudaMemcpyDeviceToDevice);
            state = cudaMemcpy(deviceBiasData, cudaBiasData + start, len * sizeof(float), cudaMemcpyDeviceToDevice);

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
                int deviceId = multiCudaCurrentDevices[i];
                int start = points[i] * weightGroup, len = (points[i + 1] - points[i]) * weightGroup;
                cudaSetDevice(deviceId);

                float *cudaScales = (float*)FastllmCudaMalloc(len * sizeof(float));
                float *cudaMins = (float*)FastllmCudaMalloc(len * sizeof(float));
                uint8_t *cudaZeropoints = (uint8_t*)FastllmCudaMalloc(len);

                state = cudaMemcpy(cudaScales, weight.scales.data() + start, len * sizeof(float), cudaMemcpyHostToDevice);
                state = cudaMemcpy(cudaMins, weight.mins.data() + start, len * sizeof(float), cudaMemcpyHostToDevice);
                state = cudaMemcpy(cudaZeropoints, zeropoints.data() + start, len, cudaMemcpyHostToDevice);
                
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
    }

    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareOutput(output);

    auto *pool = fastllm::GetAlivePool();
    std::vector<fastllm::MultiCudaMatMulFloat32SingleOp*> ops;

    int isQuantWeight = (weight.mins.size() > 0);
    int threadNum = multiCudaCurrentDevices.size();
    for (int i = 0; i < threadNum; i++) {
        int deviceId = multiCudaCurrentDevices[i];
        int start = points[i], len = points[i + 1] - points[i];
        ops.push_back(new fastllm::MultiCudaMatMulFloat32SingleOp(
            deviceId, weight.extraCudaData[i * 2], weight.dataType,
            cudaInput, cudaOutput, 
            (float*) weight.extraCudaData[i * 2 + 1],
            n, m, k, start, len, bias.dims.size() > 0,
            (float*)(isQuantWeight ? weight.extraCudaData[threadNum * 2 + i * 3] : nullptr),
            (float*)(isQuantWeight ? weight.extraCudaData[threadNum * 2 + i * 3 + 1] : nullptr),
            (uint8_t*)(isQuantWeight ? weight.extraCudaData[threadNum * 2 + i * 3 + 2] : nullptr),
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

    cudaSetDevice(0);
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmMultiCudaHalfMatMul(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    std::vector <int> points = FastllmMultiCudaGetSplitPoints(k);

    if ((weight.extraCudaHalfData.size() == 0)) {
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
            int deviceId = multiCudaCurrentDevices[i];
            int start = points[i], len = points[i + 1] - points[i];
            cudaSetDevice(deviceId);

            half *deviceBiasData;
            cudaError_t state = cudaSuccess;
            deviceBiasData = (half*)FastllmCudaMalloc(len * sizeof(half));
            state = cudaMemcpy(deviceBiasData, cudaBiasData + start, len * sizeof(half), cudaMemcpyDeviceToDevice);

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
    }

    half *cudaInput = (half *) FastllmCudaPrepareInput(input);
    half *cudaOutput = (half *) FastllmCudaPrepareOutput(output);

    auto *pool = fastllm::GetAlivePool();
    std::vector<fastllm::MultiCudaMatMulFloat16SingleOp*> ops;

    int isQuantWeight = (weight.mins.size() > 0);
    int threadNum = multiCudaCurrentDevices.size();
    for (int i = 0; i < threadNum; i++) {
        int deviceId = multiCudaCurrentDevices[i];
        int start = points[i], len = points[i + 1] - points[i];
        ops.push_back(new fastllm::MultiCudaMatMulFloat16SingleOp(
            deviceId, weight.extraCudaHalfData[i * 2], weight.dataType,
            cudaInput, cudaOutput, 
            (half *) weight.extraCudaHalfData[i * 2 + 1],
            n, m, k, start, len, bias.dims.size() > 0,
            (float*)(isQuantWeight ? weight.extraCudaHalfData[threadNum * 2 + i * 3] : nullptr),
            (float*)(isQuantWeight ? weight.extraCudaHalfData[threadNum * 2 + i * 3 + 1] : nullptr),
            (uint8_t*)(isQuantWeight ? weight.extraCudaHalfData[threadNum * 2 + i * 3 + 2] : nullptr),
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

    cudaSetDevice(0);
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}