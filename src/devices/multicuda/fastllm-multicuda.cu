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

extern __global__ void FastllmSwigluKernel(half* __restrict__ a, half* __restrict__ b, int len, int spatial, int mid);
extern __global__ void FastllmSwigluKernel(float* __restrict__ a, float* __restrict__ b, int len, int spatial, int mid);

extern __global__ void FastllmCudaBiasKernel(float *a, float *bias, int k);
extern __global__ void FastllmCudaBiasKernel(half *a, half *bias, int k);
extern __global__ void FastllmCudaFloat2HalfKernel(float* a, half *b, int len);
extern __global__ void FastllmCudaHalf2FloatKernel(half* a, float *b, int len);
extern __global__ void FastllmCudaInt42HalfKernel(uint8_t* a, float *scales, float *mins, half *b, int len, int per);
extern __global__ void FastllmCudaInt4Group2HalfKernel(uint8_t* a, float *scales, float *mins, half *b, int len, int per, int group, int groupCnt);
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
extern void LaunchFastllmGemmFp16Int4Group(half *input, uint8_t *weight, half *output, half *bias, float *scales, float *mins, int n, int m, int k, int group, int groupCnt);

extern void LaunchFastllmGemmFp32Fp16(float *input, half *weight, float *output, float *bias, int n, int m, int k);
extern void LaunchFastllmGemmFp32Int8(float *input, uint8_t *weight, float *output, float *bias, float *scales, uint8_t *zeros, int n, int m, int k);
extern void LaunchFastllmGemmFp32Int4NoZero(float *input, uint8_t *weight, float *output, float *bias, float *scales, float *mins, int n, int m, int k);
extern void LaunchFastllmGemmFp32Int4Group(float *input, uint8_t *weight, float *output, float *bias, float *scales, float *mins, int n, int m, int k, int group, int groupCnt);

extern __global__ void FastllmReduceKernel(float *output, float* input, int len, int threadNum);
extern __global__ void FastllmReduceKernel(half *output, half* input, int len, int threadNum);

std::vector <int> multiCudaCurrentDevices;

void FastllmMultiCudaSetDevice(std::vector <int> ids) {
    multiCudaCurrentDevices = ids;
}

namespace fastllm {
    struct MultiCudaPrepareInput : MultiThreadBaseOp {
        int deviceId;
        uint8_t *cudaInput, *cudaOutput;
        int inputLen, outputLen;
        uint8_t **curInput, **curOutput;
        bool forceMalloc;

        MultiCudaPrepareInput(int deviceId, uint8_t *cudaInput, uint8_t *cudaOutput, int inputLen, int outputLen, uint8_t **curInput, uint8_t **curOutput, bool forceMalloc)
            : deviceId(deviceId), cudaInput(cudaInput), cudaOutput(cudaOutput), inputLen(inputLen), outputLen(outputLen), curInput(curInput), curOutput(curOutput), forceMalloc(forceMalloc) {}

        void Run() {
            *curInput = cudaInput; 
            *curOutput = cudaOutput; 
            
            if (deviceId != 0 || forceMalloc) {
                cudaSetDevice(deviceId);
                *curInput = (uint8_t*)FastllmCudaMalloc(inputLen);                
                *curOutput = (uint8_t*)FastllmCudaMalloc(outputLen);
// cudaDeviceSynchronize();
// auto st = std::chrono::system_clock::now();
                cudaMemcpy(*curInput, cudaInput, inputLen, cudaMemcpyDeviceToDevice);
// cudaDeviceSynchronize();
// float tt = GetSpan(st, std::chrono::system_clock::now());
// printf("memcpy %d bytes spend %f s, (%f GB / s).\n", (int)inputLen, tt, inputLen / tt / 1e9);
            }
        }
    };

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
                LaunchFastllmGemmFp32Int4Group((float*)curInput, (uint8_t*)weight, (float*)curOutput, (float*)bias, scales, mins, n, m, k, group, groupCnt);
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
                    FastllmCudaInt4Group2HalfKernel <<< (k * m - 1) / threadPerBlock + 1, threadPerBlock>>>((uint8_t*)weight, scales, mins, fp16Weight, k * m, m, group, groupCnt);
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
                LaunchFastllmGemmFp16Int4Group((half*)curInput, (uint8_t*)weight, (half*)curOutput, (half*)bias, scales, mins, n, m, k, group, groupCnt);
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
                    FastllmCudaInt4Group2HalfKernel <<< (k * m - 1) / threadPerBlock + 1, threadPerBlock>>>((uint8_t*)weight, scales, mins, fp16Weight, k * m, m, group, groupCnt);
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
        T *cudaInput, *cudaOutput, *bias;
        int n, m, k, start, len;
        bool hasBias;
        float *scales, *mins;
        uint8_t *zeros;
        int group, groupCnt;

        MultiCudaMatMulSingleOp(int deviceId, void *weight, DataType weightDataType, T *cudaInput, T *cudaOutput, T *bias, 
                                    int n, int m, int k, int start, int len, bool hasBias, float *scales, float *mins, uint8_t *zeros, int group, int groupCnt)
            : deviceId(deviceId), weight(weight), weightDataType(weightDataType), cudaInput(cudaInput), cudaOutput(cudaOutput), bias(bias), 
            n(n), m(m), k(k), start(start), len(len), hasBias(hasBias), scales(scales), mins(mins), zeros(zeros), group(group), groupCnt(groupCnt) {}

        void Run() {
            cudaSetDevice(deviceId);
            T *curInput = cudaInput; 
            T *curOutput = cudaOutput; 

            if (deviceId != 0 || n > 1) {
                curInput = (T*)FastllmCudaMalloc(n * m * sizeof(T));                
                curOutput = (T*)FastllmCudaMalloc(n * len * sizeof(T));
                
                cudaMemcpy(curInput, cudaInput, n * m * sizeof(T), cudaMemcpyDeviceToDevice);
            }

            cudaDeviceSynchronize();
            RunMatmul(weight, weightDataType, bias, n, m, len, hasBias, scales, mins, zeros, group, groupCnt, curInput, curOutput);
            if (deviceId != 0 || n > 1) {
                cudaMemcpy2D(cudaOutput + start, k * sizeof(T), curOutput, len * sizeof(T), len * sizeof(T), n, cudaMemcpyDeviceToDevice);
                FastllmCudaFree(curInput);
                FastllmCudaFree(curOutput);
            }
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
            if (deviceId != 0 || n > 1) {
                *curInput = (T*)FastllmCudaMalloc(n * m * sizeof(T));                
                *curOutput = (T*)FastllmCudaMalloc(n * k2 * sizeof(T));
                // cudaMemcpy(*curInput, cudaInput, n * m * sizeof(T), cudaMemcpyDeviceToDevice);
                cudaMemcpy(*curInput, cpuInput, n * m * sizeof(T), cudaMemcpyHostToDevice);
            }
            T *mid0 = (T*)FastllmCudaMalloc(n * k1 * sizeof(T));
            T *mid1 = (T*)FastllmCudaMalloc(n * k1 / 2 * sizeof(T));
            bool isQuantWeight = weight0->mins.size() > 0;
            std::vector <void*> datas0 = weight0->extraCudaData, datas1 = weight1->extraCudaData;
            if (typeid(T) == typeid(half)) {
                datas0 = weight0->extraCudaHalfData;
                datas1 = weight1->extraCudaHalfData;
            }
            cudaDeviceSynchronize();
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
            cudaDeviceSynchronize();
            if (threadNum > 1) {
                cudaMemcpy(partOutput, *curOutput, n * k2 * sizeof(T), cudaMemcpyDeviceToDevice);
            }

            cudaDeviceSynchronize();
            FastllmCudaFree(mid0);
            FastllmCudaFree(mid1);
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
        int deviceId = multiCudaCurrentDevices[i];
        cudaSetDevice(deviceId);

        auto &div = divisionScheme[deviceId];
        int len = 0;
        for (auto &it : div) {
            len += it.second - it.first;
        }

        void *deviceWeightData;
        float *deviceBiasData;
        cudaError_t state = cudaSuccess;
        if (splitAxis == 0) {
            deviceWeightData = (void*)FastllmCudaMalloc(len * m * weight.unitSize / weight.unitSizeDiv);
            deviceBiasData = (float*)FastllmCudaMalloc(len * sizeof(float));
            int curLen = 0;
            for (auto &it : div) {
                state = cudaMemcpy((uint8_t*)deviceWeightData + curLen * m * weight.unitSize / weight.unitSizeDiv, 
                                    (uint8_t*)weight.cudaData + it.first * m * weight.unitSize / weight.unitSizeDiv, 
                                    (it.second - it.first) * m * weight.unitSize / weight.unitSizeDiv, cudaMemcpyDeviceToDevice);
                state = cudaMemcpy(deviceBiasData + curLen, cudaBiasData + it.first, (it.second - it.first) * sizeof(float), cudaMemcpyDeviceToDevice);
                curLen += (it.second - it.first);
            }
        } else {
            deviceWeightData = (void*)FastllmCudaMalloc(k * len * weight.unitSize / weight.unitSizeDiv);
            deviceBiasData = (float*)FastllmCudaMalloc(k * sizeof(float));
            int curLen = 0;
            for (auto &it : div) {
                state = cudaMemcpy2D((uint8_t*)deviceWeightData + curLen * weight.unitSize / weight.unitSizeDiv,
                                    (it.second - it.first) * weight.unitSize / weight.unitSizeDiv,
                                    (uint8_t*)weight.cudaData + it.first * weight.unitSize / weight.unitSizeDiv, 
                                    m * weight.unitSize / weight.unitSizeDiv, 
                                    (it.second - it.first) * weight.unitSize / weight.unitSizeDiv,
                                    k, cudaMemcpyDeviceToDevice);
                curLen += (it.second - it.first);
            }
            if (i == 0) {
                state = cudaMemcpy(deviceBiasData, cudaBiasData, k * sizeof(float), cudaMemcpyDeviceToDevice);
            } else {
                state = cudaMemset(deviceBiasData, 0, k * sizeof(float));
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
            int deviceId = multiCudaCurrentDevices[i];
            cudaSetDevice(deviceId);
            
            auto &div = divisionScheme[deviceId];
            int len = 0;
            for (auto &it : div) {
                len += it.second - it.first;
            }

            float *cudaScales;
            float *cudaMins;
            uint8_t *cudaZeropoints;

            if (splitAxis == 0) {
                cudaScales = (float*)FastllmCudaMalloc(len * weightGroup * sizeof(float));
                cudaMins = (float*)FastllmCudaMalloc(len * weightGroup * sizeof(float));
                cudaZeropoints = (uint8_t*)FastllmCudaMalloc(len * weightGroup);

                int curLen = 0;
                for (auto &it : div) {
                    state = cudaMemcpy(cudaScales + curLen * weightGroup, weight.scales.data() + it.first * weightGroup, (it.second - it.first) * weightGroup * sizeof(float), cudaMemcpyHostToDevice);
                    state = cudaMemcpy(cudaMins + curLen * weightGroup, weight.mins.data() + it.first * weightGroup, (it.second - it.first) * weightGroup * sizeof(float), cudaMemcpyHostToDevice);
                    state = cudaMemcpy(cudaZeropoints + curLen * weightGroup, zeropoints.data() + it.first * weight.group, (it.second - it.first) * weightGroup, cudaMemcpyHostToDevice);
                    curLen += (it.second - it.first);
                }
            } else {
                cudaScales = (float*)FastllmCudaMalloc(k * weightGroup * sizeof(float));
                cudaMins = (float*)FastllmCudaMalloc(k * weightGroup * sizeof(float));
                cudaZeropoints = (uint8_t*)FastllmCudaMalloc(k * weightGroup);
                state = cudaMemcpy(cudaScales, weight.scales.data(), k * weightGroup * sizeof(float), cudaMemcpyHostToDevice);
                state = cudaMemcpy(cudaMins, weight.mins.data(), k * weightGroup * sizeof(float), cudaMemcpyHostToDevice);
                state = cudaMemcpy(cudaZeropoints, zeropoints.data(), k * weightGroup, cudaMemcpyHostToDevice);
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
        int deviceId = multiCudaCurrentDevices[i];
        cudaSetDevice(deviceId);

        auto &div = divisionScheme[deviceId];
        int len = 0;
        for (auto &it : div) {
            len += it.second - it.first;
        }

        half *deviceBiasData;
        cudaError_t state = cudaSuccess;

        if (splitAxis == 0) {
            deviceBiasData = (half*)FastllmCudaMalloc(len * sizeof(half));
            int curLen = 0;
            for (auto &it : div) {
                state = cudaMemcpy(deviceBiasData + curLen, cudaBiasData + it.first, (it.second - it.first) * sizeof(half), cudaMemcpyDeviceToDevice);
                curLen += (it.second - it.first);
            }
        } else {
            deviceBiasData = (half*)FastllmCudaMalloc(k * sizeof(half));
            if (i == 0) {
                state = cudaMemcpy(deviceBiasData, cudaBiasData, k * sizeof(half), cudaMemcpyDeviceToDevice);
            } else {
                state = cudaMemset(deviceBiasData, 0, k * sizeof(half));
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
    std::vector <int> points = FastllmMultiCudaGetSplitPoints(k);
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
    std::vector<fastllm::MultiCudaMatMulSingleOp <T> *> ops;

    int isQuantWeight = (weight.mins.size() > 0);
    int threadNum = multiCudaCurrentDevices.size();
    for (int i = 0; i < threadNum; i++) {
        int deviceId = multiCudaCurrentDevices[i];
        int start = points[i], len = points[i + 1] - points[i];
        std::vector <void*> datas = weight.extraCudaData;
        if (typeid(T) == typeid(half)) {
            datas = weight.extraCudaHalfData;
        }
        ops.push_back(new fastllm::MultiCudaMatMulSingleOp <T> (
            deviceId, datas[i * 2], weight.dataType,
            cudaInput, cudaOutput, 
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

template <typename T>
bool FastllmMultiCudaMLPInner(const fastllm::Data &input, fastllm::Data &weight0, fastllm::Data &weight1, fastllm::Data &output) {
    std::vector <int> points = FastllmMultiCudaGetSplitPoints(weight0.dims[0] / 2);
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
auto st = std::chrono::system_clock::now();
        T *cudaInput = (T *) FastllmCudaPrepareInput(input);
        T *cudaOutput = (T *) FastllmCudaPrepareOutput(output);

        std::vector <uint8_t*> curInputs, curOutputs;
        auto *pool = fastllm::GetAlivePool();
        std::vector<fastllm::MultiCudaMLPSingleOp <T> *> ops;

        int threadNum = multiCudaCurrentDevices.size();
        curInputs.resize(threadNum);
        curOutputs.resize(threadNum);

        T *cpuInput = new T[input.Count(0)];
        cudaMemcpy(cpuInput, cudaInput, input.GetBytes(), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        T *partOutput = (T*)FastllmCudaMalloc(output.GetBytes() * threadNum);

        std::vector <int> points = FastllmMultiCudaGetSplitPoints(weight0.dims[0] / 2); // 因为要swiglu，所以先/2
        for (int i = 0; i < threadNum; i++) {
            int deviceId = multiCudaCurrentDevices[i];
            int start = points[i], len = points[i + 1] - points[i];
            ops.push_back(new fastllm::MultiCudaMLPSingleOp <T>  (
                deviceId, &weight0, &weight1,
                (int)input.Count(0) / input.dims.back(), input.dims.back(), len * 2, weight1.dims[0], 
                cudaInput, cudaOutput, (T**)&curInputs[i], (T**)&curOutputs[i], 
                cpuInput, partOutput + output.Count(0) * i, threadNum, i
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
// printf("step 0 spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));

        if (threadNum > 1) {
            int len = output.Count(0);
            for (int t = 0; t < threadNum; t++) {
                if ((long long)curOutputs[t] != (long long)cudaOutput) {
                    FastllmCudaFree(curOutputs[t]);
                }

                if ((long long)curInputs[t] != (long long)cudaInput) {
                    FastllmCudaFree(curInputs[t]);
                }
            }

            cudaSetDevice(0);
            len = output.Count(0);
            int threadPerBlock = std::min(256, len);
            FastllmReduceKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaOutput, partOutput, len, threadNum);
            FastllmCudaFree(partOutput);
        }

        /*
        if (threadNum > 0) {
            int len = output.Count(0);
            T *cpuResult = new T[len], *now = new T[len];
            for (int i = 0; i < threadNum; i++) {
                int deviceId = multiCudaCurrentDevices[i];    
                cudaSetDevice(deviceId);
                if (i == 0) {
                    cudaMemcpy(cpuResult, curOutputs[i], len * sizeof(T), cudaMemcpyDeviceToHost);
                } else {
                    cudaMemcpy(now, curOutputs[i], len * sizeof(T), cudaMemcpyDeviceToHost);
                    if (typeid(T) == typeid(half)) {
                        for (int i = 0; i < len; i++) {
                            cpuResult[i] = __hadd(((half*)cpuResult)[i], ((half*)now)[i]);
                        }
                    } else {
                        for (int i = 0; i < len; i++) {
                            cpuResult[i] = (T)((float)cpuResult[i] + (float)now[i]);
                        }
                    }
                }

                if ((long long)curOutputs[i] != (long long)cudaOutput) {
                    FastllmCudaFree(curOutputs[i]);
                }

                if ((long long)curInputs[i] != (long long)cudaInput) {
                    FastllmCudaFree(curInputs[i]);
                }
            }

            cudaSetDevice(0);
            cudaMemcpy(cudaOutput, cpuResult, len * sizeof(T), cudaMemcpyHostToDevice);

            delete[] cpuResult;
            delete[] now;
        }
        */
        
        cudaSetDevice(0);
        FastllmCudaFinishInput(input, cudaInput);
        FastllmCudaFinishOutput(output, cudaOutput);
// printf("step 1 spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
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