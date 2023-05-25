#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>

#include "fastllm-cuda.h"
#include "fastllm.h"

//static cublasHandle_t fastllmCublasHandle = nullptr;

#include <chrono>

double GetSpan(std::chrono::system_clock::time_point time1, std::chrono::system_clock::time_point time2) {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds> (time2 - time1);
    return double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
};

__global__ void FastllmGeluKernel(float* a) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float x = a[idx];
    a[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * x * (1.0f + 0.044715f * x * x)));
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmGemvInt8Kernel0(float *A, uint8_t *B, float *C,
                      float *bias, float *scales, uint8_t *zeros,
                      int m, int k) {
    __shared__ float sdata[THREAD_PER_BLOCK];

    // 1. 每个线程计算一部分
    unsigned int tid = threadIdx.x;
    unsigned int per = (m / THREAD_PER_BLOCK);
    unsigned int id = blockIdx.x * m + threadIdx.x * per;
    unsigned int len = per;
    if (tid == blockDim.x - 1) {
        len += (m - per * THREAD_PER_BLOCK);
    }
    float sum = 0.0;
    for (int i = 0; i < len; i++) {
        sum += A[threadIdx.x * per + i] * (B[id + i] - zeros[blockIdx.x]);
    }
    sdata[tid] = sum;
    __syncthreads();

    // 2. 求和
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 3. 写回结果
    if (tid == 0) {
        C[blockIdx.x] = sdata[0] * scales[blockIdx.x] + bias[blockIdx.x];
    }
}

template <int THREAD_PER_BLOCK, int SINGLE_COMPUTE, int REDUCE_NUMBER>
__global__ void FastllmGemvInt8Kernel1(float *A, uint8_t *B, float *C,
                      float *bias, float *scales, uint8_t *zeros,
                      int m, int k) {
    __shared__ float sdata[REDUCE_NUMBER];
    unsigned int tid = threadIdx.x;

    int part = m / REDUCE_NUMBER;
    // 1. 每个线程计算一部分
    for (int p = 0; p < part; p++) {
        float v[SINGLE_COMPUTE];
        for (int i = 0; i < SINGLE_COMPUTE; i++) {
            v[i] = A[p * REDUCE_NUMBER + tid * SINGLE_COMPUTE + i];
        }
        for (int i = 0; i < SINGLE_COMPUTE / part; i++) {
            float sum = 0;
            int colId = (blockIdx.x * SINGLE_COMPUTE / part + i);
            if (colId >= k) {
                sdata[i * (m / SINGLE_COMPUTE) + p * (REDUCE_NUMBER / SINGLE_COMPUTE) + tid] = 0;
                continue;
            }
            int id = colId * m + p * REDUCE_NUMBER + tid * SINGLE_COMPUTE;
            uint8_t zero = zeros[colId];
            for (int j = 0; j < SINGLE_COMPUTE; j++) {
                sum += v[j] * (B[id + j] - zero);
            }
            sdata[i * (m / SINGLE_COMPUTE) + p * (REDUCE_NUMBER / SINGLE_COMPUTE) + tid] = sum;
            __syncthreads();
        }
    }

    // 2. 求和
    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (int i = 0; i < SINGLE_COMPUTE; i++) {
                sdata[i * THREAD_PER_BLOCK + tid] += sdata[i * THREAD_PER_BLOCK + tid + s];
            }
        }
        __syncthreads();
    }

    // 3. 写回结果
    if (tid == 0) {
        for (int i = 0; i < SINGLE_COMPUTE / part; i++) {
            int id = blockIdx.x * SINGLE_COMPUTE / part  + i;
            if (id >= k) {
                continue;
            }
            float sum = 0;
            for (int p = 0; p < part; p++) {
                sum += sdata[(i * part + p) * THREAD_PER_BLOCK];
            }
            C[id] = sum * scales[id] + bias[id];
        }
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmGemvInt4Kernel0(float *A, uint8_t *B, float *C,
                      float *bias, float *scales, uint8_t *zeros,
                      int m, int k) {
    __shared__ float sdata[THREAD_PER_BLOCK];

    // 1. 每个线程计算一部分
    unsigned int tid = threadIdx.x;
    unsigned int per = (m / THREAD_PER_BLOCK);
    unsigned int id = blockIdx.x * m + threadIdx.x * per;
    unsigned int len = per;
    if (tid == blockDim.x - 1) {
        len += (m - per * THREAD_PER_BLOCK);
    }
    float sum = 0.0;
    for (int i = 0; i + 1 < len; i += 2) {
        uint8_t now = B[(id + i) / 2];
        sum += A[threadIdx.x * per + i] * ((now >> 4) - zeros[blockIdx.x]);
        sum += A[threadIdx.x * per + i + 1] * ((now & 15) - zeros[blockIdx.x]);
    }
    sdata[tid] = sum;
    __syncthreads();

    // 2. 求和
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 3. 写回结果
    if (tid == 0) {
        C[blockIdx.x] = sdata[0] * scales[blockIdx.x] + bias[blockIdx.x];
    }
}

bool FastllmMatMulFloatInt8(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    if (weight.cudaData == nullptr) {
        cudaMalloc(&weight.cudaData, weight.Count(0));
        FastllmCudaCopyFromHostToDevice(weight.cudaData, weight.cpuData, k * m);

        float *cudaScales;
        cudaMalloc(&cudaScales, k * sizeof(float));
        cudaMemcpy(cudaScales, weight.scales.data(), k * sizeof(float), cudaMemcpyHostToDevice);
        weight.extraCudaData.push_back((void*)cudaScales);

        uint8_t *cudaZeropoints;
        cudaMalloc(&cudaZeropoints, k);
        uint8_t *zeropoints = new uint8_t[k];
        for (int i = 0; i < k; i++) {
            zeropoints[i] = weight.perChannelsConfigs[i].zeroPoint;
        }
        cudaMemcpy(cudaZeropoints, zeropoints, k, cudaMemcpyHostToDevice);
        delete[] zeropoints;
        weight.extraCudaData.push_back((void*)cudaZeropoints);

        float *cudaBiasData;
        cudaMalloc(&cudaBiasData, k * sizeof(float));
        if (bias.dims.size() > 0) {
            cudaMemcpy(cudaBiasData, (uint8_t*)bias.cpuData, k * sizeof(float), cudaMemcpyHostToDevice);
        } else {
            cudaMemset(cudaBiasData, 0, k * sizeof(float));
        }
        weight.extraCudaData.push_back((void*)cudaBiasData);
    }

    float *inputData = (float *) input.cpuData;
    float *outputData = (float *) output.cpuData;

    float *cudaScales = (float*)weight.extraCudaData[0];
    uint8_t *cudaZeropoints = (uint8_t*)weight.extraCudaData[1];
    float *cudaBiasData = (float*)weight.extraCudaData[2];

    float *cudaInput = (float*)FastllmCudaMalloc(n * m * sizeof(float));
    float *cudaOutput = (float*)FastllmCudaMalloc(n * k * sizeof(float));

    cudaMemcpy(cudaInput, inputData, n * m * sizeof(float), cudaMemcpyHostToDevice);
    if (m == 4096 || m == 16384) {
        for (int i = 0; i < n; i++) {
            FastllmGemvInt8Kernel1<256, 16, 4096> <<< (k - 1) / (16 / (m / 4096)) + 1, 256 >>>(cudaInput + i * m, (uint8_t *) weight.cudaData,
                                                                              cudaOutput + i * k,
                                                                              cudaBiasData, cudaScales, cudaZeropoints,
                                                                              m, k);
        }
    } else {
        for (int i = 0; i < n; i++) {
            FastllmGemvInt8Kernel0 <256> <<< k, 256 >>> (cudaInput + i * m, (uint8_t *) weight.cudaData,
                    cudaOutput + i * k, cudaBiasData, cudaScales, cudaZeropoints, m, k);
        }
    }
    cudaDeviceSynchronize();
    cudaMemcpy(outputData, cudaOutput, n * k * sizeof(float), cudaMemcpyDeviceToHost);
    FastllmCudaFree(cudaInput);
    FastllmCudaFree(cudaOutput);
    return true;
}

bool FastllmMatMulFloatInt4(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    if (weight.cudaData == nullptr) {
        cudaMalloc(&weight.cudaData, k * m / 2);
        FastllmCudaCopyFromHostToDevice(weight.cudaData, weight.cpuData, k * m / 2);

        float *cudaScales;
        cudaMalloc(&cudaScales, k * sizeof(float));
        cudaMemcpy(cudaScales, weight.scales.data(), k * sizeof(float), cudaMemcpyHostToDevice);
        weight.extraCudaData.push_back((void*)cudaScales);

        uint8_t *cudaZeropoints;
        cudaMalloc(&cudaZeropoints, k);
        uint8_t *zeropoints = new uint8_t[k];
        for (int i = 0; i < k; i++) {
            zeropoints[i] = weight.perChannelsConfigs[i].zeroPoint;
        }
        cudaMemcpy(cudaZeropoints, zeropoints, k, cudaMemcpyHostToDevice);
        delete[] zeropoints;
        weight.extraCudaData.push_back((void*)cudaZeropoints);

        float *cudaBiasData;
        cudaMalloc(&cudaBiasData, k * sizeof(float));
        if (bias.dims.size() > 0) {
            cudaMemcpy(cudaBiasData, (uint8_t*)bias.cpuData, k * sizeof(float), cudaMemcpyHostToDevice);
        } else {
            cudaMemset(cudaBiasData, 0, k * sizeof(float));
        }
        weight.extraCudaData.push_back((void*)cudaBiasData);
    }

    float *inputData = (float *) input.cpuData;
    float *outputData = (float *) output.cpuData;

    float *cudaScales = (float*)weight.extraCudaData[0];
    uint8_t *cudaZeropoints = (uint8_t*)weight.extraCudaData[1];
    float *cudaBiasData = (float*)weight.extraCudaData[2];

    float *cudaInput = (float*)FastllmCudaMalloc(n * m * sizeof(float));
    float *cudaOutput = (float*)FastllmCudaMalloc(n * k * sizeof(float));

    cudaMemcpy(cudaInput, inputData, n * m * sizeof(float), cudaMemcpyHostToDevice);
    for (int i = 0; i < n; i++) {
        FastllmGemvInt4Kernel0 <256> <<< k, 256 >>> (cudaInput + i * m, (uint8_t *) weight.cudaData,
            cudaOutput + i * k, cudaBiasData, cudaScales, cudaZeropoints, m, k);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(outputData, cudaOutput, n * k * sizeof(float), cudaMemcpyDeviceToHost);
    FastllmCudaFree(cudaInput);
    FastllmCudaFree(cudaOutput);
    return true;
}

struct CudaMemoryBuffer {
    void *data;
    size_t size;
    bool busy;

    CudaMemoryBuffer () {}

    CudaMemoryBuffer (void *data, size_t size, bool busy) :
        data(data), size(size), busy(busy) {}
};
std::vector <CudaMemoryBuffer> cudaBuffers;

void * FastllmCudaMalloc(size_t size) {
    for (int i = 0; i < cudaBuffers.size(); i++) {
        if (cudaBuffers[i].size >= size && !cudaBuffers[i].busy) {
            cudaBuffers[i].busy = true;
            return cudaBuffers[i].data;
        }
    }
    void * ret;
    cudaMalloc(&ret, size);
    cudaBuffers.push_back(CudaMemoryBuffer(ret, size, true));
    return ret;
}

void FastllmCudaFree(void *ret) {
    for (int i = 0; i < cudaBuffers.size(); i++) {
        if (cudaBuffers[i].data == ret) {
            cudaBuffers[i].busy = false;
            break;
        }
    }
}

void FastllmCudaCopyFromHostToDevice(void *dst, void *src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

void FastllmCudaCopyFromDeviceToHost(void *dst, void *src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

bool FastllmGelu(const fastllm::Data &input, fastllm::Data &output) {
    int len = input.Count(0);
    float *temp = (float*)FastllmCudaMalloc(len * sizeof(float));
    cudaMemcpy(temp, input.cpuData, len * sizeof(float), cudaMemcpyHostToDevice);
    int threadPerBlock = min(256, len);
    FastllmGeluKernel <<<len / threadPerBlock, threadPerBlock>>> (temp);
    cudaMemcpy(output.cpuData, temp, len * sizeof(float), cudaMemcpyDeviceToHost);
    FastllmCudaFree(temp);
    return true;
}