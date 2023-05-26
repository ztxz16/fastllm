#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>

#include "fastllm-cuda.h"
#include "fastllm.h"

static cublasHandle_t fastllmCublasHandle = nullptr;

#include <chrono>

double GetSpan(std::chrono::system_clock::time_point time1, std::chrono::system_clock::time_point time2) {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds> (time2 - time1);
    return double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
};

__global__ void FastllmGeluKernel(float* a, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float x = a[idx];
    if (idx < len) {
        a[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * x * (1.0f + 0.044715f * x * x)));
    }
}

__global__ void FastllmMulKernel(float* a, float v, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        a[idx] = a[idx] * v;
    }
}

template <int THREAD_PER_BLOCK>
__global__ void SoftmaxKernelInner1(float* input, float *output, int outer, int channels) {
    int o = blockIdx.x;
    input = input + o * channels;
    output = output + o * channels;

    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ float maxV;

    // 1. 每个线程计算一部分
    unsigned int tid = threadIdx.x;
    unsigned int per = (channels / THREAD_PER_BLOCK);
    unsigned int id = threadIdx.x * per;
    unsigned int len = per;
    if (tid == blockDim.x - 1) {
        len += (channels - per * THREAD_PER_BLOCK);
    }
    float maxValue = input[id];
    for (int i = 0; i < len; i++) {
        maxValue = max(maxValue, input[id + i]);
    }
    sdata[tid] = maxValue;
    __syncthreads();

    // 2. 求max
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // 3. 记录max
    if (tid == 0) {
        maxV = sdata[0];
    }
    __syncthreads();

    // 4. 求和
    float sum = 0;
    for (int i = 0; i < len; i++) {
        output[id + i] = exp(input[id + i] - maxV);
        sum += output[id + i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    for (int i = 0; i < len; i++) {
        output[id + i] /= sdata[0];
    }
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

void *FastllmCudaPrepareInput(const fastllm::Data &input) {
    void *ret;
    if (input.dataDevice == fastllm::DataDevice::CUDA) {
        ret = (void*)input.cudaData;
    } else {
        ret = (void*)FastllmCudaMalloc(input.expansionBytes);
        cudaMemcpy(ret, input.cpuData, input.expansionBytes, cudaMemcpyHostToDevice);
    }
    return ret;
}

void FastllmCudaFinishInput(const fastllm::Data &input, void *data) {
    if (input.dataDevice != fastllm::DataDevice::CUDA) {
        FastllmCudaFree(data);
    }
}

void *FastllmCudaPrepareOutput(fastllm::Data &output) {
    void *ret;
    if (output.dataDevice == fastllm::DataDevice::CUDA) {
        ret = (float*)output.cudaData;
    } else {
        ret = (float*)FastllmCudaMalloc(output.expansionBytes);
    }
    return ret;
}

void FastllmCudaFinishOutput(fastllm::Data &output, void *data) {
    if (output.dataDevice != fastllm::DataDevice::CUDA) {
        cudaMemcpy(output.cpuData, data, output.expansionBytes, cudaMemcpyDeviceToHost);
        FastllmCudaFree(data);
    }
}

bool FastllmCudaMatMulFloatInt8(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
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

    float *cudaScales = (float*)weight.extraCudaData[0];
    uint8_t *cudaZeropoints = (uint8_t*)weight.extraCudaData[1];
    float *cudaBiasData = (float*)weight.extraCudaData[2];

    float *cudaInput = (float*)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float*)FastllmCudaPrepareOutput(output);

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

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaMatMulFloatInt4(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
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

    float *cudaScales = (float*)weight.extraCudaData[0];
    uint8_t *cudaZeropoints = (uint8_t*)weight.extraCudaData[1];
    float *cudaBiasData = (float*)weight.extraCudaData[2];

    float *cudaInput = (float*)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float*)FastllmCudaPrepareOutput(output);

    for (int i = 0; i < n; i++) {
        FastllmGemvInt4Kernel0 <256> <<< k, 256 >>> (cudaInput + i * m, (uint8_t *) weight.cudaData,
            cudaOutput + i * k, cudaBiasData, cudaScales, cudaZeropoints, m, k);
    }
    cudaDeviceSynchronize();
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
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

void FastllmCudaCopyFromDeviceToDevice(void *dst, void *src, size_t size) {
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
}

void FastllmCudaMemcpy2DDeviceToDevice(void * 	dst, size_t 	dpitch, const void * 	src,
                                       size_t 	spitch, size_t 	width, size_t 	height) {
    cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice);
}

bool FastllmCudaGeluNew(const fastllm::Data &input, fastllm::Data &output) {
    int len = input.Count(0);
    float *cudaData = (float *) FastllmCudaPrepareInput(input);
    int threadPerBlock = min(256, len);
    FastllmGeluKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaData, len);
    FastllmCudaFinishOutput(output, cudaData);
    return true;
}

bool FastllmCudaMul(const fastllm::Data &input, float v, fastllm::Data &output) {
    int len = input.Count(0);
    float *cudaData = (float *) FastllmCudaPrepareInput(input);
    int threadPerBlock = min(256, len);
    FastllmMulKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaData, v, len);
    FastllmCudaFinishOutput(output, cudaData);
    return true;
}

bool FastllmCudaSoftmax(const fastllm::Data &input, fastllm::Data &output, int axis) {
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareInput(output);

    int dimsLen = input.dims.size();
    axis = (axis % dimsLen + dimsLen) % dimsLen;
    int outer = input.Count(0) / input.Count(axis);
    int channels = input.dims[axis];
    int inner = input.Count(axis + 1);

    if (inner == 1) {
        if (channels < 8) {
            SoftmaxKernelInner1 <1> <<< outer, 1 >>> (cudaInput, cudaOutput, outer, channels);
        } else if (channels < 64) {
            SoftmaxKernelInner1 <8> <<< outer, 8 >>> (cudaInput, cudaOutput, outer, channels);
        } else if (channels < 512) {
            SoftmaxKernelInner1 <64> <<< outer, 64 >>> (cudaInput, cudaOutput, outer, channels);
        } else {
            SoftmaxKernelInner1 <256> <<< outer, 256 >>> (cudaInput, cudaOutput, outer, channels);
        }

    } else {
        printf("softmax error.\n");
        exit(0);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaBatchMatMulTransB(const fastllm::Data &input0, const fastllm::Data &input1, fastllm::Data &output,
                              int input0Spatial, int input1Spatial, int outputSpatial,
                              int input0Stride, int input1Stride,
                              int batch, int n, int m, int k, float alpha) {
    float *cudaInput0 = (float *) FastllmCudaPrepareInput(input0);
    float *cudaInput1 = (float *) FastllmCudaPrepareInput(input1);
    float *cudaOutput = (float *) FastllmCudaPrepareOutput(output);

    if (fastllmCublasHandle == nullptr) {
        cublasCreate(&fastllmCublasHandle);
    }

    float beta = 0;
    cublasStatus_t status;
    status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                       CUBLAS_OP_T, CUBLAS_OP_N,
                                       k, n, m, &alpha,
                                       cudaInput1, input1Stride, input1Spatial,
                                       cudaInput0, input0Stride, input0Spatial,
                                       &beta,
                                       cudaOutput, k, k * n, batch);

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("status = %d", status);
        printf("%d %d %d\n", k, n, m);
        printf("Error: cublas error.\n");
        exit(0);
    }
    cudaDeviceSynchronize();

    FastllmCudaFinishInput(input0, cudaInput0);
    FastllmCudaFinishInput(input1, cudaInput1);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}