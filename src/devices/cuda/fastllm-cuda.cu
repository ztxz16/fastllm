#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <chrono>

#include "fastllm-cuda.cuh"
#include "fastllm.h"

static std::map<int, cublasHandle_t> s_fastllmCublasHandleMap;
cublasHandle_t getFastllmCublasHandle() {
    int id = -1;
    cudaGetDevice(&id);
    auto it = s_fastllmCublasHandleMap.find(id);
    if (it != s_fastllmCublasHandleMap.end()) {
        return it->second;
    }
    cublasHandle_t handler = nullptr;
    auto stat = cublasCreate(&handler);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("Error: CUBLAS initialization failed. state %d.\n", stat);
        exit(0);
    } else {
        s_fastllmCublasHandleMap[id] = handler;
    }

    return handler;
}

void DeviceSync() {
    //cudaDeviceSynchronize();
}

double GetSpan(std::chrono::system_clock::time_point time1, std::chrono::system_clock::time_point time2) {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds> (time2 - time1);
    return double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
};

__global__ void FastllmCudaFloat2HalfKernel(float* a, half *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        b[idx] = __float2half(a[idx]);
    }
}

__global__ void FastllmCudaInt82HalfKernel(uint8_t* a, float *scales, uint8_t *zeros, half *b, int len, int per) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        b[idx] = __float2half(scales[idx / per] * ((float)a[idx] - zeros[idx / per]));
    }
}

__global__ void FastllmCudaInt42HalfKernel(uint8_t* a, float *scales, float *mins, half *b, int len, int per) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        if (idx % 2 == 1) {
            b[idx] = __float2half(scales[idx / per] * (a[idx / 2] & 0xF) + mins[idx / per]);
        } else {
            b[idx] = __float2half(scales[idx / per] * (a[idx / 2] >> 4) + mins[idx / per]);
        }
    }
}

__global__ void FastllmCudaHalf2FlotaKernel(half* a, float *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        b[idx] = __half2float(a[idx]);
    }
}

__global__ void FastllmCudaBiasKernel(float *a, float *bias, int k) {
    float *now = a + blockIdx.x * k;
    int stride = blockDim.x;
    for (int i = threadIdx.x; i < k; i += stride) {
        now[i] += bias[i];
    }
}

__global__ void FastllmGeluKernel(float* a, float *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        float x = a[idx];
        b[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * x * (1.0f + 0.044715f * x * x)));
    }
}

__global__ void FastllmSiluKernel(float* a, float *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        float x = a[idx];
        b[idx] = x / (1.0 + expf(-x));
    }
}

__global__ void FastllmSwigluKernel(float* a, float *b, int len, int spatial, int mid) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        int id = idx / mid * spatial + idx % mid;
        float x = a[id], y = a[id + mid];
        b[idx] = (x / (1.0 + expf(-x))) * y;
    }
}

__global__ void FastllmMulKernel(float* a, float *b, float v, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        b[idx] = a[idx] * v;
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmMulBatchKernel(float** pointer, int batch, float v) {
    float *input = pointer[blockIdx.x];
    float *output = pointer[blockIdx.x + batch];
    int len = (int)((unsigned long long)pointer[blockIdx.x + batch * 2]);
    for (int i = threadIdx.x; i < len; i += THREAD_PER_BLOCK) {
        output[i] = input[i] * v;
    }
}

__global__ void FastllmAddToKernel(float* a, float *b, float alpha, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        a[idx] += b[idx] * alpha;
    }
}

__global__ void FastllmMulToKernel(float* a, float *b, float alpha, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        a[idx] *= b[idx] * alpha;
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmAttentionMaskKernel(float* a, float *b, float maskValue, int n, int m, int spatial) {
    int on = blockIdx.x / m;
    int om = blockIdx.x % m;
    int o = on * m + om;
    int idx = threadIdx.x;
    for (int i = idx; i < spatial; i += THREAD_PER_BLOCK) {
        if (b[on * spatial + i] > 0.99) {
            a[o * spatial + i] = maskValue;
        }
    }
}

template <int THREAD_PER_BLOCK>
__global__ void SimpleMask(float* a, float *b, float maskValue, int spatial) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < spatial) {
        if (b[i] > 0.99) {
            a[i] = maskValue;
        }
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmAlibiMaskKernel(float* a, float *b, float maskValue, int n, int m, int spn, int spm, int spatial) {
    int on = blockIdx.x / m;
    int om = blockIdx.x % m;
    int o = on * m + om;
    int idx = threadIdx.x;
    float now = b[om];
    for (int i = idx; i < spatial; i += THREAD_PER_BLOCK) {
        int idi = i / spm, idj = i % spm;
        if (idj <= spm - spn + idi) {
            a[o * spatial + i] += now * idj;
        } else {
            a[o * spatial + i] = maskValue;
        }
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmApplyLognAttnKernel(float* input, float *logn, float *pos, int b, int s, int spatial) {
    int ob = blockIdx.x / s;
    int os = blockIdx.x % s;
    int o = ob * s + os;
    int idx = threadIdx.x;
    int curPos = (int)(pos[0]);

    float v = logn[os + curPos];
    float *curInput = input + o * spatial;
    for (int i = idx; i < spatial; i += THREAD_PER_BLOCK) {
        curInput[i] = curInput[i] * v;
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmTransposeByRowKernel(uint8_t *dst, uint8_t *ori, int n, int m, int k) {
    int row = blockIdx.x / m, col = blockIdx.x % m;
    uint8_t *curInput = ori + (row * m + col) * k;
    uint8_t *curOutput = dst + (col * n + row) * k;
    for (int i = threadIdx.x; i < k; i += THREAD_PER_BLOCK) {
        curOutput[i] = curInput[i];
    }
}

__global__ void FastllmPermuteKernel(float *dst, float *ori, int *temp, int axisLen, int len) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len) {
        int old = 0;
        int idx = i;
        for (int j = 0; j < axisLen; ++j) {
            int order = temp[j];
            old += (idx / temp[j + 2 * axisLen]) * temp[order + 1 * axisLen];
            idx %= temp[j + 2 * axisLen];
        }
        dst[i] = ori[old];
    }
}

__global__ void FastllmLlamaRotatePosition2DKernel(float *data, float *positionIds, float *sin, float *cos,
                                                   int len, int bs, int spatial, int n, int m, int partStride, int sinCosStride, int rotateDim) {
    int o = (blockIdx.x / n);
    int l = o % len;
    int b = o / len;
    int j = threadIdx.x;
    int index = (int) (positionIds[b * partStride + l]);

    float curSin = sin[index * sinCosStride + j];
    float curCos = cos[index * sinCosStride + j];
    float *d = (float *) data + o * spatial + j;
    int i = blockIdx.x % n;
    float va = d[i * m], vb = d[i * m + m / 2];
    d[i * m] = va * curCos - vb * curSin;
    d[i * m + m / 2] = va * curSin + vb * curCos;
}

__global__ void FastllmNearlyRotatePosition2DKernel(float *data, float *positionIds, float *sin, float *cos,
                                                    int len, int bs, int spatial, int n, int m, int partStride, int sinCosStride, int rotateDim) {
/*
    int len = data.dims[0], bs = data.dims[1];
    int spatial = data.Count(2);
    int n = data.dims[2], m = data.dims[3];
    int stride = (int)sinData.dims[1];
    for (int l = 0; l < len; l++) {
        for (int b = 0; b < bs; b++) {
            int index = (int) ((float *) positionIds.cpuData)[(b * 2) * positionIds.dims.back() + l];
            float *sin = ((float*)sinData.cpuData) + stride * index;
            float *cos = ((float*)cosData.cpuData) + stride * index;
            float *d = (float *) data.cpuData + (l * bs + b) * spatial;
            for (int i = 0; i < n; i++) {
                int j = 0;
                for (; j < rotaryDim; j += 2) {
                    float a = d[j], b = d[j + 1];
                    d[j] = a * cos[j / 2] - b * sin[j / 2];
                    d[j + 1] = a * sin[j / 2] + b * cos[j / 2];
                }
                d += m;
            }
        }
    }
*/
    int o = (blockIdx.x / n);
    int l = o / bs;
    int b = o % bs;
    int j = threadIdx.x;
    int index = (int) (positionIds[b * 2 * partStride + l]);

    float curSin = sin[index * sinCosStride + j];
    float curCos = cos[index * sinCosStride + j];
    float *d = (float *) data + o * spatial + j * 2;
    int i = blockIdx.x % n;
    float va = d[i * m], vb = d[i * m + 1];
    d[i * m] = va * curCos - vb * curSin;
    d[i * m + 1] = va * curSin + vb * curCos;
}

__global__ void FastllmRotatePosition2DKernel(float *data, float *positionIds, float *sin, float *cos,
                                              int len, int bs, int spatial, int n, int m, int partStride, int sinCosStride, int rotateDim) {
    int o = (blockIdx.x / n) / 2;
    int l = o / bs;
    int b = o % bs;
    int part = (blockIdx.x / n) % 2;
    int j = threadIdx.x;
    int index = (int) (positionIds[(b * 2 + part) * partStride + l]);

    float curSin = sin[index * sinCosStride + j];
    float curCos = cos[index * sinCosStride + j];
    float *d = (float *) data + o * spatial + part * m / 2 + j;
    int i = blockIdx.x % n;
    float va = d[i * m], vb = d[i * m + m / 4];
    d[i * m] = va * curCos - vb * curSin;
    d[i * m + m / 4] = va * curSin + vb * curCos;
}

template <int THREAD_PER_BLOCK>
__device__ void FastllmSoftmaxKernelInner1Func(float *input, float *output, int channels) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ float maxV;

    // 1. 每个线程计算一部分
    unsigned int tid = threadIdx.x;
    float maxValue = input[tid];
    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        maxValue = max(maxValue, input[i]);
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
    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        output[i] = exp(input[i] - maxV);
        sum += output[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        if (fabs(sdata[0]) < 1e-6) {
            sdata[0] = 0.1;
        }
    }
    __syncthreads();

    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        output[i] /= sdata[0];
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmSoftmaxKernelInner1(float* input, float *output, int outer, int channels) {
    int o = blockIdx.x;
    FastllmSoftmaxKernelInner1Func <THREAD_PER_BLOCK> (input + o * channels, output + o * channels, channels);
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmSoftmaxKernelBatchInner1(uint8_t** pointer) {
    int o = blockIdx.x;
    FastllmSoftmaxKernelInner1Func <THREAD_PER_BLOCK> ((float*)pointer[o * 3], (float*)pointer[o * 3 + 1],
                                                       (int)((size_t)pointer[o * 3 + 2]));
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmRMSNormKernelInner1(float *input, float *weight, float *output, int outer, int channels, float eps) {
    int o = blockIdx.x;
    input = input + o * channels;
    output = output + o * channels;

    __shared__ float sdata2[THREAD_PER_BLOCK];
    __shared__ float scale;

    // 1. 每个线程计算一部分
    unsigned int tid = threadIdx.x;
    float sum2 = 0.0;
    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        float x = input[i];
        sum2 += x * x;
    }
    sdata2[tid] = sum2;
    __syncthreads();

    // 2. 求和
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata2[tid] += sdata2[tid + s];
        }
        __syncthreads();
    }

    // 3. 计算参数
    if (tid == 0) {
        scale = 1.0 / sqrt(sdata2[0] / channels + eps);
    }
    __syncthreads();

    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        output[i] = (input[i] * scale * weight[i]);
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmLayerNormKernelInner1(float *input, float *gamma, float *beta, float *output, int outer, int channels) {
    int o = blockIdx.x;
    input = input + o * channels;
    output = output + o * channels;

    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ float sdata2[THREAD_PER_BLOCK];
    __shared__ float mean;
    __shared__ float var;

    // 1. 每个线程计算一部分
    unsigned int tid = threadIdx.x;
    float sum = 0.0, sum2 = 0.0;
    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        float x = input[i];
        sum += x;
        sum2 += x * x;
    }
    sdata[tid] = sum;
    sdata2[tid] = sum2;
    __syncthreads();

    // 2. 求和
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata2[tid] += sdata2[tid + s];
        }
        __syncthreads();
    }

    // 3. 计算参数
    if (tid == 0) {
        mean = sdata[0] / channels;
        var = sdata2[0] + mean * mean * channels - 2 * mean * channels * mean;
        var = sqrt(var / channels + 1e-10);
    }
    __syncthreads();

    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        output[i] = (input[i] - mean) / var * gamma[i] + beta[i];
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmLayerNormKernelTop1(float *input, float *output, int channels) {
    __shared__ float idData[THREAD_PER_BLOCK];
    __shared__ float maxData[THREAD_PER_BLOCK];
    float *inputData = input + blockIdx.x * channels;
    float *outputData = output + blockIdx.x * 2;
    int tid = threadIdx.x;
    maxData[tid] = -1e100;
    for (int j = tid; j < channels; j += THREAD_PER_BLOCK) {
        if (inputData[j] > maxData[tid]) {
            maxData[tid] = inputData[j];
            idData[tid] = j;
        }
    }
    __syncthreads();

    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (maxData[tid] < maxData[tid + s]) {
                maxData[tid] = maxData[tid + s];
                idData[tid] = idData[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        outputData[0] = idData[0];
        outputData[1] = maxData[0];
    }
}

template <int NBlock, int MBlock, int KBlock>
__global__ void FastllmCudaBaseGemmKernelInt8(float *A, uint8_t *B, float *C,
                                              float *bias, float *scales, uint8_t *zeros,
                                              int n, int m, int k) {
    int nStart = blockIdx.x * NBlock, nEnd = nStart + NBlock;
    int kStart = blockIdx.y * KBlock, kEnd = kStart + KBlock;

    int id = kStart + threadIdx.x;
    __shared__ float shareA[NBlock * MBlock];
    __shared__ float shareB[KBlock * MBlock];
    float localSum[NBlock] = {0.0f};
    uint8_t zero = zeros[id];
    int idx = threadIdx.x >> 3;
    int idy = threadIdx.x & 7;
    for (int l = 0; l < m; l += MBlock) {
        if (threadIdx.x < MBlock) {
            for (int i = nStart; i < nEnd; i++) {
                if (i < n && l + threadIdx.x < m) {
                    shareA[(i - nStart) * MBlock + threadIdx.x] = A[i * m + l + threadIdx.x];
                } else {
                    shareA[(i - nStart) * MBlock + threadIdx.x] = 0.0f;
                }
            }
        }
        __syncthreads();
        if (threadIdx.x < MBlock) {
            for (int i = kStart; i < kEnd; i++) {
                if (i < k && l + threadIdx.x < m) {
                    shareB[(i - kStart) * MBlock + threadIdx.x] = B[i * m + l + threadIdx.x];
                } else {
                    shareB[(i - kStart) * MBlock + threadIdx.x] = 0.0f;
                }
            }
        }
        __syncthreads();

        for (int mStart = 0; mStart < MBlock; mStart += 4) {
            float curA[32] = {0.0f}, curB[32] = {0.0f};
            for (int i = 0; i < 8; i++) {
                for (int x = l + mStart; x < l + mStart + 4 && x < m; x++) {
                    curA[i * 4 + (x - l - mStart)] = shareA[(idx * 8 + i) * MBlock + (x - l)];
                }
            }
            for (int j = 0; j < 4; j++) {
                zero = zeros[kStart + (idy * 4 + j)];
                for (int x = l + mStart; x < l + mStart + 4 && x < m; x++) {
                    curB[j * 4 + (x - l - mStart)] = shareB[(idy * 4 + j) * MBlock + (x - l)] - zero;
                }
            }
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 4; j++) {
                    int cur = i * 4 + j;
                    localSum[cur] += curA[i * 4 + 0] * curB[j * 4 + 0];
                    localSum[cur] += curA[i * 4 + 1] * curB[j * 4 + 1];
                    localSum[cur] += curA[i * 4 + 2] * curB[j * 4 + 2];
                    localSum[cur] += curA[i * 4 + 3] * curB[j * 4 + 3];
                }
            }
            __syncthreads();
        }
        __syncthreads();
    }

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 4; j++) {
            if ((nStart + idx * 8 + i) < n && (kStart + idy * 4 + j) < k) {
                C[(nStart + idx * 8 + i) * k + (kStart + idy * 4 + j)] =
                        localSum[i * 4 + j] * scales[(kStart + idy * 4 + j)] + bias[(kStart + idy * 4 + j)];
            }
        }
    }
}

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

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvFp32Fp16Kernel2(float *A, half *B, float *C, float *bias, int m, int k) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    // 1. 计算
    int st = blockIdx.x * PART;
    int end = st + PART;
    for (int p = st; p < end; p++) {
        sdata[tid] = 0;
        for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
            sdata[tid] += A[i] * (float)B[p * m + i];
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

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvInt8Kernel2(float *A, uint8_t *B, float *C,
                                       float *bias, float *scales, uint8_t *zeros,
                                       int m, int k) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    // 1. 读入fdata
    /*for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
        fdata[i] = A[i];
    }
    __syncthreads();*/

    // 2. 计算
    int st = blockIdx.x * PART;
    int end = st + PART;
    for (int p = st; p < end; p++) {
        sdata[tid] = 0;
        uint8_t zero = zeros[p];
        for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
            sdata[tid] += A[i] * (B[p * m + i] - zero);
        }
        __syncthreads();
        for (unsigned int s = 1; s < THREAD_PER_BLOCK; s *= 2) {
            if ((tid & (2 * s - 1)) == 0) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            C[p] = sdata[0] * scales[p] + bias[p];
        }
        __syncthreads();
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

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvInt4Kernel2(float *A, uint8_t *B, float *C,
                                       float *bias, float *scales, uint8_t *zeros,
                                       int m, int k) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    // 1. 计算
    int st = blockIdx.x * PART;
    int end = st + PART;
    for (int p = st; p < end; p++) {
        sdata[tid] = 0;
        uint8_t zero = zeros[p];
        for (int i = tid; i < m / 2; i += THREAD_PER_BLOCK) {
            uint8_t now = B[p * m / 2 + i];
            sdata[tid] += (A[i * 2] * ((now >> 4) - zero) + A[i * 2 + 1] * ((now & 15) - zero));
        }
        __syncthreads();
        for (unsigned int s = 1; s < THREAD_PER_BLOCK; s *= 2) {
            if ((tid & (2 * s - 1)) == 0) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            C[p] = sdata[0] * scales[p] + bias[p];
        }
        __syncthreads();
    }
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvInt4NoZeroKernel2(float *A, uint8_t *B, float *C,
                                             float *bias, float *scales, float *mins,
                                             int m, int k) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    // 1. 计算
    int st = blockIdx.x * PART;
    int end = st + PART;
    for (int p = st; p < end; p++) {
        sdata[tid] = 0;
        float minv = mins[p] / scales[p];
        for (int i = tid; i < m / 2; i += THREAD_PER_BLOCK) {
            uint8_t now = B[p * m / 2 + i];
            sdata[tid] += (A[i * 2] * (minv + (now >> 4)) + A[i * 2 + 1] * (minv + (now & 15)));
        }
        __syncthreads();
        for (unsigned int s = 1; s < THREAD_PER_BLOCK; s *= 2) {
            if ((tid & (2 * s - 1)) == 0) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            C[p] = sdata[0] * scales[p] + bias[p];
        }
        __syncthreads();
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmSplitBatchKernel(uint8_t *input, uint8_t **outputs, int outer, int channels, int inner) {
    int bid = blockIdx.x / outer, oid = blockIdx.x % outer;
    uint8_t *curInput = input + oid * channels * inner + bid * inner;
    uint8_t *curOutput = outputs[bid] + oid * inner;

    for (int i = threadIdx.x; i < inner; i += THREAD_PER_BLOCK) {
        curOutput[i] = curInput[i];
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmCatBatchKernel(uint8_t **inputs, uint8_t *output, int outer, int channels, int inner) {
    int bid = blockIdx.x / outer, oid = blockIdx.x % outer;
    uint8_t *curInput = inputs[bid] + oid * inner;
    uint8_t *curOutput = output + oid * channels * inner + bid * inner;

    for (int i = threadIdx.x; i < inner; i += THREAD_PER_BLOCK) {
        curOutput[i] = curInput[i];
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmMatMulTransBBatchKernel(uint8_t** pointer, float alpha) {
    int id = blockIdx.x;
    float *input0 = (float*)pointer[id * 8 + 0];
    float *input1 = (float*)pointer[id * 8 + 1];
    float *output = (float*)pointer[id * 8 + 2];
    int n = (int)((size_t)pointer[id * 8 + 3]);
    int m = (int)((size_t)pointer[id * 8 + 4]);
    int k = (int)((size_t)pointer[id * 8 + 5]);
    int input0Stride = (int)((size_t)pointer[id * 8 + 6]);
    int input1Stride = (int)((size_t)pointer[id * 8 + 7]);

    int tid = threadIdx.x;
    int pera = 4, perb = 4;
    float cura[4][4], curb[4][4], curc[4][4];
    int cnta = (n - 1) / pera + 1, cntb = (k - 1) / perb + 1;
    for (int taskId = tid; taskId < cnta * cntb; taskId += THREAD_PER_BLOCK) {
        int taska = taskId / cntb, taskb = taskId % cntb;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                cura[i][j] = 0;
                curb[i][j] = 0;
                curc[i][j] = 0;
            }
        }

        for (int l = 0; l < m; l += 4) {
            for (int a = taska * pera; a < (taska + 1) * pera && a < n; a++) {
#pragma unroll
                for (int x = 0; x < 4; x++) {
                    cura[a - taska * pera][x] = input0[a * input0Stride + l + x];
                }
            }
            for (int b = taskb * perb; b < (taskb + 1) * perb && b < k; b++) {
#pragma unroll
                for (int x = 0; x < 4; x++) {
                    curb[b - taskb * perb][x] = input1[b * input1Stride + l + x];
                }
            }
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        curc[i][j] += cura[i][k] * curb[j][k];
                    }
                }
            }
        }

        if ((taska + 1) * pera <= n && (taskb + 1) * perb <= k) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    output[(taska * pera + i) * k + (taskb * perb + j)] = curc[i][j] * alpha;
                }
            }
        } else {
            for (int i = 0; i < pera && taska * pera + i < n; i++) {
                for (int j = 0; j < perb && taskb * perb + j < k; j++) {
                    output[(taska * pera + i) * k + (taskb * perb + j)] = curc[i][j] * alpha;
                }
            }
        }
    }

/*
    int tid = threadIdx.x;
    for (int i = 0; i < n; i++) {
        float *curInput0 = input0 + i * input0Stride;
        for (int j = tid; j < k; j += THREAD_PER_BLOCK) {
            float *curInput1 = input1 + j * input1Stride;
            float sum = 0.0;
            for (int l = 0; l < m; l++) {
                sum += curInput0[l] * curInput1[l];
            }
            output[i * k + j] = sum * alpha;
        }
    }
*/
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmMatMulKernel(uint8_t** pointer, float alpha) {
    int id = blockIdx.x;
    float *input0 = (float*)pointer[id * 8 + 0];
    float *input1 = (float*)pointer[id * 8 + 1];
    float *output = (float*)pointer[id * 8 + 2];
    int n = (int)((size_t)pointer[id * 8 + 3]);
    int m = (int)((size_t)pointer[id * 8 + 4]);
    int k = (int)((size_t)pointer[id * 8 + 5]);
    int input0Stride = (int)((size_t)pointer[id * 8 + 6]);
    int input1Stride = (int)((size_t)pointer[id * 8 + 7]);

    int tid = threadIdx.x;
    int pera = 4, perb = 4;
    float cura[4][4], curb[4][4], curc[4][4];
    int cnta = (n - 1) / pera + 1, cntb = (k - 1) / perb + 1;
    for (int taskId = tid; taskId < cnta * cntb; taskId += THREAD_PER_BLOCK) {
        int taska = taskId / cntb, taskb = taskId % cntb;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                cura[i][j] = 0;
                curb[i][j] = 0;
                curc[i][j] = 0;
            }
        }

        for (int l = 0; l < m; l += 4) {
            for (int a = taska * pera; a < (taska + 1) * pera && a < n; a++) {
#pragma unroll
                for (int x = 0; x < 4; x++) {
                    cura[a - taska * pera][x] = l + x < m ? input0[a * input0Stride + l + x] : 0;
                }
            }
            for (int b = taskb * perb; b < (taskb + 1) * perb && b < k; b++) {
#pragma unroll
                for (int x = 0; x < 4; x++) {
                    curb[b - taskb * perb][x] = l + x < m ? input1[(l + x) * input1Stride + b] : 0;
                }
            }

#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        curc[i][j] += cura[i][k] * curb[j][k];
                    }
                }
            }
        }

        if ((taska + 1) * pera <= n && (taskb + 1) * perb <= k) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    output[(taska * pera + i) * k + (taskb * perb + j)] = curc[i][j] * alpha;
                }
            }
        } else {
            for (int i = 0; i < pera && taska * pera + i < n; i++) {
                for (int j = 0; j < perb && taskb * perb + j < k; j++) {
                    output[(taska * pera + i) * k + (taskb * perb + j)] = curc[i][j] * alpha;
                }
            }
        }
    }

/*
    //int tid = threadIdx.x;
    for (int i = 0; i < n; i++) {
        float *curInput0 = input0 + i * input0Stride;
        for (int j = tid; j < k; j += THREAD_PER_BLOCK) {
            float *curInput1 = input1 + j;
            float sum = 0.0;
            for (int l = 0; l < m; l++) {
                sum += curInput0[l] * curInput1[l * input1Stride];
            }
            output[i * k + j] = sum * alpha;
        }
    }
*/
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmAttentionKernel(float *qd, float *kd, float *vd, float *maskd, float *od,
                                       float scale, int q1, int q2, int k1, int v2,
                                       int group, int qstride, int kstride, int vstride, int ostride,
                                       float *qk, float *temp) {
    int o = blockIdx.x;
    qd += o * qstride;
    kd += (o / group) * kstride;
    vd += (o / group) * vstride;
    od += o * ostride;
    qk += o * k1;
    temp += o * k1;
    for (int i = 0; i < q1; i++) {
        for (int j = threadIdx.x; j < k1; j += THREAD_PER_BLOCK) {
            if (maskd && maskd[i * k1 + j] > 0.99) {
                qk[j] = -10000;
                continue;
            }
            float sum = 0.0f;
            float *tempQd = qd + i * q2, *tempKd = kd + j * q2;
            for (int l = 0; l < q2; l++) {
                sum += tempQd[l] * tempKd[l];
            }
            qk[j] = sum * scale;
        }
        __syncthreads();
        FastllmSoftmaxKernelInner1Func <THREAD_PER_BLOCK> (qk, temp, k1);
        __syncthreads();
        for (int j = threadIdx.x; j < v2; j += THREAD_PER_BLOCK) {
            float *curInput1 = vd + j;
            float sum = 0.0;
            for (int l = 0; l < k1; l++) {
                sum += temp[l] * curInput1[l * v2];
            }
            od[i * v2 + j] = sum;
        }
        __syncthreads();
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmAttentionBatchKernel(float** pointer, float scale, int group) {
    const int params = 16;
    int id = blockIdx.x;
    float *qd = (float*) pointer[id * params + 0];
    float *kd = (float*) pointer[id * params + 1];
    float *vd = (float*) pointer[id * params + 2];
    float *maskd = (float*) pointer[id * params + 3];
    float *od = (float*) pointer[id * params + 4];
    int q1 = (int)(unsigned long long)pointer[id * params + 5];
    int q2 = (int)(unsigned long long)pointer[id * params + 6];
    int k1 = (int)(unsigned long long)pointer[id * params + 7];
    int v2 = (int)(unsigned long long)pointer[id * params + 8];
    int qstride = (int)(unsigned long long)pointer[id * params + 9];
    int kstride = (int)(unsigned long long)pointer[id * params + 10];
    int vstride = (int)(unsigned long long)pointer[id * params + 11];
    int ostride = (int)(unsigned long long)pointer[id * params + 12];
    float *qk = (float*)pointer[id * params + 13];
    float *temp = (float*)pointer[id * params + 14];
    int q0 = (int)(unsigned long long)pointer[id * params + 15];

    for (int o = 0; o < q0; o++) {
        qd += o * qstride;
        kd += (o / group) * kstride;
        vd += (o / group) * vstride;
        od += o * ostride;
        qk += o * k1;
        temp += o * k1;

        for (int i = 0; i < q1; i++) {
            for (int j = threadIdx.x; j < k1; j += THREAD_PER_BLOCK) {
                if (maskd && maskd[i * k1 + j] > 0.99) {
                    qk[j] = -10000;
                    continue;
                }
                float sum = 0.0f;
                float *tempQd = qd + i * q2, *tempKd = kd + j * q2;
                for (int l = 0; l < q2; l++) {
                    sum += tempQd[l] * tempKd[l];
                }
                qk[j] = sum * scale;
            }
            __syncthreads();
            FastllmSoftmaxKernelInner1Func<THREAD_PER_BLOCK>(qk, temp, k1);
            __syncthreads();
            for (int j = threadIdx.x; j < v2; j += THREAD_PER_BLOCK) {
                float *curInput1 = vd + j;
                float sum = 0.0;
                for (int l = 0; l < k1; l++) {
                    sum += temp[l] * curInput1[l * v2];
                }
                od[i * v2 + j] = sum;
            }
            __syncthreads();
        }

        qd -= o * qstride;
        kd -= (o / group) * kstride;
        vd -= (o / group) * vstride;
        od -= o * ostride;
        qk -= o * k1;
        temp -= o * k1;
    }
}

void *FastllmCudaPrepareInput(const fastllm::Data &input) {
    void *ret;
    if (input.dataDevice == fastllm::DataDevice::CUDA) {
        ret = (void*)input.cudaData;
    } else {
        ret = (void*)(input.expansionBytes);
        auto state = cudaMemcpy(ret, input.cpuData, input.expansionBytes, cudaMemcpyHostToDevice);
        if (cudaSuccess != state) {
            printf("Error: CUDA error when copy from memory to GPU! state %d", state);
            return nullptr;
        }
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
        auto state = cudaMemcpy(output.cpuData, data, output.expansionBytes, cudaMemcpyDeviceToHost);
        if (cudaSuccess != state)
            printf("Error: CUDA error when copy from GPU to memory! state %d", state);
        FastllmCudaFree(data);
    }

    DeviceSync();
}

bool FastllmCudaMatMulFloatInt8(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    if (weight.cudaData == nullptr || weight.extraCudaData.size() == 0) {
        float *cudaScales;
        cudaError_t state = cudaSuccess;
        state = cudaMalloc(&cudaScales, k * sizeof(float));
        state = cudaMemcpy(cudaScales, weight.scales.data(), k * sizeof(float), cudaMemcpyHostToDevice);
        weight.extraCudaData.push_back((void*)cudaScales);

        uint8_t *cudaZeropoints;
        state = cudaMalloc(&cudaZeropoints, k);
        uint8_t *zeropoints = new uint8_t[k];
        for (int i = 0; i < k; i++) {
            zeropoints[i] = weight.perChannelsConfigs[i].zeroPoint;
        }
        state = cudaMemcpy(cudaZeropoints, zeropoints, k, cudaMemcpyHostToDevice);
        delete[] zeropoints;
        weight.extraCudaData.push_back((void*)cudaZeropoints);

        float *cudaBiasData;
        state = cudaMalloc(&cudaBiasData, k * sizeof(float));
        if (bias.dims.size() > 0) {
            state = cudaMemcpy(cudaBiasData, (uint8_t*)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(float));
        }
        if (cudaSuccess != state)
            printf("Error: CUDA error when moving bias to device! state %d\n", state);
        weight.extraCudaData.push_back((void*)cudaBiasData);
    }

    float *cudaScales = (float*)weight.extraCudaData[0];
    uint8_t *cudaZeropoints = (uint8_t*)weight.extraCudaData[1];
    float *cudaBiasData = (float*)weight.extraCudaData[2];

    float *cudaInput = (float*)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float*)FastllmCudaPrepareOutput(output);

    if (n >= 8) {
        auto fastllmCublasHandle = getFastllmCublasHandle();
        half *cudaFp16Input, *cudaFp16Output, *cudaFp16Weight;
        cudaFp16Input = (half *) FastllmCudaMalloc(n * m * sizeof(half));
        cudaFp16Output = (half *) FastllmCudaMalloc(n * k * sizeof(half));
        cudaFp16Weight = (half *) FastllmCudaMalloc(k * m * sizeof(half));

        __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
        cublasStatus_t status;

        int len = n * m;
        int threadPerBlock = min(256, len);
        FastllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaFp16Input, len);

        len = k * m;
        FastllmCudaInt82HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((uint8_t*)weight.cudaData,
                                                                                         cudaScales,
                                                                                         cudaZeropoints,
                                                                                         cudaFp16Weight, len, m);

        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              k, n, m,
                              &h_alpha, cudaFp16Weight, AType,
                              m, cudaFp16Input, BType,
                              m, &h_beta,
                              cudaFp16Output, CType,
                              k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw("cublas error");
            exit(0);
        }

        len = n * k;
        FastllmCudaHalf2FlotaKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaFp16Output, cudaOutput, len);
        FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, cudaBiasData, k);

        FastllmCudaFree(cudaFp16Input);
        FastllmCudaFree(cudaFp16Output);
        FastllmCudaFree(cudaFp16Weight);
    } else {
        for (int i = 0; i < n; i++) {
            FastllmGemvInt8Kernel2<256, 1> <<< k, 256 >>>(cudaInput + i * m,
                                                          (uint8_t *) weight.cudaData,
                                                          cudaOutput + i * k,
                                                          cudaBiasData,
                                                          cudaScales,
                                                          cudaZeropoints,
                                                          m, k);
        }
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaMatMulFloatInt4(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    if (weight.cudaData == nullptr || weight.extraCudaData.size() == 0) {
        float *cudaScales;
        cudaError_t state = cudaSuccess;
        state = cudaMalloc(&cudaScales, k * sizeof(float));
        state = cudaMemcpy(cudaScales, weight.scales.data(), k * sizeof(float), cudaMemcpyHostToDevice);
        weight.extraCudaData.push_back((void*)cudaScales);

        uint8_t *cudaZeropoints;
        state = cudaMalloc(&cudaZeropoints, k);
        uint8_t *zeropoints = new uint8_t[k];
        for (int i = 0; i < k; i++) {
            zeropoints[i] = weight.perChannelsConfigs[i].zeroPoint;
        }
        state = cudaMemcpy(cudaZeropoints, zeropoints, k, cudaMemcpyHostToDevice);
        delete[] zeropoints;
        weight.extraCudaData.push_back((void*)cudaZeropoints);

        float *cudaBiasData;
        state = cudaMalloc(&cudaBiasData, k * sizeof(float));
        if (bias.dims.size() > 0) {
            state = cudaMemcpy(cudaBiasData, (uint8_t*)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(float));
        }
        if (cudaSuccess != state)
            printf("Error: CUDA error when moving bias to device! state %d\n", state);
        weight.extraCudaData.push_back((void*)cudaBiasData);
    }

    float *cudaScales = (float*)weight.extraCudaData[0];
    uint8_t *cudaZeropoints = (uint8_t*)weight.extraCudaData[1];
    float *cudaBiasData = (float*)weight.extraCudaData[2];

    float *cudaInput = (float*)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float*)FastllmCudaPrepareOutput(output);

    for (int i = 0; i < n; i++) {
        FastllmGemvInt4Kernel2<256, 1> <<< k, 256 >>>(cudaInput + i * m,
                                                      (uint8_t *) weight.cudaData,
                                                      cudaOutput + i * k,
                                                      cudaBiasData,
                                                      cudaScales,
                                                      cudaZeropoints,
                                                      m, k);
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaMatMulFloatInt4NoZero(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    if (weight.cudaData == nullptr || weight.extraCudaData.size() == 0) {
        float *cudaScales;
        cudaError_t state = cudaSuccess;
        state = cudaMalloc(&cudaScales, k * sizeof(float));
        state = cudaMemcpy(cudaScales, weight.scales.data(), k * sizeof(float), cudaMemcpyHostToDevice);
        weight.extraCudaData.push_back((void*)cudaScales);

        float *cudaMins;
        state = cudaMalloc(&cudaMins, k * sizeof(float));
        float *mins = new float[k];
        for (int i = 0; i < k; i++) {
            mins[i] = weight.perChannelsConfigs[i].min;
        }
        state = cudaMemcpy(cudaMins, mins, k * sizeof(float), cudaMemcpyHostToDevice);
        delete[] mins;
        weight.extraCudaData.push_back((void*)cudaMins);

        float *cudaBiasData;
        state = cudaMalloc(&cudaBiasData, k * sizeof(float));
        if (bias.dims.size() > 0) {
            state = cudaMemcpy(cudaBiasData, (uint8_t*)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(float));
        }
        if (cudaSuccess != state)
            printf("Error: CUDA error when moving bias to device! state %d\n", state);
        weight.extraCudaData.push_back((void*)cudaBiasData);
    }

    float *cudaScales = (float*)weight.extraCudaData[0];
    float *cudaMins = (float*)weight.extraCudaData[1];
    float *cudaBiasData = (float*)weight.extraCudaData[2];

    float *cudaInput = (float*)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float*)FastllmCudaPrepareOutput(output);

    if (n >= 8) {
        auto fastllmCublasHandle = getFastllmCublasHandle();
        half *cudaFp16Input, *cudaFp16Output, *cudaFp16Weight;
        cudaFp16Input = (half *) FastllmCudaMalloc(n * m * sizeof(half));
        cudaFp16Output = (half *) FastllmCudaMalloc(n * k * sizeof(half));
        cudaFp16Weight = (half *) FastllmCudaMalloc(k * m * sizeof(half));

        __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
        cublasStatus_t status;

        int len = n * m;
        int threadPerBlock = min(256, len);
        FastllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaFp16Input,
                                                                                          len);

        len = k * m;
        FastllmCudaInt42HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((uint8_t *) weight.cudaData,
                                                                                         cudaScales,
                                                                                         cudaMins,
                                                                                         cudaFp16Weight, len, m);

        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              k, n, m,
                              &h_alpha, cudaFp16Weight, AType,
                              m, cudaFp16Input, BType,
                              m, &h_beta,
                              cudaFp16Output, CType,
                              k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw("cublas error");
            exit(0);
        }

        len = n * k;
        FastllmCudaHalf2FlotaKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaFp16Output, cudaOutput,
                                                                                           len);
        FastllmCudaBiasKernel <<< n, 256 >>>(cudaOutput, cudaBiasData, k);

        FastllmCudaFree(cudaFp16Input);
        FastllmCudaFree(cudaFp16Output);
        FastllmCudaFree(cudaFp16Weight);
    } else {
        for (int i = 0; i < n; i++) {
            FastllmGemvInt4NoZeroKernel2<256, 1> <<< k, 256 >>>(cudaInput + i * m,
                                                                (uint8_t *) weight.cudaData,
                                                                cudaOutput + i * k,
                                                                cudaBiasData,
                                                                cudaScales,
                                                                cudaMins,
                                                                m, k);
        }
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaMatMulFloat32(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    if (weight.cudaData == nullptr || weight.extraCudaData.size() == 0) {
        float *cudaBiasData;
        cudaError_t state = cudaSuccess;
        state = cudaMalloc(&cudaBiasData, k * sizeof(float));
        if (bias.dims.size() > 0) {
            state = cudaMemcpy(cudaBiasData, (uint8_t*)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(float));
        }
        if (cudaSuccess != state)
            printf("Error: CUDA error when moving bias to device! state %d\n", state);
        weight.extraCudaData.push_back((void*)cudaBiasData);
    }

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

        FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, (float*)weight.extraCudaData[0], k);
    } else {
        FastllmGemvFp32Fp32Kernel2<256, 1> <<< k, 256 >>>(cudaInput, (float *) weight.cudaData, cudaOutput, cudaBiasData, m, k);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaMatMulFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    if (weight.cudaData == nullptr || weight.extraCudaData.size() == 0) {
        float *cudaBiasData;
        cudaError_t state = cudaSuccess;
        state = cudaMalloc(&cudaBiasData, k * sizeof(float));
        if (bias.dims.size() > 0) {
            state = cudaMemcpy(cudaBiasData, (uint8_t*)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(float));
        }
        if (cudaSuccess != state)
            printf("Error: CUDA error when moving bias to device! state %d\n", state);
        weight.extraCudaData.push_back((void*)cudaBiasData);
    }
    float *cudaBiasData = (float*)weight.extraCudaData[0];
    float *cudaInput = (float*)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float*)FastllmCudaPrepareOutput(output);

    if (n > 1) {
        half *cudaFp16Input, *cudaFp16Output;
        cudaFp16Input = (half *) FastllmCudaMalloc(n * m * sizeof(half));
        cudaFp16Output = (half *) FastllmCudaMalloc(n * k * sizeof(half));

        __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
        auto fastllmCublasHandle = getFastllmCublasHandle();
        //cudaDeviceSynchronize();
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
        cublasStatus_t status;

        int len = n * m;
        int threadPerBlock = min(256, len);
        FastllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaFp16Input,
                                                                                          len);

        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              k, n, m,
                              &h_alpha, (half *) weight.cudaData, AType,
                              m, cudaFp16Input, BType,
                              m, &h_beta,
                              cudaFp16Output, CType,
                              k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw("cublas error");
            exit(0);
        }

        len = n * k;
        FastllmCudaHalf2FlotaKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaFp16Output, cudaOutput,
                                                                                           len);
        FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, (float*)weight.extraCudaData[0], k);
        //cudaDeviceSynchronize();

        FastllmCudaFree(cudaFp16Input);
        FastllmCudaFree(cudaFp16Output);
    } else {
        FastllmGemvFp32Fp16Kernel2<256, 1> <<< k, 256 >>>(cudaInput, (half *) weight.cudaData, cudaOutput, cudaBiasData, m, k);
    }

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
std::map<int, std::vector <CudaMemoryBuffer>> cudaBuffersMap;
std::map<int, size_t> noBusyCnt;
std::map<int, std::vector <CudaMemoryBuffer>> bigBuffersMap;

void * FastllmCudaDirectMalloc(size_t size) {
    void * ret;
    cudaError_t state = cudaMalloc(&ret, size);
    if (cudaSuccess != state) {
        printf("Error: CUDA error when allocating %d kB memory! state %d, maybe there's no enough memory left on device.\n", size >> 10, state);
        return nullptr;
    }
    return ret;
}

void FastllmCudaDirectFree(void *ret) {
    cudaError_t state = cudaFree(ret);
    if (cudaSuccess != state) {
        printf("Error: CUDA error when release memory! state %d.\n", state);
    }
}

void * FastllmCudaMalloc(size_t size) {
    int id = -1;
    cudaError_t state = cudaSuccess;
    state = cudaGetDevice(&id);
    if (cudaSuccess != state) {
        printf("Error: CUDA error when find device! state %d", state);
        return nullptr;
    }
    if (size > 1024 * 1024) {
        auto &bigBuffers = bigBuffersMap[id];
        int selId = -1;
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (bigBuffers[i].size >= size && !bigBuffers[i].busy
                && bigBuffers[i].size - size < 1 * 1024 * 1024) {
                if (selId == -1 || bigBuffers[selId].size > bigBuffers[i].size) {
                    selId = i;
                }
            }
        }
        if (selId != -1) {
            bigBuffers[selId].busy = true;
            return bigBuffers[selId].data;
        }

        void * ret;
        state = cudaMalloc(&ret, size);
        if (cudaSuccess != state) {
            printf("Error: CUDA error when allocating %d MB memory! state %d, maybe there's no enough memory left on device.\n", size >> 20, state);
            return nullptr;
        }
        bigBuffers.push_back(CudaMemoryBuffer(ret, size, true));
        return ret;
    }
    auto &cudaBuffers = cudaBuffersMap[id];
    for (int i = 0; i < cudaBuffers.size(); i++) {
        if (cudaBuffers[i].size >= size && !cudaBuffers[i].busy) {
            cudaBuffers[i].busy = true;
            noBusyCnt[id] -= cudaBuffers[i].size;
            return cudaBuffers[i].data;
        }
    }
    void * ret;
    state = cudaMalloc(&ret, size);
    if (cudaSuccess != state) {
        printf("Error: CUDA error when allocating %d KB memory! state %d, maybe there's no enough memory left on device.\n", size >> 10, state);
        return nullptr;
    }
    cudaBuffers.push_back(CudaMemoryBuffer(ret, size, true));
    return ret;
}

void FastllmCudaFree(void *ret) {
    if (ret == nullptr) {
        return;
    }
    if (cudaBuffersMap.empty())
        return;
    cudaError_t state = cudaSuccess;
    for (auto &it: cudaBuffersMap) {
        if (noBusyCnt[it.first] > 1024 * 1024 * 1024) {
            auto &cudaBuffers = it.second;
            std::vector <CudaMemoryBuffer> temp;
            for (int i = 0; i < cudaBuffers.size(); i++) {
                if (!cudaBuffers[i].busy) {
                    state = cudaSetDevice(it.first);
                    state = cudaFree(cudaBuffers[i].data);
                    if (cudaSuccess != state)
                        printf("Error: CUDA error when release memory on device %d! state %d.\n", it.first, state);
                } else {
                    temp.push_back(cudaBuffers[i]);
                }
            }
            cudaBuffers.clear();
            it.second = temp;
            noBusyCnt[it.first] = 0;
        }
    }

    for (auto &it: cudaBuffersMap) {
        auto &cudaBuffers = it.second;
        for (int i = 0; i < cudaBuffers.size(); i++) {
            if (cudaBuffers[i].data == ret) {
                noBusyCnt[it.first] += cudaBuffers[i].size;
                cudaBuffers[i].busy = false;
                return;
            }
        }
        auto &bigBuffers = bigBuffersMap[it.first];
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (bigBuffers[i].data == ret) {
                bigBuffers[i].busy = false;
                return;
            }
        }
    }
    state = cudaFree(ret);
    if (cudaSuccess != state) {
        printf("CUDA error when release memory! state %d.\n", state);
        return;
    }
}

void FastllmCudaMallocBigBuffer(size_t size) {
    void * ret;
    int id = -1;
    cudaGetDevice(&id);
    auto &bigBuffers = bigBuffersMap[id];
    cudaMalloc(&ret, size);
    auto state = cudaMalloc(&ret, size);
    if (cudaSuccess != state) {
        printf("Error: CUDA error when allocating %d MB memory! state %d. maybe there's no enough memory left on device.\n", size >> 20, state);
    }
    bigBuffers.push_back(CudaMemoryBuffer(ret, size, false));
}

void FastllmCudaClearBigBuffer() {
    int id = -1;
    cudaGetDevice(&id);
    if (bigBuffersMap.empty())
        return;
    cudaError_t state = cudaSuccess;
    for (auto &it : bigBuffersMap) {
        auto &bigBuffers = it.second;
        std::vector <CudaMemoryBuffer> temp;
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (!bigBuffers[i].busy) {
                state = cudaSetDevice(it.first);
                state = cudaFree(bigBuffers[i].data);
                if (cudaSuccess != state)
                    printf("Error: CUDA error when release memory on device %d! state %d.\n", it.first, state);
            } else {
                temp.push_back(bigBuffers[i]);
            }
        }
        bigBuffers.clear();
        bigBuffers = temp;
    }
    cudaSetDevice(id);
}

void FastllmCudaCopyFromHostToDevice(void *dst, void *src, size_t size) {
    auto state = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    if (cudaSuccess != state)
        printf("Error: CUDA error when copy from memory to GPU! state %d.\n", state);
    //cudaDeviceSynchronize();
}

void FastllmCudaCopyFromDeviceToHost(void *dst, void *src, size_t size) {
    auto state = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    if (cudaSuccess != state)
        printf("Error: CUDA error when copy from GPU to memory! state %d.\n", state);
    //cudaDeviceSynchronize();
}

void FastllmCudaCopyFromDeviceToDevice(void *dst, void *src, size_t size) {
    auto state = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    if (cudaSuccess != state)
        printf("Error: CUDA error when copy on GPU! state %d.\n", state);
    //cudaDeviceSynchronize();
}

void FastllmCudaMemcpy2DDeviceToDevice(void * 	dst, size_t 	dpitch, const void * 	src,
                                       size_t 	spitch, size_t 	width, size_t 	height) {
    cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice);
    //cudaDeviceSynchronize();
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmMemcpyBatchKernel (uint8_t** pointer) {
    int id = blockIdx.x;
    uint8_t *dst = pointer[id * 3];
    uint8_t *src = pointer[id * 3 + 1];
    size_t len = (size_t)(pointer[id * 3 + 2]);
    for (int i = threadIdx.x; i < len; i += THREAD_PER_BLOCK) {
        dst[i] = src[i];
    }
}

void FastllmCudaMemcpy2DDeviceToDeviceBatch(void ** 	dsts, size_t *	dpitchs, void ** 	srcs,
                                            size_t *	spitchs, size_t *widths, size_t *	heights,
                                            int batch) {
    int total = 0;
    for (int i = 0; i < batch; i++) {
        total += heights[i];
    }
    uint8_t ** pointers = (uint8_t**)FastllmCudaMalloc(sizeof(uint8_t*) * total * 3);
    uint8_t ** cpuPointers = new uint8_t*[total * 3];
    int cur = 0;
    for (int i = 0; i < batch; i++) {
        for (int h = 0; h < heights[i]; h++) {
            cpuPointers[cur * 3 + 0] = (uint8_t*)dsts[i] + h * dpitchs[i];
            cpuPointers[cur * 3 + 1] = (uint8_t*)srcs[i] + h * spitchs[i];
            cpuPointers[cur * 3 + 2] = (uint8_t*)(widths[i]);

            cur++;
        }
    }
    cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * total * 3, cudaMemcpyHostToDevice);
    FastllmMemcpyBatchKernel <128> <<<total, 128>>> (pointers);

    FastllmCudaFree(pointers);
    delete[] cpuPointers;

    DeviceSync();
}

bool FastllmCudaGeluNew(const fastllm::Data &input, fastllm::Data &output) {
    int len = input.Count(0);
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareOutput(output);
    int threadPerBlock = min(256, len);
    FastllmGeluKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaOutput, len);
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaSilu(const fastllm::Data &input, fastllm::Data &output) {
    int len = input.Count(0);
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareOutput(output);
    int threadPerBlock = min(256, len);
    FastllmSiluKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaOutput, len);
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaSwiglu(const fastllm::Data &input, fastllm::Data &output) {
    int len = output.Count(0);
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareOutput(output);
    int spatial = input.Count(input.dims.size() - 1), mid = spatial / 2;

    int threadPerBlock = min(256, len);
    FastllmSwigluKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaOutput, len, spatial, mid);

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaMul(const fastllm::Data &input, float v, fastllm::Data &output) {
    int len = input.Count(0);
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareOutput(output);
    int threadPerBlock = min(256, len);
    FastllmMulKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaOutput, v, len);
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaAddTo(fastllm::Data &input0, const fastllm::Data &input1, float alpha) {
    int len = input0.Count(0);
    float *cudaData = (float *) FastllmCudaPrepareInput(input0);
    float *input1Data = (float *) FastllmCudaPrepareInput(input1);

    int threadPerBlock = min(256, len);
    FastllmAddToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaData, input1Data, alpha, len);
    FastllmCudaFinishInput(input1, input1Data);
    FastllmCudaFinishOutput(input0, cudaData);
    return true;
}

bool FastllmCudaMulTo(fastllm::Data &input0, const fastllm::Data &input1, float alpha) {
    int len = input0.Count(0);
    float *cudaData = (float *) FastllmCudaPrepareInput(input0);
    float *input1Data = (float *) FastllmCudaPrepareInput(input1);

    int threadPerBlock = min(256, len);
    FastllmMulToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaData, input1Data, alpha, len);
    FastllmCudaFinishInput(input1, input1Data);
    FastllmCudaFinishOutput(input0, cudaData);
    return true;
}

bool FastllmCudaAttentionMask(fastllm::Data &input, const fastllm::Data &mask, float maskValue) {
    int spatial = input.Count(2), n = input.dims[0], m = input.dims[1];
    float *cudaData = (float *) FastllmCudaPrepareInput(input);
    float *maskData = (float *) FastllmCudaPrepareInput(mask);

    FastllmAttentionMaskKernel <256> <<< n * m, 256>>>(cudaData, maskData, maskValue,
                                                       n, m, spatial);
    FastllmCudaFinishInput(mask, maskData);
    FastllmCudaFinishOutput(input, cudaData);
    return true;
}

bool FastllmCudaAlibiMask(fastllm::Data &input, const fastllm::Data &mask, float maskValue) {
    int n = input.dims[0], m = input.dims[1];
    int spn = input.dims[2], spm = input.dims[3];
    int spatial = input.Count(2);
    float *cudaData = (float *) FastllmCudaPrepareInput(input);
    float *maskData = (float *) FastllmCudaPrepareInput(mask);

    FastllmAlibiMaskKernel <256> <<< n * m, 256>>>(cudaData, maskData, maskValue,
                                                   n, m, spn, spm, spatial);
    FastllmCudaFinishInput(mask, maskData);
    FastllmCudaFinishOutput(input, cudaData);
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
            FastllmSoftmaxKernelInner1 <1> <<< outer, 1 >>> (cudaInput, cudaOutput, outer, channels);
        } else if (channels < 64) {
            FastllmSoftmaxKernelInner1 <8> <<< outer, 8 >>> (cudaInput, cudaOutput, outer, channels);
        } else if (channels < 512) {
            FastllmSoftmaxKernelInner1 <64> <<< outer, 64 >>> (cudaInput, cudaOutput, outer, channels);
        } else {
            FastllmSoftmaxKernelInner1 <256> <<< outer, 256 >>> (cudaInput, cudaOutput, outer, channels);
        }

    } else {
        printf("softmax error.\n");
        exit(0);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaSoftmaxBatch(fastllm::Data **inputs, fastllm::Data **outputs, int axis, int batch) {
    int total = 0;
    for (int b = 0; b < batch; b++) {
        auto &input = *inputs[b];
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        int outer = input.Count(0) / input.Count(axis);
        total += outer;
    }
    uint8_t ** pointers = (uint8_t**)FastllmCudaMalloc(sizeof(uint8_t*) * total * 3);
    uint8_t ** cpuPointers = new uint8_t*[total * 3];
    int cur = 0;

    for (int b = 0; b < batch; b++) {
        auto &input = *inputs[b];
        auto &output = *outputs[b];
        float *cudaInput = (float *) input.cudaData;
        float *cudaOutput = (float *) output.cudaData;

        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        int outer = input.Count(0) / input.Count(axis);
        int channels = input.dims[axis];
        int inner = input.Count(axis + 1);

        if (inner == 1) {
            for (int o = 0; o < outer; o++) {
                cpuPointers[cur * 3 + 0] = (uint8_t*)(cudaInput + o * channels);
                cpuPointers[cur * 3 + 1] = (uint8_t*)(cudaOutput + o * channels);
                cpuPointers[cur * 3 + 2] = (uint8_t*)((size_t)channels);
                cur++;
            }
        } else {
            printf("softmax error.\n");
            exit(0);
        }
    }

    cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * total * 3, cudaMemcpyHostToDevice);
    FastllmSoftmaxKernelBatchInner1 <256> <<<total, 256>>> (pointers);

    FastllmCudaFree(pointers);
    delete[] cpuPointers;
    DeviceSync();
    return true;
}

bool FastllmCudaRMSNorm(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float eps) {
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareInput(output);

    int dimsLen = input.dims.size();
    int axis = dimsLen - 1;
    int outer = input.Count(0) / input.Count(axis);
    int channels = input.dims[axis];

    if (channels < 64) {
        FastllmRMSNormKernelInner1<1> <<< outer, 1 >>>(cudaInput, (float *) weight.cudaData, cudaOutput, outer, channels, eps);
    } else if (channels < 512) {
        FastllmRMSNormKernelInner1<64> <<< outer, 64 >>>(cudaInput, (float *) weight.cudaData, cudaOutput, outer, channels, eps);
    } else {
        FastllmRMSNormKernelInner1<512> <<< outer, 512 >>>(cudaInput, (float *) weight.cudaData, cudaOutput, outer, channels, eps);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaLayerNorm(const fastllm::Data &input, fastllm::Data &gamma, fastllm::Data &beta, fastllm::Data &output, int axis) {
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareInput(output);

    int dimsLen = input.dims.size();
    axis = (axis % dimsLen + dimsLen) % dimsLen;
    int outer = input.Count(0) / input.Count(axis);
    int channels = input.dims[axis];
    int inner = input.strides[axis];

    if (inner == 1) {
        if (channels < 64) {
            FastllmLayerNormKernelInner1<1> <<< outer, 1 >>>(cudaInput, (float *) gamma.cudaData,
                                                             (float *) beta.cudaData, cudaOutput,
                                                             outer, channels);
        } else if (channels < 512) {
            FastllmLayerNormKernelInner1<64> <<< outer, 64 >>>(cudaInput, (float *) gamma.cudaData,
                                                               (float *) beta.cudaData, cudaOutput,
                                                               outer, channels);
        } else {
            FastllmLayerNormKernelInner1<512> <<< outer, 512 >>>(cudaInput, (float *) gamma.cudaData,
                                                                 (float *) beta.cudaData, cudaOutput,
                                                                 outer, channels);
        }
    } else {
        printf("layernorm error.\n");
        exit(0);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaTopK(const fastllm::Data &input, fastllm::Data &output, int topk) {
    if (topk != 1) {
        printf("topk: unsupport topk > 1.");
        exit(0);
    }

    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareInput(output);

    int dimsLen = input.dims.size();
    int outer = input.Count(0) / input.Count(dimsLen - 1);
    int channels = input.dims[dimsLen - 1];

    FastllmLayerNormKernelTop1 <256> <<< outer, 256 >>> (cudaInput, cudaOutput, channels);
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaPermute(fastllm::Data &input, const std::vector<int> &axis) {
    if (input.dataDevice != fastllm::DataDevice::CUDA) {
        printf("permute: data should in cuda.\n");
        exit(0);
    }
    int len = input.Count(0);
    float *tempData = (float *)FastllmCudaMalloc(len * sizeof(float));
    cudaMemcpy(tempData, input.cudaData, len * sizeof(float), cudaMemcpyDeviceToDevice);

    std::vector<int> new_dims;
    for (int i = 0; i < axis.size(); i++) {
        new_dims.push_back(input.dims[axis[i]]);
    }
    if (axis == std::vector <int> {1, 0, 2}) {
        int n = input.dims[0];
        int m = input.dims[1];
        int k = input.dims[2];
        FastllmTransposeByRowKernel <256> <<< n * m, 256 >>>
                ((uint8_t*)input.cudaData, (uint8_t*)tempData, n, m, k * input.unitSize);
        input.Resize(new_dims);
    } else if (axis == std::vector <int> {2, 0, 1, 3}) {
        int n = input.dims[0] * input.dims[1];
        int m = input.dims[2];
        int k = input.dims[3];
        FastllmTransposeByRowKernel <256> <<< n * m, 256 >>>
                ((uint8_t*)input.cudaData, (uint8_t*)tempData, n, m, k * input.unitSize);
        input.Resize(new_dims);
    } else {
        std::vector<int> temp;
        int len = input.Count(0);
        for (int i = 0; i < axis.size(); i++) {
            temp.push_back(axis[i]);
        }
        for (int i = 0; i < axis.size(); i++) {
            temp.push_back(input.Count(i + 1));
        }
        input.Resize(new_dims);
        for (int i = 0; i < axis.size(); i++) {
            temp.push_back(input.Count(i + 1));
        }

        int *cudaTemp = (int *) FastllmCudaMalloc(temp.size() * sizeof(int));
        cudaMemcpy(cudaTemp, temp.data(), temp.size() * sizeof(int), cudaMemcpyHostToDevice);
        int threadPerBlock = min(256, len);
        FastllmPermuteKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>((float *) input.cudaData,
                                                                                    tempData, cudaTemp,
                                                                                    (int) axis.size(), len);
        FastllmCudaFree(cudaTemp);
    }

    FastllmCudaFree(tempData);
    return true;
}

bool FastllmCudaAttention(const fastllm::Data &q, const fastllm::Data &k, const fastllm::Data &v,
                          const fastllm::Data &mask, const fastllm::Data &output, int group, float scale) {
    int q0 = q.dims[0], q1 = q.dims[1], q2 = q.dims[2], k0 = k.dims[0], k1 = k.dims[1], v2 = v.dims[2];
    float *qd = (float*)q.cudaData;
    float *kd = (float*)k.cudaData;
    float *vd = (float*)v.cudaData;
    float *maskd = mask.dims.size() > 0 ? (float*)mask.cudaData : nullptr;
    float *od = (float*)output.cudaData;
    int batch = (mask.dims.size() == 3) ? mask.dims[0] : 1;
    int maskStride = (mask.dims.size() == 3 ? mask.strides[0] : mask.Count(0));
    if (false) {
        float *qk = (float *) FastllmCudaMalloc(q0 * k1 * sizeof(float));
        float *temp = (float *) FastllmCudaMalloc(q0 * k1 * sizeof(float));
        FastllmAttentionKernel<256> <<<q0, 256>>>(qd, kd, vd, maskd, od,
                                                  scale, q1, q2, k1, v2,
                                                  group, q.strides[0], k.strides[0], v.strides[0], output.strides[0],
                                                  qk, temp);
        FastllmCudaFree(qk);
        FastllmCudaFree(temp);
        return true;
    }

    if (q1 > 1024) {
        float *qk = (float *) FastllmCudaMalloc(q1 * k1 * sizeof(float));
        float beta = 0, one = 1;
        auto fastllmCublasHandle = getFastllmCublasHandle();
        cublasStatus_t status;


        for (int i = 0; i < q0; i++) {
            status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                               CUBLAS_OP_T, CUBLAS_OP_N,
                                               k1, q1, q2, &scale,
                                               kd + (i / group) * k.Count(1), k.strides[1], k.Count(1),
                                               qd + i * q.Count(1), q.strides[1], q.Count(1),
                                               &beta,
                                               qk, k1, k1 * q1, 1);
            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("status = %d\n", (int) status);
                printf("Error: cublas error during MatMulTransB in Attention operator.\n");
                throw ("cublas error");
                exit(0);
            }

            if (maskd) {
                SimpleMask<256> <<< (q1 * k1 / 256) + 1, 256>>>(qk, maskd + (i / (q0 / batch)) * maskStride, -10000, q1 * k1);
            }

            int outer = q1;
            if (k1 < 8) {
                FastllmSoftmaxKernelInner1<1> <<< outer, 1 >>>(qk, qk, outer, k1);
            } else if (k1 < 64) {
                FastllmSoftmaxKernelInner1<8> <<< outer, 8 >>>(qk, qk, outer, k1);
            } else if (k1 < 512) {
                FastllmSoftmaxKernelInner1<64> <<< outer, 64 >>>(qk, qk, outer, k1);
            } else {
                FastllmSoftmaxKernelInner1<256> <<< outer, 256 >>>(qk, qk, outer, k1);
            }

            status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                               CUBLAS_OP_N, CUBLAS_OP_N,
                                               v2, q1, k1, &one,
                                               vd + (i / group) * v.Count(1), v.strides[1], v.Count(1),
                                               qk, k1, k1 * q1,
                                               &beta,
                                               od + i * v2 * q1, v2, v2 * q1, 1);
            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("status = %d\n", (int) status);
                printf("Error: cublas error during MatMul in Attention operator.\n");
                throw ("cublas error");
                exit(0);
            }
        }

        FastllmCudaFree(qk);
        DeviceSync();
        return true;
    }

    if (true) {
        float *qk = (float *) FastllmCudaMalloc(q0 * q1 * k1 * sizeof(float));
        float *temp = (float *) FastllmCudaMalloc(q0 * q1 * k1 * sizeof(float));
        float beta = 0, one = 1;
        auto fastllmCublasHandle = getFastllmCublasHandle();
        cublasStatus_t status;

        status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_T, CUBLAS_OP_N,
                                           k1, q1 * group, q2, &scale,
                                           kd, k.strides[1], k.Count(1),
                                           qd, q.strides[1], q.Count(1) * group,
                                           &beta,
                                           qk, k1, k1 * q1 * group, q0 / group);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("status = %d\n", (int) status);
            printf("Error: cublas error during MatMulTransB in Attention operator.\n");
            throw ("cublas error");
            exit(0);
        }

        if (maskd) {
            int spatial = q1 * k1, n = batch, m = q0 / batch;
            FastllmAttentionMaskKernel <256> <<< n * m, 256>>>(qk, maskd, -10000, n, m, spatial);
        }

        int outer = q0 * q1;
        if (k1 < 8) {
            FastllmSoftmaxKernelInner1<1> <<< outer, 1 >>>(qk, temp, outer, k1);
        } else if (k1 < 64) {
            FastllmSoftmaxKernelInner1<8> <<< outer, 8 >>>(qk, temp, outer, k1);
        } else if (k1 < 512) {
            FastllmSoftmaxKernelInner1<64> <<< outer, 64 >>>(qk, temp, outer, k1);
        } else {
            FastllmSoftmaxKernelInner1<256> <<< outer, 256 >>>(qk, temp, outer, k1);
        }

        status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_N, CUBLAS_OP_N,
                                           v2, q1 * group, k1, &one,
                                           vd, v.strides[1], v.Count(1),
                                           temp, k1, k1 * q1 * group,
                                           &beta,
                                           od, v2, v2 * q1 * group, q0 / group);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("status = %d\n", (int) status);
            printf("Error: cublas error during MatMul in Attention operator.\n");
            throw ("cublas error");
            exit(0);
        }
        FastllmCudaFree(qk);
        FastllmCudaFree(temp);
        DeviceSync();
        return true;
    }
    return true;
}

bool FastllmCudaBatchMatMul(const fastllm::Data &input0, const fastllm::Data &input1, fastllm::Data &output,
                            int input0Spatial, int input1Spatial, int outputSpatial,
                            int input0Stride, int input1Stride,
                            int batch, int n, int m, int k, float alpha) {
    float *cudaInput0 = (float *) FastllmCudaPrepareInput(input0);
    float *cudaInput1 = (float *) FastllmCudaPrepareInput(input1);
    float *cudaOutput = (float *) FastllmCudaPrepareOutput(output);
    float beta = 0;
    auto fastllmCublasHandle = getFastllmCublasHandle();
    cublasStatus_t status;

    status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                       CUBLAS_OP_N, CUBLAS_OP_N,
                                       k, n, m, &alpha,
                                       cudaInput1, input1Stride, input1Spatial,
                                       cudaInput0, input0Stride, input0Spatial,
                                       &beta,
                                       cudaOutput, k, k * n, batch);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("status = %d\n", (int)status);
        printf("%d %d %d\n", k, n, m);
        printf("Error: cublas error in batch MatMul.\n");
        throw("cublas error");
        exit(0);
    }

    FastllmCudaFinishInput(input0, cudaInput0);
    FastllmCudaFinishInput(input1, cudaInput1);
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
    float beta = 0;
    auto fastllmCublasHandle = getFastllmCublasHandle();
    cublasStatus_t status;

    status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                       CUBLAS_OP_T, CUBLAS_OP_N,
                                       k, n, m, &alpha,
                                       cudaInput1, input1Stride, input1Spatial,
                                       cudaInput0, input0Stride, input0Spatial,
                                       &beta,
                                       cudaOutput, k, k * n, batch);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("status = %d\n", (int)status);
        printf("%d %d %d\n", k, n, m);
        printf("Error: cublas error in batch MatMulTransB.\n");
        throw("cublas error");
        exit(0);
    }

    FastllmCudaFinishInput(input0, cudaInput0);
    FastllmCudaFinishInput(input1, cudaInput1);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaRotatePosition2D(fastllm::Data &data, const fastllm::Data &positionIds,
                                 const fastllm::Data &sinData, const fastllm::Data &cosData, int rotaryDim) {
    float *cudaData = (float *) FastllmCudaPrepareInput(data);
    float *cudaPositionIds = (float *) FastllmCudaPrepareInput(positionIds);
    float *cudaSin = (float *) FastllmCudaPrepareInput(sinData);
    float *cudaCos = (float *) FastllmCudaPrepareInput(cosData);

    int outer = data.dims[0] * data.dims[1];
    int spatial = data.Count(2);
    int len = data.dims[0], bs = data.dims[1];
    int n = data.dims[2], m = data.dims[3];
    FastllmRotatePosition2DKernel <<< outer * 2 * n, min(rotaryDim, m / 4) >>> (cudaData, cudaPositionIds, cudaSin, cudaCos,
                                                                                len, bs, spatial, n, m,
                                                                                (int)positionIds.dims.back(), (int)sinData.dims[1], rotaryDim);

    FastllmCudaFinishInput(positionIds, cudaPositionIds);
    FastllmCudaFinishInput(sinData, cudaSin);
    FastllmCudaFinishInput(cosData, cudaCos);
    FastllmCudaFinishOutput(data, cudaData);

    return true;
}

bool FastllmCudaNearlyRotatePosition2D(fastllm::Data &data, const fastllm::Data &positionIds,
                                       const fastllm::Data &sinData, const fastllm::Data &cosData, int rotaryDim) {
    float *cudaData = (float *) FastllmCudaPrepareInput(data);
    float *cudaPositionIds = (float *) FastllmCudaPrepareInput(positionIds);
    float *cudaSin = (float *) FastllmCudaPrepareInput(sinData);
    float *cudaCos = (float *) FastllmCudaPrepareInput(cosData);

    int outer = data.dims[0] * data.dims[1];
    int spatial = data.Count(2);
    int len = data.dims[0], bs = data.dims[1];
    int n = data.dims[2], m = data.dims[3];
    FastllmNearlyRotatePosition2DKernel <<< outer * n, min(rotaryDim, m / 4) >>> (cudaData, cudaPositionIds, cudaSin, cudaCos,
                                                                                  len, bs, spatial, n, m,
                                                                                  (int)positionIds.dims.back(), (int)sinData.dims[1], rotaryDim);

    FastllmCudaFinishInput(positionIds, cudaPositionIds);
    FastllmCudaFinishInput(sinData, cudaSin);
    FastllmCudaFinishInput(cosData, cudaCos);
    FastllmCudaFinishOutput(data, cudaData);
    return true;
}

bool FastllmCudaLlamaRotatePosition2D(fastllm::Data &data, const fastllm::Data &positionIds,
                                      const fastllm::Data &sinData, const fastllm::Data &cosData, int rotaryDim) {
    float *cudaData = (float *) FastllmCudaPrepareInput(data);
    float *cudaPositionIds = (float *) FastllmCudaPrepareInput(positionIds);
    float *cudaSin = (float *) FastllmCudaPrepareInput(sinData);
    float *cudaCos = (float *) FastllmCudaPrepareInput(cosData);

    int outer = data.dims[0] * data.dims[1];
    int spatial = data.Count(2);
    int bs = data.dims[0], len = data.dims[1];
    int n = data.dims[2], m = data.dims[3];
    FastllmLlamaRotatePosition2DKernel <<< outer * n, min(rotaryDim, m / 2) >>> (cudaData, cudaPositionIds, cudaSin, cudaCos,
                                                                                 len, bs, spatial, n, m,
                                                                                 (int)positionIds.dims.back(), (int)sinData.dims[1], rotaryDim);

    FastllmCudaFinishInput(positionIds, cudaPositionIds);
    FastllmCudaFinishInput(sinData, cudaSin);
    FastllmCudaFinishInput(cosData, cudaCos);
    FastllmCudaFinishOutput(data, cudaData);
    return true;
}

bool FastllmCudaApplyLognAttn (fastllm::Data &input, fastllm::Data &lognAttn, fastllm::Data &positionIds) {
    float *inputData = (float *) input.cudaData;
    float *lognData = (float *) lognAttn.cudaData;
    float *posData = (float *) positionIds.cudaData;
    int batch = input.dims[0];
    int seqLen = input.dims[1];
    int spatial = input.Count(2);

    FastllmApplyLognAttnKernel <256> <<<batch * seqLen, 256>>> (inputData, lognData, posData, batch, seqLen, spatial);
    return true;
}

bool FastllmCudaAttentionBatch(fastllm::Data **q, fastllm::Data **k, fastllm::Data **v,
                               fastllm::Data **mask, fastllm::Data **output, int group, float scale, int batch) {
    int k0 = k[0]->dims[0];
    size_t memSum = 0;
    for (int b = 0; b < batch; b++) {
        memSum += q[b]->dims[0] * q[b]->dims[1] * k[b]->dims[1];
    }
    float *mem = (float*) FastllmCudaMalloc(memSum * sizeof(float));
    float **qk = new float*[batch];
    memSum = 0;
    for (int b = 0; b < batch; b++) {
        int s = q[b]->dims[0] * q[b]->dims[1] * k[b]->dims[1];
        qk[b] = mem + memSum;
        memSum += s;
    }

    if (true) {
        uint8_t ** pointers = (uint8_t**)FastllmCudaMalloc(sizeof(uint8_t*) * batch * k0 * 8);
        uint8_t ** cpuPointers = new uint8_t*[batch * k0 * 8];
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < k0; i++) {
                cpuPointers[(b * k0 + i) * 8 + 0] = (uint8_t *) q[b]->cudaData + i * group * q[b]->dims[1] * q[b]->dims[2] * sizeof(float);
                cpuPointers[(b * k0 + i) * 8 + 1] = (uint8_t *) k[b]->cudaData + i * k[b]->strides[0] * sizeof(float);
                cpuPointers[(b * k0 + i) * 8 + 2] = (uint8_t *) qk[b] + i * group * q[b]->dims[1] * k[b]->dims[1] * sizeof(float);
                cpuPointers[(b * k0 + i) * 8 + 3] = (uint8_t *) (size_t) (group * q[b]->dims[1]);
                cpuPointers[(b * k0 + i) * 8 + 4] = (uint8_t *) (size_t) q[b]->dims[2];
                cpuPointers[(b * k0 + i) * 8 + 5] = (uint8_t *) (size_t) k[b]->dims[1];
                cpuPointers[(b * k0 + i) * 8 + 6] = (uint8_t *) (size_t) q[b]->strides[1];
                cpuPointers[(b * k0 + i) * 8 + 7] = (uint8_t *) (size_t) k[b]->strides[1];
            }
        }
        cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * batch * k0 * 8, cudaMemcpyHostToDevice);
        FastllmMatMulTransBBatchKernel <128> <<<batch * k0, 128>>> (pointers, scale);
        FastllmCudaFree(pointers);
        delete[] cpuPointers;
    }
    if (true) {
        int total = 0;
        for (int b = 0; b < batch; b++) {
            int outer = q[b]->dims[0] * q[b]->dims[1];
            total += outer;
        }
        uint8_t ** pointers = (uint8_t**)FastllmCudaMalloc(sizeof(uint8_t*) * total * 3);
        uint8_t ** cpuPointers = new uint8_t*[total * 3];
        int cur = 0;
        for (int b = 0; b < batch; b++) {
            int outer = q[b]->dims[0] * q[b]->dims[1];
            int channels = k[b]->dims[1];
            for (int o = 0; o < outer; o++) {
                cpuPointers[cur * 3 + 0] = (uint8_t*)(qk[b] + o * channels);
                cpuPointers[cur * 3 + 1] = (uint8_t*)(qk[b] + o * channels);
                cpuPointers[cur * 3 + 2] = (uint8_t*)((size_t)channels);
                cur++;
            }
        }
        cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * total * 3, cudaMemcpyHostToDevice);
        FastllmSoftmaxKernelBatchInner1 <256> <<<total, 256>>> (pointers);

        FastllmCudaFree(pointers);
        delete[] cpuPointers;
    }
    if (true) {
        uint8_t ** pointers = (uint8_t**)FastllmCudaMalloc(sizeof(uint8_t*) * batch * k0 * 8);
        uint8_t ** cpuPointers = new uint8_t*[batch * k0 * 8];
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < k0; i++) {
                cpuPointers[(b * k0 + i) * 8 + 0] = (uint8_t *) qk[b] + i * group * q[b]->dims[1] * k[b]->dims[1] * sizeof(float);
                cpuPointers[(b * k0 + i) * 8 + 1] = (uint8_t *) v[b]->cudaData + i * v[b]->strides[0] * sizeof(float);
                cpuPointers[(b * k0 + i) * 8 + 2] = (uint8_t *) output[b]->cudaData + i * group * q[b]->dims[1] * v[b]->dims[2] * sizeof(float);
                cpuPointers[(b * k0 + i) * 8 + 3] = (uint8_t *) (size_t) (group * q[b]->dims[1]);
                cpuPointers[(b * k0 + i) * 8 + 4] = (uint8_t *) (size_t) k[b]->dims[1];
                cpuPointers[(b * k0 + i) * 8 + 5] = (uint8_t *) (size_t) v[b]->dims[2];
                cpuPointers[(b * k0 + i) * 8 + 6] = (uint8_t *) (size_t) k[b]->dims[1];
                cpuPointers[(b * k0 + i) * 8 + 7] = (uint8_t *) (size_t) v[b]->strides[1];
            }
        }
        cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * batch * k0 * 8, cudaMemcpyHostToDevice);
        FastllmMatMulKernel <128> <<<batch * k0, 128>>> (pointers, 1.0f);
        FastllmCudaFree(pointers);
        delete[] cpuPointers;
    }

    FastllmCudaFree(mem);
    delete[] qk;
/*
    {
        const int params = 16;
        float **pointers = (float **) FastllmCudaMalloc(sizeof(float *) * batch * params);
        float **cpuPointers = new float *[batch * params];

        float **qk = new float *[batch];
        float **temp = new float *[batch];
        for (int b = 0; b < batch; b++) {
            qk[b] = (float *) FastllmCudaMalloc(q[b]->dims[0] * k[b]->dims[1] * sizeof(float));
            temp[b] = (float *) FastllmCudaMalloc(q[b]->dims[0] * k[b]->dims[1] * sizeof(float));

            cpuPointers[b * params + 0] = (float *) q[b]->cudaData;
            cpuPointers[b * params + 1] = (float *) k[b]->cudaData;
            cpuPointers[b * params + 2] = (float *) v[b]->cudaData;
            cpuPointers[b * params + 3] = (mask[b] && mask[b]->dims.size() > 0) ? (float *) mask[b]->cudaData : nullptr;
            cpuPointers[b * params + 4] = (float *) output[b]->cudaData;
            cpuPointers[b * params + 5] = (float *) (unsigned long long) q[b]->dims[1];
            cpuPointers[b * params + 6] = (float *) (unsigned long long) q[b]->dims[2];
            cpuPointers[b * params + 7] = (float *) (unsigned long long) k[b]->dims[1];
            cpuPointers[b * params + 8] = (float *) (unsigned long long) v[b]->dims[2];
            cpuPointers[b * params + 9] = (float *) (unsigned long long) q[b]->strides[0];
            cpuPointers[b * params + 10] = (float *) (unsigned long long) k[b]->strides[0];
            cpuPointers[b * params + 11] = (float *) (unsigned long long) v[b]->strides[0];
            cpuPointers[b * params + 12] = (float *) (unsigned long long) output[b]->strides[0];
            cpuPointers[b * params + 13] = (float *) (unsigned long long) qk[b];
            cpuPointers[b * params + 14] = (float *) (unsigned long long) temp[b];
            cpuPointers[b * params + 15] = (float *) (unsigned long long) q[b]->dims[0];
        }

        cudaMemcpy(pointers, cpuPointers, sizeof(float *) * batch * params, cudaMemcpyHostToDevice);
        FastllmAttentionBatchKernel<256> <<< batch, 256 >>>(pointers, scale, group);

        for (int i = 0; i < batch; i++) {
            FastllmCudaFree(qk[i]);
            FastllmCudaFree(temp[i]);
        }
        delete[] qk;
        delete[] temp;

        FastllmCudaFree(pointers);
        delete[] cpuPointers;
    }
*/
/*
    for (int b = 0; b < batch; b++) {
        int q0 = q[b]->dims[0], q1 = q[b]->dims[1], q2 = q[b]->dims[2], k0 = k[b]->dims[0], k1 = k[b]->dims[1], v2 = v[b]->dims[2];
        float *qd = (float *) q[b]->cudaData;
        float *kd = (float *) k[b]->cudaData;
        float *vd = (float *) v[b]->cudaData;
        float *maskd = (mask[b] && mask[b]->dims.size() > 0) ? (float *) mask[b]->cudaData : nullptr;
        float *od = (float *) output[b]->cudaData;
        int maskBatch = (mask[b] && mask[b]->dims.size() > 0) ? mask[b]->dims[0] : 1;

        float *qk = (float *) FastllmCudaMalloc(q0 * k1 * sizeof(float));
        float *temp = (float *) FastllmCudaMalloc(q0 * k1 * sizeof(float));
        FastllmAttentionKernel<256> <<<q0, 256>>>(qd, kd, vd, maskd, od,
                                                  scale, q1, q2, k1, v2,
                                                  group, q[b]->strides[0], k[b]->strides[0], v[b]->strides[0],
                                                  output[b]->strides[0],
                                                  qk, temp);
    }
*/
    DeviceSync();
    return true;
}

bool FastllmCudaSplitBatch(fastllm::Data &input, fastllm::Data **outputs, int axis) {
    int part = input.dims[axis];
    int outer = input.Count(0) / input.Count(axis);
    int inputStride = input.Count(axis);
    int outputStride = outputs[0]->Count(axis);
    int inner = input.strides[axis];
    int unitSize = input.unitSize;

    uint8_t ** pointers = (uint8_t**)FastllmCudaMalloc(sizeof(uint8_t*) * part);
    uint8_t ** cpuPointers = new uint8_t*[part];
    for (int i = 0; i < part; i++) {
        cpuPointers[i] = (uint8_t*)outputs[i]->cudaData;
    }
    cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * part, cudaMemcpyHostToDevice);
    FastllmSplitBatchKernel <256> <<< part * outer, 256 >>> ((uint8_t*)input.cudaData, pointers, outer, part, inner * unitSize);

    FastllmCudaFree(pointers);
    delete[] cpuPointers;

    DeviceSync();
    return true;
}

bool FastllmCudaCatBatch(fastllm::Data **inputs, fastllm::Data &output, int axis) {
    int part = output.dims[axis];
    int outer = output.Count(0) / output.Count(axis);
    int inputStride = inputs[0]->Count(axis);
    int outputStride = output.Count(axis);
    int inner = output.strides[axis];
    int unitSize = output.unitSize;

    uint8_t ** pointers = (uint8_t**)FastllmCudaMalloc(sizeof(uint8_t*) * part);
    uint8_t ** cpuPointers = new uint8_t*[part];
    for (int i = 0; i < part; i++) {
        cpuPointers[i] = (uint8_t*)inputs[i]->cudaData;
    }
    cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * part, cudaMemcpyHostToDevice);
    FastllmCatBatchKernel <256> <<< part * outer, 256 >>> (pointers, (uint8_t*)output.cudaData, outer, part, inner * unitSize);

    FastllmCudaFree(pointers);
    delete[] cpuPointers;

    DeviceSync();
    return true;
}

bool FastllmCudaMulBatch(fastllm::Data **inputs, float v, int batch, fastllm::Data **outputs) {
    float ** pointers = (float**)FastllmCudaMalloc(sizeof(float*) * batch * 3);
    float ** cpuPointers = new float*[batch * 3];
    for (int i = 0; i < batch; i++) {
        cpuPointers[i] = (float*)inputs[i]->cudaData;
        cpuPointers[i + batch] = (float*)outputs[i]->cudaData;
        cpuPointers[i + batch * 2] = (float*)(inputs[i]->Count(0));
    }
    cudaMemcpy(pointers, cpuPointers, sizeof(float*) * batch * 3, cudaMemcpyHostToDevice);
    FastllmMulBatchKernel <256> <<< batch, 256 >>> (pointers, batch, v);

    FastllmCudaFree(pointers);
    delete[] cpuPointers;

    DeviceSync();
    return true;
}

bool FastllmCudaBatchMatMulTransBBatch(void **i0s, void **i1s, void **os,
                                       int *ns, int *ms, int *ks,
                                       int *i0Strides, int *i1Strides, float alpha, int batch) {
    uint8_t ** pointers = (uint8_t**)FastllmCudaMalloc(sizeof(uint8_t*) * batch * 8);
    uint8_t ** cpuPointers = new uint8_t*[batch * 8];
    for (int i = 0; i < batch; i++) {
        cpuPointers[i * 8 + 0] = (uint8_t *) i0s[i];
        cpuPointers[i * 8 + 1] = (uint8_t *) i1s[i];
        cpuPointers[i * 8 + 2] = (uint8_t *) os[i];
        cpuPointers[i * 8 + 3] = (uint8_t *) (size_t) ns[i];
        cpuPointers[i * 8 + 4] = (uint8_t *) (size_t) ms[i];
        cpuPointers[i * 8 + 5] = (uint8_t *) (size_t) ks[i];
        cpuPointers[i * 8 + 6] = (uint8_t *) (size_t) i0Strides[i];
        cpuPointers[i * 8 + 7] = (uint8_t *) (size_t) i1Strides[i];
    }
    cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * batch * 8, cudaMemcpyHostToDevice);
    FastllmMatMulTransBBatchKernel <128> <<<batch, 128>>> (pointers, alpha);
    FastllmCudaFree(pointers);
    delete[] cpuPointers;
    DeviceSync();
    return true;
}

bool FastllmCudaBatchMatMulBatch(void **i0s, void **i1s, void **os,
                                 int *ns, int *ms, int *ks,
                                 int *i0Strides, int *i1Strides, float alpha, int batch) {
    uint8_t ** pointers = (uint8_t**)FastllmCudaMalloc(sizeof(uint8_t*) * batch * 8);
    uint8_t ** cpuPointers = new uint8_t*[batch * 8];
    for (int i = 0; i < batch; i++) {
        cpuPointers[i * 8 + 0] = (uint8_t *) i0s[i];
        cpuPointers[i * 8 + 1] = (uint8_t *) i1s[i];
        cpuPointers[i * 8 + 2] = (uint8_t *) os[i];
        cpuPointers[i * 8 + 3] = (uint8_t *) (size_t) ns[i];
        cpuPointers[i * 8 + 4] = (uint8_t *) (size_t) ms[i];
        cpuPointers[i * 8 + 5] = (uint8_t *) (size_t) ks[i];
        cpuPointers[i * 8 + 6] = (uint8_t *) (size_t) i0Strides[i];
        cpuPointers[i * 8 + 7] = (uint8_t *) (size_t) i1Strides[i];
    }
    cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * batch * 8, cudaMemcpyHostToDevice);
    FastllmMatMulKernel <128> <<<batch, 128>>> (pointers, alpha);
    FastllmCudaFree(pointers);
    delete[] cpuPointers;
    DeviceSync();
    return true;
}

void FastllmCudaSetDevice(int gpu_id) {
    cudaSetDevice(gpu_id);
}
