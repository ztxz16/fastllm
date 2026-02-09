#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/copy.h>
#include <thrust/functional.h>

#include "fastllm-cuda.cuh"
#include "fastllm.h"

void showError(cudaError_t result, char const* const message, const char* const file,
           int const line) {
    if (cudaSuccess != result) {
        printf("%s\n  CUDA error = %d, %s at %s:%d\n  '%s'\n",
            message, result, cudaGetErrorName(result), file, line, cudaGetErrorString(result));
    }  
}

/*
size_t totalMalloc = 0;
std::map <void*, size_t> mallocMap;
std::map <size_t, int> mallocCnt;

template <typename T>
cudaError_t CCMalloc(T **ret, size_t size) {
printf("malloc %f m\n", (double)size / 1e6);
totalMalloc += size;
printf("total malloc %f m\n", (double)totalMalloc / 1e6);
    cudaError_t sta = cudaMalloc(ret, size);
mallocMap[ret[0]] = size;
mallocCnt[size]++;
    return sta;
}

template <typename T>
cudaError_t CCFree(T *ret) {
printf("free %f m\n", (double)mallocMap[ret] / 1e6);
totalMalloc -= mallocMap[ret];
mallocCnt[mallocMap[ret]]--;
printf("total malloc %f m\n", (double)totalMalloc / 1e6);
for (auto &it : mallocCnt) {
    if (it.second > 0) printf("(%f: %d) ", (double)it.first / 1e6, it.second);
}
printf("\n");
    cudaError_t sta = cudaFree(ret);
    return sta;
}
*/

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

std::vector <long long> FastllmCudaGetFreeSizes() {
    int deviceCount;
    auto error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error, cudaGetErrorString(error));
        return {};
    }
    std::vector <long long> ret;
    
    // 遍历所有设备  
    int id = -1;
    cudaGetDevice(&id);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        error = cudaGetDeviceProperties(&prop, i);
        if (error == cudaSuccess) {
            // printf("Device %d: \"%s\"\n", i, prop.name);
            // printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
            // printf("  Total global memory: %zu bytes\n", prop.totalGlobalMem);
            
            // 获取当前设备的显存使用情况  
            cudaSetDevice(i);
            size_t free = 0, total = 0;
            cudaMemGetInfo(&free, &total);
            ret.push_back(free);
            // printf("  Free memory: %zu bytes\n", free);
            // printf("  Remaining memory: %zu bytes\n", total - free);
        } else {
            printf("cudaGetDeviceProperties returned %d\n-> %s\n", (int)error, cudaGetErrorString(error));
        }
    }
    cudaSetDevice(id);
    return ret;
}

__global__ void GetCudaInfoKernel(int *infos) {
#if defined(__CUDA_ARCH__)
    infos[0] = __CUDA_ARCH__;
#else
    infos[0] = 0; // cuda arch
#endif
}

CudaInfos::CudaInfos() {
    int infoLen = 10;
    int *infos;
    cudaMalloc(&infos, infoLen * sizeof(int));
    GetCudaInfoKernel <<<1, 1>>> (infos);
    int *infosInCpu = new int[infoLen];
    cudaMemcpy(infosInCpu, infos, infoLen * sizeof(int), cudaMemcpyDeviceToHost);

    cudaArch = infosInCpu[0];
    hasTensorCore = cudaArch >= 700;

    cudaFree(infos);
    delete[] infosInCpu;

    printf("CUDA_ARCH: %d\n", cudaArch);
    printf("USE_TENSOR_CORE: %d\n", hasTensorCore);
}

CudaInfos *cudaInfos = nullptr;

CudaInfos *getCudaInfos() {
    if (cudaInfos == nullptr) {
        cudaInfos = new CudaInfos();
    }
    return cudaInfos;
}

void DeviceSync() {
    // cudaDeviceSynchronize();
}

void ForceDeviceSync() {
    cudaDeviceSynchronize();
}

double GetSpan(std::chrono::system_clock::time_point time1, std::chrono::system_clock::time_point time2) {
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds> (time2 - time1);
    return double(duration.count()) * std::chrono::nanoseconds::period::num / std::chrono::nanoseconds::period::den;
};

template <int THREAD_PER_BLOCK, typename T>
__global__ void FastllmCudaFloatEmbeddingKernel(float *input, T *weight, T *output, int embSize) {
    input += blockIdx.x;
    output += blockIdx.x * embSize;
    int token = (int)(input[0] + 1e-5);
    weight += token * embSize;
    for (int i = threadIdx.x; i < embSize; i+= THREAD_PER_BLOCK) {
        output[i] = weight[i];
    }
}

__global__ void FastllmCudaFloat2HalfKernel(float* a, half *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        b[idx] = __float2half_rz(a[idx]);
    }
}

__global__ void FastllmCudaHalf2FloatKernel(half* a, float *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        b[idx] = __half2float(a[idx]);
    }
}

__global__ void FastllmCudaBF162FloatKernel(uint16_t* a, float *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        ((uint32_t*)b)[idx] = a[idx] << 16;
    }
}

__global__ void FastllmCudaFloat2Bf16Kernel(float* a, __nv_bfloat16* b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        b[idx] = __float2bfloat16_rn(a[idx]);
    }
}

__global__ void FastllmCudaBiasKernel(__nv_bfloat16* a, __nv_bfloat16* bias, int k) {
    __nv_bfloat16* now = a + blockIdx.x * k;
    int stride = blockDim.x;
    for (int i = threadIdx.x; i < k; i += stride) {
        now[i] = __float2bfloat16_rn(__bfloat162float(now[i]) + __bfloat162float(bias[i]));
    }
}

__global__ void FastllmCudaBiasKernel(float *a, float *bias, int k) {
    float *now = a + blockIdx.x * k;
    int stride = blockDim.x;
    for (int i = threadIdx.x; i < k; i += stride) {
        now[i] += bias[i];
    }
}

__global__ void FastllmCudaBiasKernel(half *a, half *bias, int k) {
    half *now = a + blockIdx.x * k;
    int stride = blockDim.x;
    for (int i = threadIdx.x; i < k; i += stride) {
#ifdef CUDA_NO_TENSOR_CORE
        now[i] = __float2half(__half2float(now[i]) + __half2float(bias[i]));
#else
        now[i] = __hadd(now[i], bias[i]);
#endif
    }
}

__global__ void FastllmReluKernel(float* a, float *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        float x = a[idx];
        b[idx] = x > 0 ? x : 0;
    }
}

__global__ void FastllmExpKernel(float* a, float *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        float x = a[idx];
        b[idx] = exp((double)x);
    }
}

__global__ void FastllmExpKernel(half* a, half *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        float x = __half2float(a[idx]);
        b[idx] = __float2half(exp((double)x));
    }
}

__global__ void FastllmGeluKernel(float* a, float *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        float x = a[idx];
        b[idx] = x * 0.5f * (1.0f + erff(x / 1.41421));
    }
}

__global__ void FastllmGeluKernel(half* a, half *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        float x = __half2float(a[idx]);
        b[idx] = __float2half(x * 0.5f * (1.0f + erff(x / 1.41421)));
    }
}

__global__ void FastllmGeluNewKernel(float* a, float *b, int len) {
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

__global__ void FastllmSiluKernel(half* a, half *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
#ifdef CUDA_NO_TENSOR_CORE
        float x = __half2float(a[idx]);
        b[idx] = __float2half((x / (1.0 + expf(-x))));
#else
        half x = a[idx];
        b[idx] = __hdiv(x, __hadd(__float2half(1.0), hexp(-x)));
#endif
    }
}

__global__ void FastllmSigmoidKernel(float* a, float *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        float x = a[idx];
        b[idx] = 1.0 / (1.0 + expf(-x));
    }
}

__global__ void FastllmSigmoidKernel(half* a, half *b, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
#ifdef CUDA_NO_TENSOR_CORE
        float x = __half2float(a[idx]);
        b[idx] = __float2half(1.0 / (1.0 + expf(-x)));
#else
        half x = a[idx];
        b[idx] = __hdiv(1.0, __hadd(__float2half(1.0), hexp(-x)));
#endif
    }
}

__device__ float softplus(float x) {
    return  x > 20.0f ? x : log1p(expf(x));
}

__global__ void FastllmMambaSoftplusKernel(float* inputData, float *outputData, float *aLog, float *dtBias, int channels) {
    int o = blockIdx.x;
    for (int i = threadIdx.x; i < channels; i += blockDim.x) {
        outputData[o * channels + i] = -expf((double)aLog[i]) * softplus(inputData[o * channels + i] + dtBias[i]);
    }
}

__global__ void FastllmMambaSoftplusKernel(half* inputData, half *outputData, float *aLog, float *dtBias, int channels) {
    int o = blockIdx.x;
    for (int i = threadIdx.x; i < channels; i += blockDim.x) {
        outputData[o * channels + i] = __float2half(-exp((double)aLog[i]) * softplus(__half2float(inputData[o * channels + i]) + dtBias[i]));
    }
}

__global__ void FastllmSwigluKernel(float* __restrict__ a, float* __restrict__ b, int len, int spatial, int mid) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        int id = idx / mid * spatial + idx % mid;
        float x = a[id], y = a[id + mid];
        b[idx] = (x / (1.0f + expf(-x))) * y;
    }
}

__global__ void FastllmSwigluKernel(half* __restrict__ a, half* __restrict__ b, int len, int spatial, int mid) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        int id = idx / mid * spatial + idx % mid;
#ifdef CUDA_NO_TENSOR_CORE
        float x = __half2float(a[id]), y = __half2float(a[id + mid]);
        b[idx] = __float2half((x / (1.0 + expf(-x))) * y);
#else
        half x = a[id], y = a[id + mid];
        b[idx] = __hmul(__hdiv(x, __hadd(__float2half(1.0), hexp(-x))), y);
#endif
    }
}

__global__ void FastllmAddKernel(float* a, float *b, float v, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        b[idx] = a[idx] + v;
    }
}

__global__ void FastllmAddKernel(half* a, half *b, half v, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
#ifdef CUDA_NO_TENSOR_CORE
        b[idx] = __float2half(__half2float(a[idx]) + __half2float(v));
#else
        b[idx] = __hadd(a[idx], v);
#endif
    }
}

__global__ void FastllmMulKernel(float* a, float *b, float v, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        b[idx] = a[idx] * v;
    }
}

__global__ void FastllmMulKernel(half* a, half *b, half v, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
#ifdef CUDA_NO_TENSOR_CORE
        b[idx] = __float2half(__half2float(a[idx]) * __half2float(v));
#else
        b[idx] = __hmul(a[idx], v);
#endif
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


__global__ void FastllmReduceKernel(float *output, float* input, int len, int threadNum) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        output[idx] = 0;
        for (int i = 0; i < threadNum; i++) {
            output[idx] += input[idx + i * len];
        }
    }
}

__global__ void FastllmReduceKernel(half *output, half* input, int len, int threadNum) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        output[idx] = (half)0;
        for (int i = 0; i < threadNum; i++) {
            output[idx] = __hadd(output[idx], input[idx + i * len]);
        }
    }
}

__global__ void FastllmAddToKernel(float* a, float *b, float alpha, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        a[idx] += b[idx] * alpha;
    }
}

__global__ void FastllmAddToKernel(half* a, half *b, half alpha, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
#ifdef CUDA_NO_TENSOR_CORE
        a[idx] = __float2half(__half2float(a[idx]) + __half2float(b[idx]) * __half2float(alpha));
#else
        a[idx] = __hadd(a[idx], __hmul(b[idx], alpha));
#endif
    }
}

__global__ void FastllmMulToKernel(float* a, float *b, float alpha, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        a[idx] *= b[idx] * alpha;
    }
}

__global__ void FastllmMulToKernel(half* a, half *b, float alpha, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
#ifdef CUDA_NO_TENSOR_CORE
        a[idx] = __float2half(__half2float(b[idx]) * alpha * __half2float(a[idx]));
#else
        a[idx] *= (half)((float)b[idx] * alpha);
#endif
    }
}

__global__ void FastllmMulSingleToKernel(float* a, float *b, float alpha, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        a[idx] *= b[0] * alpha;
    }
}

__global__ void FastllmMulSingleToKernel(half* a, half *b, float alpha, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
#ifdef CUDA_NO_TENSOR_CORE
        a[idx] = __float2half(__half2float(b[0]) * alpha * __half2float(a[idx]));
#else
        a[idx] *= (half)((float)b[0] * alpha);
#endif
    }
}

__global__ void FastllmChannelMulToKernel(float* a, float *b, float alpha, int len, int channelLen) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        a[idx] *= b[idx / channelLen] * alpha;
    }
}

__global__ void FastllmChannelMulToKernel(half* a, half *b, float alpha, int len, int channelLen) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
#ifdef CUDA_NO_TENSOR_CORE
        a[idx] = __float2half(__half2float(b[idx / channelLen]) * alpha * __half2float(a[idx]));
#else
        a[idx] *= (half)((float)b[idx / channelLen] * alpha);
#endif
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
__global__ void FastllmRepeatPenaltyKernel(float* input, float *penalty, float *penaltyScaleData, int tokens, int vocabs) {
    unsigned int bid = blockIdx.x;
    input += bid * vocabs;
    penalty += bid * tokens;
    float scale = penaltyScaleData[bid];
    for (int i = threadIdx.x; i < tokens; i += THREAD_PER_BLOCK) {
        int token = (int)(penalty[i] + 1e-6);
        if (token >= 0) {
            input[token] = input[token] < 0 ? input[token] * scale : input[token] / scale;
        }
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

template <typename T>
__global__ void FastllmPermuteKernel(T *dst, T *ori, int *temp, int axisLen, int len) {
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

__global__ void FastllmLlamaRotatePosition2DKernel(half *data, float *positionIds, float *sin, float *cos,
                                                   int len, int bs, int spatial, int n, int m, int partStride, int sinCosStride, int rotateDim) {
    int o = (blockIdx.x / n);
    int l = o % len;
    int b = o / len;
    int j = threadIdx.x;
    int index = (int) (positionIds[b * partStride + l]);

    float curSin = sin[index * sinCosStride + j];
    float curCos = cos[index * sinCosStride + j];
    half *d = (half *) data + o * spatial + j;
    int i = blockIdx.x % n;
    float va = __half2float(d[i * m]), vb = __half2float(d[i * m + m / 2]);
    d[i * m] = __float2half(va * curCos - vb * curSin);
    d[i * m + m / 2] = __float2half(va * curSin + vb * curCos);
}

__global__ void FastllmLlamaRotatePosition2DPartKernel(float *data, float *positionIds, float *sin, float *cos,
                                                   int len, int bs, int spatial, int n, int m, int partStride, int sinCosStride, int part) {
    int o = (blockIdx.x / n);
    int l = o % len;
    int b = o / len;
    int j = threadIdx.x;
    int index = (int) (positionIds[b * partStride + l]);

    float curSin = sin[index * sinCosStride + j];
    float curCos = cos[index * sinCosStride + j];
    float *d = (float *) data + o * spatial + j;
    int i = blockIdx.x % n;
    float va = d[i * m], vb = d[i * m + part / 2];
    d[i * m] = va * curCos - vb * curSin;
    d[i * m + part / 2] = va * curSin + vb * curCos;
}

__global__ void FastllmLlamaRotatePosition2DPartKernel(half *data, float *positionIds, float *sin, float *cos,
                                                   int len, int bs, int spatial, int n, int m, int partStride, int sinCosStride, int part) {
    int o = (blockIdx.x / n);
    int l = o % len;
    int b = o / len;
    int j = threadIdx.x;
    int index = (int) (positionIds[b * partStride + l]);

    float curSin = sin[index * sinCosStride + j];
    float curCos = cos[index * sinCosStride + j];
    half *d = (half *) data + o * spatial + j;
    int i = blockIdx.x % n;
    float va = __half2float(d[i * m]), vb = __half2float(d[i * m + part / 2]);
    d[i * m] = __float2half(va * curCos - vb * curSin);
    d[i * m + part / 2] = __float2half(va * curSin + vb * curCos);
}


__global__ void FastllmNearlyRotatePosition2DKernel(float *data, float *positionIds, float *sin, float *cos,
                                                    int len, int bs, int spatial, int n, int m, int partStride, int sinCosStride, int rotateDim) {
    int o = (blockIdx.x / n);
    int l = o / bs;
    int b = o % bs;
    int j = threadIdx.x;
    int index = (int) (positionIds[b * partStride + l]);

    float curSin = sin[index * sinCosStride + j];
    float curCos = cos[index * sinCosStride + j];
    float *d = (float *) data + o * spatial + j * 2;
    int i = blockIdx.x % n;
    float va = d[i * m], vb = d[i * m + 1];
    d[i * m] = va * curCos - vb * curSin;
    d[i * m + 1] = va * curSin + vb * curCos;
}

__global__ void FastllmNearlyRotatePosition2DKernel(half *data, float *positionIds, float *sin, float *cos,
                                                    int len, int bs, int spatial, int n, int m, int partStride, int sinCosStride, int rotateDim) {
    int o = (blockIdx.x / n);
    int l = o / bs;
    int b = o % bs;
    int j = threadIdx.x;
    int index = (int) (positionIds[b * partStride + l]);

    float curSin = sin[index * sinCosStride + j];
    float curCos = cos[index * sinCosStride + j];
    half *d = (half *) data + o * spatial + j * 2;
    int i = blockIdx.x % n;
    float va = __half2float(d[i * m]), vb = __half2float(d[i * m + 1]);
    d[i * m] = __float2half(va * curCos - vb * curSin);
    d[i * m + 1] = __float2half(va * curSin + vb * curCos);
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
__global__ void FastllmRMSNormKernelInner1(half *input, float *weight, half *output, int outer, int channels, float eps) {
    int o = blockIdx.x;
    input = input + o * channels;
    output = output + o * channels;

    __shared__ float sdata2[THREAD_PER_BLOCK];
    __shared__ float scale;

    // 1. 每个线程计算一部分
    unsigned int tid = threadIdx.x;
    float sum2 = 0.0;
    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        float x = __half2float(input[i]);
        sum2 += x * x;
    }
    sdata2[tid] = sum2;
    __syncthreads();

    // 2. 求和
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata2[tid] += sdata2[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile float *now = sdata2;
        now[tid] += now[tid + 32];
        now[tid] += now[tid + 16];
        now[tid] += now[tid + 8];
        now[tid] += now[tid + 4];
        now[tid] += now[tid + 2];
        now[tid] += now[tid + 1];
    }
    __syncthreads();

    // 3. 计算参数
    if (tid == 0) {
        scale = 1.0 / sqrt(sdata2[0] / channels + eps);
    }
    __syncthreads();

    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        output[i] = __float2half(__half2float(input[i]) * scale * weight[i]);
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
__global__ void FastllmLayerNormKernelInner1(half *input, float *gamma, float *beta, half *output, int outer, int channels) {
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
        float x = __half2float(input[i]);
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
        output[i] = __float2half((__half2float(input[i]) - mean) / var * gamma[i] + beta[i]);
    }
}


template <int THREAD_PER_BLOCK>
__global__ void FastllmLayerNormKernelTop1(float *input, float *output, int channels) {
    __shared__ float idData[THREAD_PER_BLOCK];
    __shared__ float maxData[THREAD_PER_BLOCK];
    float *inputData = input + blockIdx.x * channels;
    float *outputData = output + blockIdx.x * 2;
    int tid = threadIdx.x;
    idData[tid] = tid;
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

template <int THREAD_PER_BLOCK, int MAXK>
__global__ void FastllmLayerNormKernelTopK(float *input, float *output, int K, int channels) {
    __shared__ float idData[THREAD_PER_BLOCK][MAXK];
    __shared__ float maxData[THREAD_PER_BLOCK][MAXK];
    float *inputData = input + blockIdx.x * channels;
    float *outputData = output + blockIdx.x * 2 * K;
    int tid = threadIdx.x;
    idData[tid][0] = tid;
    for (int i = 0; i < K; i++) {
        maxData[tid][i] = -1e100;
    }
    for (int j = tid; j < channels; j += THREAD_PER_BLOCK) {
        float cur = inputData[j];
        for (int l = 0; l < K; l++) {
            if (cur > maxData[tid][l]) {
                for (int x = K - 1; x > l; x--) {
                    maxData[tid][x] = maxData[tid][x - 1];
                    idData[tid][x] = idData[tid][x - 1];
                }
                maxData[tid][l] = cur;
                idData[tid][l] = j;
                break;
            }
        }
    }
    __syncthreads();

    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            int pos0 = 0, pos1 = 0;
            while (pos0 + pos1 < K) {
                if (maxData[tid][pos0] > maxData[tid + s][pos1]) {
                    pos0++;
                } else {
                    pos1++;
                }
            }
            pos0--;
            pos1--;
            int pos = K - 1;
            while (pos >= 0) {
                if (pos1 < 0 || (pos0 >= 0 && maxData[tid][pos0] < maxData[tid + s][pos1])) {
                    maxData[tid][pos] = maxData[tid][pos0];
                    idData[tid][pos] = idData[tid][pos0];
                    pos0--;
                } else {
                    maxData[tid][pos] = maxData[tid + s][pos1];
                    idData[tid][pos] = idData[tid + s][pos1];
                    pos1--;
                }
                pos--;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        for (int i = 0; i < K; i++) {
            outputData[i * 2] = idData[0][i];
            outputData[i * 2 + 1] = maxData[0][i];
        }
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

void *FastllmCudaPrepareInput(const fastllm::Data &input) {
    void *ret;
    if (input.dataDevice == fastllm::DataDevice::CUDA) {
        ret = (void*)input.cudaData;
    } else {
        ret = (void*)(input.expansionBytes);
        auto state = cudaMemcpy(ret, input.cpuData, input.expansionBytes, cudaMemcpyHostToDevice);
        if (cudaSuccess != state) {
            checkCudaErrors("Error: CUDA error when copy from memory to GPU!", state);
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
        checkCudaErrors("Error: CUDA error when copy from GPU to memory!", state);
        FastllmCudaFree(data);
    }

    DeviceSync();
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
std::map<int, int> cudaBuffersMinId; // 最小的空闲id
std::map<int, size_t> noBusyCnt;
std::map<int, std::vector <CudaMemoryBuffer>> bigBuffersMap;

void * FastllmCudaDirectMalloc(size_t size) {
    void * ret;
    cudaError_t state = cudaMalloc(&ret, size);
    if (cudaSuccess != state) {
        printf("Error: CUDA error when allocating %lu kB memory! maybe there's no enough memory left on device.", size >> 10);
        checkCudaErrors("", state);
        return nullptr;
    }
    return ret;
}

void FastllmCudaDirectFree(void *ret) {
    cudaError_t state = cudaFree(ret);
    //checkCudaErrors("Error: CUDA error when release memory!", state);
}

void FastllmCudaMemset0(void *ret, size_t size) {
    cudaMemset(ret, 0, size);
}

void * FastllmCudaMalloc(size_t size) {
    int id = -1;
    cudaError_t state = cudaSuccess;
    state = cudaGetDevice(&id);
    checkCudaErrors("Error: CUDA error when find device!", state);
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
            printf("Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory left on device.", size >> 20);
            checkCudaErrors("", state);
            return nullptr;
        }
        bigBuffers.push_back(CudaMemoryBuffer(ret, size, true));
        return ret;
    }
    auto &cudaBuffers = cudaBuffersMap[id];
    for (int i = cudaBuffersMinId[id]; i < cudaBuffers.size(); i++) {
        if (cudaBuffers[i].size >= size && !cudaBuffers[i].busy) {
            cudaBuffers[i].busy = true;
            noBusyCnt[id] -= cudaBuffers[i].size;
            while (cudaBuffersMinId[id] < cudaBuffers.size() && cudaBuffers[cudaBuffersMinId[id]].busy) {
                cudaBuffersMinId[id]++;
            }
            return cudaBuffers[i].data;
        }
    }
    void * ret;
    state = cudaMalloc(&ret, size);
    if (cudaSuccess != state) {
        printf("Error: CUDA error when allocating %lu KB memory! maybe there's no enough memory left on device.", size >> 10);
        checkCudaErrors("", state);
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
    int oriId = FastllmCudaGetDevice();
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
                        printf("Error: CUDA error when release memory on device %d!", it.first);
                    checkCudaErrors("", state);
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
                cudaBuffersMinId[it.first] = std::min(cudaBuffersMinId[it.first], i);
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
    FastllmCudaSetDevice(oriId);
    checkCudaErrors("CUDA error when release memory!", state);
}

void FastllmCudaMallocBigBuffer(size_t size) {
    void * ret;
    int id = -1;
    cudaGetDevice(&id);
    auto &bigBuffers = bigBuffersMap[id];
    cudaMalloc(&ret, size);
    auto state = cudaMalloc(&ret, size);
    if (cudaSuccess != state)
        printf("Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory left on device.", size >> 20);
    checkCudaErrors("", state);
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
        long long littleMemSum = 0;        
        long long littleMemSumLimit = 300 * 1024 * 1024; // 留一小部分复用  
        std::vector <std::pair <std::size_t, int > > v;
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (!bigBuffers[i].busy) {
                v.push_back(std::make_pair(bigBuffers[i].size, i));
            }
        }
        std::sort(v.begin(), v.end());
        std::set <int> littleMemIds;
        for (int i = 0; i < v.size(); i++) {
            littleMemSum += v[i].first;
            if (littleMemSum > littleMemSumLimit) {
                break;
            }
            littleMemIds.insert(v[i].second);
        }
        for (int i = 0; i < bigBuffers.size(); i++) {
            if (!bigBuffers[i].busy && littleMemIds.find(i) == littleMemIds.end()) {
                state = cudaSetDevice(it.first);
                state = cudaFree(bigBuffers[i].data);
                if (cudaSuccess != state)
                    printf("Error: CUDA error when release memory on device %d!", it.first);
                checkCudaErrors("", state);
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
    cudaError_t state = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    checkCudaErrors("Error: CUDA error when copy from memory to GPU!", state);
    //cudaDeviceSynchronize();
}

void FastllmCudaCopyFromDeviceToHost(void *dst, void *src, size_t size) {
    cudaError_t state = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    checkCudaErrors("Error: CUDA error when copy from GPU to memory!", state);
    //cudaDeviceSynchronize();
}

void FastllmCudaCopyFromDeviceToDevice(void *dst, void *src, size_t size) {
    cudaError_t state = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    checkCudaErrors("Error: CUDA error when copy on GPU!", state);
    //cudaDeviceSynchronize();
}

void FastllmCudaMemcpyBetweenDevices(int dstId, void *dst, int srcId, void *src, size_t size) {
    int canPeerAccess = 0;
    cudaError_t state = cudaDeviceCanAccessPeer(&canPeerAccess, srcId, dstId);
    if (canPeerAccess) {
        state = cudaMemcpyPeer(dst, dstId, src, srcId, size);
    } else {
        uint8_t *cpuData = new uint8_t[size];
        state = cudaSetDevice(srcId);
        state = cudaMemcpy(cpuData, src, size, cudaMemcpyDeviceToHost);

        state = cudaSetDevice(dstId);
        state = cudaMemcpy(dst, cpuData, size, cudaMemcpyHostToDevice);
        delete[] cpuData;
    }
    checkCudaErrors("Error: CUDA error when copy Between GPUs!", state);
    DeviceSync();
}

void FastllmCudaMemcpy2DDeviceToDevice(void * 	dst, size_t 	dpitch, const void * 	src,
                                       size_t 	spitch, size_t 	width, size_t 	height) {
    cudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice);
    DeviceSync();
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmMemcpy2DKernel (uint8_t * 	dst, size_t 	dpitch, uint8_t * 	src,
                                       size_t 	spitch, size_t 	width, size_t 	height) {
    int id = blockIdx.x;
    dst += id * dpitch;
    src += id * spitch;
    for (int i = threadIdx.x; i < width; i += THREAD_PER_BLOCK) {
        dst[i] = src[i];
    }
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

template <int THREAD_PER_BLOCK>
__global__ void FastllmRepeatKernel (void *inputOri, void *outputOri, int outer, int repeatTimes, int inputStride, int outputStride0, int outputStride1, int copyLen) {
    int id = blockIdx.x;
    int i = id / repeatTimes, j = id % repeatTimes;
    uint8_t *output = (uint8_t*)outputOri + i * outputStride0 + j * outputStride1;
    uint8_t *input = (uint8_t*)inputOri + i * inputStride;
    for (int x = threadIdx.x; x < copyLen; x += THREAD_PER_BLOCK) {
        output[x] = input[x];
    }
}

void FastllmCudaRepeat(void *input, void *output, int outer, int repeatTimes, int inputStride, int outputStride0, int outputStride1, int copyLen) {
    FastllmRepeatKernel <256> <<< outer * repeatTimes, 256 >>> (input, output, outer, repeatTimes, inputStride, outputStride0, outputStride1, copyLen);
    DeviceSync();
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
    FastllmMemcpyBatchKernel <256> <<<total, 256>>> (pointers);

    FastllmCudaFree(pointers);
    delete[] cpuPointers;

    DeviceSync();
}

bool FastllmCudaExp(const fastllm::Data &input, fastllm::Data &output) {
    int len = input.Count(0);
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareOutput(output);
    int threadPerBlock = std::min(256, len);
    if (input.dataType == fastllm::DataType::FLOAT32) {
        FastllmExpKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaOutput, len);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        FastllmExpKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((half*)cudaInput, (half*)cudaOutput, len);
    } else {
        printf("Exp datatype error.\n");
        exit(0);
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaRelu(const fastllm::Data &input, fastllm::Data &output) {
    int len = input.Count(0);
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareOutput(output);
    int threadPerBlock = std::min(256, len);
    if (input.dataType == fastllm::DataType::FLOAT32) {
        FastllmReluKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaOutput, len);
    } else {
        printf("Relu datatype error.\n");
        exit(0);
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaGelu(const fastllm::Data &input, fastllm::Data &output) {
    int len = input.Count(0);
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareOutput(output);
    int threadPerBlock = std::min(256, len);
    if (input.dataType == fastllm::DataType::FLOAT16) {
        FastllmGeluKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((half*)cudaInput, (half*)cudaOutput, len);
    } else if (input.dataType == fastllm::DataType::FLOAT32) {
        FastllmGeluKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaOutput, len);
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaGeluNew(const fastllm::Data &input, fastllm::Data &output) {
    int len = input.Count(0);
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareOutput(output);
    int threadPerBlock = std::min(256, len);
    FastllmGeluNewKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaOutput, len);
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaSilu(const fastllm::Data &input, fastllm::Data &output) {
    int len = input.Count(0);
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareOutput(output);
    int threadPerBlock = std::min(1024, len);
    if (input.dataType == fastllm::DataType::FLOAT32) {
        FastllmSiluKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaOutput, len);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        FastllmSiluKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((half*)cudaInput, (half*)cudaOutput, len);
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaSigmoid(const fastllm::Data &input, fastllm::Data &output) {
    int len = input.Count(0);
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareOutput(output);
    int threadPerBlock = std::min(1024, len);
    if (input.dataType == fastllm::DataType::FLOAT32) {
        FastllmSigmoidKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaOutput, len);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        FastllmSigmoidKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((half*)cudaInput, (half*)cudaOutput, len);
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaMambaSoftplus(const fastllm::Data &input, fastllm::Data &output, fastllm::Data &aLogData, fastllm::Data &dtBiasData) {
    int dimsLen = input.dims.size();
    int outer = input.Count(0) / input.Count(dimsLen - 1);
    int channels = input.dims[dimsLen - 1];

    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareOutput(output);
    float *aLog = (float*) FastllmCudaPrepareInput(aLogData);
    float *dtBias = (float*) FastllmCudaPrepareInput(dtBiasData);

    int threadPerBlock = std::min(64, channels);
    if (input.dataType == fastllm::DataType::FLOAT32) {
        FastllmMambaSoftplusKernel <<< outer, threadPerBlock >>> (cudaInput, cudaOutput, aLog, dtBias, channels);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        FastllmMambaSoftplusKernel <<< outer, threadPerBlock >>> ((half*)cudaInput, (half*)cudaOutput, aLog, dtBias, channels);
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishInput(aLogData, aLog);
    FastllmCudaFinishInput(dtBiasData, dtBias);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaSwiglu(const fastllm::Data &input, fastllm::Data &output) {
    int len = output.Count(0);
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareOutput(output);
    int spatial = input.Count(input.dims.size() - 1), mid = spatial / 2;

    int threadPerBlock = std::min(1024, len);
    if (input.dataType == fastllm::DataType::FLOAT32) {
        FastllmSwigluKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaOutput, len, spatial, mid);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        FastllmSwigluKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((half*)cudaInput, (half*)cudaOutput, len, spatial, mid);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaAdd(const fastllm::Data &input, float v, fastllm::Data &output) {
    int len = input.Count(0);
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareOutput(output);
    int threadPerBlock = std::min(256, len);

    if (input.dataType == fastllm::DataType::FLOAT32) {
        FastllmAddKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaOutput, v, len);
    } else {
        FastllmAddKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((half*)cudaInput, (half*)cudaOutput, __float2half_rn(v), len);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaMul(const fastllm::Data &input, float v, fastllm::Data &output) {
    int len = input.Count(0);
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareOutput(output);
    int threadPerBlock = std::min(256, len);

    if (input.dataType == fastllm::DataType::FLOAT32) {
        FastllmMulKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaOutput, v, len);
    } else {
        FastllmMulKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((half*)cudaInput, (half*)cudaOutput, __float2half_rn(v), len);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaAddTo(fastllm::Data &input0, const fastllm::Data &input1, float alpha) {
    int len = input0.Count(0);
    float *cudaData = (float *) FastllmCudaPrepareInput(input0);
    float *input1Data = (float *) FastllmCudaPrepareInput(input1);

    int threadPerBlock = std::min(1024, len);
    if (input0.dataType == fastllm::DataType::FLOAT32) {
        FastllmAddToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaData, input1Data, alpha, len);
    } else if (input0.dataType == fastllm::DataType::FLOAT16) {
        FastllmAddToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((half*)cudaData, (half*)input1Data, __float2half_rn(alpha), len);
    }

    FastllmCudaFinishInput(input1, input1Data);
    FastllmCudaFinishOutput(input0, cudaData);
    return true;
}

bool FastllmCudaMulTo(fastllm::Data &input0, const fastllm::Data &input1, float alpha) {
    int len = input0.Count(0);
    float *cudaData = (float *) FastllmCudaPrepareInput(input0);
    float *input1Data = (float *) FastllmCudaPrepareInput(input1);

    int threadPerBlock = std::min(256, len);
    if (input1.Count(0) == 1) {
        if (input0.dataType == fastllm::DataType::FLOAT32) {
            FastllmMulSingleToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaData, input1Data, alpha, len);
        } else {
            FastllmMulSingleToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((half*)cudaData, (half*)input1Data, alpha, len);
        }
    } else if (input0.dims == input1.dims) {
        if (input0.dataType == fastllm::DataType::FLOAT32) {
            FastllmMulToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaData, input1Data, alpha, len);
        } else {
            FastllmMulToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((half*)cudaData, (half*)input1Data, alpha, len);
        }
    } else {
        int channelLen = input0.Count(0) / input1.Count(0);
        if (input0.dataType == fastllm::DataType::FLOAT32) {
            FastllmChannelMulToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaData, input1Data, alpha, len, channelLen);
        } else {
            FastllmChannelMulToKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((half*)cudaData, (half*)input1Data, alpha, len, channelLen);
        }
    }
    FastllmCudaFinishInput(input1, input1Data);
    FastllmCudaFinishOutput(input0, cudaData);
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

// CUDA kernel for the main transformation
__global__ void TransferAttnKernel(float *data, int n, int m, int outer, int row_idx) {
    int o = blockIdx.z;  // batch index
    if (o >= outer) return;
    
    int j = threadIdx.x + blockIdx.x * blockDim.x;  // column index
    if (j >= row_idx) return;  // 只处理前row_idx个元素
    
    float *batchData = data + o * n * m;
    
    // 保存原始的第row_idx行的值
    float original_val = batchData[row_idx * m + j];
    
    // 计算新值: original_val + sum(row[k] * matrix[k][j] for k in [0, row_idx))
    float sum = original_val;
    for (int k = 0; k < row_idx; k++) {
        sum += batchData[row_idx * m + k] * batchData[k * m + j];
    }
    
    // 更新值
    batchData[row_idx * m + j] = sum;
}

// 使用共享内存的优化版本
__global__ void TransferAttnKernelShared(float *data, int n, int m, int outer, int row_idx) {
    extern __shared__ float shared[];
    
    int o = blockIdx.z;
    if (o >= outer) return;
    
    int tid = threadIdx.x;
    int j = tid + blockIdx.x * blockDim.x;
    
    float *batchData = data + o * n * m;
    float *row_i = shared;  // 存储第row_idx行的前row_idx个元素
    
    // 协作加载第row_idx行的前row_idx个元素到共享内存
    for (int idx = tid; idx < row_idx; idx += blockDim.x) {
        row_i[idx] = batchData[row_idx * m + idx];
    }
    __syncthreads();
    
    if (j < row_idx) {
        // 使用共享内存中的原始值计算
        float sum = row_i[j];
        
        // 累加：注意这里row_i[k]是原始值，batchData[k * m + j]是可能已更新的值
        for (int k = 0; k < row_idx; k++) {
            sum += row_i[k] * batchData[k * m + j];
        }
        
        // 写回结果
        batchData[row_idx * m + j] = sum;
    }
}

// CUDA kernel for adding identity matrix
__global__ void AddIdentityKernel(float *data, int n, int m, int outer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * n;
    
    if (idx < total) {
        int o = idx / n;
        int i = idx % n;
        data[o * n * m + i * m + i] += 1.0f;
    }
}

bool FastllmCudaTransferAttn(fastllm::Data &input) {
    float *inputData = (float *) FastllmCudaPrepareInput(input);

    int dimsLen = input.dims.size();
    int n = input.dims[dimsLen - 2];
    int m = input.dims[dimsLen - 1]; 
    int outer = input.Count(0) / input.Count(dimsLen - 2);

    // 逐行处理，从第1行开始（第0行不需要处理）
    for (int i = 1; i < n; i++) {
        // 每行只需要处理前i个元素
        int elementsToProcess = i;
        int threadsPerBlock = min(256, elementsToProcess);
        int blocksPerGrid = (elementsToProcess + threadsPerBlock - 1) / threadsPerBlock;
        
        dim3 blocks(blocksPerGrid, 1, outer);
        dim3 threads(threadsPerBlock, 1, 1);
        
        // 使用共享内存版本
        int sharedMemSize = elementsToProcess * sizeof(float);
        TransferAttnKernelShared<<<blocks, threads, sharedMemSize>>>(
            inputData, n, m, outer, i);
        
        // 必须同步，因为下一行的计算依赖于当前行的结果
        cudaDeviceSynchronize();
    }

    // 添加单位矩阵
    int totalDiag = outer * n;
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalDiag + threadsPerBlock - 1) / threadsPerBlock;
    
    AddIdentityKernel<<<blocksPerGrid, threadsPerBlock>>>(inputData, n, m, outer);

    DeviceSync();
    FastllmCudaFinishOutput(input, inputData);
    return true;
}

// CUDA核函数模板，支持float和half类型
template<typename T>
__global__ void CumSumLastDimKernel(T* data, int dim, int outer) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < outer) {
        T* row = data + tid * dim;
        
        // 对每一行进行累积和
        for (int j = 1; j < dim; j++) {
            row[j] = (T)((float)row[j] + (float)row[j - 1]);
        }
    }
}

bool FastllmCudaCumSumLastDim(fastllm::Data &input) {
    void *inputData = FastllmCudaPrepareInput(input);
    
    int dim = input.dims.back();
    int outer = input.Count(0) / dim;
    
    // 配置CUDA执行参数
    int threadsPerBlock = 256;
    int blocksPerGrid = (outer + threadsPerBlock - 1) / threadsPerBlock;
    
    // 根据数据类型调用相应的核函数
    if (input.dataType == fastllm::DataType::FLOAT32) {
        CumSumLastDimKernel<<<blocksPerGrid, threadsPerBlock>>>(
            (float*)inputData, dim, outer);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        CumSumLastDimKernel<<<blocksPerGrid, threadsPerBlock>>>(
            (half*)inputData, dim, outer);
    }

    DeviceSync();
    FastllmCudaFinishOutput(input, inputData);
    
    return true; // 添加返回值
}

// CUDA核函数模板，支持float和half
template<typename T>
__global__ void CausalMaskKernel(T *data, int n, int m, int outer, int base, T maskValue) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outer * n * m;
    
    if (idx < total) {
        int o = idx / (n * m);
        int remainder = idx % (n * m);
        int i = remainder / m;
        int j = remainder % m;
        
        if (j >= i + base) {
            data[idx] = maskValue;
        }
    }
}

bool FastllmCudaCausalMask(fastllm::Data &input, int base, float maskValue) {
    void *inputData = FastllmCudaPrepareInput(input);
    int dimsLen = input.dims.size();
    int n = input.dims[dimsLen - 2], m = input.dims[dimsLen - 1], outer = input.Count(0) / input.Count(dimsLen - 2);
    
    int total = outer * n * m;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    
    // 根据数据类型调用相应的核函数
    if (input.dataType == fastllm::DataType::FLOAT32) {
        float *floatData = (float *)inputData;
        CausalMaskKernel<float><<<gridSize, blockSize>>>(floatData, n, m, outer, base, maskValue);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        __half *halfData = (__half *)inputData;
        __half halfMaskValue = __float2half(maskValue);
        CausalMaskKernel<__half><<<gridSize, blockSize>>>(halfData, n, m, outer, base, halfMaskValue);
    }
    
    // 等待核函数执行完成
    DeviceSync();
    
    FastllmCudaFinishOutput(input, inputData);
    return true;
}

// CUDA核函数定义
template<typename T>
__global__ void MakeDecayMaskKernel(const T* input, T* output, int dim, int outer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = outer * dim * dim;
    
    if (idx < total_elements) {
        int o = idx / (dim * dim);
        int remainder = idx % (dim * dim);
        int i = remainder / dim;
        int j = remainder % dim;
        
        if (j <= i) {
            // 计算 exp(input[o * dim + i] - input[o * dim + j])
            T val_i = input[o * dim + i];
            T val_j = input[o * dim + j];
            output[idx] = exp(val_i - val_j);
        } else {
            output[idx] = T(0);
        }
    }
}

// 特化版本处理half类型
template<>
__global__ void MakeDecayMaskKernel<half>(const half* input, half* output, int dim, int outer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = outer * dim * dim;
    
    if (idx < total_elements) {
        int o = idx / (dim * dim);
        int remainder = idx % (dim * dim);
        int i = remainder / dim;
        int j = remainder % dim;
        
        if (j <= i) {
            // 对于half类型，需要转换为float进行计算
            float val_i = __half2float(input[o * dim + i]);
            float val_j = __half2float(input[o * dim + j]);
            output[idx] = __float2half(expf(val_i - val_j));
        } else {
            output[idx] = __float2half(0.0f);
        }
    }
}

bool FastllmCudaMakeDecayMask(fastllm::Data &input, fastllm::Data &output) {
    void *inputData = FastllmCudaPrepareInput(input);
    void *outputData = FastllmCudaPrepareInput(output);

    int dim = input.dims.back();
    int outer = input.Count(0) / dim;
    int total_elements = outer * dim * dim;
    
    // 配置CUDA执行参数
    int blockSize = 256;
    int gridSize = (total_elements + blockSize - 1) / blockSize;
    
    // 根据数据类型调用相应的核函数
    if (input.dataType == fastllm::DataType::FLOAT32) {
        MakeDecayMaskKernel<float><<<gridSize, blockSize>>>(
            (float*)inputData, (float*)outputData, dim, outer);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        MakeDecayMaskKernel<half><<<gridSize, blockSize>>>(
            (half*)inputData, (half*)outputData, dim, outer);
    }

    // 等待核函数执行完成
    DeviceSync();
    
    FastllmCudaFinishInput(input, inputData);
    FastllmCudaFinishOutput(output, outputData);
    
    return true;
}

bool FastllmCudaRMSNorm(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &output, float eps) {
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareInput(output);

    int dimsLen = input.dims.size();
    int axis = dimsLen - 1;
    int outer = input.Count(0) / input.Count(axis);
    int channels = input.dims[axis];

    if (input.dataType == fastllm::DataType::FLOAT32) {
        if (channels < 64) {
            FastllmRMSNormKernelInner1<1> <<< outer, 1 >>>(cudaInput, (float *) weight.cudaData, cudaOutput, outer,
                                                           channels, eps);
        } else if (channels < 512) {
            FastllmRMSNormKernelInner1<64> <<< outer, 64 >>>(cudaInput, (float *) weight.cudaData, cudaOutput, outer,
                                                             channels, eps);
        } else {
            FastllmRMSNormKernelInner1<512> <<< outer, 512 >>>(cudaInput, (float *) weight.cudaData, cudaOutput, outer,
                                                               channels, eps);
        }
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        if (channels < 512) {
            FastllmRMSNormKernelInner1<64> <<< outer, 64 >>>((half*)cudaInput, (float*) weight.cudaData, (half*)cudaOutput, outer,
                                                             channels, eps);
        } else {
            FastllmRMSNormKernelInner1<512> <<< outer, 512 >>>((half*)cudaInput, (float*) weight.cudaData, (half*)cudaOutput, outer,
                                                               channels, eps);
        }
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
        if (gamma.dataType != fastllm::DataType::FLOAT32 || beta.dataType != fastllm::DataType::FLOAT32) {
            printf("layernorm datatype error.\n");
            exit(0);    
        } else if (input.dataType == fastllm::DataType::FLOAT32) {
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
        } else if (input.dataType == fastllm::DataType::FLOAT16) {
            if (channels < 64) {
                FastllmLayerNormKernelInner1<1> <<< outer, 1 >>>((half*)cudaInput, (float *) gamma.cudaData,
                                                                (float *) beta.cudaData, (half*)cudaOutput,
                                                                outer, channels);
            } else if (channels < 512) {
                FastllmLayerNormKernelInner1<64> <<< outer, 64 >>>((half*)cudaInput, (float *) gamma.cudaData,
                                                                (float *) beta.cudaData, (half*)cudaOutput,
                                                                outer, channels);
            } else {
                FastllmLayerNormKernelInner1<512> <<< outer, 512 >>>((half*)cudaInput, (float *) gamma.cudaData,
                                                                    (float *) beta.cudaData, (half*)cudaOutput,
                                                                    outer, channels);
            }
        } else {
            printf("layernorm datatype error.\n");
            exit(0);    
        }
    } else {
        printf("layernorm error.\n");
        exit(0);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

#ifndef USE_ROCM
// 自定义函子，用于处理每一行的 TopK 操作  
struct TopKFunctor {
    float* cudaInput;        // 指向原始输入数据的设备指针  
    float* cudaOutput;       // 指向输出数据的设备指针  
    int channels;
    int topk;

    // 构造函数  
    TopKFunctor(float* cudaInput, float* cudaOutput, int channels, int topk)
        : cudaInput(cudaInput), cudaOutput(cudaOutput), channels(channels), topk(topk) {}

    // thrust::for_each 会为每个行索引 i 调用这个操作符  
    // __host__ __device__ 使得函子可以在主机和设备上被调用（Thrust 要求）  
    __host__ __device__
    void operator()(int i) const {
        thrust::device_ptr<float> d_input(cudaInput);
        thrust::device_ptr<float> d_output(cudaOutput);

        // 当前行的起始位置  
        thrust::device_ptr<float> row_start = d_input + i * channels;
        thrust::device_ptr<float> row_end = row_start + channels;
        
        // 创建索引序列 [0, 1, 2, ..., channels-1]
        thrust::device_vector<int> indices(channels);
        thrust::sequence(indices.begin(), indices.end());
        
        // 使用zip迭代器将值和索引组合在一起  
        auto begin = thrust::make_zip_iterator(
            thrust::make_tuple(row_start, indices.begin()));
        auto end = thrust::make_zip_iterator(
            thrust::make_tuple(row_end, indices.end()));
        
        // 按值降序排序  
        thrust::sort(begin, end, 
            thrust::greater<thrust::tuple<float, int>>());
        
        // 复制前topk个结果到输出  
        for (int k = 0; k < topk; ++k) {
            d_output[i * topk * 2 + k * 2] = indices[k];     // 索引  
            d_output[i * topk * 2 + k * 2 + 1] = row_start[k]; // 值  
        }
    }
};

// 主函数/调用部分  
void topk_parallel_thrust(float* d_input, float* d_output, int outer, int channels, int topk) {
    // 创建函子实例  
    TopKFunctor functor(d_input, d_output, channels, topk);

    // 使用 thrust::for_each 和 counting_iterator 来并行处理每一行  
    thrust::for_each(
        thrust::counting_iterator<int>(0),      // 起始迭代器 (0)
        thrust::counting_iterator<int>(outer),  // 结束迭代器 (outer)
        functor                                 // 应用于每个元素的函子  
    );
}
#endif

bool FastllmCudaTopK(const fastllm::Data &input, fastllm::Data &output, int topk) {
    if (topk > 50) {
        printf("topk: unsupport topk > 50.");
        exit(0);
    }

    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareInput(output);

    int dimsLen = input.dims.size();
    int outer = input.Count(0) / input.Count(dimsLen - 1);
    int channels = input.dims[dimsLen - 1];

#ifdef USE_ROCM
    if (topk == 1) {
        FastllmLayerNormKernelTop1 <256> <<< outer, 256 >>> (cudaInput, cudaOutput, channels);
    } else {
        FastllmLayerNormKernelTopK <64, 50> <<< outer, 64 >>> (cudaInput, cudaOutput, topk, channels);
    }
#else
    if (outer > 4 || topk == 1) {
        if (topk == 1) {
            FastllmLayerNormKernelTop1 <256> <<< outer, 256 >>> (cudaInput, cudaOutput, channels);
        } else {
            FastllmLayerNormKernelTopK <64, 50> <<< outer, 64 >>> (cudaInput, cudaOutput, topk, channels);
        }    
    } else {
        TopKFunctor functor(cudaInput, cudaOutput, channels, topk);
        for (int i = 0; i < outer; ++i) {
            functor(i);
        }
    }
#endif
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

// CUDA kernel for SelectExpert
template <int THREAD_PER_BLOCK, int MAXK>
__global__ void FastllmSelectExpertKernel(float *logits, float *bias, int32_t *index, float *score, 
    int n, int numExperts, int topk, int hasBias, bool needNorm, float routeScale) {
    __shared__ float idData[THREAD_PER_BLOCK][MAXK];
    __shared__ float maxData[THREAD_PER_BLOCK][MAXK];
    
    int tokenIdx = blockIdx.x;
    float *inputData = logits + tokenIdx * numExperts;
    int32_t *outputIndex = index + tokenIdx * topk;
    float *outputScore = score + tokenIdx * topk;
    
    int tid = threadIdx.x;
    
    // Initialize
    for (int i = 0; i < topk; i++) {
        maxData[tid][i] = -1e100;
        idData[tid][i] = -1;
    }
    
    // Find topk experts
    for (int j = tid; j < numExperts; j += THREAD_PER_BLOCK) {
        float cur = inputData[j];
        if (hasBias) {
            cur -= bias[j];
        }
        
        for (int l = 0; l < topk; l++) {
            if (cur > maxData[tid][l]) {
                for (int x = topk - 1; x > l; x--) {
                    maxData[tid][x] = maxData[tid][x - 1];
                    idData[tid][x] = idData[tid][x - 1];
                }
                maxData[tid][l] = cur;
                idData[tid][l] = j;
                break;
            }
        }
    }
    __syncthreads();
    
    // Merge results from all threads
    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            int pos0 = 0, pos1 = 0;
            while (pos0 + pos1 < topk) {
                if (maxData[tid][pos0] > maxData[tid + s][pos1]) {
                    pos0++;
                } else {
                    pos1++;
                }
            }
            pos0--;
            pos1--;
            int pos = topk - 1;
            while (pos >= 0) {
                if (pos1 < 0 || (pos0 >= 0 && maxData[tid][pos0] < maxData[tid + s][pos1])) {
                    maxData[tid][pos] = maxData[tid][pos0];
                    idData[tid][pos] = idData[tid][pos0];
                    pos0--;
                } else {
                    maxData[tid][pos] = maxData[tid + s][pos1];
                    idData[tid][pos] = idData[tid + s][pos1];
                    pos1--;
                }
                pos--;
            }
        }
        __syncthreads();
    }
    
    // Write output
    if (tid == 0) {
        // Calculate sum for normalization
        float sum = 1.0f;
        if (needNorm) {
            sum = 0.0f;
            for (int i = 0; i < topk; i++) {
                int expertIdx = idData[0][i];
                sum += inputData[expertIdx];
            }
        }
        
        // Write index and score
        for (int i = 0; i < topk; i++) {
            int expertIdx = idData[0][i];
            outputIndex[i] = expertIdx;
            outputScore[i] = inputData[expertIdx] / sum * routeScale;
        }
    }
}

bool FastllmCudaSelectExpert(const fastllm::Data &logits, const fastllm::Data *gateBias, 
    fastllm::Data &index, fastllm::Data &score, int topk, bool needNorm, float routeScale) {
    if (topk > 50) {
        printf("SelectExpert: unsupport topk > 50, falling back to CPU implementation.\n");
        return false; // 返回 false 表示不支持，应该回退到 CPU
    }
    
    float *cudaLogits = (float *) FastllmCudaPrepareInput(logits);
    float *cudaBias = nullptr;
    int hasBias = 0;
    if (gateBias != nullptr && gateBias->dims.size() > 0) {
        cudaBias = (float *) FastllmCudaPrepareInput(*gateBias);
        hasBias = 1;
    }
    int32_t *cudaIndex = (int32_t *) FastllmCudaPrepareInput(index);
    float *cudaScore = (float *) FastllmCudaPrepareInput(score);
    
    int dimsLen = logits.dims.size();
    int n = logits.Count(0) / logits.dims[dimsLen - 1]; // number of tokens
    int numExperts = logits.dims[dimsLen - 1]; // number of experts
    
    // Use 64 threads to stay within shared memory limit (64*50*4*2 = 25KB < 48KB)
#ifdef USE_ROCM
    FastllmSelectExpertKernel<64, 50> <<< n, 64 >>> 
        (cudaLogits, cudaBias, cudaIndex, cudaScore, n, numExperts, topk, hasBias, needNorm, routeScale);
#else
    FastllmSelectExpertKernel<64, 50> <<< n, 64 >>> 
        (cudaLogits, cudaBias, cudaIndex, cudaScore, n, numExperts, topk, hasBias, needNorm, routeScale);
#endif
    
    FastllmCudaFinishInput(logits, cudaLogits);
    if (gateBias != nullptr && gateBias->dims.size() > 0) {
        FastllmCudaFinishInput(*gateBias, cudaBias);
    }
    FastllmCudaFinishOutput(index, cudaIndex);
    FastllmCudaFinishOutput(score, cudaScore);
    return true;
}

bool FastllmCudaPermute(fastllm::Data &input, const std::vector<int> &axis) {
    if (input.dataDevice != fastllm::DataDevice::CUDA) {
        printf("permute: data should in cuda.\n");
        exit(0);
    }
    int len = input.Count(0);
    uint8_t *tempData = (uint8_t *)FastllmCudaMalloc(len * input.unitSize);
    cudaMemcpy(tempData, input.cudaData, len * input.unitSize, cudaMemcpyDeviceToDevice);

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
    } else if (axis == std::vector <int> {1, 2, 0, 3}) {
        int n = input.dims[0];
        int m = input.dims[1] * input.dims[2];
        int k = input.dims[3];
        FastllmTransposeByRowKernel <256> <<< n * m, 256 >>>
                ((uint8_t*)input.cudaData, (uint8_t*)tempData, n, m, k * input.unitSize);
        input.Resize(new_dims);
    } else if (axis == std::vector <int> {0, 2, 1, 3} && input.dims[0] == 1) {
        int n = input.dims[1];
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
        int threadPerBlock = std::min(256, len);
        if (input.unitSize == 4) {
            FastllmPermuteKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(
                    (float *) input.cudaData,(float *)tempData, cudaTemp,(int) axis.size(), len);
        } else if (input.unitSize == 2) {
            FastllmPermuteKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(
                    (uint16_t *) input.cudaData,(uint16_t *)tempData, cudaTemp,(int) axis.size(), len);
        } else if (input.unitSize == 1) {
            FastllmPermuteKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(
                    (uint8_t *) input.cudaData,(uint8_t *)tempData, cudaTemp,(int) axis.size(), len);
        }

        FastllmCudaFree(cudaTemp);
    }

    FastllmCudaFree(tempData);
    DeviceSync();
    return true;
}

bool FastllmFloatToHalf(void *a, void *b, int len) {
    int threadPerBlock = std::min(256, len);
    FastllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((float*)a, (half*)b, len);
    DeviceSync();
    return true;
}

bool FastllmHalfToFloat(void *a, void *b, int len) {
    int threadPerBlock = std::min(256, len);
    FastllmCudaHalf2FloatKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((half*)a, (float*)b, len);
    DeviceSync();
    return true;
}

bool FastllmBF16ToFloat(void *a, void *b, int len) {
    int threadPerBlock = std::min(256, len);
    FastllmCudaBF162FloatKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((uint16_t*)a, (float*)b, len);
    DeviceSync();
    return true;
}

bool FastllmCudaEmbedding(const fastllm::Data &input, const fastllm::Data &weight,fastllm::Data &output) {
    int vocabSize = weight.dims[0], embSize = weight.dims[1];
    uint64_t inputLen = input.Count(0);

    float *inputData = (float*)input.cudaData;
    float *dstOutputData = (float*)output.cudaData;

    if (weight.dataType == fastllm::DataType::FLOAT32) {
        float *outputData = (float *) dstOutputData;
        float *weightData = (float *) weight.cudaData;
        FastllmCudaFloatEmbeddingKernel <128> <<<inputLen, 128>>> (inputData, weightData, outputData, embSize);
    } else if (weight.dataType == fastllm::DataType::FLOAT16) {
        half *outputData = (half *) dstOutputData;
        half *weightData = (half *) weight.cudaData;
        FastllmCudaFloatEmbeddingKernel <128> <<<inputLen, 128>>> (inputData, weightData, outputData, embSize);
    } else if (weight.dataType == fastllm::DataType::BFLOAT16) {
        std::vector <float> cpuInputData = std::vector <float> (inputLen, 0.0f);
        FastllmCudaCopyFromDeviceToHost(cpuInputData.data(), inputData, cpuInputData.size() * sizeof(float));
        float *outputData = (float *) dstOutputData;
        uint16_t *weightData = (uint16_t *) weight.cudaData;
        for (int i = 0; i < inputLen; i++) {
            int token = (int) (cpuInputData[i] + 1e-9);
            for (int j = 0; j < embSize; j++) {
                FastllmBF16ToFloat(outputData + i * embSize, weightData + token * embSize, embSize);
            }
        }
    } else {
        
    }

    DeviceSync();
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

    if (input0.dataType == fastllm::DataType::FLOAT32 && input1.dataType == fastllm::DataType::FLOAT32) {
        status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                        CUBLAS_OP_N, CUBLAS_OP_N,
                                        k, n, m, &alpha,
                                        cudaInput1, input1Stride, input1Spatial,
                                        cudaInput0, input0Stride, input0Spatial,
                                        &beta,
                                        cudaOutput, k, k * n, batch);
    } else if (input0.dataType == fastllm::DataType::FLOAT16 && input1.dataType == fastllm::DataType::FLOAT16) {
        half h_alpha = __float2half(alpha), h_beta = __float2half(beta);
        status = cublasHgemmStridedBatched(fastllmCublasHandle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                k, n, m, &h_alpha,
                (half*)cudaInput1, input1Stride, input1Spatial,
                (half*)cudaInput0, input0Stride, input0Spatial,
                &h_beta,
                (half*)cudaOutput, k, k * n, batch);
    } else if (input0.dataType == fastllm::DataType::FLOAT32 && input1.dataType == fastllm::DataType::FLOAT16) {
        half *tempInput0 = (half*)FastllmCudaMalloc(input0.Count(0) * sizeof(half));
        half *tempOutput = (half*)FastllmCudaMalloc(output.Count(0) * sizeof(half));
        FastllmFloatToHalf(cudaInput0, tempInput0, input0.Count(0));

        half h_alpha = __float2half(alpha), h_beta = __float2half(beta);
        status = cublasHgemmStridedBatched(fastllmCublasHandle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                k, n, m, &h_alpha,
                (half*)cudaInput1, input1Stride, input1Spatial,
                (half*)tempInput0, input0Stride, input0Spatial,
                &h_beta,
                (half*)tempOutput, k, k * n, batch);
        FastllmHalfToFloat(tempOutput, cudaOutput, output.Count(0));
        FastllmCudaFree(tempInput0);
        FastllmCudaFree(tempOutput);
    }

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

    if (input0.dataType == fastllm::DataType::FLOAT32 && input1.dataType == fastllm::DataType::FLOAT32) {
        status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                       CUBLAS_OP_T, CUBLAS_OP_N,
                                       k, n, m, &alpha,
                                       cudaInput1, input1Stride, input1Spatial,
                                       cudaInput0, input0Stride, input0Spatial,
                                       &beta,
                                       cudaOutput, k, k * n, batch);
    } else if (input0.dataType == fastllm::DataType::FLOAT16 && input1.dataType == fastllm::DataType::FLOAT16) {
        half h_alpha = __float2half(alpha), h_beta = __float2half(beta);
        status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                        CUBLAS_OP_T, CUBLAS_OP_N,
                                        k, n, m, &h_alpha,
                                        (half*)cudaInput1, input1Stride, input1Spatial,
                                        (half*)cudaInput0, input0Stride, input0Spatial,
                                        &h_beta,
                                        (half*)cudaOutput, k, k * n, batch);
    } else if (input0.dataType == fastllm::DataType::FLOAT32 && input1.dataType == fastllm::DataType::FLOAT16) {
        half *tempInput0 = (half*)FastllmCudaMalloc(input0.Count(0) * sizeof(half));
        half *tempOutput = (half*)FastllmCudaMalloc(output.Count(0) * sizeof(half));
        FastllmFloatToHalf(cudaInput0, tempInput0, input0.Count(0));

        half h_alpha = __float2half(alpha), h_beta = __float2half(beta);
        status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                        CUBLAS_OP_T, CUBLAS_OP_N,
                                        k, n, m, &h_alpha,
                                        (half*)cudaInput1, input1Stride, input1Spatial,
                                        (half*)tempInput0, input0Stride, input0Spatial,
                                        &h_beta,
                                        (half*)tempOutput, k, k * n, batch);
        FastllmHalfToFloat(tempOutput, cudaOutput, output.Count(0));
        FastllmCudaFree(tempInput0);
        FastllmCudaFree(tempOutput);
    }

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
    FastllmRotatePosition2DKernel <<< outer * 2 * n, std::min(rotaryDim, m / 4) >>> (cudaData, cudaPositionIds, cudaSin, cudaCos,
                                                                                len, bs, spatial, n, m,
                                                                                (int)positionIds.dims.back(), (int)sinData.dims[1], rotaryDim);

    FastllmCudaFinishInput(positionIds, cudaPositionIds);
    FastllmCudaFinishInput(sinData, cudaSin);
    FastllmCudaFinishInput(cosData, cudaCos);
    FastllmCudaFinishOutput(data, cudaData);

    return true;
}

bool FastllmCudaNearlyRotatePosition2D(fastllm::Data &data, const fastllm::Data &positionIds,
                                       const fastllm::Data &sinData, const fastllm::Data &cosData, int rotaryDim, int positionStride) {
    float *cudaData = (float *) FastllmCudaPrepareInput(data);
    float *cudaPositionIds = (float *) FastllmCudaPrepareInput(positionIds);
    float *cudaSin = (float *) FastllmCudaPrepareInput(sinData);
    float *cudaCos = (float *) FastllmCudaPrepareInput(cosData);

    int outer = data.dims[0] * data.dims[1];
    int spatial = data.Count(2);
    int len = data.dims[0], bs = data.dims[1];
    int n = data.dims[2], m = data.dims[3];
    positionStride = (int)positionIds.dims.back() * positionStride;

    if (data.dataType == fastllm::DataType::FLOAT32) {
        FastllmNearlyRotatePosition2DKernel <<< outer * n, std::min(rotaryDim, m / 2) >>> (cudaData, cudaPositionIds, cudaSin, cudaCos,
                                                                                    len, bs, spatial, n, m,
                                                                                    positionStride, (int)sinData.dims[1], rotaryDim);
    } else if (data.dataType == fastllm::DataType::FLOAT16) {
        FastllmNearlyRotatePosition2DKernel <<< outer * n, std::min(rotaryDim, m / 2) >>> ((half*)cudaData, cudaPositionIds, cudaSin, cudaCos,
                                                                                    len, bs, spatial, n, m,
                                                                                    positionStride, (int)sinData.dims[1], rotaryDim);
    }

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

    if (data.dataType == fastllm::DataType::FLOAT32) {
        FastllmLlamaRotatePosition2DKernel <<< outer * n, std::min(rotaryDim, m / 2) >>> (cudaData, cudaPositionIds, cudaSin, cudaCos,
                                                                                 len, bs, spatial, n, m,
                                                                                 (int)positionIds.dims.back(), (int)sinData.dims[1], rotaryDim);
    } else if (data.dataType == fastllm::DataType::FLOAT16) {
        FastllmLlamaRotatePosition2DKernel <<< outer * n, std::min(rotaryDim, m / 2) >>> ((half*)cudaData, cudaPositionIds, cudaSin, cudaCos,
                                                                                 len, bs, spatial, n, m,
                                                                                 (int)positionIds.dims.back(), (int)sinData.dims[1], rotaryDim);
    }
    FastllmCudaFinishInput(positionIds, cudaPositionIds);
    FastllmCudaFinishInput(sinData, cudaSin);
    FastllmCudaFinishInput(cosData, cudaCos);
    FastllmCudaFinishOutput(data, cudaData);
    return true;
}

bool FastllmCudaLlamaRotatePosition2DPart(fastllm::Data &data, const fastllm::Data &positionIds,
                                      const fastllm::Data &sinData, const fastllm::Data &cosData, int rotaryDim, int part) {
    float *cudaData = (float *) FastllmCudaPrepareInput(data);
    float *cudaPositionIds = (float *) FastllmCudaPrepareInput(positionIds);
    float *cudaSin = (float *) FastllmCudaPrepareInput(sinData);
    float *cudaCos = (float *) FastllmCudaPrepareInput(cosData);

    int outer = data.dims[0] * data.dims[1];
    int spatial = data.Count(2);
    int bs = data.dims[0], len = data.dims[1];
    int n = data.dims[2], m = data.dims[3];

    if (data.dataType == fastllm::DataType::FLOAT32) {
        FastllmLlamaRotatePosition2DPartKernel <<< outer * n, std::min(rotaryDim, part / 2) >>> (cudaData, cudaPositionIds, cudaSin, cudaCos,
                                                                                 len, bs, spatial, n, m,
                                                                                 (int)positionIds.dims.back(), (int)sinData.dims[1], part);
    } else if (data.dataType == fastllm::DataType::FLOAT16) {
        FastllmLlamaRotatePosition2DPartKernel <<< outer * n, std::min(rotaryDim, part / 2) >>> ((half*)cudaData, cudaPositionIds, cudaSin, cudaCos,
                                                                                 len, bs, spatial, n, m,
                                                                                 (int)positionIds.dims.back(), (int)sinData.dims[1], part);
    }
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

bool FastllmCudaRepeatPenalty (fastllm::Data &input, fastllm::Data &penalty, fastllm::Data &penaltyScale) {
    float *inputData = (float*)input.cudaData;
    float *penaltyData = (float*)penalty.cudaData;
    float *penaltyScaleData = (float*)penaltyScale.cudaData;
    int batch = penalty.dims[0], tokens = penalty.dims[1];
    int vocabs = input.dims.back();

    FastllmRepeatPenaltyKernel <64> <<<batch, 64>>> (inputData, penaltyData, penaltyScaleData, tokens, vocabs);
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

__global__ void FastllmCudaNaiveConv2DKernel(float *input, float *weight, float *bias, int inputChannels, int outputChannels, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int inputHeight, int inputWidth, int outputHeight, int outputWidth, float *output) {
    int oc = blockIdx.x;
    output += oc * outputHeight * outputWidth;
    {
        float *startWeight = weight + oc * (inputChannels * kernelH * kernelW);
        for (int t = threadIdx.x; t < outputHeight * outputWidth; t += blockDim.x) {
            int oh = t / outputWidth;
            int ow = t % outputWidth;

            int ih = oh * strideH - padH;
            int iw = ow * strideW - padW;
            float value = bias[oc];
            float *curWeight = startWeight;
            for (int c = 0; c < inputChannels; c++) {
                float *curInput = (float*)input + c * inputHeight * inputWidth;
                for (int h = 0; h < kernelH; h++) {
                    for (int w = 0; w < kernelW; w++) {
                        float inputValue = 0;
                        if (ih + h >= 0 && ih + h < inputHeight && iw + w >= 0 && iw + w < inputWidth) {
                            inputValue = curInput[(ih + h) * inputWidth + (iw + w)];
                        }
                        value += inputValue * (*(curWeight++));
                    }
                }
            }
            output[oh * outputWidth + ow] = value;
        }
    }
}

__global__ void FastllmCudaNaiveConv2DHalfKernel(float *input, half *weight, float *bias, int inputChannels, int outputChannels, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int inputHeight, int inputWidth, int outputHeight, int outputWidth, float *output) {
    int oc = blockIdx.x;
    output += oc * outputHeight * outputWidth;
    {
        half *startWeight = weight + oc * (inputChannels * kernelH * kernelW);
        for (int t = threadIdx.x; t < outputHeight * outputWidth; t += blockDim.x) {
            int oh = t / outputWidth;
            int ow = t % outputWidth;

            int ih = oh * strideH - padH;
            int iw = ow * strideW - padW;
            float value = bias[oc];
            half *curWeight = startWeight;
            for (int c = 0; c < inputChannels; c++) {
                float *curInput = (float*)input + c * inputHeight * inputWidth;
                for (int h = 0; h < kernelH; h++) {
                    for (int w = 0; w < kernelW; w++) {
                        float inputValue = 0;
                        if (ih + h >= 0 && ih + h < inputHeight && iw + w >= 0 && iw + w < inputWidth) {
                            inputValue = curInput[(ih + h) * inputWidth + (iw + w)];
                        }
                        value += inputValue * __half2float(*(curWeight++));
                    }
                }
            }
            output[oh * outputWidth + ow] = value;
        }
    }
}

template <typename T>
// CUDA kernel for per-channel 1D convolution
__global__ void Conv1DPerChannelKernel(
    T* input,
    const float* weight,
    const float* bias,
    T* output,
    int batchSize,
    int inputChannels,
    int outputChannels,
    int inputLength,
    int outputLength,
    int kernelSize,
    int stride,
    int padding,
    int groups) {
    
    // 计算当前线程处理的位置
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = batchSize * outputChannels * outputLength;
    
    if (tid >= totalElements) return;
    
    // 解析输出位置 (batch, channel, position)
    int ol = tid % outputLength;
    int oc = (tid / outputLength) % outputChannels;
    int b = tid / (outputChannels * outputLength);
    
    // 对于逐通道卷积，每个输出通道对应一个输入通道
    int g = oc;  // group index (因为 groups = inputChannels)
    int ic = g;  // 对应的输入通道
    
    // 计算输入起始位置
    int il_start = ol * stride - padding;
    
    // 初始化输出值（加上bias）
    float value = bias ? bias[oc] : 0.0f;
    
    // 获取权重和输入的指针
    const float* curWeight = weight + oc * kernelSize;
    const T* curInput = input + b * inputChannels * inputLength + ic * inputLength;
    
    // 执行卷积
    #pragma unroll
    for (int k = 0; k < kernelSize; k++) {
        int inputPos = il_start + k;
        
        // 边界检查
        if (inputPos >= 0 && inputPos < inputLength) {
            value += (float)curInput[inputPos] * curWeight[k];
        }
    }
    
    // 写入输出
    output[tid] = (T)value;
}

// 主函数
bool FastllmCudaConv1DPerChannelFloat32(
    const fastllm::Data &input, 
    fastllm::Data &weight, 
    fastllm::Data &bias, 
    int inputChannels, 
    int outputChannels, 
    int kernelSize, 
    int stride, 
    int padding, 
    fastllm::Data &output) {
    
    int groups = inputChannels;
    std::vector<int> dims = input.dims;
    int batchSize = dims[0];
    int inputLength = dims[2];
    int outputLength = (inputLength + 2 * padding - kernelSize) / stride + 1;
    
    // 准备输出维度
    output.Resize({batchSize, outputChannels, outputLength});
    
    // 获取设备指针
    float *d_input = (float*)input.cudaData;
    float *d_weight = (float*)weight.cudaData;
    float *d_bias = bias.dims.size() > 0 ? (float*)bias.cudaData : nullptr;
    float *d_output = (float*)output.cudaData;
    
    // 配置kernel参数
    int totalElements = batchSize * outputChannels * outputLength;
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

    if (input.dataType == fastllm::DataType::FLOAT32) {
        Conv1DPerChannelKernel<float> <<<blocksPerGrid, threadsPerBlock>>>(
                d_input, d_weight, d_bias, d_output,
                batchSize, inputChannels, outputChannels,
                inputLength, outputLength,
                kernelSize, stride, padding, groups
        );
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        Conv1DPerChannelKernel<half> <<<blocksPerGrid, threadsPerBlock>>>(
                (half*)d_input, d_weight, d_bias, (half*)d_output,
                batchSize, inputChannels, outputChannels,
                inputLength, outputLength,
                kernelSize, stride, padding, groups
        );
    }

    DeviceSync();
    return true;
}

bool FastllmCudaConv2DFloat32(const fastllm::Data &input, fastllm::Data &weight, fastllm::Data &bias, int inputChannels, int outputChannels, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, fastllm::Data &output) {
    if (weight.cudaData == nullptr || weight.extraCudaData.size() == 0) {
        float *cudaBiasData;
        cudaError_t state = cudaSuccess;
        state = cudaMalloc(&cudaBiasData, outputChannels * sizeof(float));
        if (bias.dims.size() > 0) {
            state = cudaMemcpy(cudaBiasData, (uint8_t*)bias.cudaData, outputChannels * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            state = cudaMemset(cudaBiasData, 0, outputChannels * sizeof(float));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaData.push_back((void*)cudaBiasData);
    }

    float *cudaBiasData = (float*)weight.extraCudaData[0];
    float *cudaInput = (float*)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float*)FastllmCudaPrepareOutput(output);

    std::vector <int> dims = input.dims;
    int inputHeight = dims[2], inputWidth = dims[3];
    int outputHeight = (inputHeight + padH + padH - kernelH) / strideH + 1;
    int outputWidth = (inputWidth + padW + padW - kernelW) / strideW + 1;

    if (weight.dataType == fastllm::DataType::FLOAT16) {
        FastllmCudaNaiveConv2DHalfKernel <<< outputChannels, 256 >>> (
            cudaInput, (half*)weight.cudaData, cudaBiasData, 
            inputChannels, outputChannels, kernelH, kernelW, strideH, strideW, padH, padW, 
            inputHeight, inputWidth, outputHeight, outputWidth, 
            cudaOutput
        );
    } else {
        FastllmCudaNaiveConv2DKernel <<< outputChannels, 256 >>> (
            cudaInput, (float*)weight.cudaData, cudaBiasData, 
            inputChannels, outputChannels, kernelH, kernelW, strideH, strideW, padH, padW, 
            inputHeight, inputWidth, outputHeight, outputWidth, 
            cudaOutput
        );
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

void FastllmReduce(uint8_t *output, uint8_t* partOutput, int len, int threadNum, fastllm::DataType dataType) {
    int threadPerBlock = std::min(256, len);
    if (dataType == fastllm::DataType::FLOAT32) {
        FastllmReduceKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>> ((float*)output, (float*)partOutput, len, threadNum);
    } else if (dataType == fastllm::DataType::FLOAT16) {
        FastllmReduceKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>> ((half*)output, (half*)partOutput, len, threadNum);
    }
}

void FastllmCudaSetDevice(int gpu_id) {
    cudaSetDevice(gpu_id);
}

int FastllmCudaGetDevice() {
    int id = -1;
    cudaGetDevice(&id);
    return id;
}

int GetPointerDeviceId(void *ptr) {
    cudaPointerAttributes attributes;
    cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

    if (err == cudaSuccess) {
#if (CUDART_VERSION < 10000) && !(defined(USE_ROCM))
        if (attributes.memoryType == cudaMemoryTypeDevice) {
#else
        if (attributes.type == cudaMemoryTypeDevice) {
#endif
            int device = attributes.device;
            printf("Pointer belongs to device %d\n", device);
            return device;
        } else {
            printf("Pointer is not device memory\n");
            return -1;
        }
    } else {
        printf("Error: %s\n", cudaGetErrorString(err));
        return -1;
    }
}

int FastllmCudaGetDeviceCount() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return deviceCount;
}

__global__ void FastllmCudaResetLogitsOfEOS(int batch, int stride, float *logits, int *res_lens, int *eos_nums, int *eos_ids) {
    int base = 0;
    for (int b = 0; b < batch; b++) {
        if (res_lens[b] > 0) {
            for (int i = 0; i < eos_nums[b]; i++) {
                logits[stride * b + eos_ids[base + i]] = 0;
            }
        }
        base += eos_nums[b];
    }
    return;
}
void FastllmResetLogitsOfEOS(int batch, fastllm::Data *logits, const std::vector<int> res_lens, 
    const std::vector<int> eos_nums, const std::vector<int> eos_ids) {
    cudaError_t state = cudaSuccess;
    int *cuda_res_lens = (int*)FastllmCudaMalloc(sizeof(int) * res_lens.size());
    state = cudaMemcpy(cuda_res_lens, res_lens.data(), sizeof(int) *res_lens.size(), cudaMemcpyHostToDevice);
    int *cuda_eos_nums = (int*)FastllmCudaMalloc(sizeof(int) * eos_nums.size());
    state = cudaMemcpy(cuda_eos_nums, eos_nums.data(), sizeof(int) *eos_nums.size(), cudaMemcpyHostToDevice);
    int *cuda_eos_ids = (int*)FastllmCudaMalloc(sizeof(int) * eos_ids.size());    
    state = cudaMemcpy(cuda_eos_ids, eos_ids.data(), sizeof(int) *eos_ids.size(), cudaMemcpyHostToDevice);
    FastllmCudaResetLogitsOfEOS <<<1,1>>> (batch, logits->Count(0) / batch, (float*)logits->cudaData, cuda_res_lens, cuda_eos_nums, cuda_eos_ids);
    checkCudaErrors("Error: CUDA error when reset logtis of EOS!", state);
    FastllmCudaFree(cuda_res_lens);
    FastllmCudaFree(cuda_eos_nums);
    FastllmCudaFree(cuda_eos_ids);
    return;
}

template <typename T>
__global__ void FastllmRecurrentGatedDeltaRuleKernel(
    T* last_recurrent_state,  // [n0, n1, n2, n3]
    const T* g_t,              // [n0, n1]
    const T* k_t,              // [n0, n1, n2]
    const T* v_t,              // [n0, n1, n3]
    const T* b_t,              // [n0, n1]
    const T* q_t,              // [n0, n1, n2]
    T* core_attn_out,          // [n0, n1, n3]
    int n0, int n1, int n2, int n3, int group)
{
    // Each block handles one (n0, n1) position
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    
    if (batch_idx >= n0 || head_idx >= n1) return;
    
    int base_idx = batch_idx * n1 + head_idx;
    int tid = threadIdx.x;
    
    // Shared memory for temporary storage
    extern __shared__ float shared_mem[];
    float* kv_mem = shared_mem;  // size: n3
    float* delta = &shared_mem[n3];  // size: n3
    
    // Step 1: Scale last_recurrent_state by g_t
    float g_val = expf((float)g_t[base_idx]);
    
    // Each thread handles multiple elements if n2*n3 > blockDim.x
    for (int idx = tid; idx < n2 * n3; idx += blockDim.x) {
        int state_idx = base_idx * n2 * n3 + idx;
        last_recurrent_state[state_idx] = (T)((float)last_recurrent_state[state_idx] * g_val);
    }
    __syncthreads();
    
    // Step 2: Compute kv_mem = sum(last_recurrent_state * k_t.unsqueeze(-1), dim=-2)
    if (tid < n3) {
        float sum = 0.0f;
        for (int j = 0; j < n2; j++) {
            float k_val = (float)k_t[base_idx / group * n2 + j];
            int state_idx = base_idx * n2 * n3 + j * n3 + tid;
            sum += (float)last_recurrent_state[state_idx] * k_val;
        }
        kv_mem[tid] = sum;
    }
    __syncthreads();
    
    // Step 3: Compute delta = (v_t - kv_mem) * b_t
    float b_val = (float)b_t[base_idx];
    if (tid < n3) {
        float v_val = (float)v_t[base_idx * n3 + tid];
        delta[tid] = (v_val - kv_mem[tid]) * b_val;
    }
    __syncthreads();
    
    // Step 4: Update last_recurrent_state += k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    for (int idx = tid; idx < n2 * n3; idx += blockDim.x) {
        int j = idx / n3;
        int k = idx % n3;
        float k_val = (float)k_t[base_idx / group * n2 + j];
        int state_idx = base_idx * n2 * n3 + idx;

        last_recurrent_state[state_idx] = (T)((float)last_recurrent_state[state_idx] + k_val * delta[k]);
    }
    __syncthreads();
    
    // Step 5: Compute core_attn_out = sum(last_recurrent_state * q_t.unsqueeze(-1), dim=-2)
    if (tid < n3) {
        float sum = 0.0f;
        for (int j = 0; j < n2; j++) {
            float q_val = (float)q_t[base_idx / group * n2 + j];
            int state_idx = base_idx * n2 * n3 + j * n3 + tid;
            sum += (float)last_recurrent_state[state_idx] * q_val;
        }
        core_attn_out[base_idx * n3 + tid] = (T)sum;
    }
}

void FastllmRecurrentGatedDeltaRule(fastllm::Data &q, fastllm::Data &k, fastllm::Data &v, fastllm::Data &g, fastllm::Data &b, fastllm::Data &last_recurrent_state, fastllm::Data &core_attn_out) {
    // Get dimensions
    int n0 = last_recurrent_state.dims[0];
    int n1 = last_recurrent_state.dims[1];
    int n2 = last_recurrent_state.dims[2];
    int n3 = last_recurrent_state.dims[3];
    
    // Move data to GPU if not already there
    float *d_last_state = (float*)last_recurrent_state.cudaData;
    float *d_g = (float*)g.cudaData;
    float *d_k = (float*)k.cudaData;
    float *d_v = (float*)v.cudaData;
    float *d_b = (float*)b.cudaData;
    float *d_q = (float*)q.cudaData;
    float *d_out = (float*)core_attn_out.cudaData;

    int group = v.dims[1] / q.dims[1];
    
    // Configure kernel launch parameters
    dim3 gridDim(n0, n1);  // One block per (batch, head) pair
    int threadsPerBlock = min(256, CUDA_MAX(n2 * n3, n3));
    
    // Calculate shared memory size
    size_t sharedMemSize = 2 * n3 * sizeof(float);  // for kv_mem and delta
    
    // Launch kernel
    if (q.dataType == fastllm::DataType::FLOAT32) {
        FastllmRecurrentGatedDeltaRuleKernel <float> <<<gridDim, threadsPerBlock, sharedMemSize>>>(
            d_last_state, d_g, d_k, d_v, d_b, d_q, d_out,
            n0, n1, n2, n3, group
        ); 
    } else if (q.dataType == fastllm::DataType::FLOAT16) {
        FastllmRecurrentGatedDeltaRuleKernel <half> <<<gridDim, threadsPerBlock, sharedMemSize>>>(
            (half*)d_last_state, (half*)d_g, (half*)d_k, (half*)d_v, (half*)d_b, (half*)d_q, (half*)d_out,
            n0, n1, n2, n3, group
        ); 
    }
    
    // Synchronize if needed
    DeviceSync();
}

void FastllmPickInput(uint8_t *input, uint8_t *partInput, int rows, int cols, int *cudaIndex) {
    for (int i = 0; i < rows; i++) {
        int index = cudaIndex[i];
        for (int j = 0; j < cols; j++) {
            partInput[i * cols + j] = input[index * cols + j];
        }
    }
}

// CUDA Kernel 函数
// 每个线程负责搬运一个 uint8_t 元素
__global__ void FastllmPickInputKernel(uint8_t *input, uint8_t *partInput, int rows, int cols, int *index) {
    // blockIdx.y 对应行索引 i
    int row = blockIdx.y;
    // blockIdx.x * blockDim.x + threadIdx.x 对应列索引 j
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // 边界检查：防止越界访问
    if (row < rows && col < cols) {
        // 读取该行在源数据中对应的真实行号
        int srcRow = index[row];
        
        // 计算扁平化的内存偏移量
        // 使用 long long 防止在大模型显存较大时 int32 溢出
        long long dstOffset = (long long)row * cols + col;
        long long srcOffset = (long long)srcRow * cols + col;
        // 执行拷贝
        partInput[dstOffset] = input[srcOffset];
    }
}

// Host 调用函数
void FastllmCudaPickInput(uint8_t *input, uint8_t *partInput, int rows, int cols, int *index) {
    // 设定 Block 大小：256 是通过是一个比较通用的高性能值
    dim3 block(256);
    // 设定 Grid 大小：
    // x 维度：覆盖所有列 (cols)，向上取整除以 256
    // y 维度：覆盖所有行 (rows)
    dim3 grid((cols + 255) / 256, rows);
    // 启动 Kernel
    FastllmPickInputKernel<<<grid, block>>>(input, partInput, rows, cols, index);
}

// CUDA Kernel 函数
// 每个线程负责一个 float 元素的计算和累加
__global__ void FastllmPickOutputKernel(float *partOutput, float *output, int rows, int cols, int *index, float *scales) {
    // blockIdx.y 对应行索引 i (partOutput 中的行)
    int i = blockIdx.y;
    // blockIdx.x * blockDim.x + threadIdx.x 对应列索引 j
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    // 边界检查
    if (i < rows && j < cols) {
        // 获取目标行号 idx 和 缩放因子 sca
        int idx = index[i];
        float sca = scales[i];
        // 计算扁平化的内存偏移量
        // 使用 long long 防止大模型显存地址溢出
        long long srcOffset = (long long)i * cols + j;
        long long dstOffset = (long long)idx * cols + j;
        // 执行 CPU 逻辑: output[idx * cols + j] += sca * partOutput[i * cols + j];
        // 注意：这里假设 index 映射的目标行通常是唯一的（在 LLM Batch 推理中通常如此）。
        // 如果多个 i 映射到同一个 idx，这里存在竞争冒险，但在 FastLLM 上下文中通常是 Scatter 操作。
        output[dstOffset] += sca * partOutput[srcOffset];
    }
}
// Host 调用函数
void FastllmCudaPickOutput(float *partOutput, float *output, int rows, int cols, int *index, float *scales) {
    // 设定 Block 大小：使用 256 作为通用高性能值
    dim3 block(256);
    
    // 设定 Grid 大小：
    // x 维度：覆盖所有列 (cols)，向上取整
    // y 维度：覆盖所有行 (rows)
    dim3 grid((cols + 255) / 256, rows);
    // 启动 Kernel
    FastllmPickOutputKernel<<<grid, block>>>(partOutput, output, rows, cols, index, scales);
}

// CUDA Kernel 函数
// 每个线程负责一个 float 元素的计算和累加
__global__ void FastllmPickOutputKernel(half *partOutput, half *output, int rows, int cols, int *index, float *scales) {
    // blockIdx.y 对应行索引 i (partOutput 中的行)
    int i = blockIdx.y;
    // blockIdx.x * blockDim.x + threadIdx.x 对应列索引 j
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    // 边界检查
    if (i < rows && j < cols) {
        // 获取目标行号 idx 和 缩放因子 sca
        int idx = index[i];
        float sca = scales[i];
        // 计算扁平化的内存偏移量
        // 使用 long long 防止大模型显存地址溢出
        long long srcOffset = (long long)i * cols + j;
        long long dstOffset = (long long)idx * cols + j;
        // 执行 CPU 逻辑: output[idx * cols + j] += sca * partOutput[i * cols + j];
        // 注意：这里假设 index 映射的目标行通常是唯一的（在 LLM Batch 推理中通常如此）。
        // 如果多个 i 映射到同一个 idx，这里存在竞争冒险，但在 FastLLM 上下文中通常是 Scatter 操作。
        output[dstOffset] = (half)((float)output[dstOffset] + sca * (float)partOutput[srcOffset]);
    }
}
// Host 调用函数
void FastllmCudaPickOutput(uint8_t *partOutput, uint8_t *output, int rows, int cols, int *index, float *scales, fastllm::DataType dataType) {
    // 设定 Block 大小：使用 256 作为通用高性能值
    dim3 block(256);
    
    // 设定 Grid 大小：
    // x 维度：覆盖所有列 (cols)，向上取整
    // y 维度：覆盖所有行 (rows)
    dim3 grid((cols + 255) / 256, rows);
    // 启动 Kernel
    if (dataType == fastllm::DataType::FLOAT32) {
        FastllmPickOutputKernel<<<grid, block>>>((float*)partOutput, (float*)output, rows, cols, index, scales);
    } else if (dataType == fastllm::DataType::FLOAT16) {
        FastllmPickOutputKernel<<<grid, block>>>((half*)partOutput, (half*)output, rows, cols, index, scales);
    } else {
        printf("FastllmCudaPickOutput Error: datatype error.\n");
        exit(0);
    }
}