#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <chrono>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/copy.h>
#include <thrust/functional.h>

#include "fastllm-cuda.cuh"
#include "fastllm.h"

#ifdef USE_ROCM
#include "fastllm-hip.h"
#endif

#define max(a, b) ((a) > (b) ? (a) : (b))

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 // support tensor core
#include "mma.h"
using namespace nvcuda;
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
#define CUDA_NO_TENSOR_CORE
#endif

#define checkCudaErrors(message, val) showError(val, message, __FILE__, __LINE__)

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

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FETCH_FLOAT2(pointer) (reinterpret_cast<float2*>(&(pointer))[0])

typedef union __align__(16) {
    uint2 in;
    uint8_t out[8];
} union_char8;

typedef union __align__(16) {
    uint32_t in;
    uint8_t out[4];
} union_char4;

typedef union __align__(16) _union_half_4 {
    uint2 in;
    half out[4];
    half2 out2[2];
    __device__ _union_half_4() {
      // Do nothing
    }
} union_half4;

typedef union __align__(16) _union_half_8 {
    uint4 in;
    half out[8];
    half2 out2[4];
    __device__ _union_half_8() {
      // Do nothing
    }
} union_half8;

const size_t ST128_FP16_COUNT = 8;

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

struct CudaInfos {
    int cudaArch;
    bool hasTensorCore;

    CudaInfos () {
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
};

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

template <int BN, int BM, int BK>
__global__ void HalfFC(
    half * __restrict__ a, half * __restrict__ b, half * __restrict__ c,
    const int N, const int M, const int K,
    half scale, const int base) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 // support tensor core
    int tid = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int wid = tid >> 5;

    int stN = bx * BN;
    int stK = by * BK;
    int wrap0 = wid >> 1;
    int wrap1 = wid & 1;

    if (base + stN + BN <= stK) {
        return;
    }

    __shared__ half cur[BN][BK];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[4][8];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b[4][8];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_c[i][j], 0.0);
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            wmma::load_matrix_sync(frag_a[i][j], &a[(stN + wrap0 * 64 + i * 16) * M + j * 16], M);
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            wmma::load_matrix_sync(frag_b[i][j], &b[(stK + wrap1 * 64 + i * 16) * M + j * 16], M);
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                wmma::mma_sync(frag_c[i][j], frag_a[i][k], frag_b[j][k], frag_c[i][j]);
            }
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::store_matrix_sync(&cur[(wrap0 * 64 + i * 16)][(wrap1 * 64 + j * 16)], frag_c[i][j], BK, wmma::mem_row_major);
        }
    }
    __syncthreads();

    for (int i = 0; i < BN; i++) {
        if (base + stN + i < stK + tid) {
            cur[i][tid] = (half)0;
        }
    }

    for (int i = 0; i < BN; i++) {
        c[(stN + i) * K + stK + tid] = __hmul(cur[i][tid], scale);
    }
#endif
}

void GpuQK(half *q, half *k, half *qk, int qlen, int klen, int dim, float scale, int base) {    
    const int BQ = 128, BK = 128, DIM = 128;
    dim3 blockDim(128);
    int BX = (qlen + BQ - 1) / BQ;
    int BY = (klen + BK - 1) / BK;
    dim3 gridDim(BX, BY);
    HalfFC <BQ, DIM, BK> <<<gridDim, blockDim>>> (q, k, qk, qlen, dim, klen, (half)scale, base);
}

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

__global__ void FastllmCudaInt82HalfKernel(uint8_t* a, float *scales, uint8_t *zeros, half *b, int len, int per) {
#ifdef CUDA_NO_TENSOR_CORE
    float scalesBuffer[2];
    uint8_t zerosBuffer[2];
    int threshold = ST128_FP16_COUNT;
    int index = (threadIdx.x + blockIdx.x * blockDim.x) * ST128_FP16_COUNT;
    for (int idx = index; idx < len; idx += (gridDim.x * blockDim.x) * ST128_FP16_COUNT) {
        int startIdx = idx / per;
        int endIdx = (idx + ST128_FP16_COUNT - 1) / per;
        scalesBuffer[1] = scalesBuffer[0] = scales[startIdx];
        zerosBuffer[1] = zerosBuffer[0] = zeros[startIdx];
        if (endIdx > startIdx) {
            threshold = (idx + ST128_FP16_COUNT - 1) % per;
            scalesBuffer[1] = scales[endIdx];
            zerosBuffer[1] = zeros[endIdx];
        }
        // 读取
        union_char8 aBuffer[2];
        half bBuffer[ST128_FP16_COUNT];
        aBuffer[0].in = *reinterpret_cast<const uint2 *>(a + idx);
        // 处理
        for (int i=0; i<ST128_FP16_COUNT; i++) {
            if (idx + i < len) {
                int scaleIdx = i < threshold ? 0 : 1;
                bBuffer[i] = __float2half(scalesBuffer[scaleIdx] * ((float)aBuffer[0].out[i] - zerosBuffer[scaleIdx]));
            }
        }
        reinterpret_cast<uint4 *>(b)[idx / ST128_FP16_COUNT] = *reinterpret_cast<uint4 *>(bBuffer);
    }
#else
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) {
        b[idx] = __float2half(scales[idx / per] * ((float)a[idx] - zeros[idx / per]));
    }
#endif
}

__global__ void FastllmCudaFP8E4M32HalfKernel(uint8_t* a, float *scales, half *b, int k, int m, int blockK, int blockM) {
    unsigned int tid = threadIdx.x;
    unsigned int st = blockIdx.x;

    int ms = (m - 1) / blockM + 1;
    scales += (st / blockK) * ms;

    for (int i = tid * 4; i < m; i += blockDim.x * 4) {
        float curScale = scales[i / blockM];
        uint32_t ori = *(uint32_t*)(a + st * m + i);
        half bf0 = __ushort_as_half( (((ori >> 0) & 0x80) << 8) | (((ori >> 0) & 0x7F) << 7) );
        half bf1 = __ushort_as_half( (((ori >> 8) & 0x80) << 8) | (((ori >> 8) & 0x7F) << 7) );
        half bf2 = __ushort_as_half( (((ori >> 16) & 0x80) << 8) | (((ori >> 16) & 0x7F) << 7) );
        half bf3 = __ushort_as_half( (((ori >> 24) & 0x80) << 8) | (((ori >> 24) & 0x7F) << 7) );

        b[st * m + i + 0] = __float2half((float)bf0 * curScale);
        b[st * m + i + 1] = __float2half((float)bf1 * curScale);
        b[st * m + i + 2] = __float2half((float)bf2 * curScale);
        b[st * m + i + 3] = __float2half((float)bf3 * curScale);
    }
}

__global__ void FastllmCudaInt4Group2HalfKernel(uint8_t* a, float *scales, float *mins, half *b, int len, int per,
                                                int group, int groupCnt) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int gid = idx / per * group + (idx % per / groupCnt);
    if (idx < len) {
        if (idx % 2 == 1) {
            b[idx] = __float2half(scales[gid] * (a[idx / 2] & 0xF) + mins[gid]);
        } else {
            b[idx] = __float2half(scales[gid] * (a[idx / 2] >> 4) + mins[gid]);
        }
    }
}

__global__ void FastllmCudaInt4Group2HalfKernel(uint8_t* a, half *scales, half *mins, half *b, int k, int m, int group, int groupCnt) {
    unsigned int tid = threadIdx.x;
    unsigned int st = blockIdx.x;
    half2 scalesBuffer;
    half2 minBuffer;
    int threshold = ST128_FP16_COUNT;
    for (int i = tid * ST128_FP16_COUNT; i < m; i += blockDim.x * ST128_FP16_COUNT) {
        int index = st * m + i;
        int startIdx = st * group + i / groupCnt;
        int endIdx = st * group + (i + ST128_FP16_COUNT - 1) / groupCnt;
        scalesBuffer.x = scalesBuffer.y = __ldg(scales + startIdx);
        minBuffer.x = minBuffer.y = __ldg(mins + startIdx);
        if (endIdx > startIdx) {
            threshold = (i + ST128_FP16_COUNT - 1) % groupCnt;
            scalesBuffer.y = __ldg(scales + endIdx);
            minBuffer.y = __ldg(mins + endIdx);
        }
        // 读取
        union_char4 aBuffer;
        union_half8 bBuffer;
        aBuffer.in = *reinterpret_cast<const uint32_t *>(a + index / 2);
        // 处理
        for (int j = 0; j < ST128_FP16_COUNT / 2; j++) {
            if (i + j * 2 + 1 < m) {
                float scale = __half2float(j * 2 < threshold ? scalesBuffer.x : scalesBuffer.y);
                float min = __half2float(j * 2 < threshold ? minBuffer.x : minBuffer.y);
                bBuffer.out[j * 2] = __float2half(scale * (aBuffer.out[j] >> 4) + min);
                bBuffer.out[j * 2 + 1] = __float2half(scale * (aBuffer.out[j] & 0xF) + min);
            }
        }
        reinterpret_cast<uint4 *>(b)[index / ST128_FP16_COUNT] = bBuffer.in;
    }
}

__global__ void FastllmCudaInt42HalfKernel(uint8_t* a, float *scales, float *mins, half *b, int len, int per) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float2 scalesBuffer;
    float2 minBuffer;
    int threshold = ST128_FP16_COUNT;
    for (int index = idx * ST128_FP16_COUNT; index < len; index += (gridDim.x * blockDim.x) * ST128_FP16_COUNT) {
        int startIdx = index / per;
        int endIdx = (index + ST128_FP16_COUNT - 1) / per;
        scalesBuffer.x = scalesBuffer.y = __ldg(scales + startIdx);
        minBuffer.x = minBuffer.y = __ldg(mins + startIdx);
        if (endIdx > startIdx) {
            threshold = (idx + ST128_FP16_COUNT - 1) % per;
            scalesBuffer.y = __ldg(scales + endIdx);
            minBuffer.y = __ldg(mins + endIdx);
        }
        // 读取
        union_char4 aBuffer;
        union_half8 bBuffer;
        aBuffer.in = *reinterpret_cast<const uint32_t *>(a + index / 2);
        // 处理
        for (int i = 0; i < ST128_FP16_COUNT / 2; i++) {
            if (index + i * 2 + 1 < len) {
                float scale = i * 2 < threshold ? scalesBuffer.x : scalesBuffer.y;
                float min = i * 2 < threshold ? minBuffer.x : minBuffer.y;
                bBuffer.out[i * 2] = __float2half(scale * (aBuffer.out[i] >> 4) + min);
                bBuffer.out[i * 2 + 1] = __float2half(scale * (aBuffer.out[i] & 0xF) + min);
            }
            // if (a[index + i] != aBuffer.out[i] && index < 100)
                // printf("%d - %d : %d\n", index + i, a[index + i], aBuffer.out[i]);
        }
        reinterpret_cast<uint4 *>(b)[idx] = bBuffer.in;
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
__global__ void FastllmAttentionMaskKernel(half *a, half *b, half maskValue, int n, int m, int spatial) {
    int on = blockIdx.x / m;
    int om = blockIdx.x % m;
    int o = on * m + om;
    int idx = threadIdx.x;
    for (int i = idx; i < spatial; i += THREAD_PER_BLOCK) {
        if (__half2float(b[on * spatial + i]) > 0.99) {
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
__global__ void SimpleMask(half* a, half *b, half maskValue, int spatial) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < spatial) {
        if (__half2float(b[i]) > 0.99) {
            a[i] = maskValue;
        }
    }
}

template <int THREAD_PER_BLOCK, typename T>
__global__ void CausalMask(T* a, T maskValue, int q, int k, int base) {
    a += blockIdx.x * k;
    for (int i = base + blockIdx.x + threadIdx.x + 1; i < k; i += THREAD_PER_BLOCK) {
        a[i] = maskValue;
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

__global__ void InitBlockAtten(float *sum0, float *max0, float *sum1, float *max1, int len) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len) {
        sum0[i] = sum1[i] = 0.0f;
        max0[i] = max1[i] = -10000.0f;
    }
}

template <int THREAD_PER_BLOCK>
__global__ void AttnBlockUpdate(half *data, int n, int m, float *lastMax, float *lastSum, float *curMax, float *curSum) {
    __shared__ float scale;
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;

    if (tid == 0) {
        float diff = fminf(lastMax[bid] - curMax[bid], 0.f);
        float oldSum = lastSum[bid] * expf(diff);
        scale = (curSum[bid] > 1e-10f) ? (oldSum / curSum[bid]) : 0.0f;

        lastSum[bid] = curSum[bid];
        lastMax[bid] = curMax[bid];
    }
    __syncthreads();

    for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
        data[bid * m + i] = (half)((float)data[bid * m + i] * scale);
    }
}

template <int THREAD_PER_BLOCK>
__device__ void FastllmSoftmaxKernelInner1Func(float *input, float *output, int channels, float *maxp, float *sump) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ float maxV;

    // 1. 每个线程计算一部分
    unsigned int tid = threadIdx.x;
    float maxValue = -1e100;
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
        if (maxp != nullptr) {
            maxp[0] = sdata[0];
        }
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
            sdata[0] = 0.0001;
        }
        if (sump != nullptr) {
            sump[0] = sdata[0];
        }
    }
    __syncthreads();

    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        output[i] /= sdata[0];
    }
}

__device__ half FastllmHalfMaxFunc(const __half a, const __half b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
    return __half2float(a) >= __half2float(b) ? a : b;
#else
#if defined(CUDART_VERSION) && CUDART_VERSION > 11000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __hmax(a, b);
#else
    return __hge(a, b) ? a : b;
#endif
#endif
}

template <int THREAD_PER_BLOCK>
__device__ void FastllmSoftmaxKernelInner1Func(half *input, half *output, int channels, float *maxp, float *sump) {
    __shared__ float sdata[THREAD_PER_BLOCK];

    // 1. 每个线程计算一部分
    unsigned int tid = threadIdx.x;
    float maxValue = -1e10;
    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        maxValue = max(maxValue, (float)input[i]);
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
        if (maxp != nullptr) {
            sdata[0] = max(maxp[0], sdata[0]);
        }
    }
    __syncthreads();
    float maxV = sdata[0];
    __syncthreads();

    // 4. 求和
    float sum = 0;
    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        sum = sum + exp((float)input[i] - maxV);
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
            sdata[0] = 0.0001;
        }
        if (sump != nullptr) {
            sump[0] = sump[0] * exp(maxp[0] - maxV) + sdata[0];
            sdata[0] = sump[0];
            maxp[0] = maxV;
        }
    }
    __syncthreads();

    float scale = 1.0 / sdata[0];
    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        output[i] = (half)(exp((float)input[i] - maxV) * scale);
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmSoftmaxKernelInner1(float* input, float *output, int outer, int channels) {
    int o = blockIdx.x;
    FastllmSoftmaxKernelInner1Func <THREAD_PER_BLOCK> (input + o * channels, output + o * channels, channels, nullptr, nullptr);
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmSoftmaxKernelInner1(half* input, half *output, int outer, int channels) {
    int o = blockIdx.x;
    FastllmSoftmaxKernelInner1Func <THREAD_PER_BLOCK> (input + o * channels, output + o * channels, channels, nullptr, nullptr);
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmSoftmaxKernelInner1(half* input, half *output, int outer, int channels, float *maxp, float *sump) {
    int o = blockIdx.x;
    FastllmSoftmaxKernelInner1Func <THREAD_PER_BLOCK> (input + o * channels, output + o * channels, channels, maxp + o, sump + o);
}

template <int THREAD_PER_BLOCK, typename T>
__global__ void FastllmSoftmaxKernelInner1WithCausalMask(T* input, T *output, int outer, int channels, int base) {
    int o = blockIdx.x;
    FastllmSoftmaxKernelInner1Func <THREAD_PER_BLOCK> (input + o * channels, output + o * channels, o + base + 1, nullptr, nullptr);
}

template <int THREAD_PER_BLOCK, typename T>
__global__ void FastllmSoftmaxKernelInner1WithCausalMask(T* input, T *output, int outer, int channels, int base, float *maxp, float *sump) {
    int o = blockIdx.x;
    FastllmSoftmaxKernelInner1Func <THREAD_PER_BLOCK> (input + o * channels, output + o * channels, min(channels, o + base + 1), maxp + o, sump + o);
}

template <typename T, int THREAD_PER_BLOCK>
__global__ void FastllmSoftmaxKernelBatchInner1(uint8_t** pointer) {
    int o = blockIdx.x;
    FastllmSoftmaxKernelInner1Func <THREAD_PER_BLOCK> ((T*)pointer[o * 3], (T*)pointer[o * 3 + 1],
                                                       (int)((size_t)pointer[o * 3 + 2]), nullptr, nullptr);
}

template <typename T, int THREAD_PER_BLOCK>
__global__ void FastllmSoftmaxKernelBatchInner1(uint8_t** pointer, int outer) {
    int o = blockIdx.x;
    int channels = (int)((size_t)pointer[o / outer * 2 + 1]);
    FastllmSoftmaxKernelInner1Func <THREAD_PER_BLOCK> ((T*)pointer[o / outer * 2] + (o % outer) * channels, (T*)pointer[o / outer * 2] + (o % outer) * channels,
                                                       channels, nullptr, nullptr);
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
    const half zero = __float2half_rn(0.0);
    float4 regA;
    union_half4 regB;

    // 1. 计算
    int st = blockIdx.x * PART;
    int end = st + PART;
    for (int p = st; p < end; p++) {
        sdata[tid] = 0;
        const half *baseB = B + p * m;
#ifdef CUDA_NO_TENSOR_CORE
#pragma unroll
        for (int i = tid*4; i < m; i += THREAD_PER_BLOCK*4) {
            regA = FETCH_FLOAT4(A[i]);
            regB.in = *reinterpret_cast<const uint2 *>(baseB + i);
            float sum = 0.0f;
            if (i < m)
                sum += regA.x * __low2float(regB.out2[0]);
            if (i + 1 < m)
                sum += regA.y * __high2float(regB.out2[0]);
            if (i + 2 < m)
                sum += regA.z * __low2float(regB.out2[1]);
            if (i + 3 < m)
                sum += regA.w * __high2float(regB.out2[1]);
            sdata[tid] += sum;
        }
#else
        for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
            sdata[tid] += A[i] * (float)B[p * m + i];
        }
#endif
        __syncthreads();
        float diff = 0.0f;
        for (unsigned int s = THREAD_PER_BLOCK/2; s > 0; s >>= 1) {
            if (tid < s) {
                float other = sdata[tid + s] - diff;
                float sumTmp = sdata[tid] + other;
                diff = (sumTmp - sdata[tid]) - other;
                sdata[tid] = sumTmp;
            }
            __syncthreads();
        }

        if (tid == 0) {
            C[p] = sdata[0] + __ldg(bias + p);
        }
        __syncthreads();
    }
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvFp16Fp16Kernel2MultiRow(half *A, half *B, half *C, half *bias, int m, int k) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    const half zero = __float2half_rn(0.0);
    union_half8 regA;
    union_half8 regB;

    // 1. 计算
    int st = blockIdx.x;
    int p = st;
#pragma unroll
    for (int x = 0; x < PART; x++) sdata[x][tid] = 0;
        
    const half *baseB = B + p * m;

    if (m % 8 == 0) {
#pragma unroll
        for (int i = tid * 8; i < m; i += THREAD_PER_BLOCK * 8) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                regA.in = *reinterpret_cast<const uint4 *>(A + x * m + i);
                regB.in = *reinterpret_cast<const uint4 *>(baseB + i);
                float sum = 0.0f;
                if (i < m)
                    sum += __low2float(regA.out2[0]) * __low2float(regB.out2[0]);
                if (i + 1 < m)
                    sum += __high2float(regA.out2[0]) * __high2float(regB.out2[0]);
                if (i + 2 < m)
                    sum += __low2float(regA.out2[1]) * __low2float(regB.out2[1]);
                if (i + 3 < m)
                    sum += __high2float(regA.out2[1]) * __high2float(regB.out2[1]);
                if (i + 4 < m)
                    sum += __low2float(regA.out2[2]) * __low2float(regB.out2[2]);
                if (i + 5 < m)
                    sum += __high2float(regA.out2[2]) * __high2float(regB.out2[2]);
                if (i + 6 < m)
                    sum += __low2float(regA.out2[3]) * __low2float(regB.out2[3]);
                if (i + 7 < m)
                    sum += __high2float(regA.out2[3]) * __high2float(regB.out2[3]);
                sdata[x][tid] += sum;
            }
        }
    } else {
        for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                sdata[x][tid] += (float)A[i + x * m] * (float)baseB[i];
            }
        }
    }
    __syncthreads();
    float diff = 0.0f;
    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            #pragma unroll
            for (int x = 0; x < PART; x++) {
                float other = sdata[x][tid + s] - diff;
                float sumTmp = sdata[x][tid] + other;
                diff = (sumTmp - sdata[x][tid]) - other;
                sdata[x][tid] = sumTmp;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (bias != nullptr) {
#pragma unroll
            for (int x = 0; x < PART; x++) C[p + k * x] = (half)(sdata[x][0] + (float)(__ldg(bias + p)));
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++) C[p + k * x] = (half)(sdata[x][0]);
        }
    }
    __syncthreads();
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvFp32Fp16Kernel2MultiRow(float *A, half *B, float *C, float *bias, int m, int k) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    const half zero = __float2half_rn(0.0);
    float4 regA;
    union_half4 regB;

    // 1. 计算
    int st = blockIdx.x;
    int p = st;
#pragma unroll
    for (int x = 0; x < PART; x++) sdata[x][tid] = 0;
        
    const half *baseB = B + p * m;
    if (m % 4 == 0) {
#pragma unroll
        for (int i = tid * 4; i + 3 < m; i += THREAD_PER_BLOCK * 4) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                regA = FETCH_FLOAT4(A[i + x * m]);
                regB.in = *reinterpret_cast<const uint2 *>(baseB + i);
                float sum = 0.0f;
                if (i < m)
                    sum += regA.x * __low2float(regB.out2[0]);
                if (i + 1 < m)
                    sum += regA.y * __high2float(regB.out2[0]);
                if (i + 2 < m)
                    sum += regA.z * __low2float(regB.out2[1]);
                if (i + 3 < m)
                    sum += regA.w * __high2float(regB.out2[1]);
                sdata[x][tid] += sum;
            }
        }
    } else {
        for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
#pragma unroll
            for (int x = 0; x < PART; x++) {
                sdata[x][tid] += A[i + x * m] * (float)baseB[i];
            }
        }
    }
    __syncthreads();
    float diff = 0.0f;
    for (unsigned int s = THREAD_PER_BLOCK/2; s > 0; s >>= 1) {
        if (tid < s) {
            #pragma unroll
            for (int x = 0; x < PART; x++) {
                float other = sdata[x][tid + s] - diff;
                float sumTmp = sdata[x][tid] + other;
                diff = (sumTmp - sdata[x][tid]) - other;
                sdata[x][tid] = sumTmp;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (bias == nullptr) {
            for (int x = 0; x < PART; x++) C[p + k * x] = sdata[x][0];
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++) C[p + k * x] = sdata[x][0] + __ldg(bias + p);
        }
    }
    __syncthreads();
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

    float4 regA;
    union_char4 regB;
    
    // 2. 计算
    int st = blockIdx.x * PART;
    int end = st + PART;
    for (int p = st; p < end; p++) {
        sdata[tid] = 0;
        uint8_t zero = zeros[p];
        const uint8_t *baseB = B + p * m;
#ifdef CUDA_NO_TENSOR_CORE
#pragma unroll
        for (int i = tid*4; i < m; i += THREAD_PER_BLOCK*4) {
            regA = FETCH_FLOAT4(A[i]);
            regB.in = *reinterpret_cast<const uint32_t *>(baseB + i);
            float sum = 0.0f;
            if (i < m)
                sum += regA.x * (float)(regB.out[0] - zero);
            if (i + 1 < m)
                sum += regA.y * (float)(regB.out[1] - zero);
            if (i + 2 < m)
                sum += regA.z * (float)(regB.out[2] - zero);
            if (i + 3 < m)
                sum += regA.w * (float)(regB.out[3] - zero);
            sdata[tid] += sum;
        }
#else
        for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
            sdata[tid] += A[i] * (B[p * m + i] - zero);
        }
#endif
        __syncthreads();
        float diff = 0.0f;
        for (unsigned int s = THREAD_PER_BLOCK/2; s > 0; s >>= 1) {
            if (tid < s) {
                float other = sdata[tid + s] - diff;
                float sumTmp = sdata[tid] + other;
                diff = (sumTmp - sdata[tid]) - other;
                sdata[tid] = sumTmp;
            }
            __syncthreads();
        }

        if (tid == 0) {
            C[p] = sdata[0] * __ldg(scales + p) + __ldg(bias + p);
        }
        __syncthreads();
    }
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvFp16Int8Kernel2(half *A, uint8_t *B, half *C,
                                       half *bias, float *scales, uint8_t *zeros,
                                       int m, int k) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    union_half8 regA;
    union_char8 regB;
    
    // 2. 计算
    int st = blockIdx.x * PART;
    int end = st + PART;
    for (int p = st; p < end; p++) {
        sdata[tid] = 0;
        uint8_t zero = zeros[p];
        const uint8_t *baseB = B + p * m;
#pragma unroll
        for (int i = tid * ST128_FP16_COUNT; i < m; i += THREAD_PER_BLOCK * ST128_FP16_COUNT) {
            regA.in = *reinterpret_cast<const uint4 *>(A + i);
            regB.in = *reinterpret_cast<const uint2 *>(baseB + i);
            float sum = 0.0f;
            if (i < m)
                sum += __low2float(regA.out2[0]) * (float)(regB.out[0] - zero);
            if (i + 1 < m)
                sum += __high2float(regA.out2[0]) * (float)(regB.out[1] - zero);
            if (i + 2 < m)
                sum += __low2float(regA.out2[1]) * (float)(regB.out[2] - zero);
            if (i + 3 < m)
                sum += __high2float(regA.out2[1]) * (float)(regB.out[3] - zero);
            if (i + 4 < m)
                sum += __low2float(regA.out2[2]) * (float)(regB.out[4] - zero);
            if (i + 5 < m)
                sum += __high2float(regA.out2[2]) * (float)(regB.out[5] - zero);
            if (i + 6 < m)
                sum += __low2float(regA.out2[3]) * (float)(regB.out[6] - zero);
            if (i + 7 < m)
                sum += __high2float(regA.out2[3]) * (float)(regB.out[7] - zero);
            sdata[tid] += sum;
        }

        __syncthreads();
        float diff = 0.0f;
        for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) {
                float other = sdata[tid + s] - diff;
                float sumTmp = sdata[tid] + other;
                diff = (sumTmp - sdata[tid]) - other;
                sdata[tid] = sumTmp;
            }
            __syncthreads();
        }

        if (tid == 0) {
            if (bias != nullptr) {
                C[p] = (half)(sdata[0] * __ldg(scales + p) + (float)__ldg(bias + p));
            } else {
                C[p] = (half)(sdata[0] * __ldg(scales + p));
            }
        }

        __syncthreads();
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
__global__ void FastllmGemvInt4GroupKernel3(float *A, uint8_t *B, float *C,
                                             float *bias, half *scales, half *mins,
                                             int m, int k, int group, int groupCnt) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    // 1. 计算
    int st = blockIdx.x * PART;
    int end = st + PART;
    #pragma unroll
    for (int p = 0; p < PART; p++) {
        sdata[p][tid] = 0;
    }

    for (int i = tid * 2; i < m / 2; i += THREAD_PER_BLOCK * 2) {
        float4 aBuffer = FETCH_FLOAT4(A[i * 2]);

        for (int p = st; p < end; p++) {
            uint16_t bBuffer = *reinterpret_cast<const uint16_t *>(B + p * m / 2 + i);
            int g = p * group + (i * 2 / groupCnt);
            float curmin = __half2float(__ldg(mins + g)), curscale = __half2float(__ldg(scales + g));
            sdata[p - st][tid] += aBuffer.x * (curmin + curscale * (float)((bBuffer >> 4) & 15)) 
                         + aBuffer.y * (curmin + curscale * (float)(bBuffer & 15));
            sdata[p - st][tid] += aBuffer.z * (curmin + curscale * (float)(bBuffer >> 12)) 
                         + aBuffer.w * (curmin + curscale * (float)((bBuffer >> 8) & 15));
        }
    }
    __syncthreads();
    for (int p = 0; p < PART; p++) {
        for (unsigned int s = 1; s < THREAD_PER_BLOCK; s *= 2) {
            if ((tid & (2 * s - 1)) == 0) {
                sdata[p][tid] += sdata[p][tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            C[st + p] = sdata[p][0] + bias[st + p];
        }
        __syncthreads();
    }
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvInt4GroupKernel2(float *A, uint8_t *B, float *C,
                                             float *bias, half *scales, half *mins,
                                             int m, int k, int group, int groupCnt) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;
    // 1. 计算
    int st = blockIdx.x * PART;
    int end = st + PART;
    #pragma unroll
    for (int p = 0; p < PART; p++) {
        sdata[p][tid] = 0;
    }

    for (int i = tid; i < m / 8; i += THREAD_PER_BLOCK) {
        float4 aBuffer = FETCH_FLOAT4(A[i * 8]);
        float4 bBuffer = FETCH_FLOAT4(A[i * 8 + 4]);

        for (int p = st; p < end; p++) {
            uint8_t now0 = B[p * m / 2 + i * 4];
            uint8_t now1 = B[p * m / 2 + i * 4 + 1];
            uint8_t now2 = B[p * m / 2 + i * 4 + 2];
            uint8_t now3 = B[p * m / 2 + i * 4 + 3];
            int g = p * group + (i * 8 / groupCnt);
            float curmin = (float)mins[g], curscale = (float)scales[g];
            sdata[p - st][tid] += (aBuffer.x * (curmin + (float)curscale * (now0 >> 4)) 
                         + aBuffer.y * (curmin + (float)curscale * (now0 & 15)));
            sdata[p - st][tid] += (aBuffer.z * (curmin + (float)curscale * (now1 >> 4)) 
                         + aBuffer.w * (curmin + (float)curscale * (now1 & 15)));
            sdata[p - st][tid] += (bBuffer.x * (curmin + (float)curscale * (now2 >> 4)) 
                         + bBuffer.y * (curmin + (float)curscale * (now2 & 15)));
            sdata[p - st][tid] += (bBuffer.z * (curmin + (float)curscale * (now3 >> 4)) 
                         + bBuffer.w * (curmin + (float)curscale * (now3 & 15)));
        }
    }
    __syncthreads();
    for (int p = 0; p < PART; p++) {
        for (unsigned int s = 1; s < THREAD_PER_BLOCK; s *= 2) {
            if ((tid & (2 * s - 1)) == 0) {
                sdata[p][tid] += sdata[p][tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            C[st + p] = sdata[p][0] + bias[st + p];
        }
        __syncthreads();
    }
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvHalfInt4GroupKernelMultiRow(half *A, uint8_t *B, half *C,
                                             half *bias, half *scales, half *mins,
                                             int m, int k, int group, int groupCnt) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    // 1. 计算
    int st = blockIdx.x;
    int end = st;
#pragma unroll
    for (int x = 0; x < PART; x++) sdata[x][tid] = 0;

    union_char4 bBuffer;
    for (int i = tid; i < m / 8; i += THREAD_PER_BLOCK) {
        bBuffer.in = *reinterpret_cast<const uint32_t *>(B + st * m / 2 + i * 4);
        // uint8_t now0 = B[st * m / 2 + i * 4];
        // uint8_t now1 = B[st * m / 2 + i * 4 + 1];
        // uint8_t now2 = B[st * m / 2 + i * 4 + 2];
        // uint8_t now3 = B[st * m / 2 + i * 4 + 3];
        int g = st * group + (i * 8 / groupCnt);
        float curmin = (float)mins[g], curscale = (float)scales[g];
        for (int x = 0; x < PART; x++) {
            union_half8 aBuffer;
            aBuffer.in = *reinterpret_cast<const uint4 *>(A + x * m + i * 8);
            sdata[x][tid] += ((float)aBuffer.out[0] * (curmin + curscale * (bBuffer.out[0] >> 4)) 
                         + (float)aBuffer.out[1] * (curmin + curscale * (bBuffer.out[0] & 15)));
            sdata[x][tid] += ((float)aBuffer.out[2] * (curmin + curscale * (bBuffer.out[1] >> 4)) 
                         + (float)aBuffer.out[3] * (curmin + curscale * (bBuffer.out[1] & 15)));
            sdata[x][tid] += ((float)aBuffer.out[4] * (curmin + curscale * (bBuffer.out[2] >> 4)) 
                         + (float)aBuffer.out[5] * (curmin + curscale * (bBuffer.out[2] & 15)));
            sdata[x][tid] += ((float)aBuffer.out[6] * (curmin + curscale * (bBuffer.out[3] >> 4)) 
                         + (float)aBuffer.out[7] * (curmin + curscale * (bBuffer.out[3] & 15)));
        }
    }

    __syncthreads();
    for (unsigned int s = 1; s < THREAD_PER_BLOCK; s *= 2) {
#pragma unroll
        for (int x = 0; x < PART; x++) {
            if ((tid & (2 * s - 1)) == 0) {
                sdata[x][tid] += sdata[x][tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (bias != nullptr) {
#pragma unroll
            for (int x = 0; x < PART; x++) C[st + k * x] = (half)(sdata[x][0] + (float)(__ldg(bias + st)));
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++) C[st + k * x] = (half)(sdata[x][0]);
        }
    }
    __syncthreads();
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

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvFp16Int4NoZeroKernel2(half *A, uint8_t *B, half *C,
                                                half *bias, float *scales, float *mins,
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
            sdata[tid] += ((float)A[i * 2] * (minv + (now >> 4)) + (float)A[i * 2 + 1] * (minv + (now & 15)));
        }
        __syncthreads();
        for (unsigned int s = 1; s < THREAD_PER_BLOCK; s *= 2) {
            if ((tid & (2 * s - 1)) == 0) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            if (bias == nullptr) {
                C[p] = (half)(sdata[0] * scales[p]);
            } else {
                C[p] = (half)(sdata[0] * scales[p] + (float)bias[p]);
            }
        }
        __syncthreads();
    }
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvFP8E4M3Kernel1MultiRow(float *A, uint8_t *B, float *C,
                                                    float *bias, float *scales,
                                                    int m, int k, int blockM, int blockK) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    // 1. 计算
    int st = blockIdx.x;
#pragma unroll
    for (int x = 0; x < PART; x++) sdata[x][tid] = 0;
    int ms = (m - 1) / blockM + 1;
    const float magicScaleConstant = exp2f(120.0f);
    scales += (st / blockK) * ms;

    const uint8_t *baseB = (uint8_t*)B + st * m;
    for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
        float curScale = scales[i / blockM];
        uint32_t bb = ((uint32_t*)(baseB + i))[0];
        float bf0 = __uint_as_float( (((bb >> 0) & 0x80) << 24) | (((bb >> 0) & 0x7F) << 20) ) * curScale;
        float bf1 = __uint_as_float( (((bb >> 8) & 0x80) << 24) | (((bb >> 8) & 0x7F) << 20) ) * curScale;
        float bf2 = __uint_as_float( (((bb >> 16) & 0x80) << 24) | (((bb >> 16) & 0x7F) << 20) ) * curScale;
        float bf3 = __uint_as_float( (((bb >> 24) & 0x80) << 24) | (((bb >> 24) & 0x7F) << 20) ) * curScale;

        // float bf0 = (float)baseB[i + 0] * curScale;
        // float bf1 = (float)baseB[i + 1] * curScale;
        // float bf2 = (float)baseB[i + 2] * curScale;
        // float bf3 = (float)baseB[i + 3] * curScale;
#pragma unroll
        for (int x = 0; x < PART; x++) {
            float4 aBuffer = FETCH_FLOAT4(A[i + x * m]);

            sdata[x][tid] += aBuffer.x * bf0;
            sdata[x][tid] += aBuffer.y * bf1;
            sdata[x][tid] += aBuffer.z * bf2;
            sdata[x][tid] += aBuffer.w * bf3;
        }
    }
    __syncthreads();

    float diff = 0.0f;
    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            #pragma unroll
            for (int x = 0; x < PART; x++) {
                float other = sdata[x][tid + s] - diff;
                float sumTmp = sdata[x][tid] + other;
                diff = (sumTmp - sdata[x][tid]) - other;
                sdata[x][tid] = sumTmp;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (bias == nullptr) {
            for (int x = 0; x < PART; x++) C[st + k * x] = sdata[x][0] * magicScaleConstant;
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++) C[st + k * x] = sdata[x][0] * magicScaleConstant + bias[st];
        }
    }
    __syncthreads();
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvHalfFP8E4M3Kernel1MultiRow(half *A, uint8_t *B, half *C,
                                                    half *bias, float *scales,
                                                    int m, int k, int blockM, int blockK) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    // 1. 计算
    int st = blockIdx.x;
#pragma unroll
    for (int x = 0; x < PART; x++) sdata[x][tid] = 0;
    int ms = (m - 1) / blockM + 1;
    const float magicScaleConstant = exp2f(8.0f);
    scales += (st / blockK) * ms;

    const uint8_t *baseB = (uint8_t*)B + st * m;
    union_half4 regA;
    for (int i = tid * 4; i < m; i += THREAD_PER_BLOCK * 4) {
        float curScale = scales[i / blockM];
        uint32_t bb = ((uint32_t*)(baseB + i))[0];
        __half2 B01 = make_half2(__short_as_half( (((bb >> 0) & 0x80) << 8) | (((bb >> 0) & 0x7F) << 7) ), 
                                __short_as_half( (((bb >> 8) & 0x80) << 8) | (((bb >> 8) & 0x7F) << 7) ));
        __half2 B23 = make_half2(__short_as_half( (((bb >> 16) & 0x80) << 8) | (((bb >> 16) & 0x7F) << 7) ), 
                                __short_as_half( (((bb >> 24) & 0x80) << 8) | (((bb >> 24) & 0x7F) << 7) ));        
#pragma unroll
        for (int x = 0; x < PART; x++) {
            regA.in = *reinterpret_cast<const uint2 *>(A + x * m + i);
#if (CUDART_VERSION < 12000) || defined(CUDA_NO_TENSOR_CORE)
            sdata[x][tid] += ((float)regA.out[0] * (float)B01.x + 
                                (float)regA.out[1] * (float)B01.y +
                                (float)regA.out[2] * (float)B23.x +
                                (float)regA.out[3] * (float)B23.y) * curScale;
#else
            __half2 p01 = __hmul2(regA.out2[0], B01); // {a0b0, a1b1}
            __half2 p23 = __hmul2(regA.out2[1], B23); // {a2b2, a3b3}
            __half2 sum_halves_vec = __hadd2(p01, p23); // {a0b0+a2b2, a1b1+a3b3}
            __half sum_h = __hadd(sum_halves_vec.x, sum_halves_vec.y); // (a0b0+a2b2) + (a1b1+a3b3)
            sdata[x][tid] += __half2float(sum_h) * curScale;
#endif
        }
    }
    __syncthreads();

    float diff = 0.0f;
    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            #pragma unroll
            for (int x = 0; x < PART; x++) {
                float other = sdata[x][tid + s] - diff;
                float sumTmp = sdata[x][tid] + other;
                diff = (sumTmp - sdata[x][tid]) - other;
                sdata[x][tid] = sumTmp;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (bias == nullptr) {
            for (int x = 0; x < PART; x++) C[st + k * x] = (half)(sdata[x][0] * magicScaleConstant);
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++) C[st + k * x] = (half)(sdata[x][0] * magicScaleConstant + (float)bias[st]);
        }
    }
    __syncthreads();
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvInt4NoZeroKernel1MultiRow(float *A, uint8_t *B, float *C,
                                                     float *bias, float *scales, float *mins,
                                                     int m, int k) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    // 1. 计算
    int st = blockIdx.x;
    int p = st;
#pragma unroll
    for (int x = 0; x < PART; x++) sdata[x][tid] = 0;

    const uint8_t *baseB = B + p * m / 2;
    float minv = __ldg(mins + p) / __ldg(scales + p);
    for (int i = tid * 2; i < m / 2; i += THREAD_PER_BLOCK * 2) {
        uint16_t bBuffer = *reinterpret_cast<const uint16_t *>(baseB + i);
#pragma unroll
        for (int x = 0; x < PART; x++) {
            float4 aBuffer = FETCH_FLOAT4(A[i * 2 + x * m]);
            sdata[x][tid] += aBuffer.x * (minv + ((bBuffer >> 4) & 15)) + aBuffer.y * (minv + (bBuffer & 15));
            sdata[x][tid] += aBuffer.z * (minv + (bBuffer >> 12)) + aBuffer.w * (minv + ((bBuffer >> 8) & 15));
        }
    }
    __syncthreads();

    float diff = 0.0f;
    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            #pragma unroll
            for (int x = 0; x < PART; x++) {
                float other = sdata[x][tid + s] - diff;
                float sumTmp = sdata[x][tid] + other;
                diff = (sumTmp - sdata[x][tid]) - other;
                sdata[x][tid] = sumTmp;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (bias == nullptr) {
            for (int x = 0; x < PART; x++) C[p + k * x] = sdata[x][0] * scales[p];
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++) C[p + k * x] = sdata[x][0] * scales[p] + bias[p];
        }
    }
    __syncthreads();
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvFp16Int4NoZeroKernel1MultiRow(half *A, uint8_t *B, half *C,
                                                     half *bias, float *scales, float *mins,
                                                     int m, int k) {
    __shared__ float sdata[PART][THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    // 1. 计算
    int st = blockIdx.x;
    int p = st;
#pragma unroll
    for (int x = 0; x < PART; x++) sdata[x][tid] = 0;

    union_char4 bBuffer;
    float minv = __ldg(mins + p) / __ldg(scales + p);

    for (int i = tid; i < m / 8; i += THREAD_PER_BLOCK) {
        bBuffer.in = *reinterpret_cast<const uint32_t *>(B + st * m / 2 + i * 4);
        // uint8_t now0 = B[st * m / 2 + i * 4];
        // uint8_t now1 = B[st * m / 2 + i * 4 + 1];
        // uint8_t now2 = B[st * m / 2 + i * 4 + 2];
        // uint8_t now3 = B[st * m / 2 + i * 4 + 3];
        for (int x = 0; x < PART; x++) {
            union_half8 aBuffer;
            aBuffer.in = *reinterpret_cast<const uint4 *>(A + x * m + i * 8);
            sdata[x][tid] += (__low2float(aBuffer.out2[0]) * (minv + (bBuffer.out[0] >> 4)) 
                         + __high2float(aBuffer.out2[0]) * (minv + (bBuffer.out[0] & 15)));
            sdata[x][tid] += (__low2float(aBuffer.out2[1]) * (minv + (bBuffer.out[1] >> 4)) 
                         + __high2float(aBuffer.out2[1]) * (minv + (bBuffer.out[1] & 15)));
            sdata[x][tid] += (__low2float(aBuffer.out2[2]) * (minv + (bBuffer.out[2] >> 4)) 
                         + __high2float(aBuffer.out2[2]) * (minv + (bBuffer.out[2] & 15)));
            sdata[x][tid] += (__low2float(aBuffer.out2[3]) * (minv + (bBuffer.out[3] >> 4)) 
                         + __high2float(aBuffer.out2[3]) * (minv + (bBuffer.out[3] & 15)));
        }
    }
    __syncthreads();

    float diff = 0.0f;
    for (unsigned int s = THREAD_PER_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            #pragma unroll
            for (int x = 0; x < PART; x++) {
                float other = sdata[x][tid + s] - diff;
                float sumTmp = sdata[x][tid] + other;
                diff = (sumTmp - sdata[x][tid]) - other;
                sdata[x][tid] = sumTmp;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (bias == nullptr) {
#pragma unroll
            for (int x = 0; x < PART; x++) C[p + k * x] = (half)(sdata[x][0] * scales[p]);
        } else {
#pragma unroll
            for (int x = 0; x < PART; x++) C[p + k * x] = (half)(sdata[x][0] * scales[p] + float(bias[p]));
        }
    }
    __syncthreads();
}

template <int THREAD_PER_BLOCK, int PART>
__global__ void FastllmGemvInt4NoZeroKernel1(float *A, uint8_t *B, float *C,
                                             float *bias, float *scales, float *mins,
                                             int m, int k) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    // 1. 计算
    int st = blockIdx.x * PART;
    int end = st + PART;
    for (int p = st; p < end; p++) {
        sdata[tid] = 0;
        const uint8_t *baseB = B + p * m / 2;
        float minv = __ldg(mins + p) / __ldg(scales + p);
        for (int i = tid * 2; i < m / 2; i += THREAD_PER_BLOCK * 2) {
            float4 aBuffer = FETCH_FLOAT4(A[i * 2]);
            uint16_t bBuffer = *reinterpret_cast<const uint16_t *>(baseB + i);
            sdata[tid] += aBuffer.x * (minv + ((bBuffer >> 4) & 15)) + aBuffer.y * (minv + (bBuffer & 15));
            sdata[tid] += aBuffer.z * (minv + (bBuffer >> 12)) + aBuffer.w * (minv + ((bBuffer >> 8) & 15));
        }
        __syncthreads();

        float diff = 0.0f;
        for (unsigned int s = THREAD_PER_BLOCK/2; s > 0; s >>= 1) {
            if (tid < s) {
                float other = sdata[tid + s] - diff;
                float sumTmp = sdata[tid] + other;
                diff = (sumTmp - sdata[tid]) - other;
                sdata[tid] = sumTmp;
            }
            __syncthreads();
        }
        //if (tid <= 32)
            //warpReduce(sdata, tid);
        if (tid == 0) {
            if (bias == nullptr) {
                C[p] = sdata[0] * scales[p];
            } else {
                C[p] = sdata[0] * scales[p] + bias[p];
            }
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
__global__ void FastllmHalfMatMulTransBBatchKernel(uint8_t** pointer, float alpha) {
    int id = blockIdx.x;
    half *input0 = (half*)pointer[id * 8 + 0];
    half *input1 = (half*)pointer[id * 8 + 1];
    half *output = (half*)pointer[id * 8 + 2];
    int n = (int)((size_t)pointer[id * 8 + 3]);
    int m = (int)((size_t)pointer[id * 8 + 4]);
    int k = (int)((size_t)pointer[id * 8 + 5]);
    int input0Stride = (int)((size_t)pointer[id * 8 + 6]);
    int input1Stride = (int)((size_t)pointer[id * 8 + 7]);

    int tid = threadIdx.x;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 
    if (m == 128) {
        int wid = tid >> 5;
        int perN = 8, perK = 128;

        const int BN = 8, BK = 128;
        __shared__ float curC[BN][BK];
        half hscale = (half)alpha;

        for (int stN = 0; stN < n; stN += perN) {
            int endN = min(n, stN + perN);
            for (int stK = 0; stK < k; stK += perK) {
                int endK = min(k, stK + perK);
                wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a[8];
                wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> frag_b[8];
                wmma::fragment<wmma::accumulator, 8, 32, 16, float> frag_c;

                wmma::fill_fragment(frag_c, 0.0);
                __syncthreads();

                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    wmma::load_matrix_sync(frag_a[j], &input0[(stN) * input0Stride + j * 16], input0Stride);
                }
                __syncthreads();

                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    wmma::load_matrix_sync(frag_b[j], &input1[(stK + wid * 32) * input1Stride + j * 16], input1Stride);
                }
                __syncthreads();

                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    wmma::mma_sync(frag_c, frag_a[j], frag_b[j], frag_c);
                }
                __syncthreads();

                wmma::store_matrix_sync(&curC[0][wid * 32], frag_c, BK, wmma::mem_row_major);
                __syncthreads();

                if (stK + tid < endK) {
                    for (int i = 0; stN + i < endN; i++) {
                        output[(stN + i) * k + stK + tid] = (half)(curC[i][tid] * alpha);
                    }
                }
                __syncthreads();
            }
        }
        return;
    }
#endif
    int pera = 4, perb = 4;
    half cura[4][4], curb[4][4];
    float curc[4][4];
    int cnta = (n - 1) / pera + 1, cntb = (k - 1) / perb + 1;
    for (int taskId = tid; taskId < cnta * cntb; taskId += THREAD_PER_BLOCK) {
        int taska = taskId / cntb, taskb = taskId % cntb;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                curc[i][j] = 0.0f;
            }
        }
        for (int l = 0; l < m; l += 4) {
            for (int a = taska * pera; a < (taska + 1) * pera && a < n; a++) {
                FETCH_FLOAT2(cura[a - taska * pera]) = FETCH_FLOAT2(input0[a * input0Stride + l]);
            }
            for (int b = taskb * perb; b < (taskb + 1) * perb && b < k; b++) {
                FETCH_FLOAT2(curb[b - taskb * perb]) = FETCH_FLOAT2(input1[b * input1Stride + l]);
            }

            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        curc[i][j] += (float)cura[i][k] * (float)curb[j][k];
                    }
                }
            }
        }

        if ((taska + 1) * pera <= n && (taskb + 1) * perb <= k) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    output[(taska * pera + i) * k + (taskb * perb + j)] = (half)(curc[i][j] * alpha);
                }
            }
        } else {
            for (int i = 0; i < pera && taska * pera + i < n; i++) {
                for (int j = 0; j < perb && taskb * perb + j < k; j++) {
                    output[(taska * pera + i) * k + (taskb * perb + j)] = (half)(curc[i][j] * alpha);
                }
            }
        }
    }
/*
    int tid = threadIdx.x;
    for (int i = 0; i < n; i++) {
        half *curInput0 = input0 + i * input0Stride;
        for (int j = tid; j < k; j += THREAD_PER_BLOCK) {
            half *curInput1 = input1 + j * input1Stride;
            float sum = 0.0;
            for (int l = 0; l < m; l++) {
                sum += (float)curInput0[l] * (float)curInput1[l];
            }
            output[i * k + j] = (half)(sum * alpha);
        }
    }
*/
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
__global__ void FastllmHalfMatMulKernel(uint8_t** pointer, float alpha) {
    int id = blockIdx.x;
    half *input0 = (half*)pointer[id * 8 + 0];
    half *input1 = (half*)pointer[id * 8 + 1];
    half *output = (half*)pointer[id * 8 + 2];
    int n = (int)((size_t)pointer[id * 8 + 3]);
    int m = (int)((size_t)pointer[id * 8 + 4]);
    int k = (int)((size_t)pointer[id * 8 + 5]);
    int input0Stride = (int)((size_t)pointer[id * 8 + 6]);
    int input1Stride = (int)((size_t)pointer[id * 8 + 7]);
    int tid = threadIdx.x;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 
    if (k == 128) {
        int wid = tid >> 5;
        int perN = 8, perM = 128;
        for (int i = 0; i < n; i++) {
            output[i * k + tid] = (half)0;
        }

        __shared__ half curA[8][128];
        __shared__ float curC[8][128];

        for (int stN = 0; stN < n; stN += perN) {
            int endN = min(stN + perN, n);
            wmma::fragment<wmma::accumulator, 8, 32, 16, float> frag_c;
            wmma::fill_fragment(frag_c, 0.0);

            for (int stM = 0; stM < m; stM += perM) {
                int endM = min(stM + perM, m);
                if (stM + tid < m) {
                    for (int i = 0; stN + i < endN; i++) {
                        curA[i][tid] = input0[(stN + i) * input0Stride + stM + tid];
                    }
                } else {
                    for (int i = 0; stN + i < endN; i++) {
                        curA[i][tid] = (half)0.0;
                    }
                }

                wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a[8];
                wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> frag_b[8];
                __syncthreads();

                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    wmma::load_matrix_sync(frag_a[j], &curA[0][16 * j], 128);
                }
                __syncthreads();

                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    wmma::load_matrix_sync(frag_b[j], &input1[(stM + 16 * j) * input1Stride + wid * 32], input1Stride);
                }
                __syncthreads();

                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    wmma::mma_sync(frag_c, frag_a[j], frag_b[j], frag_c);
                }
                __syncthreads();
            }
            wmma::store_matrix_sync(&curC[0][wid * 32], frag_c, 128, wmma::mem_row_major);
            __syncthreads();

            for (int i = 0; stN + i < endN; i++) {
                output[(stN + i) * k + tid] = (half)((float)output[(stN + i) * k + tid] + (float)curC[i][tid] * alpha);
            }
            __syncthreads();
        }
        return;
    }
#endif
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
                    cura[a - taska * pera][x] = (l + x < m ? (float)input0[a * input0Stride + l + x] : 0.f);
                }
            }
            for (int b = taskb * perb; b < (taskb + 1) * perb && b < k; b++) {
#pragma unroll
                for (int x = 0; x < 4; x++) {
                    curb[b - taskb * perb][x] = (l + x < m ? (float)input1[(l + x) * input1Stride + b] : 0.f);
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
                    output[(taska * pera + i) * k + (taskb * perb + j)] = (half)(curc[i][j] * alpha);
                }
            }
        } else {
            for (int i = 0; i < pera && taska * pera + i < n; i++) {
                for (int j = 0; j < perb && taskb * perb + j < k; j++) {
                    output[(taska * pera + i) * k + (taskb * perb + j)] = (half)(curc[i][j] * alpha);
                }
            }
        }
    }
/*
    for (int i = 0; i < n; i++) {
        half *curInput0 = input0 + i * input0Stride;
        for (int j = tid; j < k; j += THREAD_PER_BLOCK) {
            half *curInput1 = input1 + j;
            float sum = 0.0;
            for (int l = 0; l < m; l++) {
                sum += (float)curInput0[l] * (float)curInput1[l * input1Stride];
            }
            output[i * k + j] = (half)(sum * alpha);
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

void LaunchFastllmGemmFp32Int8(float *input, uint8_t *weight, float *output, float *bias, float *scales, uint8_t *zeros, int n, int m, int k) {
    for (int i = 0; i < n; i++) {
        FastllmGemvInt8Kernel2<256, 1> <<< k, 256 >>>(input + i * m, weight, output + i * k, bias, scales, zeros, m, k);
    }
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
            zeropoints[i] = weight.zeros[i];
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
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
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
#ifdef CUDA_NO_TENSOR_CORE
        cudaFp16Input = (half *) FastllmCudaMalloc(n * m * sizeof(half));
        cudaFp16Weight = (half *) FastllmCudaMalloc(k * m * sizeof(half));
        float h_alpha = 1.0, h_beta = 0.0;
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_32F, ComputeType = CUDA_R_32F;
#else
        cudaFp16Input = (half *) FastllmCudaMalloc(n * m * sizeof(half));
        cudaFp16Output = (half *) FastllmCudaMalloc(n * k * sizeof(half));
        cudaFp16Weight = (half *) FastllmCudaMalloc(k * m * sizeof(half));

        __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
#endif
        cublasStatus_t status;

        int len = n * m;
        int threadPerBlock = std::min(256, len);
        FastllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaFp16Input, len);

        len = k * m;
#ifdef CUDA_NO_TENSOR_CORE
        int gridSize = (len - 1) / (threadPerBlock * ST128_FP16_COUNT) + 1;
        FastllmCudaInt82HalfKernel <<< gridSize, threadPerBlock>>>((uint8_t*)weight.cudaData,
                                                                   cudaScales,
                                                                   cudaZeropoints,
                                                                   cudaFp16Weight, len, m);

        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              k, n, m,
                              &h_alpha, cudaFp16Weight, AType,
                              m, cudaFp16Input, BType,
                              m, &h_beta,
                              cudaOutput, CType,
                              k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#else
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
#endif
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw("cublas error");
            exit(0);
        }

        len = n * k;
#ifdef CUDA_NO_TENSOR_CORE
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, cudaBiasData, k);
        }
        FastllmCudaFree(cudaFp16Input);
        FastllmCudaFree(cudaFp16Weight);
#else
        FastllmCudaHalf2FloatKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaFp16Output, cudaOutput, len);
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, cudaBiasData, k);
        }

        FastllmCudaFree(cudaFp16Input);
        FastllmCudaFree(cudaFp16Output);
        FastllmCudaFree(cudaFp16Weight);
#endif
    } else {
        LaunchFastllmGemmFp32Int8(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, cudaBiasData, cudaScales, cudaZeropoints, n, m, k);
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

void LaunchFastllmGemvInt4Kernel2(float *input, uint8_t *weight, float *output, float *bias, float *scales, uint8_t *zeros, int n, int m, int k) {
    for (int i = 0; i < n; i++) {
        FastllmGemvInt4Kernel2<256, 1> <<< k, 256 >>>(input + i * m, weight, output + i * k, bias, scales, zeros, m, k);
    }
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
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaData.push_back((void*)cudaBiasData);
    }

    float *cudaScales = (float*)weight.extraCudaData[0];
    uint8_t *cudaZeropoints = (uint8_t*)weight.extraCudaData[1];
    float *cudaBiasData = (float*)weight.extraCudaData[2];

    float *cudaInput = (float*)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float*)FastllmCudaPrepareOutput(output);
    LaunchFastllmGemvInt4Kernel2(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, cudaBiasData, cudaScales, cudaZeropoints, n, m, k);

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

void LaunchFastllmGemmFp32Int4Group(float *input, uint8_t *weight, float *output, float *bias, half *scales, half *mins, int n, int m, int k, int group, int groupCnt) {
    for (int i = 0; i < n; i++) {
#ifdef CUDA_NO_TENSOR_CORE
        FastllmGemvInt4GroupKernel3<64, 4> <<< k / 4, 64 >>>(input + i * m, weight, output + i * k, bias, scales, mins, m, k, group, groupCnt);
#else
        FastllmGemvInt4GroupKernel2<64, 4> <<< k / 4, 64 >>>(input + i * m, weight, output + i * k, bias, scales, mins, m, k, group, groupCnt);
#endif
    }
}

bool FastllmCudaMatMulFloatInt4Group(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, 
                                    int n, int m, int k) {
    int group = weight.group, groupCnt = weight.groupCnt;
    if (weight.cudaData == nullptr || weight.extraCudaData.size() == 0) {
        half *cudaScales;
        cudaError_t state = cudaSuccess;
        state = cudaMalloc(&cudaScales, k * group * sizeof(half));
        half *scales = new half[k * group];
        for (int i = 0; i < k * group; i++) {
            scales[i] = (half)weight.scales[i];
        }
        state = cudaMemcpy(cudaScales, scales, k * group * sizeof(half), cudaMemcpyHostToDevice);
        weight.extraCudaData.push_back((void*)cudaScales);
        delete[] scales;

        half *cudaMins;
        state = cudaMalloc(&cudaMins, k * group * sizeof(half));
        half *mins = new half[k * group];
        for (int i = 0; i < k * group; i++) {
            mins[i] = (half)weight.mins[i];
        }
        state = cudaMemcpy(cudaMins, mins, k * group * sizeof(half), cudaMemcpyHostToDevice);
        delete[] mins;
        weight.extraCudaData.push_back((void*)cudaMins);

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

    half *cudaScales = (half*)weight.extraCudaData[0];
    half *cudaMins = (half*)weight.extraCudaData[1];
    float *cudaBiasData = (float*)weight.extraCudaData[2];

    float *cudaInput = (float*)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float*)FastllmCudaPrepareOutput(output);
    if (n >= 8) {
        auto fastllmCublasHandle = getFastllmCublasHandle();
        half *cudaFp16Input, *cudaFp16Output, *cudaFp16Weight;
#ifdef CUDA_NO_TENSOR_CORE
        cudaFp16Input = (half *) FastllmCudaMalloc(n * m * sizeof(half));
        cudaFp16Weight = (half *) FastllmCudaMalloc(k * m * sizeof(half));

        float h_alpha = 1.0, h_beta = 0.0;
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_32F, ComputeType = CUDA_R_32F;
#else
        cudaFp16Input = (half *) FastllmCudaMalloc(n * m * sizeof(half));
        cudaFp16Output = (half *) FastllmCudaMalloc(n * k * sizeof(half));
        cudaFp16Weight = (half *) FastllmCudaMalloc(k * m * sizeof(half));

        __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
#endif
        cublasStatus_t status;

        int len = n * m;
        int threadPerBlock = std::min(256, len);
        FastllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaFp16Input,
                                                                                          len);

        len = k * m;
        FastllmCudaInt4Group2HalfKernel <<< k, 64 >>>((uint8_t*)weight.cudaData, cudaScales, cudaMins, cudaFp16Weight, k, m, group, groupCnt);

#ifdef CUDA_NO_TENSOR_CORE
        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              k, n, m,
                              &h_alpha, cudaFp16Weight, AType,
                              m, cudaFp16Input, BType,
                              m, &h_beta,
                              cudaOutput, CType,
                              k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#else
        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              k, n, m,
                              &h_alpha, cudaFp16Weight, AType,
                              m, cudaFp16Input, BType,
                              m, &h_beta,
                              cudaFp16Output, CType,
                              k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#endif
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error. status = %d\n", status);
            throw("cublas error");
            exit(0);
        }

        len = n * k;
#ifdef CUDA_NO_TENSOR_CORE
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<< n, 256 >>>(cudaOutput, cudaBiasData, k);
        }
        FastllmCudaFree(cudaFp16Input);
        FastllmCudaFree(cudaFp16Weight);
#else
        FastllmCudaHalf2FloatKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaFp16Output, cudaOutput,
                                                                                           len);
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<< n, 256 >>>(cudaOutput, cudaBiasData, k);
        }

        FastllmCudaFree(cudaFp16Input);
        FastllmCudaFree(cudaFp16Output);
        FastllmCudaFree(cudaFp16Weight);
#endif
    } else {
        LaunchFastllmGemmFp32Int4Group(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, cudaBiasData, cudaScales, cudaMins, n, m, k, group, groupCnt);
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

void LaunchFastllmGemmFp32FP8E4M3(float *input, uint8_t *weight, float *output, float *bias, float *scales, int n, int m, int k, int blockM, int blockK) {
    if (n == 1) {
        FastllmGemvFP8E4M3Kernel1MultiRow<64, 1> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 2) {
        FastllmGemvFP8E4M3Kernel1MultiRow<64, 2> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 3) {
        FastllmGemvFP8E4M3Kernel1MultiRow<64, 3> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 4) {
        FastllmGemvFP8E4M3Kernel1MultiRow<64, 4> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 5) {
        FastllmGemvFP8E4M3Kernel1MultiRow<64, 5> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 6) {
        FastllmGemvFP8E4M3Kernel1MultiRow<64, 6> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 7) {
        FastllmGemvFP8E4M3Kernel1MultiRow<64, 7> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else {
        int i = 0; 
        for (; i + 7 < n; i += 8) {
            FastllmGemvFP8E4M3Kernel1MultiRow<64, 8> <<< k, 64 >>>(input + i * m, weight, output + i * k, bias, scales, m, k, blockM, blockK);
        }
        for (; i < n; i++) {
            FastllmGemvFP8E4M3Kernel1MultiRow<64, 1> <<< k, 64 >>>(input + i * m, weight, output + i * k, bias, scales, m, k, blockM, blockK);
        }
        return;
    }
}

void LaunchFastllmGemmFp16FP8E4M3(half *input, uint8_t *weight, half *output, half *bias, float *scales, int n, int m, int k, int blockM, int blockK) {
    if (n == 1) {
        FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 1> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 2) {
        FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 2> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 3) {
        FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 3> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 4) {
        FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 4> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 5) {
        FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 5> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 6) {
        FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 6> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 7) {
        FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 7> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 8) {
        FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 8> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 9) {
        FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 9> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 10) {
        FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 10> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 11) {
        FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 11> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 12) {
        FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 12> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 13) {
        FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 13> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 14) {
        FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 14> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else if (n == 15) {
        FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 15> <<< k, 64 >>>(input, weight, output, bias, scales, m, k, blockM, blockK);
    } else {
        int i = 0; 
        for (; i + 15 < n; i += 16) {
            FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 16> <<< k, 64 >>>(input + i * m, weight, output + i * k, bias, scales, m, k, blockM, blockK);
        }
        for (; i + 7 < n; i += 8) {
            FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 8> <<< k, 64 >>>(input + i * m, weight, output + i * k, bias, scales, m, k, blockM, blockK);
        }
        for (; i + 3 < n; i += 4) {
            FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 4> <<< k, 64 >>>(input + i * m, weight, output + i * k, bias, scales, m, k, blockM, blockK);
        }
        for (; i < n; i++) {
            FastllmGemvHalfFP8E4M3Kernel1MultiRow<64, 1> <<< k, 64 >>>(input + i * m, weight, output + i * k, bias, scales, m, k, blockM, blockK);
        }
        return;
    }
}

void LaunchFastllmGemmFp32Int4NoZero(float *input, uint8_t *weight, float *output, float *bias, float *scales, float *mins, int n, int m, int k) {
   /* for (int i = 0; i < n; i++) {
        FastllmGemvInt4NoZeroKernel1<64, 1> <<< k, 64 >>>(input + i * m, weight, output + i * k, bias, scales, mins, m, k);
    }
    return;*/
    if (n == 1) {
        FastllmGemvInt4NoZeroKernel1MultiRow<64, 1> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k);
    } else if (n == 2) {
        FastllmGemvInt4NoZeroKernel1MultiRow<64, 2> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k);
    } else if (n == 3) {
        FastllmGemvInt4NoZeroKernel1MultiRow<64, 3> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k);
    } else if (n == 4) {
        FastllmGemvInt4NoZeroKernel1MultiRow<64, 4> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k);
    } else if (n == 5) {
        FastllmGemvInt4NoZeroKernel1MultiRow<64, 5> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k);
    } else if (n == 6) {
        FastllmGemvInt4NoZeroKernel1MultiRow<64, 6> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k);
    } else if (n == 7) {
        FastllmGemvInt4NoZeroKernel1MultiRow<64, 7> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k);
    } else {
        for (int i = 0; i < n; i++) {
            FastllmGemvInt4NoZeroKernel1<64, 1> <<< k, 64 >>>(input + i * m, weight, output + i * k, bias, scales, mins, m, k);
        }
        return;
    }
}

bool FastllmCudaMatMulFloatFP8E4M3(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    if (weight.cudaData == nullptr || weight.extraCudaData.size() == 0) {
        float *cudaScales;
        cudaError_t state = cudaSuccess;
        state = cudaMalloc(&cudaScales, weight.scales.size() * sizeof(float));
        state = cudaMemcpy(cudaScales, weight.scales.data(), weight.scales.size() * sizeof(float), cudaMemcpyHostToDevice);
        weight.extraCudaData.push_back((void*)cudaScales);

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

    float *cudaScales = (float*)weight.extraCudaData[0];
    float *cudaBiasData = (float*)weight.extraCudaData[1];

    float *cudaInput = (float*)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float*)FastllmCudaPrepareOutput(output);

    if (n >= 16) {
        auto fastllmCublasHandle = getFastllmCublasHandle();
        half *cudaFp16Input, *cudaFp16Output, *cudaFp16Weight;

        cudaFp16Input = (half *) FastllmCudaMalloc(n * m * sizeof(half));
        cudaFp16Output = (half *) FastllmCudaMalloc(n * k * sizeof(half));
        cudaFp16Weight = (half *) FastllmCudaMalloc(k * m * sizeof(half));

        __half h_alpha = __float2half_rn(exp2f(8.0f)), h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
        cublasStatus_t status;

        int len = n * m;
        int threadPerBlock = std::min(256, len);
        FastllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaFp16Input,
                                                                                          len);

        len = k * m;
        FastllmCudaFP8E4M32HalfKernel <<< k, 256 >>>((uint8_t*)weight.cudaData, cudaScales, cudaFp16Weight, k, m, weight.blockK, weight.blockM);
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
        FastllmCudaHalf2FloatKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaFp16Output, cudaOutput,
                                                                                           len);
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<< n, 256 >>>(cudaOutput, cudaBiasData, k);
        }

        FastllmCudaFree(cudaFp16Input);
        FastllmCudaFree(cudaFp16Output);
        FastllmCudaFree(cudaFp16Weight);
    } else {
        LaunchFastllmGemmFp32FP8E4M3(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, cudaBiasData, cudaScales, n, m, k, weight.blockM, weight.blockK);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaHalfMatMulFloatFP8E4M3(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    if (weight.cudaData == nullptr || 
        (weight.extraCudaHalfData.size() == 0 && bias.dims.size() > 0)) {
        half *cudaBiasData;
        cudaError_t state = cudaSuccess;
        state = cudaMalloc(&cudaBiasData, k * sizeof(half));
        if (bias.dims.size() > 0) {
            float *tempBiasData;
            state = cudaMalloc(&tempBiasData, k * sizeof(float));
            state = cudaMemcpy(tempBiasData, (uint8_t *) bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
            int threadPerBlock = std::min(256, k);
            FastllmCudaFloat2HalfKernel <<< (k - 1) / threadPerBlock + 1, threadPerBlock>>>(tempBiasData, cudaBiasData, k);
            state = cudaFree(tempBiasData);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(half));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaHalfData.push_back((void *) cudaBiasData);
    }

    float *cudaScales = (float*)weight.extraCudaData[0];
    half *cudaBiasData = bias.dims.size() == 0 ? nullptr : (half *) weight.extraCudaHalfData[0];
    half *cudaInput = (half*)FastllmCudaPrepareInput(input);
    half *cudaOutput = (half*)FastllmCudaPrepareOutput(output);

    if (n >= 32) {
        auto fastllmCublasHandle = getFastllmCublasHandle();
        half *cudaFp16Weight;

        cudaFp16Weight = (half *) FastllmCudaMalloc(k * m * sizeof(half));

        __half h_alpha = __float2half_rn(exp2f(8.0f)); // fp8 -> fp16的转换系数  
        __half h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
        cublasStatus_t status;

        int len = n * m;
        int threadPerBlock = std::min(256, len);

        len = k * m;

        FastllmCudaFP8E4M32HalfKernel <<< k, 256 >>>((uint8_t*)weight.cudaData, cudaScales, cudaFp16Weight, k, m, weight.blockK, weight.blockM);

        status = cublasGemmEx(fastllmCublasHandle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                k, n, m,
                                &h_alpha, cudaFp16Weight, AType,
                                m, cudaInput, BType,
                                m, &h_beta,
                                cudaOutput, CType,
                                k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));

        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw("cublas error");
            exit(0);
        }
        if (bias.dims.size() > 0) {
            half *cudaBiasData = (half*)weight.extraCudaHalfData[0];
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, cudaBiasData, k);
        }

        FastllmCudaFree(cudaFp16Weight);
    } else {
        LaunchFastllmGemmFp16FP8E4M3(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, cudaBiasData, cudaScales, n, m, k, weight.blockM, weight.blockK);
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
            mins[i] = weight.mins[i];
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
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaData.push_back((void*)cudaBiasData);
    }

    float *cudaScales = (float*)weight.extraCudaData[0];
    float *cudaMins = (float*)weight.extraCudaData[1];
    float *cudaBiasData = (float*)weight.extraCudaData[2];

    float *cudaInput = (float*)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float*)FastllmCudaPrepareOutput(output);

    if (n >= 16) {
        auto fastllmCublasHandle = getFastllmCublasHandle();
        half *cudaFp16Input, *cudaFp16Output, *cudaFp16Weight;
#ifdef CUDA_NO_TENSOR_CORE
        cudaFp16Input = (half *) FastllmCudaMalloc(n * m * sizeof(half));
        cudaFp16Weight = (half *) FastllmCudaMalloc(k * m * sizeof(half));

        float h_alpha = 1.0, h_beta = 0.0;
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_32F, ComputeType = CUDA_R_32F;
#else
        cudaFp16Input = (half *) FastllmCudaMalloc(n * m * sizeof(half));
        cudaFp16Output = (half *) FastllmCudaMalloc(n * k * sizeof(half));
        cudaFp16Weight = (half *) FastllmCudaMalloc(k * m * sizeof(half));

        __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
#endif
        cublasStatus_t status;

        int len = n * m;
        int threadPerBlock = std::min(256, len);
        FastllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaFp16Input,
                                                                                          len);

        len = k * m;
        int gridSize = (len - 1) / (threadPerBlock * 4) + 1;
        FastllmCudaInt42HalfKernel <<< gridSize, threadPerBlock>>>((uint8_t *) weight.cudaData,
                                                                   cudaScales, cudaMins,
                                                                   cudaFp16Weight, len, m);

#ifdef CUDA_NO_TENSOR_CORE
        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              k, n, m,
                              &h_alpha, cudaFp16Weight, AType,
                              m, cudaFp16Input, BType,
                              m, &h_beta,
                              cudaOutput, CType,
                              k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#else
        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              k, n, m,
                              &h_alpha, cudaFp16Weight, AType,
                              m, cudaFp16Input, BType,
                              m, &h_beta,
                              cudaFp16Output, CType,
                              k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#endif
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw("cublas error");
            exit(0);
        }

        len = n * k;
#ifdef CUDA_NO_TENSOR_CORE
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<< n, 256 >>>(cudaOutput, cudaBiasData, k);
        }
        FastllmCudaFree(cudaFp16Input);
        FastllmCudaFree(cudaFp16Weight);
#else
        FastllmCudaHalf2FloatKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaFp16Output, cudaOutput,
                                                                                           len);
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<< n, 256 >>>(cudaOutput, cudaBiasData, k);
        }

        FastllmCudaFree(cudaFp16Input);
        FastllmCudaFree(cudaFp16Output);
        FastllmCudaFree(cudaFp16Weight);
#endif
    } else {
        LaunchFastllmGemmFp32Int4NoZero(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, cudaBiasData, cudaScales, cudaMins, n, m, k);
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
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
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
    if (weight.cudaData == nullptr || weight.extraCudaData.size() == 0) {
        float *cudaBiasData;
        cudaError_t state = cudaSuccess;
        state = cudaMalloc(&cudaBiasData, k * sizeof(float));
        if (bias.dims.size() > 0) {
            state = cudaMemcpy(cudaBiasData, (uint8_t*)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(float));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaData.push_back((void*)cudaBiasData);
    }

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

void LaunchFastllmGemmFp32Fp16(float *input, half *weight, float *output, float *bias, int n, int m, int k) {
    if (n == 1) {
        FastllmGemvFp32Fp16Kernel2MultiRow<256, 1> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else if (n == 2) {
        FastllmGemvFp32Fp16Kernel2MultiRow<256, 2> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else if (n == 3) {
        FastllmGemvFp32Fp16Kernel2MultiRow<256, 3> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else if (n == 4) {
        FastllmGemvFp32Fp16Kernel2MultiRow<256, 4> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else if (n == 5) {
        FastllmGemvFp32Fp16Kernel2MultiRow<256, 5> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else if (n == 6) {
        FastllmGemvFp32Fp16Kernel2MultiRow<256, 6> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else if (n == 7) {
        FastllmGemvFp32Fp16Kernel2MultiRow<256, 7> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else {
        for (int i = 0; i < n; i++) {
            FastllmGemvFp32Fp16Kernel2MultiRow<256, 1> <<< k, 256 >>>(input + i * m, weight, output + i * k, bias, m, k);
        }
        return;

        printf("Error: LaunchFastllmGemmFp32Fp16: n > 7.\n");
        exit(0);
    }
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
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaData.push_back((void*)cudaBiasData);
    }
    float *cudaBiasData = (float*)weight.extraCudaData[0];
    float *cudaInput = (float*)FastllmCudaPrepareInput(input);
    float *cudaOutput = (float*)FastllmCudaPrepareOutput(output);

    if (n < 8) {
        LaunchFastllmGemmFp32Fp16(cudaInput, (half*)weight.cudaData, cudaOutput, cudaBiasData, n, m, k);
    } else {
        auto fastllmCublasHandle = getFastllmCublasHandle();
        //cudaDeviceSynchronize();
        half *cudaFp16Input, *cudaFp16Output;
#ifdef CUDA_NO_TENSOR_CORE
        cudaFp16Input = (half *) FastllmCudaMalloc(n * m * sizeof(half));

        float h_alpha = 1.0, h_beta = 0.0;
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_32F, ComputeType = CUDA_R_32F;
#else
        cudaFp16Input = (half *) FastllmCudaMalloc(n * m * sizeof(half));
        cudaFp16Output = (half *) FastllmCudaMalloc(n * k * sizeof(half));

        __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
#endif
        cublasStatus_t status;

        int len = n * m;
        int threadPerBlock = std::min(256, len);
        FastllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>(cudaInput, cudaFp16Input,
                                                                                          len);

#ifdef CUDA_NO_TENSOR_CORE
        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              k, n, m,
                              &h_alpha, (half *) weight.cudaData, AType,
                              m, cudaFp16Input, BType,
                              m, &h_beta,
                              cudaOutput, CType,
                              k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#else
        status = cublasGemmEx(fastllmCublasHandle,
                              CUBLAS_OP_T, CUBLAS_OP_N,
                              k, n, m,
                              &h_alpha, (half *) weight.cudaData, AType,
                              m, cudaFp16Input, BType,
                              m, &h_beta,
                              cudaFp16Output, CType,
                              k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#endif
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw("cublas error");
            exit(0);
        }

        len = n * k;
#ifdef CUDA_NO_TENSOR_CORE
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, (float*)weight.extraCudaData[0], k);
        }
        FastllmCudaFree(cudaFp16Input);
#else
        FastllmCudaHalf2FloatKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaFp16Output, cudaOutput,
                                                                                           len);

        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, (float*)weight.extraCudaData[0], k);
        }
        //cudaDeviceSynchronize();

        FastllmCudaFree(cudaFp16Input);
        FastllmCudaFree(cudaFp16Output);
#endif
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

bool FastllmCudaAttentionMask(fastllm::Data &input, const fastllm::Data &mask, float maskValue) {
    int spatial = input.Count(2), n = input.dims[0], m = input.dims[1];
    float *cudaData = (float *) FastllmCudaPrepareInput(input);
    float *maskData = (float *) FastllmCudaPrepareInput(mask);

    if (input.dataType == fastllm::DataType::FLOAT32) {
    FastllmAttentionMaskKernel <256> <<< n * m, 256>>>(cudaData, maskData, maskValue,
                                                       n, m, spatial);
    } else {
        FastllmAttentionMaskKernel <256> <<< n * m, 256>>>((half*)cudaData, (half*)maskData, __float2half(maskValue),
                                                        n, m, spatial);
    }
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


bool FastllmCudaSoftmax(const fastllm::Data &input, fastllm::Data &output, int axis) {
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareInput(output);

    int dimsLen = input.dims.size();
    axis = (axis % dimsLen + dimsLen) % dimsLen;
    int outer = input.Count(0) / input.Count(axis);
    int channels = input.dims[axis];
    int inner = input.Count(axis + 1);
    if (inner == 1) {
        if (input.dataType == fastllm::DataType::FLOAT32) {
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
            if (channels < 8) {
                FastllmSoftmaxKernelInner1 <1> <<< outer, 1 >>> ((half*)cudaInput, (half*)cudaOutput, outer, channels);
            } else if (channels < 64) {
                FastllmSoftmaxKernelInner1 <8> <<< outer, 8 >>> ((half*)cudaInput, (half*)cudaOutput, outer, channels);
            } else if (channels < 512) {
                FastllmSoftmaxKernelInner1 <64> <<< outer, 64 >>> ((half*)cudaInput, (half*)cudaOutput, outer, channels);
            } else {
                FastllmSoftmaxKernelInner1 <256> <<< outer, 256 >>> ((half*)cudaInput, (half*)cudaOutput, outer, channels);
            }
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
    FastllmSoftmaxKernelBatchInner1 <float, 256> <<<total, 256>>> (pointers);

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

bool FastllmCudaMLA(const fastllm::Data &qNope, const fastllm::Data &qPe, const fastllm::Data &kvCache, const fastllm::Data &peCache, 
    fastllm::Data &ss, fastllm::Data &output, float softmaxScale) {
    int b = qPe.dims[0], s = qPe.dims[1], h = qPe.dims[2], c = qNope.dims.back(), t = kvCache.dims[1], r = qPe.dims[3];
    auto fastllmCublasHandle = getFastllmCublasHandle();
    cublasStatus_t status;

    if (qNope.dataType == fastllm::DataType::FLOAT32) {
        float *score = (float*)FastllmCudaMalloc(b * s * h * t * sizeof(float));
        float alpha = softmaxScale, beta0 = 0.0f, beta1 = 1.0f;
        status = cublasSgemmStridedBatched(fastllmCublasHandle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            t, h, c, &alpha,
            (float*)peCache.cudaData, c, t * c,
            (float*)qNope.cudaData, c, h * c,
            &beta0,
            score, t, t * h, 1);
        status = cublasSgemmStridedBatched(fastllmCublasHandle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            t, h, r, &alpha,
            (float*)kvCache.cudaData, r, t * r,
            (float*)qPe.cudaData, r, h * r,
            &beta1,
            score, t, t * h, 1);        
        int outer = b * s * h, channels = t;
        FastllmSoftmaxKernelInner1 <64> <<< outer, 64 >>> (score, score, outer, channels);
        status = cublasSgemmStridedBatched(fastllmCublasHandle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    c, b * s * h, t, &beta1,
                    (float*)peCache.cudaData, c, t * c,
                    score, t, b * s * h * t,
                    &beta0,
                    (float*)output.cudaData, c, c * b * s * h, 1);
        FastllmCudaFree(score);
    } else if (qNope.dataType == fastllm::DataType::FLOAT16) {
        half *score = (half*)FastllmCudaMalloc(b * s * h * t * sizeof(half));
        half alpha = __float2half_rn(softmaxScale), beta0 = __float2half_rn(0.0f), beta1 = __float2half_rn(1.0f);
        status = cublasHgemmStridedBatched(fastllmCublasHandle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            t, h, c, &alpha,
            (half*)peCache.cudaData, c, t * c,
            (half*)qNope.cudaData, c, h * c,
            &beta0,
            score, t, t * h, 1);
        status = cublasHgemmStridedBatched(fastllmCublasHandle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            t, h, r, &alpha,
            (half*)kvCache.cudaData, r, t * r,
            (half*)qPe.cudaData, r, h * r,
            &beta1,
            score, t, t * h, 1);        
        int outer = b * s * h, channels = t;
        FastllmSoftmaxKernelInner1 <64> <<< outer, 64 >>> (score, score, outer, channels);
        status = cublasHgemmStridedBatched(fastllmCublasHandle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    c, b * s * h, t, &beta1,
                    (half*)peCache.cudaData, c, t * c,
                    score, t, b * s * h * t,
                    &beta0,
                    (half*)output.cudaData, c, c * b * s * h, 1);
        FastllmCudaFree(score);
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("status = %d\n", (int) status);
        printf("Error: cublas error during MatMul in MLA operator.\n");
        throw("cublas error");
    }

    DeviceSync();
    return true;
}

bool FastllmCudaAttention(const fastllm::Data &q, const fastllm::Data &k, const fastllm::Data &v,
                          const fastllm::Data &mask, const fastllm::Data &output, int group, float scale, int maskType) {
    int q0 = q.dims[0], q1 = q.dims[1], q2 = q.dims[2], k0 = k.dims[0], k1 = k.dims[1], v2 = v.dims[2];
    float *qd = (float*)q.cudaData;
    float *kd = (float*)k.cudaData;
    float *vd = (float*)v.cudaData;
    float *maskd = mask.dims.size() > 0 ? (float*)mask.cudaData : nullptr;
    float *od = (float*)output.cudaData;
    int batch = (mask.dims.size() == 3) ? mask.dims[0] : 1;
    int maskStride = (mask.dims.size() == 3 ? mask.strides[0] : mask.Count(0));

    if (q1 >= 1024 || (q1 > 1 && q1 != k1 && k1 >= 1024)) {
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

            if (batch == 1 && maskd == nullptr && maskType == 0) {
                CausalMask<256, float> <<<q1, 256>>>(qk, 0, q1, k1, k1 - q1);
                FastllmSoftmaxKernelInner1WithCausalMask<128> <<< q1, 128 >>>(qk, qk, q1, k1, k1 - q1);
            } else {
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

bool FastllmCudaHalfAttention(const fastllm::Data &q, const fastllm::Data &k, const fastllm::Data &v,
                              const fastllm::Data &mask, const fastllm::Data &output, int group, float scale, int maskType) {
    int q0 = q.dims[0], q1 = q.dims[1], q2 = q.dims[2], k0 = k.dims[0], k1 = k.dims[1], v2 = v.dims[2];
    half *qd = (half*)q.cudaData;
    half *kd = (half*)k.cudaData;
    half *vd = (half*)v.cudaData;
    half *maskd = mask.dims.size() > 0 ? (half*)mask.cudaData : nullptr;
    half *od = (half*)output.cudaData;
    int batch = (mask.dims.size() == 3) ? mask.dims[0] : 1;
    int maskStride = (mask.dims.size() == 3 ? mask.strides[0] : mask.Count(0));

    half beta = __float2half_rn(0.0f), one = __float2half_rn(1.0f), hscale = __float2half_rn(scale);
    if (q1 >= 1024 || (q1 > 1 && q1 != k1 && k1 >= 1024)) {
        int alignQ1 = q1, alignK1 = k1;
        int part = alignK1;
        bool useFastAttn = getCudaInfos()->hasTensorCore && batch == 1 && (q2 == 128 && v2 == 128) && maskType == 0;
        useFastAttn &= (q1 % 1024 == 0 && k1 % 1024 == 0);

        if (useFastAttn) {
            alignQ1 = ((q1 - 1) / 128 + 1) * 128;
            alignK1 = ((k1 - 1) / 128 + 1) * 128;
            part = (alignK1 > 8192 ? 8192 : alignK1);
        }
        half *qk = (half *) FastllmCudaMalloc(alignQ1 * part * sizeof(half));

        cudaMemset(qk, 0, alignQ1 * part * sizeof(half));
        auto fastllmCublasHandle = getFastllmCublasHandle();
        cublasStatus_t status;
        for (int i = 0; i < q0; i++) {
//DeviceSync();
//auto st = std::chrono::system_clock::now();
            if (useFastAttn) { 
                if (alignK1 > 8192) {
                    float *lastSum = (float*)FastllmCudaMalloc(alignQ1 * sizeof(float));
                    float *lastMax = (float*)FastllmCudaMalloc(alignQ1 * sizeof(float));
                    float *currentSum = (float*)FastllmCudaMalloc(alignQ1 * sizeof(float));
                    float *currentMax = (float*)FastllmCudaMalloc(alignQ1 * sizeof(float));

                    int threadPerBlock = std::min(256, alignQ1);
                    InitBlockAtten <<< (alignQ1 - 1) / threadPerBlock + 1, threadPerBlock>>> (lastSum, lastMax, currentSum, currentMax, alignQ1);

                    int part = 8192;
                    for (int st = 0; st < alignK1; st += part) {
                        int len = std::min(part, alignK1 - st);
                        status = cublasHgemm(fastllmCublasHandle,
                                            CUBLAS_OP_T, CUBLAS_OP_N,
                                            len, alignQ1, q2, &hscale,
                                            kd + (i / group) * k.Count(1) + st * k.strides[1], k.strides[1],
                                            qd + i * q.Count(1), q.strides[1],
                                            &beta, 
                                            qk, len);
                        CausalMask<256, half> <<<q1, 256>>>(qk, __float2half_rn(0.0f), alignQ1, len, k1 - q1 - st);
                        FastllmSoftmaxKernelInner1WithCausalMask<256> <<< q1, 256 >>>(qk, qk, alignQ1, len, k1 - q1 - st, currentMax, currentSum);
                        if (st > 0) {
                            AttnBlockUpdate <128> <<< alignQ1, 128 >>> (od + i * v2 * q1, alignQ1, v2, lastMax, lastSum, currentMax, currentSum);
                        } else {
                            cudaMemcpy(lastMax, currentMax, alignQ1 * sizeof(float), cudaMemcpyDeviceToDevice);
                            cudaMemcpy(lastSum, currentSum, alignQ1 * sizeof(float), cudaMemcpyDeviceToDevice);
                        }
                        half currentScale = __float2half_rn(st > 0 ? 1.0f : 0.0f);
                        status = cublasHgemm(fastllmCublasHandle,
                                            CUBLAS_OP_N, CUBLAS_OP_N,
                                            v2, alignQ1, len, &one,
                                            vd + (i / group) * v.Count(1) + st * v.strides[1], v.strides[1],
                                            qk, len,
                                            &currentScale,
                                            od + i * v2 * q1, v2);
                    }

                    FastllmCudaFree(lastSum);
                    FastllmCudaFree(lastMax);
                    FastllmCudaFree(currentSum);
                    FastllmCudaFree(currentMax);
                } else {
                    GpuQK(qd + i * q.Count(1), kd + (i / group) * k.Count(1), qk, alignQ1, alignK1, q2, scale, k1 - q1);
                    FastllmSoftmaxKernelInner1WithCausalMask<128> <<< q1, 128 >>>(qk, qk, q1, alignK1, k1 - q1);
                    status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                                CUBLAS_OP_N, CUBLAS_OP_N,
                                                v2, q1, alignK1, &one,
                                                vd + (i / group) * v.Count(1), v.strides[1], v.Count(1),
                                                qk, alignK1, alignK1 * alignQ1,
                                                &beta,
                                                od + i * v2 * q1, v2, v2 * q1, 1);
                }
            } else {
                status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                                CUBLAS_OP_T, CUBLAS_OP_N,
                                                k1, q1, q2, &hscale,
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

                if (batch == 1 && maskd == nullptr && maskType == 0) {
                    CausalMask<256, half> <<<q1, 256>>>(qk, __float2half_rn(0), q1, k1, k1 - q1);
                    FastllmSoftmaxKernelInner1WithCausalMask<128> <<< q1, 128 >>>(qk, qk, q1, k1, k1 - q1);
                } else {
                    if (maskd != nullptr) {
                        SimpleMask<256> <<< (q1 * k1 / 256) + 1, 256>>>(qk, maskd + (i / (q0 / batch)) * maskStride, __float2half_rn(-10000), q1 * k1);
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
                }

                status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                               CUBLAS_OP_N, CUBLAS_OP_N,
                                               v2, q1, k1, &one,
                                               vd + (i / group) * v.Count(1), v.strides[1], v.Count(1),
                                               qk, k1, k1 * q1,
                                               &beta,
                                               od + i * v2 * q1, v2, v2 * q1, 1);
            }

//DeviceSync(); printf("softmax spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
/*DeviceSync();
int n = k1, m = q1, k = q2;
float spend = GetSpan(st, std::chrono::system_clock::now());
float gops = (float)n * m * k * 4 / spend / 1e9;
printf("n = %d, m = %d, k = %d, spend %f s, gops = %f\n", n, m, k, spend, gops);*/
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
        half *qk = (half *) FastllmCudaMalloc(q0 * q1 * k1 * sizeof(half));
        half *temp = (half *) FastllmCudaMalloc(q0 * q1 * k1 * sizeof(half));
        auto fastllmCublasHandle = getFastllmCublasHandle();
        cublasStatus_t status;

        status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_T, CUBLAS_OP_N,
                                           k1, q1 * group, q2, &hscale,
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
            FastllmAttentionMaskKernel <256> <<< n * m, 256>>>(qk, maskd, __float2half_rn(-10000), n, m, spatial);
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

        status = cublasHgemmStridedBatched(fastllmCublasHandle,
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

template <typename T>
bool DoFastllmCudaAttentionBatch(fastllm::Data **q, fastllm::Data **k, fastllm::Data **v,
                               fastllm::Data **mask, fastllm::Data **output, int group, float scale, int batch) {
    if (false) {
        half beta = __float2half_rn(0.0f), one = __float2half_rn(1.0f), hscale = __float2half_rn(scale);
        int q0 = q[0]->dims[0], q1 = q[0]->dims[1], q2 = q[0]->dims[2], k0 = k[0]->dims[0], k1 = k[0]->dims[1], v2 = v[0]->dims[2];
        for (int i = 0; i < batch; i++) {
            q1 = max(q1, q[i]->dims[1]);
            k1 = max(k1, k[i]->dims[1]);
        }

        half *allKeys = (half*) FastllmCudaMalloc(batch * k0 * k1 * q2 * sizeof(half));
        half *allValues = (half*) FastllmCudaMalloc(batch * k0 * k1 * v2 * sizeof(half));

        std::vector <void*> dsts, srcs;
        std::vector <size_t> dpitchs, spitchs, widths, heights;
        for (int i = 0; i < batch; i++) {
            dsts.push_back((uint8_t *) (allKeys + i * k0 * k1 * q2));
            dpitchs.push_back(k1 * q2 * sizeof(half));
            srcs.push_back(k[i]->cudaData);
            spitchs.push_back(k[i]->strides[0] * sizeof(half));
            widths.push_back(k[i]->dims[1] * q2 * sizeof(half));
            heights.push_back(k0);

            dsts.push_back((uint8_t *) (allValues + i * k0 * k1 * v2));
            dpitchs.push_back(k1 * v2 * sizeof(half));
            srcs.push_back(v[i]->cudaData);
            spitchs.push_back(v[i]->strides[0] * sizeof(half));
            widths.push_back(v[i]->dims[1] * v2 * sizeof(half));
            heights.push_back(k0);
        }
        FastllmCudaMemcpy2DDeviceToDeviceBatch(dsts.data(), dpitchs.data(), srcs.data(), spitchs.data(), widths.data(), heights.data(), dsts.size());
/*
        for (int i = 0; i < batch; i++) {
            cudaMemcpy2D(
                allKeys + i * k0 * k1 * q2, k1 * q2 * sizeof(half), 
                k[i]->cudaData, k[i]->strides[0] * sizeof(half), 
                k[i]->dims[1] * q2 * sizeof(half), k0, 
                cudaMemcpyDeviceToDevice
            );
            cudaMemcpy2D(
                allValues + i * k0 * k1 * v2, k1 * v2 * sizeof(half), 
                v[i]->cudaData, v[i]->strides[0] * sizeof(half), 
                v[i]->dims[1] * v2 * sizeof(half), k0, 
                cudaMemcpyDeviceToDevice
            );
        }
*/
        half *qd = (half*)q[0]->cudaData;
        half *od = (half*)output[0]->cudaData;
        half *qk = (half *) FastllmCudaMalloc(batch * q0 * q1 * k1 * sizeof(half));
        half *temp = (half *) FastllmCudaMalloc(batch * q0 * q1 * k1 * sizeof(half));
        auto fastllmCublasHandle = getFastllmCublasHandle();
        cublasStatus_t status;

        status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_T, CUBLAS_OP_N,
                                           k1, q1 * group, q2, &hscale,
                                           allKeys, q2, k1 * q2,
                                           qd, q2, group * q1 * q2,
                                           &beta,
                                           qk, k1, k1 * q1 * group, batch * q0 / group);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("status = %d\n", (int) status);
            printf("Error: cublas error during MatMulTransB in Attention operator.\n");
            throw ("cublas error");
            exit(0);
        }

        int outer = batch * q0 * q1;
        if (k1 < 8) {
            FastllmSoftmaxKernelInner1<1> <<< outer, 1 >>>(qk, temp, outer, k1);
        } else if (k1 < 64) {
            FastllmSoftmaxKernelInner1<8> <<< outer, 8 >>>(qk, temp, outer, k1);
        } else if (k1 < 512) {
            FastllmSoftmaxKernelInner1<64> <<< outer, 64 >>>(qk, temp, outer, k1);
        } else {
            FastllmSoftmaxKernelInner1<256> <<< outer, 256 >>>(qk, temp, outer, k1);
        }

        status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_N, CUBLAS_OP_N,
                                           v2, q1 * group, k1, &one,
                                           allValues, v2, k1 * v2,
                                           temp, k1, k1 * q1 * group,
                                           &beta,
                                           od, v2, v2 * q1 * group, batch * q0 / group);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("status = %d\n", (int) status);
            printf("Error: cublas error during MatMul in Attention operator.\n");
            throw ("cublas error");
            exit(0);
        }

        FastllmCudaFree(allKeys);
        FastllmCudaFree(allValues);
        FastllmCudaFree(qk);
        FastllmCudaFree(temp);
        DeviceSync();
        return true;
    }

    int k0 = k[0]->dims[0];
    size_t memSum = 0;
    for (int b = 0; b < batch; b++) {
        memSum += q[b]->dims[0] * q[b]->dims[1] * k[b]->dims[1];
    }
    T *mem = (T*) FastllmCudaMalloc(memSum * sizeof(T));
    T **qk = new T*[batch];
    memSum = 0;
    for (int b = 0; b < batch; b++) {
        int s = q[b]->dims[0] * q[b]->dims[1] * k[b]->dims[1];
        qk[b] = mem + memSum;
        memSum += s;
    }

    uint8_t ** pointers = (uint8_t**)FastllmCudaMalloc(sizeof(uint8_t*) * batch * k0 * 8);
    uint8_t ** cpuPointers = new uint8_t*[batch * k0 * 8];
    if (true) {
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < k0; i++) {
                cpuPointers[(b * k0 + i) * 8 + 0] = (uint8_t *) q[b]->cudaData + i * group * q[b]->dims[1] * q[b]->dims[2] * sizeof(T);
                cpuPointers[(b * k0 + i) * 8 + 1] = (uint8_t *) k[b]->cudaData + i * k[b]->strides[0] * sizeof(T);
                cpuPointers[(b * k0 + i) * 8 + 2] = (uint8_t *) qk[b] + i * group * q[b]->dims[1] * k[b]->dims[1] * sizeof(T);
                cpuPointers[(b * k0 + i) * 8 + 3] = (uint8_t *) (size_t) (group * q[b]->dims[1]);
                cpuPointers[(b * k0 + i) * 8 + 4] = (uint8_t *) (size_t) q[b]->dims[2];
                cpuPointers[(b * k0 + i) * 8 + 5] = (uint8_t *) (size_t) k[b]->dims[1];
                cpuPointers[(b * k0 + i) * 8 + 6] = (uint8_t *) (size_t) q[b]->strides[1];
                cpuPointers[(b * k0 + i) * 8 + 7] = (uint8_t *) (size_t) k[b]->strides[1];
            }
        }
        cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * batch * k0 * 8, cudaMemcpyHostToDevice);
        if (typeid(T) == typeid(half)) {
            FastllmHalfMatMulTransBBatchKernel <128> <<<batch * k0, 128>>> (pointers, scale);
        } else {
            FastllmMatMulTransBBatchKernel <128> <<<batch * k0, 128>>> (pointers, scale);
        }
    }

    if (true) {
        int outer = q[0]->dims[0] * q[0]->dims[1];
        int maxChannels = 0;
        for (int b = 0; b < batch; b++) {
            int outer = q[b]->dims[0] * q[b]->dims[1];
            int channels = k[b]->dims[1];
            cpuPointers[b * 2 + 0] = (uint8_t*)(qk[b]);
            cpuPointers[b * 2 + 1] = (uint8_t*)((size_t)channels);
            maxChannels = max(maxChannels, channels);
        }
        cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * batch * 2, cudaMemcpyHostToDevice);
        if (maxChannels < 128) {
            FastllmSoftmaxKernelBatchInner1 <T, 32> <<<batch * outer, 32>>> (pointers, outer);
        } else if (maxChannels < 512) {
            FastllmSoftmaxKernelBatchInner1 <T, 64> <<<batch * outer, 64>>> (pointers, outer);
        } else {
            FastllmSoftmaxKernelBatchInner1 <T, 128> <<<batch * outer, 128>>> (pointers, outer);
        }
    }

    if (true) {
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < k0; i++) {
                cpuPointers[(b * k0 + i) * 8 + 0] = (uint8_t *) qk[b] + i * group * q[b]->dims[1] * k[b]->dims[1] * sizeof(T);
                cpuPointers[(b * k0 + i) * 8 + 1] = (uint8_t *) v[b]->cudaData + i * v[b]->strides[0] * sizeof(T);
                cpuPointers[(b * k0 + i) * 8 + 2] = (uint8_t *) output[b]->cudaData + i * group * q[b]->dims[1] * v[b]->dims[2] * sizeof(T);
                cpuPointers[(b * k0 + i) * 8 + 3] = (uint8_t *) (size_t) (group * q[b]->dims[1]);
                cpuPointers[(b * k0 + i) * 8 + 4] = (uint8_t *) (size_t) k[b]->dims[1];
                cpuPointers[(b * k0 + i) * 8 + 5] = (uint8_t *) (size_t) v[b]->dims[2];
                cpuPointers[(b * k0 + i) * 8 + 6] = (uint8_t *) (size_t) k[b]->dims[1];
                cpuPointers[(b * k0 + i) * 8 + 7] = (uint8_t *) (size_t) v[b]->strides[1];
            }
        }
        cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * batch * k0 * 8, cudaMemcpyHostToDevice);
        
        if (typeid(T) == typeid(half)) {
            FastllmHalfMatMulKernel <128> <<<batch * k0, 128>>> (pointers, 1.0f);
        } else {
            FastllmMatMulKernel <128> <<<batch * k0, 128>>> (pointers, 1.0f);
        }
    }

    FastllmCudaFree(pointers);
    delete[] cpuPointers;

    FastllmCudaFree(mem);
    delete[] qk;
    
    DeviceSync();
    return true;
}

bool FastllmCudaAttentionBatch(fastllm::Data **q, fastllm::Data **k, fastllm::Data **v,
                               fastllm::Data **mask, fastllm::Data **output, int group, float scale, int batch) {
    if (q[0]->dataType == fastllm::DataType::FLOAT32) {
        return DoFastllmCudaAttentionBatch <float> (q, k, v, mask, output, group, scale, batch);
    } else if (q[0]->dataType == fastllm::DataType::FLOAT16) {
        return DoFastllmCudaAttentionBatch <half> (q, k, v, mask, output, group, scale, batch);
    } else {
        printf("Error: attention datatype error.\n");
        throw ("Error: attention datatype error.");
        exit(0);
    }
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

void LaunchFastllmGemmFp16Fp16(half *input, half *weight, half *output, half *bias, int n, int m, int k) {
    if (n == 1) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 1> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else if (n == 2) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 2> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else if (n == 3) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 3> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else if (n == 4) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 4> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else if (n == 5) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 5> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else if (n == 6) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 6> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else if (n == 7) {
        FastllmGemvFp16Fp16Kernel2MultiRow<256, 7> <<< k, 256 >>>(input, weight, output, bias, m, k);
    } else {
        printf("Error: LaunchFastllmGemmFp16Fp16: n > 7.\n");
        exit(0);
    }
}

bool FastllmCudaHalfMatMulFloat16(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    if (weight.cudaData == nullptr || 
        (weight.extraCudaHalfData.size() == 0 && bias.dims.size() > 0)) {
        half *cudaBiasData;
        cudaError_t state = cudaSuccess;
        state = cudaMalloc(&cudaBiasData, k * sizeof(half));
        if (bias.dims.size() > 0) {
            float *tempBiasData;
            state = cudaMalloc(&tempBiasData, k * sizeof(float));
            state = cudaMemcpy(tempBiasData, (uint8_t *) bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
            int threadPerBlock = std::min(256, k);
            FastllmCudaFloat2HalfKernel <<< (k - 1) / threadPerBlock + 1, threadPerBlock>>>(tempBiasData, cudaBiasData, k);
            state = cudaFree(tempBiasData);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(half));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaHalfData.push_back((void *) cudaBiasData);
    }

    half *cudaInput = (half *) FastllmCudaPrepareInput(input);
    half *cudaOutput = (half *) FastllmCudaPrepareOutput(output);
    half *cudaBiasData = bias.dims.size() == 0 ? nullptr : (half *) weight.extraCudaHalfData[0];

    if (n < 8) {
        LaunchFastllmGemmFp16Fp16(cudaInput, (half*)weight.cudaData, cudaOutput, cudaBiasData, n, m, k);
    } else {
        auto fastllmCublasHandle = getFastllmCublasHandle();
        cublasStatus_t status;
#ifdef CUDA_NO_TENSOR_CORE
        float *cudaFp32Output = (float *) FastllmCudaMalloc(n * k * sizeof(float));
        float h_alpha = 1.0, h_beta = 0.0;
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_32F, ComputeType = CUDA_R_32F;
        status = cublasGemmEx(fastllmCublasHandle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            k, n, m,
                            &h_alpha, (half *) weight.cudaData, AType,
                            m, cudaInput, BType,
                            m, &h_beta,
                            cudaFp32Output, CType,
                            k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#else
        __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
        status = cublasGemmEx(fastllmCublasHandle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            k, n, m,
                            &h_alpha, (half *) weight.cudaData, AType,
                            m, cudaInput, BType,
                            m, &h_beta,
                            cudaOutput, CType,
                            k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#endif
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw ("cublas error");
            exit(0);
        }

#ifdef CUDA_NO_TENSOR_CORE
        int len = n * k;
        int threadPerBlock = std::min(256, len);
        FastllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaFp32Output, cudaOutput, len);
        FastllmCudaFree(cudaFp32Output);
#endif
        if (bias.dims.size() > 0) {
            FastllmCudaBiasKernel <<< n, 256 >>>(cudaOutput, (half *) weight.extraCudaHalfData[0], k);
        }
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

void LaunchFastllmGemmFp16Int8(half *input, uint8_t *weight, half *output, half *bias, float *scales, uint8_t *zeros, int n, int m, int k) {
    for (int i = 0; i < n; i++) {
        FastllmGemvFp16Int8Kernel2 <256, 1> <<< k, 256 >>>(input + i * m, weight, output + i * k, bias, scales, zeros, m, k);
    }
}

bool FastllmCudaHalfMatMulFloatInt8(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    if (weight.cudaData == nullptr || weight.extraCudaHalfData.size() == 0) {
        weight.extraCudaHalfData.push_back((void*)weight.extraCudaData[0]);
        weight.extraCudaHalfData.push_back((void*)weight.extraCudaData[1]);

        half *cudaBiasData;
        cudaError_t state = cudaMalloc(&cudaBiasData, k * sizeof(half));
        if (bias.dims.size() > 0) {
            float *tempBiasData;
            state = cudaMalloc(&tempBiasData, k * sizeof(float));
            state = cudaMemcpy(tempBiasData, (uint8_t*)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
            int threadPerBlock = std::min(256, k);
            FastllmCudaFloat2HalfKernel <<< (k - 1) / threadPerBlock + 1, threadPerBlock>>>(tempBiasData, cudaBiasData, k);
            state = cudaFree(tempBiasData);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(half));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaHalfData.push_back((void*)cudaBiasData);
    }
    float *cudaScales = (float*)weight.extraCudaHalfData[0];
    uint8_t *cudaZeropoints = (uint8_t*)weight.extraCudaHalfData[1];

    half *cudaInput = (half*)FastllmCudaPrepareInput(input);
    half *cudaOutput = (half*)FastllmCudaPrepareOutput(output);

    if (n >= 8) {
        auto fastllmCublasHandle = getFastllmCublasHandle();
        half *cudaFp16Weight;

        cudaFp16Weight = (half *) FastllmCudaMalloc(k * m * sizeof(half));

#ifdef CUDA_NO_TENSOR_CORE
        float *cudaFp32Output = (float *) FastllmCudaMalloc(n * k * sizeof(float));
        float h_alpha = 1.0, h_beta = 0.0;
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_32F, ComputeType = CUDA_R_32F;
#else
        __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
#endif
        cublasStatus_t status;

        int len = n * m;
        int threadPerBlock = std::min(256, len);

        len = k * m;

        FastllmCudaInt82HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock>>>((uint8_t*)weight.cudaData,
                                                                                            cudaScales,
                                                                                            cudaZeropoints,
                                                                                            cudaFp16Weight, len, m);

#ifdef CUDA_NO_TENSOR_CORE
        status = cublasGemmEx(fastllmCublasHandle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                k, n, m,
                                &h_alpha, cudaFp16Weight, AType,
                                m, cudaInput, BType,
                                m, &h_beta,
                                cudaFp32Output, CType,
                                k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#else
        status = cublasGemmEx(fastllmCublasHandle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                k, n, m,
                                &h_alpha, cudaFp16Weight, AType,
                                m, cudaInput, BType,
                                m, &h_beta,
                                cudaOutput, CType,
                                k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#endif

        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw("cublas error");
            exit(0);
        }

#ifdef CUDA_NO_TENSOR_CORE
        len = n * k;
        FastllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaFp32Output, cudaOutput, len);
        FastllmCudaFree(cudaFp32Output);
#endif
        if (bias.dims.size() > 0) {
            half *cudaBiasData = (half*)weight.extraCudaHalfData[2];
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, cudaBiasData, k);
        }

        FastllmCudaFree(cudaFp16Weight);
    } else {
        half *cudaBiasData = bias.dims.size() > 0 ? (half*)weight.extraCudaHalfData[2] : nullptr;
        LaunchFastllmGemmFp16Int8(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, cudaBiasData, cudaScales, cudaZeropoints, n, m, k);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

void LaunchFastllmGemmFp16Int4Group(half *input, uint8_t *weight, half *output, half *bias, half *scales, half *mins, int n, int m, int k, int group, int groupCnt) {
    if (n == 1) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 1> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 2) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 2> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 3) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 3> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 4) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 4> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 5) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 5> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 6) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 6> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 7) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 7> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 8) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 8> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 9) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 9> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 10) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 10> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 11) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 11> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 12) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 12> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 13) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 13> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 14) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 14> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 15) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 15> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else if (n == 16) {
        FastllmGemvHalfInt4GroupKernelMultiRow<64, 16> <<< k, 64 >>>(input, weight, output, bias, scales, mins, m, k, group, groupCnt);
    } else {
        for (int i = 0; i < n; i++) {
            FastllmGemvHalfInt4GroupKernelMultiRow<64, 1> <<< k, 64 >>>(input + i * m, weight, output + i * k, bias, scales, mins, m, k, group, groupCnt);
        }
        return;
    }
    
}

bool FastllmCudaHalfMatMulFloatInt4Group(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    int group = weight.group, groupCnt = weight.groupCnt;
    if (weight.cudaData == nullptr || weight.extraCudaHalfData.size() == 0) {
        weight.extraCudaHalfData.push_back((void*)weight.extraCudaData[0]);
        weight.extraCudaHalfData.push_back((void*)weight.extraCudaData[1]);

        half *cudaBiasData;
        cudaError_t state = cudaMalloc(&cudaBiasData, k * sizeof(half));
        if (bias.dims.size() > 0) {
            float *tempBiasData;
            state = cudaMalloc(&tempBiasData, k * sizeof(float));
            state = cudaMemcpy(tempBiasData, (uint8_t*)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
            int threadPerBlock = std::min(256, k);
            FastllmCudaFloat2HalfKernel <<< (k - 1) / threadPerBlock + 1, threadPerBlock>>>(tempBiasData, cudaBiasData, k);
            state = cudaFree(tempBiasData);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(half));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaHalfData.push_back((void*)cudaBiasData);
    }
    half *cudaScales = (half*)weight.extraCudaHalfData[0];
    half *cudaMins = (half*)weight.extraCudaHalfData[1];

    half *cudaInput = (half*)FastllmCudaPrepareInput(input);
    half *cudaOutput = (half*)FastllmCudaPrepareOutput(output);

    if (n > 16) {
        auto fastllmCublasHandle = getFastllmCublasHandle();
        half *cudaFp16Weight;
        cudaFp16Weight = (half *) FastllmCudaMalloc(k * m * sizeof(half));

#ifdef CUDA_NO_TENSOR_CORE
        float *cudaFp32Output = (float *) FastllmCudaMalloc(n * k * sizeof(float));
        float h_alpha = 1.0, h_beta = 0.0;
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_32F, ComputeType = CUDA_R_32F;
#else
        __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
#endif
        cublasStatus_t status;

        int len = n * m;
        int threadPerBlock = std::min(256, len);

        len = k * m;

        FastllmCudaInt4Group2HalfKernel <<< k, 256 >>>((uint8_t*)weight.cudaData, cudaScales, cudaMins, cudaFp16Weight, k, m, group, groupCnt);

#ifdef CUDA_NO_TENSOR_CORE
        status = cublasGemmEx(fastllmCublasHandle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                k, n, m,
                                &h_alpha, cudaFp16Weight, AType,
                                m, cudaInput, BType,
                                m, &h_beta,
                                cudaFp32Output, CType,
                                k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#else
        status = cublasGemmEx(fastllmCublasHandle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                k, n, m,
                                &h_alpha, cudaFp16Weight, AType,
                                m, cudaInput, BType,
                                m, &h_beta,
                                cudaOutput, CType,
                                k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));

#endif
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error. status = %d\n", status);
            throw("cublas error");
            exit(0);
        }

#ifdef CUDA_NO_TENSOR_CORE
        len = n * k;
        FastllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaFp32Output, cudaOutput, len);
        FastllmCudaFree(cudaFp32Output);
#endif
        if (bias.dims.size() > 0) {
            half *cudaBiasData = (half*)weight.extraCudaHalfData[2];
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, cudaBiasData, k);
        }

        FastllmCudaFree(cudaFp16Weight);
    } else {
        half *cudaBiasData = (half*)weight.extraCudaHalfData[2];
        LaunchFastllmGemmFp16Int4Group(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, cudaBiasData, cudaScales, cudaMins, n, m, k, group, groupCnt);
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

void LaunchFastllmGemmFp16Int4NoZero(half *input, uint8_t *weight, half *output, half *bias, float *scales, float *mins, int n, int m, int k) {
    if (n == 1) {
        FastllmGemvFp16Int4NoZeroKernel1MultiRow <64, 1> <<< k, 64 >>> (input, weight, output, bias, scales, mins, m, k);
    } else if (n == 2) {
        FastllmGemvFp16Int4NoZeroKernel1MultiRow <64, 2> <<< k, 64 >>> (input, weight, output, bias, scales, mins, m, k);
    } else if (n == 3) {
        FastllmGemvFp16Int4NoZeroKernel1MultiRow <64, 3> <<< k, 64 >>> (input, weight, output, bias, scales, mins, m, k);
    } else if (n == 4) {
        FastllmGemvFp16Int4NoZeroKernel1MultiRow <64, 4> <<< k, 64 >>> (input, weight, output, bias, scales, mins, m, k);
    } else if (n == 5) {
        FastllmGemvFp16Int4NoZeroKernel1MultiRow <64, 5> <<< k, 64 >>> (input, weight, output, bias, scales, mins, m, k);
    } else if (n == 6) {
        FastllmGemvFp16Int4NoZeroKernel1MultiRow <64, 6> <<< k, 64 >>> (input, weight, output, bias, scales, mins, m, k);
    } else if (n == 7) {
        FastllmGemvFp16Int4NoZeroKernel1MultiRow <64, 7> <<< k, 64 >>> (input, weight, output, bias, scales, mins, m, k);
    } else {
        for (int i = 0; i < n; i++) {
            FastllmGemvFp16Int4NoZeroKernel2<64, 1> <<< k / 1, 64 >>>(input + i * m, weight, output + i * k, bias, scales, mins, m, k);
        }
    }
}

bool FastllmCudaHalfMatMulFloatInt4NoZero(const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k) {
    if (weight.cudaData == nullptr || weight.extraCudaHalfData.size() == 0) {
        weight.extraCudaHalfData.push_back((void*)weight.extraCudaData[0]);
        weight.extraCudaHalfData.push_back((void*)weight.extraCudaData[1]);

        half *cudaBiasData;
        cudaError_t state = cudaMalloc(&cudaBiasData, k * sizeof(half));
        if (bias.dims.size() > 0) {
            float *tempBiasData;
            state = cudaMalloc(&tempBiasData, k * sizeof(float));
            state = cudaMemcpy(tempBiasData, (uint8_t*)bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
            int threadPerBlock = std::min(256, k);
            FastllmCudaFloat2HalfKernel <<< (k - 1) / threadPerBlock + 1, threadPerBlock>>>(tempBiasData, cudaBiasData, k);
            state = cudaFree(tempBiasData);
        } else {
            state = cudaMemset(cudaBiasData, 0, k * sizeof(half));
        }
        checkCudaErrors("Error: CUDA error when moving bias to device!", state);
        weight.extraCudaHalfData.push_back((void*)cudaBiasData);
    }
    float *cudaScales = (float*)weight.extraCudaHalfData[0];
    float *cudaMins = (float*)weight.extraCudaHalfData[1];

    half *cudaInput = (half*)FastllmCudaPrepareInput(input);
    half *cudaOutput = (half*)FastllmCudaPrepareOutput(output);

    if (n >= 8) {
        auto fastllmCublasHandle = getFastllmCublasHandle();
        half *cudaFp16Weight;

        cudaFp16Weight = (half *) FastllmCudaMalloc(k * m * sizeof(half));

#ifdef CUDA_NO_TENSOR_CORE
        float *cudaFp32Output = (float *) FastllmCudaMalloc(n * k * sizeof(float));
        float h_alpha = 1.0, h_beta = 0.0;
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_32F, ComputeType = CUDA_R_32F;
#else
        __half h_alpha = __float2half_rn(1.0), h_beta = __float2half_rn(0.0);
        cudaDataType_t AType = CUDA_R_16F, BType = CUDA_R_16F, CType = CUDA_R_16F, ComputeType = CUDA_R_16F;
#endif
        cublasStatus_t status;

        int len = n * m;
        int threadPerBlock = std::min(256, len);

        len = k * m;
        int gridSize = (len - 1) / (threadPerBlock * 4) + 1;
        FastllmCudaInt42HalfKernel <<< gridSize, threadPerBlock>>>((uint8_t *) weight.cudaData,
                                                                    cudaScales,
                                                                    cudaMins,
                                                                    cudaFp16Weight, len, m);

#ifdef CUDA_NO_TENSOR_CORE
        status = cublasGemmEx(fastllmCublasHandle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                k, n, m,
                                &h_alpha, cudaFp16Weight, AType,
                                m, cudaInput, BType,
                                m, &h_beta,
                                cudaFp32Output, CType,
                                k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#else
        status = cublasGemmEx(fastllmCublasHandle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                k, n, m,
                                &h_alpha, cudaFp16Weight, AType,
                                m, cudaInput, BType,
                                m, &h_beta,
                                cudaOutput, CType,
                                k, ComputeType, static_cast<cublasGemmAlgo_t>(CUBLAS_GEMM_DEFAULT));
#endif

        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: cublas error.\n");
            throw("cublas error");
            exit(0);
        }

#ifdef CUDA_NO_TENSOR_CORE
        len = n * k;
        FastllmCudaFloat2HalfKernel <<< (len - 1) / threadPerBlock + 1, threadPerBlock >>>(cudaFp32Output, cudaOutput, len);
        FastllmCudaFree(cudaFp32Output);
#endif
        if (bias.dims.size() > 0) {
            half *cudaBiasData = (half*)weight.extraCudaHalfData[2];
            FastllmCudaBiasKernel <<< n, 256 >>> (cudaOutput, cudaBiasData, k);
        }

        FastllmCudaFree(cudaFp16Weight);
    } else {
        half *cudaBiasData = (half*)weight.extraCudaHalfData[2];
        LaunchFastllmGemmFp16Int4NoZero(cudaInput, (uint8_t*)weight.cudaData, cudaOutput, cudaBiasData, cudaScales, cudaMins, n, m, k);
    }
    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
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
    int threadsPerBlock = min(256, max(n2 * n3, n3));
    
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
