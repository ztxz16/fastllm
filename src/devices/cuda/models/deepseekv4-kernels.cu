#include "fastllm-cuda.cuh"
#include "fastllm.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace {

constexpr int kDeepSeekV4SparseDecodeMaxKeys = 256 * 1024;
constexpr int kDeepSeekV4SparsePrefillMaxKeys = 256 * 1024;

template <typename Kernel>
bool DeepSeekV4EnsureDynamicSharedMemory(Kernel kernel, size_t sharedBytes) {
    constexpr size_t kDefaultDynamicSharedLimit = 48 * 1024;
    if (sharedBytes <= kDefaultDynamicSharedLimit) {
        return true;
    }
    int device = 0;
    cudaError_t state = cudaGetDevice(&device);
    if (state != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    int maxOptinShared = 0;
    state = cudaDeviceGetAttribute(&maxOptinShared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (state != cudaSuccess || sharedBytes > (size_t)maxOptinShared) {
        cudaGetLastError();
        return false;
    }
    state = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sharedBytes);
    if (state != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return true;
}

__device__ __forceinline__ float Dsv4ToFloat(float v) {
    return v;
}

__device__ __forceinline__ float Dsv4ToFloat(half v) {
    return __half2float(v);
}

__device__ __forceinline__ float Dsv4ToFloat(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename XT, typename WT>
__device__ __forceinline__ float Dsv4PairDot(const XT *x, const WT *w, int k) {
    return Dsv4ToFloat(x[k]) * Dsv4ToFloat(w[k]) +
           Dsv4ToFloat(x[k + 1]) * Dsv4ToFloat(w[k + 1]);
}

__device__ __forceinline__ float Dsv4PairDot(const __nv_bfloat16 *x, const __nv_bfloat16 *w, int k) {
    __nv_bfloat162 xv = *reinterpret_cast<const __nv_bfloat162 *>(x + k);
    __nv_bfloat162 wv = *reinterpret_cast<const __nv_bfloat162 *>(w + k);
    float2 xf = __bfloat1622float2(xv);
    float2 wf = __bfloat1622float2(wv);
    return xf.x * wf.x + xf.y * wf.y;
}

__device__ __forceinline__ float Dsv4PairDot(const half *x, const half *w, int k) {
    half2 xv = *reinterpret_cast<const half2 *>(x + k);
    half2 wv = *reinterpret_cast<const half2 *>(w + k);
    float2 xf = __half22float2(xv);
    float2 wf = __half22float2(wv);
    return xf.x * wf.x + xf.y * wf.y;
}

__device__ __forceinline__ float Dsv4PairDot(const float *x, const float *w, int k) {
    float2 xv = *reinterpret_cast<const float2 *>(x + k);
    float2 wv = *reinterpret_cast<const float2 *>(w + k);
    return xv.x * wv.x + xv.y * wv.y;
}

template <typename T>
__device__ __forceinline__ T Dsv4FromFloat(float v);

template <>
__device__ __forceinline__ float Dsv4FromFloat<float>(float v) {
    return v;
}

template <>
__device__ __forceinline__ half Dsv4FromFloat<half>(float v) {
    return __float2half_rn(v);
}

template <>
__device__ __forceinline__ __nv_bfloat16 Dsv4FromFloat<__nv_bfloat16>(float v) {
    return __float2bfloat16_rn(v);
}

template <typename InT, typename WT>
__global__ void DeepSeekV4WoAKernel(const InT *o, const WT *w, __nv_bfloat16 *output,
                                    int bsz, int seqlen, int heads, int headDim,
                                    int groups, int oRank) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = bsz * seqlen * groups * oRank;
    if (idx >= total) {
        return;
    }

    int r = idx % oRank;
    int tmp = idx / oRank;
    int g = tmp % groups;
    tmp /= groups;
    int s = tmp % seqlen;
    int b = tmp / seqlen;

    int headsPerGroup = heads / groups;
    int groupDim = headsPerGroup * headDim;
    const WT *wrow = w + ((uint64_t)g * oRank + r) * groupDim;
    const InT *src = o + (((uint64_t)b * seqlen + s) * heads + g * headsPerGroup) * headDim;

    double v = 0.0;
    for (int d = 0; d < groupDim; d++) {
        v += (double)Dsv4ToFloat(src[d]) * Dsv4ToFloat(wrow[d]);
    }
    output[idx] = __float2bfloat16_rn((float)v);
}

template <typename InT, typename WT>
__global__ void DeepSeekV4WoABlockReduceKernel(const InT *o, const WT *w, __nv_bfloat16 *output,
                                               int bsz, int seqlen, int heads, int headDim,
                                               int groups, int oRank) {
    extern __shared__ float partial[];
    int idx = blockIdx.x;
    int total = bsz * seqlen * groups * oRank;
    if (idx >= total) {
        return;
    }

    int r = idx % oRank;
    int tmp = idx / oRank;
    int g = tmp % groups;
    tmp /= groups;
    int s = tmp % seqlen;
    int b = tmp / seqlen;

    int headsPerGroup = heads / groups;
    int groupDim = headsPerGroup * headDim;
    const WT *wrow = w + ((uint64_t)g * oRank + r) * groupDim;
    const InT *src = o + (((uint64_t)b * seqlen + s) * heads + g * headsPerGroup) * headDim;

    float sum = 0.0f;
    for (int d = threadIdx.x; d < groupDim; d += blockDim.x) {
        sum += Dsv4ToFloat(src[d]) * Dsv4ToFloat(wrow[d]);
    }
    partial[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[idx] = __float2bfloat16_rn(partial[0]);
    }
}

template <typename InT, typename WT>
__global__ void DeepSeekV4WoAPairBlockReduceKernel(const InT *o, const WT *w, __nv_bfloat16 *output,
                                                   int bsz, int seqlen, int heads, int headDim,
                                                   int groups, int oRank) {
    extern __shared__ float partial[];
    float *partial0 = partial;
    float *partial1 = partial + blockDim.x;
    int pairsPerGroup = oRank / 2;
    int idx = blockIdx.x;
    int total = bsz * seqlen * groups * pairsPerGroup;
    if (idx >= total) {
        return;
    }

    int pair = idx % pairsPerGroup;
    int r0 = pair * 2;
    int r1 = r0 + 1;
    int tmp = idx / pairsPerGroup;
    int g = tmp % groups;
    tmp /= groups;
    int s = tmp % seqlen;
    int b = tmp / seqlen;

    int headsPerGroup = heads / groups;
    int groupDim = headsPerGroup * headDim;
    const WT *wrow0 = w + ((uint64_t)g * oRank + r0) * groupDim;
    const WT *wrow1 = wrow0 + groupDim;
    const InT *src = o + (((uint64_t)b * seqlen + s) * heads + g * headsPerGroup) * headDim;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (int d = threadIdx.x; d < groupDim; d += blockDim.x) {
        float x = Dsv4ToFloat(src[d]);
        sum0 += x * Dsv4ToFloat(wrow0[d]);
        sum1 += x * Dsv4ToFloat(wrow1[d]);
    }
    partial0[threadIdx.x] = sum0;
    partial1[threadIdx.x] = sum1;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial0[threadIdx.x] += partial0[threadIdx.x + stride];
            partial1[threadIdx.x] += partial1[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        uint64_t outBase = (((uint64_t)b * seqlen + s) * groups + g) * oRank;
        output[outBase + r0] = __float2bfloat16_rn(partial0[0]);
        output[outBase + r1] = __float2bfloat16_rn(partial1[0]);
    }
}

template <typename InT, typename WT>
__global__ void DeepSeekV4WoAFloatAccKernel(const InT *o, const WT *w, __nv_bfloat16 *output,
                                            int bsz, int seqlen, int heads, int headDim,
                                            int groups, int oRank) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = bsz * seqlen * groups * oRank;
    if (idx >= total) {
        return;
    }

    int r = idx % oRank;
    int tmp = idx / oRank;
    int g = tmp % groups;
    tmp /= groups;
    int s = tmp % seqlen;
    int b = tmp / seqlen;

    int headsPerGroup = heads / groups;
    int groupDim = headsPerGroup * headDim;
    const WT *wrow = w + ((uint64_t)g * oRank + r) * groupDim;
    const InT *src = o + (((uint64_t)b * seqlen + s) * heads + g * headsPerGroup) * headDim;

    float v = 0.0f;
    for (int d = 0; d < groupDim; d++) {
        v += Dsv4ToFloat(src[d]) * Dsv4ToFloat(wrow[d]);
    }
    output[idx] = __float2bfloat16_rn(v);
}

template <typename InT, typename WT>
__global__ void DeepSeekV4WoAKahanAccKernel(const InT *o, const WT *w, __nv_bfloat16 *output,
                                            int bsz, int seqlen, int heads, int headDim,
                                            int groups, int oRank) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = bsz * seqlen * groups * oRank;
    if (idx >= total) {
        return;
    }

    int r = idx % oRank;
    int tmp = idx / oRank;
    int g = tmp % groups;
    tmp /= groups;
    int s = tmp % seqlen;
    int b = tmp / seqlen;

    int headsPerGroup = heads / groups;
    int groupDim = headsPerGroup * headDim;
    const WT *wrow = w + ((uint64_t)g * oRank + r) * groupDim;
    const InT *src = o + (((uint64_t)b * seqlen + s) * heads + g * headsPerGroup) * headDim;

    float sum = 0.0f;
    float c = 0.0f;
    for (int d = 0; d < groupDim; d++) {
        float prod = Dsv4ToFloat(src[d]) * Dsv4ToFloat(wrow[d]);
        float y = prod - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    output[idx] = __float2bfloat16_rn(sum);
}

template <typename InT, typename WT>
__global__ void DeepSeekV4WoAPairKernel(const InT *o, const WT *w, __nv_bfloat16 *output,
                                        int bsz, int seqlen, int heads, int headDim,
                                        int groups, int oRank) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pairsPerGroup = oRank / 2;
    int total = bsz * seqlen * groups * pairsPerGroup;
    if (idx >= total) {
        return;
    }

    int pair = idx % pairsPerGroup;
    int r0 = pair * 2;
    int r1 = r0 + 1;
    int tmp = idx / pairsPerGroup;
    int g = tmp % groups;
    tmp /= groups;
    int s = tmp % seqlen;
    int b = tmp / seqlen;

    int headsPerGroup = heads / groups;
    int groupDim = headsPerGroup * headDim;
    const WT *wrow0 = w + ((uint64_t)g * oRank + r0) * groupDim;
    const WT *wrow1 = wrow0 + groupDim;

    double v0 = 0.0;
    double v1 = 0.0;
    int d = 0;
    for (int hh = 0; hh < headsPerGroup; hh++) {
        const InT *src = o + (((uint64_t)b * seqlen + s) * heads + g * headsPerGroup + hh) * headDim;
        for (int localD = 0; localD < headDim; localD++, d++) {
            double x = (double)Dsv4ToFloat(src[localD]);
            v0 += x * Dsv4ToFloat(wrow0[d]);
            v1 += x * Dsv4ToFloat(wrow1[d]);
        }
    }
    uint64_t outBase = (((uint64_t)b * seqlen + s) * groups + g) * oRank;
    output[outBase + r0] = __float2bfloat16_rn((float)v0);
    output[outBase + r1] = __float2bfloat16_rn((float)v1);
}

__device__ __forceinline__ float DeepSeekV4InvFreq(int idx, int ropeDim, float base,
                                                   int originalSeqLen, float factor,
                                                   int betaFast, int betaSlow) {
    float inv = 1.0f / powf(base, (float)(idx * 2) / (float)ropeDim);
    if (originalSeqLen > 0) {
        float lowF = ropeDim * logf((float)originalSeqLen / (betaFast * 2.0f * 3.14159265358979323846f)) /
                     (2.0f * logf(base));
        float highF = ropeDim * logf((float)originalSeqLen / (betaSlow * 2.0f * 3.14159265358979323846f)) /
                      (2.0f * logf(base));
        int low = max((int)floorf(lowF), 0);
        int high = min((int)ceilf(highF), ropeDim - 1);
        if (low == high) {
            high++;
        }
        float ramp = fminf(1.0f, fmaxf(0.0f, ((float)idx - low) / (float)(high - low)));
        float smooth = 1.0f - ramp;
        inv = inv / factor * (1.0f - smooth) + inv * smooth;
    }
    return inv;
}

template <typename InT, typename OutT>
__global__ void DeepSeekV4RMSNormKernel(const InT *input, const float *weight, OutT *output,
                                        int rows, int dim, float eps) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    extern __shared__ float red[];
    const InT *src = input + (uint64_t)row * dim;
    OutT *dst = output + (uint64_t)row * dim;
    float ss = 0.0f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float v = Dsv4ToFloat(src[d]);
        ss += v * v;
    }
    red[threadIdx.x] = ss;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] += red[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float scale = rsqrtf(red[0] / dim + eps);
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        dst[d] = Dsv4FromFloat<OutT>(Dsv4ToFloat(src[d]) * scale * weight[d]);
    }
}

template <typename T>
__global__ void DeepSeekV4ScaleQRotaryKernel(T *q, int rows, int seqlen, int heads, int dim,
                                             int ropeDim, float ropeBase, int startPos,
                                             int originalSeqLen, float ropeFactor,
                                             int betaFast, int betaSlow, float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    T *ptr = q + (uint64_t)row * dim;
    double ss = 0.0;
    for (int d = 0; d < dim; d++) {
        double v = (double)Dsv4ToFloat(ptr[d]);
        ss += v * v;
    }
    float scale = 1.0f / sqrtf((float)(ss / dim) + eps);
    for (int d = 0; d < dim; d++) {
        ptr[d] = Dsv4FromFloat<T>(Dsv4ToFloat(ptr[d]) * scale);
    }

    int s = (row / heads) % seqlen;
    int pos = startPos + s;
    int off = dim - ropeDim;
    for (int i = 0; i < ropeDim; i += 2) {
        float inv = DeepSeekV4InvFreq(i / 2, ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow);
        float ang = pos * inv;
        float c = cosf(ang);
        float sn = sinf(ang);
        float a = Dsv4ToFloat(ptr[off + i]);
        float b = Dsv4ToFloat(ptr[off + i + 1]);
        ptr[off + i] = Dsv4FromFloat<T>(a * c - b * sn);
        ptr[off + i + 1] = Dsv4FromFloat<T>(a * sn + b * c);
    }
}

template <typename T>
__global__ void DeepSeekV4ScaleQRotaryBlockKernel(T *q, int rows, int seqlen, int heads, int dim,
                                                  int ropeDim, float ropeBase, int startPos,
                                                  int originalSeqLen, float ropeFactor,
                                                  int betaFast, int betaSlow, float eps) {
    extern __shared__ float partial[];
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    T *ptr = q + (uint64_t)row * dim;
    float ss = 0.0f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float v = Dsv4ToFloat(ptr[d]);
        ss += v * v;
    }
    partial[threadIdx.x] = ss;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial[0] = rsqrtf(partial[0] / dim + eps);
    }
    __syncthreads();
    float scale = partial[0];
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        ptr[d] = Dsv4FromFloat<T>(Dsv4ToFloat(ptr[d]) * scale);
    }
    __syncthreads();

    int s = (row / heads) % seqlen;
    int pos = startPos + s;
    int off = dim - ropeDim;
    for (int i = threadIdx.x * 2; i < ropeDim; i += blockDim.x * 2) {
        float inv = DeepSeekV4InvFreq(i / 2, ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow);
        float ang = pos * inv;
        float c = cosf(ang);
        float sn = sinf(ang);
        float a = Dsv4ToFloat(ptr[off + i]);
        float b = Dsv4ToFloat(ptr[off + i + 1]);
        ptr[off + i] = Dsv4FromFloat<T>(a * c - b * sn);
        ptr[off + i + 1] = Dsv4FromFloat<T>(a * sn + b * c);
    }
}

template <typename T>
__global__ void DeepSeekV4RotaryQuantKernel(T *x, int rows, int seqlen, int heads, int dim,
                                            int ropeDim, float ropeBase, int startPos,
                                            int originalSeqLen, float ropeFactor,
                                            int betaFast, int betaSlow, int quantDim,
                                            int blockSize, int posStep) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    T *ptr = x + (uint64_t)row * dim;
    int s = (row / heads) % seqlen;
    int pos = startPos + s * posStep;
    int off = dim - ropeDim;
    for (int i = 0; i < ropeDim; i += 2) {
        float inv = DeepSeekV4InvFreq(i / 2, ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow);
        float ang = pos * inv;
        float c = cosf(ang);
        float sn = sinf(ang);
        float a = Dsv4ToFloat(ptr[off + i]);
        float b = Dsv4ToFloat(ptr[off + i + 1]);
        ptr[off + i] = Dsv4FromFloat<T>(a * c - b * sn);
        ptr[off + i + 1] = Dsv4FromFloat<T>(a * sn + b * c);
    }

    for (int start = 0; start < quantDim; start += blockSize) {
        int end = min(start + blockSize, quantDim);
        float amax = 1e-4f;
        for (int d = start; d < end; d++) {
            amax = fmaxf(amax, fabsf(Dsv4ToFloat(ptr[d])));
        }
        float qScale = powf(2.0f, ceilf(log2f(amax / 448.0f)));
        for (int d = start; d < end; d++) {
            float qv = fminf(448.0f, fmaxf(-448.0f, Dsv4ToFloat(ptr[d]) / qScale));
            float rounded = __bfloat162float(__float2bfloat16_rn(qv)) * qScale;
            ptr[d] = Dsv4FromFloat<T>(rounded);
        }
    }
}

template <typename T>
__global__ void DeepSeekV4RotaryQuantBlockKernel(T *x, int rows, int seqlen, int heads, int dim,
                                                 int ropeDim, float ropeBase, int startPos,
                                                 int originalSeqLen, float ropeFactor,
                                                 int betaFast, int betaSlow, int quantDim,
                                                 int blockSize, int posStep) {
    extern __shared__ float partial[];
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    T *ptr = x + (uint64_t)row * dim;
    int s = (row / heads) % seqlen;
    int pos = startPos + s * posStep;
    int off = dim - ropeDim;
    for (int i = threadIdx.x * 2; i < ropeDim; i += blockDim.x * 2) {
        float inv = DeepSeekV4InvFreq(i / 2, ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow);
        float ang = pos * inv;
        float c = cosf(ang);
        float sn = sinf(ang);
        float a = Dsv4ToFloat(ptr[off + i]);
        float b = Dsv4ToFloat(ptr[off + i + 1]);
        ptr[off + i] = Dsv4FromFloat<T>(a * c - b * sn);
        ptr[off + i + 1] = Dsv4FromFloat<T>(a * sn + b * c);
    }
    __syncthreads();

    for (int start = 0; start < quantDim; start += blockSize) {
        int end = min(start + blockSize, quantDim);
        float amax = 1e-4f;
        for (int d = start + threadIdx.x; d < end; d += blockDim.x) {
            amax = fmaxf(amax, fabsf(Dsv4ToFloat(ptr[d])));
        }
        partial[threadIdx.x] = amax;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            partial[0] = powf(2.0f, ceilf(log2f(partial[0] / 448.0f)));
        }
        __syncthreads();
        float qScale = partial[0];
        for (int d = start + threadIdx.x; d < end; d += blockDim.x) {
            float qv = fminf(448.0f, fmaxf(-448.0f, Dsv4ToFloat(ptr[d]) / qScale));
            float rounded = __bfloat162float(__float2bfloat16_rn(qv)) * qScale;
            ptr[d] = Dsv4FromFloat<T>(rounded);
        }
        __syncthreads();
    }
}

template <typename T>
__global__ void DeepSeekV4UpdateWindowKVCacheKernel(const T *kv, float *windowKV,
                                                    int bsz, int headDim, int startPos,
                                                    int windowSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = bsz * headDim;
    if (idx >= total) {
        return;
    }
    int b = idx / headDim;
    int d = idx % headDim;
    windowKV[((uint64_t)b * windowSize + (startPos % windowSize)) * headDim + d] =
        Dsv4ToFloat(kv[(uint64_t)b * headDim + d]);
}

__device__ __forceinline__ float DeepSeekV4Sigmoid(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

__device__ __forceinline__ float DeepSeekV4Softplus(float x) {
    if (x > 20.0f) {
        return x;
    }
    if (x < -20.0f) {
        return expf(x);
    }
    return log1pf(expf(x));
}

__global__ void DeepSeekV4RouteScoreTransformKernel(float *logits, int rows, int experts, int mode) {
    int row = blockIdx.x;
    float *rowData = logits + (uint64_t)row * experts;
    if (mode == 0) {
        float mx = -INFINITY;
        for (int e = threadIdx.x; e < experts; e += blockDim.x) {
            mx = fmaxf(mx, rowData[e]);
        }
        __shared__ float red[256];
        red[threadIdx.x] = mx;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        mx = red[0];
        double partial = 0.0;
        for (int e = threadIdx.x; e < experts; e += blockDim.x) {
            partial += (double)expf(rowData[e] - mx);
        }
        red[threadIdx.x] = (float)partial;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] += red[threadIdx.x + stride];
            }
            __syncthreads();
        }
        float sum = red[0];
        for (int e = threadIdx.x; e < experts; e += blockDim.x) {
            rowData[e] = expf(rowData[e] - mx) / sum;
        }
    } else {
        for (int e = threadIdx.x; e < experts; e += blockDim.x) {
            float raw = rowData[e];
            rowData[e] = mode == 1 ? DeepSeekV4Sigmoid(raw) : sqrtf(DeepSeekV4Softplus(raw));
        }
    }
}

__global__ void DeepSeekV4HashRouteScoreKernel(float *logits, const float *tid2eid,
                                               const int *inputIds, int32_t *index,
                                               float *score, int tokens, int experts,
                                               int topk, int mode, float routeScale) {
    int row = blockIdx.x;
    if (row >= tokens) {
        return;
    }
    float *rowData = logits + (uint64_t)row * experts;
    int32_t *outIndex = index + (uint64_t)row * topk;
    float *outScore = score + (uint64_t)row * topk;
    int tokenId = inputIds[row];
    const float *routeRow = tid2eid + (uint64_t)tokenId * topk;

    __shared__ float red[256];
    if (mode == 0) {
        float mx = -INFINITY;
        for (int e = threadIdx.x; e < experts; e += blockDim.x) {
            mx = fmaxf(mx, rowData[e]);
        }
        red[threadIdx.x] = mx;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        mx = red[0];

        float partial = 0.0f;
        for (int e = threadIdx.x; e < experts; e += blockDim.x) {
            partial += expf(rowData[e] - mx);
        }
        red[threadIdx.x] = partial;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] += red[threadIdx.x + stride];
            }
            __syncthreads();
        }
        float denom = fmaxf(red[0], 1e-30f);
        for (int k = threadIdx.x; k < topk; k += blockDim.x) {
            int expert = (int)(routeRow[k] + 0.5f);
            expert = max(0, min(expert, experts - 1));
            outIndex[k] = expert;
            outScore[k] = expf(rowData[expert] - mx) / denom * routeScale;
        }
        return;
    }

    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int k = 0; k < topk; k++) {
            int expert = (int)(routeRow[k] + 0.5f);
            expert = max(0, min(expert, experts - 1));
            float raw = rowData[expert];
            float v = mode == 1 ? DeepSeekV4Sigmoid(raw) : sqrtf(DeepSeekV4Softplus(raw));
            outIndex[k] = expert;
            outScore[k] = v;
            sum += v;
        }
        float invSum = 1.0f / fmaxf(sum, 1e-30f);
        for (int k = 0; k < topk; k++) {
            outScore[k] = outScore[k] * invSum * routeScale;
        }
    }
}

template <typename XT, typename WT>
__global__ void DeepSeekV4HcPreDotsKernel(const XT *x, const WT *w, float *dots,
                                          int tokens, int flatDim, int mixHc) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = tokens * mixHc;
    if (idx >= total) {
        return;
    }
    int m = idx % mixHc;
    int t = idx / mixHc;
    const XT *xrow = x + (uint64_t)t * flatDim;
    const WT *wrow = w + (uint64_t)m * flatDim;
    double v = 0.0;
    for (int k = 0; k < flatDim; k++) {
        v += (double)Dsv4ToFloat(xrow[k]) * Dsv4ToFloat(wrow[k]);
    }
    dots[idx] = (float)v;
}

template <typename XT, typename WT>
__global__ void DeepSeekV4HcPreDotsBlockKernel(const XT *x, const WT *w, float *dots,
                                               int tokens, int flatDim, int mixHc, int dotsStride) {
    extern __shared__ float red[];
    int idx = blockIdx.x;
    if (idx >= tokens * dotsStride) {
        return;
    }
    int m = idx % dotsStride;
    int t = idx / dotsStride;
    const XT *xrow = x + (uint64_t)t * flatDim;
    float partial = 0.0f;
    if (m == mixHc) {
        for (int k = threadIdx.x; k < flatDim; k += blockDim.x) {
            float v = Dsv4ToFloat(xrow[k]);
            partial += v * v;
        }
    } else {
        const WT *wrow = w + (uint64_t)m * flatDim;
        for (int k = threadIdx.x * 2; k + 1 < flatDim; k += blockDim.x * 2) {
            partial += Dsv4PairDot(xrow, wrow, k);
        }
        if ((flatDim & 1) && threadIdx.x == 0) {
            int k = flatDim - 1;
            partial += Dsv4ToFloat(xrow[k]) * Dsv4ToFloat(wrow[k]);
        }
    }
    red[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] += red[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        dots[(uint64_t)t * dotsStride + m] = red[0];
    }
}

template <typename XT>
__global__ void DeepSeekV4HcPreFinishKernel(const XT *x, const float *dots, const float *scale,
                                            const float *base, XT *y, float *post, float *comb,
                                            int tokens, int dim, int hcMult, int sinkhornIters,
                                            float eps, float normEps, int dotsStride) {
    extern __shared__ float shared[];
    float *mixes = shared;
    float *pre = mixes + (2 + hcMult) * hcMult;
    float *combLocal = pre + hcMult;

    int t = blockIdx.x;
    int flatDim = hcMult * dim;
    int mixHc = (2 + hcMult) * hcMult;
    const XT *xrow = x + (uint64_t)t * flatDim;

    float rsqrt = rsqrtf(dots[(uint64_t)t * dotsStride + mixHc] / flatDim + normEps);
    for (int m = threadIdx.x; m < mixHc; m += blockDim.x) {
        mixes[m] = dots[(uint64_t)t * dotsStride + m] * rsqrt;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int h = 0; h < hcMult; h++) {
            pre[h] = DeepSeekV4Sigmoid(mixes[h] * scale[0] + base[h]) + eps;
            post[(uint64_t)t * hcMult + h] =
                2.0f * DeepSeekV4Sigmoid(mixes[h + hcMult] * scale[1] + base[h + hcMult]);
        }

        for (int r = 0; r < hcMult; r++) {
            float rowMax = -INFINITY;
            for (int c = 0; c < hcMult; c++) {
                int idx = r * hcMult + c;
                int mixIdx = idx + 2 * hcMult;
                combLocal[idx] = mixes[mixIdx] * scale[2] + base[mixIdx];
                rowMax = fmaxf(rowMax, combLocal[idx]);
            }
            float rowSum = 0.0f;
            for (int c = 0; c < hcMult; c++) {
                int idx = r * hcMult + c;
                float v = expf(combLocal[idx] - rowMax);
                combLocal[idx] = v;
                rowSum += v;
            }
            for (int c = 0; c < hcMult; c++) {
                int idx = r * hcMult + c;
                combLocal[idx] = combLocal[idx] / rowSum + eps;
            }
        }
        for (int c = 0; c < hcMult; c++) {
            float colSum = 0.0f;
            for (int r = 0; r < hcMult; r++) {
                colSum += combLocal[r * hcMult + c];
            }
            for (int r = 0; r < hcMult; r++) {
                combLocal[r * hcMult + c] /= (colSum + eps);
            }
        }
        for (int it = 1; it < sinkhornIters; it++) {
            for (int r = 0; r < hcMult; r++) {
                float rowSum = 0.0f;
                for (int c = 0; c < hcMult; c++) {
                    rowSum += combLocal[r * hcMult + c];
                }
                for (int c = 0; c < hcMult; c++) {
                    combLocal[r * hcMult + c] /= (rowSum + eps);
                }
            }
            for (int c = 0; c < hcMult; c++) {
                float colSum = 0.0f;
                for (int r = 0; r < hcMult; r++) {
                    colSum += combLocal[r * hcMult + c];
                }
                for (int r = 0; r < hcMult; r++) {
                    combLocal[r * hcMult + c] /= (colSum + eps);
                }
            }
        }
        for (int i = 0; i < hcMult * hcMult; i++) {
            comb[(uint64_t)t * hcMult * hcMult + i] = combLocal[i];
        }
    }
    __syncthreads();

    XT *yrow = y + (uint64_t)t * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float v = 0.0f;
        for (int h = 0; h < hcMult; h++) {
            v += pre[h] * Dsv4ToFloat(xrow[(uint64_t)h * dim + d]);
        }
        yrow[d] = Dsv4FromFloat<XT>(v);
    }
}

template <typename XT, typename WT>
__global__ void DeepSeekV4HcPreKernel(const XT *x, const WT *fn, const float *scale,
                                      const float *base, XT *y, float *post, float *comb,
                                      int tokens, int dim, int hcMult, int sinkhornIters,
                                      float eps, float normEps) {
    extern __shared__ float shared[];
    int t = blockIdx.x;
    int flatDim = hcMult * dim;
    int mixHc = (2 + hcMult) * hcMult;
    float *red = shared;
    float *mixes = red + blockDim.x;
    float *pre = mixes + mixHc;
    float *combLocal = pre + hcMult;

    const XT *xrow = x + (uint64_t)t * flatDim;
    float partial = 0.0f;
    for (int k = threadIdx.x; k < flatDim; k += blockDim.x) {
        float v = Dsv4ToFloat(xrow[k]);
        partial += v * v;
    }
    red[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] += red[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float rsqrt = rsqrtf(red[0] / flatDim + normEps);

    for (int m = 0; m < mixHc; m++) {
        const WT *wrow = fn + (uint64_t)m * flatDim;
        partial = 0.0f;
        for (int k = threadIdx.x; k < flatDim; k += blockDim.x) {
            partial += Dsv4ToFloat(xrow[k]) * Dsv4ToFloat(wrow[k]);
        }
        red[threadIdx.x] = partial;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] += red[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            mixes[m] = red[0] * rsqrt;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        for (int h = 0; h < hcMult; h++) {
            pre[h] = DeepSeekV4Sigmoid(mixes[h] * scale[0] + base[h]) + eps;
            post[(uint64_t)t * hcMult + h] =
                2.0f * DeepSeekV4Sigmoid(mixes[h + hcMult] * scale[1] + base[h + hcMult]);
        }

        for (int r = 0; r < hcMult; r++) {
            float rowMax = -INFINITY;
            for (int c = 0; c < hcMult; c++) {
                int idx = r * hcMult + c;
                int mixIdx = idx + 2 * hcMult;
                combLocal[idx] = mixes[mixIdx] * scale[2] + base[mixIdx];
                rowMax = fmaxf(rowMax, combLocal[idx]);
            }
            float rowSum = 0.0f;
            for (int c = 0; c < hcMult; c++) {
                int idx = r * hcMult + c;
                float v = expf(combLocal[idx] - rowMax);
                combLocal[idx] = v;
                rowSum += v;
            }
            for (int c = 0; c < hcMult; c++) {
                int idx = r * hcMult + c;
                combLocal[idx] = combLocal[idx] / rowSum + eps;
            }
        }
        for (int c = 0; c < hcMult; c++) {
            float colSum = 0.0f;
            for (int r = 0; r < hcMult; r++) {
                colSum += combLocal[r * hcMult + c];
            }
            for (int r = 0; r < hcMult; r++) {
                combLocal[r * hcMult + c] /= (colSum + eps);
            }
        }
        for (int it = 1; it < sinkhornIters; it++) {
            for (int r = 0; r < hcMult; r++) {
                float rowSum = 0.0f;
                for (int c = 0; c < hcMult; c++) {
                    rowSum += combLocal[r * hcMult + c];
                }
                for (int c = 0; c < hcMult; c++) {
                    combLocal[r * hcMult + c] /= (rowSum + eps);
                }
            }
            for (int c = 0; c < hcMult; c++) {
                float colSum = 0.0f;
                for (int r = 0; r < hcMult; r++) {
                    colSum += combLocal[r * hcMult + c];
                }
                for (int r = 0; r < hcMult; r++) {
                    combLocal[r * hcMult + c] /= (colSum + eps);
                }
            }
        }
        for (int i = 0; i < hcMult * hcMult; i++) {
            comb[(uint64_t)t * hcMult * hcMult + i] = combLocal[i];
        }
    }
    __syncthreads();

    XT *yrow = y + (uint64_t)t * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float v = 0.0f;
        for (int h = 0; h < hcMult; h++) {
            v += pre[h] * Dsv4ToFloat(xrow[(uint64_t)h * dim + d]);
        }
        yrow[d] = Dsv4FromFloat<XT>(v);
    }
}

template <typename QT, typename CT>
__global__ void DeepSeekV4SparseAttentionDecodeCachedKernel(const QT *q, const float *windowKV,
                                                            const CT *compressedKV, const float *sink,
                                                            float *output, int bsz, int heads, int dim,
                                                            int windowSize, int startPos, int compressedCount,
                                                            float softmaxScale) {
    int bh = blockIdx.x;
    int b = bh / heads;
    int h = bh % heads;
    const QT *qrow = q + ((uint64_t)b * heads + h) * dim;
    float *orow = output + ((uint64_t)b * heads + h) * dim;

    int idxCount = 0;
    int pos = startPos % windowSize;
    int idxs[512];
    if (startPos >= windowSize - 1) {
        for (int i = pos + 1; i < windowSize; i++) {
            idxs[idxCount++] = i;
        }
        for (int i = 0; i <= pos; i++) {
            idxs[idxCount++] = i;
        }
    } else {
        for (int i = 0; i <= startPos; i++) {
            idxs[idxCount++] = i;
        }
    }
    for (int i = 0; i < compressedCount; i++) {
        idxs[idxCount++] = windowSize + i;
    }

    double scores[512];
    float mx = -INFINITY;
    for (int k = 0; k < idxCount; k++) {
        const float *windowRow = nullptr;
        const CT *compressedRow = nullptr;
        int idx = idxs[k];
        if (idx < windowSize) {
            windowRow = windowKV + ((uint64_t)b * windowSize + idx) * dim;
        } else {
            compressedRow = compressedKV + ((uint64_t)b * compressedCount + (idx - windowSize)) * dim;
        }
        double dot = 0.0;
        for (int d = 0; d < dim; d++) {
            float kv = windowRow != nullptr ? windowRow[d] : Dsv4ToFloat(compressedRow[d]);
            dot += (double)Dsv4ToFloat(qrow[d]) * kv;
        }
        float score = (float)dot * softmaxScale;
        scores[k] = (double)score;
        mx = fmaxf(mx, score);
    }

    float safeMx = isfinite(mx) ? mx : 0.0f;
    double denom = exp((double)sink[h] - safeMx);
    for (int k = 0; k < idxCount; k++) {
        denom += exp(scores[k] - safeMx);
    }

    for (int d = 0; d < dim; d++) {
        double v = 0.0;
        for (int k = 0; k < idxCount; k++) {
            int idx = idxs[k];
            float kv;
            if (idx < windowSize) {
                kv = windowKV[((uint64_t)b * windowSize + idx) * dim + d];
            } else {
                kv = Dsv4ToFloat(compressedKV[((uint64_t)b * compressedCount + (idx - windowSize)) * dim + d]);
            }
            double w = exp(scores[k] - safeMx) / fmax(denom, 1e-30);
            v += w * kv;
        }
        orow[d] = (float)v;
    }
}

template <typename QT, typename CT>
__global__ void DeepSeekV4SparseAttentionDecodeCachedBlockKernel(const QT *q, const float *windowKV,
                                                                 const CT *compressedKV, const float *sink,
                                                                 float *output, int bsz, int heads, int dim,
                                                                 int windowSize, int startPos, int compressedCount,
                                                                 int keyCapacity, float softmaxScale) {
    extern __shared__ float shared[];
    float *scores = shared;
    float *red = shared + keyCapacity;

    int bh = blockIdx.x;
    int b = bh / heads;
    int h = bh % heads;
    const QT *qrow = q + ((uint64_t)b * heads + h) * dim;
    float *orow = output + ((uint64_t)b * heads + h) * dim;

    int liveWindow = startPos >= windowSize - 1 ? windowSize : (startPos + 1);
    int idxCount = liveWindow + compressedCount;
    if (idxCount <= 0 || idxCount > keyCapacity || idxCount > kDeepSeekV4SparseDecodeMaxKeys) {
        return;
    }
    int pos = startPos % windowSize;

    __shared__ float mxShared;
    __shared__ float denomShared;
    if (threadIdx.x == 0) {
        mxShared = -INFINITY;
    }
    __syncthreads();

    for (int k = 0; k < idxCount; k++) {
        int idx;
        if (k < liveWindow) {
            idx = (startPos >= windowSize - 1) ? ((pos + 1 + k) % windowSize) : k;
        } else {
            idx = windowSize + (k - liveWindow);
        }

        float partial = 0.0f;
        for (int d = threadIdx.x; d < dim; d += blockDim.x) {
            float kv;
            if (idx < windowSize) {
                kv = windowKV[((uint64_t)b * windowSize + idx) * dim + d];
            } else {
                kv = Dsv4ToFloat(compressedKV[((uint64_t)b * compressedCount + (idx - windowSize)) * dim + d]);
            }
            partial += Dsv4ToFloat(qrow[d]) * kv;
        }

        red[threadIdx.x] = partial;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] += red[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            float score = red[0] * softmaxScale;
            scores[k] = score;
            mxShared = fmaxf(mxShared, score);
        }
        __syncthreads();
    }

    float safeMx = isfinite(mxShared) ? mxShared : 0.0f;
    float partialDenom = 0.0f;
    for (int k = threadIdx.x; k < idxCount; k += blockDim.x) {
        partialDenom += expf(scores[k] - safeMx);
    }
    red[threadIdx.x] = partialDenom;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] += red[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        denomShared = red[0] + expf(sink[h] - safeMx);
    }
    __syncthreads();

    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float v = 0.0f;
        for (int k = 0; k < idxCount; k++) {
            int idx;
            if (k < liveWindow) {
                idx = (startPos >= windowSize - 1) ? ((pos + 1 + k) % windowSize) : k;
            } else {
                idx = windowSize + (k - liveWindow);
            }
            float kv;
            if (idx < windowSize) {
                kv = windowKV[((uint64_t)b * windowSize + idx) * dim + d];
            } else {
                kv = Dsv4ToFloat(compressedKV[((uint64_t)b * compressedCount + (idx - windowSize)) * dim + d]);
            }
            float w = expf(scores[k] - safeMx) / fmaxf(denomShared, 1e-30f);
            v += w * kv;
        }
        orow[d] = v;
    }
}

template <typename QT, typename CT>
__global__ void DeepSeekV4SparseAttentionDecodeCachedOnlineKernel(const QT *q, const float *windowKV,
                                                                  const CT *compressedKV, const float *sink,
                                                                  float *output, int bsz, int heads, int dim,
                                                                  int windowSize, int startPos, int compressedCount,
                                                                  float softmaxScale) {
    constexpr int kMaxDimsPerThread = 4;
    __shared__ float red[256];
    __shared__ float mxShared;
    __shared__ float denomShared;
    __shared__ float alphaShared;
    __shared__ float betaShared;

    int bh = blockIdx.x;
    int b = bh / heads;
    int h = bh % heads;
    const QT *qrow = q + ((uint64_t)b * heads + h) * dim;
    float *orow = output + ((uint64_t)b * heads + h) * dim;

    int liveWindow = startPos >= windowSize - 1 ? windowSize : (startPos + 1);
    int idxCount = liveWindow + compressedCount;
    if (idxCount <= 0 || idxCount > kDeepSeekV4SparseDecodeMaxKeys ||
        dim > blockDim.x * kMaxDimsPerThread) {
        return;
    }
    int pos = startPos % windowSize;

    int localDims = 0;
    int localOffsets[kMaxDimsPerThread];
    float localAcc[kMaxDimsPerThread];
    float localKV[kMaxDimsPerThread];
    for (int d = threadIdx.x; d < dim && localDims < kMaxDimsPerThread; d += blockDim.x) {
        localOffsets[localDims] = d;
        localAcc[localDims] = 0.0f;
        localKV[localDims] = 0.0f;
        localDims++;
    }

    if (threadIdx.x == 0) {
        mxShared = -INFINITY;
        denomShared = 0.0f;
    }
    __syncthreads();

    for (int k = 0; k < idxCount; k++) {
        int idx = k < liveWindow
                      ? ((startPos >= windowSize - 1) ? ((pos + 1 + k) % windowSize) : k)
                      : (windowSize + (k - liveWindow));

        float partial = 0.0f;
        for (int i = 0; i < localDims; i++) {
            int d = localOffsets[i];
            float kv;
            if (idx < windowSize) {
                kv = windowKV[((uint64_t)b * windowSize + idx) * dim + d];
            } else {
                kv = Dsv4ToFloat(compressedKV[((uint64_t)b * compressedCount + (idx - windowSize)) * dim + d]);
            }
            localKV[i] = kv;
            partial += Dsv4ToFloat(qrow[d]) * kv;
        }

        red[threadIdx.x] = partial;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] += red[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            float score = red[0] * softmaxScale;
            float oldMx = mxShared;
            float newMx = fmaxf(oldMx, score);
            float alpha = isfinite(oldMx) ? expf(oldMx - newMx) : 0.0f;
            float beta = expf(score - newMx);
            mxShared = newMx;
            denomShared = denomShared * alpha + beta;
            alphaShared = alpha;
            betaShared = beta;
        }
        __syncthreads();
        for (int i = 0; i < localDims; i++) {
            localAcc[i] = localAcc[i] * alphaShared + betaShared * localKV[i];
        }
    }

    float finalDenom = denomShared + expf(sink[h] - mxShared);
    finalDenom = fmaxf(finalDenom, 1e-30f);
    for (int i = 0; i < localDims; i++) {
        orow[localOffsets[i]] = localAcc[i] / finalDenom;
    }
}

template <typename QT>
__global__ void DeepSeekV4SparseDecodeConvertQKernel(const QT *q, float *qFloat, uint64_t total) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        qFloat[idx] = Dsv4ToFloat(q[idx]);
    }
}

template <typename CT>
__global__ void DeepSeekV4SparseDecodeGatherKVKernel(const float *windowKV, const CT *compressedKV,
                                                     float *kvFloat, int bsz, int dim, int windowSize,
                                                     int startPos, int compressedCount, int liveWindow,
                                                     int keyCount) {
    uint64_t total = (uint64_t)bsz * keyCount * dim;
    uint64_t linear = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= total) {
        return;
    }
    int d = (int)(linear % dim);
    uint64_t tmp = linear / dim;
    int k = (int)(tmp % keyCount);
    int b = (int)(tmp / keyCount);
    int pos = startPos % windowSize;

    float kv;
    if (k < liveWindow) {
        int idx = (startPos >= windowSize - 1) ? ((pos + 1 + k) % windowSize) : k;
        kv = windowKV[((uint64_t)b * windowSize + idx) * dim + d];
    } else {
        int ck = k - liveWindow;
        kv = Dsv4ToFloat(compressedKV[((uint64_t)b * compressedCount + ck) * dim + d]);
    }
    kvFloat[((uint64_t)b * keyCount + k) * dim + d] = kv;
}

__global__ void DeepSeekV4SparseDecodeSoftmaxSinkKernel(float *scores, const float *sink,
                                                        int bsz, int heads, int keyCount) {
    extern __shared__ float red[];
    int row = blockIdx.x;
    int h = row % heads;
    float *rowScores = scores + (uint64_t)row * keyCount;

    float mx = -INFINITY;
    for (int k = threadIdx.x; k < keyCount; k += blockDim.x) {
        mx = fmaxf(mx, rowScores[k]);
    }
    red[threadIdx.x] = mx;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float safeMx = isfinite(red[0]) ? red[0] : 0.0f;

    float sum = 0.0f;
    for (int k = threadIdx.x; k < keyCount; k += blockDim.x) {
        sum += expf(rowScores[k] - safeMx);
    }
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] += red[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float denom = red[0] + expf(sink[h] - safeMx);
    denom = fmaxf(denom, 1e-30f);

    for (int k = threadIdx.x; k < keyCount; k += blockDim.x) {
        rowScores[k] = expf(rowScores[k] - safeMx) / denom;
    }
}

template <typename QT, typename KT>
__global__ void DeepSeekV4SparseAttentionPrefillBlockKernel(const QT *q, const KT *kv,
                                                            const float *sink, float *output,
                                                            int bsz, int seqlen, int heads, int dim,
                                                            int kvLen, int windowSize, int compressRatio,
                                                            int startPos, int prefixLen, float softmaxScale) {
    extern __shared__ float scores[];

    int row = blockIdx.x;
    int h = row % heads;
    int tmp = row / heads;
    int s = tmp % seqlen;
    int b = tmp / seqlen;
    if (b >= bsz) {
        return;
    }

    int realPrefixLen = max(0, min(prefixLen, kvLen - seqlen));
    int compressedStart = realPrefixLen + seqlen;
    int compressedCount = max(0, kvLen - compressedStart);
    int liveWindow = min(windowSize, realPrefixLen + s + 1);
    int beginPos = startPos + s - liveWindow + 1;
    int prefixStartPos = startPos - realPrefixLen;
    int liveCompressed = 0;
    if (compressRatio > 0 && compressedCount > 0) {
        liveCompressed = min(compressedCount, (startPos + s + 1) / compressRatio);
    }
    int idxCount = liveWindow + liveCompressed;
    if (idxCount <= 0 || idxCount > kDeepSeekV4SparsePrefillMaxKeys) {
        return;
    }

    __shared__ float mxShared;
    __shared__ float denomShared;
    const QT *qrow = q + (((uint64_t)b * seqlen + s) * heads + h) * dim;

    int lane = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;
    for (int base = 0; base < idxCount; base += warps) {
        int k = base + warpId;
        float dot = 0.0f;
        if (k < idxCount) {
            int idx;
            if (k < liveWindow) {
                int pos = beginPos + k;
                idx = (pos < startPos) ? (pos - prefixStartPos) : (realPrefixLen + pos - startPos);
            } else {
                idx = compressedStart + k - liveWindow;
            }
            const KT *kvrow = kv + ((uint64_t)b * kvLen + idx) * dim;
            for (int d = lane; d < dim; d += 32) {
                dot += Dsv4ToFloat(qrow[d]) * Dsv4ToFloat(kvrow[d]);
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        }
        if (lane == 0 && k < idxCount) {
            scores[k] = dot * softmaxScale;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float mx = -INFINITY;
        for (int k = 0; k < idxCount; k++) {
            mx = fmaxf(mx, scores[k]);
        }
        mxShared = isfinite(mx) ? mx : 0.0f;
        float denom = expf(sink[h] - mxShared);
        for (int k = 0; k < idxCount; k++) {
            denom += expf(scores[k] - mxShared);
        }
        denomShared = fmaxf(denom, 1e-30f);
    }
    __syncthreads();

    for (int k = threadIdx.x; k < idxCount; k += blockDim.x) {
        scores[k] = expf(scores[k] - mxShared) / denomShared;
    }
    __syncthreads();

    float *orow = output + (((uint64_t)b * seqlen + s) * heads + h) * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float v = 0.0f;
        for (int k = 0; k < idxCount; k++) {
            int idx;
            if (k < liveWindow) {
                int pos = beginPos + k;
                idx = (pos < startPos) ? (pos - prefixStartPos) : (realPrefixLen + pos - startPos);
            } else {
                idx = compressedStart + k - liveWindow;
            }
            const KT *kvrow = kv + ((uint64_t)b * kvLen + idx) * dim;
            v += scores[k] * Dsv4ToFloat(kvrow[d]);
        }
        orow[d] = v;
    }
}

__global__ void DeepSeekV4SparsePrefillRotaryCastKernel(const float *input, __nv_bfloat16 *output,
                                                        int rows, int seqlen, int heads, int dim,
                                                        int ropeDim, float ropeBase, int startPos,
                                                        int originalSeqLen, float ropeFactor,
                                                        int betaFast, int betaSlow) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    int s = (row / heads) % seqlen;
    int pos = startPos + s;
    const float *src = input + (uint64_t)row * dim;
    __nv_bfloat16 *dst = output + (uint64_t)row * dim;
    int off = dim - ropeDim;
    for (int d = 0; d < off; d++) {
        dst[d] = __float2bfloat16_rn(src[d]);
    }
    for (int i = 0; i < ropeDim; i += 2) {
        float inv = DeepSeekV4InvFreq(i / 2, ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow);
        float ang = pos * inv;
        float c = cosf(ang);
        float sn = -sinf(ang);
        float a = src[off + i];
        float b = src[off + i + 1];
        dst[off + i] = __float2bfloat16_rn(a * c - b * sn);
        dst[off + i + 1] = __float2bfloat16_rn(a * sn + b * c);
    }
}

__global__ void DeepSeekV4SparseDecodeRotaryCastKernel(const float *input, __nv_bfloat16 *output,
                                                       int rows, int dim, int ropeDim,
                                                       float ropeBase, int startPos,
                                                       int originalSeqLen, float ropeFactor,
                                                       int betaFast, int betaSlow) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const float *src = input + (uint64_t)row * dim;
    __nv_bfloat16 *dst = output + (uint64_t)row * dim;
    int off = dim - ropeDim;
    for (int d = 0; d < off; d++) {
        dst[d] = __float2bfloat16_rn(src[d]);
    }
    for (int i = 0; i < ropeDim; i += 2) {
        float inv = DeepSeekV4InvFreq(i / 2, ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow);
        float ang = startPos * inv;
        float c = cosf(ang);
        float sn = -sinf(ang);
        float a = src[off + i];
        float b = src[off + i + 1];
        dst[off + i] = __float2bfloat16_rn(a * c - b * sn);
        dst[off + i + 1] = __float2bfloat16_rn(a * sn + b * c);
    }
}

template <typename XT, typename RT, typename OT>
__global__ void DeepSeekV4HcPostKernel(const XT *x, const RT *residual, const float *post,
                                       const float *comb, OT *output, int tokens,
                                       int hcMult, int dim) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)tokens * hcMult * dim;
    if (idx >= total) {
        return;
    }

    int d = idx % dim;
    uint64_t tmp = idx / dim;
    int target = tmp % hcMult;
    int t = tmp / hcMult;

    double v = (double)post[(uint64_t)t * hcMult + target] * Dsv4ToFloat(x[(uint64_t)t * dim + d]);
    const float *combRow = comb + (uint64_t)t * hcMult * hcMult;
    const RT *resRow = residual + (uint64_t)t * hcMult * dim;
    for (int src = 0; src < hcMult; src++) {
        v += (double)combRow[src * hcMult + target] * Dsv4ToFloat(resRow[(uint64_t)src * dim + d]);
    }
    output[idx] = Dsv4FromFloat<OT>((float)v);
}

bool DeepSeekV4PrepareCudaOutput(fastllm::Data &output, fastllm::DataType dataType,
                                 const std::vector<int> &dims) {
    output.dataType = dataType;
    output.Resize(dims);
    output.ToDevice(fastllm::DataDevice::CUDA, false);
    output.Allocate(false);
    return output.cudaData != nullptr;
}

bool DeepSeekV4CublasDataType(fastllm::DataType dataType, cudaDataType_t &cudaType) {
    if (dataType == fastllm::DataType::BFLOAT16) {
        cudaType = CUDA_R_16BF;
        return true;
    }
    if (dataType == fastllm::DataType::FLOAT16) {
        cudaType = CUDA_R_16F;
        return true;
    }
    if (dataType == fastllm::DataType::FLOAT32) {
        cudaType = CUDA_R_32F;
        return true;
    }
    return false;
}

bool DeepSeekV4LaunchWoAGemm(const fastllm::Data &o, const fastllm::Data &woA,
                             fastllm::Data &output, int bsz, int seqlen,
                             int heads, int headDim, int groups, int oRank) {
    int tokens = bsz * seqlen;
    if (tokens < 16) {
        return false;
    }
    cudaDataType_t oType, wType;
    if (!DeepSeekV4CublasDataType(o.dataType, oType) ||
        !DeepSeekV4CublasDataType(woA.dataType, wType) ||
        output.dataType != fastllm::DataType::BFLOAT16 ||
        heads % groups != 0) {
        return false;
    }

    int headsPerGroup = heads / groups;
    int groupDim = headsPerGroup * headDim;
    int fullDim = heads * headDim;
    float alpha = 1.0f, beta = 0.0f;
    auto handle = getFastllmCublasHandle();

    if (o.dataType == fastllm::DataType::BFLOAT16 && woA.dataType == fastllm::DataType::FLOAT16) {
        half *halfO = (half *)FastllmCudaMalloc(o.Count(0) * sizeof(half));
        half *halfOut = (half *)FastllmCudaMalloc(output.Count(0) * sizeof(half));
        if (halfO != nullptr && halfOut != nullptr) {
            FastllmBF16ToHalf(o.cudaData, halfO, (int)o.Count(0));
            half hAlpha = __float2half(1.0f), hBeta = __float2half(0.0f);
            cublasStatus_t halfStatus = cublasHgemmStridedBatched(
                handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                oRank, tokens, groupDim,
                &hAlpha,
                (const half *)woA.cudaData, groupDim, (long long)oRank * groupDim,
                halfO, fullDim, groupDim,
                &hBeta,
                halfOut, groups * oRank, oRank,
                groups);
            if (halfStatus == CUBLAS_STATUS_SUCCESS) {
                FastllmHalfToBF16(halfOut, output.cudaData, (int)output.Count(0));
                FastllmCudaFree(halfO);
                FastllmCudaFree(halfOut);
                return true;
            }
            if (std::getenv("FASTLLM_DSV4_DEBUG_CUDA_WOA_GEMM") != nullptr) {
                printf("DeepSeekV4WoA half cuBLAS fallback: status=%d tokens=%d groupDim=%d groups=%d oRank=%d\n",
                       (int)halfStatus, tokens, groupDim, groups, oRank);
            }
        }
        if (halfO != nullptr) {
            FastllmCudaFree(halfO);
        }
        if (halfOut != nullptr) {
            FastllmCudaFree(halfOut);
        }
        return false;
    }

    cublasStatus_t status = cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        oRank, tokens, groupDim,
        &alpha,
        woA.cudaData, wType, groupDim, (long long)oRank * groupDim,
        o.cudaData, oType, fullDim, groupDim,
        &beta,
        output.cudaData, CUDA_R_16BF, groups * oRank, oRank,
        groups,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status == CUBLAS_STATUS_SUCCESS) {
        return true;
    }

    if (status != CUBLAS_STATUS_SUCCESS && std::getenv("FASTLLM_DSV4_DEBUG_CUDA_WOA_GEMM") != nullptr) {
        printf("DeepSeekV4WoA cuBLAS fallback: status=%d oType=%d wType=%d tokens=%d groupDim=%d groups=%d oRank=%d\n",
               (int)status, (int)o.dataType, (int)woA.dataType, tokens, groupDim, groups, oRank);
    }
    return false;
}

template <typename InT>
bool DeepSeekV4LaunchWoAByWeight(const fastllm::Data &o, const fastllm::Data &woA, int groups,
                                 int oRank, fastllm::Data &output, int bsz, int seqlen,
                                 int heads, int headDim) {
    if (std::getenv("FASTLLM_DSV4_DISABLE_CUDA_WOA_GEMM") == nullptr &&
        DeepSeekV4LaunchWoAGemm(o, woA, output, bsz, seqlen, heads, headDim, groups, oRank)) {
        return true;
    }
    bool usePair = std::getenv("FASTLLM_DSV4_ENABLE_CUDA_WOA_PAIR") != nullptr && (oRank % 2 == 0);
    bool useFloatAcc = !usePair && std::getenv("FASTLLM_DSV4_ENABLE_CUDA_WOA_FLOAT_ACC") != nullptr;
    bool useKahanAcc = !usePair && !useFloatAcc && std::getenv("FASTLLM_DSV4_ENABLE_CUDA_WOA_KAHAN_ACC") != nullptr;
    bool useBlockReduce = !usePair && !useFloatAcc && !useKahanAcc &&
                          std::getenv("FASTLLM_DSV4_DISABLE_CUDA_WOA_BLOCK") == nullptr;
    bool usePairBlockReduce = useBlockReduce && seqlen == 1 && (oRank % 2 == 0) &&
                              std::getenv("FASTLLM_DSV4_DISABLE_CUDA_WOA_PAIR_BLOCK") == nullptr;
    int total = bsz * seqlen * groups * oRank;
    int threads = std::min(256, std::max(1, total));
    int pairTotal = bsz * seqlen * groups * (oRank / 2);
    int launchTotal = (usePair || usePairBlockReduce) ? pairTotal : total;
    int blocks = (launchTotal + threads - 1) / threads;
    const InT *oData = (const InT *)o.cudaData;
    __nv_bfloat16 *outData = (__nv_bfloat16 *)output.cudaData;

    if (woA.dataType == fastllm::DataType::BFLOAT16) {
        if (usePair) {
            DeepSeekV4WoAPairKernel<<<blocks, threads>>>(oData, (const __nv_bfloat16 *)woA.cudaData, outData,
                                                         bsz, seqlen, heads, headDim, groups, oRank);
        } else if (usePairBlockReduce) {
            DeepSeekV4WoAPairBlockReduceKernel<<<pairTotal, 256, 512 * sizeof(float)>>>(
                oData, (const __nv_bfloat16 *)woA.cudaData, outData,
                bsz, seqlen, heads, headDim, groups, oRank);
        } else if (useBlockReduce) {
            DeepSeekV4WoABlockReduceKernel<<<total, 256, 256 * sizeof(float)>>>(
                oData, (const __nv_bfloat16 *)woA.cudaData, outData,
                bsz, seqlen, heads, headDim, groups, oRank);
        } else if (useFloatAcc) {
            DeepSeekV4WoAFloatAccKernel<<<blocks, threads>>>(oData, (const __nv_bfloat16 *)woA.cudaData, outData,
                                                             bsz, seqlen, heads, headDim, groups, oRank);
        } else if (useKahanAcc) {
            DeepSeekV4WoAKahanAccKernel<<<blocks, threads>>>(oData, (const __nv_bfloat16 *)woA.cudaData, outData,
                                                             bsz, seqlen, heads, headDim, groups, oRank);
        } else {
            DeepSeekV4WoAKernel<<<blocks, threads>>>(oData, (const __nv_bfloat16 *)woA.cudaData, outData,
                                                     bsz, seqlen, heads, headDim, groups, oRank);
        }
    } else if (woA.dataType == fastllm::DataType::FLOAT16) {
        if (usePair) {
            DeepSeekV4WoAPairKernel<<<blocks, threads>>>(oData, (const half *)woA.cudaData, outData,
                                                         bsz, seqlen, heads, headDim, groups, oRank);
        } else if (usePairBlockReduce) {
            DeepSeekV4WoAPairBlockReduceKernel<<<pairTotal, 256, 512 * sizeof(float)>>>(
                oData, (const half *)woA.cudaData, outData,
                bsz, seqlen, heads, headDim, groups, oRank);
        } else if (useBlockReduce) {
            DeepSeekV4WoABlockReduceKernel<<<total, 256, 256 * sizeof(float)>>>(
                oData, (const half *)woA.cudaData, outData,
                bsz, seqlen, heads, headDim, groups, oRank);
        } else if (useFloatAcc) {
            DeepSeekV4WoAFloatAccKernel<<<blocks, threads>>>(oData, (const half *)woA.cudaData, outData,
                                                             bsz, seqlen, heads, headDim, groups, oRank);
        } else if (useKahanAcc) {
            DeepSeekV4WoAKahanAccKernel<<<blocks, threads>>>(oData, (const half *)woA.cudaData, outData,
                                                             bsz, seqlen, heads, headDim, groups, oRank);
        } else {
            DeepSeekV4WoAKernel<<<blocks, threads>>>(oData, (const half *)woA.cudaData, outData,
                                                     bsz, seqlen, heads, headDim, groups, oRank);
        }
    } else if (woA.dataType == fastllm::DataType::FLOAT32) {
        if (usePair) {
            DeepSeekV4WoAPairKernel<<<blocks, threads>>>(oData, (const float *)woA.cudaData, outData,
                                                         bsz, seqlen, heads, headDim, groups, oRank);
        } else if (usePairBlockReduce) {
            DeepSeekV4WoAPairBlockReduceKernel<<<pairTotal, 256, 512 * sizeof(float)>>>(
                oData, (const float *)woA.cudaData, outData,
                bsz, seqlen, heads, headDim, groups, oRank);
        } else if (useBlockReduce) {
            DeepSeekV4WoABlockReduceKernel<<<total, 256, 256 * sizeof(float)>>>(
                oData, (const float *)woA.cudaData, outData,
                bsz, seqlen, heads, headDim, groups, oRank);
        } else if (useFloatAcc) {
            DeepSeekV4WoAFloatAccKernel<<<blocks, threads>>>(oData, (const float *)woA.cudaData, outData,
                                                             bsz, seqlen, heads, headDim, groups, oRank);
        } else if (useKahanAcc) {
            DeepSeekV4WoAKahanAccKernel<<<blocks, threads>>>(oData, (const float *)woA.cudaData, outData,
                                                             bsz, seqlen, heads, headDim, groups, oRank);
        } else {
            DeepSeekV4WoAKernel<<<blocks, threads>>>(oData, (const float *)woA.cudaData, outData,
                                                     bsz, seqlen, heads, headDim, groups, oRank);
        }
    } else {
        return false;
    }
    return true;
}

template <typename XT, typename RT>
bool DeepSeekV4LaunchHcPostByOutput(const fastllm::Data &x, const fastllm::Data &residual,
                                    const float *cudaPost, const float *cudaComb,
                                    fastllm::Data &output, int tokens, int hcMult, int dim) {
    uint64_t total = (uint64_t)tokens * hcMult * dim;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    const XT *xData = (const XT *)x.cudaData;
    const RT *resData = (const RT *)residual.cudaData;

    if (output.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4HcPostKernel<<<blocks, threads>>>(xData, resData, cudaPost, cudaComb,
                                                    (__nv_bfloat16 *)output.cudaData, tokens, hcMult, dim);
    } else if (output.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4HcPostKernel<<<blocks, threads>>>(xData, resData, cudaPost, cudaComb,
                                                    (half *)output.cudaData, tokens, hcMult, dim);
    } else if (output.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4HcPostKernel<<<blocks, threads>>>(xData, resData, cudaPost, cudaComb,
                                                    (float *)output.cudaData, tokens, hcMult, dim);
    } else {
        return false;
    }
    return true;
}

template <typename XT>
bool DeepSeekV4LaunchHcPostByResidual(const fastllm::Data &x, const fastllm::Data &residual,
                                      const float *cudaPost, const float *cudaComb,
                                      fastllm::Data &output, int tokens, int hcMult, int dim) {
    if (residual.dataType == fastllm::DataType::BFLOAT16) {
        return DeepSeekV4LaunchHcPostByOutput<XT, __nv_bfloat16>(x, residual, cudaPost, cudaComb,
                                                                output, tokens, hcMult, dim);
    }
    if (residual.dataType == fastllm::DataType::FLOAT16) {
        return DeepSeekV4LaunchHcPostByOutput<XT, half>(x, residual, cudaPost, cudaComb,
                                                       output, tokens, hcMult, dim);
    }
    if (residual.dataType == fastllm::DataType::FLOAT32) {
        return DeepSeekV4LaunchHcPostByOutput<XT, float>(x, residual, cudaPost, cudaComb,
                                                        output, tokens, hcMult, dim);
    }
    return false;
}

template <typename QT>
bool DeepSeekV4LaunchSparseDecodeCublas(const void *cudaQ, const void *cudaCompressed,
                                        fastllm::DataType compressedType, const float *cudaWindow,
                                        const float *cudaSink, float *outData,
                                        int bsz, int heads, int dim, int windowSize,
                                        int startPos, int compressedCount, int liveWindow,
                                        int keyCount, float softmaxScale) {
    if (std::getenv("FASTLLM_DSV4_DISABLE_CUBLAS_SPARSE_DECODE") != nullptr) {
        return false;
    }
    int maxCublasKeys = 256 * 1024;
    if (const char *env = std::getenv("FASTLLM_DSV4_CUBLAS_SPARSE_DECODE_MAX_KEYS")) {
        maxCublasKeys = std::max(1, std::atoi(env));
    }
    if (keyCount <= 0 || keyCount > maxCublasKeys || dim <= 0 || heads <= 0 || bsz <= 0) {
        return false;
    }

    uint64_t qElems = (uint64_t)bsz * heads * dim;
    uint64_t kvElems = (uint64_t)bsz * keyCount * dim;
    uint64_t scoreElems = (uint64_t)bsz * heads * keyCount;
    float *qFloat = (float *)FastllmCudaMalloc(qElems * sizeof(float));
    float *kvFloat = (float *)FastllmCudaMalloc(kvElems * sizeof(float));
    float *scores = (float *)FastllmCudaMalloc(scoreElems * sizeof(float));
    if (qFloat == nullptr || kvFloat == nullptr || scores == nullptr) {
        if (qFloat != nullptr) {
            FastllmCudaFree(qFloat);
        }
        if (kvFloat != nullptr) {
            FastllmCudaFree(kvFloat);
        }
        if (scores != nullptr) {
            FastllmCudaFree(scores);
        }
        return false;
    }

    int threads = 256;
    DeepSeekV4SparseDecodeConvertQKernel<QT><<<
        (int)((qElems + threads - 1) / threads), threads>>>((const QT *)cudaQ, qFloat, qElems);

    bool gathered = true;
    if (compressedCount == 0 || cudaCompressed == nullptr) {
        DeepSeekV4SparseDecodeGatherKVKernel<float><<<
            (int)((kvElems + threads - 1) / threads), threads>>>(
                cudaWindow, (const float *)nullptr, kvFloat, bsz, dim, windowSize,
                startPos, 0, liveWindow, keyCount);
    } else if (compressedType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4SparseDecodeGatherKVKernel<__nv_bfloat16><<<
            (int)((kvElems + threads - 1) / threads), threads>>>(
                cudaWindow, (const __nv_bfloat16 *)cudaCompressed, kvFloat, bsz, dim, windowSize,
                startPos, compressedCount, liveWindow, keyCount);
    } else if (compressedType == fastllm::DataType::FLOAT16) {
        DeepSeekV4SparseDecodeGatherKVKernel<half><<<
            (int)((kvElems + threads - 1) / threads), threads>>>(
                cudaWindow, (const half *)cudaCompressed, kvFloat, bsz, dim, windowSize,
                startPos, compressedCount, liveWindow, keyCount);
    } else if (compressedType == fastllm::DataType::FLOAT32) {
        DeepSeekV4SparseDecodeGatherKVKernel<float><<<
            (int)((kvElems + threads - 1) / threads), threads>>>(
                cudaWindow, (const float *)cudaCompressed, kvFloat, bsz, dim, windowSize,
                startPos, compressedCount, liveWindow, keyCount);
    } else {
        gathered = false;
    }

    bool ok = false;
    if (gathered) {
        float alpha = softmaxScale;
        float beta = 0.0f;
        auto handle = getFastllmCublasHandle();
        cublasStatus_t status = cublasSgemmStridedBatched(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            keyCount, heads, dim,
            &alpha,
            kvFloat, dim, (long long)keyCount * dim,
            qFloat, dim, (long long)heads * dim,
            &beta,
            scores, keyCount, (long long)heads * keyCount,
            bsz);

        if (status == CUBLAS_STATUS_SUCCESS) {
            DeepSeekV4SparseDecodeSoftmaxSinkKernel<<<bsz * heads, 256, 256 * sizeof(float)>>>(
                scores, cudaSink, bsz, heads, keyCount);

            alpha = 1.0f;
            status = cublasSgemmStridedBatched(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                dim, heads, keyCount,
                &alpha,
                kvFloat, dim, (long long)keyCount * dim,
                scores, keyCount, (long long)heads * keyCount,
                &beta,
                outData, dim, (long long)heads * dim,
                bsz);
        }
        ok = status == CUBLAS_STATUS_SUCCESS;
        if (!ok && std::getenv("FASTLLM_DSV4_DEBUG_CUBLAS_SPARSE_DECODE") != nullptr) {
            std::fprintf(stderr,
                         "DeepSeekV4SparseDecode cuBLAS fallback: status=%d bsz=%d heads=%d dim=%d keys=%d compressed=%d\n",
                         (int)status, bsz, heads, dim, keyCount, compressedCount);
        }
    }

    FastllmCudaFree(qFloat);
    FastllmCudaFree(kvFloat);
    FastllmCudaFree(scores);
    return ok;
}

template <typename QT>
bool DeepSeekV4LaunchSparseDecodeByCompressed(const void *cudaQ, const void *cudaCompressed,
                                              fastllm::DataType compressedType, const float *cudaWindow,
                                              const float *cudaSink, float *outData,
                                              int bsz, int heads, int dim, int windowSize,
                                              int startPos, int compressedCount, float softmaxScale,
                                              int *launchMode = nullptr) {
    const QT *qData = (const QT *)cudaQ;
    int blocks = bsz * heads;
    int liveWindow = startPos >= windowSize - 1 ? windowSize : (startPos + 1);
    int maxKeys = liveWindow + compressedCount;
    if (maxKeys <= 0 || maxKeys > kDeepSeekV4SparseDecodeMaxKeys) {
        return false;
    }
    if (launchMode != nullptr) {
        *launchMode = 0;
    }
    if (DeepSeekV4LaunchSparseDecodeCublas<QT>(
            cudaQ, cudaCompressed, compressedType, cudaWindow, cudaSink, outData,
            bsz, heads, dim, windowSize, startPos, compressedCount, liveWindow,
            maxKeys, softmaxScale)) {
        if (launchMode != nullptr) {
            *launchMode = 2;
        }
        return true;
    }
    bool useOnlineDecode = dim <= 256 * 4 &&
                           std::getenv("FASTLLM_DSV4_DISABLE_ONLINE_SPARSE_DECODE") == nullptr;
    if (useOnlineDecode) {
        if (compressedCount == 0 || cudaCompressed == nullptr) {
            DeepSeekV4SparseAttentionDecodeCachedOnlineKernel<QT, float><<<blocks, 256>>>(
                qData, cudaWindow, (const float *)nullptr, cudaSink, outData,
                bsz, heads, dim, windowSize, startPos, 0, softmaxScale);
        } else if (compressedType == fastllm::DataType::BFLOAT16) {
            DeepSeekV4SparseAttentionDecodeCachedOnlineKernel<QT, __nv_bfloat16><<<blocks, 256>>>(
                qData, cudaWindow, (const __nv_bfloat16 *)cudaCompressed, cudaSink, outData,
                bsz, heads, dim, windowSize, startPos, compressedCount, softmaxScale);
        } else if (compressedType == fastllm::DataType::FLOAT16) {
            DeepSeekV4SparseAttentionDecodeCachedOnlineKernel<QT, half><<<blocks, 256>>>(
                qData, cudaWindow, (const half *)cudaCompressed, cudaSink, outData,
                bsz, heads, dim, windowSize, startPos, compressedCount, softmaxScale);
        } else if (compressedType == fastllm::DataType::FLOAT32) {
            DeepSeekV4SparseAttentionDecodeCachedOnlineKernel<QT, float><<<blocks, 256>>>(
                qData, cudaWindow, (const float *)cudaCompressed, cudaSink, outData,
                bsz, heads, dim, windowSize, startPos, compressedCount, softmaxScale);
        } else {
            return false;
        }
        if (launchMode != nullptr) {
            *launchMode = 1;
        }
        return true;
    }
    size_t sharedBytes = ((size_t)maxKeys + 256) * sizeof(float);
    if (compressedCount == 0 || cudaCompressed == nullptr) {
        auto kernel = DeepSeekV4SparseAttentionDecodeCachedBlockKernel<QT, float>;
        if (!DeepSeekV4EnsureDynamicSharedMemory(kernel, sharedBytes)) {
            return false;
        }
        kernel<<<blocks, 256, sharedBytes>>>(
            qData, cudaWindow, (const float *)nullptr, cudaSink, outData,
            bsz, heads, dim, windowSize, startPos, 0, maxKeys, softmaxScale);
    } else if (compressedType == fastllm::DataType::BFLOAT16) {
        auto kernel = DeepSeekV4SparseAttentionDecodeCachedBlockKernel<QT, __nv_bfloat16>;
        if (!DeepSeekV4EnsureDynamicSharedMemory(kernel, sharedBytes)) {
            return false;
        }
        kernel<<<blocks, 256, sharedBytes>>>(
            qData, cudaWindow, (const __nv_bfloat16 *)cudaCompressed, cudaSink, outData,
            bsz, heads, dim, windowSize, startPos, compressedCount, maxKeys, softmaxScale);
    } else if (compressedType == fastllm::DataType::FLOAT16) {
        auto kernel = DeepSeekV4SparseAttentionDecodeCachedBlockKernel<QT, half>;
        if (!DeepSeekV4EnsureDynamicSharedMemory(kernel, sharedBytes)) {
            return false;
        }
        kernel<<<blocks, 256, sharedBytes>>>(
            qData, cudaWindow, (const half *)cudaCompressed, cudaSink, outData,
            bsz, heads, dim, windowSize, startPos, compressedCount, maxKeys, softmaxScale);
    } else if (compressedType == fastllm::DataType::FLOAT32) {
        auto kernel = DeepSeekV4SparseAttentionDecodeCachedBlockKernel<QT, float>;
        if (!DeepSeekV4EnsureDynamicSharedMemory(kernel, sharedBytes)) {
            return false;
        }
        kernel<<<blocks, 256, sharedBytes>>>(
            qData, cudaWindow, (const float *)cudaCompressed, cudaSink, outData,
            bsz, heads, dim, windowSize, startPos, compressedCount, maxKeys, softmaxScale);
    } else {
        return false;
    }
    return true;
}

template <typename QT, typename KT>
bool DeepSeekV4LaunchSparsePrefillByKV(const fastllm::Data &q, const fastllm::Data &kv,
                                       const float *cudaSink, float *outData,
                                       int bsz, int seqlen, int heads, int dim, int kvLen,
                                       int windowSize, int compressRatio, int startPos, int prefixLen,
                                       float softmaxScale) {
    int realPrefixLen = std::max(0, std::min(prefixLen, kvLen - seqlen));
    int compressedCount = std::max(0, kvLen - realPrefixLen - seqlen);
    int maxKeys = std::min(windowSize, realPrefixLen + seqlen) + (compressRatio > 0 ? compressedCount : 0);
    if (maxKeys <= 0 || maxKeys > kDeepSeekV4SparsePrefillMaxKeys) {
        return false;
    }
    int blocks = bsz * seqlen * heads;
    size_t sharedBytes = (size_t)maxKeys * sizeof(float);
    auto kernel = DeepSeekV4SparseAttentionPrefillBlockKernel<QT, KT>;
    if (!DeepSeekV4EnsureDynamicSharedMemory(kernel, sharedBytes)) {
        return false;
    }
    kernel<<<blocks, 256, sharedBytes>>>(
        (const QT *)q.cudaData, (const KT *)kv.cudaData, cudaSink, outData,
        bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio, startPos, realPrefixLen, softmaxScale);
    return true;
}

template <typename QT>
bool DeepSeekV4LaunchSparsePrefillByQ(const fastllm::Data &q, const fastllm::Data &kv,
                                      const float *cudaSink, float *outData,
                                      int bsz, int seqlen, int heads, int dim, int kvLen,
                                      int windowSize, int compressRatio, int startPos, int prefixLen,
                                      float softmaxScale) {
    if (kv.dataType == fastllm::DataType::BFLOAT16) {
        return DeepSeekV4LaunchSparsePrefillByKV<QT, __nv_bfloat16>(
            q, kv, cudaSink, outData, bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio,
            startPos, prefixLen, softmaxScale);
    }
    if (kv.dataType == fastllm::DataType::FLOAT16) {
        return DeepSeekV4LaunchSparsePrefillByKV<QT, half>(
            q, kv, cudaSink, outData, bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio,
            startPos, prefixLen, softmaxScale);
    }
    if (kv.dataType == fastllm::DataType::FLOAT32) {
        return DeepSeekV4LaunchSparsePrefillByKV<QT, float>(
            q, kv, cudaSink, outData, bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio,
            startPos, prefixLen, softmaxScale);
    }
    return false;
}

template <typename XT>
bool DeepSeekV4LaunchHcPreByWeight(const fastllm::Data &x, const fastllm::Data &hcFn,
                                   const fastllm::Data &hcScale, const fastllm::Data &hcBase,
                                   fastllm::Data &y, fastllm::Data &post, fastllm::Data &comb,
                                   float *dotsData,
                                   int tokens, int dim, int hcMult, int sinkhornIters,
                                   float eps, float normEps) {
    int threads = 256;
    int mixHc = (2 + hcMult) * hcMult;
    int flatDim = hcMult * dim;
    int dotsStride = mixHc + 1;
    int dotBlocks = tokens * dotsStride;
    size_t dotSharedBytes = (size_t)threads * sizeof(float);
    size_t finishSharedBytes = (size_t)(mixHc + hcMult + hcMult * hcMult) * sizeof(float);
    const XT *xData = (const XT *)x.cudaData;
    XT *yData = (XT *)y.cudaData;
    float *postData = (float *)post.cudaData;
    float *combData = (float *)comb.cudaData;
    const float *scaleData = (const float *)hcScale.cudaData;
    const float *baseData = (const float *)hcBase.cudaData;
    if (hcFn.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4HcPreDotsBlockKernel<<<dotBlocks, threads, dotSharedBytes>>>(
            xData, (const __nv_bfloat16 *)hcFn.cudaData, dotsData, tokens, flatDim, mixHc, dotsStride);
    } else if (hcFn.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4HcPreDotsBlockKernel<<<dotBlocks, threads, dotSharedBytes>>>(
            xData, (const half *)hcFn.cudaData, dotsData, tokens, flatDim, mixHc, dotsStride);
    } else if (hcFn.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4HcPreDotsBlockKernel<<<dotBlocks, threads, dotSharedBytes>>>(
            xData, (const float *)hcFn.cudaData, dotsData, tokens, flatDim, mixHc, dotsStride);
    } else {
        return false;
    }
    DeepSeekV4HcPreFinishKernel<<<tokens, threads, finishSharedBytes>>>(
        xData, dotsData, scaleData, baseData, yData, postData, combData,
        tokens, dim, hcMult, sinkhornIters, eps, normEps, dotsStride);
    return true;
}

template <typename InT, typename OutT>
void DeepSeekV4LaunchRMSNormTyped(const fastllm::Data &input, const fastllm::Data &weight,
                                  fastllm::Data &output, int rows, int dim, float eps) {
    int threads = 256;
    DeepSeekV4RMSNormKernel<<<rows, threads, threads * sizeof(float)>>>(
        (const InT *)input.cudaData, (const float *)weight.cudaData,
        (OutT *)output.cudaData, rows, dim, eps);
}

template <typename InT>
bool DeepSeekV4LaunchRMSNormByOutput(const fastllm::Data &input, const fastllm::Data &weight,
                                     fastllm::Data &output, int rows, int dim, float eps) {
    if (output.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4LaunchRMSNormTyped<InT, __nv_bfloat16>(input, weight, output, rows, dim, eps);
    } else if (output.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4LaunchRMSNormTyped<InT, half>(input, weight, output, rows, dim, eps);
    } else if (output.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4LaunchRMSNormTyped<InT, float>(input, weight, output, rows, dim, eps);
    } else {
        return false;
    }
    return true;
}

template <typename XT>
bool DeepSeekV4LaunchHcPreDotsByWeight(const fastllm::Data &x, const fastllm::Data &hcFn,
                                       fastllm::Data &dotsFloat, int tokens, int flatDim, int mixHc) {
    int total = tokens * mixHc;
    int threads = std::min(256, std::max(1, total));
    int blocks = (total + threads - 1) / threads;
    const XT *xData = (const XT *)x.cudaData;
    float *dotsData = (float *)dotsFloat.cudaData;
    if (hcFn.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4HcPreDotsKernel<<<blocks, threads>>>(xData, (const __nv_bfloat16 *)hcFn.cudaData,
                                                       dotsData, tokens, flatDim, mixHc);
    } else if (hcFn.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4HcPreDotsKernel<<<blocks, threads>>>(xData, (const half *)hcFn.cudaData,
                                                       dotsData, tokens, flatDim, mixHc);
    } else if (hcFn.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4HcPreDotsKernel<<<blocks, threads>>>(xData, (const float *)hcFn.cudaData,
                                                       dotsData, tokens, flatDim, mixHc);
    } else {
        return false;
    }
    return true;
}

} // namespace

extern "C" bool FastllmCudaDeepSeekV4HcPre(const fastllm::Data &x, fastllm::Data &hcFn,
                                           fastllm::Data &hcScale, fastllm::Data &hcBase,
                                           int hcMult, int sinkhornIters, float eps, float normEps,
                                           fastllm::Data &y, fastllm::Data &post, fastllm::Data &comb) {
    if (x.dataDevice != fastllm::DataDevice::CUDA || hcFn.dataDevice != fastllm::DataDevice::CUDA ||
        hcScale.dataDevice != fastllm::DataDevice::CUDA || hcBase.dataDevice != fastllm::DataDevice::CUDA ||
        x.dims.size() != 4 || hcMult <= 0 || sinkhornIters <= 0 ||
        hcScale.dataType != fastllm::DataType::FLOAT32 ||
        hcBase.dataType != fastllm::DataType::FLOAT32) {
        return false;
    }
    int bsz = x.dims[0], seqlen = x.dims[1], dim = x.dims[3];
    int flatDim = hcMult * dim;
    int mixHc = (2 + hcMult) * hcMult;
    int tokens = bsz * seqlen;
    if (x.dims[2] != hcMult || hcFn.Count(0) != (uint64_t)mixHc * flatDim ||
        hcScale.Count(0) < 3 || hcBase.Count(0) < (uint64_t)mixHc) {
        return false;
    }
    if (!DeepSeekV4PrepareCudaOutput(y, x.dataType, {bsz, seqlen, dim}) ||
        !DeepSeekV4PrepareCudaOutput(post, fastllm::DataType::FLOAT32, {bsz, seqlen, hcMult}) ||
        !DeepSeekV4PrepareCudaOutput(comb, fastllm::DataType::FLOAT32, {bsz, seqlen, hcMult, hcMult})) {
        return false;
    }
    float *dotsData = (float *)FastllmCudaMalloc((size_t)tokens * (mixHc + 1) * sizeof(float));
    if (dotsData == nullptr) {
        return false;
    }

    bool ok = false;
    if (x.dataType == fastllm::DataType::BFLOAT16) {
        ok = DeepSeekV4LaunchHcPreByWeight<__nv_bfloat16>(
            x, hcFn, hcScale, hcBase, y, post, comb, dotsData,
            tokens, dim, hcMult, sinkhornIters, eps, normEps);
    } else if (x.dataType == fastllm::DataType::FLOAT16) {
        ok = DeepSeekV4LaunchHcPreByWeight<half>(
            x, hcFn, hcScale, hcBase, y, post, comb, dotsData,
            tokens, dim, hcMult, sinkhornIters, eps, normEps);
    } else if (x.dataType == fastllm::DataType::FLOAT32) {
        ok = DeepSeekV4LaunchHcPreByWeight<float>(
            x, hcFn, hcScale, hcBase, y, post, comb, dotsData,
            tokens, dim, hcMult, sinkhornIters, eps, normEps);
    }
    DeviceSync();
    FastllmCudaFree(dotsData);
    return ok;
}

extern "C" bool FastllmCudaDeepSeekV4HcPreDots(const fastllm::Data &x, const fastllm::Data &hcFn,
                                               int hcMult, fastllm::Data &dotsFloat) {
    if (x.dataDevice != fastllm::DataDevice::CUDA || hcFn.dataDevice != fastllm::DataDevice::CUDA ||
        x.dims.size() != 4 || hcMult <= 0) {
        return false;
    }
    int bsz = x.dims[0], seqlen = x.dims[1], dim = x.dims[3];
    int flatDim = hcMult * dim;
    int mixHc = (2 + hcMult) * hcMult;
    int tokens = bsz * seqlen;
    if (x.dims[2] != hcMult || hcFn.Count(0) != (uint64_t)mixHc * flatDim) {
        return false;
    }
    dotsFloat.dataType = fastllm::DataType::FLOAT32;
    dotsFloat.Resize({tokens, mixHc});
    dotsFloat.Allocate(false);
    float *cudaOutput = (float *)FastllmCudaPrepareOutput(dotsFloat);
    dotsFloat.cudaData = cudaOutput;

    bool ok = false;
    if (x.dataType == fastllm::DataType::BFLOAT16) {
        ok = DeepSeekV4LaunchHcPreDotsByWeight<__nv_bfloat16>(x, hcFn, dotsFloat, tokens, flatDim, mixHc);
    } else if (x.dataType == fastllm::DataType::FLOAT16) {
        ok = DeepSeekV4LaunchHcPreDotsByWeight<half>(x, hcFn, dotsFloat, tokens, flatDim, mixHc);
    } else if (x.dataType == fastllm::DataType::FLOAT32) {
        ok = DeepSeekV4LaunchHcPreDotsByWeight<float>(x, hcFn, dotsFloat, tokens, flatDim, mixHc);
    }

    FastllmCudaFinishOutput(dotsFloat, cudaOutput);
    dotsFloat.cudaData = nullptr;
    dotsFloat.dataDevice = fastllm::DataDevice::CPU;
    return ok;
}

extern "C" bool FastllmCudaDeepSeekV4RMSNorm(const fastllm::Data &input, fastllm::Data &weight,
                                             float eps, fastllm::Data &output,
                                             fastllm::DataType outputType) {
    if (input.dataDevice != fastllm::DataDevice::CUDA || weight.dataDevice != fastllm::DataDevice::CUDA ||
        input.dims.empty() || weight.dataType != fastllm::DataType::FLOAT32 ||
        weight.Count(0) != (uint64_t)input.dims.back()) {
        return false;
    }

    bool inPlace = (&input == &output);
    if (inPlace && outputType != input.dataType) {
        return false;
    }
    if (!inPlace && !DeepSeekV4PrepareCudaOutput(output, outputType, input.dims)) {
        return false;
    }

    int dim = input.dims.back();
    int rows = (int)(input.Count(0) / dim);
    bool ok = false;
    if (input.dataType == fastllm::DataType::BFLOAT16) {
        ok = DeepSeekV4LaunchRMSNormByOutput<__nv_bfloat16>(input, weight, output, rows, dim, eps);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        ok = DeepSeekV4LaunchRMSNormByOutput<half>(input, weight, output, rows, dim, eps);
    } else if (input.dataType == fastllm::DataType::FLOAT32) {
        ok = DeepSeekV4LaunchRMSNormByOutput<float>(input, weight, output, rows, dim, eps);
    } else {
        return false;
    }
    DeviceSync();
    return ok;
}

extern "C" bool FastllmCudaDeepSeekV4ScaleQRotary(fastllm::Data &q, int ropeDim, float ropeBase,
                                                  int startPos, int originalSeqLen,
                                                  float ropeFactor, int betaFast, int betaSlow,
                                                  float eps) {
    if (q.dataDevice != fastllm::DataDevice::CUDA || q.dims.size() != 4 ||
        ropeDim <= 0 || ropeDim > q.dims[3]) {
        return false;
    }
    int bsz = q.dims[0], seqlen = q.dims[1], heads = q.dims[2], dim = q.dims[3];
    int rows = bsz * seqlen * heads;
    int threads = 256;
    int blocks = rows;
    size_t sharedBytes = (size_t)threads * sizeof(float);
    if (q.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4ScaleQRotaryBlockKernel<<<blocks, threads, sharedBytes>>>(
            (__nv_bfloat16 *)q.cudaData, rows, seqlen, heads, dim, ropeDim, ropeBase, startPos,
            originalSeqLen, ropeFactor, betaFast, betaSlow, eps);
    } else if (q.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4ScaleQRotaryBlockKernel<<<blocks, threads, sharedBytes>>>(
            (half *)q.cudaData, rows, seqlen, heads, dim, ropeDim, ropeBase, startPos,
            originalSeqLen, ropeFactor, betaFast, betaSlow, eps);
    } else if (q.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4ScaleQRotaryBlockKernel<<<blocks, threads, sharedBytes>>>(
            (float *)q.cudaData, rows, seqlen, heads, dim, ropeDim, ropeBase, startPos,
            originalSeqLen, ropeFactor, betaFast, betaSlow, eps);
    } else {
        return false;
    }
    DeviceSync();
    return true;
}

extern "C" bool FastllmCudaDeepSeekV4RotaryQuant(fastllm::Data &x, int ropeDim, float ropeBase,
                                                 int startPos, int originalSeqLen,
                                                 float ropeFactor, int betaFast, int betaSlow,
                                                 int quantDim, int blockSize, int posStep) {
    if (x.dataDevice != fastllm::DataDevice::CUDA || x.dims.size() < 3 || x.dims.size() > 4 ||
        ropeDim <= 0 || blockSize <= 0) {
        return false;
    }
    int bsz = x.dims[0], seqlen = x.dims[1];
    int heads = x.dims.size() == 4 ? x.dims[2] : 1;
    int dim = x.dims.size() == 4 ? x.dims[3] : x.dims[2];
    if (ropeDim > dim || quantDim < 0 || quantDim > dim) {
        return false;
    }
    int rows = bsz * seqlen * heads;
    int threads = 256;
    int blocks = rows;
    size_t sharedBytes = (size_t)threads * sizeof(float);
    if (x.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4RotaryQuantBlockKernel<<<blocks, threads, sharedBytes>>>(
            (__nv_bfloat16 *)x.cudaData, rows, seqlen, heads, dim, ropeDim, ropeBase, startPos,
            originalSeqLen, ropeFactor, betaFast, betaSlow, quantDim, blockSize, posStep);
    } else if (x.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4RotaryQuantBlockKernel<<<blocks, threads, sharedBytes>>>(
            (half *)x.cudaData, rows, seqlen, heads, dim, ropeDim, ropeBase, startPos,
            originalSeqLen, ropeFactor, betaFast, betaSlow, quantDim, blockSize, posStep);
    } else if (x.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4RotaryQuantBlockKernel<<<blocks, threads, sharedBytes>>>(
            (float *)x.cudaData, rows, seqlen, heads, dim, ropeDim, ropeBase, startPos,
            originalSeqLen, ropeFactor, betaFast, betaSlow, quantDim, blockSize, posStep);
    } else {
        return false;
    }
    DeviceSync();
    return true;
}

extern "C" bool FastllmCudaDeepSeekV4UpdateWindowKVCache(const fastllm::Data &kv, int startPos,
                                                         int windowSize, fastllm::Data &windowKV) {
    if (kv.dataDevice != fastllm::DataDevice::CUDA || kv.dims.size() != 3 || kv.dims[1] != 1 ||
        startPos < 0 || windowSize <= 0) {
        return false;
    }
    int bsz = kv.dims[0], headDim = kv.dims[2];
    if (!DeepSeekV4PrepareCudaOutput(windowKV, fastllm::DataType::FLOAT32, {bsz, windowSize, headDim})) {
        return false;
    }
    int total = bsz * headDim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (kv.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4UpdateWindowKVCacheKernel<<<blocks, threads>>>(
            (__nv_bfloat16 *)kv.cudaData, (float *)windowKV.cudaData, bsz, headDim, startPos, windowSize);
    } else if (kv.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4UpdateWindowKVCacheKernel<<<blocks, threads>>>(
            (half *)kv.cudaData, (float *)windowKV.cudaData, bsz, headDim, startPos, windowSize);
    } else if (kv.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4UpdateWindowKVCacheKernel<<<blocks, threads>>>(
            (float *)kv.cudaData, (float *)windowKV.cudaData, bsz, headDim, startPos, windowSize);
    } else {
        return false;
    }
    DeviceSync();
    return true;
}

extern "C" bool FastllmCudaDeepSeekV4RouteScoreTransform(fastllm::Data &logits, int scoreFuncMode) {
    if (logits.dataDevice != fastllm::DataDevice::CUDA || logits.dataType != fastllm::DataType::FLOAT32 ||
        logits.dims.empty() || scoreFuncMode < 0 || scoreFuncMode > 2) {
        return false;
    }
    int experts = logits.dims.back();
    int rows = (int)(logits.Count(0) / experts);
    if (experts > 256) {
        return false;
    }
    DeepSeekV4RouteScoreTransformKernel<<<rows, 256>>>((float *)logits.cudaData, rows, experts, scoreFuncMode);
    DeviceSync();
    return true;
}

extern "C" bool FastllmCudaDeepSeekV4HashRouteScore(const fastllm::Data &logits,
                                                    fastllm::Data &tid2eid,
                                                    const int *inputIds, int tokens,
                                                    int topk, int scoreFuncMode,
                                                    float routeScale,
                                                    fastllm::Data &expertIndex,
                                                    fastllm::Data &expertScore) {
    if (logits.dataDevice != fastllm::DataDevice::CUDA ||
        logits.dataType != fastllm::DataType::FLOAT32 ||
        logits.dims.empty() || inputIds == nullptr ||
        tokens <= 0 || topk <= 0 || scoreFuncMode < 0 || scoreFuncMode > 2 ||
        tid2eid.dataType != fastllm::DataType::FLOAT32) {
        return false;
    }
    int experts = logits.dims.back();
    int rows = (int)(logits.Count(0) / experts);
    if (rows != tokens || experts <= 0 || experts > 256) {
        return false;
    }
    int maxInputId = -1;
    for (int i = 0; i < tokens; i++) {
        if (inputIds[i] < 0) {
            return false;
        }
        maxInputId = std::max(maxInputId, inputIds[i]);
    }
    if (tid2eid.Count(0) < (uint64_t)(maxInputId + 1) * topk) {
        return false;
    }
    tid2eid.ToDevice(fastllm::DataDevice::CUDA);
    if (tid2eid.cudaData == nullptr ||
        !DeepSeekV4PrepareCudaOutput(expertIndex, fastllm::DataType::INT32, {tokens, topk}) ||
        !DeepSeekV4PrepareCudaOutput(expertScore, fastllm::DataType::FLOAT32, {tokens, topk})) {
        return false;
    }

    size_t inputBytes = (size_t)tokens * sizeof(int);
    int *cudaInputIds = (int *)FastllmCudaMalloc(inputBytes);
    if (cudaInputIds == nullptr) {
        return false;
    }
    FastllmCudaCopyFromHostToDevice(cudaInputIds, (void *)inputIds, inputBytes);
    DeepSeekV4HashRouteScoreKernel<<<tokens, 256>>>(
        (float *)logits.cudaData, (const float *)tid2eid.cudaData, cudaInputIds,
        (int32_t *)expertIndex.cudaData, (float *)expertScore.cudaData,
        tokens, experts, topk, scoreFuncMode, routeScale);
    DeviceSync();
    FastllmCudaFree(cudaInputIds);
    return true;
}

extern "C" bool FastllmCudaDeepSeekV4SparseAttentionPrefill(const fastllm::Data &q,
                                                            const fastllm::Data &kv,
                                                            fastllm::Data &attnSink,
                                                            int windowSize, int startPos,
                                                            int compressRatio,
                                                            int ropeDim, float ropeBase,
                                                            int originalSeqLen,
                                                            float ropeFactor, int betaFast,
                                                            int betaSlow, float softmaxScale,
                                                            fastllm::Data &output, int prefixLen) {
    if (q.dataDevice != fastllm::DataDevice::CUDA ||
        kv.dataDevice != fastllm::DataDevice::CUDA ||
        attnSink.dataDevice != fastllm::DataDevice::CUDA ||
        attnSink.dataType != fastllm::DataType::FLOAT32 ||
        q.dims.size() != 4 || kv.dims.size() != 3 ||
        windowSize <= 0 || ropeDim <= 0 || ropeDim > q.dims[3]) {
        return false;
    }
    int bsz = q.dims[0], seqlen = q.dims[1], heads = q.dims[2], dim = q.dims[3];
    int kvLen = kv.dims[1];
    int realPrefixLen = std::max(0, std::min(prefixLen, kvLen - seqlen));
    if (seqlen <= 0 || kv.dims[0] != bsz || kv.dims[2] != dim ||
        kvLen < realPrefixLen + seqlen || attnSink.Count(0) != (uint64_t)heads) {
        return false;
    }
    int compressedCount = std::max(0, kvLen - realPrefixLen - seqlen);
    int maxKeys = std::min(windowSize, realPrefixLen + seqlen) + (compressRatio > 0 ? compressedCount : 0);
    if (maxKeys <= 0 || maxKeys > kDeepSeekV4SparsePrefillMaxKeys) {
        return false;
    }
    if (!DeepSeekV4PrepareCudaOutput(output, fastllm::DataType::BFLOAT16,
                                     {bsz, seqlen, heads, dim})) {
        return false;
    }

    size_t tempBytes = (size_t)bsz * seqlen * heads * dim * sizeof(float);
    float *cudaTemp = (float *)FastllmCudaMalloc(tempBytes);
    if (cudaTemp == nullptr) {
        return false;
    }

    bool ok = false;
    if (q.dataType == fastllm::DataType::BFLOAT16) {
        ok = DeepSeekV4LaunchSparsePrefillByQ<__nv_bfloat16>(
            q, kv, (const float *)attnSink.cudaData, cudaTemp,
            bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio, startPos, realPrefixLen, softmaxScale);
    } else if (q.dataType == fastllm::DataType::FLOAT16) {
        ok = DeepSeekV4LaunchSparsePrefillByQ<half>(
            q, kv, (const float *)attnSink.cudaData, cudaTemp,
            bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio, startPos, realPrefixLen, softmaxScale);
    } else if (q.dataType == fastllm::DataType::FLOAT32) {
        ok = DeepSeekV4LaunchSparsePrefillByQ<float>(
            q, kv, (const float *)attnSink.cudaData, cudaTemp,
            bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio, startPos, realPrefixLen, softmaxScale);
    }
    if (ok) {
        int rows = bsz * seqlen * heads;
        int threads = 128;
        int blocks = (rows + threads - 1) / threads;
        DeepSeekV4SparsePrefillRotaryCastKernel<<<blocks, threads>>>(
            cudaTemp, (__nv_bfloat16 *)output.cudaData, rows, seqlen, heads, dim,
            ropeDim, ropeBase, startPos, originalSeqLen, ropeFactor, betaFast, betaSlow);
        DeviceSync();
    }

    FastllmCudaFree(cudaTemp);
    return ok;
}

extern "C" bool FastllmCudaDeepSeekV4SparseAttentionDecodeCached(const fastllm::Data &q,
                                                                 const fastllm::Data *windowKVData,
                                                                 const float *windowKV,
                                                                 const fastllm::Data &compressedKV,
                                                                 fastllm::Data &attnSink,
                                                                 int windowSize, int startPos,
                                                                 int compressedCount,
                                                                 int ropeDim, float ropeBase,
                                                                 int originalSeqLen,
                                                                 float ropeFactor, int betaFast,
                                                                 int betaSlow, float softmaxScale,
                                                                 fastllm::Data &output) {
    if (q.dims.size() != 4 || q.dims[1] != 1 ||
        windowSize <= 0 || compressedCount < 0 ||
        windowSize + compressedCount > kDeepSeekV4SparseDecodeMaxKeys ||
        ropeDim <= 0 || ropeDim > q.dims[3] ||
        q.dataDevice != fastllm::DataDevice::CUDA ||
        attnSink.dataDevice != fastllm::DataDevice::CUDA ||
        attnSink.dataType != fastllm::DataType::FLOAT32 ||
        (compressedCount > 0 && compressedKV.dataDevice != fastllm::DataDevice::CUDA)) {
        return false;
    }
    int bsz = q.dims[0], heads = q.dims[2], dim = q.dims[3];
    if (attnSink.Count(0) != (uint64_t)heads ||
        (compressedCount > 0 && compressedKV.Count(0) != (uint64_t)bsz * compressedCount * dim)) {
        return false;
    }
    const float *cudaWindow = nullptr;
    bool ownCudaWindow = false;
    if (windowKVData != nullptr && windowKVData->dataDevice == fastllm::DataDevice::CUDA &&
        windowKVData->dataType == fastllm::DataType::FLOAT32 &&
        windowKVData->Count(0) == (uint64_t)bsz * windowSize * dim) {
        cudaWindow = (const float *)windowKVData->cudaData;
    } else if (windowKV != nullptr) {
        size_t windowBytes = (size_t)bsz * windowSize * dim * sizeof(float);
        float *tmpWindow = (float *)FastllmCudaMalloc(windowBytes);
        FastllmCudaCopyFromHostToDevice(tmpWindow, (void *)windowKV, windowBytes);
        cudaWindow = tmpWindow;
        ownCudaWindow = true;
    } else {
        return false;
    }

    if (!DeepSeekV4PrepareCudaOutput(output, fastllm::DataType::BFLOAT16, {bsz, 1, heads, dim})) {
        if (ownCudaWindow) {
            FastllmCudaFree((void *)cudaWindow);
        }
        return false;
    }

    size_t tempBytes = (size_t)bsz * heads * dim * sizeof(float);
    float *cudaTemp = (float *)FastllmCudaMalloc(tempBytes);

    bool ok = false;
    bool sparseDecodeStats = std::getenv("FASTLLM_DSV4_SPARSE_DECODE_STATS") != nullptr;
    auto sparseDecodeStart = std::chrono::steady_clock::now();
    int sparseDecodeMode = 0;
    if (q.dataType == fastllm::DataType::BFLOAT16) {
        ok = DeepSeekV4LaunchSparseDecodeByCompressed<__nv_bfloat16>(
            q.cudaData, compressedCount > 0 ? compressedKV.cudaData : nullptr, compressedKV.dataType,
            cudaWindow, (const float *)attnSink.cudaData, cudaTemp,
            bsz, heads, dim, windowSize, startPos, compressedCount, softmaxScale, &sparseDecodeMode);
    } else if (q.dataType == fastllm::DataType::FLOAT16) {
        ok = DeepSeekV4LaunchSparseDecodeByCompressed<half>(
            q.cudaData, compressedCount > 0 ? compressedKV.cudaData : nullptr, compressedKV.dataType,
            cudaWindow, (const float *)attnSink.cudaData, cudaTemp,
            bsz, heads, dim, windowSize, startPos, compressedCount, softmaxScale, &sparseDecodeMode);
    } else if (q.dataType == fastllm::DataType::FLOAT32) {
        ok = DeepSeekV4LaunchSparseDecodeByCompressed<float>(
            q.cudaData, compressedCount > 0 ? compressedKV.cudaData : nullptr, compressedKV.dataType,
            cudaWindow, (const float *)attnSink.cudaData, cudaTemp,
            bsz, heads, dim, windowSize, startPos, compressedCount, softmaxScale, &sparseDecodeMode);
    }
    if (ok) {
        int rows = bsz * heads;
        int threads = 128;
        int blocks = (rows + threads - 1) / threads;
        DeepSeekV4SparseDecodeRotaryCastKernel<<<blocks, threads>>>(
            cudaTemp, (__nv_bfloat16 *)output.cudaData, rows, dim, ropeDim, ropeBase,
            startPos, originalSeqLen, ropeFactor, betaFast, betaSlow);
        DeviceSync();
    }
    if (sparseDecodeStats) {
        auto sparseDecodeEnd = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(sparseDecodeEnd - sparseDecodeStart).count();
        int liveWindow = startPos >= windowSize - 1 ? windowSize : (startPos + 1);
        int kvPasses = sparseDecodeMode == 0 ? 2 : 1;
        double windowBytes = (double)bsz * heads * dim * liveWindow * sizeof(float) * kvPasses;
        double compressedBytes = compressedCount > 0
                                     ? (double)bsz * heads * fastllm::GetDataBytes(compressedKV.dataType,
                                                                                   compressedCount, dim) * kvPasses
                                     : 0.0;
        double kvGB = (windowBytes + compressedBytes) / 1.0e9;
        static std::mutex statMutex;
        static unsigned long long statCalls = 0;
        static double statMs = 0.0;
        static double statKvGB = 0.0;
        std::lock_guard<std::mutex> statLock(statMutex);
        statCalls++;
        statMs += ms;
        statKvGB += kvGB;
        int printEvery = 43;
        if (const char *envEvery = std::getenv("FASTLLM_DSV4_SPARSE_DECODE_STATS_EVERY")) {
            printEvery = std::max(1, std::atoi(envEvery));
        }
        if (statCalls % (unsigned long long)printEvery == 0) {
            double avgMs = statMs / printEvery;
            double avgKvGB = statKvGB / printEvery;
            double avgGBps = avgMs > 0.0 ? avgKvGB / (avgMs / 1000.0) : 0.0;
            std::fprintf(stderr,
                         "DeepSeekV4SparseDecodeCachedStats calls=%llu last_ms=%.4f avg_ms=%.4f "
                         "last_kv_gb=%.6f avg_kv_gb=%.6f avg_kv_gbps=%.2f startPos=%d "
                         "liveWindow=%d compressedCount=%d dim=%d heads=%d kvPasses=%d mode=%s\n",
                         statCalls, ms, avgMs, kvGB, avgKvGB, avgGBps, startPos,
                         liveWindow, compressedCount, dim, heads, kvPasses,
                         sparseDecodeMode == 2 ? "cublas" : (sparseDecodeMode == 1 ? "online" : "two-pass"));
            std::fflush(stderr);
            statMs = 0.0;
            statKvGB = 0.0;
        }
    }

    if (ownCudaWindow) {
        FastllmCudaFree((void *)cudaWindow);
    }
    FastllmCudaFree(cudaTemp);
    return ok;
}

extern "C" bool FastllmCudaDeepSeekV4WoA(const fastllm::Data &o, const fastllm::Data &woA,
                                         int groups, int oRank, fastllm::Data &output) {
    if (o.dataDevice != fastllm::DataDevice::CUDA || woA.dataDevice != fastllm::DataDevice::CUDA ||
        o.dims.size() != 4 || groups <= 0 || oRank <= 0) {
        return false;
    }

    int bsz = o.dims[0], seqlen = o.dims[1], heads = o.dims[2], headDim = o.dims[3];
    if (heads % groups != 0) {
        return false;
    }
    int groupDim = (heads / groups) * headDim;
    if (woA.Count(0) != (uint64_t)groups * oRank * groupDim) {
        return false;
    }

    if (!DeepSeekV4PrepareCudaOutput(output, fastllm::DataType::BFLOAT16,
                                     {bsz, seqlen, groups * oRank})) {
        return false;
    }

    bool ok = false;
    if (o.dataType == fastllm::DataType::BFLOAT16) {
        ok = DeepSeekV4LaunchWoAByWeight<__nv_bfloat16>(o, woA, groups, oRank, output, bsz, seqlen, heads, headDim);
    } else if (o.dataType == fastllm::DataType::FLOAT16) {
        ok = DeepSeekV4LaunchWoAByWeight<half>(o, woA, groups, oRank, output, bsz, seqlen, heads, headDim);
    } else if (o.dataType == fastllm::DataType::FLOAT32) {
        ok = DeepSeekV4LaunchWoAByWeight<float>(o, woA, groups, oRank, output, bsz, seqlen, heads, headDim);
    }
    if (!ok) {
        return false;
    }
    DeviceSync();
    return true;
}

extern "C" bool FastllmCudaDeepSeekV4HcPost(const fastllm::Data &x, const fastllm::Data &residual,
                                            const float *post, const float *comb, int bsz,
                                            int seqlen, int hcMult, int dim, fastllm::Data &output) {
    if (x.dataDevice != fastllm::DataDevice::CUDA || residual.dataDevice != fastllm::DataDevice::CUDA ||
        post == nullptr || comb == nullptr || bsz <= 0 || seqlen <= 0 || hcMult <= 0 || dim <= 0 ||
        x.Count(0) != (uint64_t)bsz * seqlen * dim ||
        residual.Count(0) != (uint64_t)bsz * seqlen * hcMult * dim) {
        return false;
    }

    if (!DeepSeekV4PrepareCudaOutput(output, x.dataType, {bsz, seqlen, hcMult, dim})) {
        return false;
    }

    int tokens = bsz * seqlen;
    size_t postBytes = (size_t)tokens * hcMult * sizeof(float);
    size_t combBytes = (size_t)tokens * hcMult * hcMult * sizeof(float);
    float *cudaPost = (float *)FastllmCudaMalloc(postBytes);
    float *cudaComb = (float *)FastllmCudaMalloc(combBytes);
    FastllmCudaCopyFromHostToDevice(cudaPost, (void *)post, postBytes);
    FastllmCudaCopyFromHostToDevice(cudaComb, (void *)comb, combBytes);

    bool ok = false;
    if (x.dataType == fastllm::DataType::BFLOAT16) {
        ok = DeepSeekV4LaunchHcPostByResidual<__nv_bfloat16>(x, residual, cudaPost, cudaComb,
                                                            output, tokens, hcMult, dim);
    } else if (x.dataType == fastllm::DataType::FLOAT16) {
        ok = DeepSeekV4LaunchHcPostByResidual<half>(x, residual, cudaPost, cudaComb,
                                                   output, tokens, hcMult, dim);
    } else if (x.dataType == fastllm::DataType::FLOAT32) {
        ok = DeepSeekV4LaunchHcPostByResidual<float>(x, residual, cudaPost, cudaComb,
                                                    output, tokens, hcMult, dim);
    }

    DeviceSync();
    FastllmCudaFree(cudaPost);
    FastllmCudaFree(cudaComb);
    return ok;
}

extern "C" bool FastllmCudaDeepSeekV4HcPostCudaMix(const fastllm::Data &x, const fastllm::Data &residual,
                                                   const fastllm::Data &post, const fastllm::Data &comb,
                                                   int bsz, int seqlen, int hcMult, int dim,
                                                   fastllm::Data &output) {
    if (x.dataDevice != fastllm::DataDevice::CUDA || residual.dataDevice != fastllm::DataDevice::CUDA ||
        post.dataDevice != fastllm::DataDevice::CUDA || comb.dataDevice != fastllm::DataDevice::CUDA ||
        post.dataType != fastllm::DataType::FLOAT32 || comb.dataType != fastllm::DataType::FLOAT32 ||
        bsz <= 0 || seqlen <= 0 || hcMult <= 0 || dim <= 0 ||
        x.Count(0) != (uint64_t)bsz * seqlen * dim ||
        residual.Count(0) != (uint64_t)bsz * seqlen * hcMult * dim ||
        post.Count(0) != (uint64_t)bsz * seqlen * hcMult ||
        comb.Count(0) != (uint64_t)bsz * seqlen * hcMult * hcMult) {
        return false;
    }

    if (!DeepSeekV4PrepareCudaOutput(output, x.dataType, {bsz, seqlen, hcMult, dim})) {
        return false;
    }

    int tokens = bsz * seqlen;
    bool ok = false;
    const float *cudaPost = (const float *)post.cudaData;
    const float *cudaComb = (const float *)comb.cudaData;
    if (x.dataType == fastllm::DataType::BFLOAT16) {
        ok = DeepSeekV4LaunchHcPostByResidual<__nv_bfloat16>(x, residual, cudaPost, cudaComb,
                                                            output, tokens, hcMult, dim);
    } else if (x.dataType == fastllm::DataType::FLOAT16) {
        ok = DeepSeekV4LaunchHcPostByResidual<half>(x, residual, cudaPost, cudaComb,
                                                   output, tokens, hcMult, dim);
    } else if (x.dataType == fastllm::DataType::FLOAT32) {
        ok = DeepSeekV4LaunchHcPostByResidual<float>(x, residual, cudaPost, cudaComb,
                                                    output, tokens, hcMult, dim);
    }
    DeviceSync();
    return ok;
}
