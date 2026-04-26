#include "fastllm-cuda.cuh"
#include "fastllm.h"

#include <algorithm>
#include <cstdlib>

namespace {

__device__ __forceinline__ float Dsv4ToFloat(float v) {
    return v;
}

__device__ __forceinline__ float Dsv4ToFloat(half v) {
    return __half2float(v);
}

__device__ __forceinline__ float Dsv4ToFloat(__nv_bfloat16 v) {
    return __bfloat162float(v);
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

    double v = 0.0;
    int d = 0;
    for (int hh = 0; hh < headsPerGroup; hh++) {
        const InT *src = o + (((uint64_t)b * seqlen + s) * heads + g * headsPerGroup + hh) * headDim;
        for (int localD = 0; localD < headDim; localD++, d++) {
            v += (double)Dsv4ToFloat(src[localD]) * Dsv4ToFloat(wrow[d]);
        }
    }
    output[idx] = __float2bfloat16_rn((float)v);
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

template <typename T>
__global__ void DeepSeekV4RMSNormKernel(const T *input, const float *weight, T *output,
                                        int rows, int dim, float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const T *src = input + (uint64_t)row * dim;
    T *dst = output + (uint64_t)row * dim;
    double ss = 0.0;
    for (int d = 0; d < dim; d++) {
        double v = (double)Dsv4ToFloat(src[d]);
        ss += v * v;
    }
    float scale = 1.0f / sqrtf((float)(ss / dim) + eps);
    for (int d = 0; d < dim; d++) {
        dst[d] = Dsv4FromFloat<T>(Dsv4ToFloat(src[d]) * scale * weight[d]);
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

template <typename InT>
bool DeepSeekV4LaunchWoAByWeight(const fastllm::Data &o, const fastllm::Data &woA, int groups,
                                 int oRank, fastllm::Data &output, int bsz, int seqlen,
                                 int heads, int headDim) {
    bool usePair = std::getenv("FASTLLM_DSV4_ENABLE_CUDA_WOA_PAIR") != nullptr && (oRank % 2 == 0);
    int total = bsz * seqlen * groups * oRank;
    int threads = std::min(256, std::max(1, total));
    int pairTotal = bsz * seqlen * groups * (oRank / 2);
    int launchTotal = usePair ? pairTotal : total;
    int blocks = (launchTotal + threads - 1) / threads;
    const InT *oData = (const InT *)o.cudaData;
    __nv_bfloat16 *outData = (__nv_bfloat16 *)output.cudaData;

    if (woA.dataType == fastllm::DataType::BFLOAT16) {
        if (usePair) {
            DeepSeekV4WoAPairKernel<<<blocks, threads>>>(oData, (const __nv_bfloat16 *)woA.cudaData, outData,
                                                         bsz, seqlen, heads, headDim, groups, oRank);
        } else {
            DeepSeekV4WoAKernel<<<blocks, threads>>>(oData, (const __nv_bfloat16 *)woA.cudaData, outData,
                                                     bsz, seqlen, heads, headDim, groups, oRank);
        }
    } else if (woA.dataType == fastllm::DataType::FLOAT16) {
        if (usePair) {
            DeepSeekV4WoAPairKernel<<<blocks, threads>>>(oData, (const half *)woA.cudaData, outData,
                                                         bsz, seqlen, heads, headDim, groups, oRank);
        } else {
            DeepSeekV4WoAKernel<<<blocks, threads>>>(oData, (const half *)woA.cudaData, outData,
                                                     bsz, seqlen, heads, headDim, groups, oRank);
        }
    } else if (woA.dataType == fastllm::DataType::FLOAT32) {
        if (usePair) {
            DeepSeekV4WoAPairKernel<<<blocks, threads>>>(oData, (const float *)woA.cudaData, outData,
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
bool DeepSeekV4LaunchSparseDecodeByCompressed(const void *cudaQ, const void *cudaCompressed,
                                              fastllm::DataType compressedType, const float *cudaWindow,
                                              const float *cudaSink, fastllm::Data &outputFloat,
                                              int bsz, int heads, int dim, int windowSize,
                                              int startPos, int compressedCount, float softmaxScale) {
    const QT *qData = (const QT *)cudaQ;
    float *outData = (float *)outputFloat.cudaData;
    int blocks = bsz * heads;
    if (compressedCount == 0 || cudaCompressed == nullptr) {
        DeepSeekV4SparseAttentionDecodeCachedKernel<<<blocks, 1>>>(
            qData, cudaWindow, (const float *)nullptr, cudaSink, outData,
            bsz, heads, dim, windowSize, startPos, 0, softmaxScale);
    } else if (compressedType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4SparseAttentionDecodeCachedKernel<<<blocks, 1>>>(
            qData, cudaWindow, (const __nv_bfloat16 *)cudaCompressed, cudaSink, outData,
            bsz, heads, dim, windowSize, startPos, compressedCount, softmaxScale);
    } else if (compressedType == fastllm::DataType::FLOAT16) {
        DeepSeekV4SparseAttentionDecodeCachedKernel<<<blocks, 1>>>(
            qData, cudaWindow, (const half *)cudaCompressed, cudaSink, outData,
            bsz, heads, dim, windowSize, startPos, compressedCount, softmaxScale);
    } else if (compressedType == fastllm::DataType::FLOAT32) {
        DeepSeekV4SparseAttentionDecodeCachedKernel<<<blocks, 1>>>(
            qData, cudaWindow, (const float *)cudaCompressed, cudaSink, outData,
            bsz, heads, dim, windowSize, startPos, compressedCount, softmaxScale);
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
        input.dims.empty() || outputType != input.dataType || weight.dataType != fastllm::DataType::FLOAT32 ||
        weight.Count(0) != (uint64_t)input.dims.back()) {
        return false;
    }

    bool inPlace = (&input == &output);
    if (!inPlace && !DeepSeekV4PrepareCudaOutput(output, outputType, input.dims)) {
        return false;
    }

    int dim = input.dims.back();
    int rows = (int)(input.Count(0) / dim);
    int threads = 128;
    int blocks = (rows + threads - 1) / threads;
    if (input.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4RMSNormKernel<<<blocks, threads>>>((const __nv_bfloat16 *)input.cudaData,
                                                     (const float *)weight.cudaData,
                                                     (__nv_bfloat16 *)output.cudaData, rows, dim, eps);
    } else if (input.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4RMSNormKernel<<<blocks, threads>>>((const half *)input.cudaData,
                                                     (const float *)weight.cudaData,
                                                     (half *)output.cudaData, rows, dim, eps);
    } else if (input.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4RMSNormKernel<<<blocks, threads>>>((const float *)input.cudaData,
                                                     (const float *)weight.cudaData,
                                                     (float *)output.cudaData, rows, dim, eps);
    } else {
        return false;
    }
    DeviceSync();
    return true;
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
    int threads = 128;
    int blocks = (rows + threads - 1) / threads;
    if (q.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4ScaleQRotaryKernel<<<blocks, threads>>>((__nv_bfloat16 *)q.cudaData, rows, seqlen,
                                                          heads, dim, ropeDim, ropeBase, startPos,
                                                          originalSeqLen, ropeFactor, betaFast, betaSlow, eps);
    } else if (q.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4ScaleQRotaryKernel<<<blocks, threads>>>((half *)q.cudaData, rows, seqlen,
                                                          heads, dim, ropeDim, ropeBase, startPos,
                                                          originalSeqLen, ropeFactor, betaFast, betaSlow, eps);
    } else if (q.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4ScaleQRotaryKernel<<<blocks, threads>>>((float *)q.cudaData, rows, seqlen,
                                                          heads, dim, ropeDim, ropeBase, startPos,
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
    int threads = 128;
    int blocks = (rows + threads - 1) / threads;
    if (x.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4RotaryQuantKernel<<<blocks, threads>>>((__nv_bfloat16 *)x.cudaData, rows, seqlen,
                                                         heads, dim, ropeDim, ropeBase, startPos,
                                                         originalSeqLen, ropeFactor, betaFast, betaSlow,
                                                         quantDim, blockSize, posStep);
    } else if (x.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4RotaryQuantKernel<<<blocks, threads>>>((half *)x.cudaData, rows, seqlen,
                                                         heads, dim, ropeDim, ropeBase, startPos,
                                                         originalSeqLen, ropeFactor, betaFast, betaSlow,
                                                         quantDim, blockSize, posStep);
    } else if (x.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4RotaryQuantKernel<<<blocks, threads>>>((float *)x.cudaData, rows, seqlen,
                                                         heads, dim, ropeDim, ropeBase, startPos,
                                                         originalSeqLen, ropeFactor, betaFast, betaSlow,
                                                         quantDim, blockSize, posStep);
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

extern "C" bool FastllmCudaDeepSeekV4SparseAttentionDecodeCached(const fastllm::Data &q,
                                                                 const float *windowKV,
                                                                 const fastllm::Data &compressedKV,
                                                                 fastllm::Data &attnSink,
                                                                 int windowSize, int startPos,
                                                                 int compressedCount, float softmaxScale,
                                                                 fastllm::Data &outputFloat) {
    if (windowKV == nullptr || q.dims.size() != 4 || q.dims[1] != 1 ||
        windowSize <= 0 || compressedCount < 0 || windowSize + compressedCount > 512 ||
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

    outputFloat.dataType = fastllm::DataType::FLOAT32;
    outputFloat.Resize({bsz, 1, heads, dim});
    outputFloat.Allocate(false);

    size_t windowBytes = (size_t)bsz * windowSize * dim * sizeof(float);
    float *cudaWindow = (float *)FastllmCudaMalloc(windowBytes);
    FastllmCudaCopyFromHostToDevice(cudaWindow, (void *)windowKV, windowBytes);
    float *cudaOutput = (float *)FastllmCudaPrepareOutput(outputFloat);
    outputFloat.cudaData = cudaOutput;

    bool ok = false;
    if (q.dataType == fastllm::DataType::BFLOAT16) {
        ok = DeepSeekV4LaunchSparseDecodeByCompressed<__nv_bfloat16>(
            q.cudaData, compressedCount > 0 ? compressedKV.cudaData : nullptr, compressedKV.dataType,
            cudaWindow, (const float *)attnSink.cudaData, outputFloat,
            bsz, heads, dim, windowSize, startPos, compressedCount, softmaxScale);
    } else if (q.dataType == fastllm::DataType::FLOAT16) {
        ok = DeepSeekV4LaunchSparseDecodeByCompressed<half>(
            q.cudaData, compressedCount > 0 ? compressedKV.cudaData : nullptr, compressedKV.dataType,
            cudaWindow, (const float *)attnSink.cudaData, outputFloat,
            bsz, heads, dim, windowSize, startPos, compressedCount, softmaxScale);
    } else if (q.dataType == fastllm::DataType::FLOAT32) {
        ok = DeepSeekV4LaunchSparseDecodeByCompressed<float>(
            q.cudaData, compressedCount > 0 ? compressedKV.cudaData : nullptr, compressedKV.dataType,
            cudaWindow, (const float *)attnSink.cudaData, outputFloat,
            bsz, heads, dim, windowSize, startPos, compressedCount, softmaxScale);
    }

    FastllmCudaFree(cudaWindow);
    FastllmCudaFinishOutput(outputFloat, cudaOutput);
    outputFloat.cudaData = nullptr;
    outputFloat.dataDevice = fastllm::DataDevice::CPU;
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
