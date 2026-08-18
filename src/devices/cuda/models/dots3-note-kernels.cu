#include "devices/cuda/fastllm-cuda.cuh"
#include "fastllm.h"

#include <cfloat>
#include <climits>
#include <cstdint>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cub/device/device_segmented_radix_sort.cuh>

namespace {

constexpr int kIndexerHeads = 64;
constexpr int kIndexerDim = 128;
constexpr int kIndexerTopK = 2048;
constexpr int kAttentionHeads = 128;
constexpr int kAttentionQkDim = 192;
constexpr int kAttentionVDim = 128;

bool PrepareOutput(fastllm::Data &output, fastllm::DataType type,
                   const std::vector<int> &dims) {
    output.dataType = type;
    output.Resize(dims);
    output.ToDevice(fastllm::DataDevice::CUDA,
                    {FastllmCudaGetDevice()}, false);
    output.Allocate(false);
    return output.cudaData != nullptr;
}

template <typename T>
__device__ __forceinline__ float ToFloat(T value) {
    return (float)value;
}

template <>
__device__ __forceinline__ float ToFloat(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

__device__ __forceinline__ float Fp8ToFloat(uint8_t raw) {
    __nv_fp8_e4m3 value;
    value.__x = raw;
    return static_cast<float>(value);
}

template <typename QT, typename WT>
__global__ void QuantizeQKernel(const QT *q, const WT *weights,
                                uint8_t *qFp8, float *foldedWeights,
                                int rows) {
    __shared__ float maximum[kIndexerDim];
    int row = blockIdx.x;
    int d = threadIdx.x;
    if (row >= rows || d >= kIndexerDim) {
        return;
    }
    float value = ToFloat(q[(uint64_t)row * kIndexerDim + d]);
    maximum[d] = fabsf(value);
    __syncthreads();
    for (int stride = kIndexerDim / 2; stride > 0; stride >>= 1) {
        if (d < stride) {
            maximum[d] = fmaxf(maximum[d], maximum[d + stride]);
        }
        __syncthreads();
    }
    // Dots uses an ordinary FP32 scale (not UE8M0/power-of-two scaling).
    float scale = fmaxf(maximum[0], 1.0e-4f) / 448.0f;
    __nv_fp8_e4m3 quantized(
        fminf(448.0f, fmaxf(-448.0f, value / scale)));
    qFp8[(uint64_t)row * kIndexerDim + d] = quantized.__x;
    if (d == 0) {
        int token = row / kIndexerHeads;
        int head = row % kIndexerHeads;
        constexpr float scoreScale =
            0.08838834764831845f * 0.125f; // rsqrt(128) * rsqrt(64)
        foldedWeights[(uint64_t)token * kIndexerHeads + head] =
            ToFloat(weights[(uint64_t)token * kIndexerHeads + head]) *
            scale * scoreScale;
    }
}

template <typename KT>
__global__ void QuantizeKKernel(const KT *k, uint8_t *kFp8,
                                float *kScales, int rows) {
    __shared__ float maximum[kIndexerDim];
    int row = blockIdx.x;
    int d = threadIdx.x;
    if (row >= rows || d >= kIndexerDim) {
        return;
    }
    float value = ToFloat(k[(uint64_t)row * kIndexerDim + d]);
    maximum[d] = fabsf(value);
    __syncthreads();
    for (int stride = kIndexerDim / 2; stride > 0; stride >>= 1) {
        if (d < stride) {
            maximum[d] = fmaxf(maximum[d], maximum[d + stride]);
        }
        __syncthreads();
    }
    float scale = fmaxf(maximum[0], 1.0e-4f) / 448.0f;
    __nv_fp8_e4m3 quantized(
        fminf(448.0f, fmaxf(-448.0f, value / scale)));
    kFp8[(uint64_t)row * kIndexerDim + d] = quantized.__x;
    if (d == 0) {
        kScales[row] = scale;
    }
}

__global__ void PackIndexerKeyKernel(const float *rope, const float *nope,
                                     __nv_bfloat16 *output, int rows) {
    int row = blockIdx.x;
    int d = threadIdx.x;
    if (row >= rows || d >= kIndexerDim) {
        return;
    }
    const float *source = d < 64 ? rope : nope;
    int sourceDim = d < 64 ? d : d - 64;
    output[(uint64_t)row * kIndexerDim + d] =
        __float2bfloat16_rn(source[(uint64_t)row * 64 + sourceDim]);
}

__global__ void IndexerLogitsKernel(
        const uint8_t *q, const uint8_t *k, const float *kScales,
        const float *weights, float *logits, int32_t *keyIds,
        int seqlen, int totalLen, int startPos) {
    uint64_t pair = (uint64_t)blockIdx.x;
    int token = pair / totalLen;
    int key = pair - (uint64_t)token * totalLen;
    if (token >= seqlen) {
        return;
    }
    keyIds[pair] = key;
    int rowEnd = min(totalLen, startPos + token + 1);
    if (key >= rowEnd) {
        logits[pair] = -FLT_MAX;
        return;
    }
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    float contribution = 0.0f;
    for (int head = warp; head < kIndexerHeads; head += 8) {
        const uint8_t *qRow = q +
            ((uint64_t)token * kIndexerHeads + head) * kIndexerDim;
        const uint8_t *kRow = k + (uint64_t)key * kIndexerDim;
        float dot = 0.0f;
#pragma unroll
        for (int d = lane; d < kIndexerDim; d += 32) {
            dot += Fp8ToFloat(qRow[d]) * Fp8ToFloat(kRow[d]);
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
        }
        if (lane == 0) {
            contribution += fmaxf(dot, 0.0f) *
                weights[(uint64_t)token * kIndexerHeads + head];
        }
    }
    __shared__ float warpSums[8];
    if (lane == 0) {
        warpSums[warp] = contribution;
    }
    __syncthreads();
    if (warp == 0) {
        float sum = lane < 8 ? warpSums[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }
        if (lane == 0) {
            logits[pair] = sum * kScales[key];
        }
    }
}

__global__ void SegmentOffsetsKernel(int32_t *offsets, int seqlen,
                                     int totalLen) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row <= seqlen) {
        offsets[row] = row * totalLen;
    }
}

__global__ void CopyTopKKernel(const int32_t *sortedIds,
                               int32_t *topkIds, int seqlen,
                               int totalLen) {
    int row = blockIdx.x;
    for (int i = threadIdx.x; i < kIndexerTopK; i += blockDim.x) {
        topkIds[(uint64_t)row * kIndexerTopK + i] =
            sortedIds[(uint64_t)row * totalLen + i];
    }
}

__global__ void SparseAttentionKernel(
        const __nv_bfloat16 *q, uint64_t qHeadStride,
        uint64_t qTokenStride, const __nv_bfloat16 *k,
        uint64_t kHeadStride, uint64_t kTokenStride,
        const __nv_bfloat16 *v, uint64_t vHeadStride,
        uint64_t vTokenStride, const int32_t *indices,
        __nv_bfloat16 *output, int seqlen, int totalLen,
        int startPos, float scale) {
    constexpr int warps = 8;
    int row = blockIdx.x;
    int token = row / kAttentionHeads;
    int head = row % kAttentionHeads;
    if (token >= seqlen) {
        return;
    }
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int length = min(kIndexerTopK, min(totalLen, startPos + token + 1));
    const __nv_bfloat16 *qRow =
        q + (uint64_t)head * qHeadStride +
        (uint64_t)token * qTokenStride;

    float localMax = -FLT_MAX;
    for (int selected = warp; selected < length; selected += warps) {
        int keyIndex =
            indices[(uint64_t)token * kIndexerTopK + selected];
        const __nv_bfloat16 *kRow =
            k + (uint64_t)head * kHeadStride +
            (uint64_t)keyIndex * kTokenStride;
        float dot = 0.0f;
#pragma unroll
        for (int d = lane; d < kAttentionQkDim; d += 32) {
            dot += __bfloat162float(qRow[d]) * __bfloat162float(kRow[d]);
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
        }
        if (lane == 0) {
            // The reference promotes the BF16 einsum result before applying
            // the FP32 attention scale and softmax.
            float score =
                __bfloat162float(__float2bfloat16_rn(dot)) * scale;
            localMax = fmaxf(localMax, score);
        }
    }
    __shared__ float warpMax[warps];
    __shared__ float warpSum[warps];
    __shared__ float unnormalized[kIndexerTopK];
    __shared__ float warpOutput[warps][kAttentionVDim];
    __shared__ float rowMax;
    __shared__ float rowSum;
    if (lane == 0) {
        warpMax[warp] = localMax;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float maximum = warpMax[0];
#pragma unroll
        for (int i = 1; i < warps; ++i) {
            maximum = fmaxf(maximum, warpMax[i]);
        }
        rowMax = maximum;
    }
    __syncthreads();

    float localSum = 0.0f;
    for (int selected = warp; selected < length; selected += warps) {
        int keyIndex =
            indices[(uint64_t)token * kIndexerTopK + selected];
        const __nv_bfloat16 *kRow =
            k + (uint64_t)head * kHeadStride +
            (uint64_t)keyIndex * kTokenStride;
        float dot = 0.0f;
#pragma unroll
        for (int d = lane; d < kAttentionQkDim; d += 32) {
            dot += __bfloat162float(qRow[d]) * __bfloat162float(kRow[d]);
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
        }
        float probability = 0.0f;
        if (lane == 0) {
            float score =
                __bfloat162float(__float2bfloat16_rn(dot)) * scale;
            probability = expf(score - rowMax);
            unnormalized[selected] = probability;
            localSum += probability;
        }
    }
    if (lane == 0) {
        warpSum[warp] = localSum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < warps; ++i) {
            sum += warpSum[i];
        }
        rowSum = sum;
    }
    __syncthreads();

    // Transformers converts the FP32 softmax probabilities back to the query
    // dtype before multiplying V.  Preserve that BF16 rounding boundary.
    float localOutput[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int selected = warp; selected < length; selected += warps) {
        int keyIndex =
            indices[(uint64_t)token * kIndexerTopK + selected];
        float probability = 0.0f;
        if (lane == 0) {
            probability = __bfloat162float(__float2bfloat16_rn(
                unnormalized[selected] / rowSum));
        }
        probability = __shfl_sync(0xffffffffu, probability, 0);
        const __nv_bfloat16 *vRow =
            v + (uint64_t)head * vHeadStride +
            (uint64_t)keyIndex * vTokenStride;
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            int d = lane + part * 32;
            localOutput[part] +=
                probability * __bfloat162float(vRow[d]);
        }
    }
#pragma unroll
    for (int part = 0; part < 4; ++part) {
        warpOutput[warp][lane + part * 32] = localOutput[part];
    }
    __syncthreads();
    if (threadIdx.x < kAttentionVDim) {
        float numerator = 0.0f;
#pragma unroll
        for (int i = 0; i < warps; ++i) {
            numerator += warpOutput[i][threadIdx.x];
        }
        output[((uint64_t)head * seqlen + token) *
                   kAttentionVDim + threadIdx.x] =
            __float2bfloat16_rn(numerator);
    }
}

} // namespace

bool FastllmCudaDots3NotePackIndexerKey(
        const fastllm::Data &rope, const fastllm::Data &nope,
        fastllm::Data &output) {
    if (rope.dataDevice != fastllm::DataDevice::CUDA ||
        rope.dataType != fastllm::DataType::FLOAT32 ||
        rope.cudaData == nullptr || rope.dims.size() != 4 ||
        rope.dims[0] != 1 || rope.dims[2] != 1 || rope.dims[3] != 64 ||
        nope.dataDevice != fastllm::DataDevice::CUDA ||
        nope.dataType != fastllm::DataType::FLOAT32 ||
        nope.cudaData == nullptr || nope.dims != rope.dims) {
        return false;
    }
    int device = GetPointerDeviceId(rope.cudaData);
    if (device < 0 || GetPointerDeviceId(nope.cudaData) != device) {
        return false;
    }
    FastllmCudaSetDevice(device);
    int rows = rope.dims[1];
    if (rows <= 0 ||
        !PrepareOutput(output, fastllm::DataType::BFLOAT16,
                       {1, rows, kIndexerDim})) {
        return false;
    }
    PackIndexerKeyKernel<<<rows, kIndexerDim, 0, cudaStreamPerThread>>>(
        (const float *)rope.cudaData, (const float *)nope.cudaData,
        (__nv_bfloat16 *)output.cudaData, rows);
    return cudaGetLastError() == cudaSuccess;
}

bool FastllmCudaDots3NoteQuantizeIndexer(
        const fastllm::Data &q, const fastllm::Data &k,
        const fastllm::Data &weights, fastllm::Data &qFp8,
        fastllm::Data &foldedWeights, fastllm::Data &kFp8,
        fastllm::Data &kScales) {
    if (q.dataDevice != fastllm::DataDevice::CUDA || q.cudaData == nullptr ||
        (q.dataType != fastllm::DataType::FLOAT32 &&
         q.dataType != fastllm::DataType::BFLOAT16) ||
        q.dims.size() != 4 || q.dims[0] != 1 ||
        q.dims[2] != kIndexerHeads || q.dims[3] != kIndexerDim ||
        k.dataDevice != fastllm::DataDevice::CUDA || k.cudaData == nullptr ||
        (k.dataType != fastllm::DataType::FLOAT32 &&
         k.dataType != fastllm::DataType::BFLOAT16) ||
        k.dims.size() != 3 || k.dims[0] != 1 ||
        k.dims[2] != kIndexerDim ||
        weights.dataDevice != fastllm::DataDevice::CUDA ||
        weights.cudaData == nullptr || weights.dims.size() != 3 ||
        weights.dims[0] != 1 || weights.dims[1] != q.dims[1] ||
        weights.dims[2] != kIndexerHeads ||
        (weights.dataType != fastllm::DataType::FLOAT32 &&
         weights.dataType != fastllm::DataType::BFLOAT16)) {
        return false;
    }
    int device = GetPointerDeviceId(q.cudaData);
    if (device < 0 || GetPointerDeviceId(k.cudaData) != device ||
        GetPointerDeviceId(weights.cudaData) != device) {
        return false;
    }
    FastllmCudaSetDevice(device);
    int qTokens = q.dims[1];
    int kTokens = k.dims[1];
    if (!PrepareOutput(qFp8, fastllm::DataType::INT8,
                       {1, qTokens, kIndexerHeads, kIndexerDim}) ||
        !PrepareOutput(foldedWeights, fastllm::DataType::FLOAT32,
                       {1, qTokens, kIndexerHeads}) ||
        !PrepareOutput(kFp8, fastllm::DataType::INT8,
                       {1, kTokens, kIndexerDim}) ||
        !PrepareOutput(kScales, fastllm::DataType::FLOAT32,
                       {1, kTokens, 1})) {
        return false;
    }
    int qRows = qTokens * kIndexerHeads;
    if (q.dataType == fastllm::DataType::FLOAT32 &&
        weights.dataType == fastllm::DataType::BFLOAT16) {
        QuantizeQKernel<<<qRows, kIndexerDim, 0, cudaStreamPerThread>>>(
            (const float *)q.cudaData,
            (const __nv_bfloat16 *)weights.cudaData,
            (uint8_t *)qFp8.cudaData, (float *)foldedWeights.cudaData,
            qRows);
    } else if (q.dataType == fastllm::DataType::FLOAT32 &&
               weights.dataType == fastllm::DataType::FLOAT32) {
        QuantizeQKernel<<<qRows, kIndexerDim, 0, cudaStreamPerThread>>>(
            (const float *)q.cudaData, (const float *)weights.cudaData,
            (uint8_t *)qFp8.cudaData, (float *)foldedWeights.cudaData,
            qRows);
    } else if (q.dataType == fastllm::DataType::BFLOAT16 &&
               weights.dataType == fastllm::DataType::BFLOAT16) {
        QuantizeQKernel<<<qRows, kIndexerDim, 0, cudaStreamPerThread>>>(
            (const __nv_bfloat16 *)q.cudaData,
            (const __nv_bfloat16 *)weights.cudaData,
            (uint8_t *)qFp8.cudaData, (float *)foldedWeights.cudaData,
            qRows);
    } else {
        QuantizeQKernel<<<qRows, kIndexerDim, 0, cudaStreamPerThread>>>(
            (const __nv_bfloat16 *)q.cudaData,
            (const float *)weights.cudaData,
            (uint8_t *)qFp8.cudaData, (float *)foldedWeights.cudaData,
            qRows);
    }
    if (k.dataType == fastllm::DataType::BFLOAT16) {
        QuantizeKKernel<<<kTokens, kIndexerDim, 0, cudaStreamPerThread>>>(
            (const __nv_bfloat16 *)k.cudaData,
            (uint8_t *)kFp8.cudaData, (float *)kScales.cudaData,
            kTokens);
    } else {
        QuantizeKKernel<<<kTokens, kIndexerDim, 0, cudaStreamPerThread>>>(
            (const float *)k.cudaData, (uint8_t *)kFp8.cudaData,
            (float *)kScales.cudaData, kTokens);
    }
    return cudaGetLastError() == cudaSuccess;
}

bool FastllmCudaDots3NoteIndexerTopK(
        const fastllm::Data &qFp8,
        const fastllm::Data &foldedWeights,
        const fastllm::Data &kFp8, const fastllm::Data &kScales,
        int startPos, int topK, fastllm::Data &indices) {
    if (topK != kIndexerTopK || startPos < 0 ||
        qFp8.dataDevice != fastllm::DataDevice::CUDA ||
        qFp8.dataType != fastllm::DataType::INT8 ||
        qFp8.cudaData == nullptr || qFp8.dims.size() != 4 ||
        qFp8.dims[0] != 1 || qFp8.dims[2] != kIndexerHeads ||
        qFp8.dims[3] != kIndexerDim ||
        foldedWeights.dataDevice != fastllm::DataDevice::CUDA ||
        foldedWeights.dataType != fastllm::DataType::FLOAT32 ||
        foldedWeights.Count(0) !=
            (uint64_t)qFp8.dims[1] * kIndexerHeads ||
        kFp8.dataDevice != fastllm::DataDevice::CUDA ||
        kFp8.dataType != fastllm::DataType::INT8 ||
        kFp8.cudaData == nullptr || kFp8.dims.size() != 3 ||
        kFp8.dims[0] != 1 || kFp8.dims[2] != kIndexerDim ||
        kScales.dataDevice != fastllm::DataDevice::CUDA ||
        kScales.dataType != fastllm::DataType::FLOAT32 ||
        kScales.dims.size() != 3 || kScales.dims[0] != 1 ||
        kScales.dims[1] != kFp8.dims[1] || kScales.dims[2] != 1) {
        return false;
    }
    int device = GetPointerDeviceId(qFp8.cudaData);
    if (device < 0 || GetPointerDeviceId(foldedWeights.cudaData) != device ||
        GetPointerDeviceId(kFp8.cudaData) != device ||
        GetPointerDeviceId(kScales.cudaData) != device) {
        return false;
    }
    FastllmCudaSetDevice(device);
    int seqlen = qFp8.dims[1];
    int totalLen = kFp8.dims[1];
    uint64_t pairs64 = (uint64_t)seqlen * totalLen;
    if (seqlen <= 0 || totalLen <= kIndexerTopK ||
        startPos + seqlen > totalLen || pairs64 > INT_MAX ||
        !PrepareOutput(indices, fastllm::DataType::INT32,
                       {seqlen, kIndexerTopK})) {
        return false;
    }
    int pairs = (int)pairs64;
    float *logits =
        (float *)FastllmCudaMalloc((size_t)pairs * sizeof(float));
    float *sortedLogits =
        (float *)FastllmCudaMalloc((size_t)pairs * sizeof(float));
    int32_t *keyIds =
        (int32_t *)FastllmCudaMalloc((size_t)pairs * sizeof(int32_t));
    int32_t *sortedIds =
        (int32_t *)FastllmCudaMalloc((size_t)pairs * sizeof(int32_t));
    int32_t *offsets = (int32_t *)FastllmCudaMalloc(
        (size_t)(seqlen + 1) * sizeof(int32_t));
    if (logits == nullptr || sortedLogits == nullptr || keyIds == nullptr ||
        sortedIds == nullptr || offsets == nullptr) {
        if (logits) FastllmCudaFree(logits);
        if (sortedLogits) FastllmCudaFree(sortedLogits);
        if (keyIds) FastllmCudaFree(keyIds);
        if (sortedIds) FastllmCudaFree(sortedIds);
        if (offsets) FastllmCudaFree(offsets);
        return false;
    }
    IndexerLogitsKernel<<<pairs, 256, 0, cudaStreamPerThread>>>(
        (const uint8_t *)qFp8.cudaData,
        (const uint8_t *)kFp8.cudaData,
        (const float *)kScales.cudaData,
        (const float *)foldedWeights.cudaData,
        logits, keyIds, seqlen, totalLen, startPos);
    SegmentOffsetsKernel<<<(seqlen + 256) / 256, 256, 0,
                           cudaStreamPerThread>>>(
        offsets, seqlen, totalLen);

    size_t sortBytes = 0;
    cudaError_t sortState =
        cub::DeviceSegmentedRadixSort::SortPairsDescending(
            nullptr, sortBytes, logits, sortedLogits, keyIds, sortedIds,
            pairs, seqlen, offsets, offsets + 1, 0, 32,
            cudaStreamPerThread);
    void *sortScratch = sortState == cudaSuccess
                            ? FastllmCudaMalloc(sortBytes)
                            : nullptr;
    if (sortScratch != nullptr) {
        sortState = cub::DeviceSegmentedRadixSort::SortPairsDescending(
            sortScratch, sortBytes, logits, sortedLogits, keyIds, sortedIds,
            pairs, seqlen, offsets, offsets + 1, 0, 32,
            cudaStreamPerThread);
    }
    if (sortState == cudaSuccess && sortScratch != nullptr) {
        CopyTopKKernel<<<seqlen, 256, 0, cudaStreamPerThread>>>(
            sortedIds, (int32_t *)indices.cudaData, seqlen, totalLen);
    }
    cudaError_t launchState = cudaGetLastError();
    cudaError_t syncState = cudaStreamSynchronize(cudaStreamPerThread);
    bool ok = sortState == cudaSuccess && sortScratch != nullptr &&
              launchState == cudaSuccess && syncState == cudaSuccess;
    if (sortScratch) FastllmCudaFree(sortScratch);
    FastllmCudaFree(offsets);
    FastllmCudaFree(sortedIds);
    FastllmCudaFree(keyIds);
    FastllmCudaFree(sortedLogits);
    FastllmCudaFree(logits);
    return ok;
}

bool FastllmCudaDots3NoteSparseAttention(
        const fastllm::Data &q, const fastllm::Data &k,
        const fastllm::Data &v, const fastllm::Data &indices,
        int startPos, float scale, fastllm::Data &output) {
    if (startPos < 0 || q.dataDevice != fastllm::DataDevice::CUDA ||
        q.dataType != fastllm::DataType::BFLOAT16 || q.cudaData == nullptr ||
        q.dims.size() != 3 || q.dims[0] != kAttentionHeads ||
        q.dims[2] != kAttentionQkDim ||
        k.dataDevice != fastllm::DataDevice::CUDA ||
        k.dataType != fastllm::DataType::BFLOAT16 || k.cudaData == nullptr ||
        k.dims.size() != 3 || k.dims[0] != kAttentionHeads ||
        k.dims[2] != kAttentionQkDim ||
        v.dataDevice != fastllm::DataDevice::CUDA ||
        v.dataType != fastllm::DataType::BFLOAT16 || v.cudaData == nullptr ||
        v.dims.size() != 3 || v.dims[0] != kAttentionHeads ||
        v.dims[1] != k.dims[1] || v.dims[2] != kAttentionVDim ||
        indices.dataDevice != fastllm::DataDevice::CUDA ||
        indices.dataType != fastllm::DataType::INT32 ||
        indices.cudaData == nullptr || indices.dims.size() != 2 ||
        indices.dims[0] != q.dims[1] ||
        indices.dims[1] != kIndexerTopK ||
        startPos + q.dims[1] > k.dims[1]) {
        return false;
    }
    int device = GetPointerDeviceId(q.cudaData);
    if (device < 0 || GetPointerDeviceId(k.cudaData) != device ||
        GetPointerDeviceId(v.cudaData) != device ||
        GetPointerDeviceId(indices.cudaData) != device) {
        return false;
    }
    FastllmCudaSetDevice(device);
    int seqlen = q.dims[1];
    if (!PrepareOutput(output, fastllm::DataType::BFLOAT16,
                       {kAttentionHeads, seqlen, kAttentionVDim})) {
        return false;
    }
    SparseAttentionKernel<<<seqlen * kAttentionHeads, 256, 0,
                            cudaStreamPerThread>>>(
        (const __nv_bfloat16 *)q.cudaData, q.strides[0], q.strides[1],
        (const __nv_bfloat16 *)k.cudaData, k.strides[0], k.strides[1],
        (const __nv_bfloat16 *)v.cudaData, v.strides[0], v.strides[1],
        (const int32_t *)indices.cudaData,
        (__nv_bfloat16 *)output.cudaData, seqlen, k.dims[1],
        startPos, scale);
    return cudaGetLastError() == cudaSuccess;
}
