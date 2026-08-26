#include "devices/cuda/fastllm-cuda.cuh"
#include "fastllm.h"

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>

namespace {

constexpr int kIndexerHeads = 64;
constexpr int kIndexerDim = 128;
constexpr int kIndexerTopK = 2048;
constexpr int kIndexerCandidateK = 2304;
constexpr int kIndexerDefaultQueryChunk = 256;
constexpr int kIndexerKeysPerBlock = 8;
constexpr int kIndexerCandidateKeysPerBlock = 16;
constexpr int kIndexerRadixThreads = 256;
constexpr int kIndexerRadixWarps = kIndexerRadixThreads / 32;
static_assert(kIndexerRadixThreads % 32 == 0,
              "Dots3-Note radix block must contain complete warps");
constexpr size_t kIndexerHeadScoreLimit =
    1ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr int kAttentionHeads = 128;
constexpr int kAttentionQkDim = 192;
constexpr int kAttentionVDim = 128;
constexpr size_t kSparsePrefillScratchLimit =
    128ULL * 1024ULL * 1024ULL;
constexpr size_t kSlidingPrefillScratchLimit =
    64ULL * 1024ULL * 1024ULL;

bool EnvValueEnabled(const char *value) {
    return value != nullptr && value[0] != '\0' &&
           strcmp(value, "0") != 0 &&
           strcmp(value, "false") != 0 &&
           strcmp(value, "FALSE") != 0 &&
           strcmp(value, "off") != 0 &&
           strcmp(value, "OFF") != 0;
}

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

template <>
__device__ __forceinline__ float ToFloat(__half value) {
    return __half2float(value);
}

__device__ __forceinline__ float2 Fp8x2ToFloat(const uint8_t *raw) {
    __nv_fp8x2_e4m3 value;
    value.__x = *reinterpret_cast<const uint16_t *>(raw);
    return static_cast<float2>(value);
}

template <typename QT, typename WT>
__global__ void QuantizeSplitQKernel(
        const QT *rope, const QT *nope, const WT *weights,
        uint8_t *qFp8, float *foldedWeights, int rows) {
    __shared__ float maximum[kIndexerDim];
    int row = blockIdx.x;
    int d = threadIdx.x;
    if (row >= rows || d >= kIndexerDim) {
        return;
    }
    const QT *source = d < 64 ? rope : nope;
    int sourceDim = d < 64 ? d : d - 64;
    float value = ToFloat(source[(uint64_t)row * 64 + sourceDim]);
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

__global__ void PackAttentionKeyKernel(
        const __nv_bfloat16 *kv, const __nv_bfloat16 *rope,
        __nv_bfloat16 *output, uint64_t elements, int tokens,
        int nopeDim, int ropeDim,
        uint64_t kvHeadStride, uint64_t kvTokenStride,
        uint64_t ropeTokenStride, uint64_t outputHeadStride,
        uint64_t outputTokenStride) {
    uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    int qkDim = nopeDim + ropeDim;
    int d = index % qkDim;
    uint64_t row = index / qkDim;
    int token = row % tokens;
    int head = row / tokens;
    uint64_t outputOffset = (uint64_t)head * outputHeadStride +
                            (uint64_t)token * outputTokenStride + d;
    if (d < nopeDim) {
        output[outputOffset] = kv[(uint64_t)head * kvHeadStride +
                                  (uint64_t)token * kvTokenStride + d];
    } else {
        output[outputOffset] =
            rope[(uint64_t)token * ropeTokenStride + d - nopeDim];
    }
}

__global__ void IndexerLogitsTiledKernel(
        const uint8_t *q, const uint8_t *k, const float *kScales,
        const float *weights, float *logits,
        int seqlen, int totalLen, int startPos) {
    constexpr int warps = kIndexerKeysPerBlock;
    __shared__ uint8_t qShared[kIndexerHeads * kIndexerDim];
    __shared__ float weightShared[kIndexerHeads];
    __shared__ float groupSums[warps][8];

    int keyTiles = (totalLen + kIndexerKeysPerBlock - 1) /
                   kIndexerKeysPerBlock;
    int token = blockIdx.x / keyTiles;
    int keyTile = blockIdx.x - token * keyTiles;
    if (token >= seqlen) {
        return;
    }
    const uint8_t *qRow =
        q + (uint64_t)token * kIndexerHeads * kIndexerDim;
    for (int i = threadIdx.x; i < kIndexerHeads * kIndexerDim;
         i += blockDim.x) {
        qShared[i] = qRow[i];
    }
    for (int i = threadIdx.x; i < kIndexerHeads;
         i += blockDim.x) {
        weightShared[i] =
            weights[(uint64_t)token * kIndexerHeads + i];
    }
    __syncthreads();

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int key = keyTile * kIndexerKeysPerBlock + warp;
    if (key >= totalLen) {
        return;
    }
    int rowEnd = min(totalLen, startPos + token + 1);
    if (key >= rowEnd) {
        if (lane == 0) {
            logits[(uint64_t)token * totalLen + key] = -FLT_MAX;
        }
        return;
    }

    const uint8_t *kRow = k + (uint64_t)key * kIndexerDim;
    float2 keyValues[2];
#pragma unroll
    for (int part = 0; part < 2; ++part) {
        int d = lane * 2 + part * 64;
        keyValues[part] = Fp8x2ToFloat(kRow + d);
    }
    float groupedContribution[8] = {
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f};
    for (int head = 0; head < kIndexerHeads; ++head) {
        const uint8_t *qHead =
            qShared + head * kIndexerDim;
        float dot = 0.0f;
#pragma unroll
        for (int part = 0; part < 2; ++part) {
            int d = lane * 2 + part * 64;
            float2 queryValues = Fp8x2ToFloat(qHead + d);
            dot += queryValues.x * keyValues[part].x;
            dot += queryValues.y * keyValues[part].y;
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
        }
        if (lane == 0) {
            groupedContribution[head & 7] +=
                fmaxf(dot, 0.0f) * weightShared[head];
        }
    }
    if (lane == 0) {
        for (int group = 0; group < 8; ++group) {
            groupSums[warp][group] = groupedContribution[group];
        }
    }
    __syncwarp();
    float sum = lane < 8 ? groupSums[warp][lane] : 0.0f;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
        logits[(uint64_t)token * totalLen + key] =
            sum * kScales[key];
    }
}

template <typename ScoreT>
__global__ void IndexerReduceHeadScoresKernel(
        const ScoreT *headScores, const float *kScales,
        const float *weights, float *logits,
        int queryOffset, int current, int keyCount,
        int totalLen, int startPos) {
    int key = blockIdx.x * blockDim.x + threadIdx.x;
    int tokenInChunk = blockIdx.y;
    if (tokenInChunk >= current || key >= keyCount) {
        return;
    }
    int token = queryOffset + tokenInChunk;
    if (key >= startPos + token + 1) {
        logits[(uint64_t)tokenInChunk * totalLen + key] = -FLT_MAX;
        return;
    }

    uint64_t headStride = (uint64_t)current * keyCount;
    uint64_t scoreOffset =
        (uint64_t)tokenInChunk * keyCount + key;
    const float *tokenWeights =
        weights + (uint64_t)tokenInChunk * kIndexerHeads;
    float sum = 0.0f;
#pragma unroll
    for (int head = 0; head < kIndexerHeads; ++head) {
        float score = ToFloat(
            headScores[(uint64_t)head * headStride + scoreOffset]);
        sum += fmaxf(score, 0.0f) * tokenWeights[head];
    }
    logits[(uint64_t)tokenInChunk * totalLen + key] =
        sum * kScales[key];
}

template <typename IndexT>
__global__ void IndexerCandidateExactScoresKernel(
        const uint8_t *q, const uint8_t *k, const float *kScales,
        const float *weights, const IndexT *candidateIds,
        float *candidateScores, int seqlen, int totalLen,
        int startPos) {
    constexpr int warps = kIndexerCandidateKeysPerBlock;
    __shared__ uint8_t qShared[kIndexerHeads * kIndexerDim];
    __shared__ float weightShared[kIndexerHeads];
    __shared__ float groupSums[warps][8];

    constexpr int candidateTiles =
        (kIndexerCandidateK + kIndexerCandidateKeysPerBlock - 1) /
        kIndexerCandidateKeysPerBlock;
    int token = blockIdx.x / candidateTiles;
    int candidateTile = blockIdx.x - token * candidateTiles;
    if (token >= seqlen) return;
    const uint8_t *qRow =
        q + (uint64_t)token * kIndexerHeads * kIndexerDim;
    for (int i = threadIdx.x; i < kIndexerHeads * kIndexerDim;
         i += blockDim.x) {
        qShared[i] = qRow[i];
    }
    for (int i = threadIdx.x; i < kIndexerHeads;
         i += blockDim.x) {
        weightShared[i] =
            weights[(uint64_t)token * kIndexerHeads + i];
    }
    __syncthreads();

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int candidate = candidateTile * kIndexerCandidateKeysPerBlock + warp;
    if (candidate >= kIndexerCandidateK) return;
    int key = (int)candidateIds[
        (uint64_t)token * kIndexerCandidateK + candidate];
    int rowEnd = min(totalLen, startPos + token + 1);
    if (key < 0 || key >= rowEnd) {
        if (lane == 0) {
            candidateScores[
                (uint64_t)token * kIndexerCandidateK + candidate] =
                -FLT_MAX;
        }
        return;
    }

    const uint8_t *kRow = k + (uint64_t)key * kIndexerDim;
    float2 keyValues[2];
#pragma unroll
    for (int part = 0; part < 2; ++part) {
        int d = lane * 2 + part * 64;
        keyValues[part] = Fp8x2ToFloat(kRow + d);
    }
    float groupedContribution[8] = {
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f};
    for (int head = 0; head < kIndexerHeads; ++head) {
        const uint8_t *qHead = qShared + head * kIndexerDim;
        float dot = 0.0f;
#pragma unroll
        for (int part = 0; part < 2; ++part) {
            int d = lane * 2 + part * 64;
            float2 queryValues = Fp8x2ToFloat(qHead + d);
            dot += queryValues.x * keyValues[part].x;
            dot += queryValues.y * keyValues[part].y;
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffffu, dot, offset);
        }
        if (lane == 0) {
            groupedContribution[head & 7] +=
                fmaxf(dot, 0.0f) * weightShared[head];
        }
    }
    if (lane == 0) {
        for (int group = 0; group < 8; ++group) {
            groupSums[warp][group] = groupedContribution[group];
        }
    }
    __syncwarp();
    float sum = lane < 8 ? groupSums[warp][lane] : 0.0f;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
        candidateScores[
            (uint64_t)token * kIndexerCandidateK + candidate] =
            sum * kScales[key];
    }
}

__device__ __forceinline__ uint16_t IndexerFp8ToHalfBits(uint8_t bits) {
    uint16_t sign = (uint16_t)(bits & 0x80u) << 8;
    uint8_t magnitude = bits & 0x7fu;
    if (magnitude == 0) {
        return sign;
    }
    if (magnitude < 8) {
        // Normalize E4M3 subnormals before they reach MMA. All nonzero E4M3
        // values are representable as normal FP16 values, avoiding the
        // architecture-dependent handling of FP16 subnormal MMA inputs.
        int shift = (magnitude & 4u) ? 0 : ((magnitude & 2u) ? 1 : 2);
        uint16_t normalized = (uint16_t)magnitude << shift;
        uint16_t exponent = (uint16_t)(8 - shift) << 10;
        uint16_t mantissa = (normalized & 3u) << 8;
        return sign | exponent | mantissa;
    }
    if (magnitude == 0x7fu) {
        return sign | 0x7e00u;
    }
    // Add the FP16/E4M3 exponent-bias difference and expand the three E4M3
    // mantissa bits. This preserves the original finite value exactly.
    return sign | (((uint16_t)magnitude << 7) + 0x2000u);
}

__global__ void IndexerFp8ToHalfKernel(
        const uint8_t *input, __half *output, uint64_t packedCount) {
    uint64_t packed = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (packed >= packedCount) {
        return;
    }
    uint32_t bytes = reinterpret_cast<const uint32_t *>(input)[packed];
    uint32_t halves01 =
        (uint32_t)IndexerFp8ToHalfBits((uint8_t)bytes) |
        ((uint32_t)IndexerFp8ToHalfBits((uint8_t)(bytes >> 8)) << 16);
    uint32_t halves23 =
        (uint32_t)IndexerFp8ToHalfBits((uint8_t)(bytes >> 16)) |
        ((uint32_t)IndexerFp8ToHalfBits((uint8_t)(bytes >> 24)) << 16);
    reinterpret_cast<uint32_t *>(output)[packed * 2] = halves01;
    reinterpret_cast<uint32_t *>(output)[packed * 2 + 1] = halves23;
}

cublasStatus_t IndexerBatchedGemm(
        const void *k, const void *q, void *headScores,
        int current, int keyCount, cudaDataType_t inputType,
        float alpha) {
    static thread_local std::vector<cublasLtHandle_t> handles;
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if ((int)handles.size() <= device) {
        handles.resize(device + 1, nullptr);
    }
    cublasLtHandle_t &handle = handles[device];
    if (handle == nullptr) {
        cublasStatus_t state = cublasLtCreate(&handle);
        if (state != CUBLAS_STATUS_SUCCESS) {
            handle = nullptr;
            return state;
        }
    }

    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t aLayout = nullptr;
    cublasLtMatrixLayout_t bLayout = nullptr;
    cublasLtMatrixLayout_t cLayout = nullptr;
    cublasStatus_t state = cublasLtMatmulDescCreate(
        &operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t transA = CUBLAS_OP_T;
    cublasOperation_t transB = CUBLAS_OP_N;
    if (state == CUBLAS_STATUS_SUCCESS) {
        state = cublasLtMatmulDescSetAttribute(
            operation, CUBLASLT_MATMUL_DESC_TRANSA,
            &transA, sizeof(transA));
    }
    if (state == CUBLAS_STATUS_SUCCESS) {
        state = cublasLtMatmulDescSetAttribute(
            operation, CUBLASLT_MATMUL_DESC_TRANSB,
            &transB, sizeof(transB));
    }
    if (state == CUBLAS_STATUS_SUCCESS) {
        state = cublasLtMatrixLayoutCreate(
            &aLayout, inputType,
            kIndexerDim, keyCount, kIndexerDim);
    }
    if (state == CUBLAS_STATUS_SUCCESS) {
        state = cublasLtMatrixLayoutCreate(
            &bLayout, inputType,
            kIndexerDim, current, kIndexerHeads * kIndexerDim);
    }
    if (state == CUBLAS_STATUS_SUCCESS) {
        state = cublasLtMatrixLayoutCreate(
            &cLayout, CUDA_R_16F,
            keyCount, current, keyCount);
    }
    int batchCount = kIndexerHeads;
    int64_t strideA = 0;
    int64_t strideB = kIndexerDim;
    int64_t strideC = (int64_t)current * keyCount;
    auto setBatch = [&](cublasLtMatrixLayout_t layout,
                        const int64_t &stride) {
        cublasStatus_t result = cublasLtMatrixLayoutSetAttribute(
            layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
            &batchCount, sizeof(batchCount));
        if (result == CUBLAS_STATUS_SUCCESS) {
            result = cublasLtMatrixLayoutSetAttribute(
                layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                &stride, sizeof(stride));
        }
        return result;
    };
    if (state == CUBLAS_STATUS_SUCCESS) state = setBatch(aLayout, strideA);
    if (state == CUBLAS_STATUS_SUCCESS) state = setBatch(bLayout, strideB);
    if (state == CUBLAS_STATUS_SUCCESS) state = setBatch(cLayout, strideC);

    if (state == CUBLAS_STATUS_SUCCESS) {
        const float beta = 0.0f;
        state = cublasLtMatmul(
            handle, operation, &alpha,
            k, aLayout, q, bLayout, &beta,
            headScores, cLayout, headScores, cLayout,
            nullptr, nullptr, 0, cudaStreamPerThread);
    }
    if (cLayout) cublasLtMatrixLayoutDestroy(cLayout);
    if (bLayout) cublasLtMatrixLayoutDestroy(bLayout);
    if (aLayout) cublasLtMatrixLayoutDestroy(aLayout);
    if (operation) cublasLtMatmulDescDestroy(operation);
    return state;
}

cublasStatus_t IndexerFp8BatchedGemm(
        const void *k, const void *q, void *headScores,
        int current, int keyCount) {
    // The largest possible E4M3 dot is
    // 448 * 448 * 128 / 512 = 50176, which fits FP16. This common positive
    // scale preserves candidate ranking while halving the score workspace.
    return IndexerBatchedGemm(
        k, q, headScores, current, keyCount,
        CUDA_R_8F_E4M3, 1.0f / 512.0f);
}

cublasStatus_t IndexerFp16BatchedGemm(
        const void *k, const void *q, void *headScores,
        int current, int keyCount) {
    // The conversion preserves each E4M3 value exactly, so use the same score
    // scale as the native FP8 Tensor Core path.
    return IndexerBatchedGemm(
        k, q, headScores, current, keyCount,
        CUDA_R_16F, 1.0f / 512.0f);
}

__device__ __forceinline__ uint32_t OrderedFloatBits(float value) {
    uint32_t bits = __float_as_uint(value);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Select an exact Top-K set with four radix-histogram passes instead of sorting
// every score. Output indices are ascending, which makes equal-score
// truncation deterministic and matches the reference's smaller-key tie break.
template <typename IndexT, int selectedK>
__global__ void IndexerRadixSelectKernel(
        const float *logits, IndexT *selectedIds,
        const IndexT *keyMap, int seqlen,
        int totalLen, int startPos,
        IndexT *radixScratch, int radixScratchStride,
        bool parallelEmit) {
    int token = blockIdx.x;
    if (token >= seqlen) return;
    int rowEnd = min(totalLen, startPos + token + 1);
    IndexT *output = selectedIds + (uint64_t)token * selectedK;
    if (rowEnd <= selectedK) {
        for (int i = threadIdx.x; i < selectedK;
             i += blockDim.x) {
            output[i] = keyMap == nullptr
                ? (IndexT)i
                : keyMap[(uint64_t)token * totalLen + i];
        }
        return;
    }

    __shared__ uint32_t histogram[256];
    __shared__ uint32_t prefix;
    __shared__ uint32_t remaining;
    __shared__ uint32_t compactCount;
    __shared__ uint32_t emitBase;
    __shared__ uint32_t emitTileBase;
    __shared__ uint32_t emitWarpCounts[kIndexerRadixWarps];
    __shared__ uint32_t emitWarpOffsets[kIndexerRadixWarps];
    if (threadIdx.x == 0) {
        prefix = 0;
        remaining = selectedK;
    }
    __syncthreads();

    const float *row = logits + (uint64_t)token * totalLen;
#pragma unroll
    for (int round = 0; round < 4; ++round) {
        if (threadIdx.x < 256) histogram[threadIdx.x] = 0;
        __syncthreads();
        int shift = 24 - round * 8;
        uint32_t mask = round == 0
            ? 0u : (~0u << (32 - round * 8));
        uint32_t currentPrefix = prefix;
        bool useCompactBucket = radixScratch != nullptr && round > 1;
        int scanCount = useCompactBucket ? (int)compactCount : rowEnd;
        IndexT *scratchRow = radixScratch == nullptr
            ? nullptr
            : radixScratch + (uint64_t)token * radixScratchStride;
        for (int i = threadIdx.x; i < scanCount;
             i += blockDim.x) {
            int key = useCompactBucket ? (int)scratchRow[i] : i;
            uint32_t ordered = OrderedFloatBits(row[key]);
            if ((ordered & mask) == currentPrefix) {
                atomicAdd(&histogram[(ordered >> shift) & 0xffu], 1u);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            uint32_t higher = 0;
            for (int bucket = 255; bucket >= 0; --bucket) {
                uint32_t count = histogram[bucket];
                if (higher + count >= remaining) {
                    prefix = currentPrefix |
                             ((uint32_t)bucket << shift);
                    remaining -= higher;
                    break;
                }
                higher += count;
            }
        }
        __syncthreads();
        if (round == 1 && radixScratch != nullptr) {
            if (threadIdx.x == 0) {
                compactCount = 0;
            }
            __syncthreads();
            uint32_t firstTwoBytePrefix = prefix;
            for (int key = threadIdx.x; key < rowEnd;
                 key += blockDim.x) {
                uint32_t ordered = OrderedFloatBits(row[key]);
                if ((ordered & 0xffff0000u) == firstTwoBytePrefix) {
                    uint32_t position = atomicAdd(&compactCount, 1u);
                    scratchRow[position] = (IndexT)key;
                }
            }
            __syncthreads();
        }
    }

    if (!parallelEmit) {
        if (threadIdx.x == 0) {
            int count = 0;
            uint32_t pivot = prefix;
            for (int key = 0; key < rowEnd && count < selectedK; ++key) {
                if (OrderedFloatBits(row[key]) > pivot) {
                    output[count++] = keyMap == nullptr
                        ? (IndexT)key
                        : keyMap[(uint64_t)token * totalLen + key];
                }
            }
            for (int key = 0; key < rowEnd && count < selectedK; ++key) {
                if (OrderedFloatBits(row[key]) == pivot) {
                    output[count++] = keyMap == nullptr
                        ? (IndexT)key
                        : keyMap[(uint64_t)token * totalLen + key];
                }
            }
        }
        return;
    }

    // Emit in ascending key-id order with a stable CTA-wide compaction. The
    // strict-greater pass comes first, followed by only as many pivot ties as
    // needed, exactly matching the established smaller-key tie break.
    if (threadIdx.x == 0) {
        emitBase = 0;
    }
    __syncthreads();
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    uint32_t lowerLaneMask = lane == 0
        ? 0u : (0xffffffffu >> (32 - lane));
    uint32_t pivot = prefix;
    for (int equalPass = 0; equalPass < 2; ++equalPass) {
        for (int tile = 0; tile < rowEnd; tile += blockDim.x) {
            if (emitBase >= selectedK) {
                break;
            }
            int key = tile + threadIdx.x;
            bool selected = false;
            if (key < rowEnd) {
                uint32_t ordered = OrderedFloatBits(row[key]);
                selected = equalPass == 0
                    ? ordered > pivot : ordered == pivot;
            }
            uint32_t selectedMask = __ballot_sync(
                0xffffffffu, selected);
            if (lane == 0) {
                emitWarpCounts[warp] = __popc(selectedMask);
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                uint32_t offset = 0;
#pragma unroll
                for (int emitWarp = 0;
                     emitWarp < kIndexerRadixWarps; ++emitWarp) {
                    emitWarpOffsets[emitWarp] = offset;
                    offset += emitWarpCounts[emitWarp];
                }
                emitTileBase = emitBase;
                emitBase += offset;
            }
            __syncthreads();
            uint32_t position = emitTileBase + emitWarpOffsets[warp] +
                __popc(selectedMask & lowerLaneMask);
            if (selected && position < selectedK) {
                output[position] = keyMap == nullptr
                    ? (IndexT)key
                    : keyMap[(uint64_t)token * totalLen + key];
            }
            __syncthreads();
        }
    }
}

// Keep the decode-shaped path byte-for-byte equivalent to the established
// implementation. The prefill specialization below changes CTA ordering and
// reduction layout only when there are enough query rows to reuse KV in L2.
template <typename IndexT>
__global__ void SparseAttentionKernel(
        const __nv_bfloat16 *q, uint64_t qHeadStride,
        uint64_t qTokenStride, const __nv_bfloat16 *k,
        uint64_t kHeadStride, uint64_t kTokenStride,
        const __nv_bfloat16 *v, uint64_t vHeadStride,
        uint64_t vTokenStride, const IndexT *indices,
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

    __shared__ float warpMax[warps];
    __shared__ float warpSum[warps];
    __shared__ float unnormalized[kIndexerTopK];
    __shared__ float warpOutput[warps][kAttentionVDim];
    __shared__ float rowMax;
    __shared__ float rowSum;
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
            float score =
                __bfloat162float(__float2bfloat16_rn(dot)) * scale;
            unnormalized[selected] = score;
            localMax = fmaxf(localMax, score);
        }
    }
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
        float score = 0.0f;
        if (lane == 0) {
            score = unnormalized[selected];
        }
        float probability = 0.0f;
        if (lane == 0) {
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

template <typename IndexT, bool CacheIndices, int Warps>
__global__ void SparseAttentionPrefillRowKernel(
        const __nv_bfloat16 *q, uint64_t qHeadStride,
        uint64_t qTokenStride, const __nv_bfloat16 *k,
        uint64_t kHeadStride, uint64_t kTokenStride,
        const __nv_bfloat16 *v, uint64_t vHeadStride,
        uint64_t vTokenStride, const IndexT *indices,
        __nv_bfloat16 *output, int seqlen, int totalLen,
        int startPos, float scale) {
    constexpr int warps = Warps;
    static_assert(warps == 4 || warps == 8 || warps == 16,
                  "unsupported sparse prefill warp count");
    int row = blockIdx.x;
    // Keep adjacent CTAs on the same attention head. Neighboring prefill
    // queries usually select many of the same KV rows, so this ordering keeps
    // their sparse gathers hot in L2 instead of cycling through all 128 heads
    // before returning to the same KV cache.
    int head = row / seqlen;
    int token = row - head * seqlen;
    if (token >= seqlen) {
        return;
    }
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int length = min(kIndexerTopK, min(totalLen, startPos + token + 1));
    const __nv_bfloat16 *qRow =
        q + (uint64_t)head * qHeadStride +
        (uint64_t)token * qTokenStride;
    float2 qValues[kAttentionQkDim / 64];
#pragma unroll
    for (int part = 0; part < kAttentionQkDim / 64; ++part) {
        int d = lane * 2 + part * 64;
        qValues[part] = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162 *>(qRow + d));
    }

    __shared__ float warpMax[warps];
    __shared__ float warpSum[warps];
    __shared__ float unnormalized[kIndexerTopK];
    __shared__ float warpOutput[warps][kAttentionVDim];
    __shared__ float rowMax;
    __shared__ float rowSum;
    extern __shared__ __align__(4) uint8_t sparsePrefillShared[];
    IndexT *selectedIndices =
        reinterpret_cast<IndexT *>(sparsePrefillShared);
    if constexpr (CacheIndices) {
        const IndexT *rowIndices =
            indices + (uint64_t)token * kIndexerTopK;
        for (int selected = threadIdx.x; selected < length;
             selected += blockDim.x) {
            selectedIndices[selected] = rowIndices[selected];
        }
        __syncthreads();
    }
    float localMax = -FLT_MAX;
    for (int selected = warp; selected < length; selected += warps) {
        int keyIndex;
        if constexpr (CacheIndices) {
            keyIndex = selectedIndices[selected];
        } else {
            keyIndex = indices[
                (uint64_t)token * kIndexerTopK + selected];
        }
        const __nv_bfloat16 *kRow =
            k + (uint64_t)head * kHeadStride +
            (uint64_t)keyIndex * kTokenStride;
        float dot = 0.0f;
#pragma unroll
        for (int part = 0; part < kAttentionQkDim / 64; ++part) {
            int d = lane * 2 + part * 64;
            float2 keyValues = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162 *>(kRow + d));
            dot += qValues[part].x * keyValues.x;
            dot += qValues[part].y * keyValues.y;
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
            unnormalized[selected] = score;
            localMax = fmaxf(localMax, score);
        }
    }
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

    for (int selected = threadIdx.x; selected < length;
         selected += blockDim.x) {
        unnormalized[selected] =
            expf(unnormalized[selected] - rowMax);
    }
    __syncthreads();

    // Exponentiation and normalization reduction are independent across the
    // selected keys. Spread both across the complete CTA now that sparse KV
    // gathers no longer hide the lane-0 softmax serialization.
    float localSum = 0.0f;
    for (int selected = threadIdx.x; selected < length;
         selected += blockDim.x) {
        localSum += unnormalized[selected];
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        localSum += __shfl_down_sync(0xffffffffu, localSum, offset);
    }
    if (lane == 0) {
        warpSum[warp] = localSum;
    }
    __syncthreads();
    if (warp == 0) {
        float sum = lane < warps ? warpSum[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        }
        if (lane == 0) {
            rowSum = sum;
        }
    }
    __syncthreads();

    // Transformers converts the FP32 softmax probabilities back to the query
    // dtype before multiplying V.  Preserve that BF16 rounding boundary.
    float2 localOutput[2] = {
        make_float2(0.0f, 0.0f),
        make_float2(0.0f, 0.0f)};
    for (int selected = warp; selected < length; selected += warps) {
        int keyIndex;
        if constexpr (CacheIndices) {
            keyIndex = selectedIndices[selected];
        } else {
            keyIndex = indices[
                (uint64_t)token * kIndexerTopK + selected];
        }
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
        for (int part = 0; part < 2; ++part) {
            int d = lane * 2 + part * 64;
            float2 values = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162 *>(vRow + d));
            localOutput[part].x += probability * values.x;
            localOutput[part].y += probability * values.y;
        }
    }
#pragma unroll
    for (int part = 0; part < 2; ++part) {
        int d = lane * 2 + part * 64;
        warpOutput[warp][d] = localOutput[part].x;
        warpOutput[warp][d + 1] = localOutput[part].y;
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

template <typename IndexT, bool CacheIndices, int Warps>
void LaunchSparseAttentionPrefillRowWithWarps(
        const fastllm::Data &q, const fastllm::Data &k,
        const fastllm::Data &v, const fastllm::Data &indices,
        int startPos, float scale, fastllm::Data &output) {
    int seqlen = q.dims[1];
    size_t sharedBytes = CacheIndices
        ? kIndexerTopK * sizeof(IndexT) : 0;
    SparseAttentionPrefillRowKernel<IndexT, CacheIndices, Warps><<<
        seqlen * kAttentionHeads, Warps * 32, sharedBytes,
        cudaStreamPerThread>>>(
            (const __nv_bfloat16 *)q.cudaData,
            q.strides[0], q.strides[1],
            (const __nv_bfloat16 *)k.cudaData,
            k.strides[0], k.strides[1],
            (const __nv_bfloat16 *)v.cudaData,
            v.strides[0], v.strides[1],
            (const IndexT *)indices.cudaData,
            (__nv_bfloat16 *)output.cudaData,
            seqlen, k.dims[1], startPos, scale);
}

template <typename IndexT, bool CacheIndices>
void LaunchSparseAttentionPrefillRow(
        const fastllm::Data &q, const fastllm::Data &k,
        const fastllm::Data &v, const fastllm::Data &indices,
        int startPos, float scale, fastllm::Data &output) {
    // Turing exposes at most 32 resident warps per SM. A 16-warp CTA reaches
    // full occupancy with two blocks despite this kernel's shared-memory
    // footprint; later architectures retain the lower-overhead 8-warp shape.
    int warps = getCudaInfos()->cudaArch == 750 ? 16 : 8;
    if (const char *configured = std::getenv(
            "FASTLLM_DOTS3_NOTE_SPARSE_PREFILL_WARPS")) {
        int parsed = std::atoi(configured);
        if (parsed == 4 || parsed == 8 || parsed == 16) {
            warps = parsed;
        }
    }
    if (warps == 4) {
        LaunchSparseAttentionPrefillRowWithWarps<
            IndexT, CacheIndices, 4>(
                q, k, v, indices, startPos, scale, output);
    } else if (warps == 16) {
        LaunchSparseAttentionPrefillRowWithWarps<
            IndexT, CacheIndices, 16>(
                q, k, v, indices, startPos, scale, output);
    } else {
        LaunchSparseAttentionPrefillRowWithWarps<
            IndexT, CacheIndices, 8>(
                q, k, v, indices, startPos, scale, output);
    }
}

template <typename IndexT>
__global__ void SparseAttentionPrefillSoftmaxKernel(
        __nv_bfloat16 *scores, const IndexT *indices,
        uint64_t indexTokenStride, int queryOffset, int queryChunk,
        int scoreStride, int totalLen, int startPos, float scale) {
    constexpr int warps = 8;
    int row = blockIdx.x;
    int tokenInChunk = row % queryChunk;
    int token = queryOffset + tokenInChunk;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int length = min(kIndexerTopK, min(totalLen, startPos + token + 1));
    __nv_bfloat16 *rowScores =
        scores + (uint64_t)row * scoreStride;
    const IndexT *rowIndices =
        indices + (uint64_t)token * indexTokenStride;

    __shared__ float selectedProbabilities[kIndexerTopK];
    __shared__ float warpMax[warps];
    __shared__ float warpSum[warps];
    __shared__ float rowMax;
    __shared__ float rowSum;

    float localMax = -FLT_MAX;
    for (int selected = threadIdx.x; selected < length;
         selected += blockDim.x) {
        int keyIndex = rowIndices[selected];
        float score = __bfloat162float(rowScores[keyIndex]) * scale;
        selectedProbabilities[selected] = score;
        localMax = fmaxf(localMax, score);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        localMax = fmaxf(
            localMax,
            __shfl_down_sync(0xffffffffu, localMax, offset));
    }
    if (lane == 0) {
        warpMax[warp] = localMax;
    }
    __syncthreads();
    if (warp == 0) {
        float maximum = lane < warps ? warpMax[lane] : -FLT_MAX;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            maximum = fmaxf(
                maximum,
                __shfl_down_sync(0xffffffffu, maximum, offset));
        }
        if (lane == 0) {
            rowMax = maximum;
        }
    }
    __syncthreads();

    for (int selected = threadIdx.x; selected < length;
         selected += blockDim.x) {
        selectedProbabilities[selected] =
            expf(selectedProbabilities[selected] - rowMax);
    }
    __syncthreads();

    // Score gathering and exponentiation dominate this kernel, so spread
    // them across the complete block. Keep the original eight partial sums
    // and their accumulation order to avoid changing the BF16 softmax
    // rounding boundary.
    if (lane == 0) {
        float localSum = 0.0f;
        for (int selected = warp; selected < length;
             selected += warps) {
            float probability = selectedProbabilities[selected];
            localSum += probability;
        }
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

    for (int keyIndex = threadIdx.x; keyIndex < scoreStride;
         keyIndex += blockDim.x) {
        rowScores[keyIndex] = __float2bfloat16_rn(0.0f);
    }
    __syncthreads();
    for (int selected = threadIdx.x; selected < length;
         selected += blockDim.x) {
        int keyIndex = rowIndices[selected];
        rowScores[keyIndex] = __float2bfloat16_rn(
            selectedProbabilities[selected] / rowSum);
    }
}

__global__ void SlidingAttentionPrefillSoftmaxKernel(
        __nv_bfloat16 *scores, int queryOffset, int queryChunk,
        int keyStart, int keyCount, int startPos, int windowSize,
        float scale) {
    constexpr int warps = 8;
    int row = blockIdx.x;
    int token = queryOffset + row % queryChunk;
    int queryPosition = startPos + token;
    int firstKey = max(keyStart, queryPosition - windowSize + 1);
    int lastKey = min(keyStart + keyCount, queryPosition + 1);
    int visible = lastKey - firstKey;
    int visibleOffset = firstKey - keyStart;
    __nv_bfloat16 *rowScores = scores + (uint64_t)row * keyCount;
    extern __shared__ float probabilities[];
    __shared__ float warpValues[warps];
    __shared__ float rowMaximum;
    __shared__ float rowSum;

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    float localMaximum = -FLT_MAX;
    for (int i = threadIdx.x; i < visible; i += blockDim.x) {
        float value =
            __bfloat162float(rowScores[visibleOffset + i]) * scale;
        probabilities[i] = value;
        localMaximum = fmaxf(localMaximum, value);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        localMaximum = fmaxf(
            localMaximum,
            __shfl_down_sync(0xffffffffu, localMaximum, offset));
    }
    if (lane == 0) {
        warpValues[warp] = localMaximum;
    }
    __syncthreads();
    if (warp == 0) {
        float value = lane < warps ? warpValues[lane] : -FLT_MAX;
        for (int offset = 16; offset > 0; offset >>= 1) {
            value = fmaxf(
                value, __shfl_down_sync(0xffffffffu, value, offset));
        }
        if (lane == 0) {
            rowMaximum = value;
        }
    }
    __syncthreads();

    float localSum = 0.0f;
    for (int i = threadIdx.x; i < visible; i += blockDim.x) {
        float value = expf(probabilities[i] - rowMaximum);
        probabilities[i] = value;
        localSum += value;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        localSum += __shfl_down_sync(0xffffffffu, localSum, offset);
    }
    if (lane == 0) {
        warpValues[warp] = localSum;
    }
    __syncthreads();
    if (warp == 0) {
        float value = lane < warps ? warpValues[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if (lane == 0) {
            rowSum = value;
        }
    }
    __syncthreads();

    for (int key = threadIdx.x; key < keyCount; key += blockDim.x) {
        rowScores[key] = __float2bfloat16_rn(0.0f);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < visible; i += blockDim.x) {
        rowScores[visibleOffset + i] = __float2bfloat16_rn(
            probabilities[i] / rowSum);
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

bool FastllmCudaDots3NotePackAttentionKey(
        const fastllm::Data &kv, const fastllm::Data &rope,
        int nopeDim, fastllm::Data &output) {
    if (nopeDim <= 0 ||
        kv.dataDevice != fastllm::DataDevice::CUDA ||
        kv.dataType != fastllm::DataType::BFLOAT16 ||
        kv.cudaData == nullptr || kv.dims.size() != 3 ||
        kv.dims[0] <= 0 || kv.dims[1] <= 0 || kv.dims[2] <= nopeDim ||
        kv.strides[2] != 1 ||
        rope.dataDevice != fastllm::DataDevice::CUDA ||
        rope.dataType != fastllm::DataType::BFLOAT16 ||
        rope.cudaData == nullptr || rope.dims.size() != 2 ||
        rope.dims[0] != kv.dims[1] || rope.dims[1] <= 0 ||
        rope.strides[1] != 1) {
        return false;
    }
    int device = GetPointerDeviceId(kv.cudaData);
    if (device < 0 || GetPointerDeviceId(rope.cudaData) != device) {
        return false;
    }
    FastllmCudaSetDevice(device);
    int heads = kv.dims[0];
    int tokens = kv.dims[1];
    int ropeDim = rope.dims[1];
    int qkDim = nopeDim + ropeDim;
    if (!PrepareOutput(output, fastllm::DataType::BFLOAT16,
                       {heads, tokens, qkDim})) {
        return false;
    }
    uint64_t elements = (uint64_t)heads * tokens * qkDim;
    int blocks = (int)((elements + 255) / 256);
    PackAttentionKeyKernel<<<blocks, 256, 0, cudaStreamPerThread>>>(
        (const __nv_bfloat16 *)kv.cudaData,
        (const __nv_bfloat16 *)rope.cudaData,
        (__nv_bfloat16 *)output.cudaData, elements, tokens,
        nopeDim, ropeDim,
        kv.strides[0], kv.strides[1], rope.strides[0],
        output.strides[0], output.strides[1]);
    return cudaGetLastError() == cudaSuccess;
}

bool FastllmCudaDots3NoteQuantizeIndexer(
        const fastllm::Data &qRope, const fastllm::Data &qNope,
        const fastllm::Data &k,
        const fastllm::Data &weights, fastllm::Data &qFp8,
        fastllm::Data &foldedWeights, fastllm::Data &kFp8,
        fastllm::Data &kScales) {
    if (qRope.dataDevice != fastllm::DataDevice::CUDA ||
        qRope.cudaData == nullptr ||
        (qRope.dataType != fastllm::DataType::FLOAT32 &&
         qRope.dataType != fastllm::DataType::BFLOAT16) ||
        qRope.dims.size() != 4 || qRope.dims[0] != 1 ||
        qRope.dims[2] != kIndexerHeads || qRope.dims[3] != 64 ||
        qNope.dataDevice != fastllm::DataDevice::CUDA ||
        qNope.cudaData == nullptr || qNope.dataType != qRope.dataType ||
        qNope.dims != qRope.dims ||
        k.dataDevice != fastllm::DataDevice::CUDA || k.cudaData == nullptr ||
        (k.dataType != fastllm::DataType::FLOAT32 &&
         k.dataType != fastllm::DataType::BFLOAT16) ||
        k.dims.size() != 3 || k.dims[0] != 1 ||
        k.dims[2] != kIndexerDim ||
        weights.dataDevice != fastllm::DataDevice::CUDA ||
        weights.cudaData == nullptr || weights.dims.size() != 3 ||
        weights.dims[0] != 1 || weights.dims[1] != qRope.dims[1] ||
        weights.dims[2] != kIndexerHeads ||
        (weights.dataType != fastllm::DataType::FLOAT32 &&
         weights.dataType != fastllm::DataType::BFLOAT16)) {
        return false;
    }
    int device = GetPointerDeviceId(qRope.cudaData);
    if (device < 0 || GetPointerDeviceId(qNope.cudaData) != device ||
        GetPointerDeviceId(k.cudaData) != device ||
        GetPointerDeviceId(weights.cudaData) != device) {
        return false;
    }
    FastllmCudaSetDevice(device);
    int qTokens = qRope.dims[1];
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
    if (qRope.dataType == fastllm::DataType::FLOAT32 &&
        weights.dataType == fastllm::DataType::BFLOAT16) {
        QuantizeSplitQKernel<<<qRows, kIndexerDim, 0,
                               cudaStreamPerThread>>>(
            (const float *)qRope.cudaData,
            (const float *)qNope.cudaData,
            (const __nv_bfloat16 *)weights.cudaData,
            (uint8_t *)qFp8.cudaData, (float *)foldedWeights.cudaData,
            qRows);
    } else if (qRope.dataType == fastllm::DataType::FLOAT32 &&
               weights.dataType == fastllm::DataType::FLOAT32) {
        QuantizeSplitQKernel<<<qRows, kIndexerDim, 0,
                               cudaStreamPerThread>>>(
            (const float *)qRope.cudaData,
            (const float *)qNope.cudaData,
            (const float *)weights.cudaData,
            (uint8_t *)qFp8.cudaData, (float *)foldedWeights.cudaData,
            qRows);
    } else if (qRope.dataType == fastllm::DataType::BFLOAT16 &&
               weights.dataType == fastllm::DataType::BFLOAT16) {
        QuantizeSplitQKernel<<<qRows, kIndexerDim, 0,
                               cudaStreamPerThread>>>(
            (const __nv_bfloat16 *)qRope.cudaData,
            (const __nv_bfloat16 *)qNope.cudaData,
            (const __nv_bfloat16 *)weights.cudaData,
            (uint8_t *)qFp8.cudaData, (float *)foldedWeights.cudaData,
            qRows);
    } else {
        QuantizeSplitQKernel<<<qRows, kIndexerDim, 0,
                               cudaStreamPerThread>>>(
            (const __nv_bfloat16 *)qRope.cudaData,
            (const __nv_bfloat16 *)qNope.cudaData,
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

static bool Dots3NoteIndexerTopKImpl(
        const fastllm::Data &qFp8,
        const fastllm::Data &foldedWeights,
        const fastllm::Data &kFp8, const fastllm::Data &kScales,
        int startPos, int topK, fastllm::Data &indices,
        void **headScoreWorkspace,
        size_t *headScoreWorkspaceBytes,
        bool *tensorCorePathUsed) {
    if (tensorCorePathUsed != nullptr) {
        *tensorCorePathUsed = false;
    }
    if ((headScoreWorkspace == nullptr) !=
            (headScoreWorkspaceBytes == nullptr) ||
        topK != kIndexerTopK || startPos < 0 ||
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
    bool canReuseWorkspace = headScoreWorkspace != nullptr &&
                             headScoreWorkspaceBytes != nullptr;
    if (canReuseWorkspace && *headScoreWorkspace != nullptr) {
        int workspaceDevice = GetPointerDeviceId(*headScoreWorkspace);
        if (workspaceDevice < 0) {
            return false;
        }
        if (workspaceDevice != device) {
            FastllmCudaSetDevice(workspaceDevice);
            FastllmCudaDirectFree(*headScoreWorkspace);
            FastllmCudaSetDevice(device);
            *headScoreWorkspace = nullptr;
            *headScoreWorkspaceBytes = 0;
        }
    }
    int seqlen = qFp8.dims[1];
    int totalLen = kFp8.dims[1];
    bool useCompactIndices = totalLen <= 32768;
    if (seqlen <= 0 || totalLen <= kIndexerTopK ||
        startPos + seqlen > totalLen ||
        !PrepareOutput(indices,
                       useCompactIndices ? fastllm::DataType::INT16
                                         : fastllm::DataType::INT32,
                       {seqlen, kIndexerTopK})) {
        return false;
    }

    // Score eight keys per block so all of them reuse the same 8 KiB query
    // tile. The K vector is also decoded from FP8 only once per key and reused
    // across all 64 indexer heads. A four-pass radix selector then finds the
    // exact Top-2048 set without sorting every score in the row. Bounded query
    // slices keep the temporary footprint independent of the prefill size.
    // On supported Tensor Core GPUs, long rows use FP16 inputs on pre-SM89
    // targets and native FP8 inputs on SM89+. Both paths exactly rescore a
    // bounded candidate set before final selection. Short rows keep the
    // lower-overhead scalar kernel.
    int compiledArch = getCudaInfos()->cudaArch;
    bool useFp8TensorCore = compiledArch >= 890;
    bool useFp16TensorCore = compiledArch >= 750 && compiledArch < 890;
    bool useTensorCore = (useFp8TensorCore || useFp16TensorCore) &&
                         !FastllmCudaGraphIsCapturingFast();
    if (const char *env = std::getenv(
            "FASTLLM_DOTS3_NOTE_INDEXER_TENSOR_CORE")) {
        useTensorCore = useTensorCore && EnvValueEnabled(env);
    }
    int maxRowsPerChunk = INT_MAX / totalLen;
    if (maxRowsPerChunk <= 0) {
        return false;
    }
    int queryChunk = std::min(
        seqlen, std::min(kIndexerDefaultQueryChunk, maxRowsPerChunk));
    int scalarQueryChunk = queryChunk;
    useTensorCore = useTensorCore && seqlen >= 16 &&
                    seqlen % 16 == 0 && startPos % 16 == 0 &&
                    totalLen % 16 == 0 &&
                    totalLen >= 8192;
    if (useTensorCore) {
        size_t headScoreBytesPerQuery =
            (size_t)totalLen * kIndexerHeads *
            sizeof(__half);
        int tensorDefaultChunk = (int)std::min<size_t>(
            512, std::max<size_t>(
                     16,
                     kIndexerHeadScoreLimit / headScoreBytesPerQuery));
        tensorDefaultChunk = tensorDefaultChunk / 16 * 16;
        queryChunk = std::min(seqlen, tensorDefaultChunk);
        queryChunk = std::max(16, queryChunk / 16 * 16);
    }
    size_t candidateIndexBytes = useCompactIndices
        ? sizeof(int16_t) : sizeof(int32_t);
    auto additionalWorkspaceBytes = [&](int rows) {
        size_t bytes = (size_t)rows * totalLen * sizeof(float);
        if (!useTensorCore) {
            return bytes;
        }
        size_t requiredHeadScoreBytes =
            (size_t)rows * totalLen * kIndexerHeads * sizeof(__half);
        size_t reusableHeadScoreBytes =
            canReuseWorkspace && *headScoreWorkspace != nullptr
                ? *headScoreWorkspaceBytes : 0;
        if (requiredHeadScoreBytes > reusableHeadScoreBytes) {
            bytes += requiredHeadScoreBytes - reusableHeadScoreBytes;
        }
        bytes += (size_t)rows * kIndexerCandidateK *
                 (candidateIndexBytes + sizeof(float));
        if (useFp16TensorCore) {
            bytes += (size_t)totalLen * kIndexerDim * sizeof(__half);
            bytes += (size_t)rows * kIndexerHeads * kIndexerDim *
                     sizeof(__half);
        }
        return bytes;
    };
    // Treat the selected chunk as an upper bound. Late transformer layers can
    // have less free memory after their persistent KV cache is created. Count
    // all Tensor Core-only buffers, including the fixed converted K matrix.
    std::vector<long long> freeSizes = FastllmCudaGetFreeSizes();
    size_t budget = SIZE_MAX;
    if (device < (int)freeSizes.size() && freeSizes[device] > 0) {
        constexpr size_t reserveBytes = 16ULL * 1024ULL * 1024ULL;
        size_t freeBytes = (size_t)freeSizes[device];
        budget = freeBytes > reserveBytes * 2
                     ? freeBytes - reserveBytes
                     : std::max<size_t>(1, freeBytes / 2);
    }
    if (useTensorCore) {
        while (queryChunk > 16 &&
               additionalWorkspaceBytes(queryChunk) > budget) {
            queryChunk -= 16;
        }
        if (additionalWorkspaceBytes(queryChunk) > budget) {
            useTensorCore = false;
            queryChunk = scalarQueryChunk;
        }
    }
    if (!useTensorCore) {
        while (queryChunk > 1 &&
               additionalWorkspaceBytes(queryChunk) > budget) {
            queryChunk = (queryChunk + 1) / 2;
        }
    }
    int maxPairs = queryChunk * totalLen;
    float *logits = (float *)FastllmCudaMalloc(
        (size_t)maxPairs * sizeof(float));
    if (logits == nullptr) {
        return false;
    }
    __half *headScores = nullptr;
    bool ownsHeadScores = false;
    bool ownsReusableHeadScores = false;
    FastllmCudaTryMallocResult tensorSetupResult =
        FASTLLM_CUDA_TRY_MALLOC_SUCCESS;
    void *candidateIds = nullptr;
    float *candidateScores = nullptr;
    __half *fp16K = nullptr;
    __half *fp16Q = nullptr;
    auto tryPooledAllocation = [&](void **ret, size_t bytes) {
        if (tensorSetupResult == FASTLLM_CUDA_TRY_MALLOC_SUCCESS) {
            tensorSetupResult = FastllmCudaTryMalloc(ret, bytes);
        }
    };
    auto tryDirectAllocation = [&](void **ret, size_t bytes) {
        if (tensorSetupResult == FASTLLM_CUDA_TRY_MALLOC_SUCCESS) {
            tensorSetupResult = FastllmCudaTryDirectMalloc(ret, bytes);
        }
    };
    auto releaseTensorCoreSetup = [&]() {
        if (ownsHeadScores && headScores != nullptr) {
            FastllmCudaDirectFree(headScores);
        } else if (ownsReusableHeadScores &&
                   *headScoreWorkspace != nullptr) {
            FastllmCudaDirectFree(*headScoreWorkspace);
            *headScoreWorkspace = nullptr;
            *headScoreWorkspaceBytes = 0;
        }
        if (candidateIds != nullptr) FastllmCudaFree(candidateIds);
        if (candidateScores != nullptr) FastllmCudaFree(candidateScores);
        if (fp16K != nullptr) FastllmCudaFree(fp16K);
        if (fp16Q != nullptr) FastllmCudaFree(fp16Q);
        headScores = nullptr;
        candidateIds = nullptr;
        candidateScores = nullptr;
        fp16K = nullptr;
        fp16Q = nullptr;
        ownsHeadScores = false;
        ownsReusableHeadScores = false;
    };
    if (useTensorCore) {
        size_t requiredHeadScoreBytes =
            (size_t)maxPairs * kIndexerHeads *
            sizeof(__half);
        if (canReuseWorkspace &&
            (*headScoreWorkspace == nullptr ||
             *headScoreWorkspaceBytes < requiredHeadScoreBytes)) {
            if (*headScoreWorkspace != nullptr) {
                FastllmCudaDirectFree(*headScoreWorkspace);
            }
            *headScoreWorkspace = nullptr;
            tryDirectAllocation(
                headScoreWorkspace, requiredHeadScoreBytes);
            *headScoreWorkspaceBytes = *headScoreWorkspace == nullptr
                ? 0 : requiredHeadScoreBytes;
            ownsReusableHeadScores = *headScoreWorkspace != nullptr;
        }
        if (canReuseWorkspace) {
            headScores = (__half *)*headScoreWorkspace;
        } else {
            void *allocatedHeadScores = nullptr;
            tryDirectAllocation(
                &allocatedHeadScores, requiredHeadScoreBytes);
            headScores = (__half *)allocatedHeadScores;
            ownsHeadScores = headScores != nullptr;
        }
    }
    if (useTensorCore &&
        tensorSetupResult == FASTLLM_CUDA_TRY_MALLOC_SUCCESS) {
        tryPooledAllocation(
            &candidateIds,
            (size_t)queryChunk * kIndexerCandidateK * candidateIndexBytes);
        if (candidateIds != nullptr) {
            void *allocatedCandidateScores = nullptr;
            tryPooledAllocation(
                &allocatedCandidateScores,
                (size_t)queryChunk * kIndexerCandidateK * sizeof(float));
            candidateScores = (float *)allocatedCandidateScores;
        }
    }
    if (useTensorCore &&
        (headScores == nullptr || candidateIds == nullptr ||
         candidateScores == nullptr)) {
        releaseTensorCoreSetup();
        if (tensorSetupResult == FASTLLM_CUDA_TRY_MALLOC_ERROR ||
            FastllmCudaGetThreadError()) {
            FastllmCudaFree(logits);
            return false;
        }
        useTensorCore = false;
    }
    if (useTensorCore && useFp16TensorCore) {
        size_t kElements = (size_t)totalLen * kIndexerDim;
        size_t qElements = (size_t)queryChunk * kIndexerHeads *
                           kIndexerDim;
        void *allocatedFp16K = nullptr;
        tryPooledAllocation(
            &allocatedFp16K, kElements * sizeof(__half));
        fp16K = (__half *)allocatedFp16K;
        if (fp16K != nullptr) {
            void *allocatedFp16Q = nullptr;
            tryPooledAllocation(
                &allocatedFp16Q, qElements * sizeof(__half));
            fp16Q = (__half *)allocatedFp16Q;
        }
        if (fp16K != nullptr && fp16Q != nullptr) {
            int threads = 256;
            uint64_t packedCount = kElements / 4;
            IndexerFp8ToHalfKernel<<<
                (packedCount + threads - 1) / threads, threads, 0,
                cudaStreamPerThread>>>(
                    (const uint8_t *)kFp8.cudaData, fp16K,
                    packedCount);
            if (cudaGetLastError() != cudaSuccess) {
                releaseTensorCoreSetup();
                useTensorCore = false;
            }
        } else {
            releaseTensorCoreSetup();
            if (tensorSetupResult == FASTLLM_CUDA_TRY_MALLOC_ERROR ||
                FastllmCudaGetThreadError()) {
                FastllmCudaFree(logits);
                return false;
            }
            useTensorCore = false;
        }
    }
    cudaError_t state = cudaSuccess;
    bool useRadixCompaction = !EnvValueEnabled(std::getenv(
        "FASTLLM_DOTS3_NOTE_DISABLE_INDEXER_RADIX_COMPACTION"));
    bool useParallelRadixEmit = !EnvValueEnabled(std::getenv(
        "FASTLLM_DOTS3_NOTE_DISABLE_PARALLEL_RADIX_EMIT"));
    if (state == cudaSuccess) {
        const uint8_t *allQ = (const uint8_t *)qFp8.cudaData;
        const float *allWeights =
            (const float *)foldedWeights.cudaData;
        for (int queryOffset = 0; queryOffset < seqlen;
             queryOffset += queryChunk) {
            int current = std::min(queryChunk, seqlen - queryOffset);
            const uint8_t *currentQ = allQ +
                (uint64_t)queryOffset * kIndexerHeads * kIndexerDim;
            const float *currentWeights = allWeights +
                (uint64_t)queryOffset * kIndexerHeads;
            if (useTensorCore) {
                int keyCount = std::min(
                    totalLen, startPos + queryOffset + current);
                cublasStatus_t status;
                if (useFp16TensorCore) {
                    size_t qElements = (size_t)current *
                        kIndexerHeads * kIndexerDim;
                    int threads = 256;
                    uint64_t packedCount = qElements / 4;
                    IndexerFp8ToHalfKernel<<<
                        (packedCount + threads - 1) / threads,
                        threads, 0, cudaStreamPerThread>>>(
                            currentQ, fp16Q, packedCount);
                    status = cudaGetLastError() == cudaSuccess
                        ? IndexerFp16BatchedGemm(
                              fp16K, fp16Q, headScores,
                              current, keyCount)
                        : CUBLAS_STATUS_EXECUTION_FAILED;
                } else {
                    status = IndexerFp8BatchedGemm(
                        kFp8.cudaData, currentQ, headScores,
                        current, keyCount);
                }
                if (status != CUBLAS_STATUS_SUCCESS) {
                    // A trial Tensor Core path must not leave a stale runtime
                    // error that makes the scalar launch appear to fail.
                    cudaGetLastError();
                    useTensorCore = false;
                    int keyTiles =
                        (totalLen + kIndexerKeysPerBlock - 1) /
                        kIndexerKeysPerBlock;
                    IndexerLogitsTiledKernel<<<
                        current * keyTiles, 256, 0,
                        cudaStreamPerThread>>>(
                            currentQ,
                            (const uint8_t *)kFp8.cudaData,
                            (const float *)kScales.cudaData,
                            currentWeights, logits, current,
                            totalLen, startPos + queryOffset);
                    state = cudaPeekAtLastError();
                } else {
                    dim3 grid((keyCount + 255) / 256, current);
                    IndexerReduceHeadScoresKernel<__half><<<
                        grid, 256, 0, cudaStreamPerThread>>>(
                            headScores,
                            (const float *)kScales.cudaData,
                            currentWeights, logits, queryOffset,
                            current, keyCount, totalLen, startPos);
                    state = cudaPeekAtLastError();
                }
            } else {
                int keyTiles =
                    (totalLen + kIndexerKeysPerBlock - 1) /
                    kIndexerKeysPerBlock;
                IndexerLogitsTiledKernel<<<current * keyTiles, 256, 0,
                                           cudaStreamPerThread>>>(
                    currentQ, (const uint8_t *)kFp8.cudaData,
                    (const float *)kScales.cudaData, currentWeights,
                    logits, current, totalLen,
                    startPos + queryOffset);
                state = cudaPeekAtLastError();
            }
            if (state != cudaSuccess) {
                break;
            }
            // The reduction above has consumed every head score. Until the
            // next query chunk, that much larger buffer can hold one compact
            // radix bucket per query without increasing peak workspace.
            if (useTensorCore && useCompactIndices) {
                int16_t *candidates = (int16_t *)candidateIds;
                int16_t *allIndices = (int16_t *)indices.cudaData;
                IndexerRadixSelectKernel<
                    int16_t, kIndexerCandidateK><<<
                        current, kIndexerRadixThreads, 0,
                        cudaStreamPerThread>>>(
                            logits, candidates, nullptr,
                            current, totalLen,
                            startPos + queryOffset,
                            useRadixCompaction
                                ? (int16_t *)headScores : nullptr,
                            totalLen, useParallelRadixEmit);
                IndexerCandidateExactScoresKernel<int16_t><<<
                    current * (kIndexerCandidateK /
                               kIndexerCandidateKeysPerBlock),
                    kIndexerCandidateKeysPerBlock * 32,
                    0, cudaStreamPerThread>>>(
                        currentQ,
                        (const uint8_t *)kFp8.cudaData,
                        (const float *)kScales.cudaData,
                        currentWeights, candidates,
                        candidateScores, current, totalLen,
                        startPos + queryOffset);
                IndexerRadixSelectKernel<
                    int16_t, kIndexerTopK><<<
                        current, kIndexerRadixThreads, 0,
                        cudaStreamPerThread>>>(
                            candidateScores,
                            allIndices +
                                (uint64_t)queryOffset * kIndexerTopK,
                            candidates, current,
                            kIndexerCandidateK,
                            kIndexerCandidateK, nullptr, 0,
                            useParallelRadixEmit);
            } else if (useTensorCore) {
                int32_t *candidates = (int32_t *)candidateIds;
                int32_t *allIndices = (int32_t *)indices.cudaData;
                IndexerRadixSelectKernel<
                    int32_t, kIndexerCandidateK><<<
                        current, kIndexerRadixThreads, 0,
                        cudaStreamPerThread>>>(
                            logits, candidates, nullptr,
                            current, totalLen,
                            startPos + queryOffset,
                            useRadixCompaction
                                ? (int32_t *)headScores : nullptr,
                            totalLen, useParallelRadixEmit);
                IndexerCandidateExactScoresKernel<int32_t><<<
                    current * (kIndexerCandidateK /
                               kIndexerCandidateKeysPerBlock),
                    kIndexerCandidateKeysPerBlock * 32,
                    0, cudaStreamPerThread>>>(
                        currentQ,
                        (const uint8_t *)kFp8.cudaData,
                        (const float *)kScales.cudaData,
                        currentWeights, candidates,
                        candidateScores, current, totalLen,
                        startPos + queryOffset);
                IndexerRadixSelectKernel<
                    int32_t, kIndexerTopK><<<
                        current, kIndexerRadixThreads, 0,
                        cudaStreamPerThread>>>(
                            candidateScores,
                            allIndices +
                                (uint64_t)queryOffset * kIndexerTopK,
                            candidates, current,
                            kIndexerCandidateK,
                            kIndexerCandidateK, nullptr, 0,
                            useParallelRadixEmit);
            } else if (useCompactIndices) {
                int16_t *allIndices = (int16_t *)indices.cudaData;
                IndexerRadixSelectKernel<
                    int16_t, kIndexerTopK><<<
                        current, kIndexerRadixThreads, 0,
                        cudaStreamPerThread>>>(
                            logits,
                            allIndices +
                                (uint64_t)queryOffset * kIndexerTopK,
                            nullptr, current, totalLen,
                            startPos + queryOffset, nullptr, 0,
                            useParallelRadixEmit);
            } else {
                int32_t *allIndices = (int32_t *)indices.cudaData;
                IndexerRadixSelectKernel<
                    int32_t, kIndexerTopK><<<
                        current, kIndexerRadixThreads, 0,
                        cudaStreamPerThread>>>(
                            logits,
                            allIndices +
                                (uint64_t)queryOffset * kIndexerTopK,
                            nullptr, current, totalLen,
                            startPos + queryOffset, nullptr, 0,
                            useParallelRadixEmit);
            }
            state = cudaPeekAtLastError();
            if (state != cudaSuccess) break;
        }
    }
    cudaError_t syncState = cudaStreamSynchronize(cudaStreamPerThread);
    bool ok = state == cudaSuccess && syncState == cudaSuccess;
    if (ownsHeadScores) FastllmCudaDirectFree(headScores);
    if (candidateIds) FastllmCudaFree(candidateIds);
    if (candidateScores) FastllmCudaFree(candidateScores);
    if (fp16K) FastllmCudaFree(fp16K);
    if (fp16Q) FastllmCudaFree(fp16Q);
    FastllmCudaFree(logits);
    if (tensorCorePathUsed != nullptr) {
        *tensorCorePathUsed = useTensorCore && ok;
    }
    return ok;
}

bool FastllmCudaDots3NoteIndexerTopK(
        const fastllm::Data &qFp8,
        const fastllm::Data &foldedWeights,
        const fastllm::Data &kFp8, const fastllm::Data &kScales,
        int startPos, int topK, fastllm::Data &indices,
        void **headScoreWorkspace,
        size_t *headScoreWorkspaceBytes) {
    return Dots3NoteIndexerTopKImpl(
        qFp8, foldedWeights, kFp8, kScales, startPos, topK, indices,
        headScoreWorkspace, headScoreWorkspaceBytes, nullptr);
}

bool FastllmCudaDots3NoteIndexerTopKWithPathInfo(
        const fastllm::Data &qFp8,
        const fastllm::Data &foldedWeights,
        const fastllm::Data &kFp8, const fastllm::Data &kScales,
        int startPos, int topK, fastllm::Data &indices,
        void **headScoreWorkspace,
        size_t *headScoreWorkspaceBytes,
        bool *tensorCorePathUsed) {
    return Dots3NoteIndexerTopKImpl(
        qFp8, foldedWeights, kFp8, kScales, startPos, topK, indices,
        headScoreWorkspace, headScoreWorkspaceBytes, tensorCorePathUsed);
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
        (indices.dataType != fastllm::DataType::INT16 &&
         indices.dataType != fastllm::DataType::INT32) ||
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
    auto supportsVectorPairs = [](const fastllm::Data &data) {
        return data.cudaData != nullptr && data.strides.size() == 3 &&
               (reinterpret_cast<uintptr_t>(data.cudaData) & 3u) == 0 &&
               (data.strides[0] & 1u) == 0 &&
               (data.strides[1] & 1u) == 0 && data.strides[2] == 1;
    };
    bool optimizedPrefill = seqlen >= 16 && supportsVectorPairs(q) &&
                            supportsVectorPairs(k) &&
                            supportsVectorPairs(v);
    bool disableCachedSparseIndices = EnvValueEnabled(std::getenv(
        "FASTLLM_DOTS3_NOTE_DISABLE_CACHED_SPARSE_INDICES"));
    bool cachePrefillIndices = !disableCachedSparseIndices;
    if (indices.dataType == fastllm::DataType::INT16 && optimizedPrefill) {
        if (cachePrefillIndices) {
            LaunchSparseAttentionPrefillRow<int16_t, true>(
                q, k, v, indices, startPos, scale, output);
        } else {
            LaunchSparseAttentionPrefillRow<int16_t, false>(
                q, k, v, indices, startPos, scale, output);
        }
    } else if (indices.dataType == fastllm::DataType::INT32 &&
               optimizedPrefill) {
        // INT32 indices need twice as much shared storage and reduce active
        // CTAs on SM89, so keep their established direct-load path.
        LaunchSparseAttentionPrefillRow<int32_t, false>(
            q, k, v, indices, startPos, scale, output);
    } else if (indices.dataType == fastllm::DataType::INT16) {
        SparseAttentionKernel<int16_t><<<
            seqlen * kAttentionHeads, 256, 0,
            cudaStreamPerThread>>>(
                (const __nv_bfloat16 *)q.cudaData,
                q.strides[0], q.strides[1],
                (const __nv_bfloat16 *)k.cudaData,
                k.strides[0], k.strides[1],
                (const __nv_bfloat16 *)v.cudaData,
                v.strides[0], v.strides[1],
                (const int16_t *)indices.cudaData,
                (__nv_bfloat16 *)output.cudaData,
                seqlen, k.dims[1], startPos, scale);
    } else {
        SparseAttentionKernel<int32_t><<<
            seqlen * kAttentionHeads, 256, 0,
            cudaStreamPerThread>>>(
                (const __nv_bfloat16 *)q.cudaData,
                q.strides[0], q.strides[1],
                (const __nv_bfloat16 *)k.cudaData,
                k.strides[0], k.strides[1],
                (const __nv_bfloat16 *)v.cudaData,
                v.strides[0], v.strides[1],
                (const int32_t *)indices.cudaData,
                (__nv_bfloat16 *)output.cudaData,
                seqlen, k.dims[1], startPos, scale);
    }
    cudaError_t launchState = cudaGetLastError();
    DeviceSync();
    return launchState == cudaSuccess;
}

bool FastllmCudaDots3NoteSparseAttentionPrefill(
        const fastllm::Data &q, const fastllm::Data &k,
        const fastllm::Data &v, const fastllm::Data &indices,
        int startPos, float scale, fastllm::Data &output,
        void *borrowedScratch, size_t borrowedScratchBytes) {
    if (startPos < 0 || q.dataDevice != fastllm::DataDevice::CUDA ||
        q.dataType != fastllm::DataType::BFLOAT16 || q.cudaData == nullptr ||
        q.dims.size() != 3 || q.dims[0] != kAttentionHeads ||
        q.dims[2] != kAttentionQkDim || q.strides[2] != 1 ||
        k.dataDevice != fastllm::DataDevice::CUDA ||
        k.dataType != fastllm::DataType::BFLOAT16 || k.cudaData == nullptr ||
        k.dims.size() != 3 || k.dims[0] != kAttentionHeads ||
        k.dims[2] != kAttentionQkDim || k.strides[2] != 1 ||
        v.dataDevice != fastllm::DataDevice::CUDA ||
        v.dataType != fastllm::DataType::BFLOAT16 || v.cudaData == nullptr ||
        v.dims.size() != 3 || v.dims[0] != kAttentionHeads ||
        v.dims[1] != k.dims[1] || v.dims[2] != kAttentionVDim ||
        v.strides[2] != 1 ||
        indices.dataDevice != fastllm::DataDevice::CUDA ||
        (indices.dataType != fastllm::DataType::INT16 &&
         indices.dataType != fastllm::DataType::INT32) ||
        indices.cudaData == nullptr || indices.dims.size() != 2 ||
        indices.dims[0] != q.dims[1] ||
        indices.dims[1] != kIndexerTopK || indices.strides[1] != 1 ||
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
    int totalLen = k.dims[1];
    if (seqlen <= 0 || totalLen <= kIndexerTopK ||
        !PrepareOutput(output, fastllm::DataType::BFLOAT16,
                       {kAttentionHeads, seqlen, kAttentionVDim})) {
        return false;
    }
    if ((borrowedScratch == nullptr) != (borrowedScratchBytes == 0) ||
        (borrowedScratch != nullptr &&
         GetPointerDeviceId(borrowedScratch) != device)) {
        return false;
    }

    // Reuse the DeepSeek-V4 prefill strategy: bound the dense score scratch,
    // run QK and AV on Tensor Cores, and apply the sparse selection in-place
    // between the two GEMMs. Dots shares one index row across all heads, so no
    // per-head index expansion is needed.
    size_t bytesPerQuery =
        (size_t)kAttentionHeads * totalLen * sizeof(__nv_bfloat16);
    // Borrowed storage is the unused tail of the already-live KV projection,
    // so using all of it does not raise peak memory. At 8K this doubles the
    // query tile from 64 to 128 and halves the number of QK/AV launch pairs.
    size_t availableScratch = borrowedScratch == nullptr
        ? kSparsePrefillScratchLimit : borrowedScratchBytes;
    if (availableScratch < bytesPerQuery) {
        return false;
    }
    int queryChunk = (int)std::max<size_t>(
        1, availableScratch / bytesPerQuery);
    queryChunk = std::min(queryChunk, seqlen);
    size_t scratchBytes =
        (size_t)queryChunk * bytesPerQuery;
    bool ownsScores = borrowedScratch == nullptr;
    __nv_bfloat16 *scores = ownsScores
        ? (__nv_bfloat16 *)FastllmCudaMalloc(scratchBytes)
        : (__nv_bfloat16 *)borrowedScratch;
    if (scores == nullptr) {
        return false;
    }

    const __nv_bfloat16 *qData =
        (const __nv_bfloat16 *)q.cudaData;
    const __nv_bfloat16 *kData =
        (const __nv_bfloat16 *)k.cudaData;
    const __nv_bfloat16 *vData =
        (const __nv_bfloat16 *)v.cudaData;
    __nv_bfloat16 *outputData =
        (__nv_bfloat16 *)output.cudaData;
    cublasHandle_t handle = getFastllmCublasHandle();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    bool ok = true;
    for (int queryOffset = 0; queryOffset < seqlen && ok;
         queryOffset += queryChunk) {
        int current = std::min(queryChunk, seqlen - queryOffset);
        // Keys to the right of the last query in this tile are causally
        // unreachable. Omitting them from both GEMMs cuts roughly half of the
        // dense fallback work for a fresh prefill while preserving the exact
        // selected-key softmax for every row.
        int keyCount = std::min(
            totalLen, startPos + queryOffset + current);
        const __nv_bfloat16 *qChunk =
            qData + (uint64_t)queryOffset * q.strides[1];
        cublasStatus_t status = cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            keyCount, current, kAttentionQkDim, &alpha,
            kData, CUDA_R_16BF, (int)k.strides[1],
            (long long)k.strides[0],
            qChunk, CUDA_R_16BF, (int)q.strides[1],
            (long long)q.strides[0], &beta,
            scores, CUDA_R_16BF, keyCount,
            (long long)current * keyCount, kAttentionHeads,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (status != CUBLAS_STATUS_SUCCESS) {
            ok = false;
            break;
        }

        if (indices.dataType == fastllm::DataType::INT16) {
            SparseAttentionPrefillSoftmaxKernel<int16_t><<<
                current * kAttentionHeads, 256, 0,
                cudaStreamPerThread>>>(
                    scores, (const int16_t *)indices.cudaData,
                    indices.strides[0], queryOffset, current,
                    keyCount, totalLen, startPos, scale);
        } else {
            SparseAttentionPrefillSoftmaxKernel<int32_t><<<
                current * kAttentionHeads, 256, 0,
                cudaStreamPerThread>>>(
                    scores, (const int32_t *)indices.cudaData,
                    indices.strides[0], queryOffset, current,
                    keyCount, totalLen, startPos, scale);
        }
        if (cudaGetLastError() != cudaSuccess) {
            ok = false;
            break;
        }

        __nv_bfloat16 *outputChunk =
            outputData + (uint64_t)queryOffset * output.strides[1];
        status = cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            kAttentionVDim, current, keyCount, &alpha,
            vData, CUDA_R_16BF, (int)v.strides[1],
            (long long)v.strides[0],
            scores, CUDA_R_16BF, keyCount,
            (long long)current * keyCount, &beta,
            outputChunk, CUDA_R_16BF, kAttentionVDim,
            (long long)output.strides[0], kAttentionHeads,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        ok = status == CUBLAS_STATUS_SUCCESS;
    }
    cudaError_t launchState = cudaGetLastError();
    DeviceSync();
    if (ownsScores) {
        FastllmCudaFree(scores);
    }
    return ok && launchState == cudaSuccess;
}

bool FastllmCudaDots3NoteSlidingAttentionPrefill(
        const fastllm::Data &q, const fastllm::Data &k,
        const fastllm::Data &v, int startPos, int windowSize,
        float scale, fastllm::Data &output) {
    if (startPos < 0 || windowSize <= 0 || windowSize > 4096 ||
        q.dataDevice != fastllm::DataDevice::CUDA ||
        q.dataType != fastllm::DataType::BFLOAT16 ||
        q.cudaData == nullptr || q.dims.size() != 3 ||
        q.dims[0] <= 0 || q.dims[1] <= 0 || q.dims[2] <= 0 ||
        q.strides[2] != 1 ||
        k.dataDevice != fastllm::DataDevice::CUDA ||
        k.dataType != fastllm::DataType::BFLOAT16 ||
        k.cudaData == nullptr || k.dims.size() != 3 ||
        k.dims[0] != q.dims[0] || k.dims[2] != q.dims[2] ||
        k.strides[2] != 1 ||
        v.dataDevice != fastllm::DataDevice::CUDA ||
        v.dataType != fastllm::DataType::BFLOAT16 ||
        v.cudaData == nullptr || v.dims.size() != 3 ||
        v.dims[0] != q.dims[0] || v.dims[1] != k.dims[1] ||
        v.dims[2] <= 0 || v.strides[2] != 1 ||
        startPos + q.dims[1] > k.dims[1]) {
        return false;
    }
    int device = GetPointerDeviceId(q.cudaData);
    if (device < 0 || GetPointerDeviceId(k.cudaData) != device ||
        GetPointerDeviceId(v.cudaData) != device) {
        return false;
    }
    FastllmCudaSetDevice(device);
    int heads = q.dims[0];
    int seqlen = q.dims[1];
    int qkDim = q.dims[2];
    int totalLen = k.dims[1];
    int valueDim = v.dims[2];
    if (!PrepareOutput(output, fastllm::DataType::BFLOAT16,
                       {heads, seqlen, valueDim})) {
        return false;
    }

    int queryChunk = std::min(seqlen, 1024);
    size_t scratchBytes = 0;
    while (queryChunk > 0) {
        int maxKeyCount = std::min(
            totalLen, windowSize + queryChunk - 1);
        scratchBytes = (size_t)heads * queryChunk * maxKeyCount *
                       sizeof(__nv_bfloat16);
        if (scratchBytes <= kSlidingPrefillScratchLimit ||
            queryChunk == 1) {
            break;
        }
        queryChunk = (queryChunk + 1) / 2;
    }
    if (queryChunk <= 0 ||
        scratchBytes > kSlidingPrefillScratchLimit) {
        return false;
    }
    __nv_bfloat16 *scores =
        (__nv_bfloat16 *)FastllmCudaMalloc(scratchBytes);
    if (scores == nullptr) {
        return false;
    }

    const __nv_bfloat16 *qData =
        (const __nv_bfloat16 *)q.cudaData;
    const __nv_bfloat16 *kData =
        (const __nv_bfloat16 *)k.cudaData;
    const __nv_bfloat16 *vData =
        (const __nv_bfloat16 *)v.cudaData;
    __nv_bfloat16 *outputData =
        (__nv_bfloat16 *)output.cudaData;
    cublasHandle_t handle = getFastllmCublasHandle();
    const float alpha = 1.0f;
    const float beta = 0.0f;
    bool ok = true;
    for (int queryOffset = 0; queryOffset < seqlen && ok;
         queryOffset += queryChunk) {
        int current = std::min(queryChunk, seqlen - queryOffset);
        int firstQueryPosition = startPos + queryOffset;
        int keyStart = std::max(
            0, firstQueryPosition - windowSize + 1);
        int keyEnd = std::min(
            totalLen, startPos + queryOffset + current);
        int keyCount = keyEnd - keyStart;
        if (keyCount <= 0) {
            ok = false;
            break;
        }

        const __nv_bfloat16 *qChunk =
            qData + (uint64_t)queryOffset * q.strides[1];
        const __nv_bfloat16 *kChunk =
            kData + (uint64_t)keyStart * k.strides[1];
        cublasStatus_t status = cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            keyCount, current, qkDim, &alpha,
            kChunk, CUDA_R_16BF, (int)k.strides[1],
            (long long)k.strides[0],
            qChunk, CUDA_R_16BF, (int)q.strides[1],
            (long long)q.strides[0], &beta,
            scores, CUDA_R_16BF, keyCount,
            (long long)current * keyCount, heads,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (status != CUBLAS_STATUS_SUCCESS) {
            ok = false;
            break;
        }

        SlidingAttentionPrefillSoftmaxKernel<<<
            current * heads, 256,
            (size_t)windowSize * sizeof(float),
            cudaStreamPerThread>>>(
                scores, queryOffset, current, keyStart, keyCount,
                startPos, windowSize, scale);
        if (cudaGetLastError() != cudaSuccess) {
            ok = false;
            break;
        }

        const __nv_bfloat16 *vChunk =
            vData + (uint64_t)keyStart * v.strides[1];
        __nv_bfloat16 *outputChunk =
            outputData + (uint64_t)queryOffset * output.strides[1];
        status = cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            valueDim, current, keyCount, &alpha,
            vChunk, CUDA_R_16BF, (int)v.strides[1],
            (long long)v.strides[0],
            scores, CUDA_R_16BF, keyCount,
            (long long)current * keyCount, &beta,
            outputChunk, CUDA_R_16BF, valueDim,
            (long long)output.strides[0], heads,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        ok = status == CUBLAS_STATUS_SUCCESS;
    }
    cudaError_t launchState = cudaGetLastError();
    DeviceSync();
    FastllmCudaFree(scores);
    return ok && launchState == cudaSuccess;
}
