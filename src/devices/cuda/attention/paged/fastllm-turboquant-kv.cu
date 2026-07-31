//
// TurboQuant-derived packed paged KV cache primitives for FastLLM.
//
// Source algorithm: llama.cpp-turboquant (MIT), GGML_TYPE_TURBO3_0.
// This file implements only the Qwen3.5/3.6 SM70 closure used here:
//   K: q8_0 (fp16 scale + 32 int8 values per 32-value block)
//   V: Turbo3 (fp16 corrected norm + 3-bit indices per 128-value block)
// Every (page, token, head) owns an independent packed row.
//
#include "fastllm-cuda.cuh"
#include "fastllm.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace {
constexpr int kHeadDim = 256;
constexpr int kQ8BlockValues = 32;
constexpr int kQ8BlockBytes = 34;
constexpr int kTurbo3BlockValues = 128;
constexpr int kTurbo3BlockBytes = 50;
constexpr int kQ8RowBytes = (kHeadDim / kQ8BlockValues) * kQ8BlockBytes;
constexpr int kTurbo3RowBytes = (kHeadDim / kTurbo3BlockValues) * kTurbo3BlockBytes;
static_assert(kQ8RowBytes == 272 && kTurbo3RowBytes == 100,
              "Qwen3.5 TurboQuant packed KV row bytes changed");

struct Q8KvBlock {
    uint16_t scale;
    int8_t values[kQ8BlockValues];
};
static_assert(sizeof(Q8KvBlock) == kQ8BlockBytes, "Q8 KV block layout changed");

struct Turbo3KvBlock {
    uint16_t norm;
    uint8_t low2[kTurbo3BlockValues / 4];
    uint8_t high1[kTurbo3BlockValues / 8];
};
static_assert(sizeof(Turbo3KvBlock) == kTurbo3BlockBytes,
              "Turbo3 KV must remain 50 bytes per 128 values");

__device__ __constant__ float kTurbo3Centroids[8] = {
    -0.190207f, -0.118786f, -0.066822f, -0.021663f,
     0.021663f,  0.066822f,  0.118786f,  0.190207f
};
__device__ __constant__ float kTurbo3Midpoints[7] = {
    -0.154496f, -0.092804f, -0.044243f, 0.0f,
     0.044243f,  0.092804f,  0.154496f
};
__device__ __constant__ float kWhtSigns1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,
    1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,
    1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,
    1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,
    1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};
__device__ __constant__ float kWhtSigns2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,
    1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,
    -1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,
    -1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};

template <typename T>
__device__ __forceinline__ float ToFloat(T value) { return static_cast<float>(value); }
template <>
__device__ __forceinline__ float ToFloat<float>(float value) { return value; }

template <typename T>
__device__ __forceinline__ float LoadSource(const T *src, size_t index) {
    return ToFloat(src[index]);
}

__device__ __forceinline__ uint16_t FloatToHalfBits(float value) {
    union { half h; uint16_t bits; } packed;
    packed.h = __float2half_rn(value);
    return packed.bits;
}
__device__ __forceinline__ float HalfBitsToFloat(uint16_t bits) {
    union { half h; uint16_t bits; } packed;
    packed.bits = bits;
    return __half2float(packed.h);
}

__device__ __forceinline__ uint8_t NearestTurbo3(float value) {
    if (value < kTurbo3Midpoints[0]) return 0;
    if (value < kTurbo3Midpoints[1]) return 1;
    if (value < kTurbo3Midpoints[2]) return 2;
    if (value < kTurbo3Midpoints[3]) return 3;
    if (value < kTurbo3Midpoints[4]) return 4;
    if (value < kTurbo3Midpoints[5]) return 5;
    if (value < kTurbo3Midpoints[6]) return 6;
    return 7;
}

__device__ __forceinline__ void WhtStage(float *x, int lane, int h) {
    if ((lane % (2 * h)) < h) {
        float a = x[lane];
        float b = x[lane + h];
        x[lane] = a + b;
        x[lane + h] = a - b;
    }
    __syncthreads();
}

__device__ __forceinline__ void ForwardWht128(float *x, int lane) {
    x[lane] *= kWhtSigns1[lane];
    __syncthreads();
    WhtStage(x, lane, 1); WhtStage(x, lane, 2); WhtStage(x, lane, 4);
    WhtStage(x, lane, 8); WhtStage(x, lane, 16); WhtStage(x, lane, 32);
    WhtStage(x, lane, 64);
    x[lane] *= 0.08838834764831845f * kWhtSigns2[lane];
    __syncthreads();
}

__device__ __forceinline__ void InverseWht128(float *x, int lane) {
    x[lane] *= kWhtSigns2[lane];
    __syncthreads();
    WhtStage(x, lane, 1); WhtStage(x, lane, 2); WhtStage(x, lane, 4);
    WhtStage(x, lane, 8); WhtStage(x, lane, 16); WhtStage(x, lane, 32);
    WhtStage(x, lane, 64);
    x[lane] *= 0.08838834764831845f * kWhtSigns1[lane];
    __syncthreads();
}

template <typename SrcT>
__global__ void QuantizeQ8RowsKernel(
        const SrcT *src, uint8_t *dst, const int32_t *pageIds,
        const int32_t *pageOffsets, int rows, int sourceSeqLen,
        int pageLen, int numHeads, int headDim, bool batchLayout,
        int sourceTokenOffset) {
    int row = blockIdx.x;
    int block = blockIdx.y;
    int lane = threadIdx.x;
    if (row >= rows || block >= headDim / kQ8BlockValues || lane >= 32) return;
    int copiedTokens = rows / numHeads;
    int token = batchLayout ? row / numHeads : row % copiedTokens;
    int head = batchLayout ? row % numHeads : row / copiedTokens;
    int page = pageIds[batchLayout ? token : 0];
    int pageOffset = pageOffsets[batchLayout ? token : 0] +
                     (batchLayout ? 0 : token);
    if (pageOffset < 0 || pageOffset >= pageLen) return;
    size_t sourceRow = batchLayout
        ? (size_t)token * numHeads + head
        : (size_t)head * sourceSeqLen + sourceTokenOffset + token;
    float value = LoadSource(src, sourceRow * headDim + block * 32 + lane);
    float maxAbs = fabsf(value);
    for (int offset = 16; offset > 0; offset >>= 1)
        maxAbs = fmaxf(maxAbs, __shfl_down_sync(0xffffffff, maxAbs, offset));
    maxAbs = __shfl_sync(0xffffffff, maxAbs, 0);
    float scale = maxAbs > 0.0f ? maxAbs / 127.0f : 0.0f;
    int q = scale > 0.0f ? __float2int_rn(value / scale) : 0;
    q = max(-127, min(127, q));
    constexpr size_t rowBytes = kQ8RowBytes;
    size_t dstRow = ((size_t)page * pageLen + pageOffset) * numHeads + head;
    Q8KvBlock *out = reinterpret_cast<Q8KvBlock *>(dst + dstRow * rowBytes) + block;
    if (lane == 0) out->scale = FloatToHalfBits(scale);
    out->values[lane] = static_cast<int8_t>(q);
}

template <typename SrcT>
__global__ void QuantizeTurbo3RowsKernel(
        const SrcT *src, uint8_t *dst, const int32_t *pageIds,
        const int32_t *pageOffsets, int rows, int sourceSeqLen,
        int pageLen, int numHeads, int headDim, bool batchLayout,
        int sourceTokenOffset) {
    int row = blockIdx.x;
    int group = blockIdx.y;
    int lane = threadIdx.x;
    if (row >= rows || group >= headDim / 128 || lane >= 128) return;
    int copiedTokens = rows / numHeads;
    int token = batchLayout ? row / numHeads : row % copiedTokens;
    int head = batchLayout ? row % numHeads : row / copiedTokens;
    int page = pageIds[batchLayout ? token : 0];
    int pageOffset = pageOffsets[batchLayout ? token : 0] +
                     (batchLayout ? 0 : token);
    if (pageOffset < 0 || pageOffset >= pageLen) return;
    size_t sourceRow = batchLayout
        ? (size_t)token * numHeads + head
        : (size_t)head * sourceSeqLen + sourceTokenOffset + token;
    __shared__ float x[128];
    __shared__ float warpSums[4];
    x[lane] = LoadSource(src, sourceRow * headDim + group * 128 + lane);
    __syncthreads();
    float sq = x[lane] * x[lane];
    for (int off = 16; off > 0; off >>= 1)
        sq += __shfl_down_sync(0xffffffff, sq, off);
    if ((lane & 31) == 0) warpSums[lane >> 5] = sq;
    __syncthreads();
    float normSq = warpSums[0] + warpSums[1] + warpSums[2] + warpSums[3];
    float norm = sqrtf(normSq);
    x[lane] = norm > 1.0e-10f ? x[lane] / norm : 0.0f;
    __syncthreads();
    ForwardWht128(x, lane);
    uint8_t index = NearestTurbo3(x[lane]);
    float reconSq = kTurbo3Centroids[index] * kTurbo3Centroids[index];
    for (int off = 16; off > 0; off >>= 1)
        reconSq += __shfl_down_sync(0xffffffff, reconSq, off);
    if ((lane & 31) == 0) warpSums[lane >> 5] = reconSq;
    __syncthreads();
    float totalRecon = warpSums[0] + warpSums[1] + warpSums[2] + warpSums[3];
    float correctedNorm = totalRecon > 1.0e-10f ? norm / sqrtf(totalRecon) : norm;
    constexpr size_t rowBytes = kTurbo3RowBytes;
    size_t dstRow = ((size_t)page * pageLen + pageOffset) * numHeads + head;
    Turbo3KvBlock *out = reinterpret_cast<Turbo3KvBlock *>(dst + dstRow * rowBytes) + group;
    if (lane == 0) out->norm = FloatToHalfBits(correctedNorm);
    unsigned warpMask = 0xffffffffu;
    uint8_t low = index & 3;
    uint8_t packedLow = 0;
    int warpLane = lane & 31;
    #pragma unroll
    for (int i = 0; i < 4; i++)
        packedLow |= static_cast<uint8_t>(__shfl_sync(warpMask, low, (warpLane & ~3) + i)) << (2 * i);
    if ((warpLane & 3) == 0) out->low2[lane / 4] = packedLow;
    unsigned high = __ballot_sync(warpMask, (index & 4) != 0);
    if ((warpLane & 7) == 0)
        out->high1[lane / 8] = static_cast<uint8_t>((high >> ((warpLane / 8) * 8)) & 0xff);
}

__global__ void GatherQ8HeadRangeKernel(
        const uint8_t *src, const int32_t *pageIndices, int kvStart,
        int chunkLen, int pageLen, int numHeads, int headDim, int head,
        half *out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = chunkLen * headDim;
    if (idx >= total) return;
    int tokenInChunk = idx / headDim;
    int dim = idx % headDim;
    int logicalToken = kvStart + tokenInChunk;
    int pageList = logicalToken / pageLen;
    int pageOffset = logicalToken % pageLen;
    int page = pageIndices[pageList];
    constexpr size_t rowBytes = kQ8RowBytes;
    size_t row = ((size_t)page * pageLen + pageOffset) * numHeads + head;
    const Q8KvBlock *block = reinterpret_cast<const Q8KvBlock *>(src + row * rowBytes) + dim / 32;
    float scale = HalfBitsToFloat(block->scale);
    out[idx] = __float2half_rn(scale * static_cast<float>(block->values[dim & 31]));
}

__global__ void GatherTurbo3HeadRangeKernel(
        const uint8_t *src, const int32_t *pageIndices, int kvStart,
        int chunkLen, int pageLen, int numHeads, int headDim, int head,
        half *out) {
    int tokenInChunk = blockIdx.x;
    int group = blockIdx.y;
    int lane = threadIdx.x;
    if (tokenInChunk >= chunkLen || lane >= 128) return;
    int logicalToken = kvStart + tokenInChunk;
    int pageList = logicalToken / pageLen;
    int pageOffset = logicalToken % pageLen;
    int page = pageIndices[pageList];
    constexpr size_t rowBytes = kTurbo3RowBytes;
    size_t row = ((size_t)page * pageLen + pageOffset) * numHeads + head;
    const Turbo3KvBlock *block = reinterpret_cast<const Turbo3KvBlock *>(src + row * rowBytes) + group;
    uint8_t index = static_cast<uint8_t>((block->low2[lane / 4] >> (2 * (lane & 3))) & 3);
    index |= static_cast<uint8_t>(((block->high1[lane / 8] >> (lane & 7)) & 1) << 2);
    __shared__ float x[128];
    x[lane] = kTurbo3Centroids[index] * HalfBitsToFloat(block->norm);
    __syncthreads();
    InverseWht128(x, lane);
    out[(size_t)tokenInChunk * headDim + group * 128 + lane] = __float2half_rn(x[lane]);
}

template <typename SrcT>
bool LaunchCopy(uint8_t *pagedData, int pageIdx, int pageLen, int numHeads,
                int headDim, fastllm::DataType dstType, const SrcT *inputData,
                int seqLen, int inputOffset, int copyLen, int pageOffset) {
    if (headDim != kHeadDim || copyLen <= 0 || inputOffset < 0 ||
        pageOffset < 0 || pageOffset + copyLen > pageLen) return false;
    int32_t *meta = static_cast<int32_t *>(FastllmCudaMalloc(2 * sizeof(int32_t)));
    if (meta == nullptr) return false;
    int32_t host[2] = {pageIdx, pageOffset};
    cudaError_t error = cudaMemcpyAsync(meta, host, sizeof(host), cudaMemcpyHostToDevice,
                                        cudaStreamPerThread);
    if (error != cudaSuccess) { FastllmCudaFree(meta); cudaGetLastError(); return false; }
    int rows = numHeads * copyLen;
    if (dstType == fastllm::DataType::Q8_0_KV) {
        dim3 grid(rows, headDim / 32, 1);
        QuantizeQ8RowsKernel<SrcT><<<grid, 32, 0, cudaStreamPerThread>>>(
            inputData, pagedData, meta, meta + 1, rows, seqLen, pageLen,
            numHeads, headDim, false, inputOffset);
    } else if (dstType == fastllm::DataType::TURBO3_KV) {
        dim3 grid(rows, headDim / 128, 1);
        QuantizeTurbo3RowsKernel<SrcT><<<grid, 128, 0, cudaStreamPerThread>>>(
            inputData, pagedData, meta, meta + 1, rows, seqLen, pageLen,
            numHeads, headDim, false, inputOffset);
    } else {
        FastllmCudaFree(meta); return false;
    }
    error = cudaGetLastError();
    cudaError_t syncError = cudaStreamSynchronize(cudaStreamPerThread);
    FastllmCudaFree(meta);
    return error == cudaSuccess && syncError == cudaSuccess;
}

template <typename SrcT>
bool LaunchCopyBatch(uint8_t *pagedData, int32_t *pageIds,
                     int32_t *pageOffsets, int pageLen, int batch,
                     int numHeads, int headDim, fastllm::DataType dstType,
                     const SrcT *inputData, bool sync) {
    if (headDim != kHeadDim || batch <= 0 || pageIds == nullptr || pageOffsets == nullptr) return false;
    int rows = batch * numHeads;
    if (dstType == fastllm::DataType::Q8_0_KV) {
        dim3 grid(rows, headDim / 32, 1);
        QuantizeQ8RowsKernel<SrcT><<<grid, 32, 0, cudaStreamPerThread>>>(
            inputData, pagedData, pageIds, pageOffsets, rows, 1,
            pageLen, numHeads, headDim, true, 0);
    } else if (dstType == fastllm::DataType::TURBO3_KV) {
        dim3 grid(rows, headDim / 128, 1);
        QuantizeTurbo3RowsKernel<SrcT><<<grid, 128, 0, cudaStreamPerThread>>>(
            inputData, pagedData, pageIds, pageOffsets, rows, 1,
            pageLen, numHeads, headDim, true, 0);
    } else {
        return false;
    }
    cudaError_t error = cudaGetLastError();
    if (sync && error == cudaSuccess) error = cudaStreamSynchronize(cudaStreamPerThread);
    return error == cudaSuccess;
}

} // namespace

bool FastllmCudaPackedKVCacheCopy(
        uint8_t *pagedData, int pageIdx, int pageLen, int numHeads,
        int headDim, fastllm::DataType dstType, uint8_t *inputData,
        fastllm::DataType srcType, int seqLen, int inputOffset,
        int copyLen, int pageOffset) {
    if (!fastllm::IsPackedKVCacheDataType(dstType) || pagedData == nullptr || inputData == nullptr)
        return false;
    if (srcType == fastllm::DataType::FLOAT16)
        return LaunchCopy(pagedData, pageIdx, pageLen, numHeads, headDim, dstType,
                          reinterpret_cast<const half *>(inputData), seqLen,
                          inputOffset, copyLen, pageOffset);
    if (srcType == fastllm::DataType::BFLOAT16)
        return LaunchCopy(pagedData, pageIdx, pageLen, numHeads, headDim, dstType,
                          reinterpret_cast<const __nv_bfloat16 *>(inputData), seqLen,
                          inputOffset, copyLen, pageOffset);
    if (srcType == fastllm::DataType::FLOAT32)
        return LaunchCopy(pagedData, pageIdx, pageLen, numHeads, headDim, dstType,
                          reinterpret_cast<const float *>(inputData), seqLen,
                          inputOffset, copyLen, pageOffset);
    return false;
}

bool FastllmCudaPackedKVCacheCopyBatch(
        uint8_t *pagedData, int32_t *pageIds, int32_t *pageOffsets,
        int pageLen, int batch, int numHeads, int headDim,
        fastllm::DataType dstType, uint8_t *inputData,
        fastllm::DataType srcType, bool sync) {
    if (!fastllm::IsPackedKVCacheDataType(dstType) || pagedData == nullptr || inputData == nullptr)
        return false;
    if (srcType == fastllm::DataType::FLOAT16)
        return LaunchCopyBatch(pagedData, pageIds, pageOffsets, pageLen, batch,
                               numHeads, headDim, dstType,
                               reinterpret_cast<const half *>(inputData), sync);
    if (srcType == fastllm::DataType::BFLOAT16)
        return LaunchCopyBatch(pagedData, pageIds, pageOffsets, pageLen, batch,
                               numHeads, headDim, dstType,
                               reinterpret_cast<const __nv_bfloat16 *>(inputData), sync);
    if (srcType == fastllm::DataType::FLOAT32)
        return LaunchCopyBatch(pagedData, pageIds, pageOffsets, pageLen, batch,
                               numHeads, headDim, dstType,
                               reinterpret_cast<const float *>(inputData), sync);
    return false;
}

bool FastllmCudaPackedKVCacheGatherHeadRangeToHalf(
        const uint8_t *pagedData, fastllm::DataType srcType,
        const int32_t *pageIndices, int kvStart, int chunkLen,
        int pageLen, int numHeads, int headDim, int head, void *output) {
    if (pagedData == nullptr || pageIndices == nullptr || output == nullptr ||
        headDim != kHeadDim || chunkLen <= 0 || head < 0 || head >= numHeads)
        return false;
    if (srcType == fastllm::DataType::Q8_0_KV) {
        int total = chunkLen * headDim;
        GatherQ8HeadRangeKernel<<<(total + 255) / 256, 256, 0, cudaStreamPerThread>>>(
            pagedData, pageIndices, kvStart, chunkLen, pageLen,
            numHeads, headDim, head, reinterpret_cast<half *>(output));
    } else if (srcType == fastllm::DataType::TURBO3_KV) {
        dim3 grid(chunkLen, headDim / 128, 1);
        GatherTurbo3HeadRangeKernel<<<grid, 128, 0, cudaStreamPerThread>>>(
            pagedData, pageIndices, kvStart, chunkLen, pageLen,
            numHeads, headDim, head, reinterpret_cast<half *>(output));
    } else {
        return false;
    }
    return cudaGetLastError() == cudaSuccess;
}
