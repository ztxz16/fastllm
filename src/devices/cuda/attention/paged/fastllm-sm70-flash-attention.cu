// Copyright (c) 2024, flash-attention-v100 contributors.
// Copyright (c) 2025, FastLLM contributors.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE.
//
// The tiled causal/paged online-softmax structure is adapted from the local
// flash-attention-v100 paged forward kernel. This specialization targets the
// short-query FP8-KV validation shape used by Qwen3.5/3.6 MTP on Volta.

#include "devices/cuda/attention/fastllm-paged-attention-native.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

using namespace nvcuda::wmma;

constexpr int kHeadDim = 256;
constexpr int kTile = 16;
constexpr int kThreads = 512;
constexpr int kQHeads = 24;
constexpr int kKvHeads = 4;
constexpr int kGroup = 6;
constexpr int kPageLen = 128;

struct alignas(128) Sm70FlashPrefillSmem {
    half q[kTile * kHeadDim];
    half k[kTile * kHeadDim];
    half v[kTile * kHeadDim];
    float scores[kTile * kTile];
    half probabilities[kTile * kTile];
    float output[kTile * kHeadDim];
    float rowMax[kTile];
    float rowSum[kTile];
    float rescale[kTile];
};

__device__ __forceinline__ half LoadFp8Half(const __nv_fp8_e4m3 *base,
                                             int64_t offset) {
    __nv_fp8_e4m3 value;
    value.__x = __ldg(reinterpret_cast<const unsigned char *>(base) + offset);
    return __float2half_rn(static_cast<float>(value));
}

__global__ void Sm70FlashAttentionFp8PrefillKernel(
    const half *__restrict__ q,
    const __nv_fp8_e4m3 *__restrict__ kCache,
    const __nv_fp8_e4m3 *__restrict__ vCache,
    half *__restrict__ output,
    const int32_t *__restrict__ qSizes,
    const int32_t *__restrict__ pageSizes,
    const int32_t *__restrict__ pageIndices,
    const int32_t *__restrict__ lastPageLens,
    int qStrideHead,
    int qStrideToken,
    float softmaxScale) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
    extern __shared__ char rawSmem[];
    Sm70FlashPrefillSmem &smem =
        *reinterpret_cast<Sm70FlashPrefillSmem *>(rawSmem);
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int request = blockIdx.x;
    const int qHead = blockIdx.y;
    const int kvHead = qHead / kGroup;
    const int qBegin = qSizes[request];
    const int qLen = qSizes[request + 1] - qBegin;
    const int pageBegin = pageSizes[request];
    const int pageCount = pageSizes[request + 1] - pageBegin;
    const int kvLen = pageCount > 0
        ? (pageCount - 1) * kPageLen + lastPageLens[request]
        : 0;

    for (int i = tid; i < kTile * kHeadDim; i += kThreads) {
        const int row = i / kHeadDim;
        const int dim = i - row * kHeadDim;
        smem.q[i] = row < qLen
            ? q[(int64_t)qHead * qStrideHead
                + (int64_t)(qBegin + row) * qStrideToken + dim]
            : __float2half(0.0f);
        smem.output[i] = 0.0f;
    }
    if (tid < kTile) {
        smem.rowMax[tid] = -1.0e30f;
        smem.rowSum[tid] = 0.0f;
        smem.rescale[tid] = 0.0f;
    }
    __syncthreads();

    const int causalOffset = kvLen - qLen;
    for (int tileStart = 0; tileStart < kvLen; tileStart += kTile) {
        const int tileRows = min(kTile, kvLen - tileStart);
        for (int i = tid; i < kTile * kHeadDim; i += kThreads) {
            const int tokenInTile = i / kHeadDim;
            const int dim = i - tokenInTile * kHeadDim;
            half kValue = __float2half(0.0f);
            half vValue = __float2half(0.0f);
            if (tokenInTile < tileRows) {
                const int logicalToken = tileStart + tokenInTile;
                const int logicalPage = logicalToken / kPageLen;
                const int pageOffset = logicalToken - logicalPage * kPageLen;
                const int physicalPage = pageIndices[pageBegin + logicalPage];
                const int64_t cacheOffset =
                    (((int64_t)physicalPage * kPageLen + pageOffset) * kKvHeads
                        + kvHead) * kHeadDim + dim;
                kValue = LoadFp8Half(kCache, cacheOffset);
                vValue = LoadFp8Half(vCache, cacheOffset);
            }
            smem.k[i] = kValue;
            smem.v[i] = vValue;
        }
        __syncthreads();

        if (warp == 0) {
            fragment<matrix_a, kTile, kTile, kTile, half, row_major> qFrag;
            fragment<matrix_b, kTile, kTile, kTile, half, col_major> kFrag;
            fragment<accumulator, kTile, kTile, kTile, float> scoreFrag;
            fill_fragment(scoreFrag, 0.0f);
#pragma unroll
            for (int dim = 0; dim < kHeadDim; dim += kTile) {
                load_matrix_sync(qFrag, smem.q + dim, kHeadDim);
                load_matrix_sync(kFrag, smem.k + dim, kHeadDim);
                mma_sync(scoreFrag, qFrag, kFrag, scoreFrag);
            }
            store_matrix_sync(smem.scores, scoreFrag, kTile, mem_row_major);
        }
        __syncthreads();

        if (tid < qLen) {
            float tileMax = -1.0e30f;
#pragma unroll
            for (int col = 0; col < kTile; ++col) {
                const int keyPosition = tileStart + col;
                if (col < tileRows && keyPosition <= causalOffset + tid) {
                    tileMax = fmaxf(tileMax,
                                    smem.scores[tid * kTile + col]
                                        * softmaxScale);
                }
            }
            const float oldMax = smem.rowMax[tid];
            const float newMax = fmaxf(oldMax, tileMax);
            const float rescale = __expf(oldMax - newMax);
            float tileSum = 0.0f;
#pragma unroll
            for (int col = 0; col < kTile; ++col) {
                const int keyPosition = tileStart + col;
                float probability = 0.0f;
                if (col < tileRows && keyPosition <= causalOffset + tid) {
                    probability = __expf(
                        smem.scores[tid * kTile + col] * softmaxScale - newMax);
                }
                smem.probabilities[tid * kTile + col] =
                    __float2half_rn(probability);
                tileSum += probability;
            }
            smem.rowMax[tid] = newMax;
            smem.rowSum[tid] = smem.rowSum[tid] * rescale + tileSum;
            smem.rescale[tid] = rescale;
        }
        __syncthreads();

        for (int i = tid; i < qLen * kHeadDim; i += kThreads) {
            const int row = i / kHeadDim;
            smem.output[i] *= smem.rescale[row];
        }
        __syncthreads();

        if (warp < kHeadDim / kTile) {
            fragment<matrix_a, kTile, kTile, kTile, half, row_major> pFrag;
            fragment<matrix_b, kTile, kTile, kTile, half, row_major> vFrag;
            fragment<accumulator, kTile, kTile, kTile, float> outFrag;
            const int dim = warp * kTile;
            load_matrix_sync(pFrag, smem.probabilities, kTile);
            load_matrix_sync(vFrag, smem.v + dim, kHeadDim);
            load_matrix_sync(outFrag, smem.output + dim, kHeadDim, mem_row_major);
            mma_sync(outFrag, pFrag, vFrag, outFrag);
            store_matrix_sync(smem.output + dim, outFrag, kHeadDim, mem_row_major);
        }
        __syncthreads();
    }

    for (int i = tid; i < qLen * kHeadDim; i += kThreads) {
        const int row = i / kHeadDim;
        const int dim = i - row * kHeadDim;
        const float invSum = 1.0f / fmaxf(smem.rowSum[row], 1.0e-24f);
        output[((int64_t)(qBegin + row) * kQHeads + qHead) * kHeadDim
               + dim] = __float2half_rn(smem.output[i] * invSum);
    }
#endif
}

bool EnvEnabled(const char *name) {
    const char *value = std::getenv(name);
    return value != nullptr && std::strcmp(value, "0") != 0;
}

int MaxKvTokens() {
    const char *value = std::getenv("FASTLLM_CUDA_SM70_FLASH_ATTN_MAX_KV");
    if (value == nullptr) {
        return 512;
    }
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    return end != value && *end == '\0' && parsed > 0 && parsed <= INT_MAX
        ? static_cast<int>(parsed)
        : 512;
}

bool IsCudaInt32Vector(const fastllm::Data &data) {
    return data.dataType == fastllm::DataType::INT32
        && data.dataDevice == fastllm::DataDevice::CUDA
        && data.dims.size() == 1 && data.cudaData != nullptr;
}

} // namespace

bool FastllmCudaTrySm70FlashAttentionPrefill(
    fastllm::Data &q, fastllm::Data &kCaches, fastllm::Data &vCaches,
    fastllm::Data &qSizes, fastllm::Data &pageSizes,
    fastllm::Data &pageIndexs, fastllm::Data &lastPageLens,
    fastllm::Data &output, int group, float scale, int attentionType) {
    if (!EnvEnabled("FASTLLM_CUDA_SM70_FLASH_ATTN")
        || attentionType != 1 || group != kGroup
        || q.dataType != fastllm::DataType::FLOAT16
        || output.dataType != fastllm::DataType::FLOAT16
        || q.dataDevice != fastllm::DataDevice::CUDA
        || output.dataDevice != fastllm::DataDevice::CUDA
        || q.cudaData == nullptr || output.cudaData == nullptr
        || q.dims.size() != 3 || output.dims != q.dims
        || q.dims[0] != kQHeads || q.dims[1] < 2 || q.dims[1] > 50
        || q.dims[2] != kHeadDim
        || q.strides.size() != 3 || output.strides.size() != 3
        || q.strides[2] != 1 || output.strides != q.strides
        || qSizes.dims.size() != 1 || qSizes.dims[0] < 2
        || qSizes.dims[0] > 6
        || pageSizes.dims.size() != 1
        || pageSizes.dims[0] != qSizes.dims[0]
        || lastPageLens.dims.size() != 1
        || lastPageLens.dims[0] != qSizes.dims[0] - 1
        || !IsCudaInt32Vector(qSizes) || !IsCudaInt32Vector(pageSizes)
        || !IsCudaInt32Vector(pageIndexs) || !IsCudaInt32Vector(lastPageLens)) {
        return false;
    }
    fastllm::PagedCacheManager *pagedK = kCaches.pagedKVCacheData;
    fastllm::PagedCacheManager *pagedV = vCaches.pagedKVCacheData;
    if (!kCaches.isPagedKVCache || !vCaches.isPagedKVCache
        || pagedK == nullptr || pagedV == nullptr
        || pagedK->dataType != fastllm::DataType::FP8_E4M3
        || pagedV->dataType != fastllm::DataType::FP8_E4M3
        || pagedK->dataDevice != fastllm::DataDevice::CUDA
        || pagedV->dataDevice != fastllm::DataDevice::CUDA
        || pagedK->cudaData == nullptr || pagedV->cudaData == nullptr
        || pagedK->dims.size() != 4 || pagedV->dims != pagedK->dims
        || pagedK->dims[1] != kPageLen || pagedK->dims[2] != kKvHeads
        || pagedK->dims[3] != kHeadDim) {
        return false;
    }

    const int batch = qSizes.dims[0] - 1;
    if (!std::isfinite(scale) || scale <= 0.0f
        || (int)qSizes.cpuIntDatas.size() != batch + 1
        || (int)pageSizes.cpuIntDatas.size() != batch + 1
        || (int)pageIndexs.cpuIntDatas.size() < pageIndexs.dims[0]
        || (int)lastPageLens.cpuIntDatas.size() != batch
        || qSizes.cpuIntDatas.front() != 0
        || pageSizes.cpuIntDatas.front() != 0
        || qSizes.cpuIntDatas.back() != q.dims[1]
        || pageSizes.cpuIntDatas.back() != pageIndexs.dims[0]) {
        return false;
    }
    for (int request = 0; request < batch; ++request) {
        const int qLen = qSizes.cpuIntDatas[request + 1]
            - qSizes.cpuIntDatas[request];
        const int pages = pageSizes.cpuIntDatas[request + 1]
            - pageSizes.cpuIntDatas[request];
        const int lastPageLen = lastPageLens.cpuIntDatas[request];
        const int kvLen = pages > 0 ? (pages - 1) * kPageLen + lastPageLen : 0;
        if (qLen < 2 || qLen > 10 || pages <= 0
            || lastPageLen <= 0 || lastPageLen > kPageLen
            || kvLen < qLen || kvLen > MaxKvTokens()) {
            return false;
        }
        for (int page = pageSizes.cpuIntDatas[request];
             page < pageSizes.cpuIntDatas[request + 1]; ++page) {
            const int physicalPage = pageIndexs.cpuIntDatas[page];
            if (physicalPage < 0 || physicalPage >= pagedK->dims[0]) {
                return false;
            }
        }
    }

    int device = -1;
    int major = 0;
    int minor = 0;
    if (cudaGetDevice(&device) != cudaSuccess
        || cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                  device) != cudaSuccess
        || cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                                  device) != cudaSuccess
        || major != 7 || minor != 0) {
        cudaGetLastError();
        return false;
    }
    cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(cudaStreamPerThread, &captureStatus) != cudaSuccess
        || captureStatus != cudaStreamCaptureStatusNone) {
        cudaGetLastError();
        return false;
    }

    const size_t smemBytes = sizeof(Sm70FlashPrefillSmem);
    if (cudaFuncSetAttribute(Sm70FlashAttentionFp8PrefillKernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smemBytes)) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    dim3 grid(batch, kQHeads, 1);
    Sm70FlashAttentionFp8PrefillKernel<<<grid, kThreads, smemBytes,
                                         cudaStreamPerThread>>>(
        reinterpret_cast<const half *>(q.cudaData),
        reinterpret_cast<const __nv_fp8_e4m3 *>(pagedK->cudaData),
        reinterpret_cast<const __nv_fp8_e4m3 *>(pagedV->cudaData),
        reinterpret_cast<half *>(output.cudaData),
        reinterpret_cast<const int32_t *>(qSizes.cudaData),
        reinterpret_cast<const int32_t *>(pageSizes.cudaData),
        reinterpret_cast<const int32_t *>(pageIndexs.cudaData),
        reinterpret_cast<const int32_t *>(lastPageLens.cudaData),
        static_cast<int>(q.strides[0]), static_cast<int>(q.strides[1]), scale);
    if (cudaGetLastError() != cudaSuccess) {
        return false;
    }
    output.Resize({q.dims[1], kQHeads, kHeadDim});
    static thread_local bool logged = false;
    if (!logged) {
        std::printf("[FastLLM] SM70 FlashAttention FP8 paged prefill enabled "
                    "(Volta WMMA, page128, batch1..5, qLen2..10, "
                    "KV<=%d, Q24/KV4 D256 GQA6).\n", MaxKvTokens());
        logged = true;
    }
    return true;
}
