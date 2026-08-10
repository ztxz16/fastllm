// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// FastLLM integration wrapper for FlashInfer's DeepSeek-V4 SM120 sparse MLA
// decode kernel.  This translation unit is compiled for SM120 only; callers
// retain the architecture-independent CUDA implementation as their fallback.

#include "fastllm-cuda.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>

#include <attention/sparse_mla_sm120/decode_dsv4_kernel.cuh>
#include <attention/sparse_mla_sm120/model/kv_cache_traits.cuh>
#include <deep_gemm/impls/sm120_fp8_mqa_logits.cuh>

namespace flashinfer::sparse_mla_sm120 {

#define FASTLLM_DSV4_CUDA_CHECK_BOOL(call)                                  \
  do {                                                                       \
    cudaError_t e = (call);                                                  \
    if (e != cudaSuccess) {                                                  \
      std::fprintf(stderr, "DeepSeekV4 SM120 sparse MLA CUDA %s:%d: %s\n",  \
                   __FILE__, __LINE__, cudaGetErrorString(e));               \
      return false;                                                          \
    }                                                                        \
  } while (0)

template <int NUM_HEADS, int TOPK, int PAGE_BLOCK_SIZE>
static bool LaunchDecodeDsv4Impl(
    const bf16 *q, const uint8_t *kvCache, const int32_t *indices,
    bf16 *midOut, float *midLse, const int *topkLength, bf16 *output,
    const float *attnSink, const uint8_t *extraKvCache,
    const int32_t *extraIndices, const int *extraTopkLength, int extraTopk,
    size_t strideExtraKvBlock, int numTokens, int numSplits, float smScale,
    size_t strideKvBlock, cudaStream_t stream) {
  using KV = KVCacheTraits<ModelType::DSV4>;
  constexpr int H_BLOCKS = (NUM_HEADS + HPB - 1) / HPB;
  constexpr int N_V_CHUNKS = KV::D_NOPE / KV::QUANT_TILE;
  constexpr int DYN_SMEM_BYTES =
      HPB * KV::D_ROPE * (int)sizeof(bf16) +
      HPB * KV::Q_NOPE_STRIDE +
      HPB * KV::NUM_SCALES * (int)sizeof(float) +
      DSV4_KV_BUF_COUNT * DSV4_BI * KV::KV_SMEM_STRIDE +
      DSV4_KV_BUF_COUNT * DSV4_BI * KV::SCALE_BYTES_PER_TOKEN +
      DSV4_KV_BUF_COUNT * DSV4_BI * KV::D_ROPE * (int)sizeof(bf16) +
      16 + 4 * (int)sizeof(uint64_t) +
      2 * DSV4_N_WARPS * HPB * (int)sizeof(float) +
      N_V_CHUNKS * HPB * (int)sizeof(float) +
      2 * HPB * (DSV4_BI + 16);

  auto kernel =
      sparse_mla_decode_dsv4_kernel<ModelType::DSV4, NUM_HEADS, TOPK,
                                    PAGE_BLOCK_SIZE>;
  FASTLLM_DSV4_CUDA_CHECK_BOOL(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, DYN_SMEM_BYTES));

  int smCount = 0;
  int device = 0;
  FASTLLM_DSV4_CUDA_CHECK_BOOL(cudaGetDevice(&device));
  FASTLLM_DSV4_CUDA_CHECK_BOOL(
      cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));
  int chunksPerBlock = 1;
  float bestGap = 2.0f;
  constexpr int CEIL_WAVES_MAX = 3;
  const int perTokenHead = numTokens * H_BLOCKS;
  for (int cpb = 1; cpb <= numSplits; ++cpb) {
    const int effectiveSplits = (numSplits + cpb - 1) / cpb;
    const int activeBlocks = perTokenHead * effectiveSplits;
    const int ceilWaves = (activeBlocks + smCount - 1) / smCount;
    if (ceilWaves > CEIL_WAVES_MAX) {
      continue;
    }
    const float waves = (float)activeBlocks / (float)smCount;
    const float gap = (float)ceilWaves - waves;
    if (gap < bestGap - 1.0e-6f ||
        (gap < bestGap + 1.0e-6f && cpb > chunksPerBlock)) {
      bestGap = gap;
      chunksPerBlock = cpb;
    }
  }

  dim3 grid1(numTokens, H_BLOCKS, numSplits);
  kernel<<<grid1, DSV4_BLOCK_THREADS, DYN_SMEM_BYTES, stream>>>(
      q, kvCache, indices, midOut, midLse, topkLength, extraKvCache,
      extraIndices, extraTopkLength, extraTopk,
      extraKvCache == nullptr ? 0 : 64, strideExtraKvBlock, numTokens,
      numSplits, chunksPerBlock, smScale, strideKvBlock);
  FASTLLM_DSV4_CUDA_CHECK_BOOL(cudaGetLastError());

  constexpr int MERGE_THREADS = 64;
  constexpr int DIMS_PER_THREAD = KV::D_V / MERGE_THREADS;
  auto mergeKernel = sparse_mla_decode_dsv4_merge_kernel<
      NUM_HEADS, KV::D_V, MERGE_THREADS, DIMS_PER_THREAD>;
  const size_t mergeSmemBytes = (size_t)numSplits * sizeof(float);
  mergeKernel<<<dim3(numTokens, NUM_HEADS), MERGE_THREADS, mergeSmemBytes,
                stream>>>(midOut, midLse, output, nullptr, attnSink, numTokens,
                          numSplits);
  FASTLLM_DSV4_CUDA_CHECK_BOOL(cudaGetLastError());
  return true;
}

}  // namespace flashinfer::sparse_mla_sm120

extern "C" bool FastllmCudaDeepSeekV4SparseMlaSm120Raw(
    const void *q, const uint8_t *kvCache, const int32_t *indices,
    void *midOut, float *midLse, const int *topkLength, void *output,
    const float *attnSink, const uint8_t *extraKvCache,
    const int32_t *extraIndices, const int *extraTopkLength, int numTokens,
    int numHeads, int mainTopk, int extraTopk, int numSplits, float smScale,
    size_t strideKvBlock, size_t strideExtraKvBlock, void *stream) {
  if (q == nullptr || kvCache == nullptr || indices == nullptr ||
      midOut == nullptr || midLse == nullptr || topkLength == nullptr ||
      output == nullptr || attnSink == nullptr || numTokens <= 0 ||
      numTokens > 64 || mainTopk != 128 || extraTopk < 0 ||
      (extraTopk != 0 && extraTopk != 512) || numSplits <= 0 ||
      strideKvBlock == 0 || (extraTopk > 0 &&
      (extraKvCache == nullptr || extraIndices == nullptr ||
       extraTopkLength == nullptr || strideExtraKvBlock == 0))) {
    return false;
  }
  using namespace flashinfer::sparse_mla_sm120;
  cudaStream_t cudaStream = reinterpret_cast<cudaStream_t>(stream);
#define FASTLLM_DSV4_DISPATCH_HEADS(H)                                      \
  if (numHeads == (H)) {                                                     \
    if (extraTopk == 0) {                                                    \
      return LaunchDecodeDsv4Impl<(H), 128, 64>(                            \
          (const bf16 *)q, kvCache, indices, (bf16 *)midOut, midLse,         \
          topkLength, (bf16 *)output, attnSink, nullptr, nullptr, nullptr,   \
          0, 0, numTokens, numSplits, smScale, strideKvBlock, cudaStream);   \
    }                                                                        \
    return LaunchDecodeDsv4Impl<(H), 128, 64>(                              \
        (const bf16 *)q, kvCache, indices, (bf16 *)midOut, midLse,           \
        topkLength, (bf16 *)output, attnSink, extraKvCache, extraIndices,    \
        extraTopkLength, extraTopk, strideExtraKvBlock, numTokens,           \
        numSplits, smScale, strideKvBlock, cudaStream);                      \
  }
  FASTLLM_DSV4_DISPATCH_HEADS(16)
  FASTLLM_DSV4_DISPATCH_HEADS(8)
  FASTLLM_DSV4_DISPATCH_HEADS(32)
  FASTLLM_DSV4_DISPATCH_HEADS(64)
  FASTLLM_DSV4_DISPATCH_HEADS(128)
#undef FASTLLM_DSV4_DISPATCH_HEADS
  return false;
}

namespace {

constexpr int kDsv4IndexerHeads = 64;
constexpr int kDsv4IndexerHeadDim = 128;
constexpr int kDsv4IndexerBlockQ = 2;
constexpr int kDsv4IndexerBlockKV = 128;
constexpr int kDsv4IndexerQStages = 2;
constexpr int kDsv4IndexerKVStages = 3;
constexpr int kDsv4IndexerNumSms = 170;
constexpr int kDsv4IndexerTmaThreads = 128;
constexpr int kDsv4IndexerMathThreads = 256;

bool MakeDsv4IndexerTma2d(
    CUtensorMap *map, CUtensorMapDataType dtype, const void *globalAddress,
    uint64_t globalInner, uint64_t globalOuter, uint64_t globalOuterStrideBytes,
    uint32_t boxInner, uint32_t boxOuter, CUtensorMapSwizzle swizzle) {
  if (map == nullptr || globalAddress == nullptr || globalInner == 0 ||
      globalOuter == 0 || globalOuterStrideBytes == 0 || boxInner == 0 ||
      boxOuter == 0) {
    return false;
  }
  const cuuint64_t globalDims[2] = {globalInner, globalOuter};
  const cuuint64_t globalStrides[1] = {globalOuterStrideBytes};
  const cuuint32_t boxDims[2] = {boxInner, boxOuter};
  const cuuint32_t elementStrides[2] = {1, 1};
  return cuTensorMapEncodeTiled(
             map, dtype, 2, const_cast<void *>(globalAddress), globalDims,
             globalStrides, boxDims, elementStrides,
             CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
             CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

}  // namespace

// Exact SM120 FP8 MQA scorer used by DeepSeek-V4's learned sparse-attention
// indexer.  The template and launch geometry intentionally match the
// nv_dev_f8e8fb5 DeepGEMM kernel shipped with the local vLLM build.  The caller
// owns quantization and top-k and can fall back to the architecture-independent
// CUDA scorer whenever this narrowly gated launch is unavailable.
extern "C" bool FastllmCudaDeepSeekV4IndexerLogitsSm120Raw(
    const uint8_t *q, const uint8_t *kv, const float *kvScales,
    const float *weights, uint32_t *cuSeqLenKStart, uint32_t *cuSeqLenKEnd,
    float *logits, int seqLen, int kvCapacity, int logitsStride, void *stream) {
  if (q == nullptr || kv == nullptr || kvScales == nullptr ||
      weights == nullptr || cuSeqLenKStart == nullptr ||
      cuSeqLenKEnd == nullptr || logits == nullptr || seqLen <= 0 ||
      seqLen > 64 || kvCapacity <= 0 || kvCapacity % 128 != 0 ||
      logitsStride < kvCapacity || stream == nullptr) {
    return false;
  }

  int device = -1;
  int major = 0;
  int minor = 0;
  int smCount = 0;
  if (cudaGetDevice(&device) != cudaSuccess || device < 0 ||
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                             device) != cudaSuccess ||
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                             device) != cudaSuccess ||
      cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount,
                             device) != cudaSuccess ||
      major != 12 || minor != 0 || smCount != kDsv4IndexerNumSms) {
    cudaGetLastError();
    return false;
  }

  CUtensorMap tensorMapQ;
  CUtensorMap tensorMapKV;
  CUtensorMap tensorMapKVScales;
  CUtensorMap tensorMapWeights;
  const int alignedKvScales = (kvCapacity + 3) / 4 * 4;
  if (!MakeDsv4IndexerTma2d(
          &tensorMapQ, CU_TENSOR_MAP_DATA_TYPE_UINT8, q,
          kDsv4IndexerHeadDim, (uint64_t)seqLen * kDsv4IndexerHeads,
          kDsv4IndexerHeadDim, kDsv4IndexerHeadDim,
          kDsv4IndexerBlockQ * kDsv4IndexerHeads,
          CU_TENSOR_MAP_SWIZZLE_128B) ||
      !MakeDsv4IndexerTma2d(
          &tensorMapKV, CU_TENSOR_MAP_DATA_TYPE_UINT8, kv,
          kDsv4IndexerHeadDim, kvCapacity, kDsv4IndexerHeadDim,
          kDsv4IndexerHeadDim, kDsv4IndexerBlockKV,
          CU_TENSOR_MAP_SWIZZLE_128B) ||
      !MakeDsv4IndexerTma2d(
          &tensorMapKVScales, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, kvScales,
          alignedKvScales, 1, (uint64_t)alignedKvScales * sizeof(float),
          kDsv4IndexerBlockKV, 1, CU_TENSOR_MAP_SWIZZLE_NONE) ||
      !MakeDsv4IndexerTma2d(
          &tensorMapWeights, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, weights,
          kDsv4IndexerHeads, seqLen,
          (uint64_t)kDsv4IndexerHeads * sizeof(float),
          kDsv4IndexerHeads, kDsv4IndexerBlockQ,
          CU_TENSOR_MAP_SWIZZLE_NONE)) {
    return false;
  }

  int kvSplits = 1;
  const int numQBlocks = (seqLen + kDsv4IndexerBlockQ - 1) /
                         kDsv4IndexerBlockQ;
  const int maxKVBlocks = (kvCapacity + kDsv4IndexerBlockKV - 1) /
                          kDsv4IndexerBlockKV;
  if (numQBlocks < kDsv4IndexerNumSms && maxKVBlocks > 1) {
    const int maxSplits = std::max(
        1, std::min(kDsv4IndexerNumSms / std::max(1, numQBlocks) + 1,
                    maxKVBlocks / 4));
    double bestCost = 1.0e30;
    for (int split = 1; split <= maxSplits; ++split) {
      const int waves =
          (numQBlocks * split + kDsv4IndexerNumSms - 1) /
          kDsv4IndexerNumSms;
      const double cost = (double)waves / split;
      if (cost < bestCost - 1.0e-9) {
        bestCost = cost;
        kvSplits = split;
      }
    }
  }

  constexpr int qStageBytes =
      kDsv4IndexerBlockQ * kDsv4IndexerHeads * kDsv4IndexerHeadDim;
  constexpr int weightStageBytes =
      kDsv4IndexerBlockQ * kDsv4IndexerHeads * sizeof(float);
  constexpr int kvStageBytes =
      kDsv4IndexerBlockKV * kDsv4IndexerHeadDim;
  constexpr int kvScaleStageBytes =
      kDsv4IndexerBlockKV * sizeof(float);
  constexpr int dynamicSmemBytes =
      kDsv4IndexerQStages * (qStageBytes + weightStageBytes) +
      kDsv4IndexerKVStages * (kvStageBytes + kvScaleStageBytes) +
      (kDsv4IndexerQStages * 2 + kDsv4IndexerKVStages * 2) * 8 + 4;

  auto kernel = deep_gemm::sm120_fp8_mqa_logits<
      kDsv4IndexerHeads, kDsv4IndexerHeadDim, false, kDsv4IndexerBlockQ,
      kDsv4IndexerBlockKV, kDsv4IndexerQStages, kDsv4IndexerKVStages,
      kDsv4IndexerNumSms, kDsv4IndexerTmaThreads,
      kDsv4IndexerMathThreads, float>;
  if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           dynamicSmemBytes) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }

  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute.val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(kDsv4IndexerNumSms, kvSplits, 1);
  config.blockDim = dim3(kDsv4IndexerTmaThreads + kDsv4IndexerMathThreads);
  config.dynamicSmemBytes = dynamicSmemBytes;
  config.stream = reinterpret_cast<cudaStream_t>(stream);
  config.attrs = &attribute;
  config.numAttrs = 1;
  cudaError_t state = cudaLaunchKernelEx(
      &config, kernel, (uint32_t)seqLen, (uint32_t)kvCapacity,
      (uint32_t)0, (uint32_t)logitsStride, cuSeqLenKStart,
      cuSeqLenKEnd, logits, tensorMapQ, tensorMapKV, tensorMapKVScales,
      tensorMapWeights);
  if (state != cudaSuccess) {
    std::fprintf(stderr, "DeepSeekV4 SM120 indexer launch failed: %s\n",
                 cudaGetErrorString(state));
    return false;
  }
  return true;
}
