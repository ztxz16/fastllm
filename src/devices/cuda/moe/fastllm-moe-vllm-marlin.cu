/*
 * DeepSeek V4 MXFP4 MoE integration using FastLLM's in-tree Marlin kernel.
 *
 * The CUDA kernel is adapted from vLLM's Apache-2.0
 * marlin_moe_wna16 implementation and compiled directly into FastLLM.  It has
 * no Torch, Python, vLLM extension, dlopen, or machine-path dependency.
 *
 * Weight and scale permutations follow vLLM's Apache-2.0 implementations:
 *   vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py
 *   vllm/model_executor/layers/quantization/utils/marlin_utils.py
 *   csrc/libtorch_stable/quantization/marlin/gptq_marlin_repack.cu
 *
 * The original compact weights are allocated directly (outside FastLLM's
 * model-weight slab), repacked once during warmup, and then released.  This is
 * important for DeepSeek V4: retaining both layouts would cost roughly 16 GiB
 * per rank for all routed-expert layers.
 */

#include "fastllm-cuda.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#define MARLIN_NAMESPACE_NAME fastllm_marlin_moe_wna16
#include "marlin_moe/marlin_template.cuh"
#undef STATIC_ASSERT_SCALAR_TYPE_VALID
#undef MARLIN_NAMESPACE_NAME

namespace fastllm_marlin_moe {

namespace marlin_kernel = ::fastllm_marlin_moe_wna16;
namespace marlin_types = ::fastllm_marlin_moe_types;

using MarlinMoeKernelFn = void (*)(
    const int4 *, const int4 *, int4 *, int4 *, const int4 *,
    const float *, const int4 *, const float *, const int4 *, const int *,
    const int32_t *, const int32_t *, const int32_t *, const float *,
    int, bool, int, int, int, int, int *, bool, bool, bool);

static MarlinMoeKernelFn GetMarlinMoeKernel(int threadK, int threadN,
                                            int &threads) {
    if (threadK == 128 && threadN == 128) {
        threads = 256;
        return marlin_kernel::Marlin<
            marlin_types::kBFloat16.id(), marlin_types::kFE2M1f.id(),
            marlin_types::kBFloat16.id(), marlin_types::kFE8M0fnu.id(),
            256, 1, 8, 8, true, 4, 2, false>;
    }
    if (threadK == 64 && threadN == 128) {
        threads = 128;
        return marlin_kernel::Marlin<
            marlin_types::kBFloat16.id(), marlin_types::kFE2M1f.id(),
            marlin_types::kBFloat16.id(), marlin_types::kFE8M0fnu.id(),
            128, 1, 8, 4, true, 4, 2, false>;
    }
    threads = 0;
    return nullptr;
}

static MarlinMoeKernelFn GetAwqMarlinMoeKernel(bool gate, bool smallBatch,
                                               int &threads) {
    if (smallBatch) {
        // vLLM's small-M choice: an 8-row MoE tile backed by the
        // 64x128x128-thread configuration. Decode routes at most eight rows,
        // so this avoids the 32/64-row padding used by prefill.
        threads = 128;
        return marlin_kernel::Marlin<
            marlin_types::kFloat16.id(), marlin_types::kU4.id(),
            marlin_types::kFloat16.id(), marlin_types::kFloat16.id(),
            128, 1, 8, 4, true, 4, 2, false>;
    }
    if (gate) {
        threads = 256;
        return marlin_kernel::Marlin<
            marlin_types::kFloat16.id(), marlin_types::kU4.id(),
            marlin_types::kFloat16.id(), marlin_types::kFloat16.id(),
            256, 4, 16, 4, false, 4, 2, false>;
    }
    threads = 128;
    return marlin_kernel::Marlin<
        marlin_types::kFloat16.id(), marlin_types::kU4.id(),
        marlin_types::kFloat16.id(), marlin_types::kFloat16.id(),
        128, 2, 8, 4, false, 4, 2, false>;
}

static int GetAwqMarlinMoeSharedMemorySize(bool gate, bool smallBatch) {
    // Match the two steady-state configurations selected by vLLM for this
    // W4A16 MoE shape.  The values include all activation, weight, scale,
    // zero-point, reduction, and route-metadata stages.
    if (smallBatch) {
        return 27392;
    }
    return gate ? 101376 : 49664;
}

static int GetMarlinMoeSharedMemorySize(int threadK, int threadN) {
    constexpr int stages = 4;
    constexpr int threadM = 16;
    constexpr int groupSize = 32;
    constexpr int packFactor = 8;

    int groupCount = (threadK + groupSize - 1) / groupSize;
    int scaleBytes = groupCount * threadN * 2 * stages;
    int activationBytes = stages * threadM * threadK * 2;
    int weightBytes = stages * (threadK * threadN / packFactor) * 4;
    int reductionBytes = threadM * (threadN + 8) * 2;
    int biasBytes = threadN * 2;
    int temporaryBytes =
        std::max(std::max(weightBytes, reductionBytes),
                 std::min(weightBytes, reductionBytes) + biasBytes);
    int metadataBytes = threadM * 16;
    return temporaryBytes + activationBytes + scaleBytes + metadataBytes;
}

static bool MarlinMoeDeviceSupported(int device) {
    int major = 0;
    return cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                  device) == cudaSuccess &&
           major >= 8;
}

static bool PrepareMarlinMoeKernels(int device) {
    if (!MarlinMoeDeviceSupported(device)) {
        return false;
    }
    int maxSharedMemory = 0;
    if (cudaDeviceGetAttribute(&maxSharedMemory,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               device) != cudaSuccess ||
        maxSharedMemory <= 0) {
        return false;
    }

    static const int configurations[][2] = {{128, 128}, {64, 128}};
    for (const auto &configuration : configurations) {
        int threads = 0;
        MarlinMoeKernelFn kernel =
            GetMarlinMoeKernel(configuration[0], configuration[1], threads);
        if (kernel == nullptr ||
            GetMarlinMoeSharedMemorySize(configuration[0], configuration[1]) >
                maxSharedMemory ||
            cudaFuncSetAttribute(kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 maxSharedMemory) != cudaSuccess) {
            return false;
        }
    }
    return true;
}

static bool PrepareAwqMarlinMoeKernels(int device) {
    if (!MarlinMoeDeviceSupported(device)) {
        return false;
    }
    int maxSharedMemory = 0;
    if (cudaDeviceGetAttribute(&maxSharedMemory,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               device) != cudaSuccess ||
        maxSharedMemory <= 0) {
        return false;
    }
    for (bool smallBatch : {false, true}) {
        for (bool gate : {true, false}) {
            int threads = 0;
            MarlinMoeKernelFn kernel =
                GetAwqMarlinMoeKernel(gate, smallBatch, threads);
            int sharedMemory =
                GetAwqMarlinMoeSharedMemorySize(gate, smallBatch);
            if (kernel == nullptr || sharedMemory > maxSharedMemory ||
                cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    sharedMemory) != cudaSuccess) {
                return false;
            }
        }
    }
    return true;
}

static bool LaunchMarlinMoe(
        const void *activation, const void *weight, void *output,
        float *temporaryOutput, const void *scales,
        const int32_t *sortedTokenIds, const int32_t *expertIds,
        const int32_t *numTokensPadded, const float *topkWeights,
        int topk, bool multiplyTopkWeights, int numGroups, int rows,
        int outputColumns, int inputColumns, int *workspace,
        cudaStream_t stream, int threadK, int threadN, int sms,
        int blocksPerSm) {
    int threads = 0;
    MarlinMoeKernelFn kernel = GetMarlinMoeKernel(threadK, threadN, threads);
    if (kernel == nullptr || activation == nullptr || weight == nullptr ||
        output == nullptr || temporaryOutput == nullptr || scales == nullptr ||
        sortedTokenIds == nullptr || expertIds == nullptr ||
        numTokensPadded == nullptr || topkWeights == nullptr ||
        workspace == nullptr || rows <= 0 || outputColumns <= 0 ||
        inputColumns <= 0 || numGroups != inputColumns / 32 ||
        inputColumns % threadK != 0 || outputColumns % threadN != 0 ||
        sms <= 0 || blocksPerSm <= 0) {
        return false;
    }

    kernel<<<sms * blocksPerSm, threads,
             GetMarlinMoeSharedMemorySize(threadK, threadN), stream>>>(
        reinterpret_cast<const int4 *>(activation),
        reinterpret_cast<const int4 *>(weight),
        reinterpret_cast<int4 *>(output),
        reinterpret_cast<int4 *>(temporaryOutput),
        nullptr, nullptr, reinterpret_cast<const int4 *>(scales), nullptr,
        nullptr, nullptr, sortedTokenIds, expertIds, numTokensPadded,
        topkWeights, topk, multiplyTopkWeights, numGroups, rows,
        outputColumns, inputColumns, workspace, false, false, true);
    return cudaPeekAtLastError() == cudaSuccess;
}

static bool LaunchAwqMarlinMoe(
        bool gate, bool smallBatch, const void *activation,
        const void *weight, void *output,
        float *temporaryOutput, const void *scales, const void *zeros,
        const int32_t *sortedTokenIds, const int32_t *expertIds,
        const int32_t *numTokensPadded, const float *topkWeights,
        int topk, bool multiplyTopkWeights, int rows, int outputColumns,
        int inputColumns, int *workspace, cudaStream_t stream, int sms) {
    int threads = 0;
    MarlinMoeKernelFn kernel =
        GetAwqMarlinMoeKernel(gate, smallBatch, threads);
    if (kernel == nullptr || activation == nullptr || weight == nullptr ||
        output == nullptr || temporaryOutput == nullptr || scales == nullptr ||
        zeros == nullptr || sortedTokenIds == nullptr || expertIds == nullptr ||
        numTokensPadded == nullptr || topkWeights == nullptr ||
        workspace == nullptr || rows <= 0 || outputColumns <= 0 ||
        inputColumns <= 0 || inputColumns % 32 != 0 || sms <= 0) {
        return false;
    }

    const int blocksPerSm = smallBatch ? (gate ? 1 : 3) : (gate ? 1 : 2);
    kernel<<<sms * blocksPerSm, threads,
             GetAwqMarlinMoeSharedMemorySize(gate, smallBatch), stream>>>(
        reinterpret_cast<const int4 *>(activation),
        reinterpret_cast<const int4 *>(weight),
        reinterpret_cast<int4 *>(output),
        reinterpret_cast<int4 *>(temporaryOutput),
        nullptr, nullptr, reinterpret_cast<const int4 *>(scales), nullptr,
        reinterpret_cast<const int4 *>(zeros), nullptr, sortedTokenIds,
        expertIds, numTokensPadded, topkWeights, topk,
        multiplyTopkWeights, inputColumns / 32, rows, outputColumns,
        inputColumns, workspace, false, false, true);
    return cudaPeekAtLastError() == cudaSuccess;
}

template <typename T>
__device__ __forceinline__ float ToFloat(T value);

template <>
__device__ __forceinline__ float ToFloat(half value) {
    return __half2float(value);
}

template <>
__device__ __forceinline__ float ToFloat(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T FromFloat(float value);

template <>
__device__ __forceinline__ half FromFloat(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 FromFloat(float value) {
    return __float2bfloat16_rn(value);
}

// FastLLM compact MXFP4 is [N, K/2] bytes.  vLLM first views each row as
// uint32 and transposes it to GPTQ layout [K/8, N] before Marlin repacking.
__global__ void TransposePackedFp4Kernel(const uint32_t *__restrict__ source,
                                         uint32_t *__restrict__ destination,
                                         int rows, int packedK) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int count = rows * packedK;
    if (id >= count) {
        return;
    }
    int row = id / packedK;
    int k = id - row * packedK;
    destination[(size_t)k * rows + row] = source[id];
}

// Equivalent to marlin_permute_scales(..., group_size=32, is_a_8bit=False)
// followed by mxfp4_marlin_process_scales(..., input_dtype=fp16/bf16).
// E8M0 values are exact powers of two, so conversion through fp16/bf16 does
// not alter the byte; only the two permutations need to be materialized.
__global__ void PermuteMxfp4ScalesKernel(const uint8_t *__restrict__ source,
                                         uint8_t *__restrict__ destination,
                                         int outputRows, int groups) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int count = outputRows * groups;
    if (id >= count) {
        return;
    }

    // mxfp4_marlin_process_scales: [0, 2, 1, 3] in each group of four.
    constexpr int processPerm[4] = {0, 2, 1, 3};
    int block = id & ~63;
    int position = id & 63;
    int afterProcess = (position & ~3) + processPerm[position & 3];

    // get_scale_perms(): destination i*8+j reads source i+8*j.
    int scaleSource = (afterProcess >> 3) + 8 * (afterProcess & 7);
    int transposedFlat = block + scaleSource;
    int group = transposedFlat / outputRows;
    int row = transposedFlat - group * outputRows;
    destination[id] = source[(size_t)row * groups + group];
}

__global__ void BuildEpMetadataKernel(const int32_t *__restrict__ globalIndices,
                                      int32_t *__restrict__ sortedTokenIds,
                                      int32_t *__restrict__ expertIds,
                                      int32_t *__restrict__ numTokensPadded,
                                      int topk, int ownerRank, int ownerCount,
                                      int localExperts) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    int active = 0;
    for (int slot = 0; slot < topk; ++slot) {
        int expert = globalIndices[slot];
        if (ownerRank < 0 || ownerCount <= 0 || expert < 0 ||
            expert % ownerCount != ownerRank) {
            continue;
        }
        int localExpert = expert / ownerCount;
        if (localExpert < 0 || localExpert >= localExperts) {
            continue;
        }
        int base = active * 8;
        sortedTokenIds[base] = slot;
        for (int i = 1; i < 8; ++i) {
            sortedTokenIds[base + i] = topk;
        }
        expertIds[active] = localExpert;
        ++active;
    }
    numTokensPadded[0] = active * 8;
}

// Build one route list for a small multi-row verification batch. Marlin stores
// gate/up outputs by flattened route id (row * topk + slot), so the same
// metadata can be reused by the down projection with topk=1 and
// rows=rows*topk. Keep one route in row zero of each padded block, matching the
// established per-row path exactly; coalescing routes of one expert into other
// tile rows changes low-order MMA accumulation on near-tied DSpark logits.
// A single persistent Marlin launch can still consume every block, removing
// the eight complete launch sequences per layer without changing tile layout.
__global__ void BuildEpMetadataRowsKernel(
        const int32_t *__restrict__ globalIndices,
        int32_t *__restrict__ sortedTokenIds,
        int32_t *__restrict__ expertIds,
        int32_t *__restrict__ numTokensPadded,
        int rows, int topk, int ownerRank, int ownerCount,
        int localExperts) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const int routes = rows * topk;
    int activeBlocks = 0;
    for (int route = 0; route < routes; ++route) {
        int expert = globalIndices[route];
        if (ownerRank < 0 || ownerCount <= 0 || expert < 0 ||
            expert % ownerCount != ownerRank) {
            continue;
        }
        int localExpert = expert / ownerCount;
        if (localExpert < 0 || localExpert >= localExperts) {
            continue;
        }
        int base = activeBlocks * 8;
        sortedTokenIds[base] = route;
        for (int position = base + 1; position < base + 8; ++position) {
            sortedTokenIds[position] = routes;
        }
        expertIds[activeBlocks] = localExpert;
        ++activeBlocks;
    }
    numTokensPadded[0] = activeBlocks * 8;
}

__global__ void InitAwqMoeRouteCountersKernel(
        int32_t *__restrict__ expertCounts,
        int32_t *__restrict__ expertCursors, int experts) {
    int expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert < experts) {
        expertCounts[expert] = 0;
        expertCursors[expert] = 0;
    }
}

__global__ void CountAwqMoeRoutesKernel(
        const int32_t *__restrict__ indices,
        int32_t *__restrict__ expertCounts, int routes, int experts) {
    int route = blockIdx.x * blockDim.x + threadIdx.x;
    if (route >= routes) {
        return;
    }
    int expert = indices[route];
    if ((unsigned int)expert < (unsigned int)experts) {
        atomicAdd(expertCounts + expert, 1);
    }
}

__global__ void BuildAwqMoeRouteOffsetsKernel(
        const int32_t *__restrict__ expertCounts,
        int32_t *__restrict__ expertStarts,
        int32_t *__restrict__ sortedTokenIds,
        int32_t *__restrict__ gateExpertIds,
        int32_t *__restrict__ downExpertIds,
        int32_t *__restrict__ numTokensPadded, int routes, int experts,
        int gateBlock, int downBlock) {

    if (threadIdx.x == 0) {
        int32_t total = 0;
        for (int expert = 0; expert < experts; ++expert) {
            expertStarts[expert] = total;
            total += ((expertCounts[expert] + gateBlock - 1) / gateBlock) *
                     gateBlock;
        }
        numTokensPadded[0] = total;
    }
    __syncthreads();

    for (int expert = threadIdx.x; expert < experts;
         expert += blockDim.x) {
        int start = expertStarts[expert];
        int padded = ((expertCounts[expert] + gateBlock - 1) / gateBlock) *
                     gateBlock;
        for (int position = start; position < start + padded; ++position) {
            sortedTokenIds[position] = routes;
        }
        for (int position = start; position < start + padded;
             position += gateBlock) {
            gateExpertIds[position / gateBlock] = expert;
        }
        for (int position = start; position < start + padded;
             position += downBlock) {
            downExpertIds[position / downBlock] = expert;
        }
    }
}

__global__ void ScatterAwqMoeRoutesKernel(
        const int32_t *__restrict__ indices,
        const int32_t *__restrict__ expertStarts,
        int32_t *__restrict__ expertCursors,
        int32_t *__restrict__ sortedTokenIds, int routes, int experts) {
    int route = blockIdx.x * blockDim.x + threadIdx.x;
    if (route >= routes) {
        return;
    }
    int expert = indices[route];
    if ((unsigned int)expert >= (unsigned int)experts) {
        return;
    }
    int offset = atomicAdd(expertCursors + expert, 1);
    sortedTokenIds[expertStarts[expert] + offset] = route;
}

template <typename T>
__global__ void SwigluRowsKernel(const T *__restrict__ gateUp,
                                 T *__restrict__ output,
                                 int rows, int intermediate) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int count = rows * intermediate;
    if (id >= count) {
        return;
    }
    int row = id / intermediate;
    int col = id - row * intermediate;
    const T *input = gateUp + (size_t)row * intermediate * 2;
    float gate = ToFloat(input[col]);
    float up = ToFloat(input[intermediate + col]);
    output[id] = FromFloat<T>((gate / (1.0f + expf(-gate))) * up);
}

template <typename T>
__global__ void ReduceEpRowsKernel(const T *__restrict__ expertRows,
                                   const int32_t *__restrict__ globalIndices,
                                   T *__restrict__ output, int logicalRows,
                                   int topk,
                                   int ownerRank, int ownerCount, int hidden) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = logicalRows * hidden;
    if (index >= total) {
        return;
    }
    int row = index / hidden;
    int col = index - row * hidden;
    float value = 0.0f;
    if (ownerRank >= 0 && ownerCount > 0) {
        for (int slot = 0; slot < topk; ++slot) {
            int route = row * topk + slot;
            int expert = globalIndices[route];
            if (expert >= 0 && expert % ownerCount == ownerRank) {
                value += ToFloat(
                    expertRows[(size_t)route * hidden + col]);
            }
        }
    }
    output[(size_t)row * hidden + col] = FromFloat<T>(value);
}

__global__ void SumAwqMoeRowsKernel(const half *__restrict__ rows,
                                    half *__restrict__ output,
                                    int batch, int topk, int hidden) {
    int pair = blockIdx.x * blockDim.x + threadIdx.x;
    int pairsPerRow = hidden / 2;
    int count = batch * pairsPerRow;
    if (pair >= count) {
        return;
    }
    int token = pair / pairsPerRow;
    int columnPair = pair - token * pairsPerRow;
    float2 sum = make_float2(0.0f, 0.0f);
    for (int slot = 0; slot < topk; ++slot) {
        const half2 value = reinterpret_cast<const half2 *>(rows)[
            ((size_t)token * topk + slot) * pairsPerRow + columnPair];
        float2 converted = __half22float2(value);
        sum.x += converted.x;
        sum.y += converted.y;
    }
    reinterpret_cast<half2 *>(output)[
        (size_t)token * pairsPerRow + columnPair] = __floats2half2_rn(
            sum.x, sum.y);
}

__device__ __forceinline__ int AwqDecodeLoadZero(
        const uint8_t *__restrict__ zeros, size_t rowMajorIndex) {
    uint8_t packed = __ldg(zeros + (rowMajorIndex >> 1));
    return (rowMajorIndex & 1) ? (packed & 15) : (packed >> 4);
}

__device__ __forceinline__ size_t AwqMarlinWeightTileBase(
        int inputTile, int output, int outputColumns) {
    const int outputTile = output >> 6;
    const int outputInTile = output & 63;
    const int warp = outputInTile >> 4;
    const int tensorCoreColumn = outputInTile & 7;
    const int outputTiles = outputColumns >> 6;
    return ((size_t)inputTile * outputTiles + outputTile) * 128 +
           tensorCoreColumn * 16 + warp;
}

__device__ __forceinline__ half2 AwqDecodeZeroHalf2(int zero) {
    const uint32_t scalar = (uint32_t)(0x6400 + zero);
    const uint32_t packed = scalar | (scalar << 16);
    return *reinterpret_cast<const half2 *>(&packed);
}

__device__ __forceinline__ void AwqMarlinAccumulatePackedPair(
        uint32_t packed, half2 inputLow, half2 inputHigh,
        half2 zero0, half2 zero1, half2 &sum0, half2 &sum1) {
    half2 quant0[2];
    half2 quant1[2];
    marlin_kernel::dequant<
        half2, marlin_types::kU4.id(), true>(
            (int)packed, quant0);
    marlin_kernel::dequant<
        half2, marlin_types::kU4.id(), true>(
            (int)(packed >> 8), quant1);
    sum0 = __hfma2(inputLow, __hsub2(quant0[0], zero0), sum0);
    sum0 = __hfma2(inputHigh, __hsub2(quant0[1], zero0), sum0);
    sum1 = __hfma2(inputLow, __hsub2(quant1[0], zero1), sum1);
    sum1 = __hfma2(inputHigh, __hsub2(quant1[1], zero1), sum1);
}

__device__ __forceinline__ float AwqHorizontalHalf2(half2 value) {
    const float2 converted = __half22float2(value);
    return converted.x + converted.y;
}

__device__ __forceinline__ float AwqMarlinDot16(
        const half *__restrict__ input,
        const uint32_t *__restrict__ weight,
        int inputTile, int output, int outputColumns,
        float scale, int zero) {
    const int outputInTile = output & 63;
    const int nibbleBase = (outputInTile & 8) ? 2 : 0;
    const size_t tileBase =
        AwqMarlinWeightTileBase(inputTile, output, outputColumns);
    const half *inputBase = input + inputTile * 16;
    float sum = 0.0f;
#pragma unroll
    for (int rowPair = 0; rowPair < 4; ++rowPair) {
        uint32_t packed = __ldg(weight + tileBase + rowPair * 4);
        int low = rowPair * 2;
        int q0 = (packed >> ((nibbleBase + 0) * 4)) & 15;
        int q8 = (packed >> ((nibbleBase + 1) * 4)) & 15;
        int q1 = (packed >> ((nibbleBase + 4) * 4)) & 15;
        int q9 = (packed >> ((nibbleBase + 5) * 4)) & 15;
        sum += __half2float(inputBase[low]) * (q0 - zero) * scale;
        sum += __half2float(inputBase[low + 1]) * (q1 - zero) * scale;
        sum += __half2float(inputBase[low + 8]) * (q8 - zero) * scale;
        sum += __half2float(inputBase[low + 9]) * (q9 - zero) * scale;
    }
    return sum;
}

template <int WARPS_PER_BLOCK>
__global__ void AwqMarlinMoeDecodeGateupSwigluKernel(
        const half *__restrict__ input,
        const uint32_t *__restrict__ weights,
        const half *__restrict__ scales,
        const uint8_t *__restrict__ zeros,
        const int32_t *__restrict__ indices,
        half *__restrict__ middle,
        int topk, int experts, int hidden, int intermediate) {
    const int warpInBlock = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int task = blockIdx.x * WARPS_PER_BLOCK + warpInBlock;
    const int tasks = topk * intermediate;
    if (task >= tasks) {
        return;
    }
    const int route = task / intermediate;
    const int output = task - route * intermediate;
    const int expert = __ldg(indices + route);
    float gate = 0.0f;
    float up = 0.0f;
    if ((unsigned int)expert < (unsigned int)experts) {
        const int outputColumns = intermediate * 2;
        const size_t weightWords =
            (size_t)(hidden / 8) * outputColumns;
        const size_t scaleValues =
            (size_t)(hidden / 32) * outputColumns;
        const uint32_t *expertWeight =
            weights + (size_t)expert * weightWords;
        const half *expertScale =
            scales + (size_t)expert * scaleValues;
        const uint8_t *expertZero =
            zeros + (size_t)expert * (scaleValues / 2);
        for (int inputTile = lane; inputTile < hidden / 16;
             inputTile += 32) {
            int group = inputTile >> 1;
            int groups = hidden / 32;
            size_t gateMeta = (size_t)output * groups + group;
            size_t upMeta =
                (size_t)(intermediate + output) * groups + group;
            float gateScale = __half2float(__ldg(expertScale + gateMeta));
            float upScale = __half2float(__ldg(expertScale + upMeta));
            int gateZero = AwqDecodeLoadZero(expertZero, gateMeta);
            int upZero = AwqDecodeLoadZero(expertZero, upMeta);
            int gateNibbleBase = (output & 8) ? 2 : 0;
            int upOutput = intermediate + output;
            int upNibbleBase = (upOutput & 8) ? 2 : 0;
            size_t gateTileBase = AwqMarlinWeightTileBase(
                inputTile, output, outputColumns);
            size_t upTileBase = AwqMarlinWeightTileBase(
                inputTile, upOutput, outputColumns);
            union_half8 inputLow, inputHigh;
            const half *inputBase = input + inputTile * 16;
            inputLow.in =
                *reinterpret_cast<const uint4 *>(inputBase);
            inputHigh.in =
                *reinterpret_cast<const uint4 *>(inputBase + 8);
#pragma unroll
            for (int rowPair = 0; rowPair < 4; ++rowPair) {
                uint32_t gatePacked =
                    __ldg(expertWeight + gateTileBase + rowPair * 4);
                uint32_t upPacked =
                    __ldg(expertWeight + upTileBase + rowPair * 4);
                int low = rowPair * 2;
                float input0 = __half2float(inputLow.out[low]);
                float input1 = __half2float(inputLow.out[low + 1]);
                float input8 = __half2float(inputHigh.out[low]);
                float input9 = __half2float(inputHigh.out[low + 1]);
                int gateQ0 =
                    (gatePacked >> ((gateNibbleBase + 0) * 4)) & 15;
                int gateQ8 =
                    (gatePacked >> ((gateNibbleBase + 1) * 4)) & 15;
                int gateQ1 =
                    (gatePacked >> ((gateNibbleBase + 4) * 4)) & 15;
                int gateQ9 =
                    (gatePacked >> ((gateNibbleBase + 5) * 4)) & 15;
                int upQ0 =
                    (upPacked >> ((upNibbleBase + 0) * 4)) & 15;
                int upQ8 =
                    (upPacked >> ((upNibbleBase + 1) * 4)) & 15;
                int upQ1 =
                    (upPacked >> ((upNibbleBase + 4) * 4)) & 15;
                int upQ9 =
                    (upPacked >> ((upNibbleBase + 5) * 4)) & 15;
                gate += input0 * (gateQ0 - gateZero) * gateScale;
                gate += input1 * (gateQ1 - gateZero) * gateScale;
                gate += input8 * (gateQ8 - gateZero) * gateScale;
                gate += input9 * (gateQ9 - gateZero) * gateScale;
                up += input0 * (upQ0 - upZero) * upScale;
                up += input1 * (upQ1 - upZero) * upScale;
                up += input8 * (upQ8 - upZero) * upScale;
                up += input9 * (upQ9 - upZero) * upScale;
            }
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate += __shfl_down_sync(0xffffffff, gate, offset);
        up += __shfl_down_sync(0xffffffff, up, offset);
    }
    if (lane == 0) {
        float activated = gate / (1.0f + expf(-gate));
        middle[(size_t)route * intermediate + output] =
            __float2half_rn(activated * up);
    }
}

// Shape-specialized companion to the down kernel below.  Each block produces
// 16 gate/up outputs for one route.  Eight warps split the 64 quantization
// groups, and every packed load supplies the weights for outputs j and j+8.
__global__ void AwqMarlinMoeDecodeGateupH2048N256Top8Kernel(
        const half *__restrict__ input,
        const uint32_t *__restrict__ weights,
        const half *__restrict__ scales,
        const uint8_t *__restrict__ zeros,
        const int32_t *__restrict__ indices,
        half *__restrict__ middle,
        int experts) {
    __shared__ float warpValues[8][32];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int pair = lane & 7;
    const int groupPartition =
        warp * 2 + (lane >> 4);
    const int halfTile = (lane >> 3) & 1;

    constexpr int hidden = 2048;
    constexpr int intermediate = 256;
    constexpr int outputColumns = intermediate * 2;
    constexpr int groups = hidden / 32;
    constexpr size_t weightWords =
        (size_t)(hidden / 8) * outputColumns;
    constexpr size_t scaleValues =
        (size_t)groups * outputColumns;

    const int route = blockIdx.x >> 4;
    const int outputTask = blockIdx.x & 15;
    const int outputTile = outputTask >> 2;
    const int columnPairGroup = outputTask & 3;
    const int tensorCoreColumn =
        columnPairGroup * 2 + (pair >> 2);
    const int marlinWarp = pair & 3;
    const int output0 =
        outputTile * 64 + marlinWarp * 16 + tensorCoreColumn;
    const int output1 = output0 + 8;
    const int upOutput0 = intermediate + output0;

    float gate0 = 0.0f;
    float gate1 = 0.0f;
    float up0 = 0.0f;
    float up1 = 0.0f;
    const int expert = __ldg(indices + route);
    if ((unsigned int)expert < (unsigned int)experts) {
        const uint32_t *expertWeight =
            weights + (size_t)expert * weightWords;
        const half *expertScale =
            scales + (size_t)expert * scaleValues;
        const uint8_t *expertZero =
            zeros + (size_t)expert * (scaleValues / 2);
        const half2 *expertScale2 =
            reinterpret_cast<const half2 *>(expertScale);

#pragma unroll
        for (int group = groupPartition; group < groups; group += 16) {
            const int inputTile = group * 2 + halfTile;
            const size_t gatePairMeta =
                ((size_t)outputTask * groups + group) * 8 + pair;
            const size_t upPairMeta =
                ((size_t)(outputTask + 16) * groups + group) * 8 + pair;
            const float2 gateScale =
                __half22float2(__ldg(expertScale2 + gatePairMeta));
            const float2 upScale =
                __half22float2(__ldg(expertScale2 + upPairMeta));
            const uint8_t gateZero =
                __ldg(expertZero + gatePairMeta);
            const uint8_t upZero =
                __ldg(expertZero + upPairMeta);
            const int gateZero0 = gateZero >> 4;
            const int gateZero1 = gateZero & 15;
            const int upZero0 = upZero >> 4;
            const int upZero1 = upZero & 15;
            const half2 gateZeroHalf0 =
                AwqDecodeZeroHalf2(gateZero0);
            const half2 gateZeroHalf1 =
                AwqDecodeZeroHalf2(gateZero1);
            const half2 upZeroHalf0 =
                AwqDecodeZeroHalf2(upZero0);
            const half2 upZeroHalf1 =
                AwqDecodeZeroHalf2(upZero1);
            const size_t gateTileBase = AwqMarlinWeightTileBase(
                inputTile, output0, outputColumns);
            const size_t upTileBase = AwqMarlinWeightTileBase(
                inputTile, upOutput0, outputColumns);
            const half *inputBase = input + inputTile * 16;
            union_half8 inputLow, inputHigh;
            inputLow.in =
                *reinterpret_cast<const uint4 *>(inputBase);
            inputHigh.in =
                *reinterpret_cast<const uint4 *>(inputBase + 8);
            half2 gateTile0 = __float2half2_rn(0.0f);
            half2 gateTile1 = __float2half2_rn(0.0f);
            half2 upTile0 = __float2half2_rn(0.0f);
            half2 upTile1 = __float2half2_rn(0.0f);
#pragma unroll
            for (int rowPair = 0; rowPair < 4; ++rowPair) {
                const uint32_t gatePacked =
                    __ldg(expertWeight + gateTileBase + rowPair * 4);
                const uint32_t upPacked =
                    __ldg(expertWeight + upTileBase + rowPair * 4);
                AwqMarlinAccumulatePackedPair(
                    gatePacked, inputLow.out2[rowPair],
                    inputHigh.out2[rowPair], gateZeroHalf0,
                    gateZeroHalf1, gateTile0, gateTile1);
                AwqMarlinAccumulatePackedPair(
                    upPacked, inputLow.out2[rowPair],
                    inputHigh.out2[rowPair], upZeroHalf0,
                    upZeroHalf1, upTile0, upTile1);
            }
            gate0 += AwqHorizontalHalf2(gateTile0) * gateScale.x;
            gate1 += AwqHorizontalHalf2(gateTile1) * gateScale.y;
            up0 += AwqHorizontalHalf2(upTile0) * upScale.x;
            up1 += AwqHorizontalHalf2(upTile1) * upScale.y;
        }
    }

    gate0 += __shfl_down_sync(0xffffffff, gate0, 16);
    gate1 += __shfl_down_sync(0xffffffff, gate1, 16);
    up0 += __shfl_down_sync(0xffffffff, up0, 16);
    up1 += __shfl_down_sync(0xffffffff, up1, 16);
    gate0 += __shfl_down_sync(0xffffffff, gate0, 8);
    gate1 += __shfl_down_sync(0xffffffff, gate1, 8);
    up0 += __shfl_down_sync(0xffffffff, up0, 8);
    up1 += __shfl_down_sync(0xffffffff, up1, 8);
    if (lane < 8) {
        warpValues[warp][pair * 4] = gate0;
        warpValues[warp][pair * 4 + 1] = gate1;
        warpValues[warp][pair * 4 + 2] = up0;
        warpValues[warp][pair * 4 + 3] = up1;
    }
    __syncthreads();

    if (warp == 0 && lane < 8) {
        float totalGate0 = 0.0f;
        float totalGate1 = 0.0f;
        float totalUp0 = 0.0f;
        float totalUp1 = 0.0f;
#pragma unroll
        for (int warpIndex = 0; warpIndex < 8; ++warpIndex) {
            totalGate0 += warpValues[warpIndex][pair * 4];
            totalGate1 += warpValues[warpIndex][pair * 4 + 1];
            totalUp0 += warpValues[warpIndex][pair * 4 + 2];
            totalUp1 += warpValues[warpIndex][pair * 4 + 3];
        }
        const float activated0 =
            totalGate0 / (1.0f + expf(-totalGate0));
        const float activated1 =
            totalGate1 / (1.0f + expf(-totalGate1));
        half *routeOutput =
            middle + (size_t)route * intermediate;
        routeOutput[output0] =
            __float2half_rn(activated0 * totalUp0);
        routeOutput[output1] =
            __float2half_rn(activated1 * totalUp1);
    }
}

template <int ROUTES_PER_BLOCK, int LANES_PER_ROUTE>
__global__ void AwqMarlinMoeDecodeDownKernel(
        const half *__restrict__ middle,
        const uint32_t *__restrict__ weights,
        const half *__restrict__ scales,
        const uint8_t *__restrict__ zeros,
        const int32_t *__restrict__ indices,
        const float *__restrict__ scores,
        half *__restrict__ output,
        int topk, int experts, int hidden, int intermediate) {
    __shared__ float routeValues[ROUTES_PER_BLOCK];
    const int route = threadIdx.x / LANES_PER_ROUTE;
    const int lane = threadIdx.x & (LANES_PER_ROUTE - 1);
    const int outputColumn = blockIdx.x;
    float sum = 0.0f;
    if (route < topk) {
        const int expert = __ldg(indices + route);
        if ((unsigned int)expert < (unsigned int)experts) {
            const size_t weightWords =
                (size_t)(intermediate / 8) * hidden;
            const size_t scaleValues =
                (size_t)(intermediate / 32) * hidden;
            const uint32_t *expertWeight =
                weights + (size_t)expert * weightWords;
            const half *expertScale =
                scales + (size_t)expert * scaleValues;
            const uint8_t *expertZero =
                zeros + (size_t)expert * (scaleValues / 2);
            const half *routeInput =
                middle + (size_t)route * intermediate;
            for (int inputTile = lane; inputTile < intermediate / 16;
                 inputTile += LANES_PER_ROUTE) {
                int group = inputTile >> 1;
                size_t meta =
                    (size_t)outputColumn * (intermediate / 32) + group;
                float scale =
                    __half2float(__ldg(expertScale + meta));
                int zero = AwqDecodeLoadZero(expertZero, meta);
                sum += AwqMarlinDot16(
                    routeInput, expertWeight, inputTile, outputColumn,
                    hidden, scale, zero);
            }
        }
    }
#pragma unroll
    for (int offset = LANES_PER_ROUTE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(
            0xffffffff, sum, offset, LANES_PER_ROUTE);
    }
    if (lane == 0) {
        routeValues[route] =
            route < topk ? sum * __ldg(scores + route) : 0.0f;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
#pragma unroll
        for (int routeIndex = 0;
             routeIndex < ROUTES_PER_BLOCK; ++routeIndex) {
            total += routeValues[routeIndex];
        }
        output[outputColumn] = __float2half_rn(total);
    }
}

// Qwen3.5/3.6 TP=2 has H=2048, N=256 and topk=8.  In Marlin's packed
// layout, outputs separated by eight columns share every 32-bit weight word.
// The generic output-parallel kernel reads that word twice and its lanes walk
// input tiles 16 KiB apart.  This shape-specialized kernel assigns two warps
// to each route and computes 16 outputs together, so the eight useful lanes
// read adjacent Marlin columns and unpack both outputs from each word.
__global__ void AwqMarlinMoeDecodeDownH2048N256Top8Kernel(
        const half *__restrict__ middle,
        const uint32_t *__restrict__ weights,
        const half *__restrict__ scales,
        const uint8_t *__restrict__ zeros,
        const int32_t *__restrict__ indices,
        const float *__restrict__ scores,
        half *__restrict__ output,
        int topk, int experts) {
    __shared__ float routeWarpValues[8][2][16];

    const int route = threadIdx.x >> 6;
    const int warpInRoute = (threadIdx.x >> 5) & 1;
    const int lane = threadIdx.x & 31;
    const int pair = lane & 7;
    const int groupPartition =
        warpInRoute * 2 + (lane >> 4);
    const int halfTile = (lane >> 3) & 1;

    const int outputTile = blockIdx.x >> 2;
    const int columnPairGroup = blockIdx.x & 3;
    const int tensorCoreColumn =
        columnPairGroup * 2 + (pair >> 2);
    const int marlinWarp = pair & 3;
    const int output0 =
        outputTile * 64 + marlinWarp * 16 + tensorCoreColumn;
    const int output1 = output0 + 8;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    const int expert =
        route < topk ? __ldg(indices + route) : -1;
    if ((unsigned int)expert < (unsigned int)experts) {
        constexpr int hidden = 2048;
        constexpr int intermediate = 256;
        constexpr int groups = intermediate / 32;
        constexpr size_t weightWords =
            (size_t)(intermediate / 8) * hidden;
        constexpr size_t scaleValues =
            (size_t)groups * hidden;
        const uint32_t *expertWeight =
            weights + (size_t)expert * weightWords;
        const half *expertScale =
            scales + (size_t)expert * scaleValues;
        const uint8_t *expertZero =
            zeros + (size_t)expert * (scaleValues / 2);
        const half2 *expertScale2 =
            reinterpret_cast<const half2 *>(expertScale);
        const half *routeInput =
            middle + (size_t)route * intermediate;

#pragma unroll
        for (int group = groupPartition; group < groups; group += 4) {
            const int inputTile = group * 2 + halfTile;
            const size_t pairMeta =
                ((size_t)blockIdx.x * groups + group) * 8 + pair;
            const float2 scale =
                __half22float2(__ldg(expertScale2 + pairMeta));
            const uint8_t zero = __ldg(expertZero + pairMeta);
            const int zero0 = zero >> 4;
            const int zero1 = zero & 15;
            const half2 zeroHalf0 = AwqDecodeZeroHalf2(zero0);
            const half2 zeroHalf1 = AwqDecodeZeroHalf2(zero1);
            const size_t tileBase = AwqMarlinWeightTileBase(
                inputTile, output0, hidden);
            const half *inputBase = routeInput + inputTile * 16;
            union_half8 inputLow, inputHigh;
            inputLow.in =
                *reinterpret_cast<const uint4 *>(inputBase);
            inputHigh.in =
                *reinterpret_cast<const uint4 *>(inputBase + 8);
            half2 tile0 = __float2half2_rn(0.0f);
            half2 tile1 = __float2half2_rn(0.0f);
#pragma unroll
            for (int rowPair = 0; rowPair < 4; ++rowPair) {
                const uint32_t packed =
                    __ldg(expertWeight + tileBase + rowPair * 4);
                AwqMarlinAccumulatePackedPair(
                    packed, inputLow.out2[rowPair],
                    inputHigh.out2[rowPair], zeroHalf0, zeroHalf1,
                    tile0, tile1);
            }
            sum0 += AwqHorizontalHalf2(tile0) * scale.x;
            sum1 += AwqHorizontalHalf2(tile1) * scale.y;
        }
    }

    // Lanes p, p+8, p+16 and p+24 own the two half-tiles from half of
    // the quantization groups for the same output pair.
    sum0 += __shfl_down_sync(0xffffffff, sum0, 16);
    sum1 += __shfl_down_sync(0xffffffff, sum1, 16);
    sum0 += __shfl_down_sync(0xffffffff, sum0, 8);
    sum1 += __shfl_down_sync(0xffffffff, sum1, 8);
    if (lane < 8) {
        routeWarpValues[route][warpInRoute][pair * 2] = sum0;
        routeWarpValues[route][warpInRoute][pair * 2 + 1] = sum1;
    }
    __syncthreads();

    if (warpInRoute == 0 && lane < 8) {
        const float score =
            route < topk ? __ldg(scores + route) : 0.0f;
        routeWarpValues[route][0][pair * 2] =
            (routeWarpValues[route][0][pair * 2] +
             routeWarpValues[route][1][pair * 2]) * score;
        routeWarpValues[route][0][pair * 2 + 1] =
            (routeWarpValues[route][0][pair * 2 + 1] +
             routeWarpValues[route][1][pair * 2 + 1]) * score;
    }
    __syncthreads();

    if (route == 0 && warpInRoute == 0 && lane < 8) {
        float total0 = 0.0f;
        float total1 = 0.0f;
#pragma unroll
        for (int routeIndex = 0; routeIndex < 8; ++routeIndex) {
            total0 += routeWarpValues[routeIndex][0][pair * 2];
            total1 += routeWarpValues[routeIndex][0][pair * 2 + 1];
        }
        output[output0] = __float2half_rn(total0);
        output[output1] = __float2half_rn(total1);
    }
}

struct MarlinLayerCache {
    static constexpr int kSmallBatchRows = 8;

    bool ready = false;
    std::atomic<bool> retired {false};
    int device = -1;
    int experts = 0;
    int hidden = 0;
    int intermediate = 0;
    int topk = 0;
    int sms = 0;

    uint8_t *gateWeight = nullptr;
    uint8_t *downWeight = nullptr;
    uint8_t *gateScale = nullptr;
    uint8_t *downScale = nullptr;

    int32_t *sortedTokenIds = nullptr;
    int32_t *expertIds = nullptr;
    int32_t *numTokensPadded = nullptr;
    int *workspace = nullptr;
    float *temporaryOutput = nullptr;
    void *gateOutput = nullptr;
    void *downOutput = nullptr;

    std::mutex buildMutex;

    ~MarlinLayerCache();
};

static std::mutex &LayerCacheRegistryMutex() {
    static std::mutex *mutex = new std::mutex();
    return *mutex;
}

static std::map<const fastllm::Data *, std::shared_ptr<MarlinLayerCache>> &
LayerCacheRegistry() {
    // Data destruction explicitly retires entries. Keep the registry object
    // itself alive so static teardown never calls CUDA after runtime shutdown.
    static auto *registry =
        new std::map<const fastllm::Data *, std::shared_ptr<MarlinLayerCache>>();
    return *registry;
}

static thread_local std::map<const fastllm::Data *,
                             std::weak_ptr<MarlinLayerCache>> layerCacheFront;

static void *AllocateDirect(size_t bytes) {
    return bytes == 0 ? nullptr : FastllmCudaDirectMalloc(bytes);
}

static void ReleaseDirect(void *pointer) {
    if (pointer != nullptr) {
        FastllmCudaDirectFree(pointer);
    }
}

static void ReleaseCacheStorage(MarlinLayerCache &cache) {
    int originalDevice = -1;
    bool restoreDevice = false;
    if (cache.device >= 0 && cudaGetDevice(&originalDevice) == cudaSuccess &&
        originalDevice != cache.device) {
        restoreDevice = cudaSetDevice(cache.device) == cudaSuccess;
    }
    ReleaseDirect(cache.gateWeight);
    ReleaseDirect(cache.downWeight);
    ReleaseDirect(cache.gateScale);
    ReleaseDirect(cache.downScale);
    ReleaseDirect(cache.sortedTokenIds);
    ReleaseDirect(cache.expertIds);
    ReleaseDirect(cache.numTokensPadded);
    ReleaseDirect(cache.workspace);
    ReleaseDirect(cache.temporaryOutput);
    ReleaseDirect(cache.gateOutput);
    ReleaseDirect(cache.downOutput);
    cache.gateWeight = nullptr;
    cache.downWeight = nullptr;
    cache.gateScale = nullptr;
    cache.downScale = nullptr;
    cache.sortedTokenIds = nullptr;
    cache.expertIds = nullptr;
    cache.numTokensPadded = nullptr;
    cache.workspace = nullptr;
    cache.temporaryOutput = nullptr;
    cache.gateOutput = nullptr;
    cache.downOutput = nullptr;
    cache.ready = false;
    if (restoreDevice) {
        cudaSetDevice(originalDevice);
    }
}

MarlinLayerCache::~MarlinLayerCache() {
    ReleaseCacheStorage(*this);
}

static bool ValidateWeight(const fastllm::Data *weight, int rows, int columns) {
    if (weight == nullptr || weight->dataType != fastllm::DataType::NVFP4 ||
        weight->dims.size() != 2 || weight->dims[0] != rows ||
        weight->dims[1] != columns || weight->blockK != 1 ||
        weight->blockM != 32 || !weight->scales.empty() ||
        weight->cudaData == nullptr || !weight->directMemory) {
        return false;
    }
    size_t weightBytes = (size_t)rows * columns / 2;
    size_t scaleBytes = (size_t)rows * (columns / 32);
    return weight->expansionBytes >= weightBytes + scaleBytes;
}

static bool BuildLayerCache(fastllm::Data **weights, int weightsBatch,
                            int topk, MarlinLayerCache &cache) {
    if (weights == nullptr || weightsBatch < 4 || (weightsBatch & 1) != 0 ||
        topk <= 0 || topk > 16 || weights[2] == nullptr || weights[3] == nullptr ||
        weights[2]->dims.size() != 2 || weights[3]->dims.size() != 2) {
        return false;
    }

    int experts = weightsBatch / 2 - 1;
    int intermediate = weights[3]->dims[1];
    int hidden = weights[3]->dims[0];
    if (experts <= 0 || hidden <= 0 || intermediate <= 0 ||
        hidden % 128 != 0 || intermediate % 128 != 0 ||
        weights[2]->dims[0] != intermediate * 2 ||
        weights[2]->dims[1] != hidden) {
        return false;
    }
    for (int expert = 0; expert < experts; ++expert) {
        int slot = 2 + expert * 2;
        if (!ValidateWeight(weights[slot], intermediate * 2, hidden) ||
            !ValidateWeight(weights[slot + 1], hidden, intermediate)) {
            static thread_local bool warned = false;
            if (!warned) {
                std::fprintf(stderr,
                             "[FastLLM] Marlin MoE requires direct compact "
                             "MXFP4 expert allocations with block [1,32]; falling back.\n");
                std::fflush(stderr);
                warned = true;
            }
            return false;
        }
    }

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return false;
    }
    int sms = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) !=
            cudaSuccess || sms <= 0 || !PrepareMarlinMoeKernels(device)) {
        return false;
    }

    size_t gateWeightBytes = (size_t)intermediate * 2 * hidden / 2;
    size_t downWeightBytes = (size_t)hidden * intermediate / 2;
    size_t gateScaleBytes = (size_t)intermediate * 2 * (hidden / 32);
    size_t downScaleBytes = (size_t)hidden * (intermediate / 32);
    constexpr size_t scalarBytes = sizeof(uint16_t);
    const int routeCapacity = topk * MarlinLayerCache::kSmallBatchRows;
    // In the worst case every route belongs to a different expert and needs
    // its own eight-row Marlin block.
    const int paddedRouteCapacity = routeCapacity * 8;

    cache.device = device;
    cache.experts = experts;
    cache.hidden = hidden;
    cache.intermediate = intermediate;
    cache.topk = topk;
    cache.sms = sms;
    cache.gateWeight = (uint8_t *)AllocateDirect(gateWeightBytes * experts);
    cache.downWeight = (uint8_t *)AllocateDirect(downWeightBytes * experts);
    cache.gateScale = (uint8_t *)AllocateDirect(gateScaleBytes * experts);
    cache.downScale = (uint8_t *)AllocateDirect(downScaleBytes * experts);
    cache.sortedTokenIds = (int32_t *)AllocateDirect(
        (size_t)paddedRouteCapacity * sizeof(int32_t));
    cache.expertIds = (int32_t *)AllocateDirect(
        (size_t)routeCapacity * sizeof(int32_t));
    cache.numTokensPadded = (int32_t *)AllocateDirect(sizeof(int32_t));
    cache.workspace = (int *)AllocateDirect((size_t)sms * 4 * sizeof(int));
    size_t temporaryFloats = (size_t)sms * 4 * 8 * 256 * 2;
    cache.temporaryOutput = (float *)AllocateDirect(temporaryFloats * sizeof(float));
    cache.gateOutput = AllocateDirect(
        (size_t)routeCapacity * intermediate * 2 * scalarBytes);
    cache.downOutput = AllocateDirect(
        (size_t)routeCapacity * hidden * scalarBytes);
    if (cache.gateWeight == nullptr || cache.downWeight == nullptr ||
        cache.gateScale == nullptr || cache.downScale == nullptr ||
        cache.sortedTokenIds == nullptr || cache.expertIds == nullptr ||
        cache.numTokensPadded == nullptr || cache.workspace == nullptr ||
        cache.temporaryOutput == nullptr || cache.gateOutput == nullptr ||
        cache.downOutput == nullptr) {
        ReleaseCacheStorage(cache);
        return false;
    }

    size_t transposeBytes = std::max(gateWeightBytes, downWeightBytes);
    uint32_t *transposed = (uint32_t *)AllocateDirect(transposeBytes);
    if (transposed == nullptr) {
        ReleaseCacheStorage(cache);
        return false;
    }

    cudaStream_t stream = cudaStreamPerThread;
    bool success = true;
    cudaError_t state = cudaMemsetAsync(cache.workspace, 0,
                                        (size_t)sms * 4 * sizeof(int), stream);
    success = state == cudaSuccess;
    constexpr int threads = 256;
    for (int expert = 0; expert < experts && success; ++expert) {
        int slot = 2 + expert * 2;
        fastllm::Data *gate = weights[slot];
        fastllm::Data *down = weights[slot + 1];

        int gatePackedK = hidden / 8;
        int gateWords = intermediate * 2 * gatePackedK;
        TransposePackedFp4Kernel<<<(gateWords + threads - 1) / threads, threads,
                                   0, stream>>>(
            (const uint32_t *)gate->cudaData, transposed,
            intermediate * 2, gatePackedK);
        success = cudaGetLastError() == cudaSuccess &&
                  FastllmCudaGptqMarlinRepackStream(
                      transposed,
                      (uint32_t *)(cache.gateWeight + gateWeightBytes * expert),
                      hidden, intermediate * 2, (void *)stream);
        if (!success) {
            break;
        }
        int gateScaleCount = (int)gateScaleBytes;
        PermuteMxfp4ScalesKernel<<<(gateScaleCount + threads - 1) / threads,
                                   threads, 0, stream>>>(
            (const uint8_t *)gate->cudaData + gateWeightBytes,
            cache.gateScale + gateScaleBytes * expert,
            intermediate * 2, hidden / 32);

        int downPackedK = intermediate / 8;
        int downWords = hidden * downPackedK;
        TransposePackedFp4Kernel<<<(downWords + threads - 1) / threads, threads,
                                   0, stream>>>(
            (const uint32_t *)down->cudaData, transposed, hidden, downPackedK);
        success = cudaGetLastError() == cudaSuccess &&
                  FastllmCudaGptqMarlinRepackStream(
                      transposed,
                      (uint32_t *)(cache.downWeight + downWeightBytes * expert),
                      intermediate, hidden, (void *)stream);
        if (!success) {
            break;
        }
        int downScaleCount = (int)downScaleBytes;
        PermuteMxfp4ScalesKernel<<<(downScaleCount + threads - 1) / threads,
                                   threads, 0, stream>>>(
            (const uint8_t *)down->cudaData + downWeightBytes,
            cache.downScale + downScaleBytes * expert,
            hidden, intermediate / 32);
        success = cudaGetLastError() == cudaSuccess;
    }
    if (success) {
        success = cudaStreamSynchronize(stream) == cudaSuccess;
    }
    ReleaseDirect(transposed);
    if (!success) {
        ReleaseCacheStorage(cache);
        return false;
    }

    // The consolidated Marlin buffers now own the only required copy.  Source
    // expert allocations were forced to directMemory before upload, so these
    // frees return memory to CUDA instead of leaving holes in a weight slab.
    for (int expert = 0; expert < experts; ++expert) {
        int slot = 2 + expert * 2;
        for (int offset = 0; offset < 2; ++offset) {
            fastllm::Data *weight = weights[slot + offset];
            void *old = weight->cudaData;
            weight->cudaData = nullptr;
            weight->cudaDataBorrowed = false;
            FastllmCudaDirectFree(old);
        }
    }

    cache.ready = true;
    std::fprintf(stderr,
                 "[FastLLM] repacked %d local MXFP4 experts for Marlin "
                 "on GPU %d (H=%d, N=%d).\n",
                 experts, device, hidden, intermediate);
    std::fflush(stderr);
    return true;
}

static std::shared_ptr<MarlinLayerCache> GetOrBuildLayerCache(
        fastllm::Data **weights, int weightsBatch, int topk) {
    if (weights == nullptr || weightsBatch < 4 || weights[2] == nullptr) {
        return {};
    }
    const fastllm::Data *key = weights[2];
    auto local = layerCacheFront.find(key);
    if (local != layerCacheFront.end()) {
        std::shared_ptr<MarlinLayerCache> cached = local->second.lock();
        if (cached != nullptr &&
            !cached->retired.load(std::memory_order_acquire) &&
            cached->ready && cached->topk == topk) {
            return cached;
        }
        layerCacheFront.erase(local);
    }

    std::shared_ptr<MarlinLayerCache> cache;
    {
        std::lock_guard<std::mutex> lock(LayerCacheRegistryMutex());
        auto &registry = LayerCacheRegistry();
        auto it = registry.find(key);
        if (it == registry.end()) {
            cache = std::make_shared<MarlinLayerCache>();
            registry.emplace(key, cache);
        } else {
            cache = it->second;
        }
    }

    bool failed = false;
    {
        std::lock_guard<std::mutex> lock(cache->buildMutex);
        if (cache->retired.load(std::memory_order_acquire)) {
            failed = true;
        } else if (!cache->ready) {
            failed = !BuildLayerCache(weights, weightsBatch, topk, *cache);
            if (failed) {
                // Prevent another waiter from retrying this same cache between
                // build failure and registry removal.
                cache->retired.store(true, std::memory_order_release);
            }
        }
        if (!failed && cache->retired.load(std::memory_order_acquire)) {
            failed = true;
        } else if (!failed && cache->topk != topk) {
            return {};
        }
    }
    if (failed) {
        std::lock_guard<std::mutex> lock(LayerCacheRegistryMutex());
        auto &registry = LayerCacheRegistry();
        auto it = registry.find(key);
        if (it != registry.end() && it->second == cache) {
            registry.erase(it);
        }
        return {};
    }
    layerCacheFront[key] = cache;
    return cache;
}

static void ReleaseLayerCache(const fastllm::Data *key) {
    if (key == nullptr) {
        return;
    }
    std::shared_ptr<MarlinLayerCache> retired;
    {
        std::lock_guard<std::mutex> lock(LayerCacheRegistryMutex());
        auto &registry = LayerCacheRegistry();
        auto it = registry.find(key);
        if (it == registry.end()) {
            return;
        }
        it->second->retired.store(true, std::memory_order_release);
        retired = std::move(it->second);
        registry.erase(it);
    }
    // Destruction happens outside the registry lock. An in-flight invocation
    // retains its own shared_ptr until it has enqueued its final kernel.
}

__global__ void AwqInt4GroupToMarlinQWeightKernel(
        const uint8_t *__restrict__ weight,
        uint32_t *__restrict__ qweight, int inputColumns, int outputColumns) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int packsPerRow = inputColumns / 8;
    int count = packsPerRow * outputColumns;
    if (index >= count) {
        return;
    }

    int pack = index / outputColumns;
    int output = index - pack * outputColumns;
    int inputBase = pack * 8;
    const uint8_t *row =
        weight + (size_t)output * inputColumns / 2;
    uint32_t value = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint8_t packed = row[(inputBase + i) >> 1];
        uint32_t quantized =
            ((inputBase + i) & 1) ? (packed & 15) : (packed >> 4);
        value |= quantized << (i * 4);
    }
    qweight[index] = value;
}

static void AppendAwqMarlinScalesAndZeros(
        const fastllm::Data &weight, std::vector<half> &scales,
        std::vector<uint32_t> &zeros, int outputColumns) {
    const int groups = weight.group;
    const size_t values = (size_t)groups * outputColumns;
    const size_t scaleBase = scales.size();
    scales.resize(scaleBase + values);
    std::vector<float> scaleGroupOutput(values);
    std::vector<uint8_t> zeroGroupOutput(values);

    for (int group = 0; group < groups; ++group) {
        for (int output = 0; output < outputColumns; ++output) {
            size_t destination = (size_t)group * outputColumns + output;
            size_t source = (size_t)output * groups + group;
            scaleGroupOutput[destination] = weight.scales[source];
            int zero = (int)weight.zeros[source];
            zeroGroupOutput[destination] =
                (uint8_t)std::max(0, std::min(15, zero));
        }
    }

    static constexpr int scalePermutation[64] = {
        0, 8, 16, 24, 32, 40, 48, 56,
        1, 9, 17, 25, 33, 41, 49, 57,
        2, 10, 18, 26, 34, 42, 50, 58,
        3, 11, 19, 27, 35, 43, 51, 59,
        4, 12, 20, 28, 36, 44, 52, 60,
        5, 13, 21, 29, 37, 45, 53, 61,
        6, 14, 22, 30, 38, 46, 54, 62,
        7, 15, 23, 31, 39, 47, 55, 63
    };
    std::vector<uint8_t> permutedZeros(values);
    for (size_t base = 0; base < values; base += 64) {
        for (int i = 0; i < 64; ++i) {
            int source = scalePermutation[i];
            scales[scaleBase + base + i] =
                (half)scaleGroupOutput[base + source];
            permutedZeros[base + i] = zeroGroupOutput[base + source];
        }
    }

    static constexpr int zeroInterleave[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    zeros.reserve(zeros.size() + values / 8);
    for (size_t base = 0; base < values; base += 8) {
        uint32_t packed = 0;
        for (int i = 0; i < 8; ++i) {
            packed |= (uint32_t)permutedZeros[
                base + zeroInterleave[i]] << (i * 4);
        }
        zeros.push_back(packed);
    }
}

static void AppendAwqDecodeScalesAndZeros(
        const fastllm::Data &weight, std::vector<half> &scales,
        std::vector<uint8_t> &zeros) {
    const size_t values = weight.scales.size();
    const size_t scaleBase = scales.size();
    const size_t zeroBase = zeros.size();
    scales.resize(scaleBase + values);
    zeros.resize(zeroBase + (values + 1) / 2, 0);
    for (size_t index = 0; index < values; ++index) {
        scales[scaleBase + index] = (half)weight.scales[index];
        int zero = (int)weight.zeros[index];
        zero = std::max(0, std::min(15, zero));
        uint8_t &packed = zeros[zeroBase + (index >> 1)];
        if (index & 1) {
            packed |= (uint8_t)zero;
        } else {
            packed |= (uint8_t)(zero << 4);
        }
    }
}

// Keep the same byte count as the row-major decode metadata, but order it by
// the 16-output tasks used by the shape-specialized GEMV kernels.  One half2
// and one byte then describe the two outputs sharing a Marlin weight word.
static void AppendAwqPairedDecodeScalesAndZeros(
        const fastllm::Data &weight, std::vector<half> &scales,
        std::vector<uint8_t> &zeros) {
    const int outputs = weight.dims[0];
    const int groups = weight.group;
    const int tasks = outputs / 16;
    const size_t values = weight.scales.size();
    scales.reserve(scales.size() + values);
    zeros.reserve(zeros.size() + values / 2);
    for (int task = 0; task < tasks; ++task) {
        const int outputTile = task >> 2;
        const int columnPairGroup = task & 3;
        for (int group = 0; group < groups; ++group) {
            for (int pair = 0; pair < 8; ++pair) {
                const int tensorCoreColumn =
                    columnPairGroup * 2 + (pair >> 2);
                const int marlinWarp = pair & 3;
                const int output0 =
                    outputTile * 64 + marlinWarp * 16 +
                    tensorCoreColumn;
                const int output1 = output0 + 8;
                const size_t meta0 =
                    (size_t)output0 * groups + group;
                const size_t meta1 =
                    (size_t)output1 * groups + group;
                scales.push_back((half)weight.scales[meta0]);
                scales.push_back((half)weight.scales[meta1]);
                int zero0 = std::max(
                    0, std::min(15, (int)weight.zeros[meta0]));
                int zero1 = std::max(
                    0, std::min(15, (int)weight.zeros[meta1]));
                zeros.push_back(
                    (uint8_t)((zero0 << 4) | zero1));
            }
        }
    }
}

struct AwqMarlinRouteStorage {
    int32_t *sortedTokenIds = nullptr;
    int32_t *gateExpertIds = nullptr;
    int32_t *downExpertIds = nullptr;
    int32_t *numTokensPadded = nullptr;
    size_t routeCapacity = 0;
    int gateBlock = 0;
    int downBlock = 0;
};

struct AwqMarlinLayerCache {
    bool ready = false;
    bool failed = false;
    std::atomic<bool> retired {false};
    int device = -1;
    int experts = 0;
    int hidden = 0;
    int intermediate = 0;
    int sms = 0;
    int sourceDirectWeights = 0;
    size_t sourceWeightBytes = 0;

    uint32_t *gateWeight = nullptr;
    uint32_t *downWeight = nullptr;
    half *gateScale = nullptr;
    half *downScale = nullptr;
    uint32_t *gateZero = nullptr;
    uint32_t *downZero = nullptr;
    half *gateDecodeScale = nullptr;
    half *downDecodeScale = nullptr;
    uint8_t *gateDecodeZero = nullptr;
    uint8_t *downDecodeZero = nullptr;

    int32_t *expertCounts = nullptr;
    int32_t *expertStarts = nullptr;
    int32_t *expertCursors = nullptr;
    AwqMarlinRouteStorage smallRoutes;
    AwqMarlinRouteStorage largeRoutes;

    int *workspace = nullptr;
    float *temporaryOutput = nullptr;
    std::mutex buildMutex;
    std::mutex runtimeMutex;

    ~AwqMarlinLayerCache();
};

static bool AwqCompactFallbackUnavailable(
        fastllm::Data **weights, int weightsBatch) {
    if (weights == nullptr || weightsBatch < 4) {
        return false;
    }
    for (int slot = 2; slot < weightsBatch; ++slot) {
        const fastllm::Data *weight = weights[slot];
        if (weight != nullptr &&
            weight->dataType == fastllm::DataType::INT4_GROUP &&
            weight->cudaData == nullptr) {
            return true;
        }
    }
    return false;
}

[[noreturn]] static void FailAwqMarlinAfterRepack(
        const char *stage, const AwqMarlinLayerCache *cache = nullptr) {
    cudaError_t state = cudaPeekAtLastError();
    std::string message =
        "AWQ grouped-Marlin " + std::string(stage) +
        " failed after the compact expert weights were released; "
        "the legacy INT4_GROUP fallback is unavailable";
    if (cache != nullptr) {
        message += " (cuda:" + std::to_string(cache->device) +
                   ", H=" + std::to_string(cache->hidden) +
                   ", N=" + std::to_string(cache->intermediate) + ")";
    }
    if (state != cudaSuccess) {
        message += ": ";
        message += cudaGetErrorString(state);
    }
    message += ".";
    std::fprintf(stderr, "[FastLLM] %s\n", message.c_str());
    std::fflush(stderr);
    FastllmCudaSetThreadError();
    throw std::runtime_error(message);
}

static std::mutex &AwqLayerCacheRegistryMutex() {
    static auto *mutex = new std::mutex();
    return *mutex;
}

static std::map<const fastllm::Data *, std::shared_ptr<AwqMarlinLayerCache>> &
AwqLayerCacheRegistry() {
    static auto *registry = new std::map<
        const fastllm::Data *, std::shared_ptr<AwqMarlinLayerCache>>();
    return *registry;
}

static thread_local std::map<
    const fastllm::Data *, std::weak_ptr<AwqMarlinLayerCache>>
    awqLayerCacheFront;

static void ReleaseAwqRouteStorage(AwqMarlinRouteStorage &storage) {
    ReleaseDirect(storage.sortedTokenIds);
    ReleaseDirect(storage.gateExpertIds);
    ReleaseDirect(storage.downExpertIds);
    ReleaseDirect(storage.numTokensPadded);
    storage.sortedTokenIds = nullptr;
    storage.gateExpertIds = nullptr;
    storage.downExpertIds = nullptr;
    storage.numTokensPadded = nullptr;
    storage.routeCapacity = 0;
    storage.gateBlock = 0;
    storage.downBlock = 0;
}

static void ReleaseAwqCacheStorage(AwqMarlinLayerCache &cache) {
    int originalDevice = -1;
    bool restoreDevice = false;
    if (cache.device >= 0 && cudaGetDevice(&originalDevice) == cudaSuccess &&
        originalDevice != cache.device) {
        restoreDevice = cudaSetDevice(cache.device) == cudaSuccess;
    }
    ReleaseDirect(cache.gateWeight);
    ReleaseDirect(cache.downWeight);
    ReleaseDirect(cache.gateScale);
    ReleaseDirect(cache.downScale);
    ReleaseDirect(cache.gateZero);
    ReleaseDirect(cache.downZero);
    ReleaseDirect(cache.gateDecodeScale);
    ReleaseDirect(cache.downDecodeScale);
    ReleaseDirect(cache.gateDecodeZero);
    ReleaseDirect(cache.downDecodeZero);
    ReleaseDirect(cache.expertCounts);
    ReleaseDirect(cache.expertStarts);
    ReleaseDirect(cache.expertCursors);
    ReleaseAwqRouteStorage(cache.smallRoutes);
    ReleaseAwqRouteStorage(cache.largeRoutes);
    ReleaseDirect(cache.workspace);
    ReleaseDirect(cache.temporaryOutput);
    cache.gateWeight = nullptr;
    cache.downWeight = nullptr;
    cache.gateScale = nullptr;
    cache.downScale = nullptr;
    cache.gateZero = nullptr;
    cache.downZero = nullptr;
    cache.gateDecodeScale = nullptr;
    cache.downDecodeScale = nullptr;
    cache.gateDecodeZero = nullptr;
    cache.downDecodeZero = nullptr;
    cache.expertCounts = nullptr;
    cache.expertStarts = nullptr;
    cache.expertCursors = nullptr;
    cache.workspace = nullptr;
    cache.temporaryOutput = nullptr;
    cache.ready = false;
    if (restoreDevice) {
        cudaSetDevice(originalDevice);
    }
}

AwqMarlinLayerCache::~AwqMarlinLayerCache() {
    ReleaseAwqCacheStorage(*this);
}

static bool ValidateAwqWeight(const fastllm::Data *weight,
                              int outputColumns, int inputColumns) {
    return weight != nullptr &&
           weight->dataType == fastllm::DataType::INT4_GROUP &&
           weight->dims.size() == 2 &&
           weight->dims[0] == outputColumns &&
           weight->dims[1] == inputColumns &&
           weight->groupCnt == 32 &&
           weight->group == inputColumns / 32 &&
           weight->scales.size() ==
               (size_t)outputColumns * weight->group &&
           weight->zeros.size() ==
               (size_t)outputColumns * weight->group &&
           weight->cudaData != nullptr &&
           !weight->cudaDataBorrowed;
}

static void ReleaseAwqOriginalWeight(fastllm::Data &weight) {
    // Legacy INT4_GROUP setup intentionally aliases its FP16 scale/min slots
    // to extraCudaData[0/1]. Repacking may happen after that fallback metadata
    // has been initialized, so release unique allocations rather than vector
    // entries; otherwise the aliases reach cudaFree twice.
    std::set<void *> released;
    auto releaseExtra = [&](void *&pointer) {
        if (pointer != nullptr && released.insert(pointer).second) {
            FastllmCudaFree(pointer);
        }
        pointer = nullptr;
    };
    for (void *&pointer : weight.extraCudaData) {
        releaseExtra(pointer);
    }
    for (void *&pointer : weight.extraCudaHalfData) {
        releaseExtra(pointer);
    }
    if (weight.cudaData != nullptr) {
        if (weight.directMemory) {
            FastllmCudaDirectFree(weight.cudaData);
        } else {
            FastllmCudaFree(weight.cudaData);
        }
        weight.cudaData = nullptr;
        weight.cudaDataBorrowed = false;
    }
}

static bool BuildAwqLayerCache(fastllm::Data **weights, int weightsBatch,
                               AwqMarlinLayerCache &cache) {
    if (weights == nullptr || weightsBatch < 4 ||
        (weightsBatch & 1) != 0 || weights[0] != nullptr ||
        weights[1] != nullptr || weights[2] == nullptr ||
        weights[3] == nullptr || weights[2]->dims.size() != 2 ||
        weights[3]->dims.size() != 2) {
        return false;
    }

    const int experts = weightsBatch / 2 - 1;
    const int gateRows = weights[2]->dims[0];
    const int hidden = weights[2]->dims[1];
    const int intermediate = gateRows / 2;
    if (experts <= 0 || gateRows <= 0 || hidden <= 0 ||
        (gateRows & 1) != 0 || gateRows % 256 != 0 ||
        hidden % 128 != 0 || intermediate % 64 != 0 ||
        weights[3]->dims[0] != hidden ||
        weights[3]->dims[1] != intermediate) {
        return false;
    }
    for (int expert = 0; expert < experts; ++expert) {
        int slot = 2 + expert * 2;
        if (!ValidateAwqWeight(weights[slot], gateRows, hidden) ||
            !ValidateAwqWeight(weights[slot + 1], hidden, intermediate)) {
            return false;
        }
    }

    int device = 0;
    int sms = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) !=
            cudaSuccess ||
        sms <= 0 || !PrepareAwqMarlinMoeKernels(device)) {
        return false;
    }

    const size_t gateWeightBytes =
        (size_t)gateRows * hidden / 2;
    const size_t downWeightBytes =
        (size_t)hidden * intermediate / 2;
    const size_t gateScaleValues =
        (size_t)gateRows * (hidden / 32);
    const size_t downScaleValues =
        (size_t)hidden * (intermediate / 32);
    const size_t gateZeroWords = gateScaleValues / 8;
    const size_t downZeroWords = downScaleValues / 8;
    const size_t temporaryFloats =
        (size_t)sms * 4 * 8 * 256 * 2;
    const size_t persistentBytes =
        (gateWeightBytes + downWeightBytes) * experts +
        (gateScaleValues + downScaleValues) * experts * sizeof(half) +
        (gateZeroWords + downZeroWords) * experts * sizeof(uint32_t) +
        (gateScaleValues + downScaleValues) * experts * sizeof(half) +
        (gateScaleValues + downScaleValues) * experts / 2 +
        temporaryFloats * sizeof(float) +
        (size_t)sms * 4 * sizeof(int) +
        (size_t)experts * 3 * sizeof(int32_t);
    size_t freeMemory = 0;
    size_t totalMemory = 0;
    constexpr size_t reserveBytes = (size_t)1024 << 20;
    if (cudaMemGetInfo(&freeMemory, &totalMemory) != cudaSuccess ||
        freeMemory < persistentBytes + reserveBytes) {
        static std::atomic<unsigned long long> warnedDevices {0};
        unsigned long long bit =
            device < 64 ? (1ull << device) : 0;
        if (bit == 0 ||
            (warnedDevices.fetch_or(bit, std::memory_order_relaxed) & bit) ==
                0) {
            std::fprintf(
                stderr,
                "[FastLLM] AWQ grouped-Marlin prefill needs %.1f MiB for "
                "this layer but only %.1f MiB is free on GPU %d; "
                "keeping the fallback path.\n",
                persistentBytes / 1048576.0, freeMemory / 1048576.0,
                device);
            std::fflush(stderr);
        }
        return false;
    }

    cache.device = device;
    cache.experts = experts;
    cache.hidden = hidden;
    cache.intermediate = intermediate;
    cache.sms = sms;
    cache.gateWeight = (uint32_t *)AllocateDirect(
        gateWeightBytes * experts);
    cache.downWeight = (uint32_t *)AllocateDirect(
        downWeightBytes * experts);
    cache.gateScale = (half *)AllocateDirect(
        gateScaleValues * experts * sizeof(half));
    cache.downScale = (half *)AllocateDirect(
        downScaleValues * experts * sizeof(half));
    cache.gateZero = (uint32_t *)AllocateDirect(
        gateZeroWords * experts * sizeof(uint32_t));
    cache.downZero = (uint32_t *)AllocateDirect(
        downZeroWords * experts * sizeof(uint32_t));
    cache.gateDecodeScale = (half *)AllocateDirect(
        gateScaleValues * experts * sizeof(half));
    cache.downDecodeScale = (half *)AllocateDirect(
        downScaleValues * experts * sizeof(half));
    cache.gateDecodeZero = (uint8_t *)AllocateDirect(
        gateScaleValues * experts / 2);
    cache.downDecodeZero = (uint8_t *)AllocateDirect(
        downScaleValues * experts / 2);
    cache.expertCounts = (int32_t *)AllocateDirect(
        (size_t)experts * sizeof(int32_t));
    cache.expertStarts = (int32_t *)AllocateDirect(
        (size_t)experts * sizeof(int32_t));
    cache.expertCursors = (int32_t *)AllocateDirect(
        (size_t)experts * sizeof(int32_t));
    cache.workspace = (int *)AllocateDirect(
        (size_t)sms * 4 * sizeof(int));
    cache.temporaryOutput = (float *)AllocateDirect(
        temporaryFloats * sizeof(float));
    if (cache.gateWeight == nullptr || cache.downWeight == nullptr ||
        cache.gateScale == nullptr || cache.downScale == nullptr ||
        cache.gateZero == nullptr || cache.downZero == nullptr ||
        cache.gateDecodeScale == nullptr ||
        cache.downDecodeScale == nullptr ||
        cache.gateDecodeZero == nullptr ||
        cache.downDecodeZero == nullptr ||
        cache.expertCounts == nullptr || cache.expertStarts == nullptr ||
        cache.expertCursors == nullptr || cache.workspace == nullptr ||
        cache.temporaryOutput == nullptr) {
        ReleaseAwqCacheStorage(cache);
        return false;
    }

    const size_t gateQWeightCount =
        (size_t)(hidden / 8) * gateRows;
    const size_t downQWeightCount =
        (size_t)(intermediate / 8) * hidden;
    const size_t temporaryQWeightCount =
        std::max(gateQWeightCount, downQWeightCount);
    uint32_t *standardQWeight = (uint32_t *)AllocateDirect(
        temporaryQWeightCount * sizeof(uint32_t));
    if (standardQWeight == nullptr) {
        ReleaseAwqCacheStorage(cache);
        return false;
    }

    std::vector<half> gateScales;
    std::vector<half> downScales;
    std::vector<uint32_t> gateZeros;
    std::vector<uint32_t> downZeros;
    std::vector<half> gateDecodeScales;
    std::vector<half> downDecodeScales;
    std::vector<uint8_t> gateDecodeZeros;
    std::vector<uint8_t> downDecodeZeros;
    gateScales.reserve(gateScaleValues * experts);
    downScales.reserve(downScaleValues * experts);
    gateZeros.reserve(gateZeroWords * experts);
    downZeros.reserve(downZeroWords * experts);
    gateDecodeScales.reserve(gateScaleValues * experts);
    downDecodeScales.reserve(downScaleValues * experts);
    gateDecodeZeros.reserve(gateScaleValues * experts / 2);
    downDecodeZeros.reserve(downScaleValues * experts / 2);

    cudaStream_t stream = cudaStreamPerThread;
    constexpr int threads = 256;
    bool success =
        cudaMemsetAsync(cache.workspace, 0,
                        (size_t)sms * 4 * sizeof(int), stream) ==
        cudaSuccess;
    for (int expert = 0; expert < experts && success; ++expert) {
        fastllm::Data &gate = *weights[2 + expert * 2];
        fastllm::Data &down = *weights[3 + expert * 2];

        AwqInt4GroupToMarlinQWeightKernel<<<
            (gateQWeightCount + threads - 1) / threads, threads, 0,
            stream>>>((const uint8_t *)gate.cudaData, standardQWeight,
                       hidden, gateRows);
        success = cudaGetLastError() == cudaSuccess &&
                  FastllmCudaGptqMarlinRepackStream(
                      standardQWeight,
                      cache.gateWeight + gateQWeightCount * expert,
                      hidden, gateRows, (void *)stream);
        if (!success) {
            break;
        }
        AppendAwqMarlinScalesAndZeros(
            gate, gateScales, gateZeros, gateRows);
        if (hidden == 2048 && intermediate == 256) {
            AppendAwqPairedDecodeScalesAndZeros(
                gate, gateDecodeScales, gateDecodeZeros);
        } else {
            AppendAwqDecodeScalesAndZeros(
                gate, gateDecodeScales, gateDecodeZeros);
        }

        AwqInt4GroupToMarlinQWeightKernel<<<
            (downQWeightCount + threads - 1) / threads, threads, 0,
            stream>>>((const uint8_t *)down.cudaData, standardQWeight,
                       intermediate, hidden);
        success = cudaGetLastError() == cudaSuccess &&
                  FastllmCudaGptqMarlinRepackStream(
                      standardQWeight,
                      cache.downWeight + downQWeightCount * expert,
                      intermediate, hidden, (void *)stream);
        if (!success) {
            break;
        }
        AppendAwqMarlinScalesAndZeros(
            down, downScales, downZeros, hidden);
        if (hidden == 2048 && intermediate == 256) {
            AppendAwqPairedDecodeScalesAndZeros(
                down, downDecodeScales, downDecodeZeros);
        } else {
            AppendAwqDecodeScalesAndZeros(
                down, downDecodeScales, downDecodeZeros);
        }
    }

    if (success) {
        success =
            gateScales.size() == gateScaleValues * experts &&
            downScales.size() == downScaleValues * experts &&
            gateZeros.size() == gateZeroWords * experts &&
            downZeros.size() == downZeroWords * experts &&
            gateDecodeScales.size() == gateScaleValues * experts &&
            downDecodeScales.size() == downScaleValues * experts &&
            gateDecodeZeros.size() == gateScaleValues * experts / 2 &&
            downDecodeZeros.size() == downScaleValues * experts / 2;
    }
    if (success) {
        success =
            cudaMemcpyAsync(cache.gateScale, gateScales.data(),
                            gateScales.size() * sizeof(half),
                            cudaMemcpyHostToDevice, stream) == cudaSuccess &&
            cudaMemcpyAsync(cache.downScale, downScales.data(),
                            downScales.size() * sizeof(half),
                            cudaMemcpyHostToDevice, stream) == cudaSuccess &&
            cudaMemcpyAsync(cache.gateZero, gateZeros.data(),
                            gateZeros.size() * sizeof(uint32_t),
                            cudaMemcpyHostToDevice, stream) == cudaSuccess &&
            cudaMemcpyAsync(cache.downZero, downZeros.data(),
                            downZeros.size() * sizeof(uint32_t),
                            cudaMemcpyHostToDevice, stream) == cudaSuccess &&
            cudaMemcpyAsync(
                cache.gateDecodeScale, gateDecodeScales.data(),
                gateDecodeScales.size() * sizeof(half),
                cudaMemcpyHostToDevice, stream) == cudaSuccess &&
            cudaMemcpyAsync(
                cache.downDecodeScale, downDecodeScales.data(),
                downDecodeScales.size() * sizeof(half),
                cudaMemcpyHostToDevice, stream) == cudaSuccess &&
            cudaMemcpyAsync(
                cache.gateDecodeZero, gateDecodeZeros.data(),
                gateDecodeZeros.size() * sizeof(uint8_t),
                cudaMemcpyHostToDevice, stream) == cudaSuccess &&
            cudaMemcpyAsync(
                cache.downDecodeZero, downDecodeZeros.data(),
                downDecodeZeros.size() * sizeof(uint8_t),
                            cudaMemcpyHostToDevice, stream) == cudaSuccess;
    }
    if (success) {
        success = cudaStreamSynchronize(stream) == cudaSuccess;
    }
    ReleaseDirect(standardQWeight);
    if (!success) {
        ReleaseAwqCacheStorage(cache);
        return false;
    }

    // The grouped Marlin layout is now the canonical expert representation
    // for both prefill and decode. Drop compact AWQ weights and their old
    // pointer/quant-metadata caches so resident GPU memory stays single-copy.
    for (int expert = 0; expert < experts; ++expert) {
        fastllm::Data *gate = weights[2 + expert * 2];
        fastllm::Data *down = weights[3 + expert * 2];
        cache.sourceDirectWeights += gate->directMemory ? 1 : 0;
        cache.sourceDirectWeights += down->directMemory ? 1 : 0;
        cache.sourceWeightBytes += gate->GetBytes() + down->GetBytes();
        ReleaseAwqOriginalWeight(*weights[2 + expert * 2]);
        ReleaseAwqOriginalWeight(*weights[3 + expert * 2]);
    }

    cache.ready = true;
    static std::atomic<unsigned long long> announcedDevices {0};
    unsigned long long bit = device < 64 ? (1ull << device) : 0;
    if (bit == 0 ||
        (announcedDevices.fetch_or(bit, std::memory_order_relaxed) & bit) ==
            0) {
        std::fprintf(
            stderr,
            "[FastLLM] replaced compact AWQ-MoE weights with grouped Marlin "
            "on GPU %d (%d experts, H=%d, N=%d; prefill and decode; "
            "released %.1f MiB, direct %d/%d).\n",
            device, experts, hidden, intermediate,
            cache.sourceWeightBytes / 1048576.0,
            cache.sourceDirectWeights, experts * 2);
        std::fflush(stderr);
    }
    return true;
}

static AwqMarlinRouteStorage *EnsureAwqRuntimeCapacity(
        AwqMarlinLayerCache &cache, size_t routes, bool smallBatch,
        int gateBlock, int downBlock) {
    AwqMarlinRouteStorage &storage =
        smallBatch ? cache.smallRoutes : cache.largeRoutes;
    if (routes <= storage.routeCapacity &&
        storage.gateBlock == gateBlock &&
        storage.downBlock == downBlock) {
        return &storage;
    }
    std::lock_guard<std::mutex> lock(cache.runtimeMutex);
    if (routes <= storage.routeCapacity &&
        storage.gateBlock == gateBlock &&
        storage.downBlock == downBlock) {
        return &storage;
    }
    if (cudaStreamSynchronize(cudaStreamPerThread) != cudaSuccess) {
        return nullptr;
    }
    ReleaseAwqRouteStorage(storage);
    // Keep the graph-captured decode/small-batch metadata stable even if a
    // later request uses a larger prefill buffer.
    size_t routeCapacity = smallBatch ? std::max<size_t>(routes, 64 * 8) : routes;
    size_t paddedCapacity =
        routeCapacity + (size_t)cache.experts * (gateBlock - 1);
    size_t gateBlocks = (paddedCapacity + gateBlock - 1) / gateBlock;
    size_t downBlocks = (paddedCapacity + downBlock - 1) / downBlock;
    storage.sortedTokenIds = (int32_t *)AllocateDirect(
        paddedCapacity * sizeof(int32_t));
    storage.gateExpertIds = (int32_t *)AllocateDirect(
        gateBlocks * sizeof(int32_t));
    storage.downExpertIds = (int32_t *)AllocateDirect(
        downBlocks * sizeof(int32_t));
    storage.numTokensPadded =
        (int32_t *)AllocateDirect(sizeof(int32_t));
    if (storage.sortedTokenIds == nullptr ||
        storage.gateExpertIds == nullptr ||
        storage.downExpertIds == nullptr ||
        storage.numTokensPadded == nullptr) {
        ReleaseAwqRouteStorage(storage);
        return nullptr;
    }
    storage.routeCapacity = routeCapacity;
    storage.gateBlock = gateBlock;
    storage.downBlock = downBlock;
    return &storage;
}

static std::shared_ptr<AwqMarlinLayerCache> GetOrBuildAwqLayerCache(
        fastllm::Data **weights, int weightsBatch) {
    if (weights == nullptr || weightsBatch < 4 || weights[2] == nullptr) {
        return {};
    }
    const fastllm::Data *key = weights[2];
    auto local = awqLayerCacheFront.find(key);
    if (local != awqLayerCacheFront.end()) {
        std::shared_ptr<AwqMarlinLayerCache> cached = local->second.lock();
        if (cached != nullptr &&
            !cached->retired.load(std::memory_order_acquire) &&
            cached->ready && !cached->failed) {
            return cached;
        }
        awqLayerCacheFront.erase(local);
    }

    std::shared_ptr<AwqMarlinLayerCache> cache;
    {
        std::lock_guard<std::mutex> lock(AwqLayerCacheRegistryMutex());
        auto &registry = AwqLayerCacheRegistry();
        auto it = registry.find(key);
        if (it == registry.end()) {
            cache = std::make_shared<AwqMarlinLayerCache>();
            registry.emplace(key, cache);
        } else {
            cache = it->second;
        }
    }
    {
        std::lock_guard<std::mutex> lock(cache->buildMutex);
        if (cache->retired.load(std::memory_order_acquire) ||
            cache->failed) {
            return {};
        }
        if (!cache->ready &&
            !BuildAwqLayerCache(weights, weightsBatch, *cache)) {
            cache->failed = true;
            return {};
        }
    }
    awqLayerCacheFront[key] = cache;
    return cache;
}

static void ReleaseAwqLayerCache(const fastllm::Data *key) {
    if (key == nullptr) {
        return;
    }
    std::shared_ptr<AwqMarlinLayerCache> retired;
    {
        std::lock_guard<std::mutex> lock(AwqLayerCacheRegistryMutex());
        auto &registry = AwqLayerCacheRegistry();
        auto it = registry.find(key);
        if (it == registry.end()) {
            return;
        }
        it->second->retired.store(true, std::memory_order_release);
        retired = std::move(it->second);
        registry.erase(it);
    }
}

static bool RunAwqMarlinMoe(
        const fastllm::Data &input, fastllm::Data &gateOutput,
        fastllm::Data &activation, fastllm::Data &output,
        fastllm::Data **weights, int weightsBatch,
        const int32_t *indices, const float *scores,
        int batch, int topk) {
    if (input.dataDevice != fastllm::DataDevice::CUDA ||
        input.dataType != fastllm::DataType::FLOAT16 ||
        input.dims.size() != 2 || input.dims[0] != batch ||
        input.cudaData == nullptr || indices == nullptr || scores == nullptr ||
        batch <= 0 || topk <= 0 || topk > 8) {
        if (AwqCompactFallbackUnavailable(weights, weightsBatch)) {
            FailAwqMarlinAfterRepack("received an incompatible invocation");
        }
        return false;
    }
    std::shared_ptr<AwqMarlinLayerCache> cache =
        GetOrBuildAwqLayerCache(weights, weightsBatch);
    if (cache == nullptr || !cache->ready) {
        if (AwqCompactFallbackUnavailable(weights, weightsBatch)) {
            FailAwqMarlinAfterRepack("could not recover its canonical cache");
        }
        return false;
    }
    if (input.dims[1] != cache->hidden) {
        FailAwqMarlinAfterRepack("received an incompatible hidden size",
                                 cache.get());
    }

    if (batch == 1) {
        activation.dataType = fastllm::DataType::FLOAT16;
        activation.dataDevice = fastllm::DataDevice::CUDA;
        activation.dataDeviceIds = input.dataDeviceIds;
        activation.Resize({topk, cache->intermediate});
        activation.Allocate(false);
        output.dataType = fastllm::DataType::FLOAT16;
        output.dataDevice = fastllm::DataDevice::CUDA;
        output.dataDeviceIds = input.dataDeviceIds;
        output.Resize(input.dims);
        output.Allocate(false);
        if (activation.cudaData == nullptr || output.cudaData == nullptr) {
            FailAwqMarlinAfterRepack("decode buffer allocation",
                                     cache.get());
        }

        if (cache->hidden == 2048 && cache->intermediate == 256) {
            AwqMarlinMoeDecodeGateupH2048N256Top8Kernel
                <<< topk * (256 / 16), 8 * 32,
                    0, cudaStreamPerThread >>>(
                    (const half *)input.cudaData, cache->gateWeight,
                    cache->gateDecodeScale, cache->gateDecodeZero,
                    indices, (half *)activation.cudaData,
                    cache->experts);
        } else {
            constexpr int warps = 8;
            int tasks = topk * cache->intermediate;
            AwqMarlinMoeDecodeGateupSwigluKernel<warps>
                <<< (tasks + warps - 1) / warps, warps * 32,
                    0, cudaStreamPerThread >>>(
                    (const half *)input.cudaData, cache->gateWeight,
                    cache->gateDecodeScale, cache->gateDecodeZero, indices,
                    (half *)activation.cudaData, topk, cache->experts,
                    cache->hidden, cache->intermediate);
        }
        if (cache->hidden == 2048 && cache->intermediate == 256) {
            AwqMarlinMoeDecodeDownH2048N256Top8Kernel
                <<< 2048 / 16, 16 * 32, 0, cudaStreamPerThread >>>(
                    (const half *)activation.cudaData, cache->downWeight,
                    cache->downDecodeScale, cache->downDecodeZero,
                    indices, scores, (half *)output.cudaData,
                    topk, cache->experts);
        } else if (cache->intermediate == 256) {
            AwqMarlinMoeDecodeDownKernel<8, 16>
                <<< cache->hidden, 8 * 16, 0, cudaStreamPerThread >>>(
                    (const half *)activation.cudaData, cache->downWeight,
                    cache->downDecodeScale, cache->downDecodeZero,
                    indices, scores,
                    (half *)output.cudaData, topk, cache->experts,
                    cache->hidden, cache->intermediate);
        } else {
            AwqMarlinMoeDecodeDownKernel<8, 32>
                <<< cache->hidden, 8 * 32, 0, cudaStreamPerThread >>>(
                    (const half *)activation.cudaData, cache->downWeight,
                    cache->downDecodeScale, cache->downDecodeZero,
                    indices, scores,
                    (half *)output.cudaData, topk, cache->experts,
                    cache->hidden, cache->intermediate);
        }
        if (cudaPeekAtLastError() != cudaSuccess) {
            FailAwqMarlinAfterRepack("decode kernel launch", cache.get());
        }
        return true;
    }

    const int routes = batch * topk;
    const bool smallBatch = batch <= 64;
    const int gateBlock = smallBatch ? 8 : 64;
    const int downBlock = smallBatch ? 8 : 32;
    AwqMarlinRouteStorage *routeStorage =
        EnsureAwqRuntimeCapacity(
            *cache, routes, smallBatch, gateBlock, downBlock);
    if (routeStorage == nullptr) {
        FailAwqMarlinAfterRepack("route-buffer allocation", cache.get());
    }

    gateOutput.dataType = fastllm::DataType::FLOAT16;
    gateOutput.dataDevice = fastllm::DataDevice::CUDA;
    gateOutput.dataDeviceIds = input.dataDeviceIds;
    gateOutput.Resize(
        {routes, std::max(cache->hidden, cache->intermediate * 2)});
    gateOutput.Allocate(false);
    activation.dataType = fastllm::DataType::FLOAT16;
    activation.dataDevice = fastllm::DataDevice::CUDA;
    activation.dataDeviceIds = input.dataDeviceIds;
    activation.Resize({routes, cache->intermediate});
    activation.Allocate(false);
    output.dataType = fastllm::DataType::FLOAT16;
    output.dataDevice = fastllm::DataDevice::CUDA;
    output.dataDeviceIds = input.dataDeviceIds;
    output.Resize(input.dims);
    output.Allocate(false);
    if (gateOutput.cudaData == nullptr || activation.cudaData == nullptr ||
        output.cudaData == nullptr) {
        FailAwqMarlinAfterRepack("prefill buffer allocation", cache.get());
    }

    cudaStream_t stream = cudaStreamPerThread;
    constexpr int threads = 256;
    InitAwqMoeRouteCountersKernel<<<
        (cache->experts + threads - 1) / threads, threads, 0, stream>>>(
        cache->expertCounts, cache->expertCursors, cache->experts);
    CountAwqMoeRoutesKernel<<<
        (routes + threads - 1) / threads, threads, 0, stream>>>(
        indices, cache->expertCounts, routes, cache->experts);
    BuildAwqMoeRouteOffsetsKernel<<<1, threads, 0, stream>>>(
        cache->expertCounts, cache->expertStarts,
        routeStorage->sortedTokenIds, routeStorage->gateExpertIds,
        routeStorage->downExpertIds, routeStorage->numTokensPadded,
        routes, cache->experts, gateBlock, downBlock);
    ScatterAwqMoeRoutesKernel<<<
        (routes + threads - 1) / threads, threads, 0, stream>>>(
        indices, cache->expertStarts, cache->expertCursors,
        routeStorage->sortedTokenIds, routes, cache->experts);
    if (cudaPeekAtLastError() != cudaSuccess) {
        FailAwqMarlinAfterRepack("route-metadata launch", cache.get());
    }

    if (!LaunchAwqMarlinMoe(
            true, smallBatch, input.cudaData, cache->gateWeight,
            gateOutput.cudaData, cache->temporaryOutput,
            cache->gateScale, cache->gateZero,
            routeStorage->sortedTokenIds, routeStorage->gateExpertIds,
            routeStorage->numTokensPadded, scores,
            topk, false, batch, cache->intermediate * 2,
            cache->hidden, cache->workspace, stream, cache->sms)) {
        FailAwqMarlinAfterRepack("gate/up Marlin launch", cache.get());
    }
    int activationElements = routes * cache->intermediate;
    SwigluRowsKernel<<<
        (activationElements + threads - 1) / threads, threads, 0, stream>>>(
        (const half *)gateOutput.cudaData,
        (half *)activation.cudaData, routes, cache->intermediate);
    if (cudaPeekAtLastError() != cudaSuccess) {
        FailAwqMarlinAfterRepack("SwiGLU launch", cache.get());
    }

    if (!LaunchAwqMarlinMoe(
            false, smallBatch, activation.cudaData, cache->downWeight,
            gateOutput.cudaData, cache->temporaryOutput,
            cache->downScale, cache->downZero,
            routeStorage->sortedTokenIds, routeStorage->downExpertIds,
            routeStorage->numTokensPadded, scores,
            1, true, routes, cache->hidden, cache->intermediate,
            cache->workspace, stream, cache->sms)) {
        FailAwqMarlinAfterRepack("down Marlin launch", cache.get());
    }
    int outputPairs = batch * (cache->hidden / 2);
    SumAwqMoeRowsKernel<<<
        (outputPairs + threads - 1) / threads, threads, 0, stream>>>(
        (const half *)gateOutput.cudaData, (half *)output.cudaData,
        batch, topk, cache->hidden);
    if (cudaPeekAtLastError() != cudaSuccess) {
        FailAwqMarlinAfterRepack("output-reduction launch", cache.get());
    }
    return true;
}

template <typename T>
static bool RunMarlinMoe(const fastllm::Data &input, fastllm::Data &w1,
                             fastllm::Data &output, fastllm::Data **weights,
                             int weightsBatch, const int32_t *globalIndices,
                             const float *scores, int topk, int ownerRank,
                             int ownerCount) {
    if (globalIndices == nullptr || scores == nullptr || topk <= 0 ||
        topk > 16 || ownerCount <= 0 ||
        input.dims.empty() || input.dataDevice != fastllm::DataDevice::CUDA) {
        return false;
    }
    // The in-tree MXFP4 instantiations intentionally cover DeepSeek V4's BF16
    // path only. Other data types retain the native compact-kernel fallback.
    if (!std::is_same<T, __nv_bfloat16>::value) {
        return false;
    }

    std::shared_ptr<MarlinLayerCache> cache =
        GetOrBuildLayerCache(weights, weightsBatch, topk);
    if (cache == nullptr || !cache->ready || input.dims.back() != cache->hidden) {
        return false;
    }

    w1.dataDevice = input.dataDevice;
    w1.dataDeviceIds = input.dataDeviceIds;
    w1.dataType = input.dataType;
    int rows = (int)(input.Count(0) / cache->hidden);
    if (rows <= 0 || (size_t)rows * cache->hidden != input.Count(0)) {
        return false;
    }

    const bool groupedRows =
        rows > 1 && rows <= MarlinLayerCache::kSmallBatchRows;
    const int scratchRoutes = groupedRows ? rows * topk : topk;
    // The grouped DSpark path keeps all route activations resident at once;
    // larger/general inputs retain the established serial-row fallback.
    w1.Resize({scratchRoutes, cache->intermediate});
    w1.Allocate(false);
    output.dataDevice = input.dataDevice;
    output.dataDeviceIds = input.dataDeviceIds;
    output.dataType = input.dataType;
    output.Resize(input.dims);
    output.Allocate(false);

    T *activation = (T *)w1.cudaData;
    T *finalOutput = (T *)output.cudaData;
    if (activation == nullptr || finalOutput == nullptr) {
        return false;
    }

    auto checkStage = [&](const char *stage) {
        cudaError_t state = cudaGetLastError();
        if (state != cudaSuccess) {
            std::fprintf(stderr,
                         "[FastLLM] Marlin MoE failed after %s on GPU "
                         "%d: %s\n",
                         stage, cache->device, cudaGetErrorString(state));
            std::fflush(stderr);
            return false;
        }
        return true;
    };
    int gateGroups = cache->hidden / 32;
    int downGroups = cache->intermediate / 32;
    constexpr int threads = 256;
    // Match the two configurations selected in the profiled vLLM TP8 decode
    // graph. Supplying them directly also removes launcher-side search in
    // eager execution.
    constexpr int gateThreadK = 128;
    constexpr int gateThreadN = 128;
    constexpr int gateBlocksPerSm = 1;
    constexpr int downThreadK = 64;
    constexpr int downThreadN = 128;
    constexpr int downBlocksPerSm = 3;

    if (groupedRows) {
        const int routes = rows * topk;
        BuildEpMetadataRowsKernel<<<1, 1, 0, cudaStreamPerThread>>>(
            globalIndices, cache->sortedTokenIds, cache->expertIds,
            cache->numTokensPadded, rows, topk, ownerRank, ownerCount,
            cache->experts);
        if (!checkStage("grouped EP metadata")) {
            return false;
        }

        if (!LaunchMarlinMoe(
                input.cudaData, cache->gateWeight, cache->gateOutput,
                cache->temporaryOutput, cache->gateScale,
                cache->sortedTokenIds, cache->expertIds,
                cache->numTokensPadded, scores, topk, false, gateGroups,
                rows, cache->intermediate * 2, cache->hidden,
                cache->workspace, cudaStreamPerThread, gateThreadK,
                gateThreadN, cache->sms, gateBlocksPerSm) ||
            !checkStage("grouped gate/up GEMM")) {
            return false;
        }

        int actCount = routes * cache->intermediate;
        SwigluRowsKernel<<<(actCount + threads - 1) / threads, threads, 0,
                           cudaStreamPerThread>>>(
            (const T *)cache->gateOutput, activation, routes,
            cache->intermediate);
        if (!checkStage("grouped SwiGLU")) {
            return false;
        }

        if (!LaunchMarlinMoe(
                activation, cache->downWeight, cache->downOutput,
                cache->temporaryOutput, cache->downScale,
                cache->sortedTokenIds, cache->expertIds,
                cache->numTokensPadded, scores, 1, true, downGroups,
                routes, cache->hidden, cache->intermediate,
                cache->workspace, cudaStreamPerThread, downThreadK,
                downThreadN, cache->sms, downBlocksPerSm) ||
            !checkStage("grouped down GEMM")) {
            return false;
        }

        int outputCount = rows * cache->hidden;
        ReduceEpRowsKernel<<<(outputCount + threads - 1) / threads, threads,
                             0, cudaStreamPerThread>>>(
            (const T *)cache->downOutput, globalIndices, finalOutput, rows,
            topk, ownerRank, ownerCount, cache->hidden);
        return checkStage("grouped EP row reduction");
    }

    int actCount = topk * cache->intermediate;
    for (int row = 0; row < rows; ++row) {
        const T *rowInput = (const T *)input.cudaData +
                            (size_t)row * cache->hidden;
        const int32_t *rowIndices = globalIndices + (size_t)row * topk;
        const float *rowScores = scores + (size_t)row * topk;
        T *rowOutput = finalOutput + (size_t)row * cache->hidden;

        BuildEpMetadataKernel<<<1, 1, 0, cudaStreamPerThread>>>(
            rowIndices, cache->sortedTokenIds, cache->expertIds,
            cache->numTokensPadded, topk, ownerRank, ownerCount,
            cache->experts);
        if (!checkStage("EP metadata")) {
            return false;
        }

        if (!LaunchMarlinMoe(
            rowInput, cache->gateWeight, cache->gateOutput,
            cache->temporaryOutput, cache->gateScale,
            cache->sortedTokenIds, cache->expertIds, cache->numTokensPadded,
            rowScores, topk, false, gateGroups, 1,
            cache->intermediate * 2, cache->hidden, cache->workspace,
            cudaStreamPerThread, gateThreadK, gateThreadN, cache->sms,
            gateBlocksPerSm)) {
            std::fprintf(stderr,
                         "[FastLLM] Marlin MoE gate/up launch configuration "
                         "is invalid on GPU %d.\n",
                         cache->device);
            std::fflush(stderr);
            return false;
        }
        if (!checkStage("gate/up GEMM")) {
            return false;
        }

        SwigluRowsKernel<<<(actCount + threads - 1) / threads, threads, 0,
                           cudaStreamPerThread>>>(
            (const T *)cache->gateOutput, activation, topk,
            cache->intermediate);
        if (!checkStage("SwiGLU")) {
            return false;
        }

        if (!LaunchMarlinMoe(
            activation, cache->downWeight, cache->downOutput,
            cache->temporaryOutput, cache->downScale,
            cache->sortedTokenIds, cache->expertIds, cache->numTokensPadded,
            rowScores, 1, true, downGroups, topk, cache->hidden,
            cache->intermediate, cache->workspace, cudaStreamPerThread,
            downThreadK, downThreadN, cache->sms, downBlocksPerSm)) {
            std::fprintf(stderr,
                         "[FastLLM] Marlin MoE down launch configuration is "
                         "invalid on GPU %d.\n",
                         cache->device);
            std::fflush(stderr);
            return false;
        }
        if (!checkStage("down GEMM")) {
            return false;
        }

        ReduceEpRowsKernel<<<(cache->hidden + threads - 1) / threads, threads,
                             0, cudaStreamPerThread>>>(
            (const T *)cache->downOutput, rowIndices, rowOutput, 1, topk,
            ownerRank, ownerCount, cache->hidden);
        if (!checkStage("EP row reduction")) {
            return false;
        }

        // Every metadata update and Marlin launch above is submitted to the
        // same per-thread stream. CUDA stream ordering therefore protects the
        // reused scratch from the next row without a host synchronization.
        // Keeping this loop asynchronous is also required when verification
        // (typically eight rows for DSpark-7) is captured into a CUDA graph.
    }
    return true;
}

}  // namespace fastllm_marlin_moe

void FastllmCudaReleaseMergeMOEVllmMarlinCache(
        const fastllm::Data *layerKey) {
    fastllm_marlin_moe::ReleaseLayerCache(layerKey);
    fastllm_marlin_moe::ReleaseAwqLayerCache(layerKey);
}

bool FastllmCudaHalfMergeMOEInt4GroupMarlinIndexed(
        const fastllm::Data &input, fastllm::Data &gateOutput,
        fastllm::Data &activation, fastllm::Data &output,
        fastllm::Data **weights, int weightsBatch,
        const int32_t *indices, const float *scores,
        int batch, int topk) {
    return fastllm_marlin_moe::RunAwqMarlinMoe(
        input, gateOutput, activation, output, weights, weightsBatch,
        indices, scores, batch, topk);
}

bool FastllmCudaHalfMergeMOEVllmMarlinBatch1ExpertParallel(
        const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
        fastllm::Data **weights, int weightsBatch, const int32_t *globalIndices,
        const float *scores, int topk, int ownerRank, int ownerCount) {
    return fastllm_marlin_moe::RunMarlinMoe<half>(
        input, w1, output, weights, weightsBatch, globalIndices, scores, topk,
        ownerRank, ownerCount);
}

bool FastllmCudaBFloat16MergeMOEVllmMarlinBatch1ExpertParallel(
        const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &output,
        fastllm::Data **weights, int weightsBatch, const int32_t *globalIndices,
        const float *scores, int topk, int ownerRank, int ownerCount) {
    return fastllm_marlin_moe::RunMarlinMoe<__nv_bfloat16>(
        input, w1, output, weights, weightsBatch, globalIndices, scores, topk,
        ownerRank, ownerCount);
}
