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
__global__ void ReduceEpRowsKernel(const T *__restrict__ rows,
                                   const int32_t *__restrict__ globalIndices,
                                   T *__restrict__ output, int topk,
                                   int ownerRank, int ownerCount, int hidden) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= hidden) {
        return;
    }
    float value = 0.0f;
    if (ownerRank >= 0 && ownerCount > 0) {
        for (int slot = 0; slot < topk; ++slot) {
            int expert = globalIndices[slot];
            if (expert >= 0 && expert % ownerCount == ownerRank) {
                value += ToFloat(rows[(size_t)slot * hidden + col]);
            }
        }
    }
    output[col] = FromFloat<T>(value);
}

struct MarlinLayerCache {
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
    cache.sortedTokenIds = (int32_t *)AllocateDirect((size_t)topk * 8 * sizeof(int32_t));
    cache.expertIds = (int32_t *)AllocateDirect((size_t)topk * sizeof(int32_t));
    cache.numTokensPadded = (int32_t *)AllocateDirect(sizeof(int32_t));
    cache.workspace = (int *)AllocateDirect((size_t)sms * 4 * sizeof(int));
    size_t temporaryFloats = (size_t)sms * 4 * 8 * 256 * 2;
    cache.temporaryOutput = (float *)AllocateDirect(temporaryFloats * sizeof(float));
    cache.gateOutput = AllocateDirect((size_t)topk * intermediate * 2 * scalarBytes);
    cache.downOutput = AllocateDirect((size_t)topk * hidden * scalarBytes);
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

    // The Marlin scratch is reused serially for prefill rows.  Decode has a
    // single row and therefore follows the exact same graph-capturable path.
    w1.Resize({topk, cache->intermediate});
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
    int actCount = topk * cache->intermediate;
    constexpr int threads = 256;
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

        // Match the two configurations selected in the profiled vLLM TP8
        // decode graph.  Auto tuning sees only 32 EP-local experts here and
        // otherwise chooses the down-projection configuration for both GEMMs.
        // Supplying the known configuration also removes launcher-side search
        // overhead in eager mode.
        constexpr int gateThreadK = 128;
        constexpr int gateThreadN = 128;
        constexpr int gateBlocksPerSm = 1;
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

        constexpr int downThreadK = 64;
        constexpr int downThreadN = 128;
        constexpr int downBlocksPerSm = 3;
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
            (const T *)cache->downOutput, rowIndices, rowOutput, topk,
            ownerRank, ownerCount, cache->hidden);
        if (!checkStage("EP row reduction")) {
            return false;
        }

        // Prefill reuses one metadata/workspace/output scratch set for every
        // row.  The Marlin kernel can keep work in flight after it
        // returns, so the next row must not overwrite that scratch.  Decode
        // has one row and remains fully asynchronous/graph-capturable.
        if (rows > 1) {
            cudaError_t state = cudaStreamSynchronize(cudaStreamPerThread);
            if (state != cudaSuccess) {
                std::fprintf(stderr,
                             "[FastLLM] Marlin MoE failed at prefill "
                             "row boundary on GPU %d: %s\n",
                             cache->device, cudaGetErrorString(state));
                std::fflush(stderr);
                return false;
            }
        }
    }
    return true;
}

}  // namespace fastllm_marlin_moe

void FastllmCudaReleaseMergeMOEVllmMarlinCache(
        const fastllm::Data *layerKey) {
    fastllm_marlin_moe::ReleaseLayerCache(layerKey);
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
