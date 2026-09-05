//
// Graph-capturable compact NVFP4 expert cache.
//
// The checkpoint keeps E2M1 values and planar E4M3 block scales on the host.
// A device LRU cache stores complete expert records in that same compact
// representation. Misses are pulled directly from mapped pinned host memory
// by a CUDA kernel; CPU cores do not participate in the decode hot path.
//

#include "fastllm-cuda.cuh"
#include "fastllm-cuda-expert-cache.cuh"
#include "fastllm-cuda-record-copy.cuh"
#include "fastllm.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <cuda_bf16.h>

namespace {

constexpr int kMaxTopK = 16;
constexpr size_t kMinDeviceMemoryReserveBytes = 256ULL << 20;
constexpr size_t kMaxDeviceMemoryReserveBytes = 2ULL << 30;
constexpr size_t kDeviceMemoryReserveDivisor = 16;

size_t AlignUp(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

struct OffloadLayout {
    int experts = 0;
    int hidden = 0;
    int inter = 0;
    int gateBlockK = 0;
    int gateBlockM = 0;
    int downBlockK = 0;
    int downBlockM = 0;
    int gateScaleCols = 0;
    int downScaleCols = 0;
    size_t gateBytes = 0;
    size_t downBytes = 0;
    size_t downOffset = 0;
    size_t scalesOffset = 0;
    size_t recordStride = 0;
};

struct DeviceCache {
    bool attempted = false;
    bool ready = false;
    int device = -1;
    int slots = 0;
    fastllm::cuda::RecordCopyLaunch copyLaunch{0, 0};
    int ensureThreads = 0;
    uint8_t *records = nullptr;
    int32_t *keyToSlot = nullptr;
    int32_t *slotKeys = nullptr;
    unsigned long long *lastUsed = nullptr;
    unsigned long long *step = nullptr;
    int32_t *routeSlots = nullptr;
    int32_t *missExperts = nullptr;
    int32_t *missSlots = nullptr;
    int32_t *missCount = nullptr;
    unsigned long long *hitCount = nullptr;
    unsigned long long *totalMissCount = nullptr;
};

struct OffloadGroup {
    OffloadLayout layout;
    size_t totalRecords = 0;
    uint8_t *hostRecords = nullptr;
    uint8_t *deviceHostRecords = nullptr;
    std::vector<const fastllm::Data *> tableKeys;
    std::unordered_map<int, std::unique_ptr<DeviceCache> > deviceCaches;
    std::mutex mutex;
};

struct CachedTable {
    OffloadGroup *group;
    int layer;
};

std::mutex &RegistryMutex() {
    static auto *mutex = new std::mutex();
    return *mutex;
}

std::vector<std::unique_ptr<OffloadGroup> > &Groups() {
    // CUDA may already be unloading when ordinary static destructors run.
    // Process-lifetime ownership mirrors the other CUDA MoE registries.
    static auto *groups =
        new std::vector<std::unique_ptr<OffloadGroup> >();
    return *groups;
}

std::unordered_map<const fastllm::Data *, CachedTable> &TableRegistry() {
    static auto *registry =
        new std::unordered_map<const fastllm::Data *, CachedTable>();
    return *registry;
}

bool ValidateWeightPair(fastllm::Data *gate, fastllm::Data *down,
                        const OffloadLayout *expected,
                        OffloadLayout &observed) {
    if (gate == nullptr || down == nullptr ||
        gate->dataType != fastllm::DataType::NVFP4_BLOCK_16_E4M3 ||
        down->dataType != fastllm::DataType::NVFP4_BLOCK_16_E4M3 ||
        gate->dataDevice != fastllm::DataDevice::CPU ||
        down->dataDevice != fastllm::DataDevice::CPU ||
        gate->cpuData == nullptr || down->cpuData == nullptr ||
        gate->dims.size() != 2 || down->dims.size() != 2 ||
        gate->dims[0] <= 0 || gate->dims[1] <= 0 ||
        down->dims[0] <= 0 || down->dims[1] <= 0 ||
        gate->dims[0] != down->dims[1] * 2 ||
        gate->dims[1] != down->dims[0] ||
        gate->blockK <= 0 || gate->blockM <= 0 ||
        down->blockK <= 0 || down->blockM <= 0 ||
        gate->scales.size() != 2 || down->scales.size() != 1) {
        return false;
    }

    observed.hidden = gate->dims[1];
    observed.inter = down->dims[1];
    observed.gateBlockK = gate->blockK;
    observed.gateBlockM = gate->blockM;
    observed.downBlockK = down->blockK;
    observed.downBlockM = down->blockM;
    observed.gateScaleCols =
        (observed.hidden + observed.gateBlockM - 1) /
        observed.gateBlockM;
    observed.downScaleCols =
        (observed.inter + observed.downBlockM - 1) /
        observed.downBlockM;
    observed.gateBytes = gate->GetBytes();
    observed.downBytes = down->GetBytes();
    observed.downOffset = AlignUp(observed.gateBytes, 16);
    observed.scalesOffset =
        AlignUp(observed.downOffset + observed.downBytes, 16);
    // Keep each host expert record aligned for GPU reads over PCIe. The
    // trailing global scales would otherwise shift successive source addresses.
    observed.recordStride =
        AlignUp(observed.scalesOffset + 3 * sizeof(float), 128);
    if (observed.gateBytes == 0 || observed.downBytes == 0) {
        return false;
    }
    if (expected == nullptr) {
        return true;
    }
    return expected->hidden == observed.hidden &&
           expected->inter == observed.inter &&
           expected->gateBlockK == observed.gateBlockK &&
           expected->gateBlockM == observed.gateBlockM &&
           expected->downBlockK == observed.downBlockK &&
           expected->downBlockM == observed.downBlockM &&
           expected->gateBytes == observed.gateBytes &&
           expected->downBytes == observed.downBytes &&
           expected->downOffset == observed.downOffset &&
           expected->scalesOffset == observed.scalesOffset &&
           expected->recordStride == observed.recordStride;
}

size_t RequestedSlots(size_t recordStride, size_t totalRecords) {
    const uint64_t bytes = fastllm::GetMoeCudaCacheBytes();
    if (bytes == 0 || recordStride == 0) {
        return 0;
    }
    const uint64_t slots = bytes / recordStride;
    return slots >= totalRecords
        ? totalRecords : static_cast<size_t>(slots);
}

size_t DeviceMemoryReserveBytes(size_t totalBytes) {
    // Keep enough room for activations, CUDA Graph pools, and allocator
    // bookkeeping without imposing a fixed multi-GiB penalty on small GPUs.
    return std::min(
        kMaxDeviceMemoryReserveBytes,
        std::max(kMinDeviceMemoryReserveBytes,
                 totalBytes / kDeviceMemoryReserveDivisor));
}

void ReleaseDeviceCache(DeviceCache &cache) {
    if (cache.device >= 0) {
        cudaSetDevice(cache.device);
    }
    cudaFree(cache.records);
    cudaFree(cache.keyToSlot);
    cudaFree(cache.slotKeys);
    cudaFree(cache.lastUsed);
    cudaFree(cache.step);
    cudaFree(cache.routeSlots);
    cudaFree(cache.missExperts);
    cudaFree(cache.missSlots);
    cudaFree(cache.missCount);
    cudaFree(cache.hitCount);
    cudaFree(cache.totalMissCount);
    cache = DeviceCache();
}

void PrintDeviceCacheStats(const DeviceCache &cache) {
    if (!cache.ready || cache.hitCount == nullptr ||
        cache.totalMissCount == nullptr) {
        return;
    }
    cudaSetDevice(cache.device);
    unsigned long long hits = 0;
    unsigned long long misses = 0;
    const cudaError_t hitState = cudaMemcpy(
        &hits, cache.hitCount, sizeof(hits), cudaMemcpyDeviceToHost);
    const cudaError_t missState = cudaMemcpy(
        &misses, cache.totalMissCount, sizeof(misses),
        cudaMemcpyDeviceToHost);
    if (hitState != cudaSuccess || missState != cudaSuccess) {
        cudaGetLastError();
        return;
    }
    const unsigned long long routes = hits + misses;
    const double hitRate = routes == 0 ? 0.0 :
        100.0 * static_cast<double>(hits) /
            static_cast<double>(routes);
    std::fprintf(
        stderr,
        "[Fastllm] NVFP4 GPU expert cache stats cuda:%d: "
        "%llu hits, %llu misses, %.3f%% hit rate, %d slots.\n",
        cache.device, hits, misses, hitRate, cache.slots);
}

bool AllocateOne(void **pointer, size_t bytes) {
    *pointer = nullptr;
    return bytes > 0 && cudaMalloc(pointer, bytes) == cudaSuccess &&
           *pointer != nullptr;
}

DeviceCache *GetDeviceCache(OffloadGroup &group) {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0) {
        return nullptr;
    }
    std::lock_guard<std::mutex> guard(group.mutex);
    std::unique_ptr<DeviceCache> &entry = group.deviceCaches[device];
    if (!entry) {
        entry.reset(new DeviceCache());
        entry->device = device;
    }
    DeviceCache &cache = *entry;
    if (cache.ready || cache.attempted) {
        return cache.ready ? &cache : nullptr;
    }
    cache.attempted = true;

    size_t slots = RequestedSlots(
        group.layout.recordStride, group.totalRecords);
    if (slots < kMaxTopK) {
        std::fprintf(stderr,
            "[Fastllm] NVFP4 offload cache needs at least %d slots; "
            "requested %zu.\n", kMaxTopK, slots);
        return nullptr;
    }

    size_t freeBytes = 0, totalBytes = 0;
    if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess) {
        return nullptr;
    }
    const size_t reserveBytes = std::min(
        freeBytes, DeviceMemoryReserveBytes(totalBytes));
    const size_t metadataBytes =
        group.totalRecords * sizeof(int32_t) +
        slots * (sizeof(int32_t) + sizeof(unsigned long long)) + 4096;
    size_t usableBytes = freeBytes > reserveBytes
        ? freeBytes - reserveBytes : 0;
    usableBytes = usableBytes > metadataBytes
        ? usableBytes - metadataBytes : 0;
    slots = std::min(slots, usableBytes / group.layout.recordStride);
    if (slots < kMaxTopK) {
        std::fprintf(stderr,
            "[Fastllm] NVFP4 offload cache has insufficient free GPU "
            "memory after reserving %.3f GiB for runtime allocations.\n",
            static_cast<double>(reserveBytes) /
                (1024.0 * 1024.0 * 1024.0));
        return nullptr;
    }
    cache.slots = static_cast<int>(slots);

    cudaDeviceProp properties;
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        return nullptr;
    }
    cache.ensureThreads = fastllm::cuda::ExpertCacheThreads(
        cache.slots, properties.maxThreadsPerBlock);
    cache.copyLaunch = fastllm::cuda::RecordCopyConfiguration(
        group.layout.recordStride, kMaxTopK, properties);

    const bool initialized =
        AllocateOne(reinterpret_cast<void **>(&cache.records),
                    slots * group.layout.recordStride) &&
        AllocateOne(reinterpret_cast<void **>(&cache.keyToSlot),
                    group.totalRecords * sizeof(int32_t)) &&
        AllocateOne(reinterpret_cast<void **>(&cache.slotKeys),
                    slots * sizeof(int32_t)) &&
        AllocateOne(reinterpret_cast<void **>(&cache.lastUsed),
                    slots * sizeof(unsigned long long)) &&
        AllocateOne(reinterpret_cast<void **>(&cache.step),
                    sizeof(unsigned long long)) &&
        AllocateOne(reinterpret_cast<void **>(&cache.routeSlots),
                    kMaxTopK * sizeof(int32_t)) &&
        AllocateOne(reinterpret_cast<void **>(&cache.missExperts),
                    kMaxTopK * sizeof(int32_t)) &&
        AllocateOne(reinterpret_cast<void **>(&cache.missSlots),
                    kMaxTopK * sizeof(int32_t)) &&
        AllocateOne(reinterpret_cast<void **>(&cache.missCount),
                    sizeof(int32_t)) &&
        AllocateOne(reinterpret_cast<void **>(&cache.hitCount),
                    sizeof(unsigned long long)) &&
        AllocateOne(reinterpret_cast<void **>(&cache.totalMissCount),
                    sizeof(unsigned long long)) &&
        cudaMemsetAsync(cache.keyToSlot, 0xff,
            group.totalRecords * sizeof(int32_t), cudaStreamPerThread) == cudaSuccess &&
        cudaMemsetAsync(cache.slotKeys, 0xff,
            slots * sizeof(int32_t), cudaStreamPerThread) == cudaSuccess &&
        cudaMemsetAsync(cache.lastUsed, 0,
            slots * sizeof(unsigned long long), cudaStreamPerThread) == cudaSuccess &&
        cudaMemsetAsync(cache.step, 0,
            sizeof(unsigned long long), cudaStreamPerThread) == cudaSuccess &&
        cudaMemsetAsync(cache.hitCount, 0,
            sizeof(unsigned long long), cudaStreamPerThread) == cudaSuccess &&
        cudaMemsetAsync(cache.totalMissCount, 0,
            sizeof(unsigned long long), cudaStreamPerThread) == cudaSuccess;
    if (!initialized) {
        std::fprintf(stderr,
            "[Fastllm] CUDA allocation or initialization of the NVFP4 expert cache failed: "
            "%s.\n", cudaGetErrorString(cudaGetLastError()));
        ReleaseDeviceCache(cache);
        cache.attempted = true;
        cache.device = device;
        return nullptr;
    }

    cache.ready = true;

    std::fprintf(stderr,
        "[Fastllm] NVFP4 GPU expert cache: %d slots, %.3f GiB, "
        "%zu records in mapped host storage; refill grid %d x %d.\n",
        cache.slots,
        static_cast<double>(slots * group.layout.recordStride) /
            (1024.0 * 1024.0 * 1024.0),
        group.totalRecords, cache.copyLaunch.blocks, cache.copyLaunch.threads);
    return &cache;
}

OffloadGroup *FindGroup(fastllm::Data **weights, int weightsBatch,
                        int *tableId = nullptr) {
    if (weights == nullptr || weightsBatch < 4 || weights[2] == nullptr) {
        return nullptr;
    }
    std::lock_guard<std::mutex> guard(RegistryMutex());
    auto it = TableRegistry().find(weights[2]);
    if (it == TableRegistry().end()) {
        return nullptr;
    }
    OffloadGroup *group = it->second.group;
    if (weightsBatch != (group->layout.experts + 1) * 2) {
        return nullptr;
    }
    if (tableId != nullptr) {
        *tableId = it->second.layer;
    }
    return group;
}

template <typename T>
struct ActivationTraits;

template <>
struct ActivationTraits<float> {
    static constexpr fastllm::DataType type = fastllm::DataType::FLOAT32;
    __device__ static float ToFloat(float value) { return value; }
    __device__ static float FromFloat(float value) { return value; }
    __device__ static float E4M3ToFloat(uint8_t value) {
        return __half2float(__ushort_as_half(
            ((value & 0x80) << 8) | ((value & 0x7f) << 7))) *
            exp2f(8.0f);
    }
};

template <>
struct ActivationTraits<half> : ActivationTraits<float> {
    static constexpr fastllm::DataType type = fastllm::DataType::FLOAT16;
    __device__ static float ToFloat(half value) { return __half2float(value); }
    __device__ static half FromFloat(float value) { return __float2half(value); }
};

template <>
struct ActivationTraits<__nv_bfloat16> {
    static constexpr fastllm::DataType type = fastllm::DataType::BFLOAT16;
    __device__ static float ToFloat(__nv_bfloat16 value) {
        return __bfloat162float(value);
    }
    __device__ static __nv_bfloat16 FromFloat(float value) {
        return __float2bfloat16_rn(value);
    }
    __device__ static float E4M3ToFloat(uint8_t value) {
        const uint16_t bits =
            ((value & 0x80) << 8) | ((value & 0x7f) << 4);
        return __bfloat162float(
            *reinterpret_cast<const __nv_bfloat16 *>(&bits)) *
            exp2f(120.0f);
    }
};

__device__ __forceinline__ float NVFP4PseudoToFloat(uint8_t value) {
    const uint32_t bits =
        (static_cast<uint32_t>(value & 0x8) << 28) |
        (static_cast<uint32_t>(value & 0x7) << 22);
    return __uint_as_float(bits);
}

__device__ __forceinline__ float NVFP4MagicScale() {
    return __uint_as_float(253u << 23);
}

template <typename T>
__device__ __forceinline__ float CompactE4M3Scale(
        const uint8_t *scaleData, int scaleRow, int scaleCols,
        int column, int blockM, float globalScale) {
    const uint8_t scaleByte = scaleData[
        static_cast<size_t>(scaleRow) * scaleCols + column / blockM];
    return ActivationTraits<T>::E4M3ToFloat(scaleByte) * globalScale;
}

template <typename T>
__device__ __forceinline__ void AccumulateCompactE4M3(
        const T *activation, int offset, const uint8_t *weight, int row,
        int rows, int columns, int blockK, int blockM, int scaleCols,
        float globalScale, float &sum) {
    const int remaining = min(4, columns - offset);
    if (remaining <= 0) {
        return;
    }
    const int packedPerRow = (columns + 1) >> 1;
    const uint8_t *rowData = weight +
        static_cast<size_t>(row) * packedPerRow;
    const uint8_t *scaleData = weight +
        static_cast<size_t>(rows) * packedPerRow;
    const int scaleRow = row / blockK;
    if (blockK == 1 && blockM == 16 && (columns & 127) == 0 &&
        remaining == 4) {
        // Adjacent lanes consume the low/high halves of the same eight-code
        // word. Let the even lane issue one aligned 32-bit load and broadcast
        // it to its partner. Four neighboring lanes similarly share one
        // block-16 scale. The per-thread accumulation and the later two-warp
        // reduction stay unchanged, preserving the established decode
        // numerics while reducing weight/scale load instructions.
        const int lane = threadIdx.x & 31;
        uint32_t packedWord = 0;
        if ((lane & 1) == 0) {
            packedWord = *reinterpret_cast<const uint32_t *>(
                rowData + (offset >> 1));
        }
        packedWord = __shfl_sync(
            0xffffffffu, packedWord, lane & ~1);
        int scaleByte = 0;
        if ((lane & 3) == 0) {
            scaleByte = scaleData[
                static_cast<size_t>(scaleRow) * scaleCols +
                offset / blockM];
        }
        scaleByte = __shfl_sync(
            0xffffffffu, scaleByte, lane & ~3);
        const uint16_t packedHalf = (lane & 1)
            ? static_cast<uint16_t>(packedWord >> 16)
            : static_cast<uint16_t>(packedWord);
        const uint8_t packed01 = static_cast<uint8_t>(packedHalf);
        const uint8_t packed23 = static_cast<uint8_t>(packedHalf >> 8);
        const float blockSum =
            ActivationTraits<T>::ToFloat(activation[offset]) *
                NVFP4PseudoToFloat(packed01 & 0xf) +
            ActivationTraits<T>::ToFloat(activation[offset + 1]) *
                NVFP4PseudoToFloat(packed01 >> 4) +
            ActivationTraits<T>::ToFloat(activation[offset + 2]) *
                NVFP4PseudoToFloat(packed23 & 0xf) +
            ActivationTraits<T>::ToFloat(activation[offset + 3]) *
                NVFP4PseudoToFloat(packed23 >> 4);
        sum += (blockSum * NVFP4MagicScale()) *
            (ActivationTraits<T>::E4M3ToFloat(
                 static_cast<uint8_t>(scaleByte)) * globalScale);
        return;
    }
    if (remaining == 4 && offset / blockM == (offset + 3) / blockM) {
        const uint8_t packed01 = rowData[offset >> 1];
        const uint8_t packed23 = rowData[(offset + 2) >> 1];
        const float blockSum =
            ActivationTraits<T>::ToFloat(activation[offset]) *
                NVFP4PseudoToFloat(packed01 & 0xf) +
            ActivationTraits<T>::ToFloat(activation[offset + 1]) *
                NVFP4PseudoToFloat(packed01 >> 4) +
            ActivationTraits<T>::ToFloat(activation[offset + 2]) *
                NVFP4PseudoToFloat(packed23 & 0xf) +
            ActivationTraits<T>::ToFloat(activation[offset + 3]) *
                NVFP4PseudoToFloat(packed23 >> 4);
        sum += (blockSum * NVFP4MagicScale()) *
            CompactE4M3Scale<T>(scaleData, scaleRow, scaleCols, offset,
                                blockM, globalScale);
        return;
    }
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        if (item < remaining) {
            const int column = offset + item;
            const uint8_t packed = rowData[column >> 1];
            const uint8_t fp4 = (column & 1) ? packed >> 4 : packed & 0xf;
            const float product =
                ActivationTraits<T>::ToFloat(activation[column]) *
                NVFP4PseudoToFloat(fp4);
            sum += (product * NVFP4MagicScale()) *
                CompactE4M3Scale<T>(scaleData, scaleRow, scaleCols,
                                    column, blockM, globalScale);
        }
    }
}

template <typename T>
__device__ __forceinline__ void AccumulateCompactE4M3WordPair(
        const T *activation, int word, const uint8_t *weight, int row,
        int rows, int columns, int scaleCols, float globalScale,
        float &lowSum, float &highSum) {
    const int packedPerRow = columns >> 1;
    const uint8_t *rowData = weight +
        static_cast<size_t>(row) * packedPerRow;
    const uint8_t *scaleData = weight +
        static_cast<size_t>(rows) * packedPerRow;
    const uint32_t packed = reinterpret_cast<const uint32_t *>(
        rowData)[word];
    const float scale = ActivationTraits<T>::E4M3ToFloat(
        scaleData[static_cast<size_t>(row) * scaleCols + (word >> 1)]) *
        globalScale;
    const int offset = word << 3;
    const uint8_t packed01 = static_cast<uint8_t>(packed);
    const uint8_t packed23 = static_cast<uint8_t>(packed >> 8);
    const uint8_t packed45 = static_cast<uint8_t>(packed >> 16);
    const uint8_t packed67 = static_cast<uint8_t>(packed >> 24);
    const float low =
        ActivationTraits<T>::ToFloat(activation[offset]) *
            NVFP4PseudoToFloat(packed01 & 0xf) +
        ActivationTraits<T>::ToFloat(activation[offset + 1]) *
            NVFP4PseudoToFloat(packed01 >> 4) +
        ActivationTraits<T>::ToFloat(activation[offset + 2]) *
            NVFP4PseudoToFloat(packed23 & 0xf) +
        ActivationTraits<T>::ToFloat(activation[offset + 3]) *
            NVFP4PseudoToFloat(packed23 >> 4);
    const float high =
        ActivationTraits<T>::ToFloat(activation[offset + 4]) *
            NVFP4PseudoToFloat(packed45 & 0xf) +
        ActivationTraits<T>::ToFloat(activation[offset + 5]) *
            NVFP4PseudoToFloat(packed45 >> 4) +
        ActivationTraits<T>::ToFloat(activation[offset + 6]) *
            NVFP4PseudoToFloat(packed67 & 0xf) +
        ActivationTraits<T>::ToFloat(activation[offset + 7]) *
            NVFP4PseudoToFloat(packed67 >> 4);
    lowSum += (low * NVFP4MagicScale()) * scale;
    highSum += (high * NVFP4MagicScale()) * scale;
}

__device__ __forceinline__ float ReduceVirtual64(
        float low, float high) {
    // One physical lane represents two adjacent lanes from the established
    // 64-thread kernel. Half-warp shuffles reproduce its 16/8/4/2 stages;
    // adding low+high is the final offset-1 stage. Lanes 0 and 16 therefore
    // hold the two original warp partials.
    for (int offset = 8; offset > 0; offset >>= 1) {
        low += __shfl_down_sync(0xffffffffu, low, offset, 16);
        high += __shfl_down_sync(0xffffffffu, high, offset, 16);
    }
    const float half = low + high;
    return half + __shfl_sync(0xffffffffu, half, 16);
}

template <typename T, int RowPairsPerBlock>
__global__ void FastllmNVFP4OffloadGateExactWideKernel(
        const T *input, const int32_t *routeSlots,
        const uint8_t *records, T *output, int topk, int hidden, int inter,
        int scaleCols, size_t recordStride, size_t scalesOffset) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * RowPairsPerBlock + warp;
    const int route = blockIdx.y;
    if (warp >= RowPairsPerBlock || row >= inter || route >= topk) {
        return;
    }
    const int slot = routeSlots[route];
    if (slot < 0) {
        return;
    }
    const uint8_t *record = records +
        static_cast<size_t>(slot) * recordStride;
    const float *globalScales = reinterpret_cast<const float *>(
        record + scalesOffset);
    float gateLow = 0.0f, gateHigh = 0.0f;
    float upLow = 0.0f, upHigh = 0.0f;
    const int words = hidden >> 3;
    for (int word = lane; word < words; word += 32) {
        AccumulateCompactE4M3WordPair(
            input, word, record, row, inter * 2, hidden, scaleCols,
            globalScales[0], gateLow, gateHigh);
        AccumulateCompactE4M3WordPair(
            input, word, record, row + inter, inter * 2, hidden,
            scaleCols, globalScales[1], upLow, upHigh);
    }
    const float gate = ReduceVirtual64(gateLow, gateHigh);
    const float up = ReduceVirtual64(upLow, upHigh);
    if (lane == 0) {
        const float gateValue = ActivationTraits<T>::ToFloat(
            ActivationTraits<T>::FromFloat(gate));
        const float upValue = ActivationTraits<T>::ToFloat(
            ActivationTraits<T>::FromFloat(up));
        output[static_cast<size_t>(route) * inter + row] =
            ActivationTraits<T>::FromFloat(
                (gateValue / (1.0f + expf(-gateValue))) * upValue);
    }
}

template <typename T, int Threads>
__global__ void FastllmNVFP4OffloadGateKernel(
        const T *input, const int32_t *routeSlots,
        const uint8_t *records, T *output, int topk, int hidden, int inter,
        int blockK, int blockM, int scaleCols, size_t recordStride,
        size_t scalesOffset) {
    static_assert(Threads == 64, "gate reduction expects two warps");
    __shared__ float gateWarp[2];
    __shared__ float upWarp[2];
    const int row = blockIdx.x;
    const int route = blockIdx.y;
    if (route >= topk || routeSlots[route] < 0) {
        return;
    }
    const int local = threadIdx.x;
    const uint8_t *record = records +
        static_cast<size_t>(routeSlots[route]) * recordStride;
    const float *globalScales = reinterpret_cast<const float *>(
        record + scalesOffset);
    float gate = 0.0f;
    float up = 0.0f;
    for (int column = local * 4; column < hidden;
         column += Threads * 4) {
        AccumulateCompactE4M3(
            input, column, record, row, inter * 2, hidden,
            blockK, blockM, scaleCols, globalScales[0], gate);
        AccumulateCompactE4M3(
            input, column, record, row + inter, inter * 2, hidden,
            blockK, blockM, scaleCols, globalScales[1], up);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate += __shfl_down_sync(0xffffffffu, gate, offset);
        up += __shfl_down_sync(0xffffffffu, up, offset);
    }
    if ((local & 31) == 0) {
        gateWarp[local >> 5] = gate;
        upWarp[local >> 5] = up;
    }
    __syncthreads();
    if (local == 0) {
        const float gateValue = ActivationTraits<T>::ToFloat(
            ActivationTraits<T>::FromFloat(gateWarp[0] + gateWarp[1]));
        const float upValue = ActivationTraits<T>::ToFloat(
            ActivationTraits<T>::FromFloat(upWarp[0] + upWarp[1]));
        output[static_cast<size_t>(route) * inter + row] =
            ActivationTraits<T>::FromFloat(
                (gateValue / (1.0f + expf(-gateValue))) * upValue);
    }
}

template <typename T, int GroupThreads, int MaxTopK>
__global__ void FastllmNVFP4OffloadDownKernel(
        const T *input, const int32_t *routeSlots,
        const uint8_t *records, T *output, const float *scores,
        int topk, int inter, int hidden, int blockK, int blockM,
        int scaleCols, size_t recordStride, size_t downOffset,
        size_t scalesOffset) {
    __shared__ float warpPartials[MaxTopK * 2];
    __shared__ float expertOutputs[MaxTopK];
    const int route = threadIdx.x / GroupThreads;
    const int local = threadIdx.x % GroupThreads;
    const int row = blockIdx.x;
    const int slot = routeSlots[route];
    const size_t safeSlot = slot >= 0 ? static_cast<size_t>(slot) : 0;
    const uint8_t *record = records + safeSlot * recordStride;
    const uint8_t *weight = record + downOffset;
    const float *globalScales = reinterpret_cast<const float *>(
        record + scalesOffset);
    const T *expertInput = input + static_cast<size_t>(route) * inter;
    float value = 0.0f;
    if (route < topk && slot >= 0) {
        for (int column = local * 4; column < inter;
             column += GroupThreads * 4) {
            AccumulateCompactE4M3(
                expertInput, column, weight, row, hidden, inter,
                blockK, blockM, scaleCols, globalScales[2], value);
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    if ((local & 31) == 0) {
        warpPartials[route * 2 + (local >> 5)] = value;
    }
    __syncthreads();
    if (local == 0) {
        const float rounded = ActivationTraits<T>::ToFloat(
            ActivationTraits<T>::FromFloat(
                warpPartials[route * 2] + warpPartials[route * 2 + 1]));
        expertOutputs[route] = rounded * scores[route];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int item = 0; item < topk; ++item) {
            sum += expertOutputs[item];
        }
        output[row] = ActivationTraits<T>::FromFloat(sum);
    }
}

template <typename T, int MaxTopK>
__global__ void FastllmNVFP4OffloadDownExactWideKernel(
        const T *input, const int32_t *routeSlots,
        const uint8_t *records, T *output, const float *scores,
        int topk, int inter, int hidden, int scaleCols,
        size_t recordStride, size_t downOffset, size_t scalesOffset) {
    __shared__ float expertOutputs[MaxTopK];
    const int route = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x;
    float low = 0.0f, high = 0.0f;
    if (route < topk) {
        const int slot = routeSlots[route];
        if (slot >= 0) {
            const uint8_t *record = records +
                static_cast<size_t>(slot) * recordStride;
            const uint8_t *weight = record + downOffset;
            const T *expertInput = input +
                static_cast<size_t>(route) * inter;
            const int words = inter >> 3;
            const float globalScale = reinterpret_cast<const float *>(
                record + scalesOffset)[2];
            for (int word = lane; word < words; word += 32) {
                AccumulateCompactE4M3WordPair(
                    expertInput, word, weight, row, hidden, inter,
                    scaleCols, globalScale, low, high);
            }
        }
        const float value = ReduceVirtual64(low, high);
        if (lane == 0) {
            const float rounded = ActivationTraits<T>::ToFloat(
                ActivationTraits<T>::FromFloat(value));
            expertOutputs[route] = rounded * scores[route];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int item = 0; item < topk; ++item) {
            sum += expertOutputs[item];
        }
        output[row] = ActivationTraits<T>::FromFloat(sum);
    }
}

template <typename T>
bool RunTyped(const fastllm::Data &input, fastllm::Data &gateOutput,
              fastllm::Data &output, OffloadGroup &group,
              DeviceCache &cache, int tableId, const int32_t *indices,
              const float *scores, int topk) {
    const OffloadLayout &layout = group.layout;
    gateOutput.dataType = ActivationTraits<T>::type;
    gateOutput.dataDevice = input.dataDevice;
    gateOutput.dataDeviceIds = input.dataDeviceIds;
    gateOutput.Resize({topk, layout.inter});
    output.dataType = ActivationTraits<T>::type;
    output.dataDevice = input.dataDevice;
    output.dataDeviceIds = input.dataDeviceIds;
    output.Resize({1, layout.hidden});
    gateOutput.Allocate(false);
    output.Allocate(false);
    if (gateOutput.cudaData == nullptr || output.cudaData == nullptr) {
        return false;
    }

    fastllm::cuda::ExpertCacheView metadata{
        cache.keyToSlot, cache.slotKeys, cache.lastUsed, cache.step,
        cache.hitCount, cache.totalMissCount, cache.slots};
    if (!fastllm::cuda::EnsureExpertCache<kMaxTopK>(
            metadata, indices, tableId * layout.experts, layout.experts, topk,
            cache.routeSlots, cache.missExperts, cache.missSlots, cache.missCount,
            cache.ensureThreads, cudaStreamPerThread)) return false;
    const uint8_t *sourceTable = group.deviceHostRecords +
        static_cast<size_t>(tableId) * layout.experts *
        layout.recordStride;
    if (!fastllm::cuda::CopyRecords(
            {sourceTable, cache.records, layout.recordStride,
             layout.recordStride, layout.recordStride},
            cache.missExperts, cache.missSlots, cache.missCount, topk,
            cache.copyLaunch, cudaStreamPerThread)) return false;
    const bool useExactWideDecode =
        layout.gateBlockK == 1 && layout.gateBlockM == 16 &&
        layout.downBlockK == 1 && layout.downBlockM == 16 &&
        (layout.hidden & 127) == 0 && (layout.inter & 127) == 0;
    if (useExactWideDecode) {
        constexpr int rowPairsPerBlock = 4;
        const dim3 gateGrid(
            (layout.inter + rowPairsPerBlock - 1) / rowPairsPerBlock,
            topk);
        FastllmNVFP4OffloadGateExactWideKernel<T, rowPairsPerBlock>
            <<<gateGrid, rowPairsPerBlock * 32, 0,
               cudaStreamPerThread>>>(
            reinterpret_cast<const T *>(input.cudaData), cache.routeSlots,
            cache.records, reinterpret_cast<T *>(gateOutput.cudaData), topk,
            layout.hidden, layout.inter, layout.gateScaleCols,
            layout.recordStride, layout.scalesOffset);
        FastllmNVFP4OffloadDownExactWideKernel<T, kMaxTopK>
            <<<layout.hidden, topk * 32, 0, cudaStreamPerThread>>>(
            reinterpret_cast<const T *>(gateOutput.cudaData),
            cache.routeSlots, cache.records,
            reinterpret_cast<T *>(output.cudaData), scores, topk,
            layout.inter, layout.hidden, layout.downScaleCols,
            layout.recordStride, layout.downOffset,
            layout.scalesOffset);
    } else {
        const dim3 gateGrid(layout.inter, topk);
        FastllmNVFP4OffloadGateKernel<T, 64>
            <<<gateGrid, 64, 0, cudaStreamPerThread>>>(
            reinterpret_cast<const T *>(input.cudaData), cache.routeSlots,
            cache.records, reinterpret_cast<T *>(gateOutput.cudaData), topk,
            layout.hidden, layout.inter, layout.gateBlockK,
            layout.gateBlockM, layout.gateScaleCols, layout.recordStride,
            layout.scalesOffset);
        FastllmNVFP4OffloadDownKernel<T, 64, kMaxTopK>
            <<<layout.hidden, topk * 64, 0, cudaStreamPerThread>>>(
            reinterpret_cast<const T *>(gateOutput.cudaData),
            cache.routeSlots, cache.records,
            reinterpret_cast<T *>(output.cudaData), scores, topk,
            layout.inter, layout.hidden, layout.downBlockK,
            layout.downBlockM, layout.downScaleCols,
            layout.recordStride, layout.downOffset,
            layout.scalesOffset);
    }
    return cudaGetLastError() == cudaSuccess;
}

} // namespace

bool FastllmCudaMoeCacheRequested() {
    return fastllm::GetMoeCudaCacheBytes() > 0;
}

bool FastllmCudaPrepareMoeCache(
        const FastllmCudaMoeCacheLayer *layers, int layerCount) {
    if (!FastllmCudaMoeCacheRequested()) {
        return false;
    }
    if (layers == nullptr || layerCount <= 0 ||
        layers[0].weights == nullptr || layers[0].weightsBatch < 4 ||
        (layers[0].weightsBatch & 1)) {
        return false;
    }
    const int experts = layers[0].weightsBatch / 2 - 1;
    if (experts <= 0) {
        return false;
    }

    std::lock_guard<std::mutex> registryGuard(RegistryMutex());
    if (layers[0].weights[2] != nullptr &&
        TableRegistry().count(layers[0].weights[2]) != 0) {
        return true;
    }

    OffloadLayout layout;
    layout.experts = experts;
    bool first = true;
    for (int layer = 0; layer < layerCount; ++layer) {
        if (layers[layer].weights == nullptr ||
            layers[layer].weightsBatch != (experts + 1) * 2) {
            return false;
        }
        for (int expert = 0; expert < experts; ++expert) {
            const int position = (expert + 1) * 2;
            OffloadLayout observed;
            if (!ValidateWeightPair(
                    layers[layer].weights[position],
                    layers[layer].weights[position + 1],
                    first ? nullptr : &layout, observed)) {
                const fastllm::Data *gate =
                    layers[layer].weights[position];
                const fastllm::Data *down =
                    layers[layer].weights[position + 1];
                std::fprintf(stderr,
                    "[Fastllm] CUDA expert cache rejected layer %d expert %d: "
                    "requires matching host-resident compact NVFP4 weights "
                    "and gate/up/down scales (gate dtype=%d, down dtype=%d).\n",
                    layer, expert,
                    gate == nullptr ? -1 : static_cast<int>(gate->dataType),
                    down == nullptr ? -1 : static_cast<int>(down->dataType));
                return false;
            }
            if (first) {
                observed.experts = experts;
                layout = observed;
                first = false;
            }
        }
    }

    std::unique_ptr<OffloadGroup> group(new OffloadGroup());
    group->layout = layout;
    if (static_cast<size_t>(layerCount) >
            static_cast<size_t>(INT_MAX) /
                static_cast<size_t>(experts)) {
        return false;
    }
    group->totalRecords = static_cast<size_t>(layerCount) * experts;
    if (group->totalRecords > SIZE_MAX / layout.recordStride) {
        return false;
    }
    const size_t requestedSlots = RequestedSlots(
        layout.recordStride, group->totalRecords);
    if (requestedSlots < kMaxTopK) {
        std::fprintf(
            stderr,
            "[Fastllm] NVFP4 GPU expert cache needs at least %d slots; "
            "the configured budget holds %zu.\n",
            kMaxTopK, requestedSlots);
        return false;
    }
    const size_t hostBytes = group->totalRecords * layout.recordStride;
    void *host = nullptr;
    cudaError_t state = cudaHostAlloc(
        &host, hostBytes, cudaHostAllocMapped | cudaHostAllocPortable);
    if (state != cudaSuccess || host == nullptr) {
        std::fprintf(stderr,
            "[Fastllm] NVFP4 offload could not allocate %.3f GiB of mapped "
            "host storage: %s. The normal MoE backend remains active.\n",
            static_cast<double>(hostBytes) /
                (1024.0 * 1024.0 * 1024.0),
            cudaGetErrorString(state));
        cudaGetLastError();
        return false;
    }
    group->hostRecords = reinterpret_cast<uint8_t *>(host);
    void *deviceHost = nullptr;
    state = cudaHostGetDevicePointer(&deviceHost, host, 0);
    if (state != cudaSuccess || deviceHost == nullptr) {
        std::fprintf(stderr,
            "[Fastllm] NVFP4 offload could not map host storage: %s.\n",
            cudaGetErrorString(state));
        cudaFreeHost(host);
        cudaGetLastError();
        return false;
    }
    group->deviceHostRecords = reinterpret_cast<uint8_t *>(deviceHost);
    group->tableKeys.reserve(layerCount);

    for (int layer = 0; layer < layerCount; ++layer) {
        fastllm::Data *tableKey = layers[layer].weights[2];
        group->tableKeys.push_back(tableKey);
        for (int expert = 0; expert < experts; ++expert) {
            const int position = (expert + 1) * 2;
            fastllm::Data *gate = layers[layer].weights[position];
            fastllm::Data *down = layers[layer].weights[position + 1];
            uint8_t *record = group->hostRecords +
                (static_cast<size_t>(layer) * experts + expert) *
                layout.recordStride;
            std::memcpy(record, gate->cpuData, layout.gateBytes);
            std::memcpy(record + layout.downOffset,
                        down->cpuData, layout.downBytes);
            float *globalScales = reinterpret_cast<float *>(
                record + layout.scalesOffset);
            globalScales[0] = gate->scales[0];
            globalScales[1] = gate->scales[1];
            globalScales[2] = down->scales[0];
        }
    }

    OffloadGroup *rawGroup = group.get();
    for (int layer = 0; layer < layerCount; ++layer) {
        TableRegistry()[group->tableKeys[layer]] = {rawGroup, layer};
    }
    Groups().push_back(std::move(group));
    std::fprintf(stderr,
        "[Fastllm] Prepared %d x %d compact NVFP4 experts in %.3f "
        "GiB of mapped host storage (record %zu bytes).\n",
        layerCount, experts,
        static_cast<double>(hostBytes) /
            (1024.0 * 1024.0 * 1024.0),
        layout.recordStride);
    return true;
}

bool FastllmCudaCanRunMoeCache(
        fastllm::Data **weights, int weightsBatch) {
    if (!FastllmCudaMoeCacheRequested()) {
        return false;
    }
    OffloadGroup *group = FindGroup(weights, weightsBatch);
    return group != nullptr && GetDeviceCache(*group) != nullptr;
}

bool FastllmCudaCanRunMoeCacheBatch1(
        const fastllm::Data &input, const fastllm::Data &index,
        const fastllm::Data &score, fastllm::Data **weights,
        int weightsBatch, fastllm::MoeGateType gateType) {
    const bool supportedInput = FastllmCudaMoeCacheRequested() &&
           gateType == fastllm::MoeGateSwiglu &&
           input.dims.size() == 2 && input.dims[0] == 1 &&
           (input.dataType == fastllm::DataType::FLOAT32 ||
            input.dataType == fastllm::DataType::FLOAT16 ||
            input.dataType == fastllm::DataType::BFLOAT16) &&
           input.dataDevice == fastllm::DataDevice::CUDA &&
           input.cudaData != nullptr &&
           index.dims.size() == 2 && index.dims[0] == 1 &&
           index.dims[1] > 0 && index.dims[1] <= kMaxTopK &&
           index.dataDevice == fastllm::DataDevice::CUDA &&
           index.dataType == fastllm::DataType::INT32 &&
           index.cudaData != nullptr && score.dims == index.dims &&
           score.dataDevice == fastllm::DataDevice::CUDA &&
           score.dataType == fastllm::DataType::FLOAT32 &&
           score.cudaData != nullptr;
    if (!supportedInput) {
        return false;
    }
    OffloadGroup *group = FindGroup(weights, weightsBatch);
    return group != nullptr && input.dims[1] == group->layout.hidden &&
           GetDeviceCache(*group) != nullptr;
}

void FastllmCudaReleaseMoeCache(
        fastllm::Data **weights, int weightsBatch) {
    if (weights == nullptr || weightsBatch < 4 || weights[2] == nullptr) {
        return;
    }

    std::unique_ptr<OffloadGroup> released;
    {
        std::lock_guard<std::mutex> registryGuard(RegistryMutex());
        auto table = TableRegistry().find(weights[2]);
        if (table == TableRegistry().end()) {
            return;
        }
        OffloadGroup *group = table->second.group;
        if (weightsBatch != (group->layout.experts + 1) * 2) {
            return;
        }
        for (const fastllm::Data *key : group->tableKeys) {
            auto entry = TableRegistry().find(key);
            if (entry != TableRegistry().end() && entry->second.group == group) {
                TableRegistry().erase(entry);
            }
        }
        auto entry = std::find_if(
            Groups().begin(), Groups().end(),
            [group](const std::unique_ptr<OffloadGroup> &candidate) {
                return candidate.get() == group;
            });
        if (entry != Groups().end()) {
            released = std::move(*entry);
            Groups().erase(entry);
        }
    }
    if (!released) {
        return;
    }

    int originalDevice = -1;
    cudaGetDevice(&originalDevice);
    for (auto &cache : released->deviceCaches) {
        PrintDeviceCacheStats(*cache.second);
        ReleaseDeviceCache(*cache.second);
    }
    released->deviceCaches.clear();
    if (released->hostRecords != nullptr) {
        cudaFreeHost(released->hostRecords);
        released->hostRecords = nullptr;
        released->deviceHostRecords = nullptr;
    }
    if (originalDevice >= 0) {
        cudaSetDevice(originalDevice);
    }
}

bool FastllmCudaMergeMOECacheBatch1(
        const fastllm::Data &input, fastllm::Data &gateOutput,
        fastllm::Data &output, fastllm::Data **weights, int weightsBatch,
        const int32_t *indices, const float *scores, int topk) {
    if (input.dataDevice != fastllm::DataDevice::CUDA ||
        input.cudaData == nullptr || input.dims.size() != 2 ||
        input.dims[0] != 1 || indices == nullptr || scores == nullptr ||
        topk <= 0 || topk > kMaxTopK ||
        (input.dataType != fastllm::DataType::FLOAT32 &&
         input.dataType != fastllm::DataType::FLOAT16 &&
         input.dataType != fastllm::DataType::BFLOAT16)) {
        return false;
    }
    int tableId = -1;
    OffloadGroup *group = FindGroup(weights, weightsBatch, &tableId);
    if (group == nullptr || input.dims.back() != group->layout.hidden) {
        return false;
    }
    DeviceCache *cache = GetDeviceCache(*group);
    if (cache == nullptr) {
        return false;
    }
    if (input.dataType == fastllm::DataType::FLOAT32) {
        return RunTyped<float>(input, gateOutput, output, *group, *cache,
                               tableId, indices, scores, topk);
    }
    return input.dataType == fastllm::DataType::FLOAT16
        ? RunTyped<half>(input, gateOutput, output, *group, *cache,
                         tableId, indices, scores, topk)
        : RunTyped<__nv_bfloat16>(input, gateOutput, output, *group, *cache,
                                  tableId, indices, scores, topk);
}
