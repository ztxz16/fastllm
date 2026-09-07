//
// Graph-capturable FP8 and compact NVFP4 expert cache.
//
// Host records retain quantized weights and their format-specific scales.
// A device LRU cache stores complete expert records in that same compact
// representation. Misses are pulled directly from mapped pinned host memory
// by a CUDA kernel. Eager hybrid decode can leave selected experts on NUMA
// when their estimated CPU completion is earlier than a cache refill.
//

#include "fastllm-cuda.cuh"
#include "fastllm-cuda-expert-cache.cuh"
#include "fastllm-cuda-record-copy.cuh"
#include "fastllm-cuda-shared-weight.cuh"
#include "fastllm.h"
#include "utils.h"
#include "devices/moe_decode_scheduler.h"
#include "devices/cuda/fastllm-cuda-moe-policy.h"
#ifdef USE_NUMAS
#include "devices/numas/numasdevice.h"
#endif

#include <algorithm>
#include <climits>
#include <chrono>
#include <array>
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

constexpr int kMaxTopK = fastllm::MoeDecodeScheduler::maxExperts;
constexpr size_t kMinDeviceMemoryReserveBytes = 256ULL << 20;
constexpr size_t kMaxDeviceMemoryReserveBytes = 2ULL << 30;
constexpr size_t kDeviceMemoryReserveDivisor = 16;

bool PackedCacheRows(const fastllm::Data &data) {
    return data.dims.size() == 2 && data.strides.size() == 2 &&
           data.strides[1] == 1 &&
           (data.dims[0] == 1 || data.strides[0] == data.dims[1]);
}

bool SupportedCacheInput(const fastllm::Data &input) {
    return PackedCacheRows(input) && input.dims[0] > 0 &&
           input.dims[0] <= FASTLLM_CUDA_MOE_CACHE_MAX_BATCH &&
           input.dataDevice == fastllm::DataDevice::CUDA &&
           input.cudaData != nullptr &&
           (input.dataType == fastllm::DataType::FLOAT32 ||
            input.dataType == fastllm::DataType::FLOAT16 ||
            input.dataType == fastllm::DataType::BFLOAT16);
}

size_t AlignUp(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

struct OffloadLayout {
    fastllm::DataType weightType = fastllm::DataType::NVFP4_BLOCK_16_E4M3;
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
    size_t gateScaleBytes = 0;
    size_t downScaleBytes = 0;
    size_t recordStride = 0;
};

const char *FormatName(fastllm::DataType type) {
    if (type == fastllm::DataType::FP8_E4M3) return "FP8 E4M3";
    if (type == fastllm::DataType::FP8_E4M3_BLOCK_128) return "FP8 block128";
    return "NVFP4";
}

int Fp8PointerTableCount(fastllm::DataType type) {
    return type == fastllm::DataType::FP8_E4M3 ? 4 :
        type == fastllm::DataType::FP8_E4M3_BLOCK_128 ? 2 : 0;
}

struct HybridWorkspace {
    fastllm::Data gateOutput;
    float *host = nullptr, *device = nullptr;
    cudaEvent_t start = nullptr, copied = nullptr, computed = nullptr, done = nullptr;
    fastllm::MoeDecodeScheduler scheduler;
    int previousGpu = 0, previousMisses = 0;
    bool pending = false;
    ~HybridWorkspace() {
        if (pending) cudaEventSynchronize(done);
        cudaFreeHost(host);
        cudaFree(device);
        if (start) cudaEventDestroy(start);
        if (copied) cudaEventDestroy(copied);
        if (computed) cudaEventDestroy(computed);
        if (done) cudaEventDestroy(done);
    }
};

struct DecodePolicyState {
    fastllm::MoeDecodePolicy policy;
    int layers, topk;
    double startUs = 0;
    DecodePolicyState(int records, int slots, int layers, int topk)
        : policy(records, slots), layers(layers), topk(topk) {}
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
    // Contiguous slot pointer tables: gate/down weights, then external scales
    // for native E4M3. Packed block128 needs only the two weight tables.
    void **fp8Pointers = nullptr;
    void **numaPointers = nullptr;
    std::unique_ptr<HybridWorkspace> hybrid;
    std::unique_ptr<DecodePolicyState> decode;
};

struct OffloadGroup {
    OffloadLayout layout;
    size_t totalRecords = 0;
    uint8_t *hostRecords = nullptr;
    uint8_t *deviceHostRecords = nullptr;
    std::vector<const fastllm::Data *> tableKeys;
    // When NUMA storage is shared, hostRecords contains only original metadata.
    // The weight shards are borrowed from the model's prefill backend.
    std::vector<void *> numaPointers;
    bool cpuDecodeReady = false;
    fastllm::cuda::SharedExpertLayout sharedLayout;
    std::unordered_map<int, std::unique_ptr<DeviceCache> > deviceCaches;
    std::mutex mutex;
};

struct CachedTable {
    OffloadGroup *group;
    int layer;
};

// A new weight encoding supplies a compute adapter and its shape capability;
// the policy, cache admission, CPU overlap and result merge stay unchanged.
// perExpert requests unweighted FP32 [topk, hidden] results. A negative route
// slot is inactive and must never dereference a weight pointer.
struct SharedCacheStorage {
    bool (*plan)(const OffloadLayout &, fastllm::cuda::SharedExpertLayout &);
    void (*snapshot)(const OffloadLayout &, const fastllm::Data &, const fastllm::Data &, uint8_t *);
    bool (*bind)(const OffloadLayout &, const fastllm::Data &, int, fastllm::cuda::SharedWeightView &);
};

struct ExpertCacheBackend {
    fastllm::DataType type;
    bool (*supportsHybrid)(const OffloadLayout &);
    const SharedCacheStorage *storage;
    bool (*compute)(const fastllm::Data &, fastllm::Data &, fastllm::Data &,
                    const OffloadLayout &, const DeviceCache &, const float *, int, float *);
};
const ExpertCacheBackend *FindExpertCacheBackend(fastllm::DataType type);

bool PlanSharedNVFP4(const OffloadLayout &l, fastllm::cuda::SharedExpertLayout &plan) {
    if (l.gateBlockK != 1 || l.downBlockK != 1 || l.gateBlockM != 16 ||
        l.downBlockM != 16) return false;
    const uint32_t gate = l.gateBytes - fastllm::GetNVFP4WeightBytes(l.inter * 2, l.hidden);
    const uint32_t down = l.downBytes - fastllm::GetNVFP4WeightBytes(l.hidden, l.inter);
    plan.auxiliary[0] = {0, uint32_t(fastllm::GetNVFP4WeightBytes(l.inter * 2, l.hidden)), gate};
    plan.auxiliary[1] = {gate, uint32_t(l.downOffset + fastllm::GetNVFP4WeightBytes(l.hidden, l.inter)), down};
    // Include zero padding in the final span to retain aligned 16-byte loads.
    plan.auxiliary[2] = {gate + down, uint32_t(l.scalesOffset), 16};
    plan.auxiliaryBytes = gate + down + 16;
    return true;
}

void SnapshotNVFP4Metadata(const OffloadLayout &l, const fastllm::Data &gate,
                          const fastllm::Data &down, uint8_t *record) {
    const size_t gateBytes = l.gateBytes - fastllm::GetNVFP4WeightBytes(l.inter * 2, l.hidden);
    const size_t downBytes = l.downBytes - fastllm::GetNVFP4WeightBytes(l.hidden, l.inter);
    memcpy(record, fastllm::GetNVFP4ScaleData(gate), gateBytes);
    memcpy(record + gateBytes, fastllm::GetNVFP4ScaleData(down), downBytes);
    memcpy(record + gateBytes + downBytes, gate.scales.data(), 2 * sizeof(float));
    memcpy(record + gateBytes + downBytes + 2 * sizeof(float), down.scales.data(), sizeof(float));
    memset(record + gateBytes + downBytes + 3 * sizeof(float), 0, sizeof(float));
}

bool BindSharedNVFP4(const OffloadLayout &l, const fastllm::Data &w, int part,
                     fastllm::cuda::SharedWeightView &view) {
    const int columns = part == 0 ? l.hidden : l.inter;
    if (w.blockK != 1 || w.blockM != 16) return false;
    view.rowBytes = (columns + 1) / 2;
    if (w.dataType == fastllm::DataType::NVFP4_BLOCK_16_PLANAR) {
        view.tileRows = fastllm::NVFP4_PLANAR_TILE_ROWS;
        view.tileStride = view.tileRows * ((columns + 15) / 16) * 12;
        view.rowStride = ((columns + 15) / 16) * 8;
        view.blockBytes = view.blockStride = view.rowStride;
    } else if (w.dataType == fastllm::DataType::NVFP4_BLOCK_16) {
        view.tileStride = view.rowStride = ((columns + 15) / 16) * 12;
        view.blockBytes = 8; view.blockStride = 12;
    } else return false;
    return true;
}

bool PlanSharedFP8(const OffloadLayout &l, fastllm::cuda::SharedExpertLayout &plan) {
    if (l.weightType == fastllm::DataType::FP8_E4M3) {
        if ((l.gateBlockM != 128 && l.gateBlockM != l.hidden) ||
            (l.downBlockM != 128 && l.downBlockM != l.inter)) return false;
        plan.auxiliaryBytes = AlignUp(l.gateScaleBytes + l.downScaleBytes, 16);
        plan.auxiliary[0] = {0, uint32_t(l.scalesOffset), plan.auxiliaryBytes};
    }
    return true;
}

void SnapshotFP8Metadata(const OffloadLayout &l, const fastllm::Data &gate,
                        const fastllm::Data &down, uint8_t *record) {
    if (!l.gateScaleBytes) return;
    memcpy(record, gate.scales.data(), l.gateScaleBytes);
    memcpy(record + l.gateScaleBytes, down.scales.data(), l.downScaleBytes);
    const size_t bytes = l.gateScaleBytes + l.downScaleBytes;
    memset(record + bytes, 0, AlignUp(bytes, 16) - bytes);
}

bool BindSharedFP8(const OffloadLayout &l, const fastllm::Data &w, int part,
                   fastllm::cuda::SharedWeightView &view) {
    const uint32_t columns = part == 0 ? l.hidden : l.inter;
    if (l.weightType == fastllm::DataType::FP8_E4M3_BLOCK_128) {
        // Identical packed row payload: only undo gate/up row interleaving.
        if (w.dataType != l.weightType) return false;
        view.rowBytes = fastllm::GetDataBytes(w.dataType, 1, columns);
        view.tileStride = view.rowStride = view.blockBytes = view.blockStride = view.rowBytes;
    } else {
        const int block = part == 0 ? l.gateBlockM : l.downBlockM;
        const auto expected = block == 128 ? fastllm::DataType::FP8_E4M3_BLOCK_128
                                         : fastllm::DataType::FP8_E4M3_PERCHANNEL;
        if (w.dataType != expected) return false;
        view.rowBytes = columns;
        view.blockBytes = block;
        view.blockStride = block + sizeof(float);
        view.tileStride = view.rowStride = fastllm::GetDataBytes(w.dataType, 1, columns);
    }
    return true;
}

bool BindNumaWeights(OffloadGroup &group, const std::vector<fastllm::Data *> &weights,
                     const SharedCacheStorage &storage) {
    const OffloadLayout &layout = group.layout;
    const int nodes = weights.front()->numasData.size();
    if (nodes <= 0 || (layout.inter * 2) % nodes || layout.hidden % nodes) return false;
    auto &plan = group.sharedLayout;
    plan.shards = nodes;
    plan.recordBytes = layout.recordStride;
    for (int part = 0; part < 2; ++part) {
        auto &view = plan.weights[part];
        view.rows = part == 0 ? layout.inter * 2 : layout.hidden;
        view.rowsPerShard = view.rows / nodes;
        view.rowGroups = part == 0 ? 2 : 1;
        view.destinationOffset = part == 0 ? 0 : layout.downOffset;
        if (!storage.bind(layout, *weights[part], part, view)) return false;
    }
    if (!plan.Prepare()) return false;
    std::vector<void *> pointers;
    pointers.reserve(weights.size() * nodes);
    for (size_t index = 0; index < weights.size(); ++index) {
        const fastllm::Data &weight = *weights[index];
        const int part = index % 2;
        const int rows = part == 0 ? 2 * layout.inter : layout.hidden;
        const int columns = part == 0 ? layout.hidden : layout.inter;
        if (weight.dataType != weights[part]->dataType ||
            weight.blockK != weights[part]->blockK || weight.blockM != weights[part]->blockM ||
            weight.dims.size() != 2 || weight.dims[0] != rows || weight.dims[1] != columns ||
            weight.dataDevice != fastllm::DataDevice::CPU ||
            !weight.isPinned || weight.cpuData != nullptr ||
            weight.numasData.size() != size_t(nodes)) return false;
        for (uint8_t *source : weight.numasData) {
            void *mapped = nullptr;
            if (source == nullptr || (reinterpret_cast<uintptr_t>(source) & 15) != 0 ||
                cudaHostGetDevicePointer(&mapped, source, 0) != cudaSuccess || !mapped) {
                cudaGetLastError();
                return false;
            }
            pointers.push_back(source);
        }
    }
    group.numaPointers = std::move(pointers);
    return true;
}

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
        (gate->dataType != fastllm::DataType::NVFP4_BLOCK_16_E4M3 &&
         gate->dataType != fastllm::DataType::FP8_E4M3 &&
         gate->dataType != fastllm::DataType::FP8_E4M3_BLOCK_128) ||
        down->dataType != gate->dataType ||
        gate->dataDevice != fastllm::DataDevice::CPU ||
        down->dataDevice != fastllm::DataDevice::CPU ||
        gate->cpuData == nullptr || down->cpuData == nullptr ||
        gate->dims.size() != 2 || down->dims.size() != 2 ||
        gate->dims[0] <= 0 || gate->dims[1] <= 0 ||
        down->dims[0] <= 0 || down->dims[1] <= 0 ||
        down->dims[1] > INT_MAX / 2 ||
        gate->dims[0] != down->dims[1] * 2 ||
        gate->dims[1] != down->dims[0]) {
        return false;
    }

    observed.weightType = gate->dataType;
    const bool block128 = gate->dataType == fastllm::DataType::FP8_E4M3_BLOCK_128;
    if (!block128 && (gate->blockK <= 0 || gate->blockM <= 0 ||
                     down->blockK <= 0 || down->blockM <= 0)) return false;
    if (gate->dataType == fastllm::DataType::NVFP4_BLOCK_16_E4M3 &&
        (gate->scales.size() != 2 || down->scales.size() != 1)) return false;
    observed.hidden = gate->dims[1];
    observed.inter = down->dims[1];
    observed.gateBlockK = block128 ? 1 : gate->blockK;
    observed.gateBlockM = block128 ? 128 : gate->blockM;
    observed.downBlockK = block128 ? 1 : down->blockK;
    observed.downBlockM = block128 ? 128 : down->blockM;
    // Existing indexed FP8 kernels load four weights at a time. Keep their
    // alignment and scale-block assumptions explicit; unsupported layouts
    // retain the configured backend rather than issuing misaligned reads.
    if (observed.weightType != fastllm::DataType::NVFP4_BLOCK_16_E4M3 &&
        ((observed.hidden & 3) || (observed.inter & 3) ||
         (observed.gateBlockM & 3) || (observed.downBlockM & 3))) return false;
    // The existing packed block128 indexed kernels expect complete blocks.
    if (block128 && ((observed.hidden & 127) || (observed.inter & 127))) return false;
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
    if (gate->dataType == fastllm::DataType::NVFP4_BLOCK_16_E4M3) {
        observed.gateScaleBytes = 2 * sizeof(float);
        observed.downScaleBytes = sizeof(float);
    } else if (gate->dataType == fastllm::DataType::FP8_E4M3) {
        const size_t gateScaleCount =
            ((size_t(gate->dims[0]) - 1) / gate->blockK + 1) * observed.gateScaleCols;
        const size_t downScaleCount =
            ((size_t(down->dims[0]) - 1) / down->blockK + 1) * observed.downScaleCols;
        if (gate->scales.size() != gateScaleCount || down->scales.size() != downScaleCount)
            return false;
        observed.gateScaleBytes = gateScaleCount * sizeof(float);
        observed.downScaleBytes = downScaleCount * sizeof(float);
    }
    // All formats retain 128-byte alignment at every host record boundary.
    observed.recordStride = AlignUp(observed.scalesOffset +
        observed.gateScaleBytes + observed.downScaleBytes, 128);
    if (observed.gateBytes == 0 || observed.downBytes == 0) {
        return false;
    }
    if (expected == nullptr) {
        return true;
    }
    return expected->weightType == observed.weightType &&
           expected->hidden == observed.hidden &&
           expected->inter == observed.inter &&
           expected->gateBlockK == observed.gateBlockK &&
           expected->gateBlockM == observed.gateBlockM &&
           expected->downBlockK == observed.downBlockK &&
           expected->downBlockM == observed.downBlockM &&
           expected->gateBytes == observed.gateBytes &&
           expected->downBytes == observed.downBytes &&
           expected->downOffset == observed.downOffset &&
           expected->scalesOffset == observed.scalesOffset &&
           expected->gateScaleBytes == observed.gateScaleBytes &&
           expected->downScaleBytes == observed.downScaleBytes &&
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
    cache.hybrid.reset();
    cache.decode.reset();
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
    cudaFree(cache.fp8Pointers);
    cudaFree(cache.numaPointers);
    cache = DeviceCache();
}

void PrintDeviceCacheStats(const DeviceCache &cache, fastllm::DataType weightType) {
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
        "[Fastllm] %s GPU expert cache stats cuda:%d: "
        "%llu hits, %llu misses, %.3f%% hit rate, %d slots.\n",
        FormatName(weightType), cache.device, hits, misses, hitRate, cache.slots);
    if (cache.decode) {
        const auto &policy = cache.decode->policy;
        std::fprintf(stderr, "[Fastllm] MoE decode policy: %llu hybrid steps, %llu GPU steps.\n",
            (unsigned long long)policy.hybridSteps, (unsigned long long)policy.gpuSteps);
    }
}

bool AllocateOne(void **pointer, size_t bytes) {
    *pointer = nullptr;
    return bytes > 0 && cudaMalloc(pointer, bytes) == cudaSuccess &&
           *pointer != nullptr;
}

bool PrepareFp8SlotPointers(DeviceCache &cache, const OffloadLayout &layout) {
    const int tables = Fp8PointerTableCount(layout.weightType);
    if (tables == 0) return true;
    const size_t slots = cache.slots;
    std::vector<void *> pointers(slots * tables);
    for (size_t slot = 0; slot < slots; ++slot) {
        uint8_t *record = cache.records + size_t(slot) * layout.recordStride;
        pointers[slot] = record;
        pointers[slots + slot] = record + layout.downOffset;
        if (tables == 4) {
            pointers[2 * slots + slot] = record + layout.scalesOffset;
            pointers[3 * slots + slot] = record + layout.scalesOffset + layout.gateScaleBytes;
        }
    }
    const size_t bytes = pointers.size() * sizeof(void *);
    return AllocateOne(reinterpret_cast<void **>(&cache.fp8Pointers), bytes) &&
        cudaMemcpy(cache.fp8Pointers, pointers.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess;
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
            "[Fastllm] CUDA expert cache needs at least %d slots; "
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
        group.numaPointers.size() * sizeof(void *) +
        slots * (sizeof(int32_t) + sizeof(unsigned long long) +
            Fp8PointerTableCount(group.layout.weightType) * sizeof(void *)) + 4096;
    size_t usableBytes = freeBytes > reserveBytes
        ? freeBytes - reserveBytes : 0;
    usableBytes = usableBytes > metadataBytes
        ? usableBytes - metadataBytes : 0;
    slots = std::min(slots, usableBytes / group.layout.recordStride);
    if (slots < kMaxTopK) {
        std::fprintf(stderr,
            "[Fastllm] CUDA expert cache has insufficient free GPU "
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

    bool initialized =
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
            sizeof(unsigned long long), cudaStreamPerThread) == cudaSuccess &&
        PrepareFp8SlotPointers(cache, group.layout);
    if (initialized && !group.numaPointers.empty()) {
        std::vector<void *> pointers(group.numaPointers.size());
        for (size_t i = 0; initialized && i < pointers.size(); ++i) {
            initialized = cudaHostGetDevicePointer(
                &pointers[i], group.numaPointers[i], 0) == cudaSuccess;
        }
        const size_t bytes = pointers.size() * sizeof(void *);
        initialized = initialized &&
            AllocateOne(reinterpret_cast<void **>(&cache.numaPointers), bytes) &&
            cudaMemcpy(cache.numaPointers, pointers.data(), bytes,
                       cudaMemcpyHostToDevice) == cudaSuccess;
    }
    if (!initialized) {
        std::fprintf(stderr,
            "[Fastllm] CUDA expert cache allocation or initialization failed: "
            "%s.\n", cudaGetErrorString(cudaGetLastError()));
        ReleaseDeviceCache(cache);
        cache.attempted = true;
        cache.device = device;
        return nullptr;
    }

    cache.ready = true;

    std::fprintf(stderr,
        "[Fastllm] %s GPU expert cache: %d slots, %.3f GiB, "
        "%zu records in mapped host storage; refill grid %d x %d.\n",
        FormatName(group.layout.weightType), cache.slots,
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

OffloadGroup *FindHybridGroup(fastllm::Data **weights, int weightsBatch) {
#ifdef USE_NUMAS
    if (!FastllmCudaMoeCacheRequested()) return nullptr;
    auto *group = FindGroup(weights, weightsBatch);
    if (!group || !group->cpuDecodeReady) return nullptr;
    const auto *backend = FindExpertCacheBackend(group->layout.weightType);
    return backend && backend->supportsHybrid(group->layout) ? group : nullptr;
#else
    return nullptr;
#endif
}

template <typename T>
struct ActivationTraits;

template <>
struct ActivationTraits<float> {
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
    __device__ static float ToFloat(half value) { return __half2float(value); }
    __device__ static half FromFloat(float value) { return __float2half(value); }
};

template <>
struct ActivationTraits<__nv_bfloat16> {
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
        size_t recordStride, size_t downOffset, size_t scalesOffset,
        float *perExpert = nullptr) {
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
            if (perExpert != nullptr) perExpert[route * hidden + row] = rounded;
            expertOutputs[route] = rounded * scores[route];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && perExpert == nullptr) {
        float sum = 0.0f;
        for (int item = 0; item < topk; ++item) {
            sum += expertOutputs[item];
        }
        output[row] = ActivationTraits<T>::FromFloat(sum);
    }
}

bool SupportsNVFP4WideDecode(const OffloadLayout &l) {
    return l.gateBlockK == 1 && l.downBlockK == 1 && l.gateBlockM == 16 &&
        l.downBlockM == 16 && l.hidden % 128 == 0 && l.inter % 128 == 0;
}

template <typename T>
bool LaunchNVFP4Cache(const fastllm::Data &input, fastllm::Data &gateOutput,
                       fastllm::Data &output, const OffloadLayout &layout,
                       const DeviceCache &cache, const float *scores, int topk,
                       float *perExpert = nullptr) {
    if (SupportsNVFP4WideDecode(layout)) {
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
            layout.scalesOffset, perExpert);
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
        const FastllmCudaMoeCacheLayer *layers, int layerCount,
        const std::function<void()> &registerNumaWeights) {
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
                    "requires matching host-resident NVFP4 or FP8 weights "
                    "and valid scales (gate dtype=%d, down dtype=%d).\n",
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
            "[Fastllm] CUDA expert cache needs at least %d slots; "
            "the configured budget holds %zu.\n",
            kMaxTopK, requestedSlots);
        return false;
    }
    // A registered backend supplies byte layouts and only the metadata that
    // its CPU representation cannot preserve. Never snapshot shared weights,
    // including transiently during load; registration retains sole ownership.
    const auto *backend = FindExpertCacheBackend(layout.weightType);
    const auto *storage = backend ? backend->storage : nullptr;
    const bool shareNuma = bool(registerNumaWeights);
    if (shareNuma && (!storage || layout.recordStride > UINT32_MAX / kMaxTopK ||
                      !storage->plan(layout, group->sharedLayout))) return false;
    const size_t hostStride = shareNuma ? group->sharedLayout.auxiliaryBytes : layout.recordStride;
    const size_t hostBytes = group->totalRecords * hostStride;
    void *host = nullptr;
    std::unique_ptr<void, decltype(&cudaFreeHost)> hostOwner(nullptr, cudaFreeHost);
    if (hostBytes) {
        cudaError_t state = cudaHostAlloc(&host, hostBytes, cudaHostAllocMapped | cudaHostAllocPortable);
        if (state != cudaSuccess || !host) {
            std::fprintf(stderr, "[Fastllm] CUDA expert cache could not allocate %.3f GiB "
                         "of mapped host metadata: %s.\n", double(hostBytes) / (1ULL << 30),
                         cudaGetErrorString(state));
            cudaGetLastError();
            return false;
        }
        hostOwner.reset(host);
        group->hostRecords = static_cast<uint8_t *>(host);
        void *mapped = nullptr;
        if (cudaHostGetDevicePointer(&mapped, host, 0) != cudaSuccess || !mapped) {
            cudaGetLastError();
            return false;
        }
        group->deviceHostRecords = static_cast<uint8_t *>(mapped);
    }
    group->tableKeys.reserve(layerCount);
    std::vector<fastllm::Data *> expertWeights;
    if (shareNuma) expertWeights.reserve(group->totalRecords * 2);
    for (int layer = 0; layer < layerCount; ++layer) {
        group->tableKeys.push_back(layers[layer].weights[2]);
        for (int expert = 0; expert < experts; ++expert) {
            const int position = (expert + 1) * 2;
            fastllm::Data *gate = layers[layer].weights[position];
            fastllm::Data *down = layers[layer].weights[position + 1];
            uint8_t *record = hostBytes ? group->hostRecords +
                (size_t(layer) * experts + expert) * hostStride : nullptr;
            if (shareNuma) {
                expertWeights.push_back(gate);
                expertWeights.push_back(down);
                storage->snapshot(layout, *gate, *down, record);
            } else {
                memcpy(record, gate->cpuData, layout.gateBytes);
                memcpy(record + layout.downOffset, down->cpuData, layout.downBytes);
                if (layout.gateScaleBytes) {
                    memcpy(record + layout.scalesOffset, gate->scales.data(), layout.gateScaleBytes);
                    memcpy(record + layout.scalesOffset + layout.gateScaleBytes,
                           down->scales.data(), layout.downScaleBytes);
                }
            }
        }
    }
    if (shareNuma) {
        registerNumaWeights();
        if (!BindNumaWeights(*group, expertWeights, *storage)) {
            std::fprintf(stderr, "[Fastllm] CUDA expert cache requires matching pinned NUMA "
                         "shards after registration; using the configured MoE backend.\n");
            return false;
        }
#ifdef USE_NUMAS
        group->cpuDecodeReady = true;
        for (int layer = 0; layer < layerCount; ++layer)
            group->cpuDecodeReady &= fastllm::CanRunNumasMoeDecodeExperts(
                layers[layer].weights, layers[layer].weightsBatch);
#endif
        std::fprintf(stderr, "[Fastllm] %s expert cache shares %d NUMA shards per weight; "
                     "metadata uses %.3f GiB, avoiding a %.3f GiB host weight snapshot.\n",
                     FormatName(layout.weightType), group->sharedLayout.shards, double(hostBytes) / (1ULL << 30),
                     double(group->totalRecords * (layout.recordStride - hostStride)) / (1ULL << 30));
    }
    OffloadGroup *rawGroup = group.get();
    for (int layer = 0; layer < layerCount; ++layer) {
        TableRegistry()[group->tableKeys[layer]] = {rawGroup, layer};
    }
    Groups().push_back(std::move(group));
    hostOwner.release();
    std::fprintf(stderr,
        "[Fastllm] Prepared %d x %d %s experts in %.3f "
        "GiB of mapped host storage (record %zu bytes).\n",
        layerCount, experts, FormatName(layout.weightType),
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

bool FastllmCudaCanRunMoeHybrid(fastllm::Data **weights, int weightsBatch) {
    return FindHybridGroup(weights, weightsBatch) != nullptr;
}

bool FastllmCudaUseMoeHybrid(fastllm::Data **weights, int weightsBatch) {
    auto *group = FindHybridGroup(weights, weightsBatch);
    if (!group) return false;
    auto *cache = GetDeviceCache(*group);
    return cache && (!cache->decode || !cache->decode->policy.UseGpu());
}

bool FastllmCudaCanRunMoeCacheSmallBatch(
        const fastllm::Data &input, const fastllm::Data &index,
        const fastllm::Data &score, fastllm::Data **weights,
        int weightsBatch, fastllm::MoeGateType gateType) {
    const bool supportedInput = FastllmCudaMoeCacheRequested() &&
           gateType == fastllm::MoeGateSwiglu && SupportedCacheInput(input) &&
           index.dims.size() == 2 && index.dims[0] == input.dims[0] &&
           index.dims[1] > 0 && index.dims[1] <= kMaxTopK &&
           index.dataDevice == fastllm::DataDevice::CUDA &&
           index.dataType == fastllm::DataType::INT32 &&
           index.cudaData != nullptr && score.dims == index.dims &&
           score.dataDevice == fastllm::DataDevice::CUDA &&
           score.dataType == fastllm::DataType::FLOAT32 &&
           score.cudaData != nullptr &&
           PackedCacheRows(index) && PackedCacheRows(score);
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
        PrintDeviceCacheStats(*cache.second, released->layout.weightType);
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

namespace {
bool ComputeNVFP4Cache(const fastllm::Data &input, fastllm::Data &gateOutput,
        fastllm::Data &output, const OffloadLayout &layout, const DeviceCache &cache,
        const float *scores, int topk, float *perExpert) {
    switch (input.dataType) {
        case fastllm::DataType::FLOAT32:
            return LaunchNVFP4Cache<float>(input, gateOutput, output, layout, cache, scores, topk, perExpert);
        case fastllm::DataType::FLOAT16:
            return LaunchNVFP4Cache<half>(input, gateOutput, output, layout, cache, scores, topk, perExpert);
        case fastllm::DataType::BFLOAT16:
            return LaunchNVFP4Cache<__nv_bfloat16>(input, gateOutput, output, layout, cache, scores, topk, perExpert);
        default:
            return false;
    }
}

bool ComputeFP8Cache(const fastllm::Data &input, fastllm::Data &gateOutput,
        fastllm::Data &output, const OffloadLayout &layout, const DeviceCache &cache,
        const float *scores, int topk, float *perExpert) {
    const size_t slots = cache.slots;
    const bool externalScales = layout.weightType == fastllm::DataType::FP8_E4M3;
    const FastllmCudaMoeFP8CacheView view{
        layout.weightType,
        reinterpret_cast<uint8_t **>(cache.fp8Pointers),
        reinterpret_cast<uint8_t **>(cache.fp8Pointers + slots),
        externalScales ? reinterpret_cast<float **>(cache.fp8Pointers + 2 * slots) : nullptr,
        externalScales ? reinterpret_cast<float **>(cache.fp8Pointers + 3 * slots) : nullptr,
        layout.gateBlockM, layout.gateBlockK,
        layout.downBlockM, layout.downBlockK};
    return FastllmCudaMoeFP8CacheCompute(
        input, gateOutput, output, view, cache.routeSlots, scores, topk, perExpert);
}

bool FP8HybridShape(const OffloadLayout &) { return true; }

const ExpertCacheBackend *FindExpertCacheBackend(fastllm::DataType type) {
    static const SharedCacheStorage nvfp4{PlanSharedNVFP4, SnapshotNVFP4Metadata, BindSharedNVFP4};
    static const SharedCacheStorage fp8{PlanSharedFP8, SnapshotFP8Metadata, BindSharedFP8};
    static const ExpertCacheBackend backends[] = {
        {fastllm::DataType::NVFP4_BLOCK_16_E4M3, SupportsNVFP4WideDecode, &nvfp4, ComputeNVFP4Cache},
        {fastllm::DataType::FP8_E4M3, FP8HybridShape, &fp8, ComputeFP8Cache},
        {fastllm::DataType::FP8_E4M3_BLOCK_128, FP8HybridShape, &fp8, ComputeFP8Cache},
    };
    for (const auto &backend : backends) if (backend.type == type) return &backend;
    return nullptr;
}

bool EnsureCachedExperts(OffloadGroup *group, DeviceCache *cache, int tableId,
                         const int32_t *indices, int topk) {
    const auto &layout = group->layout;
    const fastllm::cuda::ExpertCacheView metadata{
        cache->keyToSlot, cache->slotKeys, cache->lastUsed, cache->step,
        cache->hitCount, cache->totalMissCount, cache->slots};
    if (!fastllm::cuda::EnsureExpertCache<kMaxTopK>(
            metadata, indices, tableId * layout.experts, layout.experts, topk,
            cache->routeSlots, cache->missExperts, cache->missSlots, cache->missCount,
            cache->ensureThreads, cudaStreamPerThread)) return false;
    const auto &shared = group->sharedLayout;
    if (shared.shards > 0) {
        void *const *pointers = cache->numaPointers +
            size_t(tableId) * layout.experts * 2 * shared.shards;
        const uint8_t *metadata = shared.auxiliaryBytes ? group->deviceHostRecords +
            size_t(tableId) * layout.experts * shared.auxiliaryBytes : nullptr;
        return fastllm::cuda::CopySharedExpertRecords(shared, pointers, metadata,
            cache->records, cache->missExperts, cache->missSlots, cache->missCount,
            topk, cache->copyLaunch, cudaStreamPerThread);
    }
    const uint8_t *sourceTable = group->deviceHostRecords +
        size_t(tableId) * layout.experts * layout.recordStride;
    if (!fastllm::cuda::CopyRecords(
            {sourceTable, cache->records, layout.recordStride,
             layout.recordStride, layout.recordStride},
            cache->missExperts, cache->missSlots, cache->missCount, topk,
            cache->copyLaunch, cudaStreamPerThread)) return false;
    return cudaGetLastError() == cudaSuccess;
}

__global__ void LookupHybridRoutes(const int32_t *indices, const int32_t *keys,
        const int32_t *slotKeys, int32_t *result, int base, int experts, int topk) {
    const int r = threadIdx.x;
    if (r < topk) {
        const int expert = indices[r];
        const int key = base + expert;
        const int slot = expert >= 0 && expert < experts ? keys[key] : -1;
        result[r] = expert;
        result[kMaxTopK + r] = slot >= 0 && slotKeys[slot] == key ? slot : -1;
    }
}

__global__ void ReduceHybridExperts(const float *cpu, const float *gpu,
        const int32_t *gpuIndices, const float *scores, float *output, int hidden, int topk) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= hidden) return;
    float sum = 0;
    for (int r = 0; r < topk; ++r) {
        const float value = (gpuIndices[r] < 0 ? cpu : gpu)[r * hidden + col];
        sum = __fadd_rn(sum, __fmul_rn(value, scores[r]));
    }
    output[col] = sum;
}

double HybridNowUs() {
    return std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

} // namespace

void *FastllmCudaBeginMoeDecode(fastllm::Data **weights, int weightsBatch, int topk) {
    if (topk < 1 || topk > kMaxTopK) return nullptr;
    auto *group = FindHybridGroup(weights, weightsBatch);
    if (!group) return nullptr;
    auto *cache = GetDeviceCache(*group);
    if (!cache) return nullptr;
    if (!cache->decode) {
        cache->decode = std::make_unique<DecodePolicyState>(
            group->totalRecords, cache->slots, group->tableKeys.size(), topk);
    }
    cache->decode->startUs = HybridNowUs();
    return cache;
}

void FastllmCudaEndMoeDecode(void *state) {
    if (!state) return;
    auto &cache = *static_cast<DeviceCache *>(state);
    auto &decode = *cache.decode;
    auto &policy = decode.policy;
    const auto before = policy.GetMode();
    const auto *scheduler = cache.hybrid ? &cache.hybrid->scheduler : nullptr;
    policy.ObserveStep(HybridNowUs() - decode.startUs,
        scheduler ? scheduler->refill.us : 0,
        scheduler ? scheduler->compute[decode.topk].us : 0, decode.layers, decode.topk);
    const auto after = policy.GetMode();
    // Report completed decisions; warmup estimates are intentionally discarded.
    if (before != after && (after == fastllm::MoeDecodePolicy::Mode::Hybrid ||
                            after == fastllm::MoeDecodePolicy::Mode::Gpu)) {
        std::fprintf(stderr, "[Fastllm] MoE decode policy -> %s: hybrid %.3f ms, GPU %.3f ms, "
            "estimated full-route miss rate %.3f%%.\n", policy.UseGpu() ? "GPU cache" : "hybrid",
            policy.HybridUs() / 1000, policy.GpuUs() / 1000, policy.MissRate() * 100);
    }
}

bool FastllmCudaMergeMOEHybrid(const fastllm::Data &input,
        const fastllm::Data &index, const fastllm::Data &score,
        fastllm::Data &output, fastllm::Data **weights, int weightsBatch, int layer) {
#ifdef USE_NUMAS
    if (input.dataType != fastllm::DataType::FLOAT32 || input.dims.size() != 2 ||
        input.dims[0] != 1 || !FastllmCudaCanRunMoeHybrid(weights, weightsBatch) ||
        !FastllmCudaCanRunMoeCacheSmallBatch(
            input, index, score, weights, weightsBatch, fastllm::MoeGateSwiglu)) return false;
    cudaStreamCaptureStatus capturing;
    if (cudaStreamIsCapturing(cudaStreamPerThread, &capturing) != cudaSuccess ||
        capturing != cudaStreamCaptureStatusNone) return false;
    int tableId;
    auto *group = FindGroup(weights, weightsBatch, &tableId);
    auto *cache = GetDeviceCache(*group);
    if (cache->decode && cache->decode->policy.UseGpu()) return false;
    const auto &layout = group->layout;
    const int hidden = layout.hidden, topk = index.dims[1];
    if (!cache->hybrid) {
        auto work = std::make_unique<HybridWorkspace>();
        const size_t metaBytes = 3 * kMaxTopK * sizeof(int32_t);
        if (cudaMallocHost(&work->host, (kMaxTopK + 1) * hidden * sizeof(float) + metaBytes) != cudaSuccess ||
            cudaMalloc(&work->device, 2 * kMaxTopK * hidden * sizeof(float) + metaBytes) != cudaSuccess ||
            cudaEventCreate(&work->start) != cudaSuccess ||
            cudaEventCreate(&work->copied) != cudaSuccess ||
            cudaEventCreate(&work->computed) != cudaSuccess ||
            cudaEventCreate(&work->done) != cudaSuccess) return false;
        cache->hybrid = std::move(work);
    }
    auto &work = *cache->hybrid;
    if (work.pending) {
        fastllm::AssertInFastLLM(cudaEventSynchronize(work.done) == cudaSuccess,
                               "Hybrid MoE completion failed.\n");
        if (work.previousGpu > 0) {
            float copyMs, computeMs;
            checkCudaErrors("Hybrid MoE", cudaEventElapsedTime(&copyMs, work.start, work.copied));
            checkCudaErrors("Hybrid MoE", cudaEventElapsedTime(&computeMs, work.copied, work.computed));
            if (work.previousMisses > 0)
                work.scheduler.refill.Observe(copyMs * 1000 / work.previousMisses);
            else work.scheduler.ensure.Observe(copyMs * 1000);
            work.scheduler.compute[work.previousGpu].Observe(computeMs * 1000);
        }
    }
    auto *hostIndices = reinterpret_cast<int32_t *>(work.host + (kMaxTopK + 1) * hidden);
    auto *resident = hostIndices + kMaxTopK;
    auto *gpuIndices = resident + kMaxTopK;
    auto *deviceMeta = reinterpret_cast<int32_t *>(work.device + 2 * kMaxTopK * hidden);
    auto *deviceGpuIndices = deviceMeta + 2 * kMaxTopK;
    float *cpuOutput = work.host + hidden;
    float *deviceCpuOutput = work.device;
    float *deviceGpuOutput = work.device + kMaxTopK * hidden;
    LookupHybridRoutes<<<1, 32, 0, cudaStreamPerThread>>>(
        static_cast<const int32_t *>(index.cudaData), cache->keyToSlot, cache->slotKeys,
        deviceMeta, tableId * layout.experts, layout.experts, topk);
    checkCudaErrors("Hybrid MoE", cudaMemcpyAsync(hostIndices, deviceMeta, 2 * kMaxTopK * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, cudaStreamPerThread));
    checkCudaErrors("Hybrid MoE", cudaMemcpyAsync(work.host, input.cudaData, hidden * sizeof(float),
                    cudaMemcpyDeviceToHost, cudaStreamPerThread));
    fastllm::AssertInFastLLM(cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess,
                           "Hybrid MoE routing copy failed.\n");
    int hits = 0;
    std::array<int, kMaxTopK> order;
    for (int r = 0; r < topk; ++r) {
        fastllm::AssertInFastLLM(hostIndices[r] >= 0 && hostIndices[r] < layout.experts,
                               "Hybrid MoE received an invalid expert index.\n");
        hits += resident[r] >= 0;
        gpuIndices[r] = -1;
    }
    if (cache->decode) cache->decode->policy.ObserveRoutes(tableId * layout.experts, hostIndices, topk);
    int next = 0;
    for (int r = 0; r < topk; ++r) if (resident[r] >= 0) order[next++] = r;
    for (int r = 0; r < topk; ++r) if (resident[r] < 0) order[next++] = r;
    const int gpu = work.scheduler.SelectGpuCount(topk, hits, group->tableKeys.size());
    for (int i = 0; i < gpu; ++i) gpuIndices[order[i]] = hostIndices[order[i]];
    output.dataType = input.dataType;
    output.dataDevice = input.dataDevice;
    output.dataDeviceIds = input.dataDeviceIds;
    output.Resize({1, hidden});
    output.Allocate(false);
    checkCudaErrors("Hybrid MoE", cudaMemcpyAsync(deviceGpuIndices, gpuIndices, topk * sizeof(int32_t),
                    cudaMemcpyHostToDevice, cudaStreamPerThread));
    auto &gateOutput = work.gateOutput;
    if (gpu > 0) {
        const double dispatchStart = HybridNowUs();
        gateOutput.dataType = input.dataType;
        gateOutput.dataDevice = input.dataDevice;
        gateOutput.dataDeviceIds = input.dataDeviceIds;
        gateOutput.Resize({topk, layout.inter});
        gateOutput.Allocate(false);
        checkCudaErrors("Hybrid MoE", cudaEventRecord(work.start, cudaStreamPerThread));
        fastllm::AssertInFastLLM(EnsureCachedExperts(group, cache, tableId, deviceGpuIndices, topk),
                               "Hybrid MoE refill failed.\n");
        checkCudaErrors("Hybrid MoE", cudaEventRecord(work.copied, cudaStreamPerThread));
        fastllm::AssertInFastLLM(FindExpertCacheBackend(layout.weightType)->compute(
            input, gateOutput, output, layout, *cache,
            static_cast<const float *>(score.cudaData), topk, deviceGpuOutput),
            "Hybrid MoE CUDA experts failed.\n");
        checkCudaErrors("Hybrid MoE", cudaEventRecord(work.computed, cudaStreamPerThread));
        work.scheduler.dispatch.Observe(HybridNowUs() - dispatchStart);
    }
    const double cpuStart = HybridNowUs();
    fastllm::NumasMoeDecodeExperts(work.host, cpuOutput, weights,
                                   hostIndices, gpuIndices, topk, layer);
    if (gpu < topk) {
        work.scheduler.cpu[topk - gpu].Observe(HybridNowUs() - cpuStart);
        checkCudaErrors("Hybrid MoE", cudaMemcpyAsync(deviceCpuOutput, cpuOutput, topk * hidden * sizeof(float),
                        cudaMemcpyHostToDevice, cudaStreamPerThread));
    }
    ReduceHybridExperts<<<(hidden + 255) / 256, 256, 0, cudaStreamPerThread>>>(
        deviceCpuOutput, deviceGpuOutput, deviceGpuIndices,
        static_cast<const float *>(score.cudaData), static_cast<float *>(output.cudaData), hidden, topk);
    checkCudaErrors("Hybrid MoE reduction", cudaGetLastError());
    checkCudaErrors("Hybrid MoE completion", cudaEventRecord(work.done, cudaStreamPerThread));
    work.pending = true;
    work.previousGpu = gpu;
    work.previousMisses = std::max(0, gpu - hits);
    ++work.scheduler.calls;
    return true;
#else
    return false;
#endif
}

bool FastllmCudaMergeMOECache(
        const fastllm::Data &input, fastllm::Data &gateOutput,
        fastllm::Data &output, fastllm::Data **weights, int weightsBatch,
        const int32_t *indices, const float *scores, int topk) {
    if (!SupportedCacheInput(input) || indices == nullptr || scores == nullptr ||
        topk <= 0 || topk > kMaxTopK) {
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
    const OffloadLayout &layout = group->layout;
    gateOutput.dataType = input.dataType;
    gateOutput.dataDevice = input.dataDevice;
    gateOutput.dataDeviceIds = input.dataDeviceIds;
    const int rows = input.dims[0];
    gateOutput.Resize({rows * topk, layout.inter});
    output.dataType = input.dataType;
    output.dataDevice = input.dataDevice;
    output.dataDeviceIds = input.dataDeviceIds;
    output.Resize({rows, layout.hidden});
    gateOutput.Allocate(false);
    output.Allocate(false);
    if (gateOutput.cudaData == nullptr || output.cudaData == nullptr) {
        return false;
    }

    auto computeRow = [&](const fastllm::Data &input, fastllm::Data &gateOutput,
                          fastllm::Data &output, const int32_t *indices,
                          const float *scores) {
        if (!EnsureCachedExperts(group, cache, tableId, indices, topk)) return false;
        const auto *backend = FindExpertCacheBackend(layout.weightType);
        return backend && backend->compute(input, gateOutput, output, layout, *cache, scores, topk, nullptr);
    };
    if (rows == 1) {
        return computeRow(input, gateOutput, output, indices, scores);
    }

    // Each row completes lookup, refill and compute on the same stream
    // before the next row reuses route metadata or evicts an expert. This
    // also preserves the single-token kernels' reduction order in a graph.
    fastllm::Data inputRow, gateRow, outputRow;
    for (int row = 0; row < rows; ++row) {
        inputRow.FakeFrom(input, size_t(row) * layout.hidden * input.unitSize);
        inputRow.Resize({1, layout.hidden});
        inputRow.dataDeviceIds = input.dataDeviceIds;
        gateRow.FakeFrom(gateOutput, size_t(row) * topk * layout.inter * input.unitSize);
        gateRow.Resize({topk, layout.inter});
        gateRow.dataDeviceIds = input.dataDeviceIds;
        outputRow.FakeFrom(output, size_t(row) * layout.hidden * input.unitSize);
        outputRow.Resize({1, layout.hidden});
        outputRow.dataDeviceIds = input.dataDeviceIds;
        if (!computeRow(inputRow, gateRow, outputRow,
                        indices + row * topk, scores + row * topk)) {
            return false;
        }
    }
    return true;
}
