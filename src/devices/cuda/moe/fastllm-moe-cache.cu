//
// Graph-capturable GGUF, FP8 and compact NVFP4 expert cache.
//
// Host records retain quantized weights and their format-specific scales.
// A device LRU cache stores complete expert records in that same compact
// representation. Misses are pulled directly from mapped pinned host memory
// by a CUDA kernel. Eager hybrid decode can leave selected experts on NUMA
// when their estimated CPU completion is earlier than a cache refill.
//

#include "fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-moe-cache-stats.h"
#include "fastllm-cuda-expert-cache.cuh"
#include "fastllm-cuda-cache-layout.h"
#include "fastllm-cuda-record-copy.cuh"
#include "fastllm-cuda-shared-weight.cuh"
#include "fastllm.h"
#include "utils.h"
#include "gguf.h"
#include "fastllm-moe-gguf-restore.cuh"
#include "devices/moe_decode_scheduler.h"
#include "devices/moe_frequency_policy.h"
#include "devices/cuda/fastllm-cuda-moe-policy.h"
#ifdef USE_NUMAS
#include "devices/numas/numasdevice.h"
#endif

#include <algorithm>
#include <atomic>
#include <climits>
#include <chrono>
#include <condition_variable>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <unordered_map>
#include <vector>

#include <cuda_bf16.h>
#include "fastllm-moe-deepseekv41-cache.cuh"
#include "fastllm-moe-glm5-cache.cuh"

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
    bool deepSeekV41 = false;
    bool glm5 = false;
    float swigluLimit = 0.0f;
    fastllm::DataType weightType = fastllm::DataType::NVFP4_BLOCK_16_E4M3;
    int gateGgmlType = -1, downGgmlType = -1;
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
    if (type == fastllm::DataType::DATA_GGUF_FORMAT) return "GGUF";
    if (type == fastllm::DataType::NVFP4_BLOCK_32_E8M0) return "DeepSeek-V4.1 NVFP4 block32";
    if (type == fastllm::DataType::FP8_E4M3) return "FP8 E4M3";
    if (type == fastllm::DataType::FP8_E4M3_BLOCK_128) return "FP8 block128";
    return "NVFP4";
}

int Fp8PointerTableCount(fastllm::DataType type) {
    return type == fastllm::DataType::FP8_E4M3 ? 4 :
        type == fastllm::DataType::FP8_E4M3_BLOCK_128 ? 2 : 0;
}

struct OffloadGroup;

struct DecodeOverlapWorkspace {
    cudaStream_t copyStream = nullptr;
    cudaEvent_t copyStart = nullptr, copyDone = nullptr;
    cudaEvent_t metadataReady = nullptr;
    cudaEvent_t residentStart = nullptr, residentDone = nullptr;
    std::vector<cudaEvent_t> expertReady, stagedStart, stagedDone;
    uint8_t *records = nullptr, *upload = nullptr;
    int32_t *slots = nullptr, *hostSlots = nullptr;
    std::vector<fastllm::MoeDecodeOverlapScheduler> layers;
    int previousLayer = -1, previousHits = 0, previousMisses = 0, previousStagedRoutes = 0;

    bool Init(size_t stride, int layerCount, int capacity = kMaxTopK, bool restore = false) {
        expertReady.resize(capacity); stagedStart.resize(capacity); stagedDone.resize(capacity);
        const size_t recordBytes = (capacity + int(restore)) * stride;
        if (cudaStreamCreateWithFlags(&copyStream, cudaStreamNonBlocking) != cudaSuccess ||
            cudaMalloc(&records, recordBytes + capacity * sizeof(int32_t)) != cudaSuccess ||
            cudaMallocHost(&hostSlots, capacity * sizeof(int32_t)) != cudaSuccess) return false;
        slots = reinterpret_cast<int32_t *>(records + recordBytes);
        if (restore) upload = records + capacity * stride;
        if (cudaEventCreateWithFlags(&metadataReady, cudaEventDisableTiming) != cudaSuccess) return false;
        for (auto *event : {&copyStart, &copyDone, &residentStart, &residentDone})
            if (cudaEventCreate(event) != cudaSuccess) return false;
        for (int i = 0; i < capacity; ++i) {
            if (cudaEventCreateWithFlags(&expertReady[i], cudaEventDisableTiming) != cudaSuccess ||
                cudaEventCreate(&stagedStart[i]) != cudaSuccess ||
                cudaEventCreate(&stagedDone[i]) != cudaSuccess) return false;
        }
        layers.resize(layerCount);
        return true;
    }
    void PrepareCopy(int slotCount) {
        // Small routing uploads on the compute stream must land before bulk
        // DMA can monopolize the copy engine. Otherwise even resident experts
        // wait for every staged record, despite using independent streams.
        checkCudaErrors("MoE metadata ready", cudaEventRecord(metadataReady, cudaStreamPerThread));
        checkCudaErrors("MoE DMA metadata", cudaStreamWaitEvent(copyStream, metadataReady, 0));
        if (slotCount) checkCudaErrors("MoE staged slots", cudaMemcpyAsync(slots, hostSlots,
            slotCount * sizeof(int32_t), cudaMemcpyHostToDevice, copyStream));
    }
    // Publish each record separately. Callers may submit its computation
    // before queuing the next DMA, without waiting on the host.
    void CopyExpert(const OffloadGroup &group, int table, int expert, int i, int count);
    // Sources are immutable pinned weights; hostSlots stays alive until done.
    void CopyExperts(const OffloadGroup &group, int table, const int *experts,
                     int count, int slotCount) {
        PrepareCopy(slotCount);
        for (int i = 0; i < count; ++i) {
            CopyExpert(group, table, experts[i], i, count);
        }
    }
    // Called only after the compute completion event (which also joins DMA).
    void Observe() {
        if (previousLayer < 0) return;
        auto &timing = layers[previousLayer];
        float ms;
        if (previousHits > 0) {
            checkCudaErrors("Decode resident timing", cudaEventElapsedTime(&ms, residentStart, residentDone));
            timing.residentExpert.Observe(ms * 1000 / previousHits);
        }
        if (previousMisses > 0) {
            checkCudaErrors("Decode PCIe timing", cudaEventElapsedTime(&ms, copyStart, copyDone));
            timing.copiedExpert.Observe(ms * 1000 / previousMisses);
            // Exclude waits for the next expert's DMA from compute cost.
            double computeMs = 0;
            for (int i = 0; i < previousMisses; ++i) {
                checkCudaErrors("Decode staged timing", cudaEventElapsedTime(
                    &ms, stagedStart[i], stagedDone[i]));
                computeMs += ms;
            }
            timing.stagedExpert.Observe(computeMs * 1000 / previousStagedRoutes);
        }
        previousLayer = -1;
    }
    ~DecodeOverlapWorkspace() {
        if (copyStream) cudaStreamSynchronize(copyStream);
        cudaFree(records);
        cudaFreeHost(hostSlots);
        for (auto event : {copyStart, copyDone, residentStart, residentDone, metadataReady})
            if (event) cudaEventDestroy(event);
        for (size_t i = 0; i < expertReady.size(); ++i)
            for (auto event : {expertReady[i], stagedStart[i], stagedDone[i]})
                if (event) cudaEventDestroy(event);
        if (copyStream) cudaStreamDestroy(copyStream);
    }
};

struct HybridWorkspace {
    fastllm::Data gateOutput, quantizedInput;
    float *host = nullptr, *device = nullptr;
    cudaEvent_t start = nullptr, copied = nullptr, computed = nullptr, done = nullptr;
    cudaEvent_t prefetchStart = nullptr, prefetchDone = nullptr;
    fastllm::MoeDecodeScheduler scheduler;
    fastllm::MoeDecodeScheduler::Estimate prefetchCost;
    std::vector<unsigned> routeHeat;
    std::vector<uint64_t> routeLastSeen;
    int previousGpu = 0, previousMisses = 0;
    uint64_t cpuRoutes = 0, gpuRoutes = 0, residentRoutes = 0, allResidentRoutes = 0;
    uint64_t prefetchedExperts = 0;
    uint64_t pureCalls = 0, pureRoutes = 0, pureBaseHits = 0, pureBaseMisses = 0;
    bool pureActive = false;
    bool previousPrefetch = false;
    bool previousFrequency = false;
    bool pending = false;
    bool overlapAttempted = false;
    std::unique_ptr<DecodeOverlapWorkspace> overlap;
    ~HybridWorkspace() {
        if (pending) cudaEventSynchronize(done);
        overlap.reset();
        cudaFreeHost(host);
        cudaFree(device);
        if (start) cudaEventDestroy(start);
        if (copied) cudaEventDestroy(copied);
        if (computed) cudaEventDestroy(computed);
        if (done) cudaEventDestroy(done);
        if (prefetchStart) cudaEventDestroy(prefetchStart);
        if (prefetchDone) cudaEventDestroy(prefetchDone);
    }
};

struct DecodePolicyState {
    fastllm::MoeDecodePolicy policy;
    int layers, topk;
    double startUs = 0;
    DecodePolicyState(int records, int slots, int layers, int topk,
                      const std::vector<fastllm::cuda::CacheSlotSpan> &spans)
        : policy(records, slots), layers(layers), topk(topk) {
        if (spans.empty()) return;
        std::unordered_map<int, int> ids;
        std::vector<int> keys(records), capacities;
        const int experts = records / layers;
        for (int layer = 0; layer < layers; ++layer) {
            auto inserted = ids.emplace(spans[layer].begin, capacities.size());
            if (inserted.second) capacities.push_back(spans[layer].count);
            std::fill_n(keys.begin() + layer * experts, experts, inserted.first->second);
        }
        policy = fastllm::MoeDecodePolicy(keys, capacities);
    }
};

// Independent buffers/estimates: verify has more routes and must not teach
// the single-token scheduler its multi-row execution cost.
constexpr int kMaxVerifyRows = 8;
constexpr int kMaxVerifyRoutes = kMaxVerifyRows * kMaxTopK;
struct VerifyWorkspace {
    uint8_t *host = nullptr;
    float *device = nullptr;
    fastllm::Data activation, quantizedInput;
    cudaEvent_t start = nullptr, computed = nullptr, done = nullptr;
    cudaEvent_t prefetchStart = nullptr, prefetchDone = nullptr;
    bool pending = false, previousPrefetch = false;
    int previousGpuRoutes = 0;
    fastllm::MoeDecodeScheduler::Estimate cpuUnit, gpuRoute, refill;
    std::vector<unsigned> heat;
    std::vector<uint64_t> lastSeen;
    uint64_t calls = 0, routes = 0, hits = 0, gpuRoutes = 0, gpuExperts = 0, admissions = 0, fallbacks = 0;
    ~VerifyWorkspace() {
        if (pending)
            cudaEventSynchronize(done);
        cudaFreeHost(host);
        cudaFree(device);
        if (start)
            cudaEventDestroy(start);
        if (computed)
            cudaEventDestroy(computed);
        if (done)
            cudaEventDestroy(done);
        if (prefetchStart)
            cudaEventDestroy(prefetchStart);
        if (prefetchDone)
            cudaEventDestroy(prefetchDone);
    }
};

struct DeviceCache {
    bool attempted = false;
    bool ready = false;
    int device = -1;
    int slots = 0;
    fastllm::cuda::RecordCopyLaunch copyLaunch{0, 0};
    int ensureThreads = 0;
    uint8_t *records = nullptr;
    size_t recordBytes = 0;
    uint64_t *slotOffsets = nullptr;
    std::vector<fastllm::cuda::CacheSlotSpan> layerSlots;
    std::vector<uint64_t> hostSlotOffsets;
    std::unique_ptr<fastllm::MoeFrequencyPolicy> frequency;
    cudaStream_t admissionStream = nullptr;
    cudaEvent_t admissionDone = nullptr;
    bool frequencyActive = false;
    bool frequencyResidencyDirty = true;
    int32_t *keyToSlot = nullptr;
    int32_t *hostKeyToSlot = nullptr, *mappedKeyToSlot = nullptr;
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
    void *ggufWorkspace = nullptr;
    size_t ggufWorkspaceBytes = 0;
    std::unique_ptr<HybridWorkspace> hybrid;
    std::unique_ptr<VerifyWorkspace> verify;
    std::shared_ptr<FastllmCudaMoeExpertParallel> batchHybrid;
    std::unique_ptr<DecodePolicyState> decode;
};

struct OffloadGroup {
    OffloadLayout layout;
    // GGUF layers keep their actual record size in host storage and share
    // GPU slot partitions with other layers of the same size.
    std::vector<OffloadLayout> layerLayouts;
    std::vector<size_t> layerHostOffsets;
    size_t totalRecords = 0;
    uint8_t *hostRecords = nullptr;
    uint8_t *deviceHostRecords = nullptr;
    std::vector<const fastllm::Data *> tableKeys;
    // When NUMA storage is shared, hostRecords contains only original metadata.
    // The weight shards are borrowed from the model's prefill backend.
    std::vector<void *> numaPointers;
    bool cpuDecodeReady = false;
    fastllm::cuda::SharedExpertLayout sharedLayout;
    // Ordinary GGUF kernels receive canonical records restored on the GPU
    // from these borrowed CPU layouts, without a second full host snapshot.
    std::vector<fastllm_gguf_restore::Record> ggufSources;
    std::unordered_map<int, std::unique_ptr<DeviceCache> > deviceCaches;
    std::mutex mutex;

    const OffloadLayout &LayerLayout(int tableId) const {
        return layerLayouts.empty() ? layout : layerLayouts[tableId];
    }
};

void DecodeOverlapWorkspace::CopyExpert(const OffloadGroup &group, int table,
        int expert, int i, int count) {
    const auto &layout = group.LayerLayout(table);
    auto *destination = records + size_t(i) * layout.recordStride;
    if (i == 0) checkCudaErrors("MoE PCIe start", cudaEventRecord(copyStart, copyStream));
    if (group.ggufSources.empty()) {
        checkCudaErrors("MoE expert DMA", cudaMemcpyAsync(destination,
            group.hostRecords + group.layerHostOffsets[table] + size_t(expert) * layout.recordStride,
            layout.recordStride, cudaMemcpyHostToDevice, copyStream));
    } else {
        const auto &source = group.ggufSources[table];
        // Leave bulk PCIe traffic on the DMA engine so resident expert
        // kernels can occupy the SMs while this expert is in flight.
        const size_t base = (size_t(table) * layout.experts + expert) * 2 * source.shards;
        for (int part = 0; part < 2; ++part) {
            const auto &weight = source.weights[part];
            const size_t bytes = weight.rows * weight.rowBytes / source.shards;
            for (int node = 0; node < source.shards; ++node)
                checkCudaErrors("MoE NUMA expert DMA", cudaMemcpyAsync(
                    upload + (part ? source.downOffset : 0) + node * bytes,
                    group.numaPointers[base + part * source.shards + node],
                    bytes, cudaMemcpyHostToDevice, copyStream));
        }
        checkCudaErrors("MoE restore expert", fastllm_gguf_restore::CopyRecords(
            source, nullptr, destination, copyStream, nullptr, nullptr, nullptr, nullptr,
            0, nullptr, -1, upload));
    }
    if (i + 1 == count) checkCudaErrors("MoE PCIe done", cudaEventRecord(copyDone, copyStream));
    checkCudaErrors("MoE expert ready", cudaEventRecord(expertReady[i], copyStream));
}

struct CachedTable {
    OffloadGroup *group;
    int layer;
};

// A new weight encoding supplies a compute adapter and its shape capability;
// the policy, cache admission, CPU overlap and result merge stay unchanged.
// perExpert requests FP32 [topk, hidden] results. Generic adapters leave them
// unweighted; V4.1 includes scores before quantization and uses its own reducer.
// A negative route slot is inactive and must never dereference a weight pointer.
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
    } else if (w.dataType == fastllm::DataType::NVFP4_BLOCK_16_E4M3_PACKED) {
        view.tileStride = view.rowStride = fastllm::GetDataBytes(w.dataType, 1, columns);
        view.sourceOffset = sizeof(float);
        view.blockBytes = 8;
        view.blockStride = 9;
    } else if (w.dataType == fastllm::DataType::NVFP4_BLOCK_16) {
        view.tileStride = view.rowStride = ((columns + 15) / 16) * 12;
        view.blockBytes = 8; view.blockStride = 12;
    } else return false;
    return true;
}

bool PlanSharedV41(const OffloadLayout &l, fastllm::cuda::SharedExpertLayout &) {
    return l.deepSeekV41 && !(l.hidden % 32) && !(l.inter % 32);
}

bool PlanSharedGlm5(const OffloadLayout &l, fastllm::cuda::SharedExpertLayout &) {
    return l.glm5 && l.weightType == fastllm::DataType::NVFP4_BLOCK_16_E4M3_PACKED &&
        l.hidden % 128 == 0 && l.inter % 128 == 0;
}
bool BindSharedGlm5(const OffloadLayout &l, const fastllm::Data &w, int,
                   fastllm::cuda::SharedWeightView &view) {
    if (w.dataType != fastllm::DataType::NVFP4_BLOCK_16_E4M3_PACKED ||
        w.numasData.empty() || l.inter % (128 * w.numasData.size())) return false;
    view.rowBytes = fastllm::GetDataBytes(w.dataType, 1, w.dims[1]);
    view.rowStride = view.tileStride = view.blockBytes = view.blockStride = view.rowBytes;
    return true;
}
void SnapshotV41(const OffloadLayout &, const fastllm::Data &, const fastllm::Data &, uint8_t *) {}
bool BindSharedV41(const OffloadLayout &l, const fastllm::Data &w, int,
                   fastllm::cuda::SharedWeightView &view) {
    // CPU activation tasks must not split one block across NUMA shards.
    if (w.dataType != fastllm::DataType::NVFP4_BLOCK_32_E8M0 || w.numasData.empty() ||
        l.inter % (32 * w.numasData.size())) return false;
    view.rowBytes = fastllm::GetDataBytes(w.dataType, 1, w.dims[1]);
    view.tileStride = view.rowStride = view.blockBytes = view.blockStride = view.rowBytes;
    return true;
}

bool PlanSharedV41GGUF(const OffloadLayout &l, fastllm::cuda::SharedExpertLayout &) {
    return l.deepSeekV41 && l.gateGgmlType == GGML_TYPE_Q2_K_R4 &&
        l.downGgmlType == GGML_TYPE_Q4_K_R4 && !(l.hidden % 256) && !(l.inter % 256);
}
void SnapshotV41GGUF(const OffloadLayout &, const fastllm::Data &, const fastllm::Data &, uint8_t *) {}
bool BindSharedV41GGUF(const OffloadLayout &l, const fastllm::Data &w, int part,
                       fastllm::cuda::SharedWeightView &view) {
    if (w.dataType != fastllm::DATA_GGUF_FORMAT || w.numasData.empty() ||
        w.ggmlType != (part ? l.downGgmlType : l.gateGgmlType) ||
        w.dims[0] % (4 * w.numasData.size())) return false;
    // Copy R4 bytes as a contiguous blob. Row-group restoration would corrupt
    // its four-row packing; the V4.1 GPU kernel reads the cross-interleaved rows.
    view.rowGroups = 1;
    view.rowBytes = fastllm::GetDataBytes(static_cast<fastllm::DataType>(
        int(fastllm::DATA_GGUF_FORMAT) + w.ggmlType), 1, w.dims[1]);
    view.tileStride = view.rowStride = view.blockBytes = view.blockStride = view.rowBytes;
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

bool AppendMappedNumaPointers(const fastllm::Data &weight, std::vector<void *> &pointers) {
    for (uint8_t *source : weight.numasData) {
        void *mapped = nullptr;
        if (!source || (reinterpret_cast<uintptr_t>(source) & 15) ||
            cudaHostGetDevicePointer(&mapped, source, 0) != cudaSuccess || !mapped) {
            cudaGetLastError();
            return false;
        }
        pointers.push_back(source);
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
        if (!AppendMappedNumaPointers(weight, pointers)) return false;
    }
    group.numaPointers = std::move(pointers);
    return true;
}

bool CanShareGGUFWeights(const FastllmCudaMoeCacheLayer *layers, int count) {
#ifdef USE_NUMAS
    if (!fastllm::NumasMoeWeightsArePinned()) return false;
    for (int layer = 0; layer < count; ++layer) {
        for (int index = 2; index < layers[layer].weightsBatch; ++index) {
            const auto &weight = *layers[layer].weights[index];
            int type = weight.ggmlType;
            if (!weight.IsRepacked && !weight.disableGGUFRepack &&
                !fastllm::GetEnableAMX()) {
                if (auto *repack = get_repack_info(static_cast<ggml_type>(type))) type = repack->new_type;
            }
            // Unsupported CPU packing keeps the existing snapshot fallback.
            if (fastllm_gguf_restore::Ordinary(type) != weight.ggmlType) return false;
        }
    }
    return true;
#else
    return false;
#endif
}

bool BindGGUFWeights(OffloadGroup &group, const std::vector<fastllm::Data *> &weights) {
    const int nodes = weights.front()->numasData.size();
    if (nodes <= 0) return false;
    group.numaPointers.reserve(weights.size() * nodes);
    for (int layer = 0; layer < int(group.layerLayouts.size()); ++layer) {
        const auto &layout = group.LayerLayout(layer);
        fastllm_gguf_restore::Record source;
        source.shards = nodes;
        source.downOffset = layout.downOffset;
        source.stride = layout.recordStride;
        for (int expert = 0; expert < layout.experts; ++expert) {
            for (int part = 0; part < 2; ++part) {
                const auto &weight = *weights[(size_t(layer) * layout.experts + expert) * 2 + part];
                const int rows = part ? layout.hidden : 2 * layout.inter;
                const int columns = part ? layout.inter : layout.hidden;
                const size_t bytes = part ? layout.downBytes : layout.gateBytes;
                if (weight.numasData.size() != size_t(nodes) || rows % nodes ||
                    weight.cpuData || !weight.isPinned || weight.GetBytes() != bytes ||
                    weight.dims.size() != 2 || weight.dims[0] != rows || weight.dims[1] != columns ||
                    fastllm_gguf_restore::Ordinary(weight.ggmlType) !=
                        (part ? layout.downGgmlType : layout.gateGgmlType)) return false;
                if (fastllm_gguf_restore::Ordinary(weight.ggmlType) != weight.ggmlType &&
                    (rows % (4 * nodes) || columns % 256)) return false;
                if (!expert) source.weights[part] = {weight.ggmlType, rows, columns, bytes / rows};
                else if (source.weights[part].type != weight.ggmlType) return false;
                if (!AppendMappedNumaPointers(weight, group.numaPointers)) return false;
            }
        }
        group.ggufSources.push_back(source);
    }
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

int GGUFCacheType(const fastllm::Data *weight) {
    if (!weight) return -1;
    if (weight->dataType == fastllm::DataType::DATA_GGUF_FORMAT) return weight->ggmlType;
    // The GGUF loader exposes F32/F16 payloads as native floating Data. Preserve
    // their source-format identity so mixed floating/quantized pairs still work.
    if (!weight->isGGUFData) return -1;
    switch (weight->dataType) {
        case fastllm::DataType::FLOAT32: return GGML_TYPE_F32;
        case fastllm::DataType::FLOAT16: return GGML_TYPE_F16;
        case fastllm::DataType::BFLOAT16: return GGML_TYPE_BF16;
        default: return -1;
    }
}

bool ValidateV41GGUFPair(fastllm::Data *gate, fastllm::Data *down,
                         const OffloadLayout *expected, OffloadLayout &observed) {
    if (!gate || !down || gate->dataType != fastllm::DATA_GGUF_FORMAT ||
        down->dataType != fastllm::DATA_GGUF_FORMAT ||
        gate->ggmlType != GGML_TYPE_Q2_K_R4 || down->ggmlType != GGML_TYPE_Q4_K_R4 ||
        !gate->IsRepacked || !down->IsRepacked || gate->numasData.empty() || down->numasData.empty() ||
        gate->dataDevice != fastllm::DataDevice::CPU || down->dataDevice != fastllm::DataDevice::CPU ||
        gate->dims.size() != 2 || down->dims.size() != 2 ||
        down->dims[0] <= 0 || down->dims[1] <= 0 || down->dims[1] > INT_MAX / 2 ||
        gate->dims[0] != 2 * down->dims[1] || gate->dims[1] != down->dims[0] ||
        gate->dims[1] % 256 || down->dims[1] % 256) return false;
    observed.weightType = fastllm::DATA_GGUF_FORMAT;
    observed.deepSeekV41 = true;
    observed.gateGgmlType = gate->ggmlType;
    observed.downGgmlType = down->ggmlType;
    observed.hidden = gate->dims[1]; observed.inter = down->dims[1];
    observed.gateBytes = gate->GetBytes(); observed.downBytes = down->GetBytes();
    observed.downOffset = AlignUp(observed.gateBytes, 16);
    observed.scalesOffset = AlignUp(observed.downOffset + observed.downBytes, 16);
    observed.recordStride = AlignUp(observed.scalesOffset, 128);
    return !expected || (expected->weightType == observed.weightType && expected->deepSeekV41 &&
        expected->gateGgmlType == observed.gateGgmlType && expected->downGgmlType == observed.downGgmlType &&
        expected->hidden == observed.hidden && expected->inter == observed.inter &&
        expected->gateBytes == observed.gateBytes && expected->downBytes == observed.downBytes);
}

bool ValidateWeightPair(fastllm::Data *gate, fastllm::Data *down,
                        const OffloadLayout *expected,
                        OffloadLayout &observed) {
    const int gateType = GGUFCacheType(gate), downType = GGUFCacheType(down);
    if (gateType >= 0 && downType >= 0) {
        if (gate->dataDevice != fastllm::DataDevice::CPU ||
            down->dataDevice != fastllm::DataDevice::CPU ||
            !gate->cpuData || !down->cpuData ||
            (gate->dataType == fastllm::DataType::DATA_GGUF_FORMAT && !gate->ggmlTensor) ||
            (down->dataType == fastllm::DataType::DATA_GGUF_FORMAT && !down->ggmlTensor) ||
            gate->IsRepacked || down->IsRepacked ||
            gate->dims.size() != 2 || down->dims.size() != 2 ||
            down->dims[0] <= 0 || down->dims[1] <= 0 || down->dims[1] > INT_MAX / 2 ||
            gate->dims[0] != 2 * down->dims[1] || gate->dims[1] != down->dims[0] ||
            !FastllmCudaMoeGGUFCacheSupported(gateType, gate->dims[1]) ||
            !FastllmCudaMoeGGUFCacheSupported(downType, down->dims[1])) return false;
        observed.weightType = fastllm::DataType::DATA_GGUF_FORMAT;
        observed.gateGgmlType = gateType;
        observed.downGgmlType = downType;
        observed.hidden = gate->dims[1];
        observed.inter = down->dims[1];
        observed.gateBytes = gate->GetBytes();
        observed.downBytes = down->GetBytes();
        observed.downOffset = AlignUp(observed.gateBytes, 16);
        observed.scalesOffset = AlignUp(observed.downOffset + observed.downBytes, 16);
        observed.recordStride = AlignUp(observed.scalesOffset, 128);
        return observed.gateBytes == size_t(gate->dims[0]) * ggml_row_size(static_cast<ggml_type>(gateType), gate->dims[1]) &&
            observed.downBytes == size_t(down->dims[0]) * ggml_row_size(static_cast<ggml_type>(downType), down->dims[1]) &&
            (!expected || (expected->weightType == observed.weightType &&
             expected->gateGgmlType == observed.gateGgmlType &&
             expected->downGgmlType == observed.downGgmlType &&
             expected->hidden == observed.hidden && expected->inter == observed.inter &&
             expected->gateBytes == observed.gateBytes && expected->downBytes == observed.downBytes));
    }
    if (gate == nullptr || down == nullptr ||
        (gate->dataType != fastllm::DataType::NVFP4_BLOCK_16_E4M3 &&
         gate->dataType != fastllm::DataType::NVFP4_BLOCK_16_E4M3_PACKED &&
         gate->dataType != fastllm::DataType::FP8_E4M3 &&
         gate->dataType != fastllm::DataType::FP8_E4M3_BLOCK_128 &&
         gate->dataType != fastllm::DataType::NVFP4_BLOCK_32_E8M0) ||
        down->dataType != gate->dataType ||
        gate->dataDevice != fastllm::DataDevice::CPU ||
        down->dataDevice != fastllm::DataDevice::CPU ||
        ((gate->cpuData == nullptr || down->cpuData == nullptr) &&
         ((gate->dataType != fastllm::DataType::NVFP4_BLOCK_32_E8M0 &&
           gate->dataType != fastllm::DataType::NVFP4_BLOCK_16_E4M3_PACKED) ||
          gate->numasData.empty() || down->numasData.empty())) ||
        gate->dims.size() != 2 || down->dims.size() != 2 ||
        gate->dims[0] <= 0 || gate->dims[1] <= 0 ||
        down->dims[0] <= 0 || down->dims[1] <= 0 ||
        down->dims[1] > INT_MAX / 2 ||
        gate->dims[0] != down->dims[1] * 2 ||
        gate->dims[1] != down->dims[0]) {
        return false;
    }

    observed.weightType = gate->dataType;
    if (gate->dataType == fastllm::DataType::NVFP4_BLOCK_32_E8M0 &&
        ((gate->dims[1] % 32) || (down->dims[1] % 32) ||
         gate->blockM != 32 || down->blockM != 32)) return false;
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

uint64_t DeviceCacheBudgetBytes(int device) {
    const std::string name = "FASTLLM_MOE_CUDA_CACHE_BYTES_" + std::to_string(device);
    const char *value = std::getenv(name.c_str());
    if (!value || !*value) return fastllm::GetMoeCudaCacheBytes();
    uint64_t bytes = 0;
    for (const char *p = value; *p; ++p) {
        if (*p < '0' || *p > '9' || bytes > (UINT64_MAX - uint64_t(*p - '0')) / 10) {
            std::fprintf(stderr, "[Fastllm] Invalid %s: expected unsigned decimal bytes; "
                "disabling expert cache on CUDA device %d.\n", name.c_str(), device);
            return 0;
        }
        bytes = bytes * 10 + uint64_t(*p - '0');
    }
    return bytes;
}

size_t RequestedSlots(size_t recordStride, size_t totalRecords,
                      uint64_t bytes = fastllm::GetMoeCudaCacheBytes()) {
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
    if (cache.admissionStream) cudaStreamSynchronize(cache.admissionStream);
    cache.hybrid.reset();
    cache.verify.reset();
    cache.batchHybrid.reset();
    cache.decode.reset();
    cudaFree(cache.records);
    cudaFree(cache.slotOffsets);
    cudaFree(cache.keyToSlot);
    cudaFreeHost(cache.hostKeyToSlot);
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
    cudaFree(cache.ggufWorkspace);
    if (cache.admissionDone) cudaEventDestroy(cache.admissionDone);
    if (cache.admissionStream) cudaStreamDestroy(cache.admissionStream);
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
        // Cache payload and metadata become visible together. An external
        // event dependency also makes later captured cache users wait safely.
        if (cache.admissionDone) {
            cudaStreamCaptureStatus capture;
            checkCudaErrors("MoE cache update capture", cudaStreamIsCapturing(cudaStreamPerThread, &capture));
            checkCudaErrors("MoE cache update dependency", cudaStreamWaitEvent(
                cudaStreamPerThread, cache.admissionDone,
                capture == cudaStreamCaptureStatusActive ? cudaEventWaitExternal : cudaEventWaitDefault));
        }
        return cache.ready ? &cache : nullptr;
    }
    cache.attempted = true;

    const uint64_t budgetBytes = DeviceCacheBudgetBytes(device);
    // A zero per-device budget can still execute streamed decode misses. No
    // resident payload or slots are allocated; all route lookups miss.
    const bool streamOnly = budgetBytes == 0 && group.cpuDecodeReady &&
        group.layout.weightType == fastllm::DataType::DATA_GGUF_FORMAT &&
        group.sharedLayout.shards == 0;
    if (budgetBytes == 0 && !streamOnly) return nullptr;
    size_t slots = RequestedSlots(
        group.layout.recordStride, group.totalRecords, budgetBytes);
    if (!streamOnly && slots < kMaxTopK) {
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
    if (group.layout.weightType == fastllm::DataType::DATA_GGUF_FORMAT) {
        // Allocate before capture; heterogeneous layers share the largest workspace.
        for (const auto &layout : group.layerLayouts) {
            cache.ggufWorkspaceBytes = std::max(cache.ggufWorkspaceBytes,
                FastllmCudaMoeGGUFCacheWorkspaceBytes(layout.hidden, layout.inter));
        }
    }
    const size_t metadataBytes =
        group.totalRecords * sizeof(int32_t) +
        group.numaPointers.size() * sizeof(void *) +
        group.totalRecords * (sizeof(int32_t) + sizeof(unsigned long long) + sizeof(uint64_t) +
            Fp8PointerTableCount(group.layout.weightType) * sizeof(void *)) + 4096 +
        cache.ggufWorkspaceBytes;
    size_t usableBytes = freeBytes > reserveBytes
        ? freeBytes - reserveBytes : 0;
    usableBytes = usableBytes > metadataBytes
        ? usableBytes - metadataBytes : 0;
    fastllm::cuda::CacheSlotPlan slotPlan;
    if (streamOnly) {
        cache.layerSlots.resize(group.layerLayouts.size(), {0, 0});
    } else if (!group.layerLayouts.empty()) {
        std::vector<size_t> strides;
        for (const auto &layout : group.layerLayouts) strides.push_back(layout.recordStride);
        slotPlan = fastllm::cuda::PlanCacheSlots(strides, group.layout.experts,
            std::min(uint64_t(usableBytes), budgetBytes), kMaxTopK);
        slots = slotPlan.offsets.size();
        cache.recordBytes = slotPlan.bytes;
        cache.layerSlots = slotPlan.layers;
        cache.hostSlotOffsets = slotPlan.offsets;
    } else {
        slots = std::min(slots, usableBytes / group.layout.recordStride);
        cache.recordBytes = slots * group.layout.recordStride;
    }
    if (!streamOnly && slots < kMaxTopK) {
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
        (cache.ggufWorkspaceBytes == 0 ||
            AllocateOne(&cache.ggufWorkspace, cache.ggufWorkspaceBytes)) &&
        (streamOnly || AllocateOne(reinterpret_cast<void **>(&cache.records),
                    cache.recordBytes)) &&
        AllocateOne(reinterpret_cast<void **>(&cache.keyToSlot),
                    group.totalRecords * sizeof(int32_t)) &&
        (streamOnly || AllocateOne(reinterpret_cast<void **>(&cache.slotKeys),
                    slots * sizeof(int32_t))) &&
        (streamOnly || AllocateOne(reinterpret_cast<void **>(&cache.lastUsed),
                    slots * sizeof(unsigned long long))) &&
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
        (streamOnly || cudaMemsetAsync(cache.slotKeys, 0xff,
            slots * sizeof(int32_t), cudaStreamPerThread) == cudaSuccess) &&
        (streamOnly || cudaMemsetAsync(cache.lastUsed, 0,
            slots * sizeof(unsigned long long), cudaStreamPerThread) == cudaSuccess) &&
        cudaMemsetAsync(cache.step, 0,
            sizeof(unsigned long long), cudaStreamPerThread) == cudaSuccess &&
        cudaMemsetAsync(cache.hitCount, 0,
            sizeof(unsigned long long), cudaStreamPerThread) == cudaSuccess &&
        cudaMemsetAsync(cache.totalMissCount, 0,
            sizeof(unsigned long long), cudaStreamPerThread) == cudaSuccess &&
        PrepareFp8SlotPointers(cache, group.layout);
    if (initialized && !slotPlan.offsets.empty()) {
        const size_t bytes = slots * sizeof(uint64_t);
        initialized = AllocateOne(reinterpret_cast<void **>(&cache.slotOffsets), bytes) &&
            cudaMemcpy(cache.slotOffsets, slotPlan.offsets.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    }
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
    if (initialized && group.cpuDecodeReady) {
        // Create the fence before any graph can capture this cache. A graph
        // captured before the first asynchronous admission must wait on later
        // admissions too, rather than retaining a fence-free cache path.
        initialized = cudaEventCreateWithFlags(&cache.admissionDone, cudaEventDisableTiming) == cudaSuccess &&
            cudaEventRecord(cache.admissionDone, cudaStreamPerThread) == cudaSuccess;
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

    // Keep residency coherent for eager CPU planning, including mutations by
    // ordinary cache calls and captured graphs. Failure only disables this
    // optimization; the device lookup path remains available.
    if (properties.canMapHostMemory &&
        cudaHostAlloc(&cache.hostKeyToSlot, group.totalRecords * sizeof(int32_t),
                      cudaHostAllocMapped) == cudaSuccess &&
        cudaHostGetDevicePointer(&cache.mappedKeyToSlot, cache.hostKeyToSlot, 0) == cudaSuccess) {
        std::fill_n(cache.hostKeyToSlot, group.totalRecords, -1);
    } else {
        cudaGetLastError();
        cudaFreeHost(cache.hostKeyToSlot);
        cache.hostKeyToSlot = cache.mappedKeyToSlot = nullptr;
    }
    cache.ready = true;

    std::fprintf(stderr,
        "[Fastllm] %s GPU expert cache: %d slots, %.3f GiB, "
        "%zu records in mapped host storage; refill grid %d x %d.\n",
        FormatName(group.layout.weightType), cache.slots,
        static_cast<double>(cache.recordBytes) /
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

OffloadGroup *FindHybridGroup(fastllm::Data **weights, int weightsBatch,
                             int *tableId = nullptr) {
#ifdef USE_NUMAS
    if (!FastllmCudaMoeCacheRequested()) return nullptr;
    auto *group = FindGroup(weights, weightsBatch, tableId);
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

    // Shared compact layouts come from NUMA registration. Ordinary GLM GGUF
    // must instead be snapshotted before NUMA repacks the original blocks.
    const bool canonicalGlmGGUF = layers[0].glm5 && layers[0].weights[2] &&
        layers[0].weights[2]->dataType == fastllm::DataType::DATA_GGUF_FORMAT;
    if ((layers[0].deepSeekV41 || layers[0].glm5) && !registerNumaWeights) return false;
    const bool registerBeforeSnapshot = layers[0].deepSeekV41 ||
        (layers[0].glm5 && !canonicalGlmGGUF);
    if (registerBeforeSnapshot) {
        registerNumaWeights();
    }
    OffloadLayout layout;
    layout.experts = experts;
    std::vector<OffloadLayout> layerLayouts;
    layerLayouts.reserve(layerCount);
    bool first = true;
    for (int layer = 0; layer < layerCount; ++layer) {
        if (layers[layer].weights == nullptr ||
            layers[layer].weightsBatch != (experts + 1) * 2 ||
            layers[layer].deepSeekV41 != layers[0].deepSeekV41 ||
            layers[layer].glm5 != layers[0].glm5 ||
            layers[layer].swigluLimit != layers[0].swigluLimit) {
            return false;
        }
        OffloadLayout layerLayout;
        for (int expert = 0; expert < experts; ++expert) {
            const int position = (expert + 1) * 2;
            OffloadLayout observed;
            const bool v41GGUF = layers[0].deepSeekV41 && layers[layer].weights[position] &&
                layers[layer].weights[position]->dataType == fastllm::DATA_GGUF_FORMAT;
            if (!(v41GGUF ? ValidateV41GGUFPair : ValidateWeightPair)(
                    layers[layer].weights[position],
                    layers[layer].weights[position + 1],
                    expert == 0 ? nullptr : &layerLayout, observed)) {
                const fastllm::Data *gate =
                    layers[layer].weights[position];
                const fastllm::Data *down =
                    layers[layer].weights[position + 1];
                std::fprintf(stderr,
                    "[Fastllm] CUDA expert cache rejected layer %d expert %d: "
                    "requires supported host-resident GGUF, NVFP4 or FP8 weights "
                    "with matching layer layouts (gate dtype=%d, down dtype=%d).\n",
                    layer, expert,
                    gate == nullptr ? -1 : static_cast<int>(gate->dataType),
                    down == nullptr ? -1 : static_cast<int>(down->dataType));
                return false;
            }
            if (layers[0].deepSeekV41 !=
                (observed.weightType == fastllm::DataType::NVFP4_BLOCK_32_E8M0 || v41GGUF)) return false;
            const bool glmGGUF = layers[0].glm5 &&
                observed.weightType == fastllm::DataType::DATA_GGUF_FORMAT;
            if (layers[0].glm5 !=
                (observed.weightType == fastllm::DataType::NVFP4_BLOCK_16_E4M3_PACKED || glmGGUF)) return false;
            if (glmGGUF && !FastllmCudaMoeGlm5GGUFCacheSupported(
                    observed.gateGgmlType, observed.downGgmlType, observed.hidden, observed.inter)) return false;
            if (expert == 0) {
                layerLayout = observed;
                layerLayout.experts = experts;
                layerLayout.deepSeekV41 = layers[0].deepSeekV41;
                layerLayout.glm5 = layers[0].glm5;
                layerLayout.swigluLimit = layers[0].swigluLimit;
            }
            if (first) {
                observed.experts = experts;
                observed.deepSeekV41 = layers[0].deepSeekV41;
                observed.glm5 = layers[0].glm5;
                observed.swigluLimit = layers[0].swigluLimit;
                layout = observed;
                first = false;
            } else if (observed.weightType == fastllm::DataType::DATA_GGUF_FORMAT && !layers[0].deepSeekV41) {
                if (layout.weightType != observed.weightType) return false;
                layout.recordStride = std::max(layout.recordStride, observed.recordStride);
            } else {
                OffloadLayout checked;
                if (!(v41GGUF ? ValidateV41GGUFPair : ValidateWeightPair)(layers[layer].weights[position],
                                        layers[layer].weights[position + 1], &layout, checked)) return false;
            }
        }
        layerLayouts.push_back(layerLayout);
    }

    std::unique_ptr<OffloadGroup> group(new OffloadGroup());
    group->layout = layout;
    if (layout.weightType == fastllm::DataType::DATA_GGUF_FORMAT) {
        group->layerLayouts = std::move(layerLayouts);
    }
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
    // Borrow supported NUMA layouts; snapshot only formats whose CPU packing
    // cannot yet be restored. GPU records always retain canonical GGUF bytes.
    const bool shareNuma = bool(registerNumaWeights) &&
                          (layout.weightType != fastllm::DataType::DATA_GGUF_FORMAT || layout.deepSeekV41);
    const bool shareGGUF = bool(registerNumaWeights) &&
        layout.weightType == fastllm::DATA_GGUF_FORMAT && !layout.deepSeekV41 && !layout.glm5 &&
        CanShareGGUFWeights(layers, layerCount);
    if (shareNuma && (!storage || layout.recordStride > UINT32_MAX / kMaxTopK ||
                      !storage->plan(layout, group->sharedLayout))) return false;
    const size_t hostStride = shareNuma ? group->sharedLayout.auxiliaryBytes : layout.recordStride;
    size_t hostBytes = 0;
    for (int layer = 0; layer < layerCount; ++layer) {
        group->layerHostOffsets.push_back(hostBytes);
        hostBytes += size_t(experts) * (shareGGUF ? 0 :
            (shareNuma ? hostStride : group->LayerLayout(layer).recordStride));
    }
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
    if (shareNuma || shareGGUF) expertWeights.reserve(group->totalRecords * 2);
    for (int layer = 0; layer < layerCount; ++layer) {
        group->tableKeys.push_back(layers[layer].weights[2]);
        const auto &recordLayout = group->LayerLayout(layer);
        for (int expert = 0; expert < experts; ++expert) {
            const int position = (expert + 1) * 2;
            fastllm::Data *gate = layers[layer].weights[position];
            fastllm::Data *down = layers[layer].weights[position + 1];
            uint8_t *record = hostBytes ? group->hostRecords +
                group->layerHostOffsets[layer] + size_t(expert) *
                    (shareNuma ? hostStride : recordLayout.recordStride) : nullptr;
            if (shareNuma || shareGGUF) {
                expertWeights.push_back(gate);
                expertWeights.push_back(down);
                if (shareNuma) storage->snapshot(layout, *gate, *down, record);
            } else {
                if (!group->layerLayouts.empty()) memset(record, 0, recordLayout.recordStride);
                memcpy(record, gate->cpuData, recordLayout.gateBytes);
                memcpy(record + recordLayout.downOffset, down->cpuData, recordLayout.downBytes);
                if (layout.gateScaleBytes) {
                    memcpy(record + layout.scalesOffset, gate->scales.data(), layout.gateScaleBytes);
                    memcpy(record + layout.scalesOffset + layout.gateScaleBytes,
                           down->scales.data(), layout.downScaleBytes);
                }
            }
        }
    }
    if (shareGGUF) {
        registerNumaWeights();
        if (!BindGGUFWeights(*group, expertWeights)) return false;
        size_t sharedBytes = 0;
        for (const auto &l : group->layerLayouts) sharedBytes += size_t(experts) * l.recordStride;
        std::fprintf(stderr, "[Fastllm] GGUF expert cache shares %d NUMA shards per weight; "
            "avoiding a %.3f GiB host weight snapshot.\n",
            group->ggufSources.front().shards, double(sharedBytes) / (1ULL << 30));
    }
    if (shareNuma) {
        if (!registerBeforeSnapshot) registerNumaWeights();
        if (!BindNumaWeights(*group, expertWeights, *storage)) {
            std::fprintf(stderr, "[Fastllm] CUDA expert cache requires matching pinned NUMA "
                         "shards after registration; using the configured MoE backend.\n");
            return false;
        }
        std::fprintf(stderr, "[Fastllm] %s expert cache shares %d NUMA shards per weight; "
                     "metadata uses %.3f GiB, avoiding a %.3f GiB host weight snapshot.\n",
                     FormatName(layout.weightType), group->sharedLayout.shards, double(hostBytes) / (1ULL << 30),
                     double(group->totalRecords * (layout.recordStride - hostStride)) / (1ULL << 30));
    }
#ifdef USE_NUMAS
    // Both shared shards and snapshot fallbacks must be ready before hybrid lookup.
    if (registerNumaWeights) {
        if (!shareNuma && !shareGGUF && layout.weightType == fastllm::DataType::DATA_GGUF_FORMAT)
            registerNumaWeights();
        group->cpuDecodeReady = true;
        for (int layer = 0; layer < layerCount; ++layer)
            group->cpuDecodeReady &= fastllm::CanRunNumasMoeDecodeExperts(
                layers[layer].weights, layers[layer].weightsBatch);
    }
#endif
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
    if (!group || group->layout.deepSeekV41 || group->layout.glm5) return false;
    auto *cache = GetDeviceCache(*group);
    return cache && cache->slots > 0;
}

bool FastllmCudaCanRunMoeHybrid(fastllm::Data **weights, int weightsBatch) {
    return FindHybridGroup(weights, weightsBatch) != nullptr;
}

bool FastllmCudaGetMoeGGUFResidents(fastllm::Data **weights, int experts,
                                  FastllmCudaMoeGGUFResidents &view) {
    view = {};
    if (!weights || experts <= 0 || !weights[2]) return false;
    OffloadGroup *group = nullptr;
    int table = -1;
    {
        std::lock_guard<std::mutex> lock(RegistryMutex());
        auto it = TableRegistry().find(weights[2]);
        if (it == TableRegistry().end()) return false;
        group = it->second.group;
        table = it->second.layer;
    }
    const auto &layout = group->LayerLayout(table);
    if (layout.weightType != fastllm::DATA_GGUF_FORMAT || layout.deepSeekV41 ||
        layout.glm5 || experts > layout.experts) return false;
    const int device = FastllmCudaGetDevice();
    std::lock_guard<std::mutex> lock(group->mutex);
    const auto it = group->deviceCaches.find(device);
    if (it == group->deviceCaches.end() || !it->second) return false;
    const auto &cache = *it->second;
    if (!cache.ready || !cache.slots || !cache.hostKeyToSlot) return false;
    // Both the mapped residency mirror and its payload must be visible to
    // the host planner. The owning model has drained its preceding kernels.
    if (cache.admissionDone) checkCudaErrors("MoE prefill cache residency",
        cudaEventSynchronize(cache.admissionDone));
    view.weights.assign(2*experts, nullptr);
    view.gateType = layout.gateGgmlType; view.downType = layout.downGgmlType;
    view.hidden = layout.hidden; view.inter = layout.inter;
    const auto span = cache.layerSlots[table];
    for (int e = 0; e < experts; ++e) {
        const int slot = cache.hostKeyToSlot[table*layout.experts+e];
        if (slot < span.begin || slot >= span.begin+span.count) continue;
        const auto *record = cache.records + cache.hostSlotOffsets[slot];
        view.weights[2*e] = record;
        view.weights[2*e+1] = record+layout.downOffset;
    }
    return true;
}

bool FastllmCudaUseMoeHybrid(fastllm::Data **weights, int weightsBatch) {
    auto *group = FindHybridGroup(weights, weightsBatch);
    if (!group) return false;
    auto *cache = GetDeviceCache(*group);
    return cache && (cache->frequency || !cache->decode || !cache->decode->policy.UseGpu());
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
    int tableId = -1;
    OffloadGroup *group = FindGroup(weights, weightsBatch, &tableId);
    if (!group) return false;
    const auto &layout = group->LayerLayout(tableId);
    auto *cache = GetDeviceCache(*group);
    return cache && (cache->slots > 0 ||
           (cache->frequencyActive && group->cpuDecodeReady && input.dims[0] == 1)) &&
           (!(layout.deepSeekV41 || layout.glm5) ||
           (input.dims[0] == 1 && input.dataType == fastllm::DataType::BFLOAT16)) &&
           input.dims[1] == layout.hidden;
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
bool Glm5HybridShape(const OffloadLayout &l) {
    return l.glm5 && l.hidden % 128 == 0 && l.inter % 128 == 0;
}
bool ComputeGlm5Cache(const fastllm::Data &input, fastllm::Data &gateOutput,
        fastllm::Data &, const OffloadLayout &layout, const DeviceCache &cache,
        const float *scores, int topk, float *perExpert) {
    if (!Glm5HybridShape(layout) || input.dataType != fastllm::DataType::BFLOAT16 || !perExpert)
        return false;
    fastllm::cuda::glm5_cache::Gate<<<dim3(layout.inter / 128, topk), 1024, 0, cudaStreamPerThread>>>(
        (const __nv_bfloat16 *)input.cudaData, cache.routeSlots, cache.records, scores,
        (__nv_bfloat16 *)gateOutput.cudaData, layout.hidden, layout.inter, layout.recordStride, layout.swigluLimit);
    fastllm::cuda::glm5_cache::Down<<<dim3(layout.hidden / 16, topk), 128, 0, cudaStreamPerThread>>>(
        (const __nv_bfloat16 *)gateOutput.cudaData, cache.routeSlots, cache.records, perExpert,
        layout.hidden, layout.inter, layout.recordStride, layout.downOffset);
    return cudaGetLastError() == cudaSuccess;
}
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

bool ComputeV41Cache(const fastllm::Data &input, fastllm::Data &activation,
        fastllm::Data &, const OffloadLayout &layout, const DeviceCache &cache,
        const float *scores, int topk, float *perExpert) {
    if (!perExpert || input.dataType != fastllm::DataType::BFLOAT16) return false;
    using namespace fastllm::cuda::dsv41_cache;
    Gate<><<<dim3(layout.inter / 32, topk), 512, 0, cudaStreamPerThread>>>(
        (const __nv_bfloat16*)input.cudaData, cache.routeSlots, cache.records, scores,
        (__nv_bfloat16*)activation.cudaData, layout.hidden, layout.inter, layout.recordStride, layout.swigluLimit);
    Down<<<dim3(layout.hidden / 8, topk), 128, 0, cudaStreamPerThread>>>(
        (const __nv_bfloat16*)activation.cudaData, cache.routeSlots, cache.records, perExpert,
        layout.hidden, layout.inter, layout.recordStride, layout.downOffset);
    return cudaGetLastError() == cudaSuccess;
}

bool FP8HybridShape(const OffloadLayout &) { return true; }
bool GGUFHybridShape(const OffloadLayout &layout) {
    if (layout.glm5)
        return FastllmCudaMoeGlm5GGUFCacheSupported(
            layout.gateGgmlType, layout.downGgmlType, layout.hidden, layout.inter);
    if (layout.deepSeekV41) {
        fastllm::cuda::SharedExpertLayout plan;
        return PlanSharedV41GGUF(layout, plan);
    }
    return FastllmCudaMoeGGUFCacheSupported(layout.gateGgmlType, layout.hidden) &&
           FastllmCudaMoeGGUFCacheSupported(layout.downGgmlType, layout.inter);
}

bool ComputeGGUFExperts(const fastllm::Data &input, fastllm::Data &gateOutput,
        fastllm::Data &output, const OffloadLayout &layout, const FastllmCudaMoeGGUFCacheView &view,
        const float *scores, int topk, float *perExpert) {
    if (layout.deepSeekV41)
        return FastllmCudaMoeV41GGUFCacheCompute(input, gateOutput, view,
            scores, topk, layout.swigluLimit, perExpert);
    if (layout.glm5)
        return FastllmCudaMoeGlm5GGUFCacheCompute(input, gateOutput, view,
            scores, topk, layout.swigluLimit, perExpert);
    return FastllmCudaMoeGGUFCacheCompute(input, gateOutput, output, view, scores, topk, perExpert);
}

bool ComputeGGUFCache(const fastllm::Data &input, fastllm::Data &gateOutput,
        fastllm::Data &output, const OffloadLayout &layout, const DeviceCache &cache,
        const float *scores, int topk, float *perExpert) {
    const FastllmCudaMoeGGUFCacheView view{
        cache.records, cache.routeSlots, layout.recordStride, layout.downOffset,
        layout.gateGgmlType, layout.downGgmlType, layout.hidden, layout.inter,
        cache.ggufWorkspace, cache.ggufWorkspaceBytes, cache.slotOffsets};
    return ComputeGGUFExperts(input, gateOutput, output, layout, view, scores, topk, perExpert);
}

const ExpertCacheBackend *FindExpertCacheBackend(fastllm::DataType type) {
    static const SharedCacheStorage nvfp4{PlanSharedNVFP4, SnapshotNVFP4Metadata, BindSharedNVFP4};
    static const SharedCacheStorage v41{PlanSharedV41, SnapshotV41, BindSharedV41};
    static const SharedCacheStorage glm5{PlanSharedGlm5, SnapshotV41, BindSharedGlm5};
    static const SharedCacheStorage fp8{PlanSharedFP8, SnapshotFP8Metadata, BindSharedFP8};
    static const SharedCacheStorage v41GGUF{PlanSharedV41GGUF, SnapshotV41GGUF, BindSharedV41GGUF};
    static const ExpertCacheBackend backends[] = {
        {fastllm::DataType::DATA_GGUF_FORMAT, GGUFHybridShape, &v41GGUF, ComputeGGUFCache},
        {fastllm::DataType::NVFP4_BLOCK_32_E8M0, FP8HybridShape, &v41, ComputeV41Cache},
        {fastllm::DataType::NVFP4_BLOCK_16_E4M3, SupportsNVFP4WideDecode, &nvfp4, ComputeNVFP4Cache},
        {fastllm::DataType::NVFP4_BLOCK_16_E4M3_PACKED, Glm5HybridShape, &glm5, ComputeGlm5Cache},
        {fastllm::DataType::FP8_E4M3, FP8HybridShape, &fp8, ComputeFP8Cache},
        {fastllm::DataType::FP8_E4M3_BLOCK_128, FP8HybridShape, &fp8, ComputeFP8Cache},
    };
    for (const auto &backend : backends) if (backend.type == type) return &backend;
    return nullptr;
}

bool EnsureCachedExperts(OffloadGroup *group, DeviceCache *cache, int tableId,
                         const int32_t *indices, int topk, cudaStream_t stream = cudaStreamPerThread,
                         const uint8_t *stagedRecord = nullptr, int stagedExpert = -1) {
    if (cache->slots == 0) return false;
    cache->frequencyResidencyDirty = true;
    const auto &layout = group->layout;
    const auto span = cache->layerSlots.empty() ? fastllm::cuda::CacheSlotSpan{0, cache->slots} :
        cache->layerSlots[tableId];
    const fastllm::cuda::ExpertCacheView metadata{
        cache->keyToSlot, cache->slotKeys, cache->lastUsed, cache->step,
        cache->hitCount, cache->totalMissCount, cache->slots, cache->mappedKeyToSlot,
        span.begin, span.count};
    if (!fastllm::cuda::EnsureExpertCache<kMaxTopK>(
            metadata, indices, tableId * layout.experts, layout.experts, topk,
            cache->routeSlots, cache->missExperts, cache->missSlots, cache->missCount,
            cache->ensureThreads, stream)) return false;
    const auto &shared = group->sharedLayout;
    if (!group->ggufSources.empty()) {
        const auto &source = group->ggufSources[tableId];
        return fastllm_gguf_restore::CopyRecords(source,
            cache->numaPointers + size_t(tableId) * layout.experts * 2 * source.shards,
            cache->records, stream, cache->missExperts, cache->missSlots,
            cache->missCount, cache->slotOffsets, 0, stagedRecord, stagedExpert) == cudaSuccess;
    }
    if (shared.shards > 0) {
        void *const *pointers = cache->numaPointers +
            size_t(tableId) * layout.experts * 2 * shared.shards;
        const uint8_t *metadata = shared.auxiliaryBytes ? group->deviceHostRecords +
            size_t(tableId) * layout.experts * shared.auxiliaryBytes : nullptr;
        return fastllm::cuda::CopySharedExpertRecords(shared, pointers, metadata,
            cache->records, cache->missExperts, cache->missSlots, cache->missCount,
            topk, cache->copyLaunch, stream);
    }
    const auto &recordLayout = group->LayerLayout(tableId);
    const uint8_t *sourceTable = group->deviceHostRecords + group->layerHostOffsets[tableId];
    if (!fastllm::cuda::CopyRecords(
            {sourceTable, cache->records, recordLayout.recordStride,
             layout.recordStride, recordLayout.recordStride, cache->slotOffsets, stagedRecord, stagedExpert},
            cache->missExperts, cache->missSlots, cache->missCount, topk,
            cache->copyLaunch, stream)) return false;
    return cudaGetLastError() == cudaSuccess;
}

__global__ void LookupHybridRoutes(const int32_t *indices, const int32_t *keys,
        const int32_t *slotKeys, int32_t *result, int base, int experts, int topk,
        int resultStride = kMaxTopK) {
    const int r = threadIdx.x;
    if (r < topk) {
        const int expert = indices[r];
        const int key = base + expert;
        const int slot = expert >= 0 && expert < experts ? keys[key] : -1;
        result[r] = expert;
        result[resultStride + r] = slot >= 0 && slotKeys[slot] == key ? slot : -1;
    }
}

// Residency is resolved on the GPU; CPU misses never demand-fill a slot.
__global__ void LookupFrequencyRoutes(const int32_t *indices, const int32_t *keys,
        const int32_t *slotKeys, int32_t *result, int32_t *gpuIndices, int32_t *routeSlots,
        unsigned long long *hits, unsigned long long *misses, int base, int experts, int topk) {
    const int r = threadIdx.x;
    const int expert = r < topk ? indices[r] : -1;
    const int key = base + expert;
    const int slot = expert >= 0 && expert < experts ? keys[key] : -1;
    const bool resident = slot >= 0 && slotKeys[slot] == key;
    const unsigned mask = __ballot_sync(0xffffffff, r < topk && resident);
    if (r == 0) {
        *hits += __popc(mask);
        *misses += topk - __popc(mask);
    }
    if (r < topk) {
        result[r] = expert;
        result[kMaxTopK + r] = resident ? slot : -1;
        gpuIndices[r] = resident ? expert : -1;
        routeSlots[r] = resident ? slot : -1;
    }
}

// Host residency was synchronized before planning. Touch the compact hit list
// without a demand-fill lookup or an empty record-copy launch for each row.
// Row-ordered ages preserve LRU ordering even for repeated expert slots.
__global__ void TouchResidentRoutes(const int32_t *slots, const int32_t *routes,
        unsigned long long *lastUsed, unsigned long long *step,
        unsigned long long *hitCount, int hits, int rows, int topk) {
    __shared__ unsigned long long base;
    if (threadIdx.x == 0) {
        base = *step;
        *step = base + rows;
        *hitCount += hits;
    }
    __syncthreads();
    for (int r = threadIdx.x; r < hits; r += blockDim.x)
        atomicMax(lastUsed + slots[r], base + routes[r] / topk + 1);
}

// Ordered after the current GPU computation and the admitted record's DMA.
__global__ void PublishFrequencyAdmission(int32_t *keys, int32_t *slotKeys,
        int32_t *mappedKeys, unsigned long long *lastUsed, unsigned long long *step,
        int key, int slot) {
    const int old = slotKeys[slot];
    if (old >= 0) {
        keys[old] = -1;
        if (mappedKeys) mappedKeys[old] = -1;
    }
    keys[key] = slot;
    slotKeys[slot] = key;
    lastUsed[slot] = ++*step;
    if (mappedKeys) mappedKeys[key] = slot;
}

// Run after current GPU experts finish reading their slots. Protect/touch
// every still-resident current route and admit at most one cold CPU route.
// Residency is checked here because current GPU refills may have evicted a
// route that was resident at the earlier host lookup.
__global__ void BuildHybridPrefetchRoutes(const int32_t *indices, const int32_t *keys,
        const int32_t *slotKeys, int32_t *requests, int base, int experts, int topk, int candidate) {
    const int r = threadIdx.x;
    if (r >= topk) return;
    const int expert = indices[r];
    const int key = base + expert;
    const int slot = expert >= 0 && expert < experts ? keys[key] : -1;
    requests[r] = (r == candidate || (slot >= 0 && slotKeys[slot] == key)) ? expert : -1;
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

__global__ void ReduceExpertParallel(const float *cpu, const float *gpu,
        const int32_t *owners, const float *scores, float *output,
        int hidden, int topk, int rank) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= hidden) return;
    const int row = blockIdx.y;
    cpu += size_t(row) * topk * hidden;
    gpu += size_t(row) * topk * hidden;
    owners += row * topk;
    scores += row * topk;
    output += row * hidden;
    float sum = 0;
    for (int r = 0; r < topk; ++r) {
        if (owners[r] == rank || (owners[r] < 0 && rank == 0)) {
            const float value = (owners[r] < 0 ? cpu : gpu)[r * hidden + col];
            // Preserve NUMA's fused FP32 weighted accumulation. Splitting the
            // product and sum adds a rounding even when every route is on CPU.
            sum = __fmaf_rn(value, scores[r], sum);
        }
    }
    output[col] = sum;
}

double HybridNowUs() {
    return std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

} // namespace

struct FastllmCudaMoeExpertParallel {
    static constexpr int maxRows = FASTLLM_CUDA_MOE_CACHE_MAX_BATCH;
    static constexpr int maxRoutes = maxRows * kMaxTopK;
    struct Rank {
        fastllm::Data ids, selected, scores, owners, lookup, gate, inputFloat;
        fastllm::Data batchSlots, batchRoutes, batchWorkspace;
        float *host = nullptr, *device = nullptr;
        int32_t *routes = nullptr;
        cudaEvent_t done = nullptr, copyStart = nullptr, copyEnd = nullptr;
        int cudaDevice = -1, hidden = 0, rows = 0, topk = 0, table = -1;
        bool pending = false, ready = false, admitted = false;
        OffloadGroup *group = nullptr;
        DeviceCache *cache = nullptr;
        std::vector<unsigned> heat;
        fastllm::MoeDecodeScheduler::Estimate refill;
        uint64_t calls = 0, gpuRoutes = 0, admissions = 0, residentRoutes = 0;
        std::unique_ptr<DecodeOverlapWorkspace> overlap;
        int overlapAttempted = 0;
        std::vector<int> missedExperts, reuse;
        std::array<int, maxRoutes> stagedSlots;

        float *CpuOutput() { return host + maxRows * hidden; }
        float *GpuOutput() { return device + size_t(maxRoutes) * hidden; }
        int32_t *Indices() { return routes; }
        int32_t *Resident() { return Indices() + maxRoutes; }
        float *Scores() { return reinterpret_cast<float *>(Resident() + maxRoutes); }
        int32_t *Selected() { return reinterpret_cast<int32_t *>(Scores() + maxRoutes); }
        int32_t *Owners() { return Selected() + maxRoutes; }
        int32_t *PrefetchIds() { return Owners() + maxRoutes; }
        int32_t *BatchSlots() { return PrefetchIds() + maxRoutes; }
        int32_t *BatchRoutes() { return BatchSlots() + maxRoutes; }

        ~Rank() {
            int previous = 0;
            cudaGetDevice(&previous);
            if (cudaDevice >= 0) cudaSetDevice(cudaDevice);
            if (pending) {
                cudaEventSynchronize(done);
                if (admitted) cudaEventSynchronize(copyEnd);
            }
            overlap.reset();
            ids.FreeSpace(); selected.FreeSpace(); scores.FreeSpace();
            owners.FreeSpace(); lookup.FreeSpace(); gate.FreeSpace(); inputFloat.FreeSpace();
            batchSlots.FreeSpace(); batchRoutes.FreeSpace(); batchWorkspace.FreeSpace();
            cudaFreeHost(host); cudaFreeHost(routes); cudaFree(device);
            if (done) cudaEventDestroy(done);
            if (copyStart) cudaEventDestroy(copyStart);
            if (copyEnd) cudaEventDestroy(copyEnd);
            cudaSetDevice(previous);
        }

        bool Prepare(int width) {
            if (pending) {
                checkCudaErrors("EP completion", cudaEventSynchronize(done));
                if (overlap) overlap->Observe();
                if (admitted) {
                    checkCudaErrors("EP admission completion", cudaEventSynchronize(copyEnd));
                    float ms = 0;
                    checkCudaErrors("EP refill timing", cudaEventElapsedTime(&ms, copyStart, copyEnd));
                    refill.Observe(ms * 1000);
                }
                pending = false;
            }
            // Peers read Selected() until the next routing rendezvous. Keep
            // routing storage stable when activation workspaces change size.
            if (!routes && cudaMallocHost(&routes, 8 * maxRoutes * sizeof(int32_t)) != cudaSuccess)
                return false;
            if (hidden == width && host && device && done && copyStart && copyEnd) return true;
            cudaFreeHost(host); host = nullptr;
            cudaFree(device); device = nullptr;
            hidden = width;
            return cudaMallocHost(&host, (maxRoutes + maxRows) * size_t(hidden) * sizeof(float)) == cudaSuccess &&
                cudaMalloc(&device, 2 * maxRoutes * size_t(hidden) * sizeof(float)) == cudaSuccess &&
                (done || cudaEventCreateWithFlags(&done, cudaEventDisableTiming) == cudaSuccess) &&
                (copyStart || cudaEventCreate(&copyStart) == cudaSuccess) &&
                (copyEnd || cudaEventCreate(&copyEnd) == cudaSuccess);
        }
    };

    std::vector<std::unique_ptr<Rank>> ranks;
    std::mutex mutex;
    std::condition_variable cv;
    std::atomic<unsigned> generation{0};
    int arrived = 0;
    uint64_t steps = 0, cpuRoutes = 0, multiGpuSteps = 0;
    fastllm::MoeDecodeScheduler::Estimate cpuTime;
    std::vector<fastllm::MoeDecodeScheduler::Estimate> cpuExpert;
    std::vector<fastllm::MoeDecodeOverlapScheduler::RankPlan> plans;

    explicit FastllmCudaMoeExpertParallel(int count) : plans(count) {
        for (int r = 0; r < count; ++r) ranks.emplace_back(new Rank());
    }
    void Barrier() {
        std::unique_lock<std::mutex> lock(mutex);
        const unsigned previous = generation.load(std::memory_order_acquire);
        if (++arrived == (int)ranks.size()) {
            arrived = 0;
            generation.fetch_add(1, std::memory_order_release);
            cv.notify_all();
        } else {
            lock.unlock();
            for (int spin = 0; spin < 2048; ++spin) {
                if (generation.load(std::memory_order_acquire) != previous) return;
#if defined(__x86_64__) || defined(__i386__)
                __builtin_ia32_pause();
#endif
            }
            lock.lock();
            fastllm::AssertInFastLLM(cv.wait_for(lock, std::chrono::seconds(60),
                [&] { return generation.load(std::memory_order_acquire) != previous; }),
                "MoE expert-parallel rendezvous timed out.\n");
        }
    }
};

std::shared_ptr<FastllmCudaMoeExpertParallel> FastllmCudaCreateMoeExpertParallel(int ranks) {
    return ranks > 1 ? std::make_shared<FastllmCudaMoeExpertParallel>(ranks) : nullptr;
}

FastllmCudaMoeExpertParallelStats FastllmCudaGetMoeExpertParallelStats(
        const FastllmCudaMoeExpertParallel &state) {
    FastllmCudaMoeExpertParallelStats result;
    result.steps = state.steps;
    result.cpuRoutes = state.cpuRoutes;
    result.multiGpuSteps = state.multiGpuSteps;
    for (const auto &rank : state.ranks) {
        result.gpuRoutes.push_back(rank->gpuRoutes);
        result.admissions.push_back(rank->admissions);
        result.residentRoutes.push_back(rank->residentRoutes);
    }
    return result;
}

bool FastllmCudaMergeMOEExpertParallel(FastllmCudaMoeExpertParallel &state, int rank,
        const fastllm::Data &input, const fastllm::Data &index, const fastllm::Data &score,
        fastllm::Data &output, fastllm::Data **weights, int weightsBatch, int layer,
        const std::function<void()> &launchParallel) {
#ifdef USE_NUMAS
    using namespace fastllm;
    const int count = state.ranks.size();
    AssertInFastLLM(rank >= 0 && rank < count, "Invalid MoE EP rank.\n");
    auto &work = *state.ranks[rank];
    auto &root = *state.ranks[0];
    constexpr int maxRoutes = FastllmCudaMoeExpertParallel::maxRoutes;
    cudaStreamCaptureStatus capture;
    int device = -1;
    cudaGetDevice(&device);
    work.ready = FastllmCudaMoeCacheRequested() && SupportedCacheInput(input) &&
        cudaStreamIsCapturing(cudaStreamPerThread, &capture) == cudaSuccess &&
        capture == cudaStreamCaptureStatusNone;
    work.group = work.ready ? FindHybridGroup(weights, weightsBatch, &work.table) : nullptr;
    work.ready = work.ready && work.group && !work.group->layout.deepSeekV41 && !work.group->layout.glm5 &&
        input.dims[1] == work.group->LayerLayout(work.table).hidden &&
        (work.cudaDevice < 0 || work.cudaDevice == device);
    if (work.ready) {
        work.cudaDevice = device;
        work.rows = input.dims[0];
        work.ready = work.Prepare(input.dims[1]);
    }
    // CPU experts and the per-expert reduction consume FP32. Convert the
    // small activation batch once per rank, retaining the buffer until its
    // completion event; both CPU and GPU experts then see the same values.
    const Data *expertInput = &input;
    if (work.ready && input.dataType != FLOAT32) {
        work.inputFloat.Resize(input.dims);
        work.inputFloat.ToDevice(CUDA, std::vector<int>{device}, false);
        work.inputFloat.Allocate(false);
        work.ready = input.dataType == FLOAT16
            ? FastllmHalfToFloat(input.cudaData, work.inputFloat.cudaData, input.Count(0))
            : FastllmBF16ToFloat(input.cudaData, work.inputFloat.cudaData, input.Count(0));
        expertInput = &work.inputFloat;
    }
    work.cache = work.ready ? GetDeviceCache(*work.group) : nullptr;
    if (rank == 0 && work.ready) {
        work.ready = PackedCacheRows(index) && PackedCacheRows(score) &&
            index.dims[0] == work.rows && index.dims == score.dims &&
            index.dims[1] > 0 && index.dims[1] <= kMaxTopK &&
            index.dataDevice == DataDevice::CUDA && index.dataType == INT32 && index.cudaData &&
            score.dataDevice == DataDevice::CUDA && score.dataType == FLOAT32 && score.cudaData;
        if (work.ready) {
            work.topk = index.dims[1];
            const int routes = work.rows * work.topk;
            checkCudaErrors("EP routes", cudaMemcpyAsync(work.Indices(), index.cudaData,
                routes * sizeof(int32_t), cudaMemcpyDeviceToHost, cudaStreamPerThread));
            checkCudaErrors("EP scores", cudaMemcpyAsync(work.Scores(), score.cudaData,
                routes * sizeof(float), cudaMemcpyDeviceToHost, cudaStreamPerThread));
            checkCudaErrors("EP input", cudaMemcpyAsync(work.host, expertInput->cudaData,
                work.rows * work.hidden * sizeof(float), cudaMemcpyDeviceToHost, cudaStreamPerThread));
            checkCudaErrors("EP routing", cudaStreamSynchronize(cudaStreamPerThread));
            for (int r = 0; r < routes; ++r)
                work.ready &= work.Indices()[r] >= 0 && work.Indices()[r] < work.group->layout.experts;
        }
    }
    if (rank != 0 && work.cache && work.cache->hostKeyToSlot) {
        // The last EP event alone is insufficient: prefill or a replayed
        // ordinary cache graph may have changed residency since that event.
        // Rank 0 already drains this stream with the input/route readback.
        checkCudaErrors("EP host residency", cudaStreamSynchronize(cudaStreamPerThread));
    }
    state.Barrier();
    bool ready = root.ready, anyCache = false;
    for (int r = 0; r < count; ++r) {
        const auto &other = *state.ranks[r];
        ready &= other.ready && other.group == root.group && other.table == root.table &&
            other.hidden == root.hidden && other.rows == root.rows;
        anyCache |= other.cache != nullptr;
        for (int j = 0; j < r; ++j) ready &= other.cudaDevice != state.ranks[j]->cudaDevice;
    }
    // A second rendezvous protects the shared validation fields from the next
    // layer when one rank rejects before the others have inspected them.
    if (!ready || !anyCache) {
        // A peer or invalid route tensor can reject after this rank queued
        // its conversion. Finish it before fallback may recycle the input.
        if (expertInput != &input)
            checkCudaErrors("EP rejected input", cudaStreamSynchronize(cudaStreamPerThread));
        state.Barrier();
        return false;
    }
    const int topk = root.topk, hidden = root.hidden;
    const int rows = root.rows, routes = rows * topk;
    const auto &layout = root.group->LayerLayout(root.table);
    const bool batchGGUF = layout.weightType == DATA_GGUF_FORMAT;
    const double cpuBudget = state.cpuTime.us;
    auto allocate = [&](Data &tensor, DataType type, std::vector<int> shape) {
        tensor.dataType = type;
        tensor.Resize(shape);
        tensor.ToDevice(DataDevice::CUDA, {device}, false);
        tensor.Allocate(false);
    };
    allocate(work.ids, INT32, {rows, topk});
    if (!batchGGUF) allocate(work.selected, INT32, {rows, topk});
    allocate(work.scores, FLOAT32, {rows, topk});
    allocate(work.owners, INT32, {rows, topk});
    allocate(work.lookup, INT32, {2 * maxRoutes});
    if (rank != 0) {
        std::copy(root.Scores(), root.Scores() + routes, work.Scores());
        std::copy(root.Indices(), root.Indices() + routes, work.Indices());
    }
    for (int r = 0; r < routes; ++r) {
        work.Selected()[r] = work.Indices()[r] % count == rank ? work.Indices()[r] : -1;
    }
    checkCudaErrors("EP score upload", cudaMemcpyAsync(work.scores.cudaData, work.Scores(),
        routes * sizeof(float), cudaMemcpyHostToDevice, cudaStreamPerThread));
    int candidate = -1;
    if (work.cache) {
        // Selected() is changed below as misses are assigned to the CPU.
        // Its upload used to be drained by the residency round trip; retain
        // a stable staging copy now that planning no longer waits for it.
        std::copy_n(work.Selected(), routes, work.PrefetchIds());
        checkCudaErrors("EP route upload", cudaMemcpyAsync(work.ids.cudaData, work.PrefetchIds(),
            routes * sizeof(int32_t), cudaMemcpyHostToDevice, cudaStreamPerThread));
        if (work.cache->hostKeyToSlot) {
            for (int r = 0; r < routes; ++r) {
                const int expert = work.Selected()[r];
                work.Resident()[r] = expert < 0 ? -1 :
                    work.cache->hostKeyToSlot[work.table * layout.experts + expert];
            }
        } else {
            LookupHybridRoutes<<<1, 256, 0, cudaStreamPerThread>>>(
                static_cast<int32_t *>(work.ids.cudaData), work.cache->keyToSlot, work.cache->slotKeys,
                static_cast<int32_t *>(work.lookup.cudaData), work.table * layout.experts, layout.experts, routes, maxRoutes);
            checkCudaErrors("EP residency", cudaMemcpyAsync(work.Resident(),
                static_cast<int32_t *>(work.lookup.cudaData) + maxRoutes,
                routes * sizeof(int32_t), cudaMemcpyDeviceToHost, cudaStreamPerThread));
            checkCudaErrors("EP residency", cudaStreamSynchronize(cudaStreamPerThread));
        }
        if (work.heat.size() != work.group->totalRecords) work.heat.assign(work.group->totalRecords, 0);
        unsigned bestHeat = 0;
        for (int r = 0; r < routes; ++r) {
            const int expert = work.Selected()[r];
            if (expert < 0) continue;
            unsigned &heat = work.heat[work.table * layout.experts + expert];
            heat = std::min(heat + 1, 65535u);
            if (work.Resident()[r] < 0) {
                work.Selected()[r] = -1;
                if (heat >= 2 && heat > bestHeat) { bestHeat = heat; candidate = r; }
            }
        }
        // Prefer copies hidden by CPU work. Sparse probes still warm a cold
        // cache when one refill costs more than the current CPU subset. A
        // coprime interval must visit every layer, not the same few layers.
        int interval = 31;
        while (std::gcd(interval, int(work.group->tableKeys.size())) != 1) ++interval;
        if (work.refill.initialized && cpuBudget > 0 && work.refill.us > cpuBudget &&
            work.calls % interval != 0) candidate = -1;
    } else {
        std::fill_n(work.Selected(), routes, -1);
    }
    // Ordinary GGUF verification borrows immutable pinned weights, restoring
    // CPU layouts during upload. Admission is separate from execution.
    // One decision per unique expert gives every row the same CPU/GPU owner.
    DecodeOverlapWorkspace *overlap = nullptr;
    const int capacity = std::min(routes, (layout.experts + count - 1 - rank) / count);
    if (capacity > 0 && work.cache &&
        layout.weightType == DATA_GGUF_FORMAT && work.group->sharedLayout.shards == 0) {
        if (work.overlapAttempted < capacity) {
            work.overlapAttempted = capacity;
            size_t stride = 0;
            for (const auto &l : work.group->layerLayouts)
                stride = std::max(stride, l.recordStride);
            auto next = std::make_unique<DecodeOverlapWorkspace>();
            if (next->Init(stride, work.group->tableKeys.size() *
                    FastllmCudaMoeExpertParallel::maxRows, capacity, !work.group->ggufSources.empty())) {
                if (work.overlap) next->layers = std::move(work.overlap->layers);
                work.overlap = std::move(next);
            } else (void)cudaGetLastError(); // Retain resident GPU + CPU fallback.
        }
        if (work.overlap && int(work.overlap->expertReady.size()) >= capacity)
            overlap = work.overlap.get();
    }
    const int timingLayer = (rows - 1) * work.group->tableKeys.size() + work.table;
    const int hits = std::count_if(work.Selected(), work.Selected() + routes,
        [](int expert) { return expert >= 0; });
    auto &missedExperts = work.missedExperts;
    auto &reuse = work.reuse;
    auto &stagedSlot = work.stagedSlots;
    missedExperts.clear(); reuse.clear();
    stagedSlot.fill(-1);
    if (batchGGUF) {
        for (int r = 0; r < routes; ++r) {
            if (work.Selected()[r] >= 0) continue;
            const int expert = work.Indices()[r];
            if (expert % count != rank) continue;
            auto found = std::find(missedExperts.begin(), missedExperts.end(), expert);
            if (found == missedExperts.end()) { missedExperts.push_back(expert); reuse.push_back(1); }
            else ++reuse[found - missedExperts.begin()];
        }
        // Spend PCIe bytes first on experts reused by the most verifier rows.
        for (int i = 0; i < int(reuse.size()); ++i)
            for (int j = i + 1; j < int(reuse.size()); ++j)
                if (reuse[j] > reuse[i]) {
                    std::swap(reuse[i], reuse[j]);
                    std::swap(missedExperts[i], missedExperts[j]);
                }
    }
    // Each rank publishes its own inputs; rank 0 fills selected after the
    // rendezvous. The fixed-size plan vector is also used for single-rank EP.
    auto &plan = state.plans[rank];
    plan = {overlap ? &overlap->layers[timingLayer] : nullptr,
        hits, int(reuse.size()), reuse.data(), 0};
    // Submit the independent shared branch before either rank starts CPU
    // expert jobs. Callbacks may lazily prepare CPU-side state, so do not run
    // arbitrary callbacks while the shared NUMA worker pool is active.
    if (launchParallel) launchParallel();
    checkCudaErrors("EP restore device", cudaSetDevice(device));
    state.Barrier();
    if (batchGGUF) {
        if (rank == 0) {
            if (count == 1) {
                if (overlap) plan.selected = overlap->layers[timingLayer].SelectMisses(reuse.size(), hits, reuse.data());
            } else {
                state.cpuExpert.resize(root.group->tableKeys.size() * FastllmCudaMoeExpertParallel::maxRows);
                fastllm::MoeDecodeOverlapScheduler::SelectParallelMisses(state.plans, state.cpuExpert[timingLayer]);
            }
            for (int owner = 0; owner < count; ++owner) {
                auto &other = *state.ranks[owner];
                for (int i = 0; i < state.plans[owner].selected; ++i)
                    for (int r = 0; r < routes; ++r)
                        if (other.Indices()[r] == other.missedExperts[i]) other.stagedSlots[r] = i;
            }
        }
        // Publish one ownership plan before either GPU or the shared NUMA
        // pool starts. No expert is uploaded or computed on both ranks.
        if (count > 1) state.Barrier();
    }
    const int staged = plan.selected;
    const int stagedRoutes = std::count_if(stagedSlot.begin(), stagedSlot.begin() + routes,
        [](int slot) { return slot >= 0; });
    int cpuCount = 0, activeRanks = 0;
    if (rank == 0) {
        for (const auto &other : state.plans)
            activeRanks += other.hits > 0 || other.selected > 0;
    }
    for (int r = 0; r < routes; ++r) {
        const int owner = work.Indices()[r] % count;
        auto &assigned = *state.ranks[owner];
        work.Owners()[r] = assigned.Selected()[r] >= 0 || assigned.stagedSlots[r] >= 0 ? owner : -1;
        cpuCount += work.Owners()[r] < 0;
    }
    const int gpuCount = hits + stagedRoutes;
    std::array<int, maxRoutes + 1> stagedOffsets;
    stagedOffsets[0] = hits;
    if (batchGGUF && gpuCount) {
        // Group resident routes by slot so neighboring CTAs reuse weight
        // cache lines. Staged routes are grouped by their individual DMA.
        int at = 0;
        for (int r = 0; r < routes; ++r)
            if (work.Selected()[r] >= 0) work.BatchRoutes()[at++] = r;
        std::sort(work.BatchRoutes(), work.BatchRoutes() + hits, [&](int a, int b) {
            const int left = work.Resident()[a], right = work.Resident()[b];
            return left != right ? left < right : a < b;
        });
        for (int i = 0; i < hits; ++i) work.BatchSlots()[i] = work.Resident()[work.BatchRoutes()[i]];
        for (int i = 0; i < staged; ++i) {
            for (int r = 0; r < routes; ++r) if (stagedSlot[r] == i) {
                work.BatchSlots()[at] = i;
                work.BatchRoutes()[at++] = r;
            }
            stagedOffsets[i + 1] = at;
        }
        allocate(work.batchSlots, INT32, {routes});
        allocate(work.batchRoutes, INT32, {routes});
        const size_t bytes = FastllmCudaMoeGGUFCacheBatchWorkspaceBytes(hidden, layout.inter, rows, topk);
        if (bytes) allocate(work.batchWorkspace, INT8, {int(bytes)});
    }
    work.admitted = candidate >= 0 && work.cache->slots > 0 && cpuCount + stagedRoutes > 0;
    if (work.admitted && !work.cache->admissionStream) {
        checkCudaErrors("EP admission stream", cudaStreamCreateWithFlags(
            &work.cache->admissionStream, cudaStreamNonBlocking));
    }
    auto submitGpu = [&] {
        const double dispatchStart = HybridNowUs();
        allocate(output, FLOAT32, {rows, hidden});
        if (!batchGGUF)
            checkCudaErrors("EP selected upload", cudaMemcpyAsync(work.selected.cudaData, work.Selected(),
                routes * sizeof(int32_t), cudaMemcpyHostToDevice, cudaStreamPerThread));
        checkCudaErrors("EP owner upload", cudaMemcpyAsync(work.owners.cudaData, work.Owners(),
            routes * sizeof(int32_t), cudaMemcpyHostToDevice, cudaStreamPerThread));
        FastllmCudaMoeGGUFCacheView batchView{};
        if (batchGGUF && gpuCount) {
            checkCudaErrors("Verify slot upload", cudaMemcpyAsync(work.batchSlots.cudaData, work.BatchSlots(),
                gpuCount * sizeof(int32_t), cudaMemcpyHostToDevice, cudaStreamPerThread));
            checkCudaErrors("Verify row upload", cudaMemcpyAsync(work.batchRoutes.cudaData, work.BatchRoutes(),
                gpuCount * sizeof(int32_t), cudaMemcpyHostToDevice, cudaStreamPerThread));
            // An EP rank without a usable cache contributes CPU routes only.
            batchView = {work.cache->records, static_cast<int32_t *>(work.batchSlots.cudaData),
                layout.recordStride, layout.downOffset,
                layout.gateGgmlType, layout.downGgmlType, hidden, layout.inter,
                work.batchWorkspace.cudaData, size_t(work.batchWorkspace.GetBytes()), work.cache->slotOffsets};
            batchView.routeMap = static_cast<int32_t *>(work.batchRoutes.cudaData);
            batchView.routeCount = hits;
        }
        // Allocate before starting DMA: a growing scratch buffer may call
        // cudaMalloc, which must not turn the first batch into a serial copy.
        if (gpuCount) allocate(work.gate, FLOAT32, {batchGGUF ? routes : topk, layout.inter});
        if (staged) overlap->PrepareCopy(0);
        if (hits) {
            if (overlap) checkCudaErrors("Verify resident start", cudaEventRecord(overlap->residentStart, cudaStreamPerThread));
            if (batchGGUF) {
                TouchResidentRoutes<<<1, 256, 0, cudaStreamPerThread>>>(
                    batchView.routeSlots, batchView.routeMap, work.cache->lastUsed,
                    work.cache->step, work.cache->hitCount, hits, rows, topk);
                AssertInFastLLM(ComputeGGUFExperts(*expertInput, work.gate, output,
                    layout, batchView, static_cast<float *>(work.scores.cudaData), topk,
                    work.GpuOutput()), "Verify resident batch failed.\n");
            } else {
                // Snapshot residency for the entire verifier before any admission.
                // Repeated experts consequently keep one owner across all rows and
                // CPU rows can share weight reads in the grouped NUMA path.
                for (int row = 0; row < rows; ++row) {
                    if (std::none_of(work.Selected() + row * topk,
                            work.Selected() + (row + 1) * topk, [](int expert) { return expert >= 0; })) continue;
                    Data inputRow, outputRow;
                    inputRow.FakeFrom(*expertInput, size_t(row) * hidden * sizeof(float));
                    inputRow.Resize({1, hidden}); inputRow.dataDeviceIds = expertInput->dataDeviceIds;
                    outputRow.FakeFrom(output, size_t(row) * hidden * sizeof(float));
                    outputRow.Resize({1, hidden}); outputRow.dataDeviceIds = output.dataDeviceIds;
                    AssertInFastLLM(EnsureCachedExperts(work.group, work.cache, work.table,
                        static_cast<int32_t *>(work.selected.cudaData) + row * topk, topk), "EP resident lookup failed.\n");
                    AssertInFastLLM(FindExpertCacheBackend(layout.weightType)->compute(inputRow, work.gate, outputRow,
                        layout, *work.cache, static_cast<float *>(work.scores.cudaData) + row * topk, topk,
                        work.GpuOutput() + size_t(row) * topk * hidden), "EP expert compute failed.\n");
                }
            }
        }
        if (hits && overlap)
            checkCudaErrors("Verify resident done", cudaEventRecord(overlap->residentDone, cudaStreamPerThread));
        if (staged) {
            FastllmCudaMoeGGUFCacheView temporary{
                overlap->records, nullptr, layout.recordStride, layout.downOffset,
                layout.gateGgmlType, layout.downGgmlType, layout.hidden, layout.inter,
                batchView.workspace, batchView.workspaceBytes};
            // All calls retain the same input-row layout and Q8 prefix, even
            // when each expert serves a different number of verifier rows.
            temporary.q8InputPrepared = hits > 0;
            for (int i = 0; i < staged; ++i) {
                overlap->CopyExpert(*work.group, work.table, missedExperts[i], i, staged);
                checkCudaErrors("Verify wait expert", cudaStreamWaitEvent(cudaStreamPerThread, overlap->expertReady[i], 0));
                checkCudaErrors("Verify staged start", cudaEventRecord(overlap->stagedStart[i], cudaStreamPerThread));
                temporary.routeSlots = batchView.routeSlots + stagedOffsets[i];
                temporary.routeMap = batchView.routeMap + stagedOffsets[i];
                temporary.routeCount = stagedOffsets[i + 1] - stagedOffsets[i];
                AssertInFastLLM(ComputeGGUFExperts(*expertInput, work.gate, output, layout, temporary,
                    static_cast<float *>(work.scores.cudaData), topk, work.GpuOutput()),
                    "Verify staged expert batch failed.\n");
                temporary.q8InputPrepared = true;
                checkCudaErrors("Verify staged done", cudaEventRecord(overlap->stagedDone[i], cudaStreamPerThread));
            }
            overlap->layers[timingLayer].dispatch.Observe(HybridNowUs() - dispatchStart);
        }
    };
    const bool overlapCpu = rank == 0 && cpuCount > 0 && (rows == 1 || batchGGUF);
    double cpuStart = 0;
    if (overlapCpu) {
        cpuStart = HybridNowUs();
        NumasMoeDecodeExpertsBatchWithOverlap(root.host, root.CpuOutput(), rows, weights, weightsBatch,
            root.Indices(), root.Owners(), root.Scores(), topk, layer, submitGpu);
    } else {
        submitGpu();
    }
    if (rank == 0) {
        const double start = HybridNowUs();
        if (cpuCount) {
            if (!overlapCpu)
                NumasMoeDecodeExpertsBatch(root.host, root.CpuOutput(), rows, weights, weightsBatch,
                    root.Indices(), root.Owners(), root.Scores(), topk, layer);
            const double cpuUs = HybridNowUs() - (overlapCpu ? cpuStart : start);
            state.cpuTime.Observe(cpuUs);
            if (batchGGUF && count > 1) state.cpuExpert[timingLayer].Observe(cpuUs / cpuCount);
            else if (overlap) overlap->layers[timingLayer].cpuExpert.Observe(cpuUs / cpuCount);
            checkCudaErrors("EP CPU output", cudaMemcpyAsync(work.device, root.CpuOutput(),
                size_t(routes) * hidden * sizeof(float), cudaMemcpyHostToDevice, cudaStreamPerThread));
        }
        ++state.steps;
        state.cpuRoutes += cpuCount;
        state.multiGpuSteps += activeRanks > 1;
    }
    ReduceExpertParallel<<<dim3((hidden + 255) / 256, rows), 256, 0, cudaStreamPerThread>>>(
        work.device, work.GpuOutput(),
        static_cast<int32_t *>(work.owners.cudaData), static_cast<float *>(work.scores.cudaData),
        static_cast<float *>(output.cudaData), hidden, topk, rank);
    checkCudaErrors("EP reduction", cudaGetLastError());
    checkCudaErrors("EP done", cudaEventRecord(work.done, cudaStreamPerThread));
    if (work.admitted) {
        // The decision is unchanged; execute it after this layer has finished
        // reading its slots, beside the following layer's attention/dense work.
        const auto stream = work.cache->admissionStream;
        checkCudaErrors("EP admission dependency", cudaStreamWaitEvent(stream, work.done, 0));
        checkCudaErrors("EP refill start", cudaEventRecord(work.copyStart, stream));
        BuildHybridPrefetchRoutes<<<1, 32, 0, stream>>>(
            static_cast<int32_t *>(work.ids.cudaData) + (candidate / topk) * topk,
            work.cache->keyToSlot, work.cache->slotKeys,
            static_cast<int32_t *>(work.lookup.cudaData), work.table * layout.experts,
            layout.experts, topk, candidate % topk);
        // A staged expert is already in VRAM. Promote those exact packed bytes
        // rather than reading the same host record over PCIe a second time.
        const int stagedIndex = stagedSlot[candidate];
        const uint8_t *record = stagedIndex >= 0
            ? overlap->records + size_t(stagedIndex) * layout.recordStride : nullptr;
        AssertInFastLLM(EnsureCachedExperts(work.group, work.cache, work.table,
            static_cast<int32_t *>(work.lookup.cudaData), topk, stream,
            record, work.Indices()[candidate]), "EP cache admission failed.\n");
        checkCudaErrors("EP refill end", cudaEventRecord(work.copyEnd, stream));
        checkCudaErrors("EP publish admission", cudaEventRecord(work.cache->admissionDone, stream));
        ++work.admissions;
    }
    work.pending = true;
    work.gpuRoutes += gpuCount;
    work.residentRoutes += hits;
    if (overlap) {
        overlap->previousLayer = timingLayer;
        overlap->previousHits = hits;
        overlap->previousMisses = staged;
        overlap->previousStagedRoutes = stagedRoutes;
        ++overlap->layers[timingLayer].calls;
    }
    ++work.calls;
    // Each rank has its own copy of the authoritative routes. GPU decisions
    // are overwritten only after the next call's first rendezvous, so peers
    // may enqueue their TP collective while rank 0 finishes the CPU subset.
    return true;
#else
    return false;
#endif
}

void *FastllmCudaBeginMoeDecode(fastllm::Data **weights, int weightsBatch, int topk) {
    if (topk < 1 || topk > kMaxTopK) return nullptr;
    auto *group = FindHybridGroup(weights, weightsBatch);
    if (!group) return nullptr;
    auto *cache = GetDeviceCache(*group);
    if (!cache) return nullptr;
    if (group->layout.weightType == fastllm::DataType::DATA_GGUF_FORMAT &&
        cache->slots < int(group->totalRecords)) {
        if (!cache->frequency) {
            std::vector<int> keys(group->totalRecords), slots(cache->slots);
            std::unordered_map<int, int> partitions;
            for (int layer = 0; layer < int(group->tableKeys.size()); ++layer) {
                const auto span = cache->layerSlots[layer];
                const int id = partitions.emplace(span.begin, partitions.size()).first->second;
                std::fill_n(keys.begin() + layer * group->layout.experts, group->layout.experts, id);
                std::fill_n(slots.begin() + span.begin, span.count, id);
            }
            cache->frequency = std::make_unique<fastllm::MoeFrequencyPolicy>(
                std::move(keys), std::move(slots), group->tableKeys.size());
        }
        if (cache->frequencyResidencyDirty) {
            std::vector<int> owners(cache->slots);
            if (!owners.empty()) checkCudaErrors("MoE frequency residency", cudaMemcpyAsync(owners.data(), cache->slotKeys,
                owners.size() * sizeof(int), cudaMemcpyDeviceToHost, cudaStreamPerThread));
            checkCudaErrors("MoE frequency residency", cudaStreamSynchronize(cudaStreamPerThread));
            cache->frequency->SetResidents(owners.data());
            cache->frequencyResidencyDirty = false;
        }
        cache->frequency->BeginStep();
        cache->frequencyActive = true;
        return cache;
    }
    // GLM's full-resident and non-GGUF caches retain their existing scheduler.
    if (group->layout.glm5) return nullptr;
    if (!cache->decode) {
        cache->decode = std::make_unique<DecodePolicyState>(
            group->totalRecords, cache->slots, group->tableKeys.size(), topk, cache->layerSlots);
    }
    cache->decode->startUs = HybridNowUs();
    return cache;
}

void FastllmCudaEndMoeDecode(void *state) {
    if (!state) return;
    auto &cache = *static_cast<DeviceCache *>(state);
    if (cache.frequency) {
        cache.frequencyActive = false;
        return;
    }
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

namespace {
#ifdef USE_NUMAS
int V41CacheOption(const char *name, int fallback) {
    const char *v = std::getenv(name);
    if (!v || !*v)
        return fallback;
    char *end = nullptr;
    long n = std::strtol(v, &end, 10);
    return end != v && !*end && n >= 0 && n <= 1000000 ? int(n) : fallback;
}

__global__ void LookupVerifyRoutes(const int32_t *indices, const int32_t *keys, const int32_t *slotKeys,
                                   unsigned long long *lastUsed, unsigned long long *step, int32_t *result, int base,
                                   int experts, int routes) {
    __shared__ unsigned long long stamp;
    if (threadIdx.x == 0)
        stamp = atomicAdd(step, 1ULL) + 1;
    __syncthreads();
    const int r = threadIdx.x;
    if (r >= routes)
        return;
    const int id = indices[r], key = base + id;
    const int slot = id >= 0 && id < experts ? keys[key] : -1;
    const bool hit = slot >= 0 && slotKeys[slot] == key;
    result[r] = id;
    result[kMaxVerifyRoutes + r] = hit ? slot : -1;
    if (hit)
        atomicMax(lastUsed + slot, stamp);
}

bool TryV41VerifyHybrid(const fastllm::Data &input, const fastllm::Data &index, const fastllm::Data &score,
                        fastllm::Data &output, fastllm::Data **weights, int weightsBatch, int layer,
                        const std::function<void()> &launchParallel) {
    if (!SupportedCacheInput(input) || input.dataType != fastllm::DataType::BFLOAT16 || input.dims[0] < 2 ||
        input.dims[0] > kMaxVerifyRows || !PackedCacheRows(index) || index.dims[0] != input.dims[0] ||
        index.dims[1] < 1 || index.dims[1] > kMaxTopK || index.dataType != fastllm::DataType::INT32 ||
        index.dataDevice != fastllm::DataDevice::CUDA || !index.cudaData || !PackedCacheRows(score) ||
        score.dims != index.dims || score.dataType != fastllm::DataType::FLOAT32 ||
        score.dataDevice != fastllm::DataDevice::CUDA || !score.cudaData)
        return false;
    const char *mode = std::getenv("FASTLLM_DSV41_MOE_CACHE_MODE");
    if (mode && std::strcmp(mode, "gpu") == 0)
        return false;
    auto *group = FindHybridGroup(weights, weightsBatch);
    if (!group || !group->layout.deepSeekV41 || input.dims[1] != group->layout.hidden)
        return false;
    cudaStreamCaptureStatus capture;
    if (cudaStreamIsCapturing(cudaStreamPerThread, &capture) != cudaSuccess || capture != cudaStreamCaptureStatusNone)
        return false;
    int tableId;
    FindGroup(weights, weightsBatch, &tableId);
    auto *cache = GetDeviceCache(*group);
    if (!cache)
        return false;
    const auto &layout = group->layout;
    const int hidden = layout.hidden, rows = input.dims[0], topk = index.dims[1], routes = rows * topk;
    if (!cache->verify) {
        auto w = std::make_unique<VerifyWorkspace>();
        const size_t hostBytes = kMaxVerifyRows * hidden * sizeof(uint16_t) + 5 * kMaxVerifyRoutes * sizeof(int32_t) +
                                 size_t(kMaxVerifyRoutes) * hidden * sizeof(float);
        const size_t deviceBytes =
            2 * size_t(kMaxVerifyRoutes) * hidden * sizeof(float) + 5 * kMaxVerifyRoutes * sizeof(int32_t);
        if (cudaMallocHost(&w->host, hostBytes) != cudaSuccess || cudaMalloc(&w->device, deviceBytes) != cudaSuccess ||
            cudaEventCreate(&w->start) != cudaSuccess || cudaEventCreate(&w->computed) != cudaSuccess ||
            cudaEventCreate(&w->done) != cudaSuccess || cudaEventCreate(&w->prefetchStart) != cudaSuccess ||
            cudaEventCreate(&w->prefetchDone) != cudaSuccess)
            return false;
        w->heat.resize(group->totalRecords, 0);
        w->lastSeen.resize(group->totalRecords, 0);
        cache->verify = std::move(w);
    }
    auto &w = *cache->verify;
    if (w.pending) {
        checkCudaErrors("Verify cache completion", cudaEventSynchronize(w.done));
        if (w.previousGpuRoutes) {
            float ms;
            checkCudaErrors("Verify cache timing", cudaEventElapsedTime(&ms, w.start, w.computed));
            w.gpuRoute.Observe(ms * 1000 / w.previousGpuRoutes);
        }
        if (w.previousPrefetch) {
            float ms;
            checkCudaErrors("Verify refill timing", cudaEventElapsedTime(&ms, w.prefetchStart, w.prefetchDone));
            w.refill.Observe(ms * 1000);
        }
    }
    w.previousGpuRoutes = 0;
    w.previousPrefetch = false;
    auto *hostInput = reinterpret_cast<uint16_t *>(w.host);
    auto *ids = reinterpret_cast<int32_t *>(hostInput + kMaxVerifyRows * hidden);
    auto *slots = ids + kMaxVerifyRoutes;
    auto *gpuIds = slots + kMaxVerifyRoutes;
    auto *scores = reinterpret_cast<float *>(gpuIds + kMaxVerifyRoutes);
    auto *requests = reinterpret_cast<int32_t *>(scores + kMaxVerifyRoutes);
    auto *cpu = reinterpret_cast<float *>(requests + kMaxVerifyRoutes);
    float *dCpu = w.device, *dGpu = dCpu + size_t(kMaxVerifyRoutes) * hidden;
    auto *meta = reinterpret_cast<int32_t *>(dGpu + size_t(kMaxVerifyRoutes) * hidden);
    auto *dSlots = meta + 2 * kMaxVerifyRoutes, *dGpuIds = dSlots + kMaxVerifyRoutes,
         *dRequests = dGpuIds + kMaxVerifyRoutes;
    LookupVerifyRoutes<<<1, kMaxVerifyRoutes, 0, cudaStreamPerThread>>>(
        (const int32_t *)index.cudaData, cache->keyToSlot, cache->slotKeys, cache->lastUsed, cache->step, meta,
        tableId * layout.experts, layout.experts, routes);
    checkCudaErrors("Verify route lookup", cudaMemcpyAsync(ids, meta, 2 * kMaxVerifyRoutes * sizeof(int32_t),
                                                           cudaMemcpyDeviceToHost, cudaStreamPerThread));
    checkCudaErrors("Verify input staging", cudaMemcpyAsync(hostInput, input.cudaData, size_t(rows) * hidden * 2,
                                                            cudaMemcpyDeviceToHost, cudaStreamPerThread));
    checkCudaErrors("Verify scores staging", cudaMemcpyAsync(scores, score.cudaData, routes * sizeof(float),
                                                             cudaMemcpyDeviceToHost, cudaStreamPerThread));
    checkCudaErrors("Verify staging completion", cudaStreamSynchronize(cudaStreamPerThread));
    struct Expert {
        int id, slot, count;
    };
    std::vector<Expert> experts;
    experts.reserve(routes);
    int hits = 0;
    for (int r = 0; r < routes; ++r) {
        fastllm::AssertInFastLLM(ids[r] >= 0 && ids[r] < layout.experts, "Invalid V4.1 verify expert index.\n");
        hits += slots[r] >= 0;
        auto it = std::find_if(experts.begin(), experts.end(), [&](const Expert &e) { return e.id == ids[r]; });
        if (it == experts.end())
            experts.push_back({ids[r], slots[r], 1});
        else
            ++it->count;
    }
    ++w.calls;
    w.routes += routes;
    w.hits += hits;
    // Repeated rows already share CPU weight reads. Offload the cheapest
    // resident groups first, preserving whole groups on either backend.
    std::stable_sort(experts.begin(), experts.end(), [](const Expert &a, const Expert &b) {
        if ((a.slot >= 0) != (b.slot >= 0))
            return a.slot >= 0;
        return a.count < b.count;
    });
    const int resident = std::count_if(experts.begin(), experts.end(), [](const Expert &e) { return e.slot >= 0; });
    const double cpuUnit = w.cpuUnit.initialized ? w.cpuUnit.us : 70.0;
    const double gpuUnit = w.gpuRoute.initialized ? w.gpuRoute.us : 80.0;
    const double allCpuUnits = experts.size() + 0.35 * (routes - experts.size());
    int gpu = 0, gpuRoutes = 0, candidateRoutes = 0;
    double selectedCpu = allCpuUnits * cpuUnit, selectedGpu = 0, best = selectedCpu;
    for (int n = 1; n <= std::min(resident, kMaxTopK - 1); ++n) {
        candidateRoutes += experts[n - 1].count;
        const double cpuUs = (experts.size() - n + 0.35 * (routes - candidateRoutes - (experts.size() - n))) * cpuUnit;
        const double gpuUs = 20 + candidateRoutes * gpuUnit;
        if (std::max(cpuUs, gpuUs) + 20 < best) {
            best = std::max(cpuUs, gpuUs) + 20;
            gpu = n;
            gpuRoutes = candidateRoutes;
            selectedCpu = cpuUs;
            selectedGpu = gpuUs;
        }
    }
    const int forced = V41CacheOption("FASTLLM_DSV41_MOE_CACHE_GPU_EXPERTS", 1000000);
    if (forced < 1000000) {
        gpu = std::min({forced, resident, kMaxTopK - 1});
        gpuRoutes = 0;
        for (int i = 0; i < gpu; ++i)
            gpuRoutes += experts[i].count;
        selectedCpu = (experts.size() - gpu + 0.35 * (routes - gpuRoutes - (experts.size() - gpu))) * cpuUnit;
        selectedGpu = gpuRoutes ? 20 + gpuRoutes * gpuUnit : 0;
    }
    int prefetch = -1;
    unsigned bestHeat = 1;
    const bool prefetchEnabled = V41CacheOption("FASTLLM_DSV41_MOE_CACHE_PREFETCH", 31) > 0;
    const double refill = w.refill.initialized ? w.refill.us : 750.0;
    for (const auto &e : experts) {
        const int key = tableId * layout.experts + e.id;
        if (w.calls - w.lastSeen[key] > group->tableKeys.size() * 64)
            w.heat[key] = 0;
        w.heat[key] = std::min(w.heat[key] + 1, 65535u);
        w.lastSeen[key] = w.calls;
        if (prefetchEnabled && e.slot < 0 && w.heat[key] > bestHeat && selectedCpu - selectedGpu > refill * 1.1) {
            prefetch = e.id;
            bestHeat = w.heat[key];
        }
    }
    auto report = [&]() {
        if (V41CacheOption("FASTLLM_DSV41_MOE_CACHE_TRACE", 0) && (w.calls == 1 || w.calls % 1024 == 0))
            std::fprintf(stderr,
                         "[Fastllm] V4.1 verify cache: device %d, %llu calls, %llu routes, %llu resident routes, %llu "
                         "GPU routes, %llu GPU groups, %llu admissions, %llu fallbacks; CPU %.3f us/unit, GPU %.3f "
                         "us/route, refill %.3f us.\n",
                         cache->device, (unsigned long long)w.calls, (unsigned long long)w.routes,
                         (unsigned long long)w.hits, (unsigned long long)w.gpuRoutes, (unsigned long long)w.gpuExperts,
                         (unsigned long long)w.admissions, (unsigned long long)w.fallbacks, cpuUnit, gpuUnit, refill);
    };
    if (!gpu && prefetch < 0) {
        ++w.fallbacks;
        report();
        return false;
    }
    for (int r = 0; r < routes; ++r) {
        bool selected = false;
        for (int j = 0; j < gpu; ++j)
            selected |= ids[r] == experts[j].id;
        gpuIds[r] = selected ? ids[r] : -1;
        if (!selected)
            slots[r] = -1;
    }
    output.dataType = fastllm::DataType::BFLOAT16;
    output.Resize({rows, hidden});
    output.ToDevice(fastllm::DataDevice::CUDA, {cache->device}, false);
    output.Allocate(false);
    if (gpu) {
        checkCudaErrors("Verify GPU slots", cudaMemcpyAsync(dSlots, slots, routes * sizeof(int32_t),
                                                            cudaMemcpyHostToDevice, cudaStreamPerThread));
        checkCudaErrors("Verify GPU split", cudaMemcpyAsync(dGpuIds, gpuIds, routes * sizeof(int32_t),
                                                            cudaMemcpyHostToDevice, cudaStreamPerThread));
        w.quantizedInput.dataType = input.dataType;
        w.quantizedInput.Resize(input.dims);
        w.quantizedInput.ToDevice(fastllm::DataDevice::CUDA, {cache->device}, false);
        fastllm::AssertInFastLLM(FastllmCudaDeepSeekV41QuantizeActivation(input, w.quantizedInput),
                                 "Verify input quantization failed.\n");
        w.activation.dataType = input.dataType;
        w.activation.Resize({routes, layout.inter});
        w.activation.ToDevice(fastllm::DataDevice::CUDA, {cache->device}, false);
        w.activation.Allocate(false);
        checkCudaErrors("Verify GPU start", cudaEventRecord(w.start, cudaStreamPerThread));
        using namespace fastllm::cuda::dsv41_cache;
        if (layout.weightType == fastllm::DATA_GGUF_FORMAT) {
            const FastllmCudaMoeGGUFCacheView view{cache->records, dSlots, layout.recordStride, layout.downOffset,
                layout.gateGgmlType, layout.downGgmlType, hidden, layout.inter,
                cache->ggufWorkspace, cache->ggufWorkspaceBytes};
            fastllm::AssertInFastLLM(FastllmCudaMoeV41GGUFCacheCompute(w.quantizedInput, w.activation,
                view, (const float *)score.cudaData, topk, layout.swigluLimit, dGpu),
                "Verify Q2 CUDA experts failed.\n");
        } else {
            Gate<true><<<dim3(layout.inter / 32, routes), 512, 0, cudaStreamPerThread>>>(
                (const __nv_bfloat16 *)w.quantizedInput.cudaData, dSlots, cache->records, (const float *)score.cudaData,
                (__nv_bfloat16 *)w.activation.cudaData, hidden, layout.inter, layout.recordStride, layout.swigluLimit,
                topk);
            Down<<<dim3(hidden / 8, routes), 128, 0, cudaStreamPerThread>>>(
                (const __nv_bfloat16 *)w.activation.cudaData, dSlots, cache->records, dGpu, hidden, layout.inter,
                layout.recordStride, layout.downOffset);
        }
        checkCudaErrors("Verify CUDA experts", cudaGetLastError());
        checkCudaErrors("Verify GPU finish", cudaEventRecord(w.computed, cudaStreamPerThread));
        w.previousGpuRoutes = gpuRoutes;
        w.gpuRoutes += gpuRoutes;
        w.gpuExperts += gpu;
    }
    if (prefetch >= 0) {
        for (int i = 0; i < gpu; ++i)
            requests[i] = experts[i].id;
        requests[gpu] = prefetch;
        checkCudaErrors("Verify admission list", cudaMemcpyAsync(dRequests, requests, (gpu + 1) * sizeof(int32_t),
                                                                 cudaMemcpyHostToDevice, cudaStreamPerThread));
        checkCudaErrors("Verify refill start", cudaEventRecord(w.prefetchStart, cudaStreamPerThread));
        // Current GPU kernels finish before eviction/refill. Protect selected
        // residents as well, then overlap at most one cold copy with CPU work.
        fastllm::AssertInFastLLM(EnsureCachedExperts(group, cache, tableId, dRequests, gpu + 1),
                                 "Verify cache admission failed.\n");
        checkCudaErrors("Verify refill end", cudaEventRecord(w.prefetchDone, cudaStreamPerThread));
        w.previousPrefetch = true;
        ++w.admissions;
    }
    // Finish routing reads and all fallback decisions before launching TP
    // shared experts; their GPU work can overlap the remaining NUMA subset.
    if (launchParallel) {
        launchParallel();
        checkCudaErrors("Verify restore device", cudaSetDevice(cache->device));
    }
    const auto start = HybridNowUs();
    fastllm::NumasMoeVerifyExperts(hostInput, cpu, rows, weights, weightsBatch, ids, gpuIds, scores, topk, layer,
                                   layout.swigluLimit, gpu > 0);
    const double units = experts.size() - gpu + 0.35 * (routes - gpuRoutes - (experts.size() - gpu));
    if (units > 0)
        w.cpuUnit.Observe((HybridNowUs() - start) / units);
    if (gpu) {
        if (gpuRoutes < routes)
            checkCudaErrors("Verify CPU expert output",
                            cudaMemcpyAsync(dCpu, cpu, size_t(routes) * hidden * sizeof(float), cudaMemcpyHostToDevice,
                                            cudaStreamPerThread));
        fastllm::cuda::dsv41_cache::Reduce<<<dim3((hidden + 255) / 256, rows), 256, 0, cudaStreamPerThread>>>(
            dCpu, dGpu, dGpuIds, (const int32_t *)index.cudaData, (__nv_bfloat16 *)output.cudaData, hidden, topk);
        checkCudaErrors("Verify ordered reduction", cudaGetLastError());
    } else
        checkCudaErrors("Verify CPU sum", cudaMemcpyAsync(output.cudaData, cpu, size_t(rows) * hidden * 2,
                                                          cudaMemcpyHostToDevice, cudaStreamPerThread));
    checkCudaErrors("Verify completion", cudaEventRecord(w.done, cudaStreamPerThread));
    w.pending = true;
    report();
    return true;
}
#endif
} // namespace

bool FastllmCudaMergeMOEHybrid(const fastllm::Data &input,
        const fastllm::Data &index, const fastllm::Data &score,
        fastllm::Data &output, fastllm::Data **weights, int weightsBatch, int layer,
        const std::function<void()> &launchParallel) {
#ifdef USE_NUMAS
    if (input.dims.size() == 2 && input.dims[0] > 1) {
        if (input.dataType == fastllm::DataType::BFLOAT16)
            return TryV41VerifyHybrid(input, index, score, output, weights, weightsBatch, layer, launchParallel);
        if (input.dataType != fastllm::DataType::FLOAT32 || !SupportedCacheInput(input)) return false;
        auto *group = FindHybridGroup(weights, weightsBatch);
        if (!group || group->layout.deepSeekV41 || group->layout.glm5) return false;
        auto *cache = GetDeviceCache(*group);
        if (!cache) return false;
        // Keep resident experts on GPU and dynamically split GGUF misses
        // between NUMA and streamed GPU execution, deduplicating weight DMA
        // across verifier rows. Other layouts retain the resident/NUMA path.
        if (!cache->batchHybrid)
            cache->batchHybrid = std::make_shared<FastllmCudaMoeExpertParallel>(1);
        return FastllmCudaMergeMOEExpertParallel(*cache->batchHybrid, 0,
            input, index, score, output, weights, weightsBatch, layer, launchParallel);
    }
    cudaStreamCaptureStatus capturing;
    if (cudaStreamIsCapturing(cudaStreamPerThread, &capturing) != cudaSuccess ||
        capturing != cudaStreamCaptureStatusNone) return false;
    if ((input.dataType != fastllm::DataType::FLOAT32 && input.dataType != fastllm::DataType::BFLOAT16) ||
        input.dims.size() != 2 ||
        input.dims[0] != 1 || !FastllmCudaCanRunMoeHybrid(weights, weightsBatch) ||
        !FastllmCudaCanRunMoeCacheSmallBatch(
            input, index, score, weights, weightsBatch, fastllm::MoeGateSwiglu)) return false;
    int tableId;
    auto *group = FindGroup(weights, weightsBatch, &tableId);
    auto *cache = GetDeviceCache(*group);
    const auto &layout = group->LayerLayout(tableId);
    auto *frequency = cache->frequencyActive ? cache->frequency.get() : nullptr;
    if (!frequency && cache->decode && cache->decode->policy.UseGpu()) return false;
    const bool scoredBfloat = layout.deepSeekV41 || layout.glm5;
    if (scoredBfloat != (input.dataType == fastllm::DataType::BFLOAT16)) return false;
    const int hidden = layout.hidden, topk = index.dims[1];
    if (!cache->hybrid) {
        auto work = std::make_unique<HybridWorkspace>();
        // GGUF layers can have different hidden widths in the same pool.
        // Each call uses its layer's offsets within this maximum-sized buffer.
        int workspaceHidden = hidden;
        for (const auto &layerLayout : group->layerLayouts)
            workspaceHidden = std::max(workspaceHidden, layerLayout.hidden);
        const size_t metaBytes = 4 * kMaxTopK * sizeof(int32_t);
        if (cudaMallocHost(&work->host, (kMaxTopK + 1) * workspaceHidden * sizeof(float) + metaBytes) != cudaSuccess ||
            cudaMalloc(&work->device, 2 * kMaxTopK * workspaceHidden * sizeof(float) + metaBytes) != cudaSuccess ||
            cudaEventCreate(&work->start) != cudaSuccess ||
            cudaEventCreate(&work->copied) != cudaSuccess ||
            cudaEventCreate(&work->computed) != cudaSuccess ||
            cudaEventCreate(&work->done) != cudaSuccess ||
            cudaEventCreate(&work->prefetchStart) != cudaSuccess ||
            cudaEventCreate(&work->prefetchDone) != cudaSuccess) return false;
        cache->hybrid = std::move(work);
    }
    auto &work = *cache->hybrid;
    const char *cacheMode = std::getenv("FASTLLM_DSV41_MOE_CACHE_MODE");
    if (layout.deepSeekV41 && cacheMode && std::strcmp(cacheMode, "gpu") == 0) {
        // Keep routing, demand refill and expert execution on the same CUDA
        // stream. In particular, do not copy activations/routes to NUMA or
        // synchronize each layer merely to decide a CPU/GPU split.
        auto snapshot = [&]() {
            auto *counts = reinterpret_cast<unsigned long long *>(work.host);
            checkCudaErrors("Pure MoE hits", cudaMemcpyAsync(counts, cache->hitCount,
                sizeof(*counts), cudaMemcpyDeviceToHost, cudaStreamPerThread));
            checkCudaErrors("Pure MoE misses", cudaMemcpyAsync(counts + 1, cache->totalMissCount,
                sizeof(*counts), cudaMemcpyDeviceToHost, cudaStreamPerThread));
            checkCudaErrors("Pure MoE statistics", cudaStreamSynchronize(cudaStreamPerThread));
            return std::array<uint64_t, 2>{counts[0], counts[1]};
        };
        if (!work.pureActive) {
            // This also finishes any hybrid work using the pinned buffer.
            const auto counts = snapshot();
            work.pureBaseHits = counts[0];
            work.pureBaseMisses = counts[1];
            work.pureCalls = work.pureRoutes = 0;
            work.previousGpu = work.previousMisses = 0;
            work.previousPrefetch = false;
            work.pureActive = true;
        }
        output.dataType = input.dataType;
        output.Resize({1, hidden});
        output.ToDevice(fastllm::DataDevice::CUDA, {cache->device}, false);
        output.Allocate(false);
        work.quantizedInput.dataType = input.dataType;
        work.quantizedInput.dataDevice = input.dataDevice;
        work.quantizedInput.dataDeviceIds = input.dataDeviceIds;
        work.quantizedInput.Resize(input.dims);
        fastllm::AssertInFastLLM(FastllmCudaDeepSeekV41QuantizeActivation(input, work.quantizedInput),
                               "Pure V4.1 cache input quantization failed.\n");
        auto &gateOutput = work.gateOutput;
        gateOutput.dataType = input.dataType;
        gateOutput.dataDevice = input.dataDevice;
        gateOutput.dataDeviceIds = input.dataDeviceIds;
        gateOutput.Resize({topk, layout.inter});
        gateOutput.Allocate(false);
        const auto *indices = static_cast<const int32_t *>(index.cudaData);
        fastllm::AssertInFastLLM(EnsureCachedExperts(group, cache, tableId, indices, topk),
                               "Pure V4.1 cache refill failed.\n");
        float *gpuOutput = work.device + kMaxTopK * hidden;
        fastllm::AssertInFastLLM(FindExpertCacheBackend(layout.weightType)->compute(
            work.quantizedInput, gateOutput, output, layout, *cache,
            static_cast<const float *>(score.cudaData), topk, gpuOutput),
            "Pure V4.1 cache CUDA experts failed.\n");
        fastllm::cuda::dsv41_cache::Reduce<<<(hidden + 255) / 256, 256, 0, cudaStreamPerThread>>>(
            work.device, gpuOutput, indices, indices,
            (__nv_bfloat16*)output.cudaData, hidden, topk);
        checkCudaErrors("Pure MoE reduction", cudaGetLastError());
        checkCudaErrors("Pure MoE completion", cudaEventRecord(work.done, cudaStreamPerThread));
        work.pending = true;
        ++work.pureCalls;
        work.pureRoutes += topk;
        if (V41CacheOption("FASTLLM_DSV41_MOE_CACHE_TRACE", 0) &&
            (work.pureCalls == 1 || work.pureCalls % 1024 == 0)) {
            const auto counts = snapshot();
            std::fprintf(stderr, "[Fastllm] V4.1 pure expert cache: %llu layers, %llu GPU routes, "
                "%llu hits, %llu misses; %d slots; device %d.\n",
                (unsigned long long)work.pureCalls, (unsigned long long)work.pureRoutes,
                (unsigned long long)(counts[0] - work.pureBaseHits),
                (unsigned long long)(counts[1] - work.pureBaseMisses), cache->slots, cache->device);
        }
        if (launchParallel) {
            launchParallel();
            checkCudaErrors("Pure MoE restore device", cudaSetDevice(cache->device));
        }
        return true;
    }
    work.pureActive = false;
    if (work.pending) {
        fastllm::AssertInFastLLM(cudaEventSynchronize(work.done) == cudaSuccess,
                               "Hybrid MoE completion failed.\n");
        if (work.overlap) work.overlap->Observe();
        if (!work.previousFrequency && work.previousGpu > 0) {
            float copyMs, computeMs;
            checkCudaErrors("Hybrid MoE", cudaEventElapsedTime(&copyMs, work.start, work.copied));
            checkCudaErrors("Hybrid MoE", cudaEventElapsedTime(&computeMs, work.copied, work.computed));
            if (work.previousMisses > 0)
                work.scheduler.refill.Observe(copyMs * 1000 / work.previousMisses);
            else work.scheduler.ensure.Observe(copyMs * 1000);
            work.scheduler.compute[work.previousGpu].Observe(computeMs * 1000);
        }
        if (work.previousPrefetch) {
            float elapsedMs;
            checkCudaErrors("Hybrid MoE prefetch", cudaEventElapsedTime(&elapsedMs, work.prefetchStart, work.prefetchDone));
            work.prefetchCost.Observe(elapsedMs * 1000);
        }
    }
    // Stream GGUF from pinned canonical records or the sole NUMA copy.
    // Scratch allocation failure retains the resident GPU + CPU fallback.
    if (frequency && !work.overlapAttempted && group->sharedLayout.shards == 0) {
        work.overlapAttempted = true;
        size_t maxStride = layout.recordStride;
        for (const auto &l : group->layerLayouts)
            maxStride = std::max(maxStride, l.recordStride);
        auto stream = std::make_unique<DecodeOverlapWorkspace>();
        if (stream->Init(maxStride, group->tableKeys.size(), kMaxTopK, !group->ggufSources.empty()))
            work.overlap = std::move(stream);
        else (void)cudaGetLastError();
    }
    auto *overlap = frequency ? work.overlap.get() : nullptr;
    work.previousPrefetch = false;
    auto *hostIndices = reinterpret_cast<int32_t *>(work.host + (kMaxTopK + 1) * hidden);
    auto *resident = hostIndices + kMaxTopK;
    auto *gpuIndices = resident + kMaxTopK;
    auto *hostScores = reinterpret_cast<float *>(gpuIndices + kMaxTopK);
    auto *deviceMeta = reinterpret_cast<int32_t *>(work.device + 2 * kMaxTopK * hidden);
    auto *deviceGpuIndices = deviceMeta + 2 * kMaxTopK;
    float *cpuOutput = work.host + hidden;
    float *deviceCpuOutput = work.device;
    float *deviceGpuOutput = work.device + kMaxTopK * hidden;
    if (frequency) {
        LookupFrequencyRoutes<<<1, 32, 0, cudaStreamPerThread>>>(
            static_cast<const int32_t *>(index.cudaData), cache->keyToSlot, cache->slotKeys,
            deviceMeta, deviceGpuIndices, cache->routeSlots, cache->hitCount, cache->totalMissCount,
            tableId * layout.experts, layout.experts, topk);
    } else LookupHybridRoutes<<<1, 32, 0, cudaStreamPerThread>>>(
        static_cast<const int32_t *>(index.cudaData), cache->keyToSlot, cache->slotKeys,
        deviceMeta, tableId * layout.experts, layout.experts, topk);
    checkCudaErrors("Hybrid MoE", cudaMemcpyAsync(hostIndices, deviceMeta, 2 * kMaxTopK * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, cudaStreamPerThread));
    checkCudaErrors("Hybrid MoE", cudaMemcpyAsync(work.host, input.cudaData, hidden * input.unitSize,
                    cudaMemcpyDeviceToHost, cudaStreamPerThread));
    if (scoredBfloat) {
        checkCudaErrors("Hybrid MoE scores", cudaMemcpyAsync(hostScores, score.cudaData, topk * sizeof(float),
                        cudaMemcpyDeviceToHost, cudaStreamPerThread));
    }
    fastllm::AssertInFastLLM(cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess,
                           "Hybrid MoE routing copy failed.\n");
    if (scoredBfloat) {
        // Expand backwards in the same pinned allocation.
        const uint16_t *bits = reinterpret_cast<const uint16_t *>(work.host);
        for (int col = hidden - 1; col >= 0; --col) {
            const uint32_t value = uint32_t(bits[col]) << 16;
            memcpy(work.host + col, &value, sizeof(value));
        }
    }
    int hits = 0;
    std::array<int, kMaxTopK> order;
    for (int r = 0; r < topk; ++r) {
        fastllm::AssertInFastLLM(hostIndices[r] >= 0 && hostIndices[r] < layout.experts,
                               "Hybrid MoE received an invalid expert index.\n");
        hits += resident[r] >= 0;
        gpuIndices[r] = -1;
    }
    if (frequency) {
        frequency->Observe(tableId * layout.experts, hostIndices, topk);
    } else if (cache->decode) {
        cache->decode->policy.ObserveRoutes(tableId * layout.experts, hostIndices, topk);
    }
    int next = 0;
    for (int r = 0; r < topk; ++r) if (resident[r] >= 0) order[next++] = r;
    for (int r = 0; r < topk; ++r) if (resident[r] < 0) order[next++] = r;
    const int staged = overlap ? overlap->layers[tableId].SelectMisses(topk - hits, hits) : 0;
    int gpu = frequency ? hits + staged : work.scheduler.SelectGpuCount(topk, hits, group->tableKeys.size());
    if (scoredBfloat && !frequency) {
        // Optional calibration override; unset uses the measured scheduler.
        const int count = V41CacheOption(layout.glm5 ? "FASTLLM_GLM5_MOE_CACHE_GPU_EXPERTS" :
                                         "FASTLLM_DSV41_MOE_CACHE_GPU_EXPERTS", topk + 1);
        if (count <= topk) gpu = count;
    }
    for (int i = 0; i < gpu; ++i) gpuIndices[order[i]] = hostIndices[order[i]];
    if (staged) {
        std::fill_n(overlap->hostSlots, topk, -1);
        for (int i = 0; i < staged; ++i) overlap->hostSlots[order[hits + i]] = i;
    }
    int prefetchRoute = -1;
    if (scoredBfloat && !frequency) {
        const int requested = V41CacheOption(layout.glm5 ? "FASTLLM_GLM5_MOE_CACHE_PREFETCH" :
                                             "FASTLLM_DSV41_MOE_CACHE_PREFETCH", 31);
        if (requested > 0) {
            const int layers = group->tableKeys.size();
            int interval = requested;
            // A fixed stride must not keep visiting only the same layers.
            while (std::gcd(interval, layers) != 1) ++interval;
            if (work.routeHeat.empty()) {
                work.routeHeat.resize(group->totalRecords, 0);
                work.routeLastSeen.resize(group->totalRecords, 0);
            }
            unsigned bestHeat = 1;
            const uint64_t now = work.scheduler.calls + 1;
            for (int r = 0; r < topk; ++r) {
                bool duplicate = false;
                for (int k = 0; k < r; ++k) duplicate |= hostIndices[k] == hostIndices[r];
                if (duplicate) continue;
                const int key = tableId * layout.experts + hostIndices[r];
                unsigned &heat = work.routeHeat[key];
                if (now - work.routeLastSeen[key] > uint64_t(layers) * 64) heat = 0;
                heat = std::min(heat + 1, 65535u);
                work.routeLastSeen[key] = now;
                bool onGpu = false;
                for (int k = 0; k < topk; ++k) onGpu |= gpuIndices[k] == hostIndices[r];
                if (now % interval == 0 && !onGpu && resident[r] < 0 && heat > bestHeat) {
                    prefetchRoute = r;
                    bestHeat = heat;
                }
            }
        }
    }
    output.dataType = input.dataType;
    output.Resize({1, hidden});
    // The model reuses this tensor across layer-partition boundaries. Move
    // its allocation before launching kernels; relabeling the device alone
    // would leave a pointer owned by the previous GPU. No old output is used.
    output.ToDevice(fastllm::DataDevice::CUDA, {cache->device}, false);
    output.Allocate(false);
    if (!frequency || staged) checkCudaErrors("Hybrid MoE", cudaMemcpyAsync(deviceGpuIndices, gpuIndices, topk * sizeof(int32_t),
                    cudaMemcpyHostToDevice, cudaStreamPerThread));
    auto submitGpu = [&]() {
        double stagedDispatchUs = 0;
        if (staged) {
            const double begin = HybridNowUs();
            // The NUMA gate/up workers are already running when this callback
            // executes. DMA starts before resident kernels and does not wait
            // for them. The packed temporary records do not alter residency.
            std::array<int, kMaxTopK> experts;
            for (int i = 0; i < staged; ++i) experts[i] = hostIndices[order[hits + i]];
            overlap->CopyExperts(*group, tableId, experts.data(), staged, topk);
            stagedDispatchUs += HybridNowUs() - begin;
        }
        // The frequency path has already started NUMA workers and staged DMA.
        // Launch the shared expert before waiting for any staged expert.
        // All rejection and host routing reads precede this callback. It may
        // submit a TP graph on another device, so restore our device afterward.
        if (launchParallel) {
            launchParallel();
            checkCudaErrors("Hybrid MoE restore device", cudaSetDevice(cache->device));
        }
        auto &gateOutput = work.gateOutput;
        if ((frequency ? hits : gpu) > 0 || staged > 0) {
            if (layout.deepSeekV41) {
                work.quantizedInput.dataType = input.dataType;
                work.quantizedInput.dataDevice = input.dataDevice;
                work.quantizedInput.dataDeviceIds = input.dataDeviceIds;
                work.quantizedInput.Resize(input.dims);
                fastllm::AssertInFastLLM(FastllmCudaDeepSeekV41QuantizeActivation(input, work.quantizedInput),
                                       "V4.1 cache input quantization failed.\n");
            }
            const double dispatchStart = frequency ? 0 : HybridNowUs();
            gateOutput.dataType = input.dataType;
            gateOutput.dataDevice = input.dataDevice;
            gateOutput.dataDeviceIds = input.dataDeviceIds;
            gateOutput.Resize({topk, layout.inter});
            gateOutput.Allocate(false);
            if (!frequency) {
                checkCudaErrors("Hybrid MoE", cudaEventRecord(work.start, cudaStreamPerThread));
                fastllm::AssertInFastLLM(EnsureCachedExperts(group, cache, tableId, deviceGpuIndices, topk),
                                       "Hybrid MoE refill failed.\n");
                checkCudaErrors("Hybrid MoE", cudaEventRecord(work.copied, cudaStreamPerThread));
            }
            if (!frequency || hits > 0) {
                if (overlap) checkCudaErrors("Decode resident start", cudaEventRecord(overlap->residentStart, cudaStreamPerThread));
                fastllm::AssertInFastLLM(FindExpertCacheBackend(layout.weightType)->compute(
                    layout.deepSeekV41 ? work.quantizedInput : input, gateOutput, output, layout, *cache,
                    static_cast<const float *>(score.cudaData), topk, deviceGpuOutput),
                    "Hybrid MoE CUDA experts failed.\n");
                if (overlap) checkCudaErrors("Decode resident end", cudaEventRecord(overlap->residentDone, cudaStreamPerThread));
            }
            if (!frequency) checkCudaErrors("Hybrid MoE", cudaEventRecord(work.computed, cudaStreamPerThread));
            if (!frequency) work.scheduler.dispatch.Observe(HybridNowUs() - dispatchStart);
        }
        if (prefetchRoute >= 0) {
            int32_t *prefetchIndices = deviceGpuIndices + kMaxTopK;
            checkCudaErrors("Hybrid MoE prefetch", cudaEventRecord(work.prefetchStart, cudaStreamPerThread));
            BuildHybridPrefetchRoutes<<<1, 32, 0, cudaStreamPerThread>>>(
                (const int32_t*)index.cudaData, cache->keyToSlot, cache->slotKeys, prefetchIndices,
                tableId * layout.experts, layout.experts, topk, prefetchRoute);
            fastllm::AssertInFastLLM(EnsureCachedExperts(group, cache, tableId, prefetchIndices, topk),
                                   "Hybrid MoE prefetch failed.\n");
            checkCudaErrors("Hybrid MoE prefetch", cudaEventRecord(work.prefetchDone, cudaStreamPerThread));
            work.previousPrefetch = true;
            ++work.prefetchedExperts;
        }
        if (frequency) {
            auto &policy = *frequency;
            while (true) {
                const auto admission = policy.Select(tableId, tableId * layout.experts, hostIndices, topk);
                if (admission.key < 0) break;
                // Admission only changes future residency. This call uses its
                // original CPU subset or independent staged expert records.
                auto *destination = cache->records + cache->hostSlotOffsets[admission.slot];
                const int expert = admission.key % layout.experts;
                if (group->ggufSources.empty()) {
                    checkCudaErrors("MoE frequency admission", cudaMemcpyAsync(destination,
                        group->hostRecords + group->layerHostOffsets[tableId] + size_t(expert) * layout.recordStride,
                        layout.recordStride, cudaMemcpyHostToDevice, cudaStreamPerThread));
                } else {
                    const auto &source = group->ggufSources[tableId];
                    checkCudaErrors("MoE frequency admission", fastllm_gguf_restore::CopyRecords(source,
                        cache->numaPointers + size_t(tableId) * layout.experts * 2 * source.shards,
                        destination, cudaStreamPerThread, nullptr, nullptr, nullptr, nullptr, expert));
                }
                PublishFrequencyAdmission<<<1, 1, 0, cudaStreamPerThread>>>(
                    cache->keyToSlot, cache->slotKeys, cache->mappedKeyToSlot,
                    cache->lastUsed, cache->step, admission.key, admission.slot);
                checkCudaErrors("MoE frequency admission", cudaGetLastError());
                policy.Admit(tableId, admission);
            }
        }
        if (staged) {
            const double begin = HybridNowUs();
            FastllmCudaMoeGGUFCacheView temporary{
                overlap->records, nullptr, layout.recordStride, layout.downOffset,
                layout.gateGgmlType, layout.downGgmlType, layout.hidden, layout.inter,
                cache->ggufWorkspace, cache->ggufWorkspaceBytes};
            // A resident call, if present, has already prepared this input.
            // All staged calls use the same prefix and disjoint output rows.
            temporary.q8InputPrepared = hits > 0;
            for (int i = 0; i < staged; ++i) {
                const int route = order[hits + i];
                checkCudaErrors("Decode wait expert", cudaStreamWaitEvent(cudaStreamPerThread, overlap->expertReady[i], 0));
                checkCudaErrors("Decode staged start", cudaEventRecord(overlap->stagedStart[i], cudaStreamPerThread));
                // The compute stream reuses activation scratch serially, while
                // the copy stream fills disjoint records for subsequent experts.
                // Write each result in original route order for the final sum.
                temporary.routeSlots = overlap->slots + route;
                fastllm::AssertInFastLLM(ComputeGGUFExperts(
                    input, gateOutput, output, layout, temporary,
                    static_cast<const float *>(score.cudaData) + route, 1, deviceGpuOutput + size_t(route) * hidden),
                    "Staged decode CUDA expert failed.\n");
                temporary.q8InputPrepared = true;
                checkCudaErrors("Decode staged end", cudaEventRecord(overlap->stagedDone[i], cudaStreamPerThread));
            }
            stagedDispatchUs += HybridNowUs() - begin;
            overlap->layers[tableId].dispatch.Observe(stagedDispatchUs);
        }
    };
    if (!frequency) submitGpu();
    const double cpuStart = HybridNowUs();
    if (frequency) {
        fastllm::NumasMoeDecodeExpertsWithOverlap(work.host, cpuOutput, weights,
            hostIndices, gpuIndices, topk, layer, submitGpu,
            scoredBfloat ? hostScores : nullptr, layout.swigluLimit, layout.glm5 ? 128 : 32);
    } else {
        fastllm::NumasMoeDecodeExperts(work.host, cpuOutput, weights,
            hostIndices, gpuIndices, topk, layer,
            scoredBfloat ? hostScores : nullptr, layout.swigluLimit, layout.glm5 ? 128 : 32);
    }
    if (gpu < topk) {
        if (!frequency) work.scheduler.cpu[topk - gpu].Observe(HybridNowUs() - cpuStart);
        if (overlap) overlap->layers[tableId].cpuExpert.Observe((HybridNowUs() - cpuStart) / (topk - gpu));
        checkCudaErrors("Hybrid MoE", cudaMemcpyAsync(deviceCpuOutput, cpuOutput, topk * hidden * sizeof(float),
                        cudaMemcpyHostToDevice, cudaStreamPerThread));
    }
    if (scoredBfloat) {
        fastllm::cuda::dsv41_cache::Reduce<<<(hidden + 255) / 256, 256, 0, cudaStreamPerThread>>>(
            deviceCpuOutput, deviceGpuOutput, deviceGpuIndices, (const int32_t*)index.cudaData,
            (__nv_bfloat16*)output.cudaData, hidden, topk);
    } else ReduceHybridExperts<<<(hidden + 255) / 256, 256, 0, cudaStreamPerThread>>>(
        deviceCpuOutput, deviceGpuOutput, deviceGpuIndices,
        static_cast<const float *>(score.cudaData), static_cast<float *>(output.cudaData), hidden, topk);
    checkCudaErrors("Hybrid MoE reduction", cudaGetLastError());
    checkCudaErrors("Hybrid MoE completion", cudaEventRecord(work.done, cudaStreamPerThread));
    work.pending = true;
    work.previousFrequency = frequency != nullptr;
    if (overlap) {
        overlap->previousLayer = tableId;
        overlap->previousHits = hits;
        overlap->previousMisses = staged;
        overlap->previousStagedRoutes = staged;
        ++overlap->layers[tableId].calls;
    }
    work.previousGpu = gpu;
    work.previousMisses = std::max(0, gpu - hits);
    ++work.scheduler.calls;
    work.cpuRoutes += topk - gpu;
    work.gpuRoutes += gpu;
    work.residentRoutes += std::min(gpu, hits);
    work.allResidentRoutes += hits;
    if (scoredBfloat && V41CacheOption(layout.glm5 ? "FASTLLM_GLM5_MOE_CACHE_TRACE" :
                                                 "FASTLLM_DSV41_MOE_CACHE_TRACE", 0) &&
        (work.scheduler.calls == 1 || work.scheduler.calls % 1024 == 0 ||
         (work.previousPrefetch && work.prefetchedExperts == 1))) {
        std::fprintf(stderr, "[Fastllm] %s expert cache: %llu layers, %llu CPU routes, %llu GPU routes, "
            "%llu resident GPU routes, %d slots; current GPU split %d/%d; device %d; "
            "%llu all resident routes; %llu prefetched experts, %.3f us/prefetch.\n",
            layout.glm5 ? "GLM" : "V4.1",
            (unsigned long long)work.scheduler.calls, (unsigned long long)work.cpuRoutes,
            (unsigned long long)work.gpuRoutes, (unsigned long long)work.residentRoutes, cache->slots, gpu, topk, cache->device,
            (unsigned long long)work.allResidentRoutes, (unsigned long long)work.prefetchedExperts, work.prefetchCost.us);
    }
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
    if (group == nullptr || group->layout.deepSeekV41 || group->layout.glm5) return false;
    const auto &layout = group->LayerLayout(tableId);
    if (input.dims.back() != layout.hidden) return false;
    DeviceCache *cache = GetDeviceCache(*group);
    if (cache == nullptr) {
        return false;
    }
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

extern "C" bool fastllm_moe_cuda_cache_route_stats(int device, uint64_t *values) {
    if (!values || device < 0) return false;
    std::fill_n(values, 8, uint64_t(0));
    int previous = 0;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(device) != cudaSuccess) return false;
    bool ok = cudaDeviceSynchronize() == cudaSuccess;
    std::lock_guard<std::mutex> guard(RegistryMutex());
    for (const auto &group : Groups()) {
        std::lock_guard<std::mutex> lock(group->mutex);
        auto it = group->deviceCaches.find(device);
        if (it == group->deviceCaches.end() || !it->second || !it->second->ready) continue;
        if (it->second->batchHybrid) {
            const auto &batch = *it->second->batchHybrid;
            const auto &rank = *batch.ranks[0];
            const uint64_t routes = batch.cpuRoutes + rank.gpuRoutes;
            values[0] += batch.steps;
            values[1] += routes;
            values[2] += rank.residentRoutes;
            values[3] += routes - rank.residentRoutes;
            values[4] += rank.gpuRoutes;
            values[5] += batch.cpuRoutes;
            values[6] += rank.residentRoutes;
            values[7] += rank.admissions;
        }
        if (!it->second->hybrid) continue;
        const auto &work = *it->second->hybrid;
        const uint64_t routes = work.cpuRoutes + work.gpuRoutes;
        values[0] += work.scheduler.calls;
        values[1] += routes;
        values[2] += work.allResidentRoutes;
        values[3] += routes - work.allResidentRoutes;
        values[4] += work.gpuRoutes;
        values[5] += work.cpuRoutes;
        values[6] += work.residentRoutes;
        values[7] += work.prefetchedExperts;
    }
    ok &= cudaSetDevice(previous) == cudaSuccess;
    return ok;
}

extern "C" bool fastllm_moe_cuda_cache_stats(int device, uint64_t *values, bool reset) {
    if (!values || device < 0) return false;
    std::fill_n(values, 5, uint64_t(0));
    int previous = 0;
    if (cudaGetDevice(&previous) != cudaSuccess || cudaSetDevice(device) != cudaSuccess) return false;
    bool ok = cudaDeviceSynchronize() == cudaSuccess;
    std::lock_guard<std::mutex> guard(RegistryMutex());
    for (const auto &group : Groups()) {
        std::lock_guard<std::mutex> lock(group->mutex);
        auto it = group->deviceCaches.find(device);
        if (it == group->deviceCaches.end() || !it->second || !it->second->ready) continue;
        const auto &cache = *it->second;
        unsigned long long counts[2] = {};
        ok &= cudaMemcpy(counts, cache.hitCount, sizeof(counts[0]), cudaMemcpyDeviceToHost) == cudaSuccess;
        ok &= cudaMemcpy(counts + 1, cache.totalMissCount, sizeof(counts[1]), cudaMemcpyDeviceToHost) == cudaSuccess;
        values[0] += counts[0]; values[1] += counts[1];
        values[2] += cache.recordBytes;
        values[3] += cache.slots; values[4] += group->totalRecords;
        if (reset) {
            ok &= cudaMemset(cache.hitCount, 0, sizeof(counts[0])) == cudaSuccess;
            ok &= cudaMemset(cache.totalMissCount, 0, sizeof(counts[1])) == cudaSuccess;
        }
    }
    ok &= cudaSetDevice(previous) == cudaSuccess;
    return ok;
}
