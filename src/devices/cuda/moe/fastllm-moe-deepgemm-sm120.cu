/*
 * DeepSeek-V4 MXFP4 MoE decode path for NVIDIA SM120.
 *
 * The two grouped FP8xFP4 GEMMs use the same DeepGEMM specializations as the
 * local vLLM build.  Activations are dynamically quantized to E4M3 with
 * packed UE8M0 scales, while the checkpoint's compact FP4 bytes are kept in
 * their native row-major layout.  Other architectures retain FastLLM's
 * Marlin/native CUDA paths; this translation unit is compiled for SM120 only.
 */

#include "fastllm-cuda.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace fastllm_deepgemm_moe {

using namespace deep_gemm;

constexpr int kHidden = 4096;
constexpr int kIntermediate = 2048;
constexpr int kGateColumns = kIntermediate * 2;
constexpr int kLocalExperts = 32;
constexpr int kExpertBlockM = 64;
constexpr int kWorkspaceRows = kLocalExperts * kExpertBlockM;
constexpr int kMaxChunkRows = 8;
constexpr int kMaxTopK = 8;
constexpr int kMaxRoutes = kMaxChunkRows * kMaxTopK;
constexpr int kNumSms = 170;
constexpr int kThreads = 384;
constexpr int kDynamicSharedBytes = 92672;

using DeepGemmKernel = void (*)(
    cutlass::bfloat16_t *, const cutlass::bfloat16_t *, __nv_fp8_e4m3 *,
    __nv_fp8_e4m3 *, int *, cute::TmaDescriptor *, float *, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    cute::TmaDescriptor, cute::TmaDescriptor, cute::TmaDescriptor,
    cute::TmaDescriptor, cute::TmaDescriptor);

static DeepGemmKernel GateKernel() {
    return &sm120_fp8_fp4_gemm_1d1d_impl<
        0, kGateColumns, kHidden, 128, 32, kLocalExperts,
        64, 128, 128, 128, 128, 128, 3, 128, 256, kNumSms,
        GemmType::MGroupedContiguous, false, cutlass::bfloat16_t,
        epilogue::transform::EpilogueIdentity,
        false, true, false, true, false, 64, 1>;
}

static DeepGemmKernel DownKernel() {
    return &sm120_fp8_fp4_gemm_1d1d_impl<
        0, kHidden, kIntermediate, 128, 32, kLocalExperts,
        64, 128, 128, 128, 128, 128, 3, 128, 256, kNumSms,
        GemmType::MGroupedContiguous, false, cutlass::bfloat16_t,
        epilogue::transform::EpilogueIdentity,
        false, true, false, true, false, 64, 1>;
}

static void *AllocateDirect(size_t bytes) {
    return bytes == 0 ? nullptr : FastllmCudaDirectMalloc(bytes);
}

static void ReleaseDirect(void *pointer) {
    if (pointer != nullptr) {
        FastllmCudaDirectFree(pointer);
    }
}

static bool CompactExpertWeightsReleased(
        fastllm::Data **weights, int weightsBatch) {
    if (weights == nullptr || weightsBatch < 4) {
        return false;
    }
    for (int slot = 2; slot < weightsBatch; ++slot) {
        if (weights[slot] != nullptr && weights[slot]->cudaData == nullptr) {
            return true;
        }
    }
    return false;
}

[[noreturn]] static void FailDeepGemmAfterRepack(
        const char *stage, int device = -1) {
    cudaError_t state = cudaPeekAtLastError();
    std::string message =
        "DeepSeek-V4 SM120 DeepGEMM MoE " + std::string(stage) +
        " failed after the compact expert weights were released; "
        "the generic MoE fallback is unavailable";
    if (device >= 0) {
        message += " (cuda:" + std::to_string(device) + ")";
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

struct RuntimeWorkspace {
    int device = -1;
    bool ready = false;
    uint8_t *gateInput = nullptr;
    uint32_t *gateInputScales = nullptr;
    __nv_bfloat16 *gateOutput = nullptr;
    uint8_t *downInput = nullptr;
    uint32_t *downInputScales = nullptr;
    __nv_bfloat16 *downOutput = nullptr;
    int32_t *groupedLayout = nullptr;
    int32_t *routePositions = nullptr;
    std::mutex mutex;
};

static void ReleaseRuntimeWorkspaceStorage(RuntimeWorkspace &workspace) {
    ReleaseDirect(workspace.gateInput);
    ReleaseDirect(workspace.gateInputScales);
    ReleaseDirect(workspace.gateOutput);
    ReleaseDirect(workspace.downInput);
    ReleaseDirect(workspace.downInputScales);
    ReleaseDirect(workspace.downOutput);
    ReleaseDirect(workspace.groupedLayout);
    ReleaseDirect(workspace.routePositions);
    workspace.gateInput = nullptr;
    workspace.gateInputScales = nullptr;
    workspace.gateOutput = nullptr;
    workspace.downInput = nullptr;
    workspace.downInputScales = nullptr;
    workspace.downOutput = nullptr;
    workspace.groupedLayout = nullptr;
    workspace.routePositions = nullptr;
    workspace.device = -1;
    workspace.ready = false;
}

static std::mutex &WorkspaceRegistryMutex() {
    static std::mutex *mutex = new std::mutex();
    return *mutex;
}

static std::map<int, std::shared_ptr<RuntimeWorkspace> > &WorkspaceRegistry() {
    // Intentionally process-lived. CUDA graphs retain these addresses, and
    // avoiding CUDA frees during static teardown is safer than relying on
    // runtime destruction order.
    static auto *registry =
        new std::map<int, std::shared_ptr<RuntimeWorkspace> >();
    return *registry;
}

static bool PrepareRuntimeWorkspace(RuntimeWorkspace &workspace, int device) {
    std::lock_guard<std::mutex> lock(workspace.mutex);
    if (workspace.ready) {
        return workspace.device == device;
    }
    ReleaseRuntimeWorkspaceStorage(workspace);
    workspace.device = device;
    workspace.gateInput = (uint8_t *)AllocateDirect(
        (size_t)kWorkspaceRows * kHidden);
    workspace.gateInputScales = (uint32_t *)AllocateDirect(
        (size_t)kWorkspaceRows * (kHidden / 128 / 4) * sizeof(uint32_t));
    workspace.gateOutput = (__nv_bfloat16 *)AllocateDirect(
        (size_t)kWorkspaceRows * kGateColumns * sizeof(__nv_bfloat16));
    workspace.downInput = (uint8_t *)AllocateDirect(
        (size_t)kWorkspaceRows * kIntermediate);
    workspace.downInputScales = (uint32_t *)AllocateDirect(
        (size_t)kWorkspaceRows * (kIntermediate / 128 / 4) * sizeof(uint32_t));
    workspace.downOutput = (__nv_bfloat16 *)AllocateDirect(
        (size_t)kWorkspaceRows * kHidden * sizeof(__nv_bfloat16));
    workspace.groupedLayout = (int32_t *)AllocateDirect(
        (size_t)kWorkspaceRows * sizeof(int32_t));
    workspace.routePositions = (int32_t *)AllocateDirect(
        (size_t)kMaxRoutes * sizeof(int32_t));
    workspace.ready = workspace.gateInput != nullptr &&
        workspace.gateInputScales != nullptr && workspace.gateOutput != nullptr &&
        workspace.downInput != nullptr && workspace.downInputScales != nullptr &&
        workspace.downOutput != nullptr && workspace.groupedLayout != nullptr &&
        workspace.routePositions != nullptr;
    if (!workspace.ready) {
        ReleaseRuntimeWorkspaceStorage(workspace);
    }
    return workspace.ready;
}

static std::shared_ptr<RuntimeWorkspace> GetRuntimeWorkspace(int device) {
    std::shared_ptr<RuntimeWorkspace> workspace;
    {
        std::lock_guard<std::mutex> lock(WorkspaceRegistryMutex());
        auto &registry = WorkspaceRegistry();
        auto it = registry.find(device);
        if (it == registry.end()) {
            workspace = std::make_shared<RuntimeWorkspace>();
            registry.emplace(device, workspace);
        } else {
            workspace = it->second;
        }
    }
    return PrepareRuntimeWorkspace(*workspace, device) ? workspace : nullptr;
}

static bool MakeTma2d(cute::TmaDescriptor &descriptor,
                      CUtensorMapDataType dtype, void *pointer,
                      uint64_t inner, uint64_t outer,
                      uint64_t outerStrideBytes, uint32_t boxInner,
                      uint32_t boxOuter, CUtensorMapSwizzle swizzle) {
    if (pointer == nullptr) {
        return false;
    }
    const cuuint64_t dims[2] = {inner, outer};
    const cuuint64_t strides[1] = {outerStrideBytes};
    const cuuint32_t box[2] = {boxInner, boxOuter};
    const cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        reinterpret_cast<CUtensorMap *>(&descriptor), dtype, 2, pointer,
        dims, strides, box, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

struct LayerCache {
    bool ready = false;
    std::atomic<bool> retired {false};
    int device = -1;
    uint8_t *gateWeight = nullptr;
    uint32_t *gateWeightScales = nullptr;
    uint8_t *downWeight = nullptr;
    uint32_t *downWeightScales = nullptr;
    std::shared_ptr<RuntimeWorkspace> workspace;
    cute::TmaDescriptor gateA;
    cute::TmaDescriptor gateB;
    cute::TmaDescriptor gateSfa;
    cute::TmaDescriptor gateSfb;
    cute::TmaDescriptor gateD;
    cute::TmaDescriptor downA;
    cute::TmaDescriptor downB;
    cute::TmaDescriptor downSfa;
    cute::TmaDescriptor downSfb;
    cute::TmaDescriptor downD;
    std::mutex buildMutex;

    ~LayerCache();
};

static std::mutex &LayerRegistryMutex() {
    static std::mutex *mutex = new std::mutex();
    return *mutex;
}

static std::map<const fastllm::Data *, std::shared_ptr<LayerCache> > &
LayerRegistry() {
    static auto *registry =
        new std::map<const fastllm::Data *, std::shared_ptr<LayerCache> >();
    return *registry;
}

static thread_local std::map<const fastllm::Data *, std::weak_ptr<LayerCache> >
    layerCacheFront;

static void ReleaseLayerStorage(LayerCache &cache) {
    int originalDevice = -1;
    bool restoreDevice = false;
    if (cache.device >= 0 && cudaGetDevice(&originalDevice) == cudaSuccess &&
        originalDevice != cache.device) {
        restoreDevice = cudaSetDevice(cache.device) == cudaSuccess;
    }
    ReleaseDirect(cache.gateWeight);
    ReleaseDirect(cache.gateWeightScales);
    ReleaseDirect(cache.downWeight);
    ReleaseDirect(cache.downWeightScales);
    cache.gateWeight = nullptr;
    cache.gateWeightScales = nullptr;
    cache.downWeight = nullptr;
    cache.downWeightScales = nullptr;
    cache.ready = false;
    if (restoreDevice) {
        cudaSetDevice(originalDevice);
    }
}

LayerCache::~LayerCache() {
    ReleaseLayerStorage(*this);
}

static bool ValidateWeight(const fastllm::Data *weight, int rows,
                           int columns, bool requireDirectMemory = true) {
    if (weight == nullptr || weight->dataType != fastllm::DataType::NVFP4 ||
        weight->dims.size() != 2 || weight->dims[0] != rows ||
        weight->dims[1] != columns || weight->blockK != 1 ||
        weight->blockM != 32 || !weight->scales.empty() ||
        weight->cudaData == nullptr ||
        (requireDirectMemory && !weight->directMemory)) {
        return false;
    }
    const size_t weightBytes = (size_t)rows * columns / 2;
    const size_t scaleBytes = (size_t)rows * (columns / 32);
    return weight->expansionBytes >= weightBytes + scaleBytes;
}

// Transform [N, K/32] checkpoint scale bytes into DeepGEMM's packed
// [K/128, N] INT32 layout. Four consecutive UE8M0 bytes form one word.
__global__ void TransformWeightScalesKernel(
        const uint8_t *__restrict__ source,
        uint32_t *__restrict__ destination, int n, int scaleGroups) {
    const int packs = scaleGroups / 4;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * packs;
    if (id >= total) {
        return;
    }
    int pack = id / n;
    int row = id - pack * n;
    const uint8_t *src = source + (size_t)row * scaleGroups + pack * 4;
    destination[id] = (uint32_t)src[0] |
                      ((uint32_t)src[1] << 8) |
                      ((uint32_t)src[2] << 16) |
                      ((uint32_t)src[3] << 24);
}

static bool PrepareKernelFunctions() {
    return cudaFuncSetAttribute(
               GateKernel(), cudaFuncAttributeMaxDynamicSharedMemorySize,
               kDynamicSharedBytes) == cudaSuccess &&
           cudaFuncSetAttribute(
               DownKernel(), cudaFuncAttributeMaxDynamicSharedMemorySize,
               kDynamicSharedBytes) == cudaSuccess;
}

static bool BuildTmaDescriptors(LayerCache &cache) {
    RuntimeWorkspace &workspace = *cache.workspace;
    constexpr int gateScalePacks = kHidden / 128 / 4;
    constexpr int downScalePacks = kIntermediate / 128 / 4;
    return
        MakeTma2d(cache.gateA, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                  workspace.gateInput, kHidden, kWorkspaceRows, kHidden,
                  128, 64, CU_TENSOR_MAP_SWIZZLE_128B) &&
        MakeTma2d(cache.gateB, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
                  cache.gateWeight, kHidden,
                  (uint64_t)kGateColumns * kLocalExperts, kHidden / 2,
                  128, 128, CU_TENSOR_MAP_SWIZZLE_128B) &&
        MakeTma2d(cache.gateSfa, CU_TENSOR_MAP_DATA_TYPE_INT32,
                  workspace.gateInputScales, kWorkspaceRows, gateScalePacks,
                  kWorkspaceRows * sizeof(uint32_t), 64, 1,
                  CU_TENSOR_MAP_SWIZZLE_NONE) &&
        MakeTma2d(cache.gateSfb, CU_TENSOR_MAP_DATA_TYPE_INT32,
                  cache.gateWeightScales, kGateColumns,
                  (uint64_t)(kHidden / 32 / 4) * kLocalExperts,
                  kGateColumns * sizeof(uint32_t), 128, 1,
                  CU_TENSOR_MAP_SWIZZLE_NONE) &&
        MakeTma2d(cache.gateD, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                  workspace.gateOutput, kGateColumns, kWorkspaceRows,
                  kGateColumns * sizeof(__nv_bfloat16), 64, 64,
                  CU_TENSOR_MAP_SWIZZLE_128B) &&
        MakeTma2d(cache.downA, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                  workspace.downInput, kIntermediate, kWorkspaceRows,
                  kIntermediate, 128, 64, CU_TENSOR_MAP_SWIZZLE_128B) &&
        MakeTma2d(cache.downB, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
                  cache.downWeight, kIntermediate,
                  (uint64_t)kHidden * kLocalExperts,
                  kIntermediate / 2, 128, 128,
                  CU_TENSOR_MAP_SWIZZLE_128B) &&
        MakeTma2d(cache.downSfa, CU_TENSOR_MAP_DATA_TYPE_INT32,
                  workspace.downInputScales, kWorkspaceRows, downScalePacks,
                  kWorkspaceRows * sizeof(uint32_t), 64, 1,
                  CU_TENSOR_MAP_SWIZZLE_NONE) &&
        MakeTma2d(cache.downSfb, CU_TENSOR_MAP_DATA_TYPE_INT32,
                  cache.downWeightScales, kHidden,
                  (uint64_t)(kIntermediate / 32 / 4) * kLocalExperts,
                  kHidden * sizeof(uint32_t),
                  128, 1, CU_TENSOR_MAP_SWIZZLE_NONE) &&
        MakeTma2d(cache.downD, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                  workspace.downOutput, kHidden, kWorkspaceRows,
                  kHidden * sizeof(__nv_bfloat16), 64, 64,
                  CU_TENSOR_MAP_SWIZZLE_128B);
}

static bool BuildLayerCache(fastllm::Data **weights, int weightsBatch,
                            LayerCache &cache) {
    if (weights == nullptr || weightsBatch != 2 + kLocalExperts * 2) {
        return false;
    }
    for (int expert = 0; expert < kLocalExperts; ++expert) {
        const int slot = 2 + expert * 2;
        if (!ValidateWeight(weights[slot], kGateColumns, kHidden) ||
            !ValidateWeight(weights[slot + 1], kHidden, kIntermediate)) {
            return false;
        }
    }

    int device = -1;
    int major = 0;
    int minor = 0;
    int sms = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                               device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                               device) != cudaSuccess ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                               device) != cudaSuccess ||
        major != 12 || minor != 0 || sms != kNumSms ||
        FastllmCudaGraphIsCapturing() ||
        !PrepareKernelFunctions()) {
        cudaGetLastError();
        return false;
    }

    cache.device = device;
    cache.workspace = GetRuntimeWorkspace(device);
    if (cache.workspace == nullptr) {
        return false;
    }

    constexpr size_t gateWeightBytes =
        (size_t)kGateColumns * kHidden / 2;
    constexpr size_t gateScaleBytes =
        (size_t)kGateColumns * (kHidden / 32);
    constexpr size_t downWeightBytes =
        (size_t)kHidden * kIntermediate / 2;
    constexpr size_t downScaleBytes =
        (size_t)kHidden * (kIntermediate / 32);
    cache.gateWeight = (uint8_t *)AllocateDirect(
        gateWeightBytes * kLocalExperts);
    cache.gateWeightScales = (uint32_t *)AllocateDirect(
        gateScaleBytes * kLocalExperts);
    cache.downWeight = (uint8_t *)AllocateDirect(
        downWeightBytes * kLocalExperts);
    cache.downWeightScales = (uint32_t *)AllocateDirect(
        downScaleBytes * kLocalExperts);
    if (cache.gateWeight == nullptr || cache.gateWeightScales == nullptr ||
        cache.downWeight == nullptr || cache.downWeightScales == nullptr) {
        ReleaseLayerStorage(cache);
        return false;
    }

    constexpr int threads = 256;
    cudaStream_t stream = cudaStreamPerThread;
    bool success = true;
    for (int expert = 0; expert < kLocalExperts && success; ++expert) {
        const int slot = 2 + expert * 2;
        fastllm::Data *gate = weights[slot];
        fastllm::Data *down = weights[slot + 1];
        success = cudaMemcpyAsync(
            cache.gateWeight + gateWeightBytes * expert, gate->cudaData,
            gateWeightBytes, cudaMemcpyDeviceToDevice, stream) == cudaSuccess;
        if (!success) {
            break;
        }
        const int gatePacks = kGateColumns * (kHidden / 32 / 4);
        TransformWeightScalesKernel<<<
            (gatePacks + threads - 1) / threads, threads, 0, stream>>>(
            (const uint8_t *)gate->cudaData + gateWeightBytes,
            cache.gateWeightScales +
                (size_t)expert * gateScaleBytes / sizeof(uint32_t),
            kGateColumns, kHidden / 32);
        success = cudaGetLastError() == cudaSuccess &&
            cudaMemcpyAsync(
                cache.downWeight + downWeightBytes * expert, down->cudaData,
                downWeightBytes, cudaMemcpyDeviceToDevice, stream) ==
                cudaSuccess;
        if (!success) {
            break;
        }
        const int downPacks = kHidden * (kIntermediate / 32 / 4);
        TransformWeightScalesKernel<<<
            (downPacks + threads - 1) / threads, threads, 0, stream>>>(
            (const uint8_t *)down->cudaData + downWeightBytes,
            cache.downWeightScales +
                (size_t)expert * downScaleBytes / sizeof(uint32_t),
            kHidden, kIntermediate / 32);
        success = cudaGetLastError() == cudaSuccess;
    }
    if (success) {
        success = cudaStreamSynchronize(stream) == cudaSuccess &&
                  BuildTmaDescriptors(cache);
    }
    if (!success) {
        ReleaseLayerStorage(cache);
        return false;
    }

    // The native DeepGEMM layout has the same footprint as the compact
    // checkpoint data. Once every copy and TMA descriptor is valid, release
    // the direct source allocations so model memory does not double.
    for (int expert = 0; expert < kLocalExperts; ++expert) {
        const int slot = 2 + expert * 2;
        for (int offset = 0; offset < 2; ++offset) {
            fastllm::Data *weight = weights[slot + offset];
            void *old = weight->cudaData;
            weight->cudaData = nullptr;
            weight->cudaDataBorrowed = false;
            FastllmCudaDirectFree(old);
        }
    }

    cache.ready = true;
    return true;
}

static std::shared_ptr<LayerCache> GetOrBuildLayerCache(
        fastllm::Data **weights, int weightsBatch) {
    if (weights == nullptr || weightsBatch < 4 || weights[2] == nullptr) {
        return nullptr;
    }
    const fastllm::Data *key = weights[2];
    auto front = layerCacheFront.find(key);
    if (front != layerCacheFront.end()) {
        std::shared_ptr<LayerCache> cached = front->second.lock();
        if (cached != nullptr && cached->ready &&
            !cached->retired.load(std::memory_order_acquire)) {
            return cached;
        }
        layerCacheFront.erase(front);
    }

    std::shared_ptr<LayerCache> cache;
    {
        std::lock_guard<std::mutex> lock(LayerRegistryMutex());
        auto &registry = LayerRegistry();
        auto it = registry.find(key);
        if (it == registry.end()) {
            cache = std::make_shared<LayerCache>();
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
            failed = !BuildLayerCache(weights, weightsBatch, *cache);
            if (failed) {
                cache->retired.store(true, std::memory_order_release);
            }
        }
    }
    if (failed) {
        std::lock_guard<std::mutex> lock(LayerRegistryMutex());
        auto &registry = LayerRegistry();
        auto it = registry.find(key);
        if (it != registry.end() && it->second == cache) {
            registry.erase(it);
        }
        return nullptr;
    }
    layerCacheFront[key] = cache;
    return cache;
}

static void ReleaseLayerCache(const fastllm::Data *key) {
    if (key == nullptr) {
        return;
    }
    std::shared_ptr<LayerCache> retired;
    {
        std::lock_guard<std::mutex> lock(LayerRegistryMutex());
        auto &registry = LayerRegistry();
        auto it = registry.find(key);
        if (it == registry.end()) {
            return;
        }
        it->second->retired.store(true, std::memory_order_release);
        retired = std::move(it->second);
        registry.erase(it);
    }
    layerCacheFront.erase(key);
}

__global__ void BuildRouteMetadataKernel(
        const int32_t *__restrict__ globalIndices,
        int32_t *__restrict__ groupedLayout,
        int32_t *__restrict__ routePositions,
        int rows, int topk, int ownerRank, int ownerCount) {
    if (blockIdx.x != 0) {
        return;
    }
    const int thread = threadIdx.x;
    if (thread < kLocalExperts) {
        groupedLayout[thread * kExpertBlockM] = -1;
    }
    __syncthreads();

    const int routes = rows * topk;
    if (thread >= routes) {
        return;
    }
    routePositions[thread] = -1;
    int expert = globalIndices[thread];
    if (expert < 0 || ownerCount <= 0 || ownerRank < 0 ||
        expert % ownerCount != ownerRank) {
        return;
    }
    int localExpert = expert / ownerCount;
    if ((unsigned int)localExpert >= (unsigned int)kLocalExperts) {
        return;
    }

    // Preserve the serial implementation's route order exactly.  An atomic
    // counter is a little cheaper in source code, but makes the route-to-row
    // assignment scheduling-dependent and complicates logits-level audits.
    // DSpark has at most 64 routes here, so each thread can cheaply count the
    // matching valid routes that precede it.
    int localOffset = 0;
    for (int route = 0; route < thread; ++route) {
        int previousExpert = globalIndices[route];
        if (previousExpert >= 0 &&
            previousExpert % ownerCount == ownerRank &&
            previousExpert / ownerCount == localExpert) {
            ++localOffset;
        }
    }
    if (localOffset >= kExpertBlockM) {
        return;
    }
    routePositions[thread] =
        localExpert * kExpertBlockM + localOffset;
    if (localOffset == 0) {
        groupedLayout[localExpert * kExpertBlockM] = localExpert;
    }
}

__device__ __forceinline__ uint8_t Ue8m0ScaleByte(float maxAbs,
                                                   float &inverseScale) {
    // Match the DeepGEMM/vLLM 1x128 quantizer exactly: the epsilon is a lower
    // bound on the scale, rather than on the unscaled absolute maximum.
    float scaleRaw = fmaxf(maxAbs * (1.0f / 448.0f), 1.0e-10f);
    float exponent = ceilf(log2f(scaleRaw));
    float scale = exp2f(exponent);
    inverseScale = 1.0f / scale;
    int encoded = (int)exponent + 127;
    return (uint8_t)max(0, min(255, encoded));
}

template <bool FusedSilu, int WorkspaceRows = kWorkspaceRows>
__global__ void QuantizeRoutedRowsKernel(
        const __nv_bfloat16 *__restrict__ input,
        const int32_t *__restrict__ routePositions,
        uint8_t *__restrict__ output,
        uint32_t *__restrict__ packedScales,
        int routes, int topk, int inputColumns, int outputColumns,
        float swigluLimit) {
    int route = blockIdx.x;
    if (route >= routes) {
        return;
    }
    int destinationRow = routePositions[route];
    if (destinationRow < 0) {
        return;
    }
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    constexpr int warps = 8;
    const int groups = outputColumns / 128;
    const int sourceRow = FusedSilu ? destinationRow : route / topk;
    for (int group = warp; group < groups; group += warps) {
        float values[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int column = group * 128 + lane * 4 + i;
            if constexpr (FusedSilu) {
                const __nv_bfloat16 *row =
                    input + (size_t)sourceRow * inputColumns;
                float gate = __bfloat162float(row[column]);
                float up = __bfloat162float(row[outputColumns + column]);
                if (swigluLimit > 0.0f) {
                    gate = fminf(gate, swigluLimit);
                    up = fminf(swigluLimit, fmaxf(-swigluLimit, up));
                }
                float value = (gate / (1.0f + __expf(-gate))) * up;
                values[i] = __bfloat162float(__float2bfloat16_rn(value));
            } else {
                values[i] = __bfloat162float(
                    input[(size_t)sourceRow * inputColumns + column]);
            }
        }
        float maxAbs = fmaxf(fmaxf(fabsf(values[0]), fabsf(values[1])),
                             fmaxf(fabsf(values[2]), fabsf(values[3])));
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            maxAbs = fmaxf(maxAbs,
                __shfl_down_sync(0xffffffff, maxAbs, offset));
        }
        maxAbs = __shfl_sync(0xffffffff, maxAbs, 0);
        float inverseScale = 0.0f;
        uint8_t scaleByte = Ue8m0ScaleByte(maxAbs, inverseScale);
        if (lane == 0) {
            int pack = group >> 2;
            int byte = group & 3;
            uint8_t *scaleBytes = reinterpret_cast<uint8_t *>(packedScales);
            scaleBytes[((size_t)pack * WorkspaceRows + destinationRow) * 4 +
                       byte] = scaleByte;
        }
        uint32_t packed = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float q = fminf(448.0f,
                            fmaxf(-448.0f, values[i] * inverseScale));
            uint8_t value = __nv_cvt_float_to_fp8(
                q, __NV_SATFINITE, __NV_E4M3);
            packed |= (uint32_t)value << (i * 8);
        }
        reinterpret_cast<uint32_t *>(
            output + (size_t)destinationRow * outputColumns)[lane + group * 32] =
                packed;
    }
}

__global__ void ReduceRoutedRowsKernel(
        const __nv_bfloat16 *__restrict__ expertRows,
        const int32_t *__restrict__ routePositions,
        const float *__restrict__ scores,
        __nv_bfloat16 *__restrict__ output,
        int rows, int topk) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * kHidden;
    if (id >= total) {
        return;
    }
    int row = id / kHidden;
    int column = id - row * kHidden;
    float value = 0.0f;
    for (int slot = 0; slot < topk; ++slot) {
        int route = row * topk + slot;
        int position = routePositions[route];
        if (position >= 0) {
            value += __bfloat162float(
                         expertRows[(size_t)position * kHidden + column]) *
                     scores[route];
        }
    }
    output[(size_t)row * kHidden + column] = __float2bfloat16_rn(value);
}

static bool LaunchDeepGemm(DeepGemmKernel kernel,
                           __nv_bfloat16 *output, uint8_t *input,
                           uint8_t *weight, int32_t *layout,
                           int shapeK,
                           const cute::TmaDescriptor &tmaA,
                           const cute::TmaDescriptor &tmaB,
                           const cute::TmaDescriptor &tmaSfa,
                           const cute::TmaDescriptor &tmaSfb,
                           const cute::TmaDescriptor &tmaD,
                           cudaStream_t stream) {
    kernel<<<kNumSms, kThreads, kDynamicSharedBytes, stream>>>(
        reinterpret_cast<cutlass::bfloat16_t *>(output), nullptr,
        reinterpret_cast<__nv_fp8_e4m3 *>(input),
        reinterpret_cast<__nv_fp8_e4m3 *>(weight), layout, nullptr, nullptr,
        kWorkspaceRows, kHidden, shapeK, kHidden, 0,
        kWorkspaceRows * kHidden,
        tmaA, tmaB, tmaSfa, tmaSfb, tmaD);
    return cudaGetLastError() == cudaSuccess;
}

static bool RunDeepGemmMoe(
        const fastllm::Data &input, fastllm::Data &output,
        fastllm::Data **weights, int weightsBatch,
        const int32_t *globalIndices, const float *scores,
        int topk, int ownerRank, int ownerCount, float swigluLimit) {
    if (input.dataType != fastllm::DataType::BFLOAT16 ||
        input.dataDevice != fastllm::DataDevice::CUDA ||
        input.dims.empty() || input.dims.back() != kHidden ||
        globalIndices == nullptr || scores == nullptr ||
        topk <= 0 || topk > kMaxTopK || ownerRank < 0 || ownerCount <= 0) {
        if (CompactExpertWeightsReleased(weights, weightsBatch)) {
            FailDeepGemmAfterRepack(
                "expert-parallel invocation validation");
        }
        return false;
    }
    int rows = (int)(input.Count(0) / kHidden);
    if (rows <= 0 || (size_t)rows * kHidden != input.Count(0)) {
        if (CompactExpertWeightsReleased(weights, weightsBatch)) {
            FailDeepGemmAfterRepack(
                "expert-parallel input-shape validation");
        }
        return false;
    }
    std::shared_ptr<LayerCache> cache =
        GetOrBuildLayerCache(weights, weightsBatch);
    if (cache == nullptr || !cache->ready || cache->workspace == nullptr) {
        if (CompactExpertWeightsReleased(weights, weightsBatch)) {
            FailDeepGemmAfterRepack(
                "expert-parallel cache recovery");
        }
        return false;
    }

    output.dataDevice = input.dataDevice;
    output.dataDeviceIds = input.dataDeviceIds;
    output.dataType = input.dataType;
    output.Resize(input.dims);
    output.Allocate(false);
    if (output.cudaData == nullptr) {
        FailDeepGemmAfterRepack(
            "expert-parallel output allocation", cache->device);
    }

    RuntimeWorkspace &workspace = *cache->workspace;
    cudaStream_t stream = cudaStreamPerThread;
    constexpr int quantThreads = 256;
    constexpr int reduceThreads = 256;
    for (int rowBase = 0; rowBase < rows; rowBase += kMaxChunkRows) {
        int chunkRows = std::min(kMaxChunkRows, rows - rowBase);
        int routes = chunkRows * topk;
        const int32_t *chunkIndices =
            globalIndices + (size_t)rowBase * topk;
        const float *chunkScores = scores + (size_t)rowBase * topk;
        const __nv_bfloat16 *chunkInput =
            (const __nv_bfloat16 *)input.cudaData + (size_t)rowBase * kHidden;
        __nv_bfloat16 *chunkOutput =
            (__nv_bfloat16 *)output.cudaData + (size_t)rowBase * kHidden;

        BuildRouteMetadataKernel<<<1, kMaxRoutes, 0, stream>>>(
            chunkIndices, workspace.groupedLayout,
            workspace.routePositions, chunkRows, topk,
            ownerRank, ownerCount);
        QuantizeRoutedRowsKernel<false><<<routes, quantThreads, 0, stream>>>(
            chunkInput, workspace.routePositions, workspace.gateInput,
            workspace.gateInputScales, routes, topk, kHidden, kHidden,
            swigluLimit);
        if (cudaGetLastError() != cudaSuccess ||
            !LaunchDeepGemm(
                GateKernel(), workspace.gateOutput, workspace.gateInput,
                cache->gateWeight, workspace.groupedLayout, kHidden,
                cache->gateA, cache->gateB, cache->gateSfa,
                cache->gateSfb, cache->gateD, stream)) {
            FailDeepGemmAfterRepack(
                "expert-parallel gate launch", cache->device);
        }
        QuantizeRoutedRowsKernel<true><<<routes, quantThreads, 0, stream>>>(
            workspace.gateOutput, workspace.routePositions,
            workspace.downInput, workspace.downInputScales, routes, topk,
            kGateColumns, kIntermediate, swigluLimit);
        if (cudaGetLastError() != cudaSuccess ||
            !LaunchDeepGemm(
                DownKernel(), workspace.downOutput, workspace.downInput,
                cache->downWeight, workspace.groupedLayout, kIntermediate,
                cache->downA, cache->downB, cache->downSfa,
                cache->downSfb, cache->downD, stream)) {
            FailDeepGemmAfterRepack(
                "expert-parallel down launch", cache->device);
        }
        int outputCount = chunkRows * kHidden;
        ReduceRoutedRowsKernel<<<
            (outputCount + reduceThreads - 1) / reduceThreads,
            reduceThreads, 0, stream>>>(
            workspace.downOutput, workspace.routePositions, chunkScores,
            chunkOutput, chunkRows, topk);
        if (cudaGetLastError() != cudaSuccess) {
            FailDeepGemmAfterRepack(
                "expert-parallel output reduction", cache->device);
        }
    }
    return true;
}

// The no-EP DeepSeek-V4 layout shards every routed expert's intermediate
// dimension across all TP ranks.  All ranks therefore execute the same route
// list and contribute a 256-wide partial result before the existing allreduce.
// This is the layout used by vLLM's DeepGemmFP4Experts path and, importantly,
// it removes the per-layer straggler caused by assigning complete experts to
// ranks for a single-token request.
namespace tensor_parallel {

constexpr int kTpIntermediate = 256;
constexpr int kTpGateColumns = kTpIntermediate * 2;
constexpr int kTpExperts = 256;
constexpr int kTpExpertBlockM = 64;
constexpr int kTpWorkspaceRows = kMaxRoutes * kTpExpertBlockM;

static DeepGemmKernel GateKernel() {
    // Exact SM120 specialization selected by the local vLLM DeepGEMM cache
    // for M-grouped [M, 4096] x [256, 512, 4096].
    return &sm120_fp8_fp4_gemm_1d1d_impl<
        0, kTpGateColumns, kHidden, 128, 32, kTpExperts,
        64, 128, 128, 128, 128, 128, 3, 128, 256, kNumSms,
        GemmType::MGroupedContiguous, false, cutlass::bfloat16_t,
        epilogue::transform::EpilogueIdentity,
        false, true, false, true, false, 64, 1>;
}

static DeepGemmKernel DownKernel() {
    // Exact SM120 specialization selected by vLLM for
    // M-grouped [M, 256] x [256, 4096, 256].
    return &sm120_fp8_fp4_gemm_1d1d_impl<
        0, kHidden, kTpIntermediate, 128, 32, kTpExperts,
        64, 128, 128, 128, 128, 128, 3, 128, 256, kNumSms,
        GemmType::MGroupedContiguous, false, cutlass::bfloat16_t,
        epilogue::transform::EpilogueIdentity,
        false, true, false, true, false, 64, 1>;
}

struct RuntimeWorkspace {
    int device = -1;
    bool ready = false;
    uint8_t *gateInput = nullptr;
    uint32_t *gateInputScales = nullptr;
    __nv_bfloat16 *gateOutput = nullptr;
    uint8_t *downInput = nullptr;
    uint32_t *downInputScales = nullptr;
    __nv_bfloat16 *downOutput = nullptr;
    int32_t *groupedLayout = nullptr;
    int32_t *routePositions = nullptr;
    std::mutex mutex;
};

static void ReleaseRuntimeWorkspaceStorage(RuntimeWorkspace &workspace) {
    ReleaseDirect(workspace.gateInput);
    ReleaseDirect(workspace.gateInputScales);
    ReleaseDirect(workspace.gateOutput);
    ReleaseDirect(workspace.downInput);
    ReleaseDirect(workspace.downInputScales);
    ReleaseDirect(workspace.downOutput);
    ReleaseDirect(workspace.groupedLayout);
    ReleaseDirect(workspace.routePositions);
    workspace.gateInput = nullptr;
    workspace.gateInputScales = nullptr;
    workspace.gateOutput = nullptr;
    workspace.downInput = nullptr;
    workspace.downInputScales = nullptr;
    workspace.downOutput = nullptr;
    workspace.groupedLayout = nullptr;
    workspace.routePositions = nullptr;
    workspace.device = -1;
    workspace.ready = false;
}

static std::mutex &WorkspaceRegistryMutex() {
    static std::mutex *mutex = new std::mutex();
    return *mutex;
}

static std::map<int, std::shared_ptr<RuntimeWorkspace> > &WorkspaceRegistry() {
    static auto *registry =
        new std::map<int, std::shared_ptr<RuntimeWorkspace> >();
    return *registry;
}

static bool PrepareRuntimeWorkspace(RuntimeWorkspace &workspace, int device) {
    std::lock_guard<std::mutex> lock(workspace.mutex);
    if (workspace.ready) {
        return workspace.device == device;
    }
    ReleaseRuntimeWorkspaceStorage(workspace);
    constexpr int gateScalePacks = (kHidden / 128 + 3) / 4;
    constexpr int downScalePacks = (kTpIntermediate / 128 + 3) / 4;
    workspace.device = device;
    workspace.gateInput = (uint8_t *)AllocateDirect(
        (size_t)kTpWorkspaceRows * kHidden);
    workspace.gateInputScales = (uint32_t *)AllocateDirect(
        (size_t)kTpWorkspaceRows * gateScalePacks * sizeof(uint32_t));
    workspace.gateOutput = (__nv_bfloat16 *)AllocateDirect(
        (size_t)kTpWorkspaceRows * kTpGateColumns * sizeof(__nv_bfloat16));
    workspace.downInput = (uint8_t *)AllocateDirect(
        (size_t)kTpWorkspaceRows * kTpIntermediate);
    workspace.downInputScales = (uint32_t *)AllocateDirect(
        (size_t)kTpWorkspaceRows * downScalePacks * sizeof(uint32_t));
    workspace.downOutput = (__nv_bfloat16 *)AllocateDirect(
        (size_t)kTpWorkspaceRows * kHidden * sizeof(__nv_bfloat16));
    workspace.groupedLayout = (int32_t *)AllocateDirect(
        (size_t)kTpWorkspaceRows * sizeof(int32_t));
    workspace.routePositions = (int32_t *)AllocateDirect(
        (size_t)kMaxRoutes * sizeof(int32_t));
    workspace.ready = workspace.gateInput != nullptr &&
        workspace.gateInputScales != nullptr && workspace.gateOutput != nullptr &&
        workspace.downInput != nullptr && workspace.downInputScales != nullptr &&
        workspace.downOutput != nullptr && workspace.groupedLayout != nullptr &&
        workspace.routePositions != nullptr;
    if (!workspace.ready) {
        ReleaseRuntimeWorkspaceStorage(workspace);
    }
    return workspace.ready;
}

static std::shared_ptr<RuntimeWorkspace> GetRuntimeWorkspace(int device) {
    std::shared_ptr<RuntimeWorkspace> workspace;
    {
        std::lock_guard<std::mutex> lock(WorkspaceRegistryMutex());
        auto &registry = WorkspaceRegistry();
        auto it = registry.find(device);
        if (it == registry.end()) {
            workspace = std::make_shared<RuntimeWorkspace>();
            registry.emplace(device, workspace);
        } else {
            workspace = it->second;
        }
    }
    return PrepareRuntimeWorkspace(*workspace, device) ? workspace : nullptr;
}

struct LayerCache {
    bool ready = false;
    std::atomic<bool> retired {false};
    int device = -1;
    uint8_t *gateWeight = nullptr;
    uint32_t *gateWeightScales = nullptr;
    uint8_t *downWeight = nullptr;
    uint32_t *downWeightScales = nullptr;
    std::shared_ptr<RuntimeWorkspace> workspace;
    cute::TmaDescriptor gateA;
    cute::TmaDescriptor gateB;
    cute::TmaDescriptor gateSfa;
    cute::TmaDescriptor gateSfb;
    cute::TmaDescriptor gateD;
    cute::TmaDescriptor downA;
    cute::TmaDescriptor downB;
    cute::TmaDescriptor downSfa;
    cute::TmaDescriptor downSfb;
    cute::TmaDescriptor downD;
    std::mutex buildMutex;

    ~LayerCache();
};

static std::mutex &LayerRegistryMutex() {
    static std::mutex *mutex = new std::mutex();
    return *mutex;
}

static std::map<const fastllm::Data *, std::shared_ptr<LayerCache> > &
LayerRegistry() {
    static auto *registry =
        new std::map<const fastllm::Data *, std::shared_ptr<LayerCache> >();
    return *registry;
}

static thread_local std::map<const fastllm::Data *, std::weak_ptr<LayerCache> >
    layerCacheFront;

static void ReleaseLayerStorage(LayerCache &cache) {
    int originalDevice = -1;
    bool restoreDevice = false;
    if (cache.device >= 0 && cudaGetDevice(&originalDevice) == cudaSuccess &&
        originalDevice != cache.device) {
        restoreDevice = cudaSetDevice(cache.device) == cudaSuccess;
    }
    ReleaseDirect(cache.gateWeight);
    ReleaseDirect(cache.gateWeightScales);
    ReleaseDirect(cache.downWeight);
    ReleaseDirect(cache.downWeightScales);
    cache.gateWeight = nullptr;
    cache.gateWeightScales = nullptr;
    cache.downWeight = nullptr;
    cache.downWeightScales = nullptr;
    cache.ready = false;
    if (restoreDevice) {
        cudaSetDevice(originalDevice);
    }
}

LayerCache::~LayerCache() {
    ReleaseLayerStorage(*this);
}

static bool PrepareKernelFunctions() {
    return cudaFuncSetAttribute(
               GateKernel(), cudaFuncAttributeMaxDynamicSharedMemorySize,
               kDynamicSharedBytes) == cudaSuccess &&
           cudaFuncSetAttribute(
               DownKernel(), cudaFuncAttributeMaxDynamicSharedMemorySize,
               kDynamicSharedBytes) == cudaSuccess;
}

static bool BuildTmaDescriptors(LayerCache &cache) {
    RuntimeWorkspace &workspace = *cache.workspace;
    constexpr int gateScalePacks = (kHidden / 128 + 3) / 4;
    constexpr int downScalePacks = (kTpIntermediate / 128 + 3) / 4;
    return
        MakeTma2d(cache.gateA, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                  workspace.gateInput, kHidden, kTpWorkspaceRows, kHidden,
                  128, 64, CU_TENSOR_MAP_SWIZZLE_128B) &&
        MakeTma2d(cache.gateB, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
                  cache.gateWeight, kHidden,
                  (uint64_t)kTpGateColumns * kTpExperts, kHidden / 2,
                  128, 128, CU_TENSOR_MAP_SWIZZLE_128B) &&
        MakeTma2d(cache.gateSfa, CU_TENSOR_MAP_DATA_TYPE_INT32,
                  workspace.gateInputScales, kTpWorkspaceRows, gateScalePacks,
                  kTpWorkspaceRows * sizeof(uint32_t), 64, 1,
                  CU_TENSOR_MAP_SWIZZLE_NONE) &&
        MakeTma2d(cache.gateSfb, CU_TENSOR_MAP_DATA_TYPE_INT32,
                  cache.gateWeightScales, kTpGateColumns,
                  (uint64_t)(kHidden / 32 / 4) * kTpExperts,
                  kTpGateColumns * sizeof(uint32_t), 128, 1,
                  CU_TENSOR_MAP_SWIZZLE_NONE) &&
        MakeTma2d(cache.gateD, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                  workspace.gateOutput, kTpGateColumns, kTpWorkspaceRows,
                  kTpGateColumns * sizeof(__nv_bfloat16), 64, 64,
                  CU_TENSOR_MAP_SWIZZLE_128B) &&
        MakeTma2d(cache.downA, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                  workspace.downInput, kTpIntermediate, kTpWorkspaceRows,
                  kTpIntermediate, 128, 64, CU_TENSOR_MAP_SWIZZLE_128B) &&
        MakeTma2d(cache.downB, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
                  cache.downWeight, kTpIntermediate,
                  (uint64_t)kHidden * kTpExperts, kTpIntermediate / 2,
                  128, 128, CU_TENSOR_MAP_SWIZZLE_128B) &&
        MakeTma2d(cache.downSfa, CU_TENSOR_MAP_DATA_TYPE_INT32,
                  workspace.downInputScales, kTpWorkspaceRows, downScalePacks,
                  kTpWorkspaceRows * sizeof(uint32_t), 64, 1,
                  CU_TENSOR_MAP_SWIZZLE_NONE) &&
        MakeTma2d(cache.downSfb, CU_TENSOR_MAP_DATA_TYPE_INT32,
                  cache.downWeightScales, kHidden,
                  (uint64_t)(kTpIntermediate / 32 / 4) * kTpExperts,
                  kHidden * sizeof(uint32_t), 128, 1,
                  CU_TENSOR_MAP_SWIZZLE_NONE) &&
        MakeTma2d(cache.downD, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                  workspace.downOutput, kHidden, kTpWorkspaceRows,
                  kHidden * sizeof(__nv_bfloat16), 64, 64,
                  CU_TENSOR_MAP_SWIZZLE_128B);
}

static bool BuildLayerCache(fastllm::Data **weights, int weightsBatch,
                            LayerCache &cache) {
    if (weights == nullptr || weightsBatch != 2 + kTpExperts * 2) {
        return false;
    }
    for (int expert = 0; expert < kTpExperts; ++expert) {
        const int slot = 2 + expert * 2;
        if (!ValidateWeight(weights[slot], kTpGateColumns, kHidden, false) ||
            !ValidateWeight(weights[slot + 1], kHidden, kTpIntermediate,
                            false)) {
            return false;
        }
    }

    int device = -1;
    int major = 0;
    int minor = 0;
    int sms = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device < 0 ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                               device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                               device) != cudaSuccess ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                               device) != cudaSuccess ||
        major != 12 || minor != 0 || sms != kNumSms ||
        FastllmCudaGraphIsCapturing() ||
        !PrepareKernelFunctions()) {
        cudaGetLastError();
        return false;
    }

    cache.device = device;
    cache.workspace = GetRuntimeWorkspace(device);
    if (cache.workspace == nullptr) {
        return false;
    }

    constexpr size_t gateWeightBytes =
        (size_t)kTpGateColumns * kHidden / 2;
    constexpr size_t gateScaleBytes =
        (size_t)kTpGateColumns * (kHidden / 32);
    constexpr size_t downWeightBytes =
        (size_t)kHidden * kTpIntermediate / 2;
    constexpr size_t downScaleBytes =
        (size_t)kHidden * (kTpIntermediate / 32);
    cache.gateWeight = (uint8_t *)AllocateDirect(
        gateWeightBytes * kTpExperts);
    cache.gateWeightScales = (uint32_t *)AllocateDirect(
        gateScaleBytes * kTpExperts);
    cache.downWeight = (uint8_t *)AllocateDirect(
        downWeightBytes * kTpExperts);
    cache.downWeightScales = (uint32_t *)AllocateDirect(
        downScaleBytes * kTpExperts);
    if (cache.gateWeight == nullptr || cache.gateWeightScales == nullptr ||
        cache.downWeight == nullptr || cache.downWeightScales == nullptr) {
        ReleaseLayerStorage(cache);
        return false;
    }

    constexpr int threads = 256;
    cudaStream_t stream = cudaStreamPerThread;
    bool success = true;
    for (int expert = 0; expert < kTpExperts && success; ++expert) {
        const int slot = 2 + expert * 2;
        fastllm::Data *gate = weights[slot];
        fastllm::Data *down = weights[slot + 1];
        success = cudaMemcpyAsync(
            cache.gateWeight + gateWeightBytes * expert, gate->cudaData,
            gateWeightBytes, cudaMemcpyDeviceToDevice, stream) == cudaSuccess;
        if (!success) {
            break;
        }
        const int gatePacks = kTpGateColumns * (kHidden / 32 / 4);
        TransformWeightScalesKernel<<<
            (gatePacks + threads - 1) / threads, threads, 0, stream>>>(
            (const uint8_t *)gate->cudaData + gateWeightBytes,
            cache.gateWeightScales +
                (size_t)expert * gateScaleBytes / sizeof(uint32_t),
            kTpGateColumns, kHidden / 32);
        success = cudaGetLastError() == cudaSuccess &&
            cudaMemcpyAsync(
                cache.downWeight + downWeightBytes * expert, down->cudaData,
                downWeightBytes, cudaMemcpyDeviceToDevice, stream) ==
                cudaSuccess;
        if (!success) {
            break;
        }
        const int downPacks = kHidden * (kTpIntermediate / 32 / 4);
        TransformWeightScalesKernel<<<
            (downPacks + threads - 1) / threads, threads, 0, stream>>>(
            (const uint8_t *)down->cudaData + downWeightBytes,
            cache.downWeightScales +
                (size_t)expert * downScaleBytes / sizeof(uint32_t),
            kHidden, kTpIntermediate / 32);
        success = cudaGetLastError() == cudaSuccess;
    }
    if (success) {
        success = cudaStreamSynchronize(stream) == cudaSuccess &&
                  BuildTmaDescriptors(cache);
    }
    if (!success) {
        ReleaseLayerStorage(cache);
        return false;
    }

    for (int expert = 0; expert < kTpExperts; ++expert) {
        const int slot = 2 + expert * 2;
        for (int offset = 0; offset < 2; ++offset) {
            fastllm::Data *weight = weights[slot + offset];
            void *old = weight->cudaData;
            weight->cudaData = nullptr;
            weight->cudaDataBorrowed = false;
            if (weight->directMemory) {
                FastllmCudaDirectFree(old);
            } else {
                FastllmCudaFree(old);
            }
        }
    }

    cache.ready = true;
    return true;
}

static std::shared_ptr<LayerCache> GetOrBuildLayerCache(
        fastllm::Data **weights, int weightsBatch) {
    if (weights == nullptr || weightsBatch < 4 || weights[2] == nullptr) {
        return nullptr;
    }
    const fastllm::Data *key = weights[2];
    auto front = layerCacheFront.find(key);
    if (front != layerCacheFront.end()) {
        std::shared_ptr<LayerCache> cached = front->second.lock();
        if (cached != nullptr && cached->ready &&
            !cached->retired.load(std::memory_order_acquire)) {
            return cached;
        }
        layerCacheFront.erase(front);
    }

    std::shared_ptr<LayerCache> cache;
    {
        std::lock_guard<std::mutex> lock(LayerRegistryMutex());
        auto &registry = LayerRegistry();
        auto it = registry.find(key);
        if (it == registry.end()) {
            cache = std::make_shared<LayerCache>();
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
            failed = !BuildLayerCache(weights, weightsBatch, *cache);
            if (failed) {
                cache->retired.store(true, std::memory_order_release);
            }
        }
    }
    if (failed) {
        std::lock_guard<std::mutex> lock(LayerRegistryMutex());
        auto &registry = LayerRegistry();
        auto it = registry.find(key);
        if (it != registry.end() && it->second == cache) {
            registry.erase(it);
        }
        return nullptr;
    }
    layerCacheFront[key] = cache;
    return cache;
}

static void ReleaseLayerCache(const fastllm::Data *key) {
    if (key == nullptr) {
        return;
    }
    std::shared_ptr<LayerCache> retired;
    {
        std::lock_guard<std::mutex> lock(LayerRegistryMutex());
        auto &registry = LayerRegistry();
        auto it = registry.find(key);
        if (it == registry.end()) {
            return;
        }
        it->second->retired.store(true, std::memory_order_release);
        retired = std::move(it->second);
        registry.erase(it);
    }
    layerCacheFront.erase(key);
}

// Reference implementation retained for precision/performance A/B testing.
// The production kernel below must preserve this exact first-appearance
// ordering because DeepGEMM uses groupedLayout and routePositions together.
__global__ void BuildRouteMetadataReferenceKernel(
        const int32_t *__restrict__ globalIndices,
        int32_t *__restrict__ groupedLayout,
        int32_t *__restrict__ routePositions, int routes) {
    if (blockIdx.x != 0) {
        return;
    }
    const int thread = threadIdx.x;
    if (thread < kMaxRoutes) {
        groupedLayout[thread * kTpExpertBlockM] = -1;
    }
    __syncthreads();
    if (thread >= routes) {
        return;
    }

    routePositions[thread] = -1;
    const int expert = globalIndices[thread];
    if ((unsigned int)expert >= (unsigned int)kTpExperts) {
        return;
    }

    int firstRoute = thread;
    for (int previous = 0; previous < thread; ++previous) {
        if (globalIndices[previous] == expert) {
            firstRoute = previous;
            break;
        }
    }

    int localOffset = 0;
    for (int previous = 0; previous < thread; ++previous) {
        localOffset += globalIndices[previous] == expert;
    }
    if (localOffset >= kTpExpertBlockM) {
        return;
    }

    // Blocks are assigned by the order in which each expert first appears.
    // Counting all unique experts before `thread` is incorrect for duplicate
    // routes: a later occurrence would move to a new, uninitialised block.
    int uniqueBlock = 0;
    for (int previous = 0; previous < firstRoute; ++previous) {
        const int previousExpert = globalIndices[previous];
        if ((unsigned int)previousExpert >= (unsigned int)kTpExperts) {
            continue;
        }
        bool firstOccurrence = true;
        for (int earlier = 0; earlier < previous; ++earlier) {
            if (globalIndices[earlier] == previousExpert) {
                firstOccurrence = false;
                break;
            }
        }
        if (firstOccurrence) {
            ++uniqueBlock;
        }
    }
    const int position = uniqueBlock * kTpExpertBlockM + localOffset;
    routePositions[thread] = position;
    if (localOffset == 0) {
        groupedLayout[uniqueBlock * kTpExpertBlockM] = expert;
    }
}

__global__ void BuildRouteMetadataKernel(
        const int32_t *__restrict__ globalIndices,
        int32_t *__restrict__ groupedLayout,
        int32_t *__restrict__ routePositions, int routes) {
    if (blockIdx.x != 0) {
        return;
    }

    __shared__ int32_t sharedExperts[kMaxRoutes];
    __shared__ uint32_t firstMasks[2];
    const int thread = threadIdx.x;
    const bool active = thread < routes;
    const int32_t expert = active ? globalIndices[thread] : -1;
    sharedExperts[thread] = expert;
    groupedLayout[thread * kTpExpertBlockM] = -1;
    if (active) {
        routePositions[thread] = -1;
    }
    __syncthreads();

    const bool valid = active &&
        (unsigned int)expert < (unsigned int)kTpExperts;
    int firstRoute = thread;
    int localOffset = 0;
    if (valid) {
        // Stage the route list once in shared memory.  This single ordered
        // scan replaces the reference kernel's repeated global scans and its
        // nested first-occurrence search, while keeping local route order.
        for (int previous = 0; previous < thread; ++previous) {
            if (sharedExperts[previous] == expert) {
                if (localOffset == 0) {
                    firstRoute = previous;
                }
                ++localOffset;
            }
        }
    }

    const bool firstOccurrence = valid && localOffset == 0;
    const uint32_t firstMask = __ballot_sync(0xffffffffu, firstOccurrence);
    if ((thread & 31) == 0) {
        firstMasks[thread >> 5] = firstMask;
    }
    __syncthreads();

    if (!valid || localOffset >= kTpExpertBlockM) {
        return;
    }

    // Count first appearances before this route's first occurrence.  The
    // bitmask prefix is deterministic and therefore assigns exactly the same
    // grouped block as BuildRouteMetadataReferenceKernel.
    int uniqueBlock;
    if (firstRoute < 32) {
        const uint32_t prefix = firstRoute == 0
            ? 0u : ((1u << firstRoute) - 1u);
        uniqueBlock = __popc(firstMasks[0] & prefix);
    } else {
        const int upperBits = firstRoute - 32;
        const uint32_t prefix = upperBits == 0
            ? 0u : ((1u << upperBits) - 1u);
        uniqueBlock = __popc(firstMasks[0]) +
            __popc(firstMasks[1] & prefix);
    }
    const int position = uniqueBlock * kTpExpertBlockM + localOffset;
    routePositions[thread] = position;
    if (firstOccurrence) {
        groupedLayout[uniqueBlock * kTpExpertBlockM] = expert;
    }
}

static bool UseReferenceRouteMetadata() {
    static const bool enabled = []() {
        const char *env = std::getenv(
            "FASTLLM_DSV4_REFERENCE_ROUTE_METADATA");
        return env != nullptr && env[0] != '\0' &&
            std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool LaunchDeepGemm(DeepGemmKernel kernel,
                           __nv_bfloat16 *output, int outputColumns,
                           uint8_t *input, uint8_t *weight, int32_t *layout,
                           int shapeM, int shapeK,
                           const cute::TmaDescriptor &tmaA,
                           const cute::TmaDescriptor &tmaB,
                           const cute::TmaDescriptor &tmaSfa,
                           const cute::TmaDescriptor &tmaSfb,
                           const cute::TmaDescriptor &tmaD,
                           cudaStream_t stream) {
    kernel<<<kNumSms, kThreads, kDynamicSharedBytes, stream>>>(
        reinterpret_cast<cutlass::bfloat16_t *>(output), nullptr,
        reinterpret_cast<__nv_fp8_e4m3 *>(input),
        reinterpret_cast<__nv_fp8_e4m3 *>(weight), layout, nullptr, nullptr,
        shapeM, outputColumns, shapeK, outputColumns, 0,
        kTpWorkspaceRows * outputColumns,
        tmaA, tmaB, tmaSfa, tmaSfb, tmaD);
    return cudaGetLastError() == cudaSuccess;
}

static bool RunDeepGemmMoe(
        const fastllm::Data &input, fastllm::Data &output,
        fastllm::Data **weights, int weightsBatch,
        const int32_t *globalIndices, const float *scores, int topk,
        float swigluLimit) {
    if (input.dataType != fastllm::DataType::BFLOAT16 ||
        input.dataDevice != fastllm::DataDevice::CUDA ||
        input.dims.empty() || input.dims.back() != kHidden ||
        globalIndices == nullptr || scores == nullptr ||
        topk <= 0 || topk > kMaxTopK) {
        if (CompactExpertWeightsReleased(weights, weightsBatch)) {
            FailDeepGemmAfterRepack(
                "tensor-parallel invocation validation");
        }
        return false;
    }
    const int rows = (int)(input.Count(0) / kHidden);
    if (rows <= 0 || (size_t)rows * kHidden != input.Count(0)) {
        if (CompactExpertWeightsReleased(weights, weightsBatch)) {
            FailDeepGemmAfterRepack(
                "tensor-parallel input-shape validation");
        }
        return false;
    }
    std::shared_ptr<LayerCache> cache =
        GetOrBuildLayerCache(weights, weightsBatch);
    if (cache == nullptr || !cache->ready || cache->workspace == nullptr) {
        if (CompactExpertWeightsReleased(weights, weightsBatch)) {
            FailDeepGemmAfterRepack(
                "tensor-parallel cache recovery");
        }
        return false;
    }

    output.dataDevice = input.dataDevice;
    output.dataDeviceIds = input.dataDeviceIds;
    output.dataType = input.dataType;
    output.Resize(input.dims);
    output.Allocate(false);
    if (output.cudaData == nullptr) {
        FailDeepGemmAfterRepack(
            "tensor-parallel output allocation", cache->device);
    }

    RuntimeWorkspace &workspace = *cache->workspace;
    cudaStream_t stream = cudaStreamPerThread;
    constexpr int quantThreads = 256;
    constexpr int reduceThreads = 256;
    for (int rowBase = 0; rowBase < rows; rowBase += kMaxChunkRows) {
        const int chunkRows = std::min(kMaxChunkRows, rows - rowBase);
        const int routes = chunkRows * topk;
        const int activeWorkspaceRows = routes * kTpExpertBlockM;
        const int32_t *chunkIndices =
            globalIndices + (size_t)rowBase * topk;
        const float *chunkScores = scores + (size_t)rowBase * topk;
        const __nv_bfloat16 *chunkInput =
            (const __nv_bfloat16 *)input.cudaData + (size_t)rowBase * kHidden;
        __nv_bfloat16 *chunkOutput =
            (__nv_bfloat16 *)output.cudaData + (size_t)rowBase * kHidden;

        // Keep single-row draft/ordinary decode on the established kernel.
        // The verification bottleneck is the multi-row (normally M=8) path;
        // limiting the new scheduling there also keeps draft proposal timing
        // unchanged while its metadata work is already negligible.
        if (UseReferenceRouteMetadata() || chunkRows == 1) {
            BuildRouteMetadataReferenceKernel<<<1, kMaxRoutes, 0, stream>>>(
                chunkIndices, workspace.groupedLayout,
                workspace.routePositions, routes);
        } else {
            BuildRouteMetadataKernel<<<1, kMaxRoutes, 0, stream>>>(
                chunkIndices, workspace.groupedLayout,
                workspace.routePositions, routes);
        }
        QuantizeRoutedRowsKernel<false, kTpWorkspaceRows><<<
            routes, quantThreads, 0, stream>>>(
            chunkInput, workspace.routePositions, workspace.gateInput,
            workspace.gateInputScales, routes, topk, kHidden, kHidden,
            swigluLimit);
        if (cudaGetLastError() != cudaSuccess ||
            !LaunchDeepGemm(
                GateKernel(), workspace.gateOutput, kTpGateColumns,
                workspace.gateInput, cache->gateWeight,
                workspace.groupedLayout, activeWorkspaceRows, kHidden,
                cache->gateA, cache->gateB, cache->gateSfa,
                cache->gateSfb, cache->gateD, stream)) {
            FailDeepGemmAfterRepack(
                "tensor-parallel gate launch", cache->device);
        }
        QuantizeRoutedRowsKernel<true, kTpWorkspaceRows><<<
            routes, quantThreads, 0, stream>>>(
            workspace.gateOutput, workspace.routePositions,
            workspace.downInput, workspace.downInputScales, routes, topk,
            kTpGateColumns, kTpIntermediate, swigluLimit);
        if (cudaGetLastError() != cudaSuccess ||
            !LaunchDeepGemm(
                DownKernel(), workspace.downOutput, kHidden,
                workspace.downInput, cache->downWeight,
                workspace.groupedLayout, activeWorkspaceRows, kTpIntermediate,
                cache->downA, cache->downB, cache->downSfa,
                cache->downSfb, cache->downD, stream)) {
            FailDeepGemmAfterRepack(
                "tensor-parallel down launch", cache->device);
        }
        const int outputCount = chunkRows * kHidden;
        ReduceRoutedRowsKernel<<<
            (outputCount + reduceThreads - 1) / reduceThreads,
            reduceThreads, 0, stream>>>(
            workspace.downOutput, workspace.routePositions, chunkScores,
            chunkOutput, chunkRows, topk);
        if (cudaGetLastError() != cudaSuccess) {
            FailDeepGemmAfterRepack(
                "tensor-parallel output reduction", cache->device);
        }
    }
    return true;
}

}  // namespace tensor_parallel

}  // namespace fastllm_deepgemm_moe

bool FastllmCudaBFloat16MergeMOEDeepGemmSm120ExpertParallel(
        const fastllm::Data &input, fastllm::Data &output,
        fastllm::Data **weights, int weightsBatch,
        const int32_t *globalIndices, const float *scores,
        int topk, int ownerRank, int ownerCount, float swigluLimit) {
    return fastllm_deepgemm_moe::RunDeepGemmMoe(
        input, output, weights, weightsBatch, globalIndices, scores,
        topk, ownerRank, ownerCount, swigluLimit);
}

bool FastllmCudaBFloat16MergeMOEDeepGemmSm120TensorParallel(
        const fastllm::Data &input, fastllm::Data &output,
        fastllm::Data **weights, int weightsBatch,
        const int32_t *globalIndices, const float *scores, int topk,
        float swigluLimit) {
    return fastllm_deepgemm_moe::tensor_parallel::RunDeepGemmMoe(
        input, output, weights, weightsBatch, globalIndices, scores, topk,
        swigluLimit);
}

void FastllmCudaReleaseMergeMOEDeepGemmSm120Cache(
        const fastllm::Data *layerKey) {
    fastllm_deepgemm_moe::ReleaseLayerCache(layerKey);
    fastllm_deepgemm_moe::tensor_parallel::ReleaseLayerCache(layerKey);
}
