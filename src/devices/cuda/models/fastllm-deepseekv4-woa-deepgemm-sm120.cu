/*
 * DeepSeek-V4 DSpark WoA projection for NVIDIA SM120.
 *
 * TP8 leaves one output group per rank.  For seven/eight speculative rows,
 * vLLM swaps the narrow GEMM to [1024, 4096] x [N, 4096]^T and uses a
 * split-K=4 DeepGEMM specialization.  This file mirrors that schedule while
 * retaining the generic Triton/CUDA implementations as fallbacks for every
 * other shape and architecture.
 */

#include "fastllm-cuda.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh>
#include <deep_gemm/impls/sm120_split_k_reduce.cuh>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace fastllm_deepgemm_woa {

using namespace deep_gemm;

constexpr int kHidden = 4096;
constexpr int kOutputRank = 1024;
constexpr int kTargetRows = 8;
constexpr int kDraftRows = 7;
constexpr int kScaleGranularity = 128;
constexpr int kScalePacks = kHidden / kScaleGranularity / 4;
constexpr int kSplitK = 4;
constexpr int kNumSms = 170;
constexpr int kThreads = 384;
constexpr int kDynamicSharedBytes = 95872;
constexpr int kAlignedRows = 8;

using DeepGemmKernel = void (*)(
    cutlass::bfloat16_t *, const cutlass::bfloat16_t *, __nv_fp8_e4m3 *,
    __nv_fp8_e4m3 *, int *, cute::TmaDescriptor *, float *, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    cute::TmaDescriptor, cute::TmaDescriptor, cute::TmaDescriptor,
    cute::TmaDescriptor, cute::TmaDescriptor);

using SplitKReduceKernel = void (*)(
    cutlass::bfloat16_t *, const float *, uint32_t, uint32_t, int, int);

static DeepGemmKernel TargetKernel() {
    return &sm120_fp8_fp4_gemm_1d1d_impl<
        0, kTargetRows, kHidden,
        128, 128, 1,
        64, 16, 128,
        128, 128, 0,
        9, 128, 256, kNumSms,
        GemmType::Normal, false, cutlass::bfloat16_t,
        epilogue::transform::EpilogueIdentity,
        false, false, false, true, false, 64, kSplitK>;
}

static DeepGemmKernel DraftKernel() {
    return &sm120_fp8_fp4_gemm_1d1d_impl<
        0, kDraftRows, kHidden,
        128, 128, 1,
        64, 16, 128,
        128, 128, 0,
        9, 128, 256, kNumSms,
        GemmType::Normal, false, cutlass::bfloat16_t,
        epilogue::transform::EpilogueIdentity,
        false, false, false, true, false, 64, kSplitK>;
}

static SplitKReduceKernel ReduceKernel() {
    return &sm120_split_k_reduce_impl<cutlass::bfloat16_t, kSplitK>;
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

struct DeviceWorkspace {
    bool ready = false;
    int device = -1;
    uint8_t *input = nullptr;
    uint32_t *inputScales = nullptr;
    float *splitK = nullptr;
    __nv_bfloat16 *dummyOutput = nullptr;
    cute::TmaDescriptor inputTarget;
    cute::TmaDescriptor inputDraft;
    cute::TmaDescriptor inputScale;
    cute::TmaDescriptor dummyD;
    std::mutex mutex;
};

struct WeightCache {
    bool ready = false;
    int device = -1;
    const void *weightPointer = nullptr;
    uint32_t *packedScales = nullptr;
    cute::TmaDescriptor weight;
    cute::TmaDescriptor scales;
    std::mutex mutex;
};

static void ReleaseWorkspaceStorage(DeviceWorkspace &workspace) {
    if (workspace.input != nullptr) {
        FastllmCudaDirectFree(workspace.input);
    }
    if (workspace.inputScales != nullptr) {
        FastllmCudaDirectFree(workspace.inputScales);
    }
    if (workspace.splitK != nullptr) {
        FastllmCudaDirectFree(workspace.splitK);
    }
    if (workspace.dummyOutput != nullptr) {
        FastllmCudaDirectFree(workspace.dummyOutput);
    }
    workspace.input = nullptr;
    workspace.inputScales = nullptr;
    workspace.splitK = nullptr;
    workspace.dummyOutput = nullptr;
    workspace.device = -1;
    workspace.ready = false;
}

static void ReleaseWeightCacheStorage(WeightCache &cache) {
    if (cache.packedScales != nullptr) {
        FastllmCudaDirectFree(cache.packedScales);
    }
    cache.packedScales = nullptr;
    cache.weightPointer = nullptr;
    cache.device = -1;
    cache.ready = false;
}

static std::mutex &RegistryMutex() {
    static auto *mutex = new std::mutex();
    return *mutex;
}

static std::map<int, std::shared_ptr<DeviceWorkspace> > &WorkspaceRegistry() {
    static auto *registry =
        new std::map<int, std::shared_ptr<DeviceWorkspace> >();
    return *registry;
}

using WeightKey = std::pair<int, const void *>;

static std::map<WeightKey, std::shared_ptr<WeightCache> > &WeightRegistry() {
    static auto *registry =
        new std::map<WeightKey, std::shared_ptr<WeightCache> >();
    return *registry;
}

static bool PrepareKernelFunctions() {
    return cudaFuncSetAttribute(
               TargetKernel(), cudaFuncAttributeMaxDynamicSharedMemorySize,
               kDynamicSharedBytes) == cudaSuccess &&
           cudaFuncSetAttribute(
               DraftKernel(), cudaFuncAttributeMaxDynamicSharedMemorySize,
               kDynamicSharedBytes) == cudaSuccess;
}

static bool PrepareWorkspace(DeviceWorkspace &workspace, int device) {
    std::lock_guard<std::mutex> guard(workspace.mutex);
    if (workspace.ready) {
        return workspace.device == device;
    }
    if (FastllmCudaGraphIsCapturing()) {
        return false;
    }

    ReleaseWorkspaceStorage(workspace);
    workspace.device = device;
    workspace.input = reinterpret_cast<uint8_t *>(
        FastllmCudaDirectMalloc((size_t)kTargetRows * kHidden));
    workspace.inputScales = reinterpret_cast<uint32_t *>(
        FastllmCudaDirectMalloc(
            (size_t)kScalePacks * kAlignedRows * sizeof(uint32_t)));
    workspace.splitK = reinterpret_cast<float *>(
        FastllmCudaDirectMalloc(
            (size_t)kSplitK * kOutputRank * kTargetRows * sizeof(float)));
    workspace.dummyOutput = reinterpret_cast<__nv_bfloat16 *>(
        FastllmCudaDirectMalloc(
            (size_t)kOutputRank * kTargetRows * sizeof(__nv_bfloat16)));
    if (workspace.input == nullptr || workspace.inputScales == nullptr ||
        workspace.splitK == nullptr || workspace.dummyOutput == nullptr ||
        !PrepareKernelFunctions()) {
        ReleaseWorkspaceStorage(workspace);
        return false;
    }

    workspace.ready =
        MakeTma2d(
            workspace.inputTarget, CU_TENSOR_MAP_DATA_TYPE_UINT8,
            workspace.input, kHidden, kTargetRows, kHidden,
            128, 16, CU_TENSOR_MAP_SWIZZLE_128B) &&
        MakeTma2d(
            workspace.inputDraft, CU_TENSOR_MAP_DATA_TYPE_UINT8,
            workspace.input, kHidden, kDraftRows, kHidden,
            128, 16, CU_TENSOR_MAP_SWIZZLE_128B) &&
        MakeTma2d(
            workspace.inputScale, CU_TENSOR_MAP_DATA_TYPE_INT32,
            workspace.inputScales, kAlignedRows, kScalePacks,
            kAlignedRows * sizeof(uint32_t), 16, 1,
            CU_TENSOR_MAP_SWIZZLE_NONE) &&
        MakeTma2d(
            workspace.dummyD, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            workspace.dummyOutput, kTargetRows, kOutputRank,
            kTargetRows * sizeof(__nv_bfloat16), 8, 64,
            CU_TENSOR_MAP_SWIZZLE_NONE);
    if (!workspace.ready) {
        ReleaseWorkspaceStorage(workspace);
    }
    return workspace.ready;
}

static std::shared_ptr<DeviceWorkspace> GetWorkspace(int device) {
    std::shared_ptr<DeviceWorkspace> workspace;
    {
        std::lock_guard<std::mutex> guard(RegistryMutex());
        auto &registry = WorkspaceRegistry();
        auto it = registry.find(device);
        if (it == registry.end()) {
            workspace = std::make_shared<DeviceWorkspace>();
            registry.emplace(device, workspace);
        } else {
            workspace = it->second;
        }
    }
    return PrepareWorkspace(*workspace, device) ? workspace : nullptr;
}

static uint8_t Ue8m0Byte(float scale) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(scale), "unexpected float size");
    std::memcpy(&bits, &scale, sizeof(bits));
    return static_cast<uint8_t>((bits >> 23) & 0xffu);
}

static bool PrepareWeightCache(WeightCache &cache,
                               const fastllm::Data &weight,
                               int device) {
    std::lock_guard<std::mutex> guard(cache.mutex);
    if (cache.ready) {
        return cache.device == device &&
               cache.weightPointer == weight.cudaData;
    }
    if (FastllmCudaGraphIsCapturing()) {
        return false;
    }

    constexpr int outputScaleBlocks = kOutputRank / kScaleGranularity;
    constexpr int kScaleBlocks = kHidden / kScaleGranularity;
    if (weight.scales.size() !=
        (size_t)outputScaleBlocks * kScaleBlocks) {
        return false;
    }

    std::vector<uint32_t> packed((size_t)kScalePacks * kOutputRank);
    for (int pack = 0; pack < kScalePacks; ++pack) {
        for (int row = 0; row < kOutputRank; ++row) {
            const int outputBlock = row / kScaleGranularity;
            uint32_t value = 0;
            for (int byte = 0; byte < 4; ++byte) {
                const float scale = weight.scales[
                    (size_t)outputBlock * kScaleBlocks + pack * 4 + byte];
                value |= static_cast<uint32_t>(Ue8m0Byte(scale)) <<
                         (byte * 8);
            }
            packed[(size_t)pack * kOutputRank + row] = value;
        }
    }

    ReleaseWeightCacheStorage(cache);
    cache.device = device;
    cache.weightPointer = weight.cudaData;
    cache.packedScales = reinterpret_cast<uint32_t *>(
        FastllmCudaDirectMalloc(packed.size() * sizeof(uint32_t)));
    if (cache.packedScales == nullptr ||
        cudaMemcpyAsync(cache.packedScales, packed.data(),
                        packed.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice,
                        cudaStreamPerThread) != cudaSuccess) {
        ReleaseWeightCacheStorage(cache);
        return false;
    }

    cache.ready =
        MakeTma2d(
            cache.weight, CU_TENSOR_MAP_DATA_TYPE_UINT8,
            weight.cudaData, kHidden, kOutputRank, kHidden,
            128, 64, CU_TENSOR_MAP_SWIZZLE_128B) &&
        MakeTma2d(
            cache.scales, CU_TENSOR_MAP_DATA_TYPE_INT32,
            cache.packedScales, kOutputRank, kScalePacks,
            kOutputRank * sizeof(uint32_t), 64, 1,
            CU_TENSOR_MAP_SWIZZLE_NONE);
    if (!cache.ready) {
        ReleaseWeightCacheStorage(cache);
    }
    return cache.ready;
}

static std::shared_ptr<WeightCache> GetWeightCache(
        const fastllm::Data &weight, int device) {
    const WeightKey key(device, weight.cudaData);
    std::shared_ptr<WeightCache> cache;
    {
        std::lock_guard<std::mutex> guard(RegistryMutex());
        auto &registry = WeightRegistry();
        auto it = registry.find(key);
        if (it == registry.end()) {
            cache = std::make_shared<WeightCache>();
            registry.emplace(key, cache);
        } else {
            cache = it->second;
        }
    }
    return PrepareWeightCache(*cache, weight, device) ? cache : nullptr;
}

__device__ __forceinline__ uint8_t Ue8m0ScaleByte(float maximum,
                                                   float &inverseScale) {
    float rawScale = fmaxf(maximum, 1.0e-10f) * (1.0f / 448.0f);
    float exponent = ceilf(log2f(rawScale));
    float scale = exp2f(exponent);
    inverseScale = 1.0f / scale;
    int encoded = static_cast<int>(exponent) + 127;
    return static_cast<uint8_t>(max(0, min(255, encoded)));
}

// Keep one CTA per K/128 group.  DSpark target decode has 8 * 32 = 256 such
// groups, enough to fill all 170 SMs; grouping four reductions into one CTA
// left more than half of the GPU idle.  Each CTA writes one byte of the
// packed INT32 scale buffer, so adjacent groups do not need an atomic update.
__global__ void QuantizeInputKernel(
        const __nv_bfloat16 *__restrict__ input,
        uint8_t *__restrict__ output,
        uint8_t *__restrict__ scaleBytes,
        int rows) {
    __shared__ float warpMax[4];
    __shared__ float inverseScale;
    __shared__ uint8_t encodedScale;

    constexpr int groupsPerRow = kHidden / kScaleGranularity;
    const int token = blockIdx.x / groupsPerRow;
    const int group = blockIdx.x - token * groupsPerRow;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int column = group * kScaleGranularity + threadIdx.x;
    const float value = __bfloat162float(
        input[(size_t)token * kHidden + column]);
    float maximum = fabsf(value);
    for (int delta = 16; delta > 0; delta >>= 1) {
        maximum = fmaxf(
            maximum,
            __shfl_down_sync(0xffffffff, maximum, delta));
    }
    if (lane == 0) {
        warpMax[warp] = maximum;
    }
    __syncthreads();

    if (warp == 0) {
        maximum = lane < 4 ? warpMax[lane] : 0.0f;
        for (int delta = 16; delta > 0; delta >>= 1) {
            maximum = fmaxf(maximum,
                __shfl_down_sync(0xffffffff, maximum, delta));
        }
        if (lane == 0) {
            encodedScale =
                Ue8m0ScaleByte(maximum, inverseScale);
        }
    }
    __syncthreads();

    float q = fminf(448.0f, fmaxf(
        -448.0f, value * inverseScale));
    output[(size_t)token * kHidden + column] =
        __nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
    if (threadIdx.x == 0) {
        const int pack = group >> 2;
        const int byte = group & 3;
        scaleBytes[((size_t)pack * kAlignedRows + token) * 4 + byte] =
            encodedScale;
    }
    cudaTriggerProgrammaticLaunchCompletion();
}

static bool Launch(const fastllm::Data &input,
                   const fastllm::Data &weight,
                   fastllm::Data &output,
                   int rows) {
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
        major != 12 || minor != 0 || sms != kNumSms) {
        cudaGetLastError();
        return false;
    }

    std::shared_ptr<DeviceWorkspace> workspace = GetWorkspace(device);
    std::shared_ptr<WeightCache> weightCache =
        GetWeightCache(weight, device);
    if (workspace == nullptr || weightCache == nullptr) {
        return false;
    }

    cudaStream_t stream = cudaStreamPerThread;
    constexpr int groupsPerRow = kHidden / kScaleGranularity;
    QuantizeInputKernel<<<rows * groupsPerRow, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(input.cudaData),
        workspace->input,
        reinterpret_cast<uint8_t *>(workspace->inputScales), rows);
    if (cudaGetLastError() != cudaSuccess) {
        return false;
    }

    DeepGemmKernel kernel = rows == kTargetRows ?
        TargetKernel() : DraftKernel();
    const cute::TmaDescriptor &inputDescriptor = rows == kTargetRows ?
        workspace->inputTarget : workspace->inputDraft;
    cudaLaunchAttribute launchAttribute = {};
    launchAttribute.id =
        cudaLaunchAttributeProgrammaticStreamSerialization;
    launchAttribute.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t launchConfig = {};
    launchConfig.gridDim = dim3(kNumSms);
    launchConfig.blockDim = dim3(kThreads);
    launchConfig.dynamicSmemBytes = kDynamicSharedBytes;
    launchConfig.stream = stream;
    launchConfig.attrs = &launchAttribute;
    launchConfig.numAttrs = 1;
    cudaError_t state = cudaLaunchKernelEx(
        &launchConfig, kernel,
        reinterpret_cast<cutlass::bfloat16_t *>(output.cudaData), nullptr,
        reinterpret_cast<__nv_fp8_e4m3 *>(weight.cudaData),
        reinterpret_cast<__nv_fp8_e4m3 *>(workspace->input),
        nullptr, nullptr, workspace->splitK,
        kOutputRank, rows, kHidden,
        1, kOutputRank, kOutputRank * rows,
        weightCache->weight, inputDescriptor,
        weightCache->scales, workspace->inputScale,
        workspace->dummyD);
    if (state != cudaSuccess) {
        return false;
    }

    constexpr int reduceThreads = 256;
    const int elements = kOutputRank * rows;
    launchConfig.gridDim =
        dim3((elements + reduceThreads - 1) / reduceThreads);
    launchConfig.blockDim = dim3(reduceThreads);
    launchConfig.dynamicSmemBytes = 0;
    state = cudaLaunchKernelEx(
        &launchConfig, ReduceKernel(),
            reinterpret_cast<cutlass::bfloat16_t *>(output.cudaData),
            workspace->splitK, kOutputRank, rows, 1, kOutputRank);
    return state == cudaSuccess;
}

static bool Run(const fastllm::Data &input,
                const fastllm::Data &weight,
                int groups, int outputRank,
                fastllm::Data &output) {
    if (input.dataDevice != fastllm::DataDevice::CUDA ||
        input.dataType != fastllm::DataType::BFLOAT16 ||
        input.cudaData == nullptr || input.dims.size() != 4 ||
        weight.dataDevice != fastllm::DataDevice::CUDA ||
        weight.dataType != fastllm::DataType::FP8_E4M3 ||
        weight.cudaData == nullptr || weight.dims.size() != 2 ||
        weight.blockK != kScaleGranularity ||
        weight.blockM != kScaleGranularity ||
        output.dataDevice != fastllm::DataDevice::CUDA ||
        output.dataType != fastllm::DataType::BFLOAT16 ||
        output.cudaData == nullptr || groups != 1 ||
        outputRank != kOutputRank) {
        return false;
    }

    const int rows = input.dims[0] * input.dims[1];
    const int localHidden = input.dims[2] * input.dims[3];
    if ((rows != kDraftRows && rows != kTargetRows) ||
        localHidden != kHidden ||
        input.Count(0) != (uint64_t)rows * kHidden ||
        weight.dims[0] != kOutputRank ||
        weight.dims[1] != kHidden ||
        output.Count(0) != (uint64_t)rows * kOutputRank) {
        return false;
    }
    return Launch(input, weight, output, rows);
}

}  // namespace fastllm_deepgemm_woa

extern "C" bool FastllmCudaDeepSeekV4WoADeepGemmSm120(
        const fastllm::Data &input, const fastllm::Data &weight,
        int groups, int outputRank, fastllm::Data &output) {
    return fastllm_deepgemm_woa::Run(
        input, weight, groups, outputRank, output);
}
