#include "fastllm-cuda.cuh"
#include "fastllm.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

#ifdef FASTLLM_ENABLE_CUTLASS_FP8

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"

namespace fastllm_cuda_cutlass_fp8 {

template <typename Kernel>
struct FastllmEnableSm120Family : Kernel {
    template <typename... Args>
    CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined(__CUDA_ARCH__)
#if (__CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300)
        Kernel::operator()(std::forward<Args>(args)...);
#else
        printf("This kernel only supports sm120/sm121.\n");
        asm("trap;");
#endif
#endif
    }
};

using namespace cute;

template <class OutType, int ScaleGranularityM, int ScaleGranularityN, int ScaleGranularityK,
          class MmaTileShape, class ClusterShape, class EpilogueScheduler,
          class MainloopScheduler, bool SwapAB = false>
struct FastllmCutlassFp8BlockwiseGemm {
    static constexpr bool swap_ab = SwapAB;
    using ElementAB = cutlass::float_e4m3_t;

    using ElementA = ElementAB;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutATranspose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

    using ElementB = ElementAB;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutBTranspose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

    using ElementD = OutType;
    using LayoutD = cutlass::layout::RowMajor;
    using LayoutDTranspose = typename cutlass::layout::LayoutTranspose<LayoutD>::type;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    using ElementC = void;
    using LayoutC = LayoutD;
    using LayoutCTranspose = LayoutDTranspose;
    static constexpr int AlignmentC = AlignmentD;

    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementBlockScale = float;

    using ScaleConfig = std::conditional_t<
        SwapAB,
        cutlass::detail::Sm120BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN,
                                                   ScaleGranularityK, UMMA::Major::K,
                                                   UMMA::Major::MN>,
        cutlass::detail::Sm120BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN,
                                                   ScaleGranularityK, UMMA::Major::MN,
                                                   UMMA::Major::K>>;

    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

    using ArchTag = cutlass::arch::Sm120;
    using OperatorClass = cutlass::arch::OpClassTensorOp;

    static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
    using ElementScalar = float;
    using DefaultOperation = cutlass::epilogue::fusion::LinearCombination<
        ElementD, ElementCompute, ElementC, ElementScalar, RoundStyle>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,
        ElementC, std::conditional_t<SwapAB, LayoutCTranspose, LayoutC>, AlignmentC,
        ElementD, std::conditional_t<SwapAB, LayoutDTranspose, LayoutD>, AlignmentD,
        EpilogueScheduler, DefaultOperation>::CollectiveOp;

    using CollectiveMainloop = std::conditional_t<
        SwapAB,
        typename cutlass::gemm::collective::CollectiveBuilder<
            ArchTag, OperatorClass, ElementB, cute::tuple<LayoutBTranspose, LayoutSFA>,
            AlignmentB, ElementA, cute::tuple<LayoutATranspose, LayoutSFB>, AlignmentA,
            ElementAccumulator, MmaTileShape, ClusterShape,
            cutlass::gemm::collective::StageCountAutoCarveout<
                static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
            MainloopScheduler>::CollectiveOp,
        typename cutlass::gemm::collective::CollectiveBuilder<
            ArchTag, OperatorClass, ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
            ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB, ElementAccumulator,
            MmaTileShape, ClusterShape,
            cutlass::gemm::collective::StageCountAutoCarveout<
                static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
            MainloopScheduler>::CollectiveOp>;

    using KernelType = FastllmEnableSm120Family<cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>>;

    struct GemmKernel : public KernelType {};
};

template <typename OutType>
struct FastllmSm120Fp8DefaultConfig {
    using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
    using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
    using TileShape = Shape<_128, _128, _128>;
    using ClusterShape = Shape<_1, _1, _1>;
    using Gemm = FastllmCutlassFp8BlockwiseGemm<
        OutType, 1, 128, 128, TileShape, ClusterShape, EpilogueSchedule, KernelSchedule>;
};

template <typename OutType>
struct FastllmSm120Fp8PingpongConfig {
    using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120;
    using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
    using TileShape = Shape<_64, _128, _128>;
    using ClusterShape = Shape<_1, _1, _1>;
    using Gemm = FastllmCutlassFp8BlockwiseGemm<
        OutType, 1, 128, 128, TileShape, ClusterShape, EpilogueSchedule, KernelSchedule>;
};

template <typename OutType>
struct FastllmSm120Fp8SwapABConfig {
    using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedBlockwiseCooperativeSm120;
    using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
    using TileShape = Shape<_128, _32, _128>;
    using ClusterShape = Shape<_1, _1, _1>;
    using Gemm = FastllmCutlassFp8BlockwiseGemm<
        OutType, 128, 1, 128, TileShape, ClusterShape, EpilogueSchedule, KernelSchedule, true>;
};

struct FastllmCutlassFp8Scratch {
    cutlass::float_e4m3_t *input = nullptr;
    float *inputScales = nullptr;
    size_t inputElems = 0;
    size_t scaleElems = 0;
};

struct FastllmCutlassFp8WeightCache {
    cutlass::float_e4m3_t *weightTN = nullptr;
    float *weightScales = nullptr;
    const float *hostScales = nullptr;
    size_t scaleCount = 0;
    int inFeatures = 0;
    int outFeatures = 0;
    int blockM = 0;
    int blockK = 0;
};

static std::mutex g_cutlassScratchMutex;
static std::map<int, FastllmCutlassFp8Scratch> g_cutlassScratchByDevice;
static std::mutex g_cutlassWeightMutex;
static std::map<std::pair<int, const void*>, FastllmCutlassFp8WeightCache> g_cutlassWeightCache;

static bool FastllmCutlassIsStreamCapturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    cudaError_t state = cudaStreamIsCapturing(stream, &status);
    return state == cudaSuccess && status != cudaStreamCaptureStatusNone;
}

static bool FastllmCutlassFp8CompiledForRuntimeArch(int arch) {
    switch (arch) {
#if defined(FASTLLM_CUTLASS_FP8_ENABLE_SM120)
    case 120:
        return true;
#endif
#if defined(FASTLLM_CUTLASS_FP8_ENABLE_SM121)
    case 121:
        return true;
#endif
    default:
        return false;
    }
}

__device__ inline float FastllmCutlassToFloat(half x) {
    return __half2float(x);
}

__device__ inline float FastllmCutlassToFloat(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <typename T>
struct FastllmCutlassInputFp8QuantTraits;

template <>
struct FastllmCutlassInputFp8QuantTraits<half> {
    __device__ static inline float ToFloat(uint16_t bits) {
        return __half2float(__ushort_as_half(bits));
    }
    __device__ static inline half FromBits(uint16_t bits) {
        return __ushort_as_half(bits);
    }
};

template <>
struct FastllmCutlassInputFp8QuantTraits<__nv_bfloat16> {
    __device__ static inline float ToFloat(uint16_t bits) {
        return __bfloat162float(__ushort_as_bfloat16(bits));
    }
    __device__ static inline __nv_bfloat16 FromBits(uint16_t bits) {
        return __ushort_as_bfloat16(bits);
    }
};

template <typename T>
__device__ inline void FastllmCutlassLoad4AsFloat(const T *__restrict__ ptr, float (&values)[4]) {
    static_assert(sizeof(T) == 2, "FP8 input quantization expects 16-bit source elements");
    uint2 packed = *reinterpret_cast<const uint2 *>(ptr);
    values[0] = FastllmCutlassInputFp8QuantTraits<T>::ToFloat((uint16_t)(packed.x & 0xffffu));
    values[1] = FastllmCutlassInputFp8QuantTraits<T>::ToFloat((uint16_t)(packed.x >> 16));
    values[2] = FastllmCutlassInputFp8QuantTraits<T>::ToFloat((uint16_t)(packed.y & 0xffffu));
    values[3] = FastllmCutlassInputFp8QuantTraits<T>::ToFloat((uint16_t)(packed.y >> 16));
}

__device__ inline uint8_t FastllmCutlassFloatToFp8Byte(float v) {
    v = fminf(448.0f, fmaxf(-448.0f, v));
    return cutlass::float_e4m3_t(v).storage;
}

__device__ inline uint32_t FastllmCutlassPackFp8x4(float v0, float v1, float v2, float v3) {
    uint32_t b0 = FastllmCutlassFloatToFp8Byte(v0);
    uint32_t b1 = FastllmCutlassFloatToFp8Byte(v1);
    uint32_t b2 = FastllmCutlassFloatToFp8Byte(v2);
    uint32_t b3 = FastllmCutlassFloatToFp8Byte(v3);
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

template <typename T>
__global__ void FastllmCutlassQuantInputFp8Kernel(
    const T *input, cutlass::float_e4m3_t *quant, float *scales,
    int rows, int cols, int scaleCols) {
    int row = blockIdx.x;
    int group = blockIdx.y;
    int base = group * 128;
    __shared__ float reduce[256];

    float maxAbs = 0.0f;
    for (int i = threadIdx.x; i < 128; i += blockDim.x) {
        int col = base + i;
        float v = col < cols ? FastllmCutlassToFloat(input[(size_t)row * cols + col]) : 0.0f;
        maxAbs = fmaxf(maxAbs, fabsf(v));
    }
    reduce[threadIdx.x] = maxAbs;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    float scale = fmaxf(reduce[0], 1.0e-10f) * (1.0f / 448.0f);
    for (int i = threadIdx.x; i < 128; i += blockDim.x) {
        int col = base + i;
        if (col < cols) {
            float v = FastllmCutlassToFloat(input[(size_t)row * cols + col]) / scale;
            v = fminf(448.0f, fmaxf(-448.0f, v));
            quant[(size_t)row * cols + col] = cutlass::float_e4m3_t(v);
        }
    }
    if (threadIdx.x == 0) {
        // CUTLASS blockwise SFA layout is physically K-block-major:
        // offset = k_block * rows + row.
        scales[(size_t)group * rows + row] = scale;
    }
}

template <typename T, int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(256) FastllmCutlassQuantInputFp8PackedWarpKernel(
    const T *__restrict__ input, cutlass::float_e4m3_t *__restrict__ quant,
    float *__restrict__ scales, int rows, int cols, int scaleCols) {
    int warpId = threadIdx.x >> 5;
    int laneId = threadIdx.x & 31;
    int task = blockIdx.x * WARPS_PER_BLOCK + warpId;
    int totalTasks = rows * scaleCols;
    if (task >= totalTasks) {
        return;
    }

    int row = task / scaleCols;
    int group = task - row * scaleCols;
    int base = group * 128;
    size_t blockOffset = (size_t)row * cols + base;

    float values[4];
    FastllmCutlassLoad4AsFloat(input + blockOffset + laneId * 4, values);

    float maxAbs = fmaxf(fmaxf(fabsf(values[0]), fabsf(values[1])),
                         fmaxf(fabsf(values[2]), fabsf(values[3])));
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maxAbs = fmaxf(maxAbs, __shfl_down_sync(0xffffffff, maxAbs, offset));
    }
    maxAbs = __shfl_sync(0xffffffff, maxAbs, 0);

    float scale = fmaxf(maxAbs, 1.0e-10f) * (1.0f / 448.0f);
    if (laneId == 0) {
        // CUTLASS blockwise SFA layout is physically K-block-major:
        // offset = k_block * rows + row.
        scales[(size_t)group * rows + row] = scale;
    }
    float invScale = 1.0f / scale;

    uint32_t packed = FastllmCutlassPackFp8x4(values[0] * invScale, values[1] * invScale,
                                             values[2] * invScale, values[3] * invScale);
    reinterpret_cast<uint32_t *>(quant + blockOffset)[laneId] = packed;
}

__device__ inline float FastllmCutlassRoundedSwiglu(half gate, half up) {
#ifdef CUDA_NO_TENSOR_CORE
    float x = __half2float(gate);
    float y = __half2float(up);
    half rounded = __float2half((x / (1.0 + expf(-x))) * y);
#else
    half rounded = __hmul(__hdiv(gate, __hadd(__float2half(1.0), hexp(-gate))), up);
#endif
    return __half2float(rounded);
}

__device__ inline float FastllmCutlassRoundedSwiglu(__nv_bfloat16 gate, __nv_bfloat16 up) {
    float x = __bfloat162float(gate);
    float y = __bfloat162float(up);
    __nv_bfloat16 rounded = __float2bfloat16((x / (1.0f + expf(-x))) * y);
    return __bfloat162float(rounded);
}

template <typename T>
__device__ inline void FastllmCutlassLoad4RoundedSwiglu(
    const T *__restrict__ gatePtr, const T *__restrict__ upPtr, float (&values)[4]) {
    uint2 gatePacked = *reinterpret_cast<const uint2 *>(gatePtr);
    uint2 upPacked = *reinterpret_cast<const uint2 *>(upPtr);
    values[0] = FastllmCutlassRoundedSwiglu(
        FastllmCutlassInputFp8QuantTraits<T>::FromBits((uint16_t)(gatePacked.x & 0xffffu)),
        FastllmCutlassInputFp8QuantTraits<T>::FromBits((uint16_t)(upPacked.x & 0xffffu)));
    values[1] = FastllmCutlassRoundedSwiglu(
        FastllmCutlassInputFp8QuantTraits<T>::FromBits((uint16_t)(gatePacked.x >> 16)),
        FastllmCutlassInputFp8QuantTraits<T>::FromBits((uint16_t)(upPacked.x >> 16)));
    values[2] = FastllmCutlassRoundedSwiglu(
        FastllmCutlassInputFp8QuantTraits<T>::FromBits((uint16_t)(gatePacked.y & 0xffffu)),
        FastllmCutlassInputFp8QuantTraits<T>::FromBits((uint16_t)(upPacked.y & 0xffffu)));
    values[3] = FastllmCutlassRoundedSwiglu(
        FastllmCutlassInputFp8QuantTraits<T>::FromBits((uint16_t)(gatePacked.y >> 16)),
        FastllmCutlassInputFp8QuantTraits<T>::FromBits((uint16_t)(upPacked.y >> 16)));
}

template <typename T, int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(256) FastllmCutlassSwigluQuantInputFp8PackedWarpKernel(
    const T *__restrict__ gateup, cutlass::float_e4m3_t *__restrict__ quant,
    float *__restrict__ scales, int rows, int cols, int gateupStride, int scaleCols) {
    int warpId = threadIdx.x >> 5;
    int laneId = threadIdx.x & 31;
    int task = blockIdx.x * WARPS_PER_BLOCK + warpId;
    int totalTasks = rows * scaleCols;
    if (task >= totalTasks) {
        return;
    }

    int row = task / scaleCols;
    int group = task - row * scaleCols;
    int base = group * 128;
    size_t gateOffset = (size_t)row * gateupStride + base + laneId * 4;
    size_t blockOffset = (size_t)row * cols + base;

    float values[4];
    FastllmCutlassLoad4RoundedSwiglu(gateup + gateOffset, gateup + gateOffset + cols, values);

    float maxAbs = fmaxf(fmaxf(fabsf(values[0]), fabsf(values[1])),
                         fmaxf(fabsf(values[2]), fabsf(values[3])));
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maxAbs = fmaxf(maxAbs, __shfl_down_sync(0xffffffff, maxAbs, offset));
    }
    maxAbs = __shfl_sync(0xffffffff, maxAbs, 0);

    float scale = fmaxf(maxAbs, 1.0e-10f) * (1.0f / 448.0f);
    if (laneId == 0) {
        scales[(size_t)group * rows + row] = scale;
    }
    float invScale = 1.0f / scale;

    uint32_t packed = FastllmCutlassPackFp8x4(values[0] * invScale, values[1] * invScale,
                                             values[2] * invScale, values[3] * invScale);
    reinterpret_cast<uint32_t *>(quant + blockOffset)[laneId] = packed;
}

static bool FastllmCutlassUseWarpQuant() {
    static const bool enabled = []() {
        const char *env = std::getenv("FASTLLM_CUDA_CUTLASS_LINEAR_FP8_WARP_QUANT");
        if (env == nullptr || env[0] == '\0') {
            return true;
        }
        return std::strcmp(env, "0") != 0 &&
               std::strcmp(env, "false") != 0 && std::strcmp(env, "FALSE") != 0 &&
               std::strcmp(env, "off") != 0 && std::strcmp(env, "OFF") != 0 &&
               std::strcmp(env, "no") != 0 && std::strcmp(env, "NO") != 0;
    }();
    return enabled;
}

static bool FastllmCutlassUseFusedSwigluQuant() {
    static const bool enabled = []() {
        const char *env = std::getenv("FASTLLM_CUDA_CUTLASS_LINEAR_FP8_SWIGLU_QUANT");
        if (env == nullptr || env[0] == '\0') {
            return true;
        }
        return std::strcmp(env, "0") != 0 &&
               std::strcmp(env, "false") != 0 && std::strcmp(env, "FALSE") != 0 &&
               std::strcmp(env, "off") != 0 && std::strcmp(env, "OFF") != 0 &&
               std::strcmp(env, "no") != 0 && std::strcmp(env, "NO") != 0;
    }();
    return enabled;
}

static int FastllmCutlassEnvInt(const char *name, int fallback) {
    const char *v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return fallback;
    }
    char *end = nullptr;
    long value = std::strtol(v, &end, 10);
    if (end == v || value <= 0 || value > 4096) {
        return fallback;
    }
    return (int)value;
}

template <typename T>
__global__ void FastllmCutlassAddFloatBiasKernel(T *output, const float *bias, int rows, int cols) {
    size_t total = (size_t)rows * cols;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (size_t)blockDim.x * gridDim.x) {
        int col = idx % cols;
        float v = FastllmCutlassToFloat(output[idx]) + bias[col];
        output[idx] = T(v);
    }
}

static bool FastllmCutlassEnsureScratch(
    int rows, int cols, cudaStream_t stream, FastllmCutlassFp8Scratch *&scratch) {
    int scaleCols = (cols + 127) / 128;
    size_t inputElems = (size_t)rows * cols;
    size_t scaleElems = (size_t)rows * scaleCols;
    int device = FastllmCudaGetDevice();
    std::lock_guard<std::mutex> guard(g_cutlassScratchMutex);
    auto &deviceScratch = g_cutlassScratchByDevice[device];
    if (FastllmCutlassIsStreamCapturing(stream) &&
        (deviceScratch.inputElems < inputElems || deviceScratch.scaleElems < scaleElems)) {
        return false;
    }
    if (deviceScratch.inputElems < inputElems) {
        if (deviceScratch.input != nullptr) {
            FastllmCudaFree(deviceScratch.input);
        }
        deviceScratch.input = (cutlass::float_e4m3_t*)FastllmCudaMalloc(inputElems);
        deviceScratch.inputElems = inputElems;
    }
    if (deviceScratch.scaleElems < scaleElems) {
        if (deviceScratch.inputScales != nullptr) {
            FastllmCudaFree(deviceScratch.inputScales);
        }
        deviceScratch.inputScales = (float*)FastllmCudaMalloc(scaleElems * sizeof(float));
        deviceScratch.scaleElems = scaleElems;
    }
    scratch = &deviceScratch;
    return scratch->input != nullptr && scratch->inputScales != nullptr;
}

static bool FastllmCutlassEnsureWeightCache(
    fastllm::Data &weight, int inFeatures, int outFeatures,
    cudaStream_t stream, FastllmCutlassFp8WeightCache *&cache) {
    const void *key = weight.cudaData;
    if (key == nullptr || weight.scales.empty() || weight.blockM != 128 || weight.blockK != 128) {
        return false;
    }
    int device = FastllmCudaGetDevice();
    std::lock_guard<std::mutex> guard(g_cutlassWeightMutex);
    auto cacheKey = std::make_pair(device, key);
    auto it = g_cutlassWeightCache.find(cacheKey);
    if (it != g_cutlassWeightCache.end()) {
        auto &entry = it->second;
        if (entry.weightTN != nullptr &&
            entry.inFeatures == inFeatures && entry.outFeatures == outFeatures &&
            entry.blockM == weight.blockM && entry.blockK == weight.blockK &&
            entry.hostScales == weight.scales.data() && entry.scaleCount == weight.scales.size()) {
            cache = &entry;
            return true;
        }
    }
    if (FastllmCutlassIsStreamCapturing(stream)) {
        return false;
    }
    auto &entry = g_cutlassWeightCache[cacheKey];
    if (entry.weightTN != nullptr) {
        FastllmCudaFree(entry.weightTN);
        entry.weightTN = nullptr;
    }
    if (entry.weightScales != nullptr) {
        FastllmCudaFree(entry.weightScales);
        entry.weightScales = nullptr;
    }

    size_t weightBytes = (size_t)inFeatures * outFeatures;
    size_t scaleBytes = weight.scales.size() * sizeof(float);
    entry.weightTN = (cutlass::float_e4m3_t*)FastllmCudaMalloc(weightBytes);
    entry.weightScales = (float*)FastllmCudaMalloc(scaleBytes);
    if (entry.weightTN == nullptr || entry.weightScales == nullptr) {
        return false;
    }
    // CUTLASS reads B as a logical transpose with column-major layout, just like
    // vLLM passes B.T while keeping the original [out][in] physical storage.
    cudaError_t state = cudaMemcpyAsync(entry.weightTN, weight.cudaData, weightBytes,
                                        cudaMemcpyDeviceToDevice, stream);
    if (state != cudaSuccess) {
        return false;
    }
    // CUTLASS SFB layout for B scales matches FastLLM's native
    // [out_block][in_block] row-major order.
    state = cudaMemcpyAsync(entry.weightScales, weight.scales.data(), scaleBytes,
                            cudaMemcpyHostToDevice, stream);
    if (state != cudaSuccess) {
        return false;
    }
    state = cudaStreamSynchronize(stream);
    if (state != cudaSuccess) {
        return false;
    }
    entry.inFeatures = inFeatures;
    entry.outFeatures = outFeatures;
    entry.blockM = weight.blockM;
    entry.blockK = weight.blockK;
    entry.hostScales = weight.scales.data();
    entry.scaleCount = weight.scales.size();
    cache = &entry;
    return true;
}

template <typename Gemm>
static bool FastllmRunCutlassFp8Blockwise(
    cutlass::float_e4m3_t *input, cutlass::float_e4m3_t *weightTN,
    float *inputScales, float *weightScales, typename Gemm::ElementD *output,
    int batch, int outFeatures, int inFeatures, cudaStream_t stream) {
    static constexpr bool swapAB = Gemm::swap_ab;
    using GemmKernel = typename Gemm::GemmKernel;
    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideC = typename GemmKernel::StrideC;
    using StrideD = typename GemmKernel::StrideD;
    using LayoutSFA = typename Gemm::LayoutSFA;
    using LayoutSFB = typename Gemm::LayoutSFB;
    using ScaleConfig = typename Gemm::ScaleConfig;

    StrideA aStride = cutlass::make_cute_packed_stride(
        StrideA{}, cute::make_shape(batch, inFeatures, 1));
    StrideB bStride = cutlass::make_cute_packed_stride(
        StrideB{}, cute::make_shape(outFeatures, inFeatures, 1));
    StrideC cStride = cutlass::make_cute_packed_stride(
        StrideC{}, swapAB ? cute::make_shape(outFeatures, batch, 1)
                          : cute::make_shape(batch, outFeatures, 1));
    StrideD dStride = cutlass::make_cute_packed_stride(
        StrideD{}, swapAB ? cute::make_shape(outFeatures, batch, 1)
                          : cute::make_shape(batch, outFeatures, 1));

    LayoutSFA layoutSFA = swapAB
        ? ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(outFeatures, batch, inFeatures, 1))
        : ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(batch, outFeatures, inFeatures, 1));
    LayoutSFB layoutSFB = swapAB
        ? ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(outFeatures, batch, inFeatures, 1))
        : ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(batch, outFeatures, inFeatures, 1));

    typename GemmKernel::MainloopArguments mainloopArgs{};
    mainloopArgs.layout_SFA = layoutSFA;
    mainloopArgs.layout_SFB = layoutSFB;
    if constexpr (swapAB) {
        mainloopArgs.ptr_A = weightTN;
        mainloopArgs.dA = bStride;
        mainloopArgs.ptr_B = input;
        mainloopArgs.dB = aStride;
        mainloopArgs.ptr_SFA = weightScales;
        mainloopArgs.ptr_SFB = inputScales;
    } else {
        mainloopArgs.ptr_A = input;
        mainloopArgs.dA = aStride;
        mainloopArgs.ptr_B = weightTN;
        mainloopArgs.dB = bStride;
        mainloopArgs.ptr_SFA = inputScales;
        mainloopArgs.ptr_SFB = weightScales;
    }

    auto problemShape = swapAB ? cute::make_shape(outFeatures, batch, inFeatures, 1)
                               : cute::make_shape(batch, outFeatures, inFeatures, 1);
    typename GemmKernel::EpilogueArguments epilogueArgs{{}, output, cStride, output, dStride};
    cutlass::KernelHardwareInfo hwInfo;
    int device = 0;
    cudaGetDevice(&device);
    hwInfo.device_id = device;
    hwInfo.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device);

    typename GemmKernel::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problemShape,
        mainloopArgs,
        epilogueArgs,
        hwInfo};

    using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    GemmOp gemm;
    cutlass::Status status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        return false;
    }
    size_t workspaceBytes = GemmOp::get_workspace_size(args);
    void *workspace = nullptr;
    if (workspaceBytes > 0) {
        if (FastllmCutlassIsStreamCapturing(stream)) {
            return false;
        }
        workspace = FastllmCudaMalloc(workspaceBytes);
        if (workspace == nullptr) {
            return false;
        }
    }
    status = gemm.run(args, workspace, stream);
    if (workspace != nullptr) {
        FastllmCudaFree(workspace);
    }
    return status == cutlass::Status::kSuccess;
}

template <typename OutType>
static bool FastllmDispatchCutlassFp8Blockwise(
    cutlass::float_e4m3_t *input, cutlass::float_e4m3_t *weightTN,
    float *inputScales, float *weightScales, OutType *output,
    int batch, int outFeatures, int inFeatures, cudaStream_t stream) {
    bool swapAB = (batch <= 64) || (batch % 4 != 0);
    if (swapAB) {
        using Gemm = typename FastllmSm120Fp8SwapABConfig<OutType>::Gemm;
        return FastllmRunCutlassFp8Blockwise<Gemm>(
            input, weightTN, inputScales, weightScales, output, batch, outFeatures, inFeatures, stream);
    }
    if (batch <= 256) {
        using Gemm = typename FastllmSm120Fp8PingpongConfig<OutType>::Gemm;
        return FastllmRunCutlassFp8Blockwise<Gemm>(
            input, weightTN, inputScales, weightScales, output, batch, outFeatures, inFeatures, stream);
    }
    using Gemm = typename FastllmSm120Fp8DefaultConfig<OutType>::Gemm;
    return FastllmRunCutlassFp8Blockwise<Gemm>(
        input, weightTN, inputScales, weightScales, output, batch, outFeatures, inFeatures, stream);
}

} // namespace fastllm_cuda_cutlass_fp8

#endif

bool FastllmCudaCutlassLinearFP8E4M3Block128(
    const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias,
    fastllm::Data &output, int n, int m, int k) {
#if defined(FASTLLM_ENABLE_CUTLASS_FP8) && \
    (defined(FASTLLM_CUTLASS_FP8_ENABLE_SM120) || defined(FASTLLM_CUTLASS_FP8_ENABLE_SM121))
    using namespace fastllm_cuda_cutlass_fp8;

    if (n <= 0 || m <= 0 || k <= 0 || (m % 128) != 0 || (k % 128) != 0 ||
        input.cudaData == nullptr || weight.cudaData == nullptr ||
        weight.dataType != fastllm::DataType::FP8_E4M3 ||
        weight.blockM != 128 || weight.blockK != 128 || weight.scales.empty() ||
        output.dataType != input.dataType ||
        (input.dataType != fastllm::DataType::FLOAT16 &&
         input.dataType != fastllm::DataType::BFLOAT16) ||
        (bias.dims.size() > 0 && bias.dataType != fastllm::DataType::FLOAT32)) {
        return false;
    }
    int arch = FastllmCudaRuntimeArch();
    if (!FastllmCutlassFp8CompiledForRuntimeArch(arch)) {
        return false;
    }

    FastllmCutlassFp8Scratch *scratch = nullptr;
    FastllmCutlassFp8WeightCache *cache = nullptr;
    cudaStream_t stream = 0;
    if (!FastllmCutlassEnsureScratch(n, m, stream, scratch) ||
        !FastllmCutlassEnsureWeightCache(weight, m, k, stream, cache)) {
        return false;
    }

    void *inputData = FastllmCudaPrepareInput(input);
    void *outputData = FastllmCudaPrepareOutput(output);
    if (inputData == nullptr || outputData == nullptr) {
        FastllmCudaFinishInput(input, inputData);
        FastllmCudaFinishOutput(output, outputData);
        return false;
    }

    int scaleCols = (m + 127) / 128;
    if (FastllmCutlassUseWarpQuant() && (m % 128) == 0) {
        constexpr int warpQuantWarps = 8;
        int tasks = n * scaleCols;
        dim3 grid((tasks + warpQuantWarps - 1) / warpQuantWarps);
        if (input.dataType == fastllm::DataType::FLOAT16) {
            FastllmCutlassQuantInputFp8PackedWarpKernel<half, warpQuantWarps><<<grid, warpQuantWarps * 32, 0, stream>>>(
                (const half*)inputData, scratch->input, scratch->inputScales, n, m, scaleCols);
        } else {
            FastllmCutlassQuantInputFp8PackedWarpKernel<__nv_bfloat16, warpQuantWarps><<<grid, warpQuantWarps * 32, 0, stream>>>(
                (const __nv_bfloat16*)inputData, scratch->input, scratch->inputScales, n, m, scaleCols);
        }
    } else {
        dim3 grid(n, scaleCols);
        if (input.dataType == fastllm::DataType::FLOAT16) {
            FastllmCutlassQuantInputFp8Kernel<<<grid, 256, 0, stream>>>(
                (const half*)inputData, scratch->input, scratch->inputScales, n, m, scaleCols);
        } else {
            FastllmCutlassQuantInputFp8Kernel<<<grid, 256, 0, stream>>>(
                (const __nv_bfloat16*)inputData, scratch->input, scratch->inputScales, n, m, scaleCols);
        }
    }
    if (cudaGetLastError() != cudaSuccess) {
        FastllmCudaFinishInput(input, inputData);
        FastllmCudaFinishOutput(output, outputData);
        return false;
    }

    bool ok = false;
    if (input.dataType == fastllm::DataType::FLOAT16) {
        ok = FastllmDispatchCutlassFp8Blockwise(
            scratch->input, cache->weightTN, scratch->inputScales, cache->weightScales,
            (cutlass::half_t*)outputData, n, k, m, stream);
        if (ok && bias.dims.size() > 0) {
            int threads = 256;
            int blocks = (int)std::min<size_t>(4096, ((size_t)n * k + threads - 1) / threads);
            FastllmCutlassAddFloatBiasKernel<<<blocks, threads, 0, stream>>>(
                (half*)outputData, (const float*)bias.cudaData, n, k);
            ok = cudaGetLastError() == cudaSuccess;
        }
    } else {
        ok = FastllmDispatchCutlassFp8Blockwise(
            scratch->input, cache->weightTN, scratch->inputScales, cache->weightScales,
            (cutlass::bfloat16_t*)outputData, n, k, m, stream);
        if (ok && bias.dims.size() > 0) {
            int threads = 256;
            int blocks = (int)std::min<size_t>(4096, ((size_t)n * k + threads - 1) / threads);
            FastllmCutlassAddFloatBiasKernel<<<blocks, threads, 0, stream>>>(
                (__nv_bfloat16*)outputData, (const float*)bias.cudaData, n, k);
            ok = cudaGetLastError() == cudaSuccess;
        }
    }

    FastllmCudaFinishInput(input, inputData);
    FastllmCudaFinishOutput(output, outputData);
    return ok;
#else
    (void)input;
    (void)weight;
    (void)bias;
    (void)output;
    (void)n;
    (void)m;
    (void)k;
    return false;
#endif
}

bool FastllmCudaCutlassLinearFP8E4M3Block128FromSwiglu(
    const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias,
    fastllm::Data &output, int n, int m, int k) {
#if defined(FASTLLM_ENABLE_CUTLASS_FP8) && \
    (defined(FASTLLM_CUTLASS_FP8_ENABLE_SM120) || defined(FASTLLM_CUTLASS_FP8_ENABLE_SM121))
    using namespace fastllm_cuda_cutlass_fp8;

    if (!FastllmCutlassUseFusedSwigluQuant() || !FastllmCutlassUseWarpQuant()) {
        return false;
    }
    int minBatch = FastllmCutlassEnvInt("FASTLLM_CUDA_CUTLASS_LINEAR_FP8_MIN_BATCH", 7);
    if (n < minBatch) {
        return false;
    }
    int maxBatch = FastllmCutlassEnvInt("FASTLLM_CUDA_CUTLASS_LINEAR_FP8_MAX_BATCH", 0);
    if (maxBatch > 0 && n > maxBatch) {
        return false;
    }
    if (n <= 0 || m <= 0 || k <= 0 || (m % 128) != 0 || (k % 128) != 0 ||
        input.cudaData == nullptr || weight.cudaData == nullptr ||
        input.dims.empty() || input.dims.back() != m * 2 ||
        weight.dims.size() != 2 || weight.dims[0] != k || weight.dims[1] != m ||
        weight.dataType != fastllm::DataType::FP8_E4M3 ||
        weight.blockM != 128 || weight.blockK != 128 || weight.scales.empty() ||
        output.dataType != input.dataType ||
        (input.dataType != fastllm::DataType::FLOAT16 &&
         input.dataType != fastllm::DataType::BFLOAT16) ||
        (bias.dims.size() > 0 && bias.dataType != fastllm::DataType::FLOAT32)) {
        return false;
    }
    int arch = FastllmCudaRuntimeArch();
    if (!FastllmCutlassFp8CompiledForRuntimeArch(arch)) {
        return false;
    }

    FastllmCutlassFp8Scratch *scratch = nullptr;
    FastllmCutlassFp8WeightCache *cache = nullptr;
    cudaStream_t stream = 0;
    if (!FastllmCutlassEnsureScratch(n, m, stream, scratch) ||
        !FastllmCutlassEnsureWeightCache(weight, m, k, stream, cache)) {
        return false;
    }

    void *inputData = FastllmCudaPrepareInput(input);
    void *outputData = FastllmCudaPrepareOutput(output);
    if (inputData == nullptr || outputData == nullptr) {
        FastllmCudaFinishInput(input, inputData);
        FastllmCudaFinishOutput(output, outputData);
        return false;
    }

    constexpr int warpQuantWarps = 8;
    int scaleCols = (m + 127) / 128;
    int tasks = n * scaleCols;
    dim3 grid((tasks + warpQuantWarps - 1) / warpQuantWarps);
    if (input.dataType == fastllm::DataType::FLOAT16) {
        FastllmCutlassSwigluQuantInputFp8PackedWarpKernel<half, warpQuantWarps><<<grid, warpQuantWarps * 32, 0, stream>>>(
            (const half*)inputData, scratch->input, scratch->inputScales, n, m, m * 2, scaleCols);
    } else {
        FastllmCutlassSwigluQuantInputFp8PackedWarpKernel<__nv_bfloat16, warpQuantWarps><<<grid, warpQuantWarps * 32, 0, stream>>>(
            (const __nv_bfloat16*)inputData, scratch->input, scratch->inputScales, n, m, m * 2, scaleCols);
    }
    if (cudaGetLastError() != cudaSuccess) {
        FastllmCudaFinishInput(input, inputData);
        FastllmCudaFinishOutput(output, outputData);
        return false;
    }

    bool ok = false;
    if (input.dataType == fastllm::DataType::FLOAT16) {
        ok = FastllmDispatchCutlassFp8Blockwise(
            scratch->input, cache->weightTN, scratch->inputScales, cache->weightScales,
            (cutlass::half_t*)outputData, n, k, m, stream);
        if (ok && bias.dims.size() > 0) {
            int threads = 256;
            int blocks = (int)std::min<size_t>(4096, ((size_t)n * k + threads - 1) / threads);
            FastllmCutlassAddFloatBiasKernel<<<blocks, threads, 0, stream>>>(
                (half*)outputData, (const float*)bias.cudaData, n, k);
            ok = cudaGetLastError() == cudaSuccess;
        }
    } else {
        ok = FastllmDispatchCutlassFp8Blockwise(
            scratch->input, cache->weightTN, scratch->inputScales, cache->weightScales,
            (cutlass::bfloat16_t*)outputData, n, k, m, stream);
        if (ok && bias.dims.size() > 0) {
            int threads = 256;
            int blocks = (int)std::min<size_t>(4096, ((size_t)n * k + threads - 1) / threads);
            FastllmCutlassAddFloatBiasKernel<<<blocks, threads, 0, stream>>>(
                (__nv_bfloat16*)outputData, (const float*)bias.cudaData, n, k);
            ok = cudaGetLastError() == cudaSuccess;
        }
    }

    FastllmCudaFinishInput(input, inputData);
    FastllmCudaFinishOutput(output, outputData);
    return ok;
#else
    (void)input;
    (void)weight;
    (void)bias;
    (void)output;
    (void)n;
    (void)m;
    (void)k;
    return false;
#endif
}
