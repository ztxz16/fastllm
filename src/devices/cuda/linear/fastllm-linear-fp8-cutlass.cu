// Blockwise FP8 W8A8 CUTLASS实现。激活使用(1,128)动态scale，权重
// 使用(128,128)静态scale；标准per-token/per-channel W8A8位于
// fastllm-linear-w8a8-cutlass.cu。
#include "fastllm-cuda.cuh"
#include "fastllm.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(FASTLLM_ENABLE_CUTLASS_FP8) && \
    (defined(FASTLLM_CUTLASS_FP8_ENABLE_SM90) || \
     defined(FASTLLM_CUTLASS_FP8_ENABLE_SM120) || \
     defined(FASTLLM_CUTLASS_FP8_ENABLE_SM121))

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"

namespace fastllm_w8a8_blockwise_detail {

template <typename Kernel>
struct FastllmEnableSm90Only : Kernel {
    template <typename... Args>
    CUTLASS_DEVICE void operator()(Args&&... args) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)
        Kernel::operator()(std::forward<Args>(args)...);
#elif defined(__CUDA_ARCH__)
        asm("trap;");
#endif
    }
};

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

template <class T>
struct FastllmCutlassIdentity {
    CUTLASS_HOST_DEVICE T operator()(T const &value) const {
        return value;
    }
};

template <class OutType>
using FastllmCutlassExactResidualCallbacks =
    cutlass::epilogue::fusion::Sm90EVT<
        cutlass::epilogue::fusion::Sm90Compute<
            cutlass::plus, OutType, OutType,
            cutlass::FloatRoundStyle::round_to_nearest>,
        cutlass::epilogue::fusion::Sm90EVT<
            cutlass::epilogue::fusion::Sm90Compute<
                FastllmCutlassIdentity, OutType, float,
                cutlass::FloatRoundStyle::round_to_nearest>,
            cutlass::epilogue::fusion::Sm90AccFetch>,
        cutlass::epilogue::fusion::Sm90SrcFetch<OutType>>;

#if defined(FASTLLM_CUTLASS_FP8_ENABLE_SM120) || defined(FASTLLM_CUTLASS_FP8_ENABLE_SM121)
template <class OutType, int ScaleGranularityM, int ScaleGranularityN, int ScaleGranularityK,
          class MmaTileShape, class ClusterShape, class EpilogueScheduler,
          class MainloopScheduler, bool SwapAB = false, bool ExactResidual = false>
struct FastllmCutlassFp8BlockwiseGemm {
    static constexpr bool swap_ab = SwapAB;
    static constexpr bool exact_residual = ExactResidual;
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

    using ElementC = std::conditional_t<ExactResidual, OutType, void>;
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
    using DefaultOperation = std::conditional_t<
        ExactResidual,
        FastllmCutlassExactResidualCallbacks<OutType>,
        cutlass::epilogue::fusion::LinearCombination<
            ElementD, ElementCompute, ElementC, ElementScalar, RoundStyle>>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, MmaTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
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
#endif

#if defined(FASTLLM_CUTLASS_FP8_ENABLE_SM90)
template <class OutType, bool SwapAB = false>
struct FastllmCutlassSm90Fp8BlockwiseGemm {
    static constexpr bool swap_ab = SwapAB;
    using ElementAB = cutlass::float_e4m3_t;
    using ElementA = ElementAB;
    using ElementB = ElementAB;
    using ElementD = OutType;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementBlockScale = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutD = cutlass::layout::RowMajor;
    using LayoutAT = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
    using LayoutBT = typename cutlass::layout::LayoutTranspose<LayoutB>::type;
    using LayoutDT = typename cutlass::layout::LayoutTranspose<LayoutD>::type;
    static constexpr int AlignmentA = 16;
    static constexpr int AlignmentB = 16;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
    using TileShape = std::conditional_t<SwapAB, Shape<_128, _16, _128>, Shape<_128, _128, _128>>;
    using ClusterShape = std::conditional_t<SwapAB, Shape<_1, _1, _1>, Shape<_1, _2, _1>>;
    using ScaleConfig = std::conditional_t<SwapAB,
        cutlass::detail::Sm90BlockwiseScaleConfig<128, 1, 128, GMMA::Major::K, GMMA::Major::MN>,
        cutlass::detail::Sm90BlockwiseScaleConfig<1, 128, 128, GMMA::Major::MN, GMMA::Major::K>>;
    using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
    using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
    using EpilogueSchedule = std::conditional_t<SwapAB,
        cutlass::epilogue::TmaWarpSpecialized,
        cutlass::epilogue::TmaWarpSpecializedCooperative>;
    using MainloopSchedule = std::conditional_t<SwapAB,
        cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8BlockScaledAccum,
        cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum>;
    using Operation = cutlass::epilogue::fusion::LinearCombination<
        ElementD, ElementCompute, void, float, cutlass::FloatRoundStyle::round_to_nearest>;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
        ElementCompute, void, std::conditional_t<SwapAB, LayoutDT, LayoutD>, AlignmentD,
        ElementD, std::conditional_t<SwapAB, LayoutDT, LayoutD>, AlignmentD,
        EpilogueSchedule, Operation>::CollectiveOp;
    using CollectiveMainloop = std::conditional_t<SwapAB,
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
            ElementB, cute::tuple<LayoutBT, LayoutSFA>, AlignmentB,
            ElementA, cute::tuple<LayoutAT, LayoutSFB>, AlignmentA,
            ElementAccumulator, TileShape, ClusterShape,
            cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
            MainloopSchedule>::CollectiveOp,
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
            ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
            ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
            ElementAccumulator, TileShape, ClusterShape,
            cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
            MainloopSchedule>::CollectiveOp>;
    using KernelType = FastllmEnableSm90Only<cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>>;
    struct GemmKernel : KernelType {};
};
#endif

#if defined(FASTLLM_CUTLASS_FP8_ENABLE_SM120) || defined(FASTLLM_CUTLASS_FP8_ENABLE_SM121)
template <typename OutType, bool ExactResidual = false>
struct FastllmSm120Fp8DefaultConfig {
    using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
    using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
    using TileShape = Shape<_128, _128, _128>;
    using ClusterShape = Shape<_1, _1, _1>;
    using Gemm = FastllmCutlassFp8BlockwiseGemm<
        OutType, 1, 128, 128, TileShape, ClusterShape, EpilogueSchedule,
        KernelSchedule, false, ExactResidual>;
};

template <typename OutType, bool ExactResidual = false>
struct FastllmSm120Fp8PingpongConfig {
    using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120;
    using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
    using TileShape = Shape<_64, _128, _128>;
    using ClusterShape = Shape<_1, _1, _1>;
    using Gemm = FastllmCutlassFp8BlockwiseGemm<
        OutType, 1, 128, 128, TileShape, ClusterShape, EpilogueSchedule,
        KernelSchedule, false, ExactResidual>;
};

template <typename OutType, bool ExactResidual = false>
struct FastllmSm120Fp8SwapABConfig {
    using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedBlockwiseCooperativeSm120;
    using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
    using TileShape = Shape<_128, _32, _128>;
    using ClusterShape = Shape<_1, _1, _1>;
    using Gemm = FastllmCutlassFp8BlockwiseGemm<
        OutType, 128, 1, 128, TileShape, ClusterShape, EpilogueSchedule,
        KernelSchedule, true, ExactResidual>;
};
#endif

struct FastllmCutlassFp8Scratch {
    cutlass::float_e4m3_t *input = nullptr;
    float *inputScales = nullptr;
    void *workspace = nullptr;
    size_t inputElems = 0;
    size_t scaleElems = 0;
    size_t workspaceBytes = 0;
    // CUDA graphs retain the exact scratch addresses observed during capture.
    // Qwen3.5 pre-captures increasing batch sizes, so growing this buffer must
    // not free storage that an earlier graph still references.
    std::vector<cutlass::float_e4m3_t*> retiredInputs;
    std::vector<float*> retiredInputScales;
    std::vector<void*> retiredWorkspaces;
};

struct FastllmCutlassFp8WeightCache {
    cutlass::float_e4m3_t *weightTN = nullptr;
    float *weightScales = nullptr;
    const float *hostScales = nullptr;
    bool ownsWeightTN = false;
    size_t scaleCount = 0;
    int inFeatures = 0;
    int outFeatures = 0;
    int blockM = 0;
    int blockK = 0;
    const void *weightSource = nullptr;
};

static std::mutex g_cutlassScratchMutex;
static std::map<int, FastllmCutlassFp8Scratch> g_cutlassScratchByDevice;
static std::mutex g_cutlassWeightMutex;
static std::map<std::pair<const fastllm::Data *, int>,
                FastllmCutlassFp8WeightCache> g_cutlassWeightCache;
static std::mutex g_cutlassBackendMutex;
static std::map<std::pair<const fastllm::Data *, int>,
                FastllmCudaFp8W8A8BackendState> g_cutlassBackendStates;

static bool FastllmCutlassBlockwiseEnabled() {
    const char *value = std::getenv("FASTLLM_CUDA_W8A8");
    return value == nullptr || value[0] == '\0' || value[0] != '0';
}

static bool FastllmCutlassBlockwiseStrictEnabled() {
    const char *value = std::getenv("FASTLLM_CUDA_W8A8_STRICT");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static bool FastllmCutlassBlockwiseTraceEnabled() {
    const char *value = std::getenv("FASTLLM_CUDA_W8A8_TRACE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static bool FastllmCutlassBlockwiseForceFailureForTest() {
    const char *value = std::getenv(
        "FASTLLM_CUDA_W8A8_TEST_FORCE_GEMM_FAILURE");
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

static void FastllmCutlassBlockwiseTrace(
    const char *reason, int m, int n, int k) {
    if (!FastllmCutlassBlockwiseTraceEnabled()) return;

    std::fprintf(
        stderr,
        "[fastllm][w8a8] path=w8a8-block128-cutlass "
        "reason=%s m=%d n=%d k=%d sm=%d\n",
        reason, m, n, k, FastllmCudaRuntimeArch());
}

static FastllmCudaFp8W8A8BackendState FastllmCutlassGetBackendState(
    const fastllm::Data &weight, int device) {
    std::lock_guard<std::mutex> guard(g_cutlassBackendMutex);
    auto found = g_cutlassBackendStates.find({&weight, device});
    return found == g_cutlassBackendStates.end()
        ? FastllmCudaFp8W8A8BackendState::Uninitialized
        : found->second;
}

static void FastllmCutlassSetBackendState(
    const fastllm::Data &weight, int device,
    FastllmCudaFp8W8A8BackendState state) {
    std::lock_guard<std::mutex> guard(g_cutlassBackendMutex);
    g_cutlassBackendStates[{&weight, device}] = state;
}

static bool FastllmCutlassIsStreamCapturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    cudaError_t state = cudaStreamIsCapturing(stream, &status);
    return state == cudaSuccess && status != cudaStreamCaptureStatusNone;
}

static bool FastllmCutlassFp8CompiledForRuntimeArch(int arch) {
    switch (arch) {
#if defined(FASTLLM_CUTLASS_FP8_ENABLE_SM90)
    case 90:
        return true;
#endif
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

// Fuse Qwen3.5 GDN's exact 128-wide output RMSNorm, FP16 SiLU gate and
// blockwise FP8 activation quantization.  The reduction and FP16 rounding
// deliberately match FastllmRMSNormSiluMulHalf128HeadMajorCombinedGateExactKernel
// followed by FastllmCutlassQuantInputFp8PackedWarpKernel.
template <int WARPS_PER_BLOCK>
__global__ void __launch_bounds__(256)
FastllmCutlassGdnOutputGateQuantInputFp8Half128HeadMajorKernel(
    const half *__restrict__ headMajorInput,
    const float *__restrict__ normWeight,
    const half *__restrict__ combinedGateInput,
    cutlass::float_e4m3_t *__restrict__ quant,
    float *__restrict__ scales,
    int rows, int seqLen, int paddedSeqLen,
    int gateStride, int gateOffset, int gateHeads, float eps) {
    constexpr int channels = 128;
    int warpId = threadIdx.x >> 5;
    int laneId = threadIdx.x & 31;
    int task = blockIdx.x * WARPS_PER_BLOCK + warpId;
    int totalTasks = rows * gateHeads;
    if (task >= totalTasks) {
        return;
    }

    int row = task / gateHeads;
    int head = task - row * gateHeads;
    int batch = row / seqLen;
    int tokenInBatch = row - batch * seqLen;
    const half *input =
        headMajorInput +
        (((size_t)batch * gateHeads + head) * paddedSeqLen +
         tokenInBatch) * channels;
    const half *gate =
        combinedGateInput + (size_t)row * gateStride +
        gateOffset + head * channels;

    // Preserve the old kernel's two independent 64-value reductions and the
    // final lane-0 addition.  This is needed for bitwise-stable FP16 output.
    const half2 *input2 = reinterpret_cast<const half2 *>(input);
    float2 value0 = __half22float2(input2[laneId]);
    float2 value1 = __half22float2(input2[laneId + 32]);
    float sum0 = value0.x * value0.x + value0.y * value0.y;
    float sum1 = value1.x * value1.x + value1.y * value1.y;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down_sync(0xffffffffu, sum0, offset);
        sum1 += __shfl_down_sync(0xffffffffu, sum1, offset);
    }
    float rowScale = 0.0f;
    if (laneId == 0) {
        rowScale = rsqrtf((sum0 + sum1) / channels + eps);
    }
    rowScale = __shfl_sync(0xffffffffu, rowScale, 0);

    int col = laneId * 4;
    uint2 packedInput =
        *reinterpret_cast<const uint2 *>(input + col);
    uint2 packedGate =
        *reinterpret_cast<const uint2 *>(gate + col);
    half inputValues[4] = {
        __ushort_as_half((uint16_t)(packedInput.x & 0xffffu)),
        __ushort_as_half((uint16_t)(packedInput.x >> 16)),
        __ushort_as_half((uint16_t)(packedInput.y & 0xffffu)),
        __ushort_as_half((uint16_t)(packedInput.y >> 16)),
    };
    half gateValues[4] = {
        __ushort_as_half((uint16_t)(packedGate.x & 0xffffu)),
        __ushort_as_half((uint16_t)(packedGate.x >> 16)),
        __ushort_as_half((uint16_t)(packedGate.y & 0xffffu)),
        __ushort_as_half((uint16_t)(packedGate.y >> 16)),
    };
    float values[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
#ifdef CUDA_NO_TENSOR_CORE
        float gateFloat = __half2float(gateValues[item]);
        half siluGate = __float2half(
            gateFloat / (1.0f + expf(-gateFloat)));
#else
        half siluGate = __hdiv(
            gateValues[item],
            __hadd(__float2half(1.0f), hexp(-gateValues[item])));
#endif
        half rms = __float2half_rn(
            __half2float(inputValues[item]) * rowScale *
            __ldg(normWeight + col + item));
#ifdef CUDA_NO_TENSOR_CORE
        half rounded = __float2half(
            __half2float(rms) * __half2float(siluGate));
#else
        half rounded = __hmul(rms, siluGate);
#endif
        values[item] = __half2float(rounded);
    }

    float maxAbs =
        fmaxf(fmaxf(fabsf(values[0]), fabsf(values[1])),
              fmaxf(fabsf(values[2]), fabsf(values[3])));
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maxAbs = fmaxf(
            maxAbs,
            __shfl_down_sync(0xffffffffu, maxAbs, offset));
    }
    maxAbs = __shfl_sync(0xffffffffu, maxAbs, 0);
    float scale =
        fmaxf(maxAbs, 1.0e-10f) * (1.0f / 448.0f);
    if (laneId == 0) {
        scales[(size_t)head * rows + row] = scale;
    }
    float invScale = 1.0f / scale;
    size_t blockOffset =
        (size_t)row * gateHeads * channels +
        head * channels;
    uint32_t packed = FastllmCutlassPackFp8x4(
        values[0] * invScale, values[1] * invScale,
        values[2] * invScale, values[3] * invScale);
    reinterpret_cast<uint32_t *>(
        quant + blockOffset)[laneId] = packed;
}

// Fuse the exact 5120-wide RMSNorm used by Qwen3.5/3.6 with the following
// blockwise FP8 activation quantization.  The first phase deliberately
// preserves FastllmRMSNormHalfVirtual1024Kernel's virtual-thread reduction
// order.  The quantization phase rounds each normalized value through FP16
// before computing the per-128 scale, matching the materialized RMSNorm
// output consumed by FastllmCutlassQuantInputFp8PackedWarpKernel.  Some GDN
// projections also need that FP16 value for a second, non-FP8 projection;
// MaterializeNorm stores the same rounded value without another input pass.
template <bool MaterializeNorm>
__global__ __launch_bounds__(512)
void FastllmCutlassRMSNormQuantInputFp8Half5120Kernel(
    const half *__restrict__ input, const float *__restrict__ normWeight,
    half *__restrict__ normOutput,
    cutlass::float_e4m3_t *__restrict__ quant,
    float *__restrict__ scales, int rows, float eps) {
    constexpr int kChannels = 5120;
    constexpr int kScaleCols = kChannels / 128;
    constexpr int kPhysicalThreads = 512;
    constexpr int kVirtualThreads = 1024;
    constexpr int kPhysicalWarps = kPhysicalThreads / 32;
    __shared__ float warpSums[kVirtualThreads / 32];
    __shared__ float rowScaleShared;

    int row = blockIdx.x;
    input += (size_t)row * kChannels;
    quant += (size_t)row * kChannels;
    if constexpr (MaterializeNorm) {
        normOutput += (size_t)row * kChannels;
    }
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    constexpr int kHalf2Channels = kChannels / 2;
    const half2 *input2 = reinterpret_cast<const half2 *>(input);

    float sum0 = 0.0f;
    for (int i = tid; i < kHalf2Channels; i += kVirtualThreads) {
        float2 value = __half22float2(input2[i]);
        sum0 += value.x * value.x + value.y * value.y;
    }
    float sum1 = 0.0f;
    for (int i = tid + kPhysicalThreads;
         i < kHalf2Channels; i += kVirtualThreads) {
        float2 value = __half22float2(input2[i]);
        sum1 += value.x * value.x + value.y * value.y;
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down_sync(0xffffffffu, sum0, offset);
        sum1 += __shfl_down_sync(0xffffffffu, sum1, offset);
    }
    if (lane == 0) {
        warpSums[warp] = sum0;
        warpSums[warp + kPhysicalWarps] = sum1;
    }
    __syncthreads();

    if (warp == 0) {
        float value = warpSums[lane];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if (lane == 0) {
            rowScaleShared = rsqrtf(value / kChannels + eps);
        }
    }
    __syncthreads();

    float rowScale = rowScaleShared;
    for (int group = warp; group < kScaleCols;
         group += kPhysicalWarps) {
        int col = group * 128 + lane * 4;
        uint2 packedInput =
            *reinterpret_cast<const uint2 *>(input + col);
        half inputValues[4] = {
            __ushort_as_half((uint16_t)(packedInput.x & 0xffffu)),
            __ushort_as_half((uint16_t)(packedInput.x >> 16)),
            __ushort_as_half((uint16_t)(packedInput.y & 0xffffu)),
            __ushort_as_half((uint16_t)(packedInput.y >> 16)),
        };
        half roundedValues[4];
        float values[4];
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            roundedValues[item] = __float2half_rn(
                __half2float(inputValues[item]) * rowScale *
                __ldg(normWeight + col + item));
            values[item] = __half2float(roundedValues[item]);
        }
        if constexpr (MaterializeNorm) {
            uint2 packedNorm;
            packedNorm.x =
                (uint32_t)__half_as_ushort(roundedValues[0]) |
                ((uint32_t)__half_as_ushort(roundedValues[1]) << 16);
            packedNorm.y =
                (uint32_t)__half_as_ushort(roundedValues[2]) |
                ((uint32_t)__half_as_ushort(roundedValues[3]) << 16);
            *reinterpret_cast<uint2 *>(normOutput + col) = packedNorm;
        }

        float maxAbs =
            fmaxf(fmaxf(fabsf(values[0]), fabsf(values[1])),
                  fmaxf(fabsf(values[2]), fabsf(values[3])));
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            maxAbs = fmaxf(
                maxAbs,
                __shfl_down_sync(0xffffffffu, maxAbs, offset));
        }
        maxAbs = __shfl_sync(0xffffffffu, maxAbs, 0);
        float scale =
            fmaxf(maxAbs, 1.0e-10f) * (1.0f / 448.0f);
        if (lane == 0) {
            scales[(size_t)group * rows + row] = scale;
        }
        float invScale = 1.0f / scale;
        uint32_t packed = FastllmCutlassPackFp8x4(
            values[0] * invScale, values[1] * invScale,
            values[2] * invScale, values[3] * invScale);
        reinterpret_cast<uint32_t *>(quant + group * 128)[lane] =
            packed;
    }
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
        cutlass::float_e4m3_t *newInput =
            (cutlass::float_e4m3_t*)FastllmCudaMalloc(inputElems);
        if (newInput == nullptr) {
            return false;
        }
        if (deviceScratch.input != nullptr) {
            deviceScratch.retiredInputs.push_back(deviceScratch.input);
        }
        deviceScratch.input = newInput;
        deviceScratch.inputElems = inputElems;
    }
    if (deviceScratch.scaleElems < scaleElems) {
        float *newInputScales =
            (float*)FastllmCudaMalloc(scaleElems * sizeof(float));
        if (newInputScales == nullptr) {
            return false;
        }
        if (deviceScratch.inputScales != nullptr) {
            deviceScratch.retiredInputScales.push_back(deviceScratch.inputScales);
        }
        deviceScratch.inputScales = newInputScales;
        deviceScratch.scaleElems = scaleElems;
    }
    scratch = &deviceScratch;
    return scratch->input != nullptr && scratch->inputScales != nullptr;
}

static void *FastllmCutlassGetWorkspace(
    size_t bytes, cudaStream_t stream) {
    if (bytes == 0) return nullptr;

    const int device = FastllmCudaGetDevice();
    std::lock_guard<std::mutex> guard(g_cutlassScratchMutex);
    auto &scratch = g_cutlassScratchByDevice[device];

    if (scratch.workspace != nullptr && scratch.workspaceBytes >= bytes) {
        return scratch.workspace;
    }

    if (FastllmCutlassIsStreamCapturing(stream)) return nullptr;

    void *replacement = FastllmCudaMalloc(bytes);
    if (replacement == nullptr) return nullptr;

    if (scratch.workspace != nullptr) {
        // 已捕获的CUDA Graph可能仍引用旧workspace。扩容时保留旧地址，
        // 避免后续较大形状破坏先前Batch对应的Graph。
        scratch.retiredWorkspaces.push_back(scratch.workspace);
    }
    scratch.workspace = replacement;
    scratch.workspaceBytes = bytes;
    return scratch.workspace;
}

static bool FastllmCutlassEnsureWeightCache(
    fastllm::Data &weight, int inFeatures, int outFeatures,
    cudaStream_t stream, FastllmCutlassFp8WeightCache *&cache) {
    // FP8 Marlin warmup repacks cudaData in place. CUTLASS aliases cudaData
    // as row-major FP8, so accepting that layout silently produces bad logits.
    if (FastllmCudaHasFp8MarlinLayout(weight)) {
        return false;
    }
    if (weight.cudaData == nullptr || weight.scales.empty() ||
        weight.blockM != 128 || weight.blockK != 128) {
        return false;
    }
    int device = FastllmCudaGetDevice();
    std::lock_guard<std::mutex> guard(g_cutlassWeightMutex);
    auto cacheKey = std::make_pair(&weight, device);
    auto it = g_cutlassWeightCache.find(cacheKey);
    if (it != g_cutlassWeightCache.end()) {
        auto &entry = it->second;
        if (entry.weightTN != nullptr &&
            entry.weightSource == weight.cudaData &&
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
    if (entry.weightTN != nullptr && entry.ownsWeightTN) {
        FastllmCudaFree(entry.weightTN);
    }
    entry.weightTN = nullptr;
    entry.ownsWeightTN = false;
    if (entry.weightScales != nullptr) {
        FastllmCudaFree(entry.weightScales);
        entry.weightScales = nullptr;
    }

    size_t scaleBytes = weight.scales.size() * sizeof(float);
    // CUTLASS consumes FastLLM's native packed [out][in] FP8 bytes directly.
    // Keep an alias here; allocating another weightTN copy doubles FP8 weight VRAM.
    entry.weightTN = (cutlass::float_e4m3_t*)weight.cudaData;
    entry.weightSource = weight.cudaData;
    entry.ownsWeightTN = false;
    entry.weightScales = (float*)FastllmCudaMalloc(scaleBytes);
    if (entry.weightTN == nullptr || entry.weightScales == nullptr) {
        return false;
    }
    // CUTLASS SFB layout for B scales matches FastLLM's native
    // [out_block][in_block] row-major order.
    cudaError_t state = cudaMemcpyAsync(entry.weightScales, weight.scales.data(), scaleBytes,
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

static void FastllmCutlassReleaseWeightCache(
    FastllmCutlassFp8WeightCache &cache) {
    if (cache.weightTN != nullptr && cache.ownsWeightTN) {
        FastllmCudaFree(cache.weightTN);
    }
    if (cache.weightScales != nullptr) {
        FastllmCudaFree(cache.weightScales);
    }
    cache = FastllmCutlassFp8WeightCache();
}

static void FastllmCutlassReleaseWeightCacheForDevice(
    const fastllm::Data *weight, int device) {
    std::lock_guard<std::mutex> guard(g_cutlassWeightMutex);
    auto found = g_cutlassWeightCache.find({weight, device});
    if (found == g_cutlassWeightCache.end()) return;

    const int originalDevice = FastllmCudaGetDevice();
    FastllmCudaSetDevice(device);
    FastllmCutlassReleaseWeightCache(found->second);
    g_cutlassWeightCache.erase(found);
    FastllmCudaSetDevice(originalDevice);
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
    void *workspace = FastllmCutlassGetWorkspace(workspaceBytes, stream);
    if (workspaceBytes > 0 && workspace == nullptr) return false;

    status = gemm.run(args, workspace, stream);
    return status == cutlass::Status::kSuccess;
}

template <bool ExactResidual = false, typename OutType>
static bool FastllmDispatchCutlassFp8Blockwise(
    cutlass::float_e4m3_t *input, cutlass::float_e4m3_t *weightTN,
    float *inputScales, float *weightScales, OutType *output,
    int batch, int outFeatures, int inFeatures, cudaStream_t stream) {
    const int arch = FastllmCudaRuntimeArch();
#if defined(FASTLLM_CUTLASS_FP8_ENABLE_SM90)
    if (arch == 90) {
        if ((batch % 4) != 0) {
            using Gemm = FastllmCutlassSm90Fp8BlockwiseGemm<OutType, true>;
            return FastllmRunCutlassFp8Blockwise<Gemm>(
                input, weightTN, inputScales, weightScales, output,
                batch, outFeatures, inFeatures, stream);
        }
        using Gemm = FastllmCutlassSm90Fp8BlockwiseGemm<OutType, false>;
        return FastllmRunCutlassFp8Blockwise<Gemm>(
            input, weightTN, inputScales, weightScales, output,
            batch, outFeatures, inFeatures, stream);
    }
#endif
#if defined(FASTLLM_CUTLASS_FP8_ENABLE_SM120) || defined(FASTLLM_CUTLASS_FP8_ENABLE_SM121)
    // 忠实采用vLLM SM120 blockwise分派：小M或M不能被4整除时交换A/B；
    // 其余M<=256使用Pingpong，大M使用默认Cooperative配置。
    const bool swapAB = batch <= 64 || batch % 4 != 0;
    if (swapAB) {
        using Gemm =
            typename FastllmSm120Fp8SwapABConfig<OutType, ExactResidual>::Gemm;
        return FastllmRunCutlassFp8Blockwise<Gemm>(
            input, weightTN, inputScales, weightScales, output, batch, outFeatures, inFeatures, stream);
    }
    if (batch <= 256) {
        using Gemm =
            typename FastllmSm120Fp8PingpongConfig<OutType, ExactResidual>::Gemm;
        return FastllmRunCutlassFp8Blockwise<Gemm>(
            input, weightTN, inputScales, weightScales, output,
            batch, outFeatures, inFeatures, stream);
    }

    using Gemm = typename FastllmSm120Fp8DefaultConfig<
        OutType, ExactResidual>::Gemm;
    return FastllmRunCutlassFp8Blockwise<Gemm>(
        input, weightTN, inputScales, weightScales, output, batch, outFeatures, inFeatures, stream);
#else
    (void)input; (void)weightTN; (void)inputScales; (void)weightScales;
    (void)output; (void)batch; (void)outFeatures; (void)inFeatures;
    (void)arch; (void)stream;
    return false;
#endif
}

} // namespace fastllm_w8a8_blockwise_detail

#endif

static bool FastllmCudaCutlassLinearFP8E4M3Block128Impl(
    const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias,
    fastllm::Data &output, int n, int m, int k, bool exactResidual) {
#if defined(FASTLLM_ENABLE_CUTLASS_FP8) && \
    (defined(FASTLLM_CUTLASS_FP8_ENABLE_SM90) || defined(FASTLLM_CUTLASS_FP8_ENABLE_SM120) || defined(FASTLLM_CUTLASS_FP8_ENABLE_SM121))
    using namespace fastllm_w8a8_blockwise_detail;

    if (n <= 0 || m <= 0 || k <= 0 || (m % 128) != 0 || (k % 128) != 0 ||
        input.cudaData == nullptr || weight.cudaData == nullptr || output.cudaData == nullptr ||
        weight.dataType != fastllm::DataType::FP8_E4M3 ||
        weight.dims != std::vector<int>({k, m}) ||
        weight.blockM != 128 || weight.blockK != 128 ||
        weight.scales.size() != (size_t)(k / 128) * (m / 128) ||
        output.dataType != input.dataType ||
        (input.dataType != fastllm::DataType::FLOAT16 &&
         input.dataType != fastllm::DataType::BFLOAT16) ||
        (exactResidual && (input.dataType != fastllm::DataType::FLOAT16 ||
                           !bias.dims.empty())) ||
        (bias.dims.size() > 0 &&
         (bias.dataType != fastllm::DataType::FLOAT32 ||
          bias.cudaData == nullptr || bias.Count(0) != k))) {
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
        if (exactResidual) {
            ok = FastllmDispatchCutlassFp8Blockwise<true>(
                scratch->input, cache->weightTN, scratch->inputScales, cache->weightScales,
                (cutlass::half_t*)outputData, n, k, m, stream);
        } else {
            ok = FastllmDispatchCutlassFp8Blockwise(
                scratch->input, cache->weightTN, scratch->inputScales, cache->weightScales,
                (cutlass::half_t*)outputData, n, k, m, stream);
        }
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
    (void)exactResidual;
    return false;
#endif
}

bool FastllmCudaCutlassLinearFP8E4M3Block128(
    const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias,
    fastllm::Data &output, int n, int m, int k) {
    return FastllmCudaCutlassLinearFP8E4M3Block128Impl(
        input, weight, bias, output, n, m, k, false);
}

/**
 * 清除指定Blockwise FP8权重的CUTLASS后端状态与设备scale缓存。
 *
 * 权重本体由Data管理，本函数只释放按“Data对象、GPU”保存的设备scale
 * 和后端选择状态。张量覆盖、迁移GPU或析构时由统一FP8生命周期入口
 * 调用，避免新权重继承旧地址对应的Cutlass/Rejected状态。
 *
 * @param weight 即将覆盖、迁移或销毁的FP8权重；nullptr表示无需处理。
 */
void FastllmCudaReleaseFp8W8A8Block128BackendState(
        const fastllm::Data *weight) {
#if defined(FASTLLM_ENABLE_CUTLASS_FP8) && \
    (defined(FASTLLM_CUTLASS_FP8_ENABLE_SM90) || \
     defined(FASTLLM_CUTLASS_FP8_ENABLE_SM120) || \
     defined(FASTLLM_CUTLASS_FP8_ENABLE_SM121))
    using namespace fastllm_w8a8_blockwise_detail;
    if (weight == nullptr) return;

    {
        std::lock_guard<std::mutex> guard(g_cutlassBackendMutex);
        for (auto it = g_cutlassBackendStates.begin();
             it != g_cutlassBackendStates.end();) {
            if (it->first.first == weight) {
                it = g_cutlassBackendStates.erase(it);
            } else {
                ++it;
            }
        }
    }

    std::lock_guard<std::mutex> guard(g_cutlassWeightMutex);
    const int originalDevice = FastllmCudaGetDevice();
    for (auto it = g_cutlassWeightCache.begin();
         it != g_cutlassWeightCache.end();) {
        if (it->first.first == weight) {
            FastllmCudaSetDevice(it->first.second);
            FastllmCutlassReleaseWeightCache(it->second);
            it = g_cutlassWeightCache.erase(it);
        } else {
            ++it;
        }
    }
    FastllmCudaSetDevice(originalDevice);
#else
    (void)weight;
#endif
}

FastllmCudaFp8W8A8BackendState
FastllmCudaGetFp8W8A8Block128BackendState(
        const fastllm::Data &weight, int device) {
#if defined(FASTLLM_ENABLE_CUTLASS_FP8) && \
    (defined(FASTLLM_CUTLASS_FP8_ENABLE_SM90) || \
     defined(FASTLLM_CUTLASS_FP8_ENABLE_SM120) || \
     defined(FASTLLM_CUTLASS_FP8_ENABLE_SM121))
    return fastllm_w8a8_blockwise_detail::FastllmCutlassGetBackendState(
        weight, device);
#else
    (void)weight;
    (void)device;
    return FastllmCudaFp8W8A8BackendState::Rejected;
#endif
}

/**
 * 在Blockwise FP8权重上传GPU后准备CUTLASS固定后端。
 *
 * 本阶段只验证与运行时M无关的权重语义：权重为FP8_E4M3 [N,K]，
 * N和K均按128对齐，FP32 scale按[N/128,K/128]组织，blockK和blockM
 * 均为128。验证通过后创建稳定的设备scale缓存并记录Prepared；首次
 * 真实GEMM负责验证kernel并最终固定Cutlass或Rejected。
 *
 * @param weight 已上传当前GPU的Blockwise FP8二维Linear权重。
 * @return true表示已处于Prepared/Cutlass；false表示不满足目标后端。
 */
bool FastllmCudaTryPrepareFp8W8A8Block128Weight(
        fastllm::Data &weight) {
#if defined(FASTLLM_ENABLE_CUTLASS_FP8) && \
    (defined(FASTLLM_CUTLASS_FP8_ENABLE_SM90) || \
     defined(FASTLLM_CUTLASS_FP8_ENABLE_SM120) || \
     defined(FASTLLM_CUTLASS_FP8_ENABLE_SM121))
    using namespace fastllm_w8a8_blockwise_detail;
    if (weight.dataType != fastllm::DataType::FP8_E4M3 ||
        weight.weightType != fastllm::WeightType::LINEAR ||
        weight.dims.size() != 2) return false;

    const int device = FastllmCudaGetDevice();
    const auto state = FastllmCutlassGetBackendState(weight, device);
    if (state == FastllmCudaFp8W8A8BackendState::Prepared ||
        state == FastllmCudaFp8W8A8BackendState::Cutlass) return true;
    if (state == FastllmCudaFp8W8A8BackendState::Rejected) return false;

    const int n = weight.dims[0];
    const int k = weight.dims[1];
    const int arch = FastllmCudaRuntimeArch();
    const bool supported = FastllmCutlassBlockwiseEnabled() &&
        FastllmCutlassFp8CompiledForRuntimeArch(arch) &&
        n > 0 && k > 0 && n % 128 == 0 && k % 128 == 0 &&
        weight.blockK == 128 && weight.blockM == 128 &&
        weight.scales.size() ==
            (size_t)(n / 128) * (k / 128) &&
        weight.cudaData != nullptr;
    if (!supported) {
        FastllmCutlassSetBackendState(
            weight, device, FastllmCudaFp8W8A8BackendState::Rejected);
        FastllmCutlassBlockwiseTrace(
            "weight-load backend selection rejected", 1, n, k);
        if (FastllmCutlassBlockwiseStrictEnabled()) {
            throw std::runtime_error(
                "strict FP8 W8A8 Block128 weight-load selection failed");
        }
        return false;
    }

    FastllmCutlassFp8WeightCache *cache = nullptr;
    if (!FastllmCutlassEnsureWeightCache(weight, k, n, 0, cache) ||
        cache == nullptr || cache->weightScales == nullptr) {
        FastllmCutlassReleaseWeightCacheForDevice(&weight, device);
        FastllmCutlassSetBackendState(
            weight, device, FastllmCudaFp8W8A8BackendState::Rejected);
        FastllmCutlassBlockwiseTrace(
            "weight scale cache creation failed", 1, n, k);
        if (FastllmCutlassBlockwiseStrictEnabled()) {
            throw std::runtime_error(
                "strict FP8 W8A8 Block128 scale cache creation failed");
        }
        return false;
    }

    FastllmCutlassSetBackendState(
        weight, device, FastllmCudaFp8W8A8BackendState::Prepared);
    FastllmCutlassBlockwiseTrace(
        "weight prepared; first GEMM validation pending", 1, n, k);
    return true;
#else
    (void)weight;
    return false;
#endif
}

/**
 * 执行一次正式Blockwise FP8 W8A8 CUTLASS线性计算。
 *
 * 数学语义为output[M,N] = input[M,K] * weight[N,K]^T + bias[N]。
 * FP16/BF16激活按每个token的连续128个K元素动态量化为FP8；权重使用
 * 静态128x128 FP32 scale。SM90与SM120按编译及运行架构选择对应CUTLASS
 * kernel，SM120的SwapAB、Pingpong和默认配置沿用vLLM形状分派。
 *
 * 后端按权重与GPU固定。首次GEMM同步验证成功后记录Cutlass；首次失败
 * 固定Rejected并允许非严格模式退出。Cutlass固定后任何语义或运行错误
 * 都抛出异常，禁止同一层静默切换Legacy。
 *
 * @param input  CUDA FP16/BF16激活，逻辑形状为[m,k]。
 * @param weight CUDA FP8_E4M3权重，逻辑形状为[n,k]，blockK=blockM=128。
 * @param bias   可选CUDA FP32偏置，长度为n。
 * @param output CUDA FP16/BF16输出，逻辑形状为[m,n]，类型与input相同。
 * @param m      GEMM的M维，通常为token数。
 * @param k      GEMM的K维，必须按128对齐。
 * @param n      GEMM的N维，必须按128对齐。
 * @return true表示CUTLASS计算成功提交；false仅表示首次选择被拒绝。
 */
bool FastllmCudaCutlassLinearFp8W8A8Block128(
        const fastllm::Data &input, fastllm::Data &weight,
        const fastllm::Data &bias, fastllm::Data &output,
        int m, int k, int n) {
#if defined(FASTLLM_ENABLE_CUTLASS_FP8) && \
    (defined(FASTLLM_CUTLASS_FP8_ENABLE_SM90) || \
     defined(FASTLLM_CUTLASS_FP8_ENABLE_SM120) || \
     defined(FASTLLM_CUTLASS_FP8_ENABLE_SM121))
    using namespace fastllm_w8a8_blockwise_detail;
    if (weight.dataType != fastllm::DataType::FP8_E4M3) return false;

    const int device = FastllmCudaGetDevice();
    auto state = FastllmCutlassGetBackendState(weight, device);
    if (state == FastllmCudaFp8W8A8BackendState::Uninitialized) {
        if (!FastllmCudaTryPrepareFp8W8A8Block128Weight(weight)) {
            return false;
        }
        state = FastllmCutlassGetBackendState(weight, device);
    }
    if (state == FastllmCudaFp8W8A8BackendState::Rejected) {
        if (FastllmCutlassBlockwiseStrictEnabled()) {
            throw std::runtime_error(
                "strict FP8 W8A8 Block128 requires the CUTLASS backend");
        }
        return false;
    }

    const bool selecting =
        state == FastllmCudaFp8W8A8BackendState::Prepared;
    const char *failure = nullptr;
    if (m <= 0 || k <= 0 || n <= 0) failure = "invalid GEMM dimensions";
    else if (k % 128 != 0 || n % 128 != 0) {
        failure = "K/N is not aligned to 128";
    } else if (input.cudaData == nullptr || weight.cudaData == nullptr ||
               output.cudaData == nullptr) {
        failure = "CUDA tensor pointer is null";
    } else if (weight.dims != std::vector<int>({n, k})) {
        failure = "weight shape mismatch";
    } else if (weight.blockK != 128 || weight.blockM != 128 ||
               weight.scales.size() !=
                   (size_t)(n / 128) * (k / 128)) {
        failure = "weight scale layout mismatch";
    } else if (input.dataType != fastllm::DataType::FLOAT16 &&
               input.dataType != fastllm::DataType::BFLOAT16) {
        failure = "input is not FP16/BF16";
    } else if (output.dataType != input.dataType) {
        failure = "output dtype differs from input";
    } else if (!bias.dims.empty() &&
               (bias.dataType != fastllm::DataType::FLOAT32 ||
                bias.cudaData == nullptr || bias.Count(0) != n)) {
        failure = "bias must be empty or FP32[N] on CUDA";
    }

    if (failure != nullptr) {
        FastllmCutlassBlockwiseTrace(failure, m, n, k);
        if (selecting) {
            FastllmCutlassReleaseWeightCacheForDevice(&weight, device);
            FastllmCutlassSetBackendState(
                weight, device,
                FastllmCudaFp8W8A8BackendState::Rejected);
            if (FastllmCutlassBlockwiseStrictEnabled()) {
                throw std::runtime_error(
                    std::string("strict FP8 W8A8 Block128 selection failed: ") +
                    failure);
            }
            return false;
        }
        throw std::runtime_error(
            std::string("fixed FP8 W8A8 Block128 CUTLASS backend failed: ") +
            failure);
    }

    const bool ok = FastllmCudaCutlassLinearFP8E4M3Block128Impl(
        input, weight, bias, output, m, k, n, false);
    cudaError_t validation = selecting && ok
        ? cudaStreamSynchronize(0)
        : (ok ? cudaSuccess : cudaErrorUnknown);
    const bool forcedFailure = ok && validation == cudaSuccess &&
        FastllmCutlassBlockwiseForceFailureForTest();
    if (forcedFailure && !selecting) {
        validation = cudaStreamSynchronize(0);
    }

    if (!ok || validation != cudaSuccess || forcedFailure) {
        const char *reason = !ok ? "CUTLASS launch failed" :
            (forcedFailure ? "forced CUTLASS failure" :
             "CUTLASS asynchronous validation failed");
        FastllmCutlassBlockwiseTrace(reason, m, n, k);
        if (selecting) {
            FastllmCutlassReleaseWeightCacheForDevice(&weight, device);
            FastllmCutlassSetBackendState(
                weight, device,
                FastllmCudaFp8W8A8BackendState::Rejected);
            if (FastllmCutlassBlockwiseStrictEnabled()) {
                throw std::runtime_error(
                    std::string("strict FP8 W8A8 Block128 failed: ") + reason);
            }
            return false;
        }
        throw std::runtime_error(
            std::string("fixed FP8 W8A8 Block128 CUTLASS backend failed: ") +
            reason);
    }

    if (selecting) {
        FastllmCutlassSetBackendState(
            weight, device, FastllmCudaFp8W8A8BackendState::Cutlass);
        FastllmCutlassBlockwiseTrace("backend fixed", m, n, k);
    }
    return true;
#else
    (void)input;
    (void)weight;
    (void)bias;
    (void)output;
    (void)m;
    (void)k;
    (void)n;
    return false;
#endif
}

bool FastllmCudaCutlassLinearFP8E4M3Block128Add(
    const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias,
    fastllm::Data &output, int n, int m, int k) {
    return FastllmCudaCutlassLinearFP8E4M3Block128Impl(
        input, weight, bias, output, n, m, k, true);
}

static bool FastllmCudaCutlassLinearFP8E4M3Block128FromRMSNormImpl(
    const fastllm::Data &input, fastllm::Data &normWeight, float eps,
    fastllm::Data *normOutput,
    fastllm::Data &weight, const fastllm::Data &bias,
    fastllm::Data &output, int n, int m, int k) {
#if defined(FASTLLM_ENABLE_CUTLASS_FP8) && \
    (defined(FASTLLM_CUTLASS_FP8_ENABLE_SM120) || defined(FASTLLM_CUTLASS_FP8_ENABLE_SM121))
    using namespace fastllm_w8a8_blockwise_detail;

    if (!FastllmCutlassUseWarpQuant() ||
        n < 64 || m != 5120 || k <= 0 || (k % 128) != 0 ||
        input.cudaData == nullptr || normWeight.cudaData == nullptr ||
        weight.cudaData == nullptr || output.cudaData == nullptr ||
        input.dataType != fastllm::DataType::FLOAT16 ||
        output.dataType != fastllm::DataType::FLOAT16 ||
        input.dims.empty() || input.dims.back() != m ||
        (normOutput != nullptr &&
         (normOutput->cudaData == nullptr ||
          normOutput->dataType != fastllm::DataType::FLOAT16 ||
          normOutput->dims != input.dims)) ||
        normWeight.dataType != fastllm::DataType::FLOAT32 ||
        normWeight.Count(0) != m ||
        weight.dims.size() != 2 ||
        weight.dims[0] != k || weight.dims[1] != m ||
        weight.dataType != fastllm::DataType::FP8_E4M3 ||
        weight.blockM != 128 || weight.blockK != 128 ||
        weight.scales.empty() ||
        (bias.dims.size() > 0 &&
         bias.dataType != fastllm::DataType::FLOAT32)) {
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
        !FastllmCutlassEnsureWeightCache(
            weight, m, k, stream, cache)) {
        return false;
    }

    void *inputData = FastllmCudaPrepareInput(input);
    void *outputData = FastllmCudaPrepareOutput(output);
    void *normOutputData = normOutput == nullptr ?
        nullptr : FastllmCudaPrepareOutput(*normOutput);
    if (inputData == nullptr || outputData == nullptr ||
        (normOutput != nullptr && normOutputData == nullptr)) {
        FastllmCudaFinishInput(input, inputData);
        if (normOutput != nullptr) {
            FastllmCudaFinishOutput(*normOutput, normOutputData);
        }
        FastllmCudaFinishOutput(output, outputData);
        return false;
    }

    if (normOutput != nullptr) {
        FastllmCutlassRMSNormQuantInputFp8Half5120Kernel<true>
            <<<n, 512, 0, stream>>>(
                (const half *)inputData,
                (const float *)normWeight.cudaData,
                (half *)normOutputData,
                scratch->input, scratch->inputScales, n, eps);
    } else {
        FastllmCutlassRMSNormQuantInputFp8Half5120Kernel<false>
            <<<n, 512, 0, stream>>>(
                (const half *)inputData,
                (const float *)normWeight.cudaData,
                nullptr,
                scratch->input, scratch->inputScales, n, eps);
    }
    bool ok = cudaGetLastError() == cudaSuccess;
    if (ok) {
        ok = FastllmDispatchCutlassFp8Blockwise(
            scratch->input, cache->weightTN,
            scratch->inputScales, cache->weightScales,
            (cutlass::half_t *)outputData, n, k, m, stream);
    }
    if (ok && bias.dims.size() > 0) {
        int threads = 256;
        int blocks = (int)std::min<size_t>(
            4096, ((size_t)n * k + threads - 1) / threads);
        FastllmCutlassAddFloatBiasKernel
            <<<blocks, threads, 0, stream>>>(
                (half *)outputData,
                (const float *)bias.cudaData, n, k);
        ok = cudaGetLastError() == cudaSuccess;
    }

    FastllmCudaFinishInput(input, inputData);
    if (normOutput != nullptr) {
        FastllmCudaFinishOutput(*normOutput, normOutputData);
    }
    FastllmCudaFinishOutput(output, outputData);
    return ok;
#else
    (void)input;
    (void)normWeight;
    (void)eps;
    (void)normOutput;
    (void)weight;
    (void)bias;
    (void)output;
    (void)n;
    (void)m;
    (void)k;
    return false;
#endif
}

bool FastllmCudaCutlassLinearFP8E4M3Block128FromRMSNorm(
    const fastllm::Data &input, fastllm::Data &normWeight, float eps,
    fastllm::Data &weight, const fastllm::Data &bias,
    fastllm::Data &output, int n, int m, int k) {
    return FastllmCudaCutlassLinearFP8E4M3Block128FromRMSNormImpl(
        input, normWeight, eps, nullptr,
        weight, bias, output, n, m, k);
}

bool FastllmCudaCutlassLinearFP8E4M3Block128FromRMSNormMaterialize(
    const fastllm::Data &input, fastllm::Data &normWeight, float eps,
    fastllm::Data &normOutput,
    fastllm::Data &weight, const fastllm::Data &bias,
    fastllm::Data &output, int n, int m, int k) {
    return FastllmCudaCutlassLinearFP8E4M3Block128FromRMSNormImpl(
        input, normWeight, eps, &normOutput,
        weight, bias, output, n, m, k);
}

static bool FastllmCudaCutlassLinearFP8E4M3Block128FromGdnOutputGateImpl(
    const fastllm::Data &headMajorInput,
    fastllm::Data &normWeight,
    const fastllm::Data &combinedGateInput,
    int batch, int seqLen, int gateOffset, int gateHeads, float eps,
    fastllm::Data &weight, const fastllm::Data &bias,
    fastllm::Data &output, int n, int m, int k,
    bool exactResidual) {
#if defined(FASTLLM_ENABLE_CUTLASS_FP8) && \
    (defined(FASTLLM_CUTLASS_FP8_ENABLE_SM120) || defined(FASTLLM_CUTLASS_FP8_ENABLE_SM121))
    using namespace fastllm_w8a8_blockwise_detail;

    auto isDense = [](const fastllm::Data &data) {
        if (data.dims.empty() ||
            data.strides.size() != data.dims.size()) {
            return false;
        }
        uint64_t expected = 1;
        for (int i = (int)data.dims.size() - 1; i >= 0; --i) {
            if (data.strides[i] != expected) {
                return false;
            }
            expected *= (uint64_t)data.dims[i];
        }
        return true;
    };

    int minBatch = FastllmCutlassEnvInt(
        "FASTLLM_CUDA_CUTLASS_LINEAR_FP8_MIN_BATCH", 8);
    if (!FastllmCutlassUseWarpQuant() ||
        batch <= 0 || seqLen <= 0 || gateOffset < 0 ||
        gateHeads <= 0 || n < minBatch ||
        n != batch * seqLen || m != gateHeads * 128 ||
        m <= 0 || k <= 0 || (m % 128) != 0 ||
        (k % 128) != 0 || !std::isfinite(eps) || eps < 0.0f ||
        headMajorInput.dataDevice != fastllm::DataDevice::CUDA ||
        combinedGateInput.dataDevice != fastllm::DataDevice::CUDA ||
        normWeight.dataDevice != fastllm::DataDevice::CUDA ||
        headMajorInput.cudaData == nullptr ||
        combinedGateInput.cudaData == nullptr ||
        normWeight.cudaData == nullptr ||
        weight.cudaData == nullptr ||
        headMajorInput.dataType != fastllm::DataType::FLOAT16 ||
        combinedGateInput.dataType != fastllm::DataType::FLOAT16 ||
        normWeight.dataType != fastllm::DataType::FLOAT32 ||
        output.dataType != fastllm::DataType::FLOAT16 ||
        normWeight.Count(0) != 128 ||
        !isDense(headMajorInput) || !isDense(combinedGateInput) ||
        headMajorInput.dims.size() != 4 ||
        headMajorInput.dims[0] != batch ||
        headMajorInput.dims[1] != gateHeads ||
        headMajorInput.dims[2] < seqLen ||
        headMajorInput.dims[3] != 128 ||
        combinedGateInput.dims.empty() ||
        weight.dims.size() != 2 ||
        weight.dims[0] != k || weight.dims[1] != m ||
        weight.dataType != fastllm::DataType::FP8_E4M3 ||
        weight.blockM != 128 || weight.blockK != 128 ||
        weight.scales.empty() ||
        output.dims.empty() || output.dims.back() != k ||
        output.Count(0) != (uint64_t)n * k ||
        (exactResidual && !bias.dims.empty()) ||
        (bias.dims.size() > 0 &&
         bias.dataType != fastllm::DataType::FLOAT32)) {
        return false;
    }

    int gateStride = combinedGateInput.dims.back();
    if (gateOffset > gateStride ||
        m > gateStride - gateOffset ||
        combinedGateInput.Count(0) !=
            (uint64_t)n * gateStride) {
        return false;
    }
    int64_t tasks64 = (int64_t)n * gateHeads;
    if (tasks64 <= 0 ||
        tasks64 > std::numeric_limits<int>::max()) {
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
        !FastllmCutlassEnsureWeightCache(
            weight, m, k, stream, cache)) {
        return false;
    }

    void *inputData = FastllmCudaPrepareInput(headMajorInput);
    void *gateData = FastllmCudaPrepareInput(combinedGateInput);
    void *outputData = FastllmCudaPrepareOutput(output);
    if (inputData == nullptr ||
        gateData == nullptr || outputData == nullptr) {
        FastllmCudaFinishInput(headMajorInput, inputData);
        FastllmCudaFinishInput(combinedGateInput, gateData);
        FastllmCudaFinishOutput(output, outputData);
        return false;
    }

    constexpr int warpQuantWarps = 8;
    int tasks = (int)tasks64;
    dim3 grid(
        (tasks + warpQuantWarps - 1) / warpQuantWarps);
    FastllmCutlassGdnOutputGateQuantInputFp8Half128HeadMajorKernel<
        warpQuantWarps>
        <<<grid, warpQuantWarps * 32, 0, stream>>>(
            (const half *)inputData,
            (const float *)normWeight.cudaData,
            (const half *)gateData,
            scratch->input, scratch->inputScales,
            n, seqLen, headMajorInput.dims[2],
            gateStride, gateOffset, gateHeads, eps);
    bool ok = cudaGetLastError() == cudaSuccess;
    if (ok) {
        if (exactResidual) {
            ok = FastllmDispatchCutlassFp8Blockwise<true>(
                scratch->input, cache->weightTN,
                scratch->inputScales, cache->weightScales,
                (cutlass::half_t *)outputData,
                n, k, m, stream);
        } else {
            ok = FastllmDispatchCutlassFp8Blockwise(
                scratch->input, cache->weightTN,
                scratch->inputScales, cache->weightScales,
                (cutlass::half_t *)outputData,
                n, k, m, stream);
        }
    }
    if (ok && bias.dims.size() > 0) {
        int threads = 256;
        int blocks = (int)std::min<size_t>(
            4096, ((size_t)n * k + threads - 1) / threads);
        FastllmCutlassAddFloatBiasKernel
            <<<blocks, threads, 0, stream>>>(
                (half *)outputData,
                (const float *)bias.cudaData, n, k);
        ok = cudaGetLastError() == cudaSuccess;
    }

    FastllmCudaFinishInput(headMajorInput, inputData);
    FastllmCudaFinishInput(combinedGateInput, gateData);
    FastllmCudaFinishOutput(output, outputData);
    return ok;
#else
    (void)headMajorInput;
    (void)normWeight;
    (void)combinedGateInput;
    (void)batch;
    (void)seqLen;
    (void)gateOffset;
    (void)gateHeads;
    (void)eps;
    (void)weight;
    (void)bias;
    (void)output;
    (void)n;
    (void)m;
    (void)k;
    (void)exactResidual;
    return false;
#endif
}

bool FastllmCudaCutlassLinearFP8E4M3Block128FromGdnOutputGate(
    const fastllm::Data &headMajorInput,
    fastllm::Data &normWeight,
    const fastllm::Data &combinedGateInput,
    int batch, int seqLen, int gateOffset, int gateHeads, float eps,
    fastllm::Data &weight, const fastllm::Data &bias,
    fastllm::Data &output, int n, int m, int k) {
    return FastllmCudaCutlassLinearFP8E4M3Block128FromGdnOutputGateImpl(
        headMajorInput, normWeight, combinedGateInput,
        batch, seqLen, gateOffset, gateHeads, eps,
        weight, bias, output, n, m, k, false);
}

bool FastllmCudaCutlassLinearFP8E4M3Block128FromGdnOutputGateAdd(
    const fastllm::Data &headMajorInput,
    fastllm::Data &normWeight,
    const fastllm::Data &combinedGateInput,
    int batch, int seqLen, int gateOffset, int gateHeads, float eps,
    fastllm::Data &weight, const fastllm::Data &bias,
    fastllm::Data &output, int n, int m, int k) {
    return FastllmCudaCutlassLinearFP8E4M3Block128FromGdnOutputGateImpl(
        headMajorInput, normWeight, combinedGateInput,
        batch, seqLen, gateOffset, gateHeads, eps,
        weight, bias, output, n, m, k, true);
}

static bool FastllmCudaCutlassLinearFP8E4M3Block128FromSwigluImpl(
    const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias,
    fastllm::Data &output, int n, int m, int k, bool exactResidual) {
#if defined(FASTLLM_ENABLE_CUTLASS_FP8) && \
    (defined(FASTLLM_CUTLASS_FP8_ENABLE_SM90) || defined(FASTLLM_CUTLASS_FP8_ENABLE_SM120) || defined(FASTLLM_CUTLASS_FP8_ENABLE_SM121))
    using namespace fastllm_w8a8_blockwise_detail;

    if (!FastllmCutlassUseFusedSwigluQuant() || !FastllmCutlassUseWarpQuant()) {
        return false;
    }
    int minBatch = FastllmCutlassEnvInt("FASTLLM_CUDA_CUTLASS_LINEAR_FP8_MIN_BATCH", 8);
    if (n < minBatch) {
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
        (exactResidual && (input.dataType != fastllm::DataType::FLOAT16 ||
                           !bias.dims.empty())) ||
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
        if (exactResidual) {
            ok = FastllmDispatchCutlassFp8Blockwise<true>(
                scratch->input, cache->weightTN, scratch->inputScales, cache->weightScales,
                (cutlass::half_t*)outputData, n, k, m, stream);
        } else {
            ok = FastllmDispatchCutlassFp8Blockwise(
                scratch->input, cache->weightTN, scratch->inputScales, cache->weightScales,
                (cutlass::half_t*)outputData, n, k, m, stream);
        }
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
    (void)exactResidual;
    return false;
#endif
}

bool FastllmCudaCutlassLinearFP8E4M3Block128FromSwiglu(
    const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias,
    fastllm::Data &output, int n, int m, int k) {
    return FastllmCudaCutlassLinearFP8E4M3Block128FromSwigluImpl(
        input, weight, bias, output, n, m, k, false);
}

bool FastllmCudaCutlassLinearFP8E4M3Block128FromSwigluAdd(
    const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias,
    fastllm::Data &output, int n, int m, int k) {
    return FastllmCudaCutlassLinearFP8E4M3Block128FromSwigluImpl(
        input, weight, bias, output, n, m, k, true);
}
