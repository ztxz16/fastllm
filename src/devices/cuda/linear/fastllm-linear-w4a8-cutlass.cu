#include "fastllm-cuda.cuh"
#include "fastllm-w4a8-utils.cuh"
#include "libtorch_stable/quantization/vectorization_utils.cuh"

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/mixed_dtype_utils.hpp"
#include "cutlass/util/packed_stride.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

namespace {

static constexpr uint64_t W4A8_CACHE_MAGIC = 0x5734413843414348ULL; // "W4A8CACH"
static constexpr int W4A8_GROUP_SIZE = 128;
static constexpr int W4A8_SCALE_PACK_SIZE = 8;
static constexpr int W4A8_FP8_QUANT_THREADS = 256;
static constexpr float W4A8_FP8_MAX = 448.0f;
static constexpr float W4A8_FP8_MIN_SCALE =
    1.0f / (W4A8_FP8_MAX * 512.0f);

struct W4A8MaxOp {
    __device__ __forceinline__ float operator()(float lhs, float rhs) const {
        return fmaxf(lhs, rhs);
    }
};

struct W4A8ActivationScratch {
    cutlass::float_e4m3_t *fp8 = nullptr;
    float *tokenScales = nullptr;
    int tokens = 0;
    int hidden = 0;
    size_t fp8Bytes = 0;
    size_t scaleBytes = 0;
};

using W4A8MmaType = cutlass::float_e4m3_t;
using W4A8QuantType = cutlass::int4b_t;
static constexpr int W4A8_TILE_SHAPE_K = 128 * 8 / cute::sizeof_bits<W4A8MmaType>::value;

using W4A8LayoutA = cutlass::layout::RowMajor;
using W4A8LayoutATranspose = typename cutlass::layout::LayoutTranspose<W4A8LayoutA>::type;
using W4A8StrideA = cutlass::detail::TagToStrideA_t<W4A8LayoutA>;
using W4A8LayoutB = cutlass::layout::ColumnMajor;
using W4A8StrideB = cutlass::detail::TagToStrideB_t<W4A8LayoutB>;
using W4A8LayoutAtomQuant = decltype(cutlass::compute_memory_reordering_atom<W4A8MmaType>());
using W4A8LayoutBReordered = decltype(cute::tile_to_shape(
    W4A8LayoutAtomQuant{}, cute::Layout<cute::Shape<int, int, int>, W4A8StrideB>{}));

template <typename ElementAcc, typename ElementD, typename TileShape>
struct FastllmCudaW4A8ScaledEpilogue {
    using Accum = cutlass::epilogue::fusion::Sm90AccFetch;
    using ChannelScale = cutlass::epilogue::fusion::Sm90ColBroadcast<
        0, TileShape, float, float, cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>>;
    using TokenScale = cutlass::epilogue::fusion::Sm90RowBroadcast<
        0, TileShape, float, float, cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

    using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
        cutlass::multiplies, float, float,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using EVTCompute0 =
        cutlass::epilogue::fusion::Sm90EVT<Compute0, TokenScale, Accum>;

    using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
        cutlass::multiplies, ElementD, float,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using EVTCompute =
        cutlass::epilogue::fusion::Sm90EVT<Compute1, ChannelScale, EVTCompute0>;
    using ArgumentType = typename EVTCompute::Arguments;

    static ArgumentType prepare_args(const float *channelScales,
                                     const float *tokenScales) {
        typename ChannelScale::Arguments channelArgs{channelScales, 0.0f, {}};
        typename TokenScale::Arguments tokenArgs{tokenScales, 0.0f, {}};
        typename EVTCompute0::Arguments evt0Args{tokenArgs, {}, {}};
        return ArgumentType{channelArgs, evt0Args, {}};
    }
};

template <typename OutType, class TileShapeMN, class ClusterShapeMNK>
struct FastllmCudaW4A8GemmKernel {
    using TileShape =
        decltype(cute::append(TileShapeMN{}, cute::Int<W4A8_TILE_SHAPE_K>{}));
    using ClusterShape = ClusterShapeMNK;
    using ElementA = W4A8MmaType;
    using ElementB = W4A8QuantType;
    using ElementScale = W4A8MmaType;
    using ElementSChannel = float;
    using ElementAccumulator = float;
    using ElementCompute = float;
    using ElementC = OutType;
    using ElementD = OutType;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = LayoutC;
    using ChTokScalesEpilogue =
        FastllmCudaW4A8ScaledEpilogue<ElementAccumulator, ElementD, TileShape>;
    using EVTCompute = typename ChTokScalesEpilogue::EVTCompute;

    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape,
            ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
            ElementAccumulator, ElementSChannel, ElementC,
            typename cutlass::layout::LayoutTranspose<LayoutC>::type,
            AlignmentC, ElementD,
            typename cutlass::layout::LayoutTranspose<LayoutD>::type,
            AlignmentD, cutlass::epilogue::TmaWarpSpecializedCooperative,
            EVTCompute>::CollectiveOp;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
            cute::tuple<ElementB, cutlass::Array<ElementScale, W4A8_SCALE_PACK_SIZE>>,
            W4A8LayoutBReordered, AlignmentB, ElementA, W4A8LayoutATranspose,
            AlignmentA, ElementAccumulator, TileShape, ClusterShape,
            cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
                sizeof(typename CollectiveEpilogue::SharedStorage))>,
            cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideD = typename GemmKernel::StrideD;
    using StrideS = typename CollectiveMainloop::StrideScale;

    static bool run(const W4A8MmaType *activation,
                    const W4A8QuantType *packedWeight,
                    const cutlass::Array<ElementScale, W4A8_SCALE_PACK_SIZE> *packedGroupScales,
                    const float *channelScales,
                    const float *tokenScales,
                    ElementD *output,
                    int tokens,
                    int outChannels,
                    int inChannels,
                    cudaStream_t stream) {
        auto shapeB = cute::make_shape(outChannels, inChannels, 1);
        W4A8LayoutBReordered layoutBReordered =
            cute::tile_to_shape(W4A8LayoutAtomQuant{}, shapeB);

        W4A8StrideA strideA = cutlass::make_cute_packed_stride(
            W4A8StrideA{}, cute::make_shape(tokens, inChannels, 1));
        StrideD strideD = cutlass::make_cute_packed_stride(
            StrideD{}, cute::make_shape(outChannels, tokens, 1));
        StrideS strideS = cutlass::make_cute_packed_stride(
            StrideS{}, cute::make_shape(outChannels, inChannels / W4A8_GROUP_SIZE, 1));

        using Args = typename Gemm::Arguments;
        using MainloopArguments = typename GemmKernel::MainloopArguments;
        using EpilogueArguments = typename GemmKernel::EpilogueArguments;

        MainloopArguments mainloopArguments{
            packedWeight,
            layoutBReordered,
            activation,
            strideA,
            packedGroupScales,
            strideS,
            W4A8_GROUP_SIZE};

        EpilogueArguments epilogueArguments{
            ChTokScalesEpilogue::prepare_args(channelScales, tokenScales),
            nullptr,
            {},
            output,
            strideD};

        Args arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                       {outChannels, tokens, inChannels, 1},
                       mainloopArguments,
                       epilogueArguments};

        size_t workspaceSize = Gemm::get_workspace_size(arguments);
        void *workspace = workspaceSize == 0 ? nullptr : FastllmCudaMalloc(workspaceSize);
        if (workspaceSize != 0 && workspace == nullptr) {
            return false;
        }

        Gemm gemm;
        cutlass::Status status = gemm.can_implement(arguments);
        if (status == cutlass::Status::kSuccess) {
            status = gemm.initialize(arguments, workspace, stream);
        }
        if (status == cutlass::Status::kSuccess) {
            status = gemm.run(stream);
        }
        if (workspace != nullptr) {
            FastllmCudaFree(workspace);
        }
        return status == cutlass::Status::kSuccess && cudaGetLastError() == cudaSuccess;
    }
};

static std::mutex g_w4a8WeightCacheMutex;

static bool FastllmCudaW4A8PrepareCacheEnabled() {
    const char *env = std::getenv("FASTLLM_CUDA_W4A8_PREPARE_CACHE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool FastllmCudaW4A8PrepareActivationEnabled() {
    const char *env = std::getenv("FASTLLM_CUDA_W4A8_PREPARE_ACTIVATION");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool FastllmCudaW4A8PrepareOutputEnabled() {
    const char *env = std::getenv("FASTLLM_CUDA_W4A8_PREPARE_OUTPUT");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool FastllmCudaW4A8GemmEnabled() {
    const char *env = std::getenv("FASTLLM_CUDA_W4A8_ENABLE_GEMM");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool FastllmCudaW4A8ValidateEnabled() {
    const char *env = std::getenv("FASTLLM_CUDA_W4A8_VALIDATE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool FastllmCudaW4A8TraceEnabled() {
    const char *env = std::getenv("FASTLLM_CUDA_W4A8_TRACE");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

static bool FastllmCudaW4A8RuntimeArchSupported(int arch) {
    return arch == 90;
}

static void FastllmCudaW4A8TraceSkip(const char *reason,
                                     int n,
                                     int m,
                                     int k,
                                     int arch,
                                     const fastllm::Data &input,
                                     const fastllm::Data &weight,
                                     const fastllm::Data &bias,
                                     const fastllm::Data &output) {
    if (!FastllmCudaW4A8TraceEnabled() && !FastllmCudaW4A8ValidateEnabled()) {
        return;
    }
    printf("[FastLLM][W4A8] skip: %s n=%d m=%d k=%d arch=%d input=%d weight=%d bias=%d output=%d\n",
           reason, n, m, k, arch, (int)input.dataType, (int)weight.dataType,
           (int)bias.dataType, (int)output.dataType);
}

static void FastllmCudaW4A8TraceReady(const char *stage,
                                      int n,
                                      int m,
                                      int k,
                                      int arch) {
    if (!FastllmCudaW4A8TraceEnabled() && !FastllmCudaW4A8ValidateEnabled()) {
        return;
    }
    printf("[FastLLM][W4A8] %s ready: n=%d m=%d k=%d arch=%d\n", stage, n, m, k, arch);
}

static bool FastllmCudaW4A8BasicShapeSupported(int m, int k) {
    return m > 0 && k > 0 && (m % W4A8_GROUP_SIZE) == 0 && (k % W4A8_GROUP_SIZE) == 0;
}

static bool FastllmCudaW4A8CanUseWeightSource(const fastllm::Data &weight,
                                              int m,
                                              int k,
                                              const char **reason = nullptr);

static bool FastllmCudaW4A8CanQuantizeActivation(const fastllm::Data &input,
                                                 int n,
                                                 int m) {
    return n > 0 &&
           m > 0 &&
           (input.dataType == fastllm::DataType::FLOAT16 ||
            input.dataType == fastllm::DataType::BFLOAT16);
}

static bool FastllmCudaW4A8LinearSemanticsSupported(const fastllm::Data &input,
                                                    const fastllm::Data &weight,
                                                    const fastllm::Data &bias,
                                                    const fastllm::Data &output,
                                                    int n,
                                                    int m,
                                                    int k,
                                                    const char **reason = nullptr) {
    if (n <= 0 || m <= 0 || k <= 0) {
        if (reason != nullptr) {
            *reason = "invalid n/m/k";
        }
        return false;
    }
    if (input.dims.empty() || output.dims.empty()) {
        if (reason != nullptr) {
            *reason = "empty input/output dims";
        }
        return false;
    }
    if (input.dims.back() != m || output.dims.back() != k) {
        if (reason != nullptr) {
            *reason = "linear dimension mismatch";
        }
        return false;
    }
    if (input.Count(0) != n * m || output.Count(0) != n * k) {
        if (reason != nullptr) {
            *reason = "flattened shape mismatch";
        }
        return false;
    }
    if (output.dataType != input.dataType) {
        if (reason != nullptr) {
            *reason = "output dtype mismatch";
        }
        return false;
    }
    if (!FastllmCudaW4A8CanQuantizeActivation(input, n, m)) {
        if (reason != nullptr) {
            *reason = "input dtype is not fp16/bf16";
        }
        return false;
    }
    if (!FastllmCudaW4A8BasicShapeSupported(m, k)) {
        if (reason != nullptr) {
            *reason = "m/k is not 128-aligned";
        }
        return false;
    }

    if (bias.dims.size() > 0 &&
        (bias.dataType != fastllm::DataType::FLOAT32 ||
         bias.cudaData == nullptr ||
         bias.Count(0) != k)) {
        if (reason != nullptr) {
            *reason = "bias is not float32[k]";
        }
        return false;
    }
    return FastllmCudaW4A8CanUseWeightSource(weight, m, k, reason);
}

static bool FastllmCudaW4A8CanUseWeightSource(const fastllm::Data &weight,
                                              int m,
                                              int k,
                                              const char **reason) {
    bool legacyInt4Group = weight.dataType == fastllm::DataType::INT4_GROUP;
    bool compressedW4A8 = weight.dataType == fastllm::DataType::INT4_W4A8;
    if (!legacyInt4Group && !compressedW4A8) {
        if (reason != nullptr) {
            *reason = "weight dtype is not INT4_GROUP/INT4_W4A8";
        }
        return false;
    }
    if (weight.cudaData == nullptr) {
        if (reason != nullptr) {
            *reason = "weight cudaData is null";
        }
        return false;
    }
    if (weight.dims.size() != 2 || weight.dims[0] != k || weight.dims[1] != m) {
        if (reason != nullptr) {
            *reason = "weight shape is not [k, m]";
        }
        return false;
    }
    if (weight.groupCnt != W4A8_GROUP_SIZE || weight.group != m / W4A8_GROUP_SIZE) {
        if (reason != nullptr) {
            *reason = "weight group is not 128";
        }
        return false;
    }
    if (!FastllmCudaW4A8BasicShapeSupported(m, k)) {
        if (reason != nullptr) {
            *reason = "m/k is not 128-aligned";
        }
        return false;
    }

    size_t expectedScaleCount = (size_t)k * weight.group;
    if (compressedW4A8) {
        std::string validationReason;
        if (!weight.ValidateW4A8Weight(&validationReason)) {
            if (reason != nullptr) {
                *reason = "INT4_W4A8 source metadata is inconsistent";
            }
            return false;
        }
        return true;
    }

    if (weight.scales.size() != expectedScaleCount ||
        weight.mins.size() != expectedScaleCount) {
        if (reason != nullptr) {
            *reason = "weight scale/min shape mismatch";
        }
        return false;
    }

    // The legacy source is valid only when its affine representation is
    // exactly the same symmetric signed INT4 quantization used by W4A8.
    for (size_t i = 0; i < weight.mins.size(); i++) {
        float expectedMin = -8.0f * weight.scales[i];
        float tol = 1e-6f * (std::fabs(weight.scales[i]) > 1.0f ? std::fabs(weight.scales[i]) : 1.0f);
        if (std::fabs(weight.mins[i] - expectedMin) > tol) {
            if (reason != nullptr) {
                *reason = "weight min/scale is not signed-int4 compatible";
            }
            return false;
        }
    }
    return true;
}

__global__ void FastllmCudaW4A8PackInt4GroupToCutlassBKernel(const uint8_t *src,
                                                             uint32_t *dst,
                                                             int inChannels,
                                                             int outChannels) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)(inChannels / 8) * outChannels;
    if (idx >= total) {
        return;
    }

    int packRow = (int)(idx / outChannels);
    int out = (int)(idx - (size_t)packRow * outChannels);
    int inBase = packRow * 8;
    const uint8_t *row = src + (size_t)out * (inChannels / 2);

    uint32_t packed = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        int in = inBase + i;
        uint8_t byte = row[in / 2];
        uint32_t q = (in & 1) ? (byte & 0xF) : (byte >> 4);
        uint32_t signedQ = (q - 8) & 0xF;
        packed |= signedQ << (i * 4);
    }
    dst[(size_t)out * (inChannels / 8) + packRow] = packed;
}

__global__ void FastllmCudaW4A8ConvertUint4B8ToSignedInt4Kernel(
    const uint8_t *src, uint8_t *dst, size_t bytes) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bytes) {
        return;
    }
    uint8_t packed = src[idx];
    uint8_t low = ((packed & 0xF) - 8) & 0xF;
    uint8_t high = (((packed >> 4) & 0xF) - 8) & 0xF;
    dst[idx] = low | (high << 4);
}

__device__ inline float FastllmCudaW4A8ToFloat(half v) {
    return __half2float(v);
}

__device__ inline float FastllmCudaW4A8ToFloat(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename T>
__device__ inline T FastllmCudaW4A8FromFloat(float v);

template <>
__device__ inline half FastllmCudaW4A8FromFloat<half>(float v) {
    return __float2half(v);
}

template <>
__device__ inline __nv_bfloat16 FastllmCudaW4A8FromFloat<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template <typename T>
__global__ void FastllmCudaW4A8AddFloatBiasKernel(T *output,
                                                  const float *bias,
                                                  int tokens,
                                                  int outChannels) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)tokens * outChannels;
    if (idx >= total) {
        return;
    }
    int col = (int)(idx % outChannels);
    output[idx] = FastllmCudaW4A8FromFloat<T>(FastllmCudaW4A8ToFloat(output[idx]) + bias[col]);
}

template <typename T>
__global__ void FastllmCudaW4A8QuantizeActivationPerTokenKernel(
    const T *input,
    cutlass::float_e4m3_t *fp8,
    float *tokenScales,
    int tokens,
    int hidden) {
    int row = blockIdx.x;
    if (row >= tokens) {
        return;
    }

    const int tid = threadIdx.x;
    const T *tokenInput = input + (size_t)row * hidden;
    cutlass::float_e4m3_t *tokenFp8 = fp8 + (size_t)row * hidden;

    float localMax = 0.0f;
    fastllm::vectorize_read_with_alignment<16>(
        tokenInput, hidden, tid, blockDim.x,
        [&] __device__(T value) {
            localMax = fmaxf(
                localMax, fabsf(FastllmCudaW4A8ToFloat(value)));
        });

    using BlockReduce = cub::BlockReduce<float, W4A8_FP8_QUANT_THREADS>;
    __shared__ typename BlockReduce::TempStorage reduceStorage;
    float rowMax = BlockReduce(reduceStorage).Reduce(localMax, W4A8MaxOp{});

    __shared__ float tokenScale;
    if (threadIdx.x == 0) {
        tokenScale = fmaxf(rowMax / W4A8_FP8_MAX, W4A8_FP8_MIN_SCALE);
        tokenScales[row] = tokenScale;
    }
    __syncthreads();

    fastllm::vectorize_with_alignment<16>(
        tokenInput, tokenFp8, hidden, tid, blockDim.x,
        [=] __device__(cutlass::float_e4m3_t &dst, const T &src) {
            float value = FastllmCudaW4A8ToFloat(src) / tokenScale;
            value = fminf(
                W4A8_FP8_MAX, fmaxf(-W4A8_FP8_MAX, value));
            dst = cutlass::float_e4m3_t(value);
        });
}

static bool FastllmCudaW4A8LaunchActivationQuant(
    const fastllm::Data &input,
    const void *inputData,
    cutlass::float_e4m3_t *fp8,
    float *tokenScales,
    int n,
    int m,
    cudaStream_t stream) {
    if (!FastllmCudaW4A8CanQuantizeActivation(input, n, m) ||
        inputData == nullptr || fp8 == nullptr || tokenScales == nullptr) {
        return false;
    }

    dim3 grid(n);
    if (input.dataType == fastllm::DataType::FLOAT16) {
        FastllmCudaW4A8QuantizeActivationPerTokenKernel<<<
            grid, W4A8_FP8_QUANT_THREADS, 0, stream>>>(
                (const half*)inputData, fp8, tokenScales, n, m);
    } else {
        FastllmCudaW4A8QuantizeActivationPerTokenKernel<<<
            grid, W4A8_FP8_QUANT_THREADS, 0, stream>>>(
                (const __nv_bfloat16*)inputData, fp8, tokenScales, n, m);
    }
    return cudaGetLastError() == cudaSuccess;
}

static bool FastllmCudaW4A8EncodeAndReorderInt4B(uint8_t *rawPackedWeight,
                                                 uint8_t *cutlassPackedWeight,
                                                 int inChannels,
                                                 int outChannels) {
    using MmaType = cutlass::float_e4m3_t;
    using QuantType = cutlass::int4b_t;

    auto rawPtr = reinterpret_cast<QuantType const*>(rawPackedWeight);
    auto packedPtr = reinterpret_cast<QuantType*>(cutlassPackedWeight);
    size_t numInt4Elems = (size_t)inChannels * outChannels;

    if (!fastllm::cuda::w4a8::EncodeInt4ForCutlass(rawPtr, packedPtr, numInt4Elems)) {
        return false;
    }

    auto shapeB = cute::make_shape(outChannels, inChannels, 1);
    auto layoutB = cute::make_layout(shapeB, cute::LayoutRight{});
    auto layoutAtomQuant = cutlass::compute_memory_reordering_atom<MmaType>();
    auto layoutBReordered = cute::tile_to_shape(layoutAtomQuant, shapeB);
    cutlass::reorder_tensor(packedPtr, layoutB, layoutBReordered);
    return true;
}

template <typename OutType>
static bool FastllmCudaW4A8DispatchForOutputType(
    const W4A8MmaType *activation,
    const W4A8QuantType *packedWeight,
    const cutlass::Array<W4A8MmaType, W4A8_SCALE_PACK_SIZE> *packedGroupScales,
    const float *channelScales,
    const float *tokenScales,
    OutType *output,
    int n,
    int m,
    int k,
    cudaStream_t stream) {
    using namespace cute;
    std::string schedule;
    if (n <= 16) {
        schedule = (m == 16384 && k == 18432) ? "256x16_1x1x1" : "128x16_1x1x1";
    } else if (n <= 32) {
        schedule = (m == 16384 && k == 18432) ? "256x32_1x1x1" : "128x32_1x1x1";
    } else if (n <= 64) {
        if (m == 16384 && k == 18432) {
            schedule = "256x64_1x1x1";
        } else if (k <= 8192 && m <= 8192) {
            schedule = "128x32_1x1x1";
        } else {
            schedule = "128x64_1x1x1";
        }
    } else if (n <= 128) {
        if (m == 16384 && k == 18432) {
            schedule = "256x128_1x1x1";
        } else if (k <= 8192) {
            schedule = "128x64_1x1x1";
        } else {
            schedule = "128x128_1x1x1";
        }
    } else if (n <= 256) {
        if (k <= 4096) {
            schedule = "128x64_1x1x1";
        } else if (k <= 8192) {
            schedule = "128x128_1x1x1";
        } else {
            schedule = "128x256_1x1x1";
        }
    } else if (n <= 512 && k <= 4096) {
        schedule = "128x128_1x1x1";
    } else if (n <= 1024) {
        schedule = "128x256_1x1x1";
    } else {
        schedule = "128x256_2x1x1";
    }

#define FASTLLM_W4A8_DISPATCH(SCHEDULE, TILE_M, TILE_N, CLUSTER_M, CLUSTER_N, CLUSTER_K) \
    if (schedule == SCHEDULE) { \
        using Kernel = FastllmCudaW4A8GemmKernel<OutType, Shape<Int<TILE_M>, Int<TILE_N>>, \
                                                Shape<Int<CLUSTER_M>, Int<CLUSTER_N>, Int<CLUSTER_K>>>; \
        return Kernel::run(activation, packedWeight, packedGroupScales, channelScales, tokenScales, \
                           output, n, k, m, stream); \
    }

    FASTLLM_W4A8_DISPATCH("256x128_1x1x1", 256, 128, 1, 1, 1)
    FASTLLM_W4A8_DISPATCH("256x64_1x1x1", 256, 64, 1, 1, 1)
    FASTLLM_W4A8_DISPATCH("256x32_1x1x1", 256, 32, 1, 1, 1)
    FASTLLM_W4A8_DISPATCH("256x16_1x1x1", 256, 16, 1, 1, 1)
    FASTLLM_W4A8_DISPATCH("128x256_2x1x1", 128, 256, 2, 1, 1)
    FASTLLM_W4A8_DISPATCH("128x256_1x1x1", 128, 256, 1, 1, 1)
    FASTLLM_W4A8_DISPATCH("128x128_1x1x1", 128, 128, 1, 1, 1)
    FASTLLM_W4A8_DISPATCH("128x64_1x1x1", 128, 64, 1, 1, 1)
    FASTLLM_W4A8_DISPATCH("128x32_1x1x1", 128, 32, 1, 1, 1)
    FASTLLM_W4A8_DISPATCH("128x16_1x1x1", 128, 16, 1, 1, 1)
#undef FASTLLM_W4A8_DISPATCH
    return false;
}

static bool FastllmCudaW4A8RunGemm(const fastllm::Data &output,
                                   const W4A8ActivationScratch &activation,
                                   const fastllm::Data &weight,
                                   void *outputData,
                                   int n,
                                   int m,
                                   int k,
                                   cudaStream_t stream) {
    int deviceId = FastllmCudaGetDevice();
    auto cacheIt = weight.w4a8CudaCaches.find(deviceId);
    if (activation.fp8 == nullptr ||
        activation.tokenScales == nullptr ||
        outputData == nullptr ||
        cacheIt == weight.w4a8CudaCaches.end()) {
        return false;
    }

    const fastllm::W4A8CudaWeightCache &cache = cacheIt->second;
    if (cache.packedWeight == nullptr || cache.packedGroupScales == nullptr ||
        cache.channelScales == nullptr) {
        return false;
    }
    const auto *packedWeight =
        reinterpret_cast<const W4A8QuantType*>(cache.packedWeight);
    const auto *packedGroupScales =
        reinterpret_cast<const cutlass::Array<W4A8MmaType, W4A8_SCALE_PACK_SIZE>*>(
            cache.packedGroupScales);
    const float *channelScales =
        reinterpret_cast<const float*>(cache.channelScales);

    if (output.dataType == fastllm::DataType::FLOAT16) {
        return FastllmCudaW4A8DispatchForOutputType(
            activation.fp8, packedWeight, packedGroupScales, channelScales,
            activation.tokenScales, (cutlass::half_t*)outputData, n, m, k, stream);
    }
    if (output.dataType == fastllm::DataType::BFLOAT16) {
        return FastllmCudaW4A8DispatchForOutputType(
            activation.fp8, packedWeight, packedGroupScales, channelScales,
            activation.tokenScales, (cutlass::bfloat16_t*)outputData, n, m, k, stream);
    }
    return false;
}

static size_t FastllmCudaW4A8SourceScaleCount(const fastllm::Data &weight) {
    return weight.dataType == fastllm::DataType::INT4_W4A8
        ? weight.w4a8GroupScales.size()
        : weight.scales.size();
}

static const void *FastllmCudaW4A8SourceScaleData(const fastllm::Data &weight) {
    return weight.dataType == fastllm::DataType::INT4_W4A8
        ? static_cast<const void*>(weight.w4a8GroupScales.data())
        : static_cast<const void*>(weight.scales.data());
}

static float FastllmCudaW4A8BFloat16ToFloat(uint16_t bits) {
    uint32_t value = (uint32_t)bits << 16;
    float result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

static float FastllmCudaW4A8SourceScaleAt(const fastllm::Data &weight,
                                          size_t index) {
    if (weight.dataType == fastllm::DataType::INT4_W4A8) {
        return FastllmCudaW4A8BFloat16ToFloat(weight.w4a8GroupScales[index]);
    }
    return weight.scales[index];
}

static bool FastllmCudaW4A8BuildScaleCaches(
    const fastllm::Data &weight,
    int m,
    int k,
    cutlass::Array<W4A8MmaType, W4A8_SCALE_PACK_SIZE> **packedGroupScales,
    float **channelScales,
    size_t *packedGroupScaleBytes,
    size_t *channelScaleBytes) {
    if (packedGroupScales == nullptr ||
        channelScales == nullptr ||
        packedGroupScaleBytes == nullptr ||
        channelScaleBytes == nullptr ||
        weight.group != m / W4A8_GROUP_SIZE ||
        FastllmCudaW4A8SourceScaleCount(weight) != (size_t)k * weight.group) {
        return false;
    }

    *packedGroupScales = nullptr;
    *channelScales = nullptr;
    *packedGroupScaleBytes = 0;
    *channelScaleBytes = 0;

    size_t scaleCount = FastllmCudaW4A8SourceScaleCount(weight);
    std::vector<W4A8MmaType> groupScalesFp8(scaleCount);
    std::vector<float> hostChannelScales(k, 1.0f);
    for (size_t i = 0; i < scaleCount; ++i) {
        float scale = FastllmCudaW4A8SourceScaleAt(weight, i);
        if (!std::isfinite(scale) || scale < 0.0f) {
            return false;
        }
    }
    for (int out = 0; out < k; out++) {
        float channelAbsMax = 0.0f;
        size_t rowOffset = (size_t)out * weight.group;
        for (int group = 0; group < weight.group; group++) {
            channelAbsMax = std::max(
                channelAbsMax,
                std::fabs(FastllmCudaW4A8SourceScaleAt(weight, rowOffset + group)));
        }
        // Per-channel FP8 quantization maps the largest group scale to 56,
        // preserving E4M3 dynamic-range usage.
        float channelScale = std::max(channelAbsMax, 1.0e-10f) / 56.0f;
        hostChannelScales[out] = channelScale;
        for (int group = 0; group < weight.group; group++) {
            groupScalesFp8[(size_t)group * k + out] =
                W4A8MmaType(
                    FastllmCudaW4A8SourceScaleAt(weight, rowOffset + group) /
                    channelScale);
        }
    }

    W4A8MmaType *deviceGroupScales =
        (W4A8MmaType*)FastllmCudaMalloc(scaleCount * sizeof(W4A8MmaType));
    auto *devicePackedGroupScales =
        (cutlass::Array<W4A8MmaType, W4A8_SCALE_PACK_SIZE>*)FastllmCudaMalloc(
            scaleCount * sizeof(cutlass::Array<W4A8MmaType, W4A8_SCALE_PACK_SIZE>));
    float *deviceChannelScales = (float*)FastllmCudaMalloc((size_t)k * sizeof(float));
    if (deviceGroupScales == nullptr ||
        devicePackedGroupScales == nullptr ||
        deviceChannelScales == nullptr) {
        if (deviceGroupScales != nullptr) {
            FastllmCudaFree(deviceGroupScales);
        }
        if (devicePackedGroupScales != nullptr) {
            FastllmCudaFree(devicePackedGroupScales);
        }
        if (deviceChannelScales != nullptr) {
            FastllmCudaFree(deviceChannelScales);
        }
        return false;
    }

    FastllmCudaCopyFromHostToDevice(deviceGroupScales, groupScalesFp8.data(),
                                    scaleCount * sizeof(W4A8MmaType));
    FastllmCudaCopyFromHostToDevice(deviceChannelScales, hostChannelScales.data(),
                                    (size_t)k * sizeof(float));
    bool packed = cutlass::pack_scale_fp8(deviceGroupScales, devicePackedGroupScales, scaleCount);
    FastllmCudaFree(deviceGroupScales);
    if (!packed) {
        FastllmCudaFree(devicePackedGroupScales);
        FastllmCudaFree(deviceChannelScales);
        return false;
    }

    *packedGroupScales = devicePackedGroupScales;
    *channelScales = deviceChannelScales;
    *packedGroupScaleBytes =
        scaleCount * sizeof(cutlass::Array<W4A8MmaType, W4A8_SCALE_PACK_SIZE>);
    *channelScaleBytes = (size_t)k * sizeof(float);
    return true;
}

static void FastllmCudaW4A8ReleaseActivationScratch(W4A8ActivationScratch &scratch) {
    if (scratch.fp8 != nullptr) {
        FastllmCudaFree(scratch.fp8);
    }
    if (scratch.tokenScales != nullptr) {
        FastllmCudaFree(scratch.tokenScales);
    }
    scratch = W4A8ActivationScratch{};
}

static bool FastllmCudaW4A8QuantizeActivation(const fastllm::Data &input,
                                              const void *inputData,
                                              int n,
                                              int m,
                                              W4A8ActivationScratch &scratch) {
    if (!FastllmCudaW4A8CanQuantizeActivation(input, n, m) || inputData == nullptr) {
        return false;
    }

    FastllmCudaW4A8ReleaseActivationScratch(scratch);
    scratch.tokens = n;
    scratch.hidden = m;
    scratch.fp8Bytes = (size_t)n * m * sizeof(cutlass::float_e4m3_t);
    scratch.scaleBytes = (size_t)n * sizeof(float);
    scratch.fp8 = (cutlass::float_e4m3_t*)FastllmCudaMalloc(scratch.fp8Bytes);
    scratch.tokenScales = (float*)FastllmCudaMalloc(scratch.scaleBytes);
    if (scratch.fp8 == nullptr || scratch.tokenScales == nullptr) {
        FastllmCudaW4A8ReleaseActivationScratch(scratch);
        return false;
    }

    if (!FastllmCudaW4A8LaunchActivationQuant(
            input, inputData, scratch.fp8, scratch.tokenScales, n, m, 0)) {
        FastllmCudaW4A8ReleaseActivationScratch(scratch);
        return false;
    }
    return true;
}

static bool FastllmCudaW4A8ApplyFloatBias(const fastllm::Data &output,
                                          void *outputData,
                                          const fastllm::Data &bias,
                                          int n,
                                          int k) {
    if (bias.dims.empty()) {
        return true;
    }
    if (outputData == nullptr ||
        bias.cudaData == nullptr ||
        bias.dataType != fastllm::DataType::FLOAT32 ||
        bias.Count(0) != k) {
        return false;
    }

    int threads = 256;
    int blocks = (int)std::min<size_t>(4096, ((size_t)n * k + threads - 1) / threads);
    if (output.dataType == fastllm::DataType::FLOAT16) {
        FastllmCudaW4A8AddFloatBiasKernel<<<blocks, threads>>>(
            (half*)outputData, (const float*)bias.cudaData, n, k);
    } else if (output.dataType == fastllm::DataType::BFLOAT16) {
        FastllmCudaW4A8AddFloatBiasKernel<<<blocks, threads>>>(
            (__nv_bfloat16*)outputData, (const float*)bias.cudaData, n, k);
    } else {
        return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

static void FastllmCudaW4A8ReleaseCacheEntry(fastllm::W4A8CudaWeightCache &cache) {
    int originalDevice = FastllmCudaGetDevice();
    if (cache.deviceId >= 0 && cache.deviceId != originalDevice) {
        FastllmCudaSetDevice(cache.deviceId);
    }
    if (cache.packedWeight != nullptr) {
        FastllmCudaFree(cache.packedWeight);
    }
    if (cache.packedGroupScales != nullptr) {
        FastllmCudaFree(cache.packedGroupScales);
    }
    if (cache.channelScales != nullptr) {
        FastllmCudaFree(cache.channelScales);
    }
    if (cache.deviceId >= 0 && cache.deviceId != originalDevice) {
        FastllmCudaSetDevice(originalDevice);
    }
    cache = fastllm::W4A8CudaWeightCache();
}

static bool FastllmCudaW4A8HasPackedWeightCache(const fastllm::Data &weight,
                                                int m,
                                                int k,
                                                int deviceId) {
    auto it = weight.w4a8CudaCaches.find(deviceId);
    if (it == weight.w4a8CudaCaches.end()) {
        return false;
    }
    const fastllm::W4A8CudaWeightCache &meta = it->second;
    size_t sourceScaleCount = FastllmCudaW4A8SourceScaleCount(weight);
    return meta.magic == W4A8_CACHE_MAGIC &&
           meta.deviceId == deviceId &&
           meta.sourceType == weight.dataType &&
           meta.inChannels == m &&
           meta.outChannels == k &&
           meta.groupCnt == weight.groupCnt &&
           meta.group == weight.group &&
           meta.sourceCudaData == weight.cudaData &&
           meta.hostScales == FastllmCudaW4A8SourceScaleData(weight) &&
           meta.scaleCount == sourceScaleCount &&
           meta.sourceEncoding == weight.w4a8WeightEncoding &&
           meta.packedWeightBytes == (size_t)m * k / 2 &&
           meta.packedGroupScaleBytes ==
               sourceScaleCount * sizeof(cutlass::Array<W4A8MmaType, W4A8_SCALE_PACK_SIZE>) &&
           meta.channelScaleBytes == (size_t)k * sizeof(float);
}

static bool FastllmCudaW4A8EnsurePackedWeightCache(fastllm::Data &weight,
                                                   int m,
                                                   int k) {
    if (!FastllmCudaW4A8CanUseWeightSource(weight, m, k)) {
        return false;
    }
    int deviceId = FastllmCudaGetDevice();
    std::lock_guard<std::mutex> guard(g_w4a8WeightCacheMutex);
    if (FastllmCudaW4A8HasPackedWeightCache(weight, m, k, deviceId)) {
        return true;
    }

    auto stale = weight.w4a8CudaCaches.find(deviceId);
    if (stale != weight.w4a8CudaCaches.end()) {
        FastllmCudaW4A8ReleaseCacheEntry(stale->second);
        weight.w4a8CudaCaches.erase(stale);
    }

    size_t packedBytes = (size_t)m * k / 2;
    uint8_t *rawPackedWeight = (uint8_t*)FastllmCudaMalloc(packedBytes);
    uint8_t *cutlassPackedWeight = (uint8_t*)FastllmCudaMalloc(packedBytes);
    if (rawPackedWeight == nullptr || cutlassPackedWeight == nullptr) {
        if (rawPackedWeight != nullptr) {
            FastllmCudaFree(rawPackedWeight);
        }
        if (cutlassPackedWeight != nullptr) {
            FastllmCudaFree(cutlassPackedWeight);
        }
        return false;
    }

    int threads = 256;
    if (weight.dataType == fastllm::DataType::INT4_W4A8) {
        FastllmCudaW4A8ConvertUint4B8ToSignedInt4Kernel<<<
            (packedBytes + threads - 1) / threads, threads>>>(
                (const uint8_t*)weight.cudaData, rawPackedWeight, packedBytes);
    } else {
        size_t words = (size_t)(m / 8) * k;
        FastllmCudaW4A8PackInt4GroupToCutlassBKernel<<<
            (words + threads - 1) / threads, threads>>>(
                (const uint8_t*)weight.cudaData, (uint32_t*)rawPackedWeight, m, k);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess ||
        !FastllmCudaW4A8EncodeAndReorderInt4B(rawPackedWeight, cutlassPackedWeight, m, k)) {
        FastllmCudaFree(rawPackedWeight);
        FastllmCudaFree(cutlassPackedWeight);
        return false;
    }

    cutlass::Array<W4A8MmaType, W4A8_SCALE_PACK_SIZE> *packedGroupScales = nullptr;
    float *channelScales = nullptr;
    size_t packedGroupScaleBytes = 0;
    size_t channelScaleBytes = 0;
    if (!FastllmCudaW4A8BuildScaleCaches(weight, m, k, &packedGroupScales,
                                         &channelScales, &packedGroupScaleBytes,
                                         &channelScaleBytes)) {
        FastllmCudaFree(rawPackedWeight);
        FastllmCudaFree(cutlassPackedWeight);
        return false;
    }

    FastllmCudaFree(rawPackedWeight);
    fastllm::W4A8CudaWeightCache cache;
    cache.magic = W4A8_CACHE_MAGIC;
    cache.deviceId = deviceId;
    cache.sourceType = weight.dataType;
    cache.sourceEncoding = weight.w4a8WeightEncoding;
    cache.inChannels = m;
    cache.outChannels = k;
    cache.groupCnt = weight.groupCnt;
    cache.group = weight.group;
    cache.packedWeightBytes = packedBytes;
    cache.packedGroupScaleBytes = packedGroupScaleBytes;
    cache.channelScaleBytes = channelScaleBytes;
    cache.sourceCudaData = weight.cudaData;
    cache.hostScales = FastllmCudaW4A8SourceScaleData(weight);
    cache.scaleCount = FastllmCudaW4A8SourceScaleCount(weight);
    cache.packedWeight = cutlassPackedWeight;
    cache.packedGroupScales = packedGroupScales;
    cache.channelScales = channelScales;
    weight.w4a8CudaCaches.emplace(deviceId, cache);
    return true;
}

} // namespace

bool FastllmCudaW4A8PrepareWeightCache(fastllm::Data &weight,
                                       int inChannels,
                                       int outChannels) {
    return FastllmCudaW4A8EnsurePackedWeightCache(
        weight, inChannels, outChannels);
}

void FastllmCudaReleaseW4A8WeightCache(fastllm::Data &weight) {
    std::lock_guard<std::mutex> guard(g_w4a8WeightCacheMutex);
    for (auto &entry : weight.w4a8CudaCaches) {
        FastllmCudaW4A8ReleaseCacheEntry(entry.second);
    }
    weight.w4a8CudaCaches.clear();
}

bool FastllmCudaW4A8QuantizeActivationPerToken(
    const fastllm::Data &input, int n, int m,
    void *fp8Data, float *tokenScales) {
    void *inputData = FastllmCudaPrepareInput(input);
    if (inputData == nullptr) {
        return false;
    }
    bool ready = FastllmCudaW4A8LaunchActivationQuant(
        input, inputData, (cutlass::float_e4m3_t*)fp8Data,
        tokenScales, n, m, 0);
    FastllmCudaFinishInput(input, inputData);
    return ready;
}

bool FastllmCudaW4A8QuantizeActivationPerTokenStream(
    const fastllm::Data &input, int n, int m,
    void *fp8Data, float *tokenScales, void *streamPtr) {
    void *inputData = FastllmCudaPrepareInput(input);
    if (inputData == nullptr) {
        return false;
    }
    cudaStream_t stream = streamPtr == nullptr
        ? cudaStreamPerThread : static_cast<cudaStream_t>(streamPtr);
    bool ready = FastllmCudaW4A8LaunchActivationQuant(
        input, inputData, static_cast<cutlass::float_e4m3_t*>(fp8Data),
        tokenScales, n, m, stream);
    FastllmCudaFinishInput(input, inputData);
    return ready;
}

bool FastllmCudaInspectW4A8Activation(
    const fastllm::Data &input, int n, int m,
    std::vector<uint8_t> &fp8Bytes, std::vector<float> &tokenScales) {
    fp8Bytes.clear();
    tokenScales.clear();
    void *inputData = FastllmCudaPrepareInput(input);
    if (inputData == nullptr) {
        return false;
    }

    W4A8ActivationScratch scratch;
    bool ready = FastllmCudaW4A8QuantizeActivation(input, inputData, n, m, scratch);
    bool copied = false;
    if (ready) {
        fp8Bytes.resize(scratch.fp8Bytes);
        tokenScales.resize(n);
        cudaError_t fp8State = cudaMemcpy(
            fp8Bytes.data(), scratch.fp8, scratch.fp8Bytes, cudaMemcpyDeviceToHost);
        cudaError_t scaleState = cudaMemcpy(
            tokenScales.data(), scratch.tokenScales, scratch.scaleBytes,
            cudaMemcpyDeviceToHost);
        copied = fp8State == cudaSuccess && scaleState == cudaSuccess;
    }
    FastllmCudaW4A8ReleaseActivationScratch(scratch);
    FastllmCudaFinishInput(input, inputData);
    if (!copied) {
        fp8Bytes.clear();
        tokenScales.clear();
    }
    return copied;
}
bool TryCudaCutlassW4A8(const fastllm::Data &input, fastllm::Data &weight,
                        const fastllm::Data &bias, fastllm::Data &output,
                        int n, int m, int k) {
    // INT4_W4A8 is the model-owned compressed-tensors format and therefore
    // uses the production path automatically. The environment switches remain
    // development-only opt-ins for the legacy signed INT4_GROUP adapter.
    bool productionW4A8 = weight.dataType == fastllm::DataType::INT4_W4A8;
    bool prepareCache = productionW4A8 || FastllmCudaW4A8PrepareCacheEnabled();
    bool prepareActivation = productionW4A8 || FastllmCudaW4A8PrepareActivationEnabled();
    bool prepareOutput = productionW4A8 || FastllmCudaW4A8PrepareOutputEnabled();
    bool enableGemm = productionW4A8 || FastllmCudaW4A8GemmEnabled();
    bool validate = FastllmCudaW4A8ValidateEnabled();
    if (!prepareCache && !prepareActivation && !prepareOutput && !enableGemm && !validate) {
        return false;
    }

    int arch = FastllmCudaRuntimeArch();
    if (!FastllmCudaW4A8RuntimeArchSupported(arch)) {
        FastllmCudaW4A8TraceSkip("runtime arch is not SM90", n, m, k, arch,
                                 input, weight, bias, output);
        return false;
    }

    const char *skipReason = nullptr;
    if (!FastllmCudaW4A8LinearSemanticsSupported(input, weight, bias, output, n, m, k, &skipReason)) {
        FastllmCudaW4A8TraceSkip(skipReason == nullptr ? "linear semantics unsupported" : skipReason,
                                 n, m, k, arch, input, weight, bias, output);
        return false;
    }
    FastllmCudaW4A8TraceReady("validation", n, m, k, arch);
    if (validate && !prepareCache && !prepareActivation && !prepareOutput && !enableGemm) {
        return false;
    }

    // Enabling GEMM implies all preparation stages. For INT4_W4A8 this is the
    // normal production route; for INT4_GROUP it is still an explicit opt-in.
    if (enableGemm) {
        prepareCache = true;
        prepareActivation = true;
        prepareOutput = true;
    }

    bool weightCacheReady = false;
    if (prepareCache) {
        weightCacheReady = FastllmCudaW4A8EnsurePackedWeightCache(weight, m, k);
        if (weightCacheReady) {
            FastllmCudaW4A8TraceReady("weight-cache", n, m, k, arch);
        } else {
            FastllmCudaW4A8TraceSkip("weight cache preparation failed", n, m, k, arch,
                                     input, weight, bias, output);
            if (enableGemm) {
                return false;
            }
        }
    }

    void *inputData = nullptr;
    void *outputData = nullptr;
    if (prepareActivation) {
        inputData = FastllmCudaPrepareInput(input);
        if (inputData == nullptr) {
            FastllmCudaW4A8TraceSkip("input prepare failed", n, m, k, arch,
                                     input, weight, bias, output);
            return false;
        }
    }
    if (prepareOutput) {
        outputData = FastllmCudaPrepareOutput(output);
        if (outputData == nullptr) {
            if (inputData != nullptr) {
                FastllmCudaFinishInput(input, inputData);
            }
            FastllmCudaW4A8TraceSkip("output prepare failed", n, m, k, arch,
                                     input, weight, bias, output);
            return false;
        }
        FastllmCudaW4A8TraceReady("output-buffer", n, m, k, arch);
    }

    W4A8ActivationScratch activation;
    bool activationReady = false;
    if (prepareActivation) {
        activationReady = FastllmCudaW4A8QuantizeActivation(input, inputData, n, m, activation);
        if (activationReady) {
            FastllmCudaW4A8TraceReady("activation-quant", n, m, k, arch);
        } else {
            FastllmCudaW4A8TraceSkip("activation quantization failed", n, m, k, arch,
                                     input, weight, bias, output);
        }
    }

    bool gemmOk = false;
    if (enableGemm) {
        if (!weightCacheReady || !activationReady || outputData == nullptr) {
            FastllmCudaW4A8TraceSkip("gemm prerequisites not ready", n, m, k, arch,
                                     input, weight, bias, output);
        } else {
            cudaStream_t stream = 0;
            gemmOk = FastllmCudaW4A8RunGemm(output, activation, weight, outputData,
                                            n, m, k, stream);
            if (gemmOk) {
                gemmOk = FastllmCudaW4A8ApplyFloatBias(output, outputData, bias, n, k);
            }
            if (gemmOk) {
                FastllmCudaW4A8TraceReady("gemm", n, m, k, arch);
            } else {
                FastllmCudaW4A8TraceSkip("gemm launch failed", n, m, k, arch,
                                         input, weight, bias, output);
            }
        }
        FastllmCudaW4A8ReleaseActivationScratch(activation);
    } else {
        FastllmCudaW4A8ReleaseActivationScratch(activation);
    }
    if (prepareActivation && outputData == nullptr) {
        DeviceSync();
    }
    if (inputData != nullptr) {
        FastllmCudaFinishInput(input, inputData);
    }
    if (outputData != nullptr) {
        FastllmCudaFinishOutput(output, outputData);
    }

    return gemmOk;
}
