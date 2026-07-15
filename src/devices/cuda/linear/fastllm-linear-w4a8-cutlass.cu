#include "fastllm-cuda.cuh"
#include "libtorch_stable/cutlass_extensions/epilogue/broadcast_load_epilogue_c3x.hpp"
#include "libtorch_stable/quantization/cutlass_w4a8/w4a8_utils.cuh"

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

static constexpr int W4A8_PACKED_WEIGHT_IDX = 64;
static constexpr int W4A8_PACKED_GROUP_SCALES_IDX = 65;
static constexpr int W4A8_CHANNEL_SCALES_IDX = 66;
static constexpr uint64_t W4A8_CACHE_MAGIC = 0x5734413843414348ULL; // "W4A8CACH"
static constexpr int W4A8_GROUP_SIZE = 128;
static constexpr int W4A8_SCALE_PACK_SIZE = 8;

struct W4A8WeightCacheMeta {
    uint64_t magic = W4A8_CACHE_MAGIC;
    fastllm::DataType sourceType = fastllm::DataType::FLOAT32;
    int inChannels = 0;
    int outChannels = 0;
    int groupCnt = 0;
    int group = 0;
    size_t packedWeightBytes = 0;
    size_t packedGroupScaleBytes = 0;
    size_t channelScaleBytes = 0;
    const void *sourceCudaData = nullptr;
    const float *hostScales = nullptr;
    size_t scaleCount = 0;
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
    using ChannelScale = cutlass::epilogue::fusion::Sm90ColOrScalarBroadcast<
        0, TileShape, float, cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>>;
    using TokenScale = cutlass::epilogue::fusion::Sm90RowOrScalarBroadcast<
        0, TileShape, float, cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

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
        typename ChannelScale::Arguments channelArgs{channelScales, true};
        typename TokenScale::Arguments tokenArgs{tokenScales, true};
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

static std::unordered_map<const fastllm::Data*, W4A8WeightCacheMeta> g_w4a8WeightCacheMetas;

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
    return arch >= 900 && arch < 1000;
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

static bool FastllmCudaW4A8CanUseInt4GroupSource(const fastllm::Data &weight,
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
    return FastllmCudaW4A8CanUseInt4GroupSource(weight, m, k, reason);
}

static bool FastllmCudaW4A8CanUseInt4GroupSource(const fastllm::Data &weight,
                                                 int m,
                                                 int k,
                                                 const char **reason) {
    if (weight.dataType != fastllm::DataType::INT4_GROUP) {
        if (reason != nullptr) {
            *reason = "weight dtype is not INT4_GROUP";
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
    if (weight.groupCnt != 128 || weight.group != m / 128) {
        if (reason != nullptr) {
            *reason = "weight group is not 128";
        }
        return false;
    }
    if (weight.scales.size() != (size_t)k * weight.group ||
        weight.mins.size() != (size_t)k * weight.group) {
        if (reason != nullptr) {
            *reason = "weight scale/min shape mismatch";
        }
        return false;
    }
    if (!FastllmCudaW4A8BasicShapeSupported(m, k)) {
        if (reason != nullptr) {
            *reason = "m/k is not 128-aligned";
        }
        return false;
    }

    // vLLM cutlass_w4a8 consumes signed INT4 weights with scale-only group
    // quantization. FastLLM INT4_GROUP can be repacked without changing
    // quantization math only when min + uint4 * scale == (uint4 - 8) * scale.
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

__global__ void FastllmCudaW4A8PackInt4GroupToVllmBKernel(const uint8_t *src,
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
    dst[idx] = packed;
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

    __shared__ float reduce[256];
    float localMax = 0.0f;
    size_t rowOffset = (size_t)row * hidden;
    for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
        float v = FastllmCudaW4A8ToFloat(input[rowOffset + col]);
        localMax = fmaxf(localMax, fabsf(v));
    }
    reduce[threadIdx.x] = localMax;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    float scale = fmaxf(reduce[0], 1.0e-10f) * (1.0f / 448.0f);
    for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
        float v = FastllmCudaW4A8ToFloat(input[rowOffset + col]) / scale;
        v = fminf(448.0f, fmaxf(-448.0f, v));
        fp8[rowOffset + col] = cutlass::float_e4m3_t(v);
    }
    if (threadIdx.x == 0) {
        tokenScales[row] = scale;
    }
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

    if (!vllm::cutlass_w4a8_utils::unified_encode_int4b(rawPtr, packedPtr, numInt4Elems)) {
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
    if (activation.fp8 == nullptr ||
        activation.tokenScales == nullptr ||
        outputData == nullptr ||
        (int)weight.extraCudaData.size() <= W4A8_CHANNEL_SCALES_IDX ||
        weight.extraCudaData[W4A8_PACKED_WEIGHT_IDX] == nullptr ||
        weight.extraCudaData[W4A8_PACKED_GROUP_SCALES_IDX] == nullptr ||
        weight.extraCudaData[W4A8_CHANNEL_SCALES_IDX] == nullptr) {
        return false;
    }

    const auto *packedWeight =
        reinterpret_cast<const W4A8QuantType*>(weight.extraCudaData[W4A8_PACKED_WEIGHT_IDX]);
    const auto *packedGroupScales =
        reinterpret_cast<const cutlass::Array<W4A8MmaType, W4A8_SCALE_PACK_SIZE>*>(
            weight.extraCudaData[W4A8_PACKED_GROUP_SCALES_IDX]);
    const float *channelScales =
        reinterpret_cast<const float*>(weight.extraCudaData[W4A8_CHANNEL_SCALES_IDX]);

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
        weight.scales.size() != (size_t)k * weight.group) {
        return false;
    }

    *packedGroupScales = nullptr;
    *channelScales = nullptr;
    *packedGroupScaleBytes = 0;
    *channelScaleBytes = 0;

    size_t scaleCount = weight.scales.size();
    std::vector<W4A8MmaType> groupScalesFp8(scaleCount);
    std::vector<float> hostChannelScales(k, 1.0f);
    for (int out = 0; out < k; out++) {
        float channelAbsMax = 0.0f;
        size_t rowOffset = (size_t)out * weight.group;
        for (int group = 0; group < weight.group; group++) {
            channelAbsMax = std::max(channelAbsMax, std::fabs(weight.scales[rowOffset + group]));
        }
        float channelScale = std::max(channelAbsMax, 1.0e-10f);
        hostChannelScales[out] = channelScale;
        for (int group = 0; group < weight.group; group++) {
            groupScalesFp8[rowOffset + group] =
                W4A8MmaType(weight.scales[rowOffset + group] / channelScale);
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

    dim3 grid(n);
    if (input.dataType == fastllm::DataType::FLOAT16) {
        FastllmCudaW4A8QuantizeActivationPerTokenKernel<<<grid, 256>>>(
            (const half*)inputData, scratch.fp8, scratch.tokenScales, n, m);
    } else {
        FastllmCudaW4A8QuantizeActivationPerTokenKernel<<<grid, 256>>>(
            (const __nv_bfloat16*)inputData, scratch.fp8, scratch.tokenScales, n, m);
    }
    if (cudaGetLastError() != cudaSuccess) {
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

static void FastllmCudaW4A8ReleasePackedWeightCache(fastllm::Data &weight) {
    for (int idx : {W4A8_PACKED_WEIGHT_IDX,
                    W4A8_PACKED_GROUP_SCALES_IDX,
                    W4A8_CHANNEL_SCALES_IDX}) {
        if ((int)weight.extraCudaData.size() > idx) {
            if (weight.extraCudaData[idx] != nullptr) {
                FastllmCudaFree(weight.extraCudaData[idx]);
                weight.extraCudaData[idx] = nullptr;
            }
        }
    }
    g_w4a8WeightCacheMetas.erase(&weight);
}

static bool FastllmCudaW4A8HasPackedWeightCache(const fastllm::Data &weight,
                                                int m,
                                                int k) {
    if ((int)weight.extraCudaData.size() <= W4A8_CHANNEL_SCALES_IDX ||
        weight.extraCudaData[W4A8_PACKED_WEIGHT_IDX] == nullptr ||
        weight.extraCudaData[W4A8_PACKED_GROUP_SCALES_IDX] == nullptr ||
        weight.extraCudaData[W4A8_CHANNEL_SCALES_IDX] == nullptr) {
        return false;
    }
    auto it = g_w4a8WeightCacheMetas.find(&weight);
    if (it == g_w4a8WeightCacheMetas.end()) {
        return false;
    }
    const W4A8WeightCacheMeta &meta = it->second;
    return meta.magic == W4A8_CACHE_MAGIC &&
           meta.sourceType == weight.dataType &&
           meta.inChannels == m &&
           meta.outChannels == k &&
           meta.groupCnt == weight.groupCnt &&
           meta.group == weight.group &&
           meta.sourceCudaData == weight.cudaData &&
           meta.hostScales == weight.scales.data() &&
           meta.scaleCount == weight.scales.size() &&
           meta.packedWeightBytes == (size_t)m * k / 2 &&
           meta.packedGroupScaleBytes ==
               weight.scales.size() * sizeof(cutlass::Array<W4A8MmaType, W4A8_SCALE_PACK_SIZE>) &&
           meta.channelScaleBytes == (size_t)k * sizeof(float);
}

static bool FastllmCudaW4A8EnsurePackedWeightCache(fastllm::Data &weight,
                                                   int m,
                                                   int k) {
    if (!FastllmCudaW4A8CanUseInt4GroupSource(weight, m, k)) {
        return false;
    }
    if (FastllmCudaW4A8HasPackedWeightCache(weight, m, k)) {
        return true;
    }

    FastllmCudaW4A8ReleasePackedWeightCache(weight);

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
    size_t words = (size_t)(m / 8) * k;
    FastllmCudaW4A8PackInt4GroupToVllmBKernel<<<(words + threads - 1) / threads, threads>>>(
        (const uint8_t*)weight.cudaData, (uint32_t*)rawPackedWeight, m, k);
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
    if ((int)weight.extraCudaData.size() <= W4A8_CHANNEL_SCALES_IDX) {
        weight.extraCudaData.resize(W4A8_CHANNEL_SCALES_IDX + 1, nullptr);
    }
    weight.extraCudaData[W4A8_PACKED_WEIGHT_IDX] = cutlassPackedWeight;
    weight.extraCudaData[W4A8_PACKED_GROUP_SCALES_IDX] = packedGroupScales;
    weight.extraCudaData[W4A8_CHANNEL_SCALES_IDX] = channelScales;
    g_w4a8WeightCacheMetas[&weight] = W4A8WeightCacheMeta{
        W4A8_CACHE_MAGIC,
        weight.dataType,
        m,
        k,
        weight.groupCnt,
        weight.group,
        packedBytes,
        packedGroupScaleBytes,
        channelScaleBytes,
        weight.cudaData,
        weight.scales.data(),
        weight.scales.size(),
    };
    return true;
}

} // namespace

bool TryCudaCutlassW4A8(const fastllm::Data &input, fastllm::Data &weight,
                        const fastllm::Data &bias, fastllm::Data &output,
                        int n, int m, int k) {
    bool prepareCache = FastllmCudaW4A8PrepareCacheEnabled();
    bool prepareActivation = FastllmCudaW4A8PrepareActivationEnabled();
    bool prepareOutput = FastllmCudaW4A8PrepareOutputEnabled();
    bool enableGemm = FastllmCudaW4A8GemmEnabled();
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

    // Development guard: W4A8 GEMM is allowed to take over DoCudaLinear only
    // when this explicit switch is set. Later stages can remove this or flip it
    // to an opt-out guard after correctness and performance are verified.
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
