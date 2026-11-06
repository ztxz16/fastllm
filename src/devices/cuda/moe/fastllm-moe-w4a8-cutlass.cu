#include "fastllm-cuda.cuh"

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/mixed_dtype_utils.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "libtorch_stable/cutlass_extensions/epilogue/broadcast_load_epilogue_array_c3x.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

// Native FastLLM adaptation of vLLM's SM90 grouped W4A8 CUTLASS entry.
namespace fastllm_w4a8_moe_detail {

using namespace cute;

constexpr int W4A8_GROUP_SIZE = 128;
constexpr int W4A8_SCALE_PACK_SIZE = 8;

using W4A8ProblemShape =
    cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using W4A8MmaType = cutlass::float_e4m3_t;
using W4A8QuantType = cutlass::int4b_t;
using W4A8OutputType = cutlass::bfloat16_t;
using W4A8LayoutA = cutlass::layout::RowMajor;
using W4A8LayoutB = cutlass::layout::ColumnMajor;
using W4A8LayoutC = cutlass::layout::RowMajor;
using W4A8StrideA =
    cute::remove_pointer_t<cutlass::detail::TagToStrideA_t<W4A8LayoutA*>>;
using W4A8StrideB =
    cute::remove_pointer_t<cutlass::detail::TagToStrideB_t<W4A8LayoutB*>>;
using W4A8LayoutAtomQuant =
    decltype(cutlass::compute_memory_reordering_atom<W4A8MmaType>());
using W4A8LayoutBReordered = decltype(cute::tile_to_shape(
    W4A8LayoutAtomQuant{},
    Layout<Shape<int, int, Int<1>>, W4A8StrideB>{}));

template <typename ElementAcc, typename ElementD, typename TileShape>
struct FastllmW4A8GroupedScaledEpilogue {
    using Accum = cutlass::epilogue::fusion::Sm90AccFetch;
    using ChannelScale =
        cutlass::epilogue::fusion::Sm90ColOrScalarBroadcastArray<
            0, TileShape, float, Stride<Int<1>, Int<0>, Int<0>>>;
    using TokenScale =
        cutlass::epilogue::fusion::Sm90RowOrScalarBroadcastArray<
            0, TileShape, float, Stride<Int<0>, Int<1>, Int<0>>>;
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

    static ArgumentType prepare_args(const float *const *channelScales,
                                     const float *const *tokenScales) {
        typename ChannelScale::Arguments channelArgs{
            channelScales, true, {}};
        typename TokenScale::Arguments tokenArgs{
            tokenScales, true, {}};
        typename EVTCompute0::Arguments evt0Args{tokenArgs, {}, {}};
        return ArgumentType{channelArgs, evt0Args, {}};
    }
};

template <class TileShapeMN, class ClusterShapeMNK>
struct FastllmW4A8GroupedGemmKernel {
    static constexpr int TileShapeK =
        W4A8_GROUP_SIZE * W4A8_SCALE_PACK_SIZE /
        sizeof_bits<W4A8MmaType>::value;
    using TileShape =
        decltype(cute::append(TileShapeMN{}, cute::Int<TileShapeK>{}));
    using ClusterShape = ClusterShapeMNK;
    using Epilogue = FastllmW4A8GroupedScaledEpilogue<
        float, W4A8OutputType, TileShape>;
    using EVTCompute = typename Epilogue::EVTCompute;
    static constexpr int AlignmentA =
        128 / cutlass::sizeof_bits<W4A8MmaType>::value;
    static constexpr int AlignmentB =
        128 / cutlass::sizeof_bits<W4A8QuantType>::value;
    static constexpr int AlignmentD =
        128 / cutlass::sizeof_bits<W4A8OutputType>::value;

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape,
            ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
            float, float, W4A8OutputType,
            typename cutlass::layout::LayoutTranspose<W4A8LayoutC>::type*,
            AlignmentD, W4A8OutputType,
            typename cutlass::layout::LayoutTranspose<W4A8LayoutC>::type*,
            AlignmentD,
            cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative,
            EVTCompute>::CollectiveOp;

    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
            cute::tuple<W4A8QuantType,
                        cutlass::Array<W4A8MmaType, W4A8_SCALE_PACK_SIZE>>,
            W4A8LayoutBReordered*, AlignmentB, W4A8MmaType,
            typename cutlass::layout::LayoutTranspose<W4A8LayoutA>::type*,
            AlignmentA, float, TileShape, ClusterShape,
            cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
                sizeof(typename CollectiveEpilogue::SharedStorage))>,
            cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>::
            CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        W4A8ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideD = typename GemmKernel::InternalStrideD;
    using StrideS = typename CollectiveMainloop::StrideScale;

    static bool run(
        const std::vector<const W4A8MmaType*> &activationPtrs,
        const std::vector<const W4A8QuantType*> &weightPtrs,
        const std::vector<const cutlass::Array<
            W4A8MmaType, W4A8_SCALE_PACK_SIZE>*> &groupScalePtrs,
        const std::vector<const float*> &channelScalePtrs,
        const std::vector<const float*> &tokenScalePtrs,
        const std::vector<W4A8OutputType*> &outputPtrs,
        const std::vector<int> &expertCounts, int inChannels,
        int outChannels, cudaStream_t stream) {
        const int groups = static_cast<int>(expertCounts.size());
        if (groups == 0) {
            return false;
        }

        std::vector<typename W4A8ProblemShape::UnderlyingProblemShape>
            hostProblems(groups);
        std::vector<W4A8StrideA> hostStridesA(groups);
        std::vector<W4A8LayoutBReordered> hostLayoutsB(groups);
        std::vector<StrideD> hostStridesD(groups);
        std::vector<StrideS> hostStridesS(groups);
        for (int i = 0; i < groups; ++i) {
            hostProblems[i] =
                make_shape(outChannels, expertCounts[i], inChannels);
            hostStridesA[i] = W4A8StrideA{
                int64_t(inChannels), Int<1>{}, Int<0>{}};
            hostLayoutsB[i] = tile_to_shape(
                W4A8LayoutAtomQuant{},
                make_shape(outChannels, inChannels, Int<1>{}));
            hostStridesD[i] = StrideD{
                Int<1>{}, int64_t(outChannels), Int<0>{}};
            hostStridesS[i] = StrideS{
                Int<1>{}, int64_t(outChannels), int64_t(0)};
        }

        std::vector<void*> allocations;
        bool copyOk = true;
        auto copyToDevice = [&](const void *source, size_t bytes) -> void* {
            void *device = FastllmCudaMalloc(bytes);
            if (device == nullptr) {
                copyOk = false;
                return nullptr;
            }
            allocations.push_back(device);
            if (cudaMemcpyAsync(device, source, bytes, cudaMemcpyHostToDevice,
                                stream) != cudaSuccess) {
                copyOk = false;
                return nullptr;
            }
            return device;
        };

#define FASTLLM_W4A8_COPY_VECTOR(VECTOR) \
        copyToDevice((VECTOR).data(), (VECTOR).size() * sizeof((VECTOR)[0]))
        auto *deviceProblems =
            static_cast<typename W4A8ProblemShape::UnderlyingProblemShape*>(
                FASTLLM_W4A8_COPY_VECTOR(hostProblems));
        auto *deviceActivationPtrs = static_cast<const W4A8MmaType**>(
            FASTLLM_W4A8_COPY_VECTOR(activationPtrs));
        auto *deviceWeightPtrs = static_cast<const W4A8QuantType**>(
            FASTLLM_W4A8_COPY_VECTOR(weightPtrs));
        auto *deviceGroupScalePtrs =
            static_cast<const cutlass::Array<
                W4A8MmaType, W4A8_SCALE_PACK_SIZE>**>(
                FASTLLM_W4A8_COPY_VECTOR(groupScalePtrs));
        auto *deviceChannelScalePtrs = static_cast<const float**>(
            FASTLLM_W4A8_COPY_VECTOR(channelScalePtrs));
        auto *deviceTokenScalePtrs = static_cast<const float**>(
            FASTLLM_W4A8_COPY_VECTOR(tokenScalePtrs));
        auto *deviceOutputPtrs = static_cast<W4A8OutputType**>(
            FASTLLM_W4A8_COPY_VECTOR(outputPtrs));
        auto *deviceStridesA =
            static_cast<W4A8StrideA*>(FASTLLM_W4A8_COPY_VECTOR(hostStridesA));
        auto *deviceLayoutsB = static_cast<W4A8LayoutBReordered*>(
            FASTLLM_W4A8_COPY_VECTOR(hostLayoutsB));
        auto *deviceStridesD =
            static_cast<StrideD*>(FASTLLM_W4A8_COPY_VECTOR(hostStridesD));
        auto *deviceStridesS =
            static_cast<StrideS*>(FASTLLM_W4A8_COPY_VECTOR(hostStridesS));
#undef FASTLLM_W4A8_COPY_VECTOR

        bool copied = copyOk && deviceProblems != nullptr &&
                      deviceActivationPtrs != nullptr &&
                      deviceWeightPtrs != nullptr &&
                      deviceGroupScalePtrs != nullptr &&
                      deviceChannelScalePtrs != nullptr &&
                      deviceTokenScalePtrs != nullptr &&
                      deviceOutputPtrs != nullptr &&
                      deviceStridesA != nullptr &&
                      deviceLayoutsB != nullptr &&
                      deviceStridesD != nullptr &&
                      deviceStridesS != nullptr;
        bool ok = false;
        if (copied) {
            W4A8ProblemShape problemShape{groups, deviceProblems, nullptr};
            typename GemmKernel::MainloopArguments mainloop{
                deviceWeightPtrs, deviceLayoutsB,
                deviceActivationPtrs, deviceStridesA,
                deviceGroupScalePtrs, deviceStridesS, W4A8_GROUP_SIZE};
            typename GemmKernel::EpilogueArguments epilogue{
                Epilogue::prepare_args(
                    deviceChannelScalePtrs, deviceTokenScalePtrs),
                nullptr, deviceStridesD, deviceOutputPtrs, deviceStridesD};
            int deviceId = FastllmCudaGetDevice();
            cutlass::KernelHardwareInfo hardware{
                deviceId,
                cutlass::KernelHardwareInfo::
                    query_device_multiprocessor_count(deviceId)};
            typename Gemm::Arguments arguments{
                cutlass::gemm::GemmUniversalMode::kGrouped,
                problemShape, mainloop, epilogue, hardware};

            size_t workspaceBytes = Gemm::get_workspace_size(arguments);
            void *workspace = workspaceBytes == 0
                ? nullptr : FastllmCudaMalloc(workspaceBytes);
            if (workspaceBytes == 0 || workspace != nullptr) {
                Gemm gemm;
                cutlass::Status status = gemm.can_implement(arguments);
                if (status == cutlass::Status::kSuccess) {
                    status = gemm.initialize(arguments, workspace, stream);
                }
                if (status == cutlass::Status::kSuccess) {
                    status = gemm.run(stream);
                }
                ok = status == cutlass::Status::kSuccess &&
                     cudaGetLastError() == cudaSuccess;
            }
            if (workspace != nullptr) {
                FastllmCudaFree(workspace);
            }
        }
        for (void *allocation : allocations) {
            FastllmCudaFree(allocation);
        }
        return ok;
    }
};

template <typename Kernel>
bool FastllmW4A8RunGroupedKernel(
    const std::vector<const W4A8MmaType*> &activationPtrs,
    const std::vector<const W4A8QuantType*> &weightPtrs,
    const std::vector<const cutlass::Array<
        W4A8MmaType, W4A8_SCALE_PACK_SIZE>*> &groupScalePtrs,
    const std::vector<const float*> &channelScalePtrs,
    const std::vector<const float*> &tokenScalePtrs,
    const std::vector<W4A8OutputType*> &outputPtrs,
    const std::vector<int> &expertCounts, int inChannels,
    int outChannels, cudaStream_t stream) {
    return Kernel::run(
        activationPtrs, weightPtrs, groupScalePtrs, channelScalePtrs,
        tokenScalePtrs, outputPtrs, expertCounts, inChannels, outChannels,
        stream);
}

static bool FastllmW4A8DispatchGroupedGemm(
    const std::vector<const W4A8MmaType*> &activationPtrs,
    const std::vector<const W4A8QuantType*> &weightPtrs,
    const std::vector<const cutlass::Array<
        W4A8MmaType, W4A8_SCALE_PACK_SIZE>*> &groupScalePtrs,
    const std::vector<const float*> &channelScalePtrs,
    const std::vector<const float*> &tokenScalePtrs,
    const std::vector<W4A8OutputType*> &outputPtrs,
    const std::vector<int> &expertCounts, int totalTasks,
    int inChannels, int outChannels, cudaStream_t stream) {
    int averageTasks =
        (totalTasks + static_cast<int>(expertCounts.size()) - 1) /
        static_cast<int>(expertCounts.size());
#define FASTLLM_W4A8_GROUPED_RUN(TILE_M, TILE_N, CLUSTER_M) \
    FastllmW4A8RunGroupedKernel<FastllmW4A8GroupedGemmKernel< \
        Shape<Int<TILE_M>, Int<TILE_N>>, \
        Shape<Int<CLUSTER_M>, Int<1>, Int<1>>>>( \
            activationPtrs, weightPtrs, groupScalePtrs, channelScalePtrs, \
            tokenScalePtrs, outputPtrs, expertCounts, inChannels, \
            outChannels, stream)
    if (averageTasks <= 16) {
        return FASTLLM_W4A8_GROUPED_RUN(128, 16, 2);
    }
    if (averageTasks <= 32) {
        return FASTLLM_W4A8_GROUPED_RUN(256, 32, 1);
    }
    if (averageTasks <= 64) {
        return FASTLLM_W4A8_GROUPED_RUN(256, 64, 1);
    }
    if (averageTasks <= 128) {
        return FASTLLM_W4A8_GROUPED_RUN(256, 128, 2);
    }
    return FASTLLM_W4A8_GROUPED_RUN(128, 256, 2);
#undef FASTLLM_W4A8_GROUPED_RUN
}

__global__ void FastllmW4A8GatherRowsKernel(
    const __nv_bfloat16 *input, const int *routeRows,
    __nv_bfloat16 *output, int totalTasks, int hidden) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t count = (size_t)totalTasks * hidden;
    if (index < count) {
        int task = static_cast<int>(index / hidden);
        int col = static_cast<int>(index % hidden);
        output[index] = input[(size_t)routeRows[task] * hidden + col];
    }
}

__global__ void FastllmW4A8ReduceRoutesKernel(
    const __nv_bfloat16 *routeOutput, const int *routePositions,
    const float *routeScales, __nv_bfloat16 *output,
    int batch, int topk, int hidden) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t count = (size_t)batch * hidden;
    if (index >= count) {
        return;
    }
    int token = static_cast<int>(index / hidden);
    int col = static_cast<int>(index % hidden);
    float value = 0.0f;
    for (int slot = 0; slot < topk; ++slot) {
        int route = routePositions[token * topk + slot];
        value += __bfloat162float(routeOutput[(size_t)route * hidden + col]) *
                 routeScales[route];
    }
    output[index] = __float2bfloat16_rn(value);
}

static bool FastllmW4A8WeightShapeSupported(
    const fastllm::Data &weight, int inChannels, int outChannels) {
    return weight.dataType == fastllm::DataType::INT4_W4A8 &&
           weight.dims.size() == 2 &&
           weight.dims[0] == outChannels &&
           weight.dims[1] == inChannels &&
           weight.groupCnt == W4A8_GROUP_SIZE &&
           weight.group == inChannels / W4A8_GROUP_SIZE &&
           inChannels % 256 == 0 &&
           outChannels % 16 == 0;
}

static bool FastllmW4A8CollectCaches(
    fastllm::Data **weights, int weightsBatch, bool gate,
    int inChannels, int outChannels, const int *expertCounts,
    const int *expertStarts, const W4A8MmaType *activation,
    const float *tokenScales, W4A8OutputType *output,
    std::vector<const W4A8MmaType*> &activationPtrs,
    std::vector<const W4A8QuantType*> &weightPtrs,
    std::vector<const cutlass::Array<
        W4A8MmaType, W4A8_SCALE_PACK_SIZE>*> &groupScalePtrs,
    std::vector<const float*> &channelScalePtrs,
    std::vector<const float*> &tokenScalePtrs,
    std::vector<W4A8OutputType*> &outputPtrs,
    std::vector<int> &activeCounts) {
    int experts = weightsBatch / 2 - 1;
    int deviceId = FastllmCudaGetDevice();
    for (int expert = 0; expert < experts; ++expert) {
        if (expertCounts[expert] == 0) {
            continue;
        }
        fastllm::Data *weight =
            weights[(expert + 1) * 2 + (gate ? 0 : 1)];
        if (weight == nullptr ||
            !FastllmW4A8WeightShapeSupported(
                *weight, inChannels, outChannels) ||
            !FastllmCudaW4A8PrepareWeightCache(
                *weight, inChannels, outChannels)) {
            return false;
        }
        auto cache = weight->w4a8CudaCaches.find(deviceId);
        if (cache == weight->w4a8CudaCaches.end()) {
            return false;
        }
        const auto &entry = cache->second;
        int start = expertStarts[expert];
        activationPtrs.push_back(activation + (size_t)start * inChannels);
        weightPtrs.push_back(
            reinterpret_cast<const W4A8QuantType*>(entry.packedWeight));
        groupScalePtrs.push_back(
            reinterpret_cast<const cutlass::Array<
                W4A8MmaType, W4A8_SCALE_PACK_SIZE>*>(
                    entry.packedGroupScales));
        channelScalePtrs.push_back(
            reinterpret_cast<const float*>(entry.channelScales));
        tokenScalePtrs.push_back(tokenScales + start);
        outputPtrs.push_back(output + (size_t)start * outChannels);
        activeCounts.push_back(expertCounts[expert]);
    }
    return !activeCounts.empty();
}

} // namespace fastllm_w4a8_moe_detail

using namespace fastllm_w4a8_moe_detail;

bool FastllmCudaBFloat16MergeMOEW4A8GroupedIndexed(
    const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &w2,
    fastllm::Data &output, fastllm::Data **weights, int weightsBatch,
    const int *routeRows, const float *routeScales,
    const int *routePositions, const int *expertStarts,
    const int *expertCounts, int batch, int topk, int totalTasks,
    int hidden, int inter) {
    if (FastllmCudaRuntimeArch() != 90 ||
        input.dataType != fastllm::DataType::BFLOAT16 ||
        input.dataDevice != fastllm::DataDevice::CUDA ||
        weights == nullptr || weightsBatch < 4 || (weightsBatch & 1) ||
        routeRows == nullptr || routeScales == nullptr ||
        routePositions == nullptr || expertStarts == nullptr ||
        expertCounts == nullptr || batch <= 0 || topk <= 0 ||
        totalTasks != batch * topk || hidden <= 0 || inter <= 0 ||
        hidden % 256 != 0 || inter % 256 != 0) {
        return false;
    }

    int experts = weightsBatch / 2 - 1;
    for (int expert = 0; expert < experts; ++expert) {
        fastllm::Data *gateWeight = weights[(expert + 1) * 2];
        fastllm::Data *downWeight = weights[(expert + 1) * 2 + 1];
        if (gateWeight == nullptr || downWeight == nullptr ||
            !FastllmW4A8WeightShapeSupported(
                *gateWeight, hidden, inter * 2) ||
            !FastllmW4A8WeightShapeSupported(
                *downWeight, inter, hidden)) {
            return false;
        }
    }

    w1.dataDevice = fastllm::DataDevice::CUDA;
    w1.dataDeviceIds = input.dataDeviceIds;
    w1.dataType = fastllm::DataType::BFLOAT16;
    w1.Resize({totalTasks, inter});
    w1.Allocate(false);
    w2.dataDevice = fastllm::DataDevice::CUDA;
    w2.dataDeviceIds = input.dataDeviceIds;
    w2.dataType = fastllm::DataType::BFLOAT16;
    w2.Resize({totalTasks, hidden});
    w2.Allocate(false);
    output.dataDevice = fastllm::DataDevice::CUDA;
    output.dataDeviceIds = input.dataDeviceIds;
    output.dataType = fastllm::DataType::BFLOAT16;
    output.Resize(input.dims);
    output.Allocate(false);

    fastllm::Data gathered;
    gathered.dataDevice = fastllm::DataDevice::CUDA;
    gathered.dataDeviceIds = input.dataDeviceIds;
    gathered.dataType = fastllm::DataType::BFLOAT16;
    gathered.Resize({totalTasks, hidden});
    gathered.Allocate(false);
    fastllm::Data gateOutput;
    gateOutput.dataDevice = fastllm::DataDevice::CUDA;
    gateOutput.dataDeviceIds = input.dataDeviceIds;
    gateOutput.dataType = fastllm::DataType::BFLOAT16;
    gateOutput.Resize({totalTasks, inter * 2});
    gateOutput.Allocate(false);

    int *cudaRouteRows =
        (int*)FastllmCudaMalloc((size_t)totalTasks * sizeof(int));
    float *cudaRouteScales =
        (float*)FastllmCudaMalloc((size_t)totalTasks * sizeof(float));
    int *cudaRoutePositions =
        (int*)FastllmCudaMalloc((size_t)batch * topk * sizeof(int));
    W4A8MmaType *gateActivation =
        (W4A8MmaType*)FastllmCudaMalloc(
            (size_t)totalTasks * hidden * sizeof(W4A8MmaType));
    float *gateTokenScales =
        (float*)FastllmCudaMalloc((size_t)totalTasks * sizeof(float));
    W4A8MmaType *downActivation =
        (W4A8MmaType*)FastllmCudaMalloc(
            (size_t)totalTasks * inter * sizeof(W4A8MmaType));
    float *downTokenScales =
        (float*)FastllmCudaMalloc((size_t)totalTasks * sizeof(float));
    if (cudaRouteRows == nullptr || cudaRouteScales == nullptr ||
        cudaRoutePositions == nullptr || gateActivation == nullptr ||
        gateTokenScales == nullptr || downActivation == nullptr ||
        downTokenScales == nullptr) {
        FastllmCudaFree(cudaRouteRows);
        FastllmCudaFree(cudaRouteScales);
        FastllmCudaFree(cudaRoutePositions);
        FastllmCudaFree(gateActivation);
        FastllmCudaFree(gateTokenScales);
        FastllmCudaFree(downActivation);
        FastllmCudaFree(downTokenScales);
        return false;
    }

    cudaStream_t stream = 0;
    bool ok =
        cudaMemcpyAsync(cudaRouteRows, routeRows,
                        (size_t)totalTasks * sizeof(int),
                        cudaMemcpyHostToDevice, stream) == cudaSuccess &&
        cudaMemcpyAsync(cudaRouteScales, routeScales,
                        (size_t)totalTasks * sizeof(float),
                        cudaMemcpyHostToDevice, stream) == cudaSuccess &&
        cudaMemcpyAsync(cudaRoutePositions, routePositions,
                        (size_t)batch * topk * sizeof(int),
                        cudaMemcpyHostToDevice, stream) == cudaSuccess;
    if (ok) {
        int threads = 256;
        size_t elements = (size_t)totalTasks * hidden;
        FastllmW4A8GatherRowsKernel<<<
            (elements + threads - 1) / threads, threads, 0, stream>>>(
                (const __nv_bfloat16*)input.cudaData, cudaRouteRows,
                (__nv_bfloat16*)gathered.cudaData, totalTasks, hidden);
        ok = cudaGetLastError() == cudaSuccess &&
             FastllmCudaW4A8QuantizeActivationPerToken(
                 gathered, totalTasks, hidden,
                 gateActivation, gateTokenScales);
    }

    auto runGrouped = [&](bool gate) {
        std::vector<const W4A8MmaType*> activationPtrs;
        std::vector<const W4A8QuantType*> weightPtrs;
        std::vector<const cutlass::Array<
            W4A8MmaType, W4A8_SCALE_PACK_SIZE>*> groupScalePtrs;
        std::vector<const float*> channelScalePtrs;
        std::vector<const float*> tokenScalePtrs;
        std::vector<W4A8OutputType*> outputPtrs;
        std::vector<int> activeCounts;
        int inChannels = gate ? hidden : inter;
        int outChannels = gate ? inter * 2 : hidden;
        const W4A8MmaType *activation =
            gate ? gateActivation : downActivation;
        const float *tokenScales =
            gate ? gateTokenScales : downTokenScales;
        W4A8OutputType *gemmOutput =
            reinterpret_cast<W4A8OutputType*>(
                gate ? gateOutput.cudaData : w2.cudaData);
        return FastllmW4A8CollectCaches(
                   weights, weightsBatch, gate, inChannels, outChannels,
                   expertCounts, expertStarts, activation, tokenScales,
                   gemmOutput, activationPtrs, weightPtrs, groupScalePtrs,
                   channelScalePtrs, tokenScalePtrs, outputPtrs,
                   activeCounts) &&
               FastllmW4A8DispatchGroupedGemm(
                   activationPtrs, weightPtrs, groupScalePtrs,
                   channelScalePtrs, tokenScalePtrs, outputPtrs,
                   activeCounts, totalTasks, inChannels, outChannels, stream);
    };

    if (ok) {
        ok = runGrouped(true);
    }
    if (ok) {
        ok = FastllmCudaSwiglu(gateOutput, w1) &&
             FastllmCudaW4A8QuantizeActivationPerToken(
                 w1, totalTasks, inter, downActivation, downTokenScales);
    }
    if (ok) {
        ok = runGrouped(false);
    }
    if (ok) {
        int threads = 256;
        size_t elements = (size_t)batch * hidden;
        FastllmW4A8ReduceRoutesKernel<<<
            (elements + threads - 1) / threads, threads, 0, stream>>>(
                (const __nv_bfloat16*)w2.cudaData, cudaRoutePositions,
                cudaRouteScales, (__nv_bfloat16*)output.cudaData,
                batch, topk, hidden);
        ok = cudaGetLastError() == cudaSuccess;
    }

    FastllmCudaFree(cudaRouteRows);
    FastllmCudaFree(cudaRouteScales);
    FastllmCudaFree(cudaRoutePositions);
    FastllmCudaFree(gateActivation);
    FastllmCudaFree(gateTokenScales);
    FastllmCudaFree(downActivation);
    FastllmCudaFree(downTokenScales);
    return ok;
}
