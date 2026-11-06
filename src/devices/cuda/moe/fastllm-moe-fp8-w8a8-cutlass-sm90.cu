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
#include "cutlass/util/packed_stride.hpp"
#include "libtorch_stable/cutlass_extensions/epilogue/broadcast_load_epilogue_array_c3x.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>
#include <vector>

namespace fastllm_fp8_w8a8_moe_sm90_detail {

using namespace cute;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using Fp8Element = cutlass::float_e4m3_t;

template <typename ElementAccumulator, typename ElementOutput,
          typename TileShape>
struct ScaledEpilogue {
    using Accumulator = cutlass::epilogue::fusion::Sm90AccFetch;
    using ChannelScale =
        cutlass::epilogue::fusion::Sm90ColOrScalarBroadcastArray<
            0, TileShape, float, Stride<Int<1>, Int<0>, Int<0>>>;
    using TokenScale =
        cutlass::epilogue::fusion::Sm90RowOrScalarBroadcastArray<
            0, TileShape, float, Stride<Int<0>, Int<1>, Int<0>>>;
    using MultiplyToken = cutlass::epilogue::fusion::Sm90Compute<
        cutlass::multiplies, float, float,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using TokenScaledAccumulator =
        cutlass::epilogue::fusion::Sm90EVT<
            MultiplyToken, TokenScale, Accumulator>;
    using MultiplyChannel = cutlass::epilogue::fusion::Sm90Compute<
        cutlass::multiplies, ElementOutput, float,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using EVTCompute = cutlass::epilogue::fusion::Sm90EVT<
        MultiplyChannel, ChannelScale, TokenScaledAccumulator>;
    using ArgumentType = typename EVTCompute::Arguments;

    static ArgumentType prepare_args(
        const float *const *channelScales,
        const float *const *tokenScales) {
        typename ChannelScale::Arguments channelArgs{
            channelScales, true, {}};
        typename TokenScale::Arguments tokenArgs{
            tokenScales, true, {}};
        typename TokenScaledAccumulator::Arguments tokenScaledArgs{
            tokenArgs, {}, {}};
        return ArgumentType{channelArgs, tokenScaledArgs, {}};
    }
};

template <typename ElementOutput, typename TileShape, typename ClusterShape,
          bool SwapAB>
struct GroupedKernel {
    using Epilogue = ScaledEpilogue<float, ElementOutput, TileShape>;
    using EVTCompute = typename Epilogue::EVTCompute;
    static constexpr int AlignmentAB = 16;
    static constexpr int AlignmentD =
        128 / cutlass::sizeof_bits<ElementOutput>::value;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutD = cutlass::layout::RowMajor;
    using MainloopLayoutA = std::conditional_t<
        SwapAB, typename cutlass::layout::LayoutTranspose<LayoutB>::type,
        LayoutA>;
    using MainloopLayoutB = std::conditional_t<
        SwapAB, typename cutlass::layout::LayoutTranspose<LayoutA>::type,
        LayoutB>;
    using EpilogueLayout = std::conditional_t<
        SwapAB, typename cutlass::layout::LayoutTranspose<LayoutD>::type,
        LayoutD>;

    using CollectiveEpilogue =
        typename cutlass::epilogue::collective::CollectiveBuilder<
            cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape,
            ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
            float, float, void, EpilogueLayout*, AlignmentD,
            ElementOutput, EpilogueLayout*, AlignmentD,
            cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong,
            EVTCompute>::CollectiveOp;
    using CollectiveMainloop =
        typename cutlass::gemm::collective::CollectiveBuilder<
            cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
            Fp8Element, MainloopLayoutA*, AlignmentAB,
            Fp8Element, MainloopLayoutB*, AlignmentAB,
            float, TileShape, ClusterShape,
            cutlass::gemm::collective::StageCountAutoCarveout<
                static_cast<int>(
                    sizeof(typename CollectiveEpilogue::SharedStorage))>,
            cutlass::gemm::
                KernelPtrArrayTmaWarpSpecializedPingpongFP8FastAccum>::
            CollectiveOp;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA = typename GemmKernel::InternalStrideA;
    using StrideB = typename GemmKernel::InternalStrideB;
    using StrideD = typename GemmKernel::InternalStrideD;

    /**
     * 执行一个SM90 FP8 W8A8 grouped GEMM配置。
     *
     * 激活为按token动态量化的FP8 E4M3，权重为按输出通道静态量化的
     * FP8 E4M3。SwapAB配置按照vLLM的小M策略交换矩阵角色，以减少M维
     * padding；Scale角色和输出布局随矩阵交换同步转换。
     *
     * @param activationPtrs 各活跃expert的激活首地址。
     * @param weightPtrs     各活跃expert的权重首地址。
     * @param tokenScalePtrs 各活跃expert的per-token Scale首地址。
     * @param channelScalePtrs 各活跃expert的per-channel Scale首地址。
     * @param outputPtrs     各活跃expert的输出首地址。
     * @param counts         各活跃expert的token行数。
     * @param inChannels     GEMM K维。
     * @param outChannels    GEMM N维。
     * @param stream         正式推理使用的CUDA流。
     * @return true表示CUTLASS接受并提交kernel；false表示准备或提交失败。
     */
    static bool run(
        const std::vector<const Fp8Element*> &activationPtrs,
        const std::vector<const Fp8Element*> &weightPtrs,
        const std::vector<const float*> &tokenScalePtrs,
        const std::vector<const float*> &channelScalePtrs,
        const std::vector<ElementOutput*> &outputPtrs,
        const std::vector<int> &counts, int inChannels,
        int outChannels, cudaStream_t stream) {
        const int groups = static_cast<int>(counts.size());
        if (groups <= 0) {
            return false;
        }

        std::vector<typename ProblemShape::UnderlyingProblemShape>
            problems(groups);
        std::vector<StrideA> stridesA(groups);
        std::vector<StrideB> stridesB(groups);
        std::vector<StrideD> stridesD(groups);
        for (int i = 0; i < groups; ++i) {
            if constexpr (SwapAB) {
                problems[i] =
                    make_shape(outChannels, counts[i], inChannels);
                stridesA[i] = cutlass::make_cute_packed_stride(
                    StrideA{}, make_shape(outChannels, inChannels, 1));
                stridesB[i] = cutlass::make_cute_packed_stride(
                    StrideB{}, make_shape(counts[i], inChannels, 1));
                stridesD[i] = cutlass::make_cute_packed_stride(
                    StrideD{}, make_shape(outChannels, counts[i], 1));
            } else {
                problems[i] =
                    make_shape(counts[i], outChannels, inChannels);
                stridesA[i] = cutlass::make_cute_packed_stride(
                    StrideA{}, make_shape(counts[i], inChannels, 1));
                stridesB[i] = cutlass::make_cute_packed_stride(
                    StrideB{}, make_shape(outChannels, inChannels, 1));
                stridesD[i] = cutlass::make_cute_packed_stride(
                    StrideD{}, make_shape(counts[i], outChannels, 1));
            }
        }

        std::vector<void*> allocations;
        bool copied = true;
        auto upload = [&](const void *source, size_t bytes) -> void* {
            void *destination = FastllmCudaMalloc(bytes);
            if (destination == nullptr ||
                cudaMemcpyAsync(destination, source, bytes,
                                cudaMemcpyHostToDevice, stream) != cudaSuccess) {
                copied = false;
            }
            if (destination != nullptr) {
                allocations.push_back(destination);
            }
            return destination;
        };

#define FASTLLM_FP8_MOE_UPLOAD(VECTOR) \
        upload((VECTOR).data(), (VECTOR).size() * sizeof((VECTOR)[0]))
        auto *deviceProblems =
            static_cast<typename ProblemShape::UnderlyingProblemShape*>(
                FASTLLM_FP8_MOE_UPLOAD(problems));
        auto *deviceActivations = static_cast<const Fp8Element**>(
            FASTLLM_FP8_MOE_UPLOAD(activationPtrs));
        auto *deviceWeights = static_cast<const Fp8Element**>(
            FASTLLM_FP8_MOE_UPLOAD(weightPtrs));
        auto *deviceTokenScales = static_cast<const float**>(
            FASTLLM_FP8_MOE_UPLOAD(tokenScalePtrs));
        auto *deviceChannelScales = static_cast<const float**>(
            FASTLLM_FP8_MOE_UPLOAD(channelScalePtrs));
        auto *deviceOutputs = static_cast<ElementOutput**>(
            FASTLLM_FP8_MOE_UPLOAD(outputPtrs));
        auto *deviceStridesA = static_cast<StrideA*>(
            FASTLLM_FP8_MOE_UPLOAD(stridesA));
        auto *deviceStridesB = static_cast<StrideB*>(
            FASTLLM_FP8_MOE_UPLOAD(stridesB));
        auto *deviceStridesD = static_cast<StrideD*>(
            FASTLLM_FP8_MOE_UPLOAD(stridesD));
#undef FASTLLM_FP8_MOE_UPLOAD

        bool ok = false;
        if (copied) {
            ProblemShape problemShape{groups, deviceProblems, nullptr};
            auto mainloop = [&]() {
                if constexpr (SwapAB) {
                    return typename GemmKernel::MainloopArguments{
                    deviceWeights, deviceStridesA,
                    deviceActivations, deviceStridesB};
                } else {
                    return typename GemmKernel::MainloopArguments{
                    deviceActivations, deviceStridesA,
                    deviceWeights, deviceStridesB};
                }
            }();
            auto epilogue = [&]() {
                if constexpr (SwapAB) {
                    return typename GemmKernel::EpilogueArguments{
                        Epilogue::prepare_args(
                            deviceTokenScales, deviceChannelScales),
                        nullptr, deviceStridesD,
                        deviceOutputs, deviceStridesD};
                } else {
                    return typename GemmKernel::EpilogueArguments{
                    Epilogue::prepare_args(
                        deviceChannelScales, deviceTokenScales),
                    nullptr, deviceStridesD, deviceOutputs, deviceStridesD};
                }
            }();

            const int device = FastllmCudaGetDevice();
            cutlass::KernelHardwareInfo hardware{
                device,
                cutlass::KernelHardwareInfo::
                    query_device_multiprocessor_count(device)};
            typename Gemm::Arguments arguments{
                cutlass::gemm::GemmUniversalMode::kGrouped,
                problemShape, mainloop, epilogue, hardware};
            const size_t workspaceBytes =
                Gemm::get_workspace_size(arguments);
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

template <typename ElementOutput>
static bool DispatchGrouped(
    const std::vector<const Fp8Element*> &activationPtrs,
    const std::vector<const Fp8Element*> &weightPtrs,
    const std::vector<const float*> &tokenScalePtrs,
    const std::vector<const float*> &channelScalePtrs,
    const std::vector<ElementOutput*> &outputPtrs,
    const std::vector<int> &counts, int totalTasks,
    int inChannels, int outChannels, cudaStream_t stream) {
#define FASTLLM_FP8_MOE_RUN(TM, TN, TK, CM, CN, CK, SWAP) \
    GroupedKernel<ElementOutput, Shape<Int<TM>, Int<TN>, Int<TK>>, \
                  Shape<Int<CM>, Int<CN>, Int<CK>>, SWAP>::run( \
        activationPtrs, weightPtrs, tokenScalePtrs, channelScalePtrs, \
        outputPtrs, counts, inChannels, outChannels, stream)
    // 与vLLM grouped_mm_c3x_sm90保持相同的分派顺序和Tile配置。
    if (totalTasks <= 4) {
        return FASTLLM_FP8_MOE_RUN(128, 16, 128, 1, 1, 1, true);
    }
    if (totalTasks <= 64) {
        return FASTLLM_FP8_MOE_RUN(128, 16, 256, 2, 1, 1, true);
    }
    if (outChannels >= 8192) {
        return FASTLLM_FP8_MOE_RUN(64, 128, 256, 1, 8, 1, false);
    }
    if (inChannels >= 8192) {
        return FASTLLM_FP8_MOE_RUN(128, 128, 128, 1, 8, 1, false);
    }
    return FASTLLM_FP8_MOE_RUN(64, 256, 128, 1, 2, 1, false);
#undef FASTLLM_FP8_MOE_RUN
}

template <typename T>
__global__ void GatherRowsKernel(
    const T *input, const int *routeRows, T *output,
    int totalTasks, int hidden) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = static_cast<size_t>(totalTasks) * hidden;
    if (index < count) {
        const int task = static_cast<int>(index / hidden);
        const int column = static_cast<int>(index % hidden);
        output[index] =
            input[static_cast<size_t>(routeRows[task]) * hidden + column];
    }
}

template <typename T>
__device__ __forceinline__ float OutputToFloat(T value);

template <>
__device__ __forceinline__ float OutputToFloat(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <>
__device__ __forceinline__ float OutputToFloat(__half value) {
    return __half2float(value);
}

template <typename T>
__device__ __forceinline__ T FloatToOutput(float value);

template <>
__device__ __forceinline__ __nv_bfloat16 FloatToOutput(float value) {
    return __float2bfloat16_rn(value);
}

template <>
__device__ __forceinline__ __half FloatToOutput(float value) {
    return __float2half_rn(value);
}

template <typename T>
__global__ void ReduceRoutesKernel(
    const T *routeOutput, const int *routePositions,
    const float *routeScales, T *output,
    int batch, int topk, int hidden) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = static_cast<size_t>(batch) * hidden;
    if (index >= count) {
        return;
    }

    const int token = static_cast<int>(index / hidden);
    const int column = static_cast<int>(index % hidden);
    float value = 0.0f;
    for (int slot = 0; slot < topk; ++slot) {
        const int route = routePositions[token * topk + slot];
        value += OutputToFloat(
            routeOutput[static_cast<size_t>(route) * hidden + column]) *
            routeScales[route];
    }
    output[index] = FloatToOutput<T>(value);
}

static bool StandardWeightSupported(
    const fastllm::Data &weight, int inChannels, int outChannels) {
    return weight.dataType == fastllm::DataType::FP8_E4M3 &&
           weight.cudaData != nullptr &&
           weight.dims == std::vector<int>({outChannels, inChannels}) &&
           weight.blockK == 1 && weight.blockM == inChannels &&
           (weight.scales.size() == static_cast<size_t>(outChannels) ||
            weight.scales.size() == 1);
}

template <typename InputOutput, typename CutlassOutput>
static bool RunMoe(
    const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &w2,
    fastllm::Data &output, fastllm::Data **weights, int weightsBatch,
    const int *routeRows, const float *routeScales,
    const int *routePositions, const int *expertStarts,
    const int *expertCounts, int batch, int topk, int totalTasks,
    int hidden, int inter) {
    const int experts = weightsBatch / 2 - 1;
    for (int expert = 0; expert < experts; ++expert) {
        if (weights[(expert + 1) * 2] == nullptr ||
            weights[(expert + 1) * 2 + 1] == nullptr ||
            !StandardWeightSupported(
                *weights[(expert + 1) * 2], hidden, inter * 2) ||
            !StandardWeightSupported(
                *weights[(expert + 1) * 2 + 1], inter, hidden)) {
            return false;
        }
    }

    auto makeCuda = [&](fastllm::Data &data,
                        const std::vector<int> &dims) -> bool {
        data.dataDevice = fastllm::DataDevice::CUDA;
        data.dataDeviceIds = input.dataDeviceIds;
        data.dataType = input.dataType;
        data.Resize(dims);
        data.Allocate(false);
        return data.cudaData != nullptr;
    };

    fastllm::Data gathered;
    fastllm::Data gateOutput;
    if (!makeCuda(gathered, {totalTasks, hidden}) ||
        !makeCuda(gateOutput, {totalTasks, inter * 2}) ||
        !makeCuda(w1, {totalTasks, inter}) ||
        !makeCuda(w2, {totalTasks, hidden}) ||
        !makeCuda(output, input.dims)) {
        return false;
    }

    cudaStream_t stream = cudaStreamPerThread;
    int *cudaRows = static_cast<int*>(
        FastllmCudaMalloc(static_cast<size_t>(totalTasks) * sizeof(int)));
    float *cudaRouteScales = static_cast<float*>(
        FastllmCudaMalloc(static_cast<size_t>(totalTasks) * sizeof(float)));
    int *cudaPositions = static_cast<int*>(
        FastllmCudaMalloc(static_cast<size_t>(batch) * topk * sizeof(int)));
    Fp8Element *gateActivation = static_cast<Fp8Element*>(
        FastllmCudaMalloc(static_cast<size_t>(totalTasks) * hidden));
    Fp8Element *downActivation = static_cast<Fp8Element*>(
        FastllmCudaMalloc(static_cast<size_t>(totalTasks) * inter));
    float *gateScales = static_cast<float*>(
        FastllmCudaMalloc(static_cast<size_t>(totalTasks) * sizeof(float)));
    float *downScales = static_cast<float*>(
        FastllmCudaMalloc(static_cast<size_t>(totalTasks) * sizeof(float)));
    bool ok = cudaRows != nullptr && cudaRouteScales != nullptr &&
              cudaPositions != nullptr && gateActivation != nullptr &&
              downActivation != nullptr && gateScales != nullptr &&
              downScales != nullptr;
    if (ok) {
        ok = cudaMemcpyAsync(
                 cudaRows, routeRows,
                 static_cast<size_t>(totalTasks) * sizeof(int),
                 cudaMemcpyHostToDevice, stream) == cudaSuccess &&
             cudaMemcpyAsync(
                 cudaRouteScales, routeScales,
                 static_cast<size_t>(totalTasks) * sizeof(float),
                 cudaMemcpyHostToDevice, stream) == cudaSuccess &&
             cudaMemcpyAsync(
                 cudaPositions, routePositions,
                 static_cast<size_t>(batch) * topk * sizeof(int),
                 cudaMemcpyHostToDevice, stream) == cudaSuccess;
    }
    if (ok) {
        constexpr int threads = 256;
        const size_t elements = static_cast<size_t>(totalTasks) * hidden;
        GatherRowsKernel<InputOutput><<<
            (elements + threads - 1) / threads, threads, 0, stream>>>(
                static_cast<const InputOutput*>(input.cudaData), cudaRows,
                static_cast<InputOutput*>(gathered.cudaData),
                totalTasks, hidden);
        ok = cudaGetLastError() == cudaSuccess &&
             FastllmCudaW4A8QuantizeActivationPerTokenStream(
                 gathered, totalTasks, hidden,
                 gateActivation, gateScales, stream);
    }

    auto grouped = [&](bool gate) -> bool {
        const int inChannels = gate ? hidden : inter;
        const int outChannels = gate ? inter * 2 : hidden;
        Fp8Element *activation =
            gate ? gateActivation : downActivation;
        float *activationScales = gate ? gateScales : downScales;
        CutlassOutput *destination = static_cast<CutlassOutput*>(
            gate ? gateOutput.cudaData : w2.cudaData);
        std::vector<const Fp8Element*> activationPtrs;
        std::vector<const Fp8Element*> weightPtrs;
        std::vector<const float*> tokenScalePtrs;
        std::vector<const float*> channelScalePtrs;
        std::vector<CutlassOutput*> outputPtrs;
        std::vector<int> activeCounts;
        fastllm::Data emptyBias;
        for (int expert = 0; expert < experts; ++expert) {
            if (expertCounts[expert] == 0) {
                continue;
            }
            fastllm::Data &weight =
                *weights[(expert + 1) * 2 + (gate ? 0 : 1)];
            FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(
                weight, emptyBias, outChannels);
            if (weight.extraCudaData.empty() ||
                weight.extraCudaData[0] == nullptr) {
                return false;
            }
            const int start = expertStarts[expert];
            activationPtrs.push_back(
                activation + static_cast<size_t>(start) * inChannels);
            weightPtrs.push_back(
                static_cast<const Fp8Element*>(weight.cudaData));
            tokenScalePtrs.push_back(activationScales + start);
            channelScalePtrs.push_back(
                static_cast<const float*>(weight.extraCudaData[0]));
            outputPtrs.push_back(
                destination + static_cast<size_t>(start) * outChannels);
            activeCounts.push_back(expertCounts[expert]);
        }
        return !activeCounts.empty() && DispatchGrouped(
            activationPtrs, weightPtrs, tokenScalePtrs, channelScalePtrs,
            outputPtrs, activeCounts, totalTasks,
            inChannels, outChannels, stream);
    };

    if (ok) {
        ok = grouped(true);
    }
    if (ok) {
        ok = FastllmCudaSwiglu(gateOutput, w1) &&
             FastllmCudaW4A8QuantizeActivationPerTokenStream(
                 w1, totalTasks, inter, downActivation, downScales, stream);
    }
    if (ok) {
        ok = grouped(false);
    }
    if (ok) {
        constexpr int threads = 256;
        const size_t elements = static_cast<size_t>(batch) * hidden;
        ReduceRoutesKernel<InputOutput><<<
            (elements + threads - 1) / threads, threads, 0, stream>>>(
                static_cast<const InputOutput*>(w2.cudaData),
                cudaPositions, cudaRouteScales,
                static_cast<InputOutput*>(output.cudaData),
                batch, topk, hidden);
        ok = cudaGetLastError() == cudaSuccess;
    }

    FastllmCudaFree(cudaRows);
    FastllmCudaFree(cudaRouteScales);
    FastllmCudaFree(cudaPositions);
    FastllmCudaFree(gateActivation);
    FastllmCudaFree(downActivation);
    FastllmCudaFree(gateScales);
    FastllmCudaFree(downScales);
    return ok;
}

} // namespace fastllm_fp8_w8a8_moe_sm90_detail

/**
 * 执行SM90标准FP8 W8A8 CUTLASS grouped MoE。
 *
 * 本入口接收CPU端已经按expert整理的路由元数据，在cudaStreamPerThread上
 * 完成路由激活聚集、两次per-token FP8动态量化、两次CUTLASS grouped
 * GEMM、SwiGLU以及top-k加权归并。权重为FP8 E4M3 [N,K]，Scale为
 * tensorwise或per-output-channel；blockwise权重由其他后端处理。
 *
 * @param input          FP16或BF16输入，形状为[batch, hidden]。
 * @param w1             SwiGLU输出临时张量，形状为[batch*topk, inter]。
 * @param w2             down输出临时张量，形状为[batch*topk, hidden]。
 * @param output         最终输出，形状与input相同、类型与input相同。
 * @param weights        expert权重数组，第0对保留给shared expert。
 * @param weightsBatch   weights中的指针数量。
 * @param routeRows      按expert排列的原始token行号。
 * @param routeScales    按expert排列的路由权重。
 * @param routePositions 原始[token,topk]位置到grouped行号的映射。
 * @param expertStarts   每个routed expert的起始行。
 * @param expertCounts   每个routed expert的有效行数。
 * @param batch          输入token数。
 * @param topk           每个token选择的expert数。
 * @param totalTasks     routed行数，必须等于batch*topk。
 * @param hidden         模型隐藏维度。
 * @param inter          expert中间维度。
 * @return true表示完整routed MoE成功；false表示语义不支持或CUDA提交失败。
 */
bool FastllmCudaMergeMOEFp8W8A8CutlassSm90(
    const fastllm::Data &input, fastllm::Data &w1, fastllm::Data &w2,
    fastllm::Data &output, fastllm::Data **weights, int weightsBatch,
    const int *routeRows, const float *routeScales,
    const int *routePositions, const int *expertStarts,
    const int *expertCounts, int batch, int topk, int totalTasks,
    int hidden, int inter) {
    using namespace fastllm_fp8_w8a8_moe_sm90_detail;

    if (FastllmCudaRuntimeArch() != 90 ||
        (input.dataType != fastllm::DataType::FLOAT16 &&
         input.dataType != fastllm::DataType::BFLOAT16) ||
        input.dataDevice != fastllm::DataDevice::CUDA ||
        weights == nullptr || weightsBatch < 4 || (weightsBatch & 1) ||
        routeRows == nullptr || routeScales == nullptr ||
        routePositions == nullptr || expertStarts == nullptr ||
        expertCounts == nullptr || batch <= 0 || topk <= 0 ||
        totalTasks != batch * topk || hidden <= 0 || inter <= 0 ||
        hidden % 16 != 0 || inter % 16 != 0) {
        return false;
    }

    if (input.dataType == fastllm::DataType::FLOAT16) {
        return RunMoe<__half, cutlass::half_t>(
            input, w1, w2, output, weights, weightsBatch,
            routeRows, routeScales, routePositions,
            expertStarts, expertCounts, batch, topk,
            totalTasks, hidden, inter);
    }
    return RunMoe<__nv_bfloat16, cutlass::bfloat16_t>(
        input, w1, w2, output, weights, weightsBatch,
        routeRows, routeScales, routePositions,
        expertStarts, expertCounts, batch, topk,
        totalTasks, hidden, inter);
}
