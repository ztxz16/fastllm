// SPDX-License-Identifier: Apache-2.0
// Adapted from vLLM 643c125fab66d5ed5ec3143b7e764a77e7ae8ac7.
// Local changes: raw-pointer API, no PyTorch dependency, A16 group scales/offsets only.
#pragma once

#include <stdexcept>

// clang-format off
// The cutlass include order matters (annoyingly)
#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
// clang-format on

#include "cute_utils.cuh"
#include "fastllm_numeric_conversion.cuh"
#include "machete_collective_builder.cuh"
#include "machete_prepacked_layout.cuh"
#include "machete_interleaving_utils.cuh"

namespace machete {

using namespace cute;

// NOTE This kernel computes D = alpha * A * B + beta * C by computing
//   D^t = alpha * B^t * A^t + beta * C^t, this is because the wgmma
//   instructions only support sourcing from registers for the left-hand
//   operand, we want to upconvert/decompress the quantized operand in
//   register. Since the primary use case we want to support is Y = XW^t where
//   W is quantized, in this situation or right-hand operand is quantized so
//   we compute the transpose to move it to the left-hand side.
template <typename ElementA_, typename ElementB_, typename ElementD_,
          typename AccumulatorT, typename GroupScaleT, typename GroupZeroT,
          typename ChannelScaleT, typename TokenScaleT, class KernelSchedule,
          typename ScheduleConfig>
struct MacheteKernelTemplate {
  static constexpr bool with_C = false;  // not ever used
  static constexpr bool with_group_scales = !std::is_same_v<GroupScaleT, void>;
  static constexpr bool with_group_zeropoints =
      !std::is_same_v<GroupZeroT, void>;
  static constexpr bool with_channel_scales =
      !std::is_same_v<ChannelScaleT, void>;
  static constexpr bool with_token_scales = !std::is_same_v<TokenScaleT, void>;

  using MmaType = ElementA_;
  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using ElementD = ElementD_;
  using ElementC = cute::conditional_t<with_C, ElementD, void>;
  using ElementAccumulator = AccumulatorT;
  using ElementCompute = AccumulatorT;  // For Epilogue
  // Use dummy values when we don't have scales or zeropoints
  using ElementZGroup =
      cute::conditional_t<with_group_zeropoints, GroupZeroT, MmaType>;
  using ElementSGroup =
      cute::conditional_t<with_group_scales, GroupScaleT, MmaType>;
  using ElementConvertGroup =
      cute::conditional_t<with_group_scales, GroupScaleT, MmaType>;
  using ElementSChannel =
      cute::conditional_t<with_channel_scales, ChannelScaleT, AccumulatorT>;
  using ElementSToken =
      cute::conditional_t<with_token_scales, TokenScaleT, AccumulatorT>;

  using BTypeTuple = cute::conditional_t<
      with_group_scales,
      cute::conditional_t<with_group_zeropoints,
                          cute::tuple<ElementB, ElementSGroup, ElementZGroup>,
                          cute::tuple<ElementB, ElementSGroup>>,
      ElementB>;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = LayoutC;
  using LayoutScale = cutlass::layout::RowMajor;
  // not actually used since B has the prepacked layout, but required by cutlass
  using _LayoutB = cutlass::layout::ColumnMajor;

  // Interface strides expected by create_arguments (will get transposed)
  using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
  using StrideC = cutlass::detail::TagToStrideA_t<LayoutC>;
  using StrideD = cutlass::detail::TagToStrideA_t<LayoutD>;
  using StrideSGroup = cutlass::detail::TagToStrideA_t<LayoutScale>;
  using StrideZGroup = StrideSGroup;

  using LayoutA_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutA>::type;
  using LayoutC_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutC>::type;
  using LayoutD_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutD>::type;

  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using PrepackedLayoutB =
      PrepackedLayoutBTemplate<ElementA_, ElementB_, ElementConvertGroup,
                               AccumulatorT, LayoutA_Transpose, KernelSchedule>;

  static int constexpr TileShapeK =
      128 * 8 / cutlass::sizeof_bits<MmaType>::value;
  static int constexpr AlignmentA = 128 / cutlass::sizeof_bits_v<ElementA>;
  static int constexpr AlignmentB = 128 / cutlass::sizeof_bits_v<ElementB>;
  static int constexpr AlignmentC =
      (with_C) ? 128 / cutlass::sizeof_bits_v<ElementC> : 0;
  static int constexpr AlignmentD = 128 / cutlass::sizeof_bits_v<ElementD>;

  using TileShape = decltype(append(typename ScheduleConfig::TileShapeNM{},
                                    cute::Int<TileShapeK>{}));
  using ClusterShape = typename ScheduleConfig::ClusterShape;
  using EpilogueSchedule = typename ScheduleConfig::EpilogueSchedule;
  using EpilogueTileType = typename ScheduleConfig::EpilogueTileType;
  using TileScheduler = typename ScheduleConfig::TileScheduler;

  static_assert(
      (!with_channel_scales && !with_token_scales) ||
          ((with_channel_scales && with_token_scales) &&
           std::is_same_v<ElementSChannel, ElementSToken>),
      "Currently token and channel scales (if present) must be the same type");

  // Currently only supports float scales
  static_assert((with_channel_scales || with_token_scales) ||
                    (std::is_same_v<ElementSChannel, float> &&
                     std::is_same_v<ElementSToken, float>),
                "Currently token and channel scales (if present) must be float "
                "(and if one is present the other must be too)");

  using StoreEpilogueCompute = typename cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90AccFetch>;

  using EVTCompute = StoreEpilogueCompute;
  static_assert(!with_channel_scales && !with_token_scales);

  // EVTCompute
  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, TileShape, ClusterShape, EpilogueTileType,
          ElementAccumulator, ElementSChannel, ElementC, LayoutC_Transpose,
          AlignmentC, ElementD, LayoutD_Transpose, AlignmentD, EpilogueSchedule,
          EVTCompute>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::VLLMCollectiveBuilder<
          cutlass::gemm::collective::MacheteKernelTag, ArchTag, OperatorClass,
          BTypeTuple, PrepackedLayoutB, AlignmentB, ElementA, LayoutA_Transpose,
          AlignmentA, ElementAccumulator, TileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,  // Indicates ProblemShape
      CollectiveMainloop, CollectiveEpilogue, TileScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  // stride_B is unused (since B is prepacked), but still required by cutlass
  using _StrideB = cutlass::detail::TagToStrideB_t<_LayoutB>;

  using Arguments = typename Gemm::Arguments;
  using MainloopArguments = typename GemmKernel::MainloopArguments;
  using EpilogueArguments = typename GemmKernel::EpilogueArguments;

  /**
   * 将FastLLM裸指针组装为Machete的转置GEMM参数，不分配内存。
   * @param M 输入token数；A为[M,K]，D为[M,N]连续行主序。
   * @param N 输出通道数；B为预打包权重，S/Z为[K/group_size,N]。
   * @param K 输入通道数；group_size为每组连续K元素数量。
   * @return CUTLASS参数；调用方必须先检查can_implement。
   */
  static Arguments create_arguments(
      int M, int N, int K, void const* A, void const* B, void* D,
      void const* S, void const* Z, int group_size) {
    StrideA stride_A = make_stride(int64_t(K), _1{}, int64_t(0));
    auto stride_Dt = make_stride(_1{}, int64_t(N), int64_t(0));
    auto stride_S = make_stride(_1{}, int64_t(N), int64_t(0));
    MainloopArguments mainloop{};
    if constexpr (with_group_zeropoints) {
      mainloop = MainloopArguments{
          static_cast<ElementB const*>(B), _StrideB{},
          static_cast<ElementA const*>(A), stride_A,
          static_cast<ElementSGroup const*>(S), stride_S, group_size,
          static_cast<ElementZGroup const*>(Z)};
    } else {
      mainloop = MainloopArguments{
          static_cast<ElementB const*>(B), _StrideB{},
          static_cast<ElementA const*>(A), stride_A,
          static_cast<ElementSGroup const*>(S), stride_S, group_size};
    }
    EpilogueArguments epilogue{{}, nullptr, {}, static_cast<ElementD*>(D), stride_Dt};
    return Arguments{cutlass::gemm::GemmUniversalMode::kGemm,
                     {N, M, K, 1}, mainloop, epilogue};
  }

  static size_t get_workspace_size(Arguments const& args) {
    return Gemm::get_workspace_size(args);
  }

  static bool can_implement(Arguments const& args) {
    return Gemm::can_implement(args) == cutlass::Status::kSuccess;
  }

  static void run(Arguments const& args, void* workspace, cudaStream_t stream) {
    Gemm gemm_op;

    cutlass::Status status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) throw std::runtime_error("Machete kernel failed to initialize workspace");

    status = gemm_op.run(stream);
    if (status != cutlass::Status::kSuccess) throw std::runtime_error("Machete kernel failed");
  }
};

};  // namespace machete
