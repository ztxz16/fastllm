// SPDX-License-Identifier: Apache-2.0
// Adapted from vLLM 643c125fab66d5ed5ec3143b7e764a77e7ae8ac7.
// Local changes: FastLLM header names and standalone include paths; see NOTICE.md.
#pragma once

#include "cutlass/gemm/collective/collective_builder.hpp"

namespace cutlass::gemm::collective {
using namespace cute;

//
// VLLMCollectiveBuilder is a wrapper around CollectiveBuilder that allows for
// for custom kernel tags, allowing you to build custom collectives. Without
// touching the cutlass library headers, using `CutlassKernelTag` will mean it
// will resort to using the standard cutlass collective builder.
//

// Use the default Cutlass collective builder, i.e. use an unmodified cutless
// collective
struct CutlassKernelTag {};

template <class KernelTag, class ArchTag, class OpClass, class ElementA,
          class GmemLayoutA, int AlignmentA, class ElementB, class GmemLayoutB,
          int AlignmentB, class ElementAccumulator, class TileShape_MNK,
          class ClusterShape_MNK, class StageCountType,
          class KernelScheduleType, class Enable = void>
struct VLLMCollectiveBuilder {
  static_assert(sizeof(ElementA) == 0,
                "Could not build a collective for given parameters.");
};

template <class ArchTag, class OpClass, class ElementA, class GmemLayoutA,
          int AlignmentA, class ElementB, class GmemLayoutB, int AlignmentB,
          class ElementAccumulator, class TileShape_MNK, class ClusterShape_MNK,
          class StageCountType, class KernelScheduleType>
struct VLLMCollectiveBuilder<
    CutlassKernelTag, ArchTag, OpClass, ElementA, GmemLayoutA, AlignmentA,
    ElementB, GmemLayoutB, AlignmentB, ElementAccumulator, TileShape_MNK,
    ClusterShape_MNK, StageCountType, KernelScheduleType> {
  using CollectiveOp = typename CollectiveBuilder<
      ArchTag, OpClass, ElementA, GmemLayoutA, AlignmentA, ElementB,
      GmemLayoutB, AlignmentB, ElementAccumulator, TileShape_MNK,
      ClusterShape_MNK, StageCountType, KernelScheduleType>::CollectiveOp;
};

};  // namespace cutlass::gemm::collective
