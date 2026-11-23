#pragma once

#include "naivert/common/executor_traits.hpp"
#include <cuda_bf16.h>
#include "naivert/common/block128_activation.cuh"

#include "naivert/common/mx_scale.cuh"
#include "naivert/common/model_geometry.cuh"
#include "naivert/common/role_layout.hpp"

namespace naivert_replica_expert_select_up_gate_silu {

using naivert::ExecutorTraits;
using naivert::StandardRoleLayout;

constexpr int kHidden = naivert::geometry::kHidden;
constexpr int kRoutedExperts = naivert::geometry::kRoutedExperts;
constexpr int kSharedExperts = naivert::geometry::kSharedExperts;
constexpr int kTotalExperts = naivert::geometry::kTotalExperts;
constexpr int kSelectedExperts = naivert::geometry::kSelectedExperts;
constexpr int kOutputSlots = naivert::geometry::kExpertSlots;
constexpr int kRouteKeptGroups = naivert::geometry::kRouteKeptGroups;
constexpr int kOutputGroups = 16;
constexpr int kOutputsPerGroup = 16;
constexpr int kPackedRows = 32;
constexpr int kPageK = 1024;
constexpr int kPages = kHidden / kPageK;
constexpr int kSlotDepth = 4;

constexpr int kMetadataOffset = 0x400;
constexpr int kBarrierGroup0Offset = 0x400 + 0x3800;
constexpr int kBarrierGroup1Offset = 0x400 + 0x3880;
constexpr int kPrivateSyncOffset = 0x400 + 0x140;
constexpr int kPaneBaseOffset = 0x400 + 0x3c00;
// Expert weight page. fp8: 32 rows x 1024 K bytes + 16 bf16 block-128 scales (32 B visible
// of 128 packed). MX8: the fp8 byte
// order + ue8m0 scales [branch 2][K block 32][row 16] (1 KiB), pane-visible in full.
constexpr bool kMx = naivert::geometry::kExpertWeightsMx;
constexpr bool kBlockScales = kMx;   // one scale per (row, 32-wide K block) in the page
constexpr int kMxPageScaleBytes = 2 * (kPageK / 32) * kOutputsPerGroup;
constexpr int kPackedPageStride =
    kMx ? 32 * kPageK + kMxPageScaleBytes : 32 * kPageK + 128;
constexpr int kBlock128PaneBytes = 0x8000 + 16 * sizeof(naivert::ExpertWeightScale);
constexpr int kPaneStride = kMx ? 0x8400 : kBlock128PaneBytes;
constexpr int kPaneVisibleBytes = kBlockScales ? kPackedPageStride : kBlock128PaneBytes;
static_assert(kPaneVisibleBytes <= kPaneStride, "a weight page must fit its pane");

#if defined(__CUDACC__)
// The page's per-(row, K block) scale at `index` = (branch * 32 + block) * 16 + row: ue8m0 (MX8).
__device__ __forceinline__ float page_block_scale(const unsigned char* scale_bytes, int index) {
  return naivert::mx::ue8m0_to_float(scale_bytes[index]);
}
#endif
constexpr int kActivationScratchOffset = kPaneBaseOffset + kSlotDepth * kPaneStride;
constexpr int kActivationScratchBytes = 0x7400;
constexpr int kStoreMetaOffset = kActivationScratchOffset + kActivationScratchBytes;
constexpr int kStoreMetaBytes = 0x400;
constexpr int kSharedBytes = kStoreMetaOffset + kStoreMetaBytes;

constexpr int kBlockThreads = StandardRoleLayout::kThreadCount;
constexpr int kConsumerThreads = StandardRoleLayout::kConsumerActiveThreads;
constexpr int kHelperThreads = StandardRoleLayout::kHelperThreads;
constexpr int kTotalTiles = kOutputSlots * kOutputGroups;

static_assert(kConsumerThreads == 256, "expert_select_up_gate_silu assumes 256 consumers");
static_assert(kHelperThreads == 128, "expert_select_up_gate_silu assumes 128 helpers");
static_assert(kHidden % kPageK == 0, "expert_select stages hidden in 1024-wide pages");
static_assert(kRoutedExperts == 256 && kSelectedExperts == 8,
              "the router owns one 32-expert group per consumer warp and selects eight");
static_assert(kBlockThreads == 384, "expert_select_up_gate_silu assumes 384 threads");

using ExpertSelectUpGateSiluTraits =
    ExecutorTraits<kSlotDepth>;

}  // namespace naivert_replica_expert_select_up_gate_silu
