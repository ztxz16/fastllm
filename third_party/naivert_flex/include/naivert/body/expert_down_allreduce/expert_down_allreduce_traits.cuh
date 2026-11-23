#pragma once

#include <cstdint>

#include "naivert/common/executor_traits.hpp"
#include "naivert/common/mx_scale.cuh"
#include "naivert/common/model_geometry.cuh"
#include "naivert/common/role_layout.hpp"


#ifndef NAIVERT_EXPERT_DOWN_ALLREDUCE_INTER
#define NAIVERT_EXPERT_DOWN_ALLREDUCE_INTER 256
#endif

#ifndef NAIVERT_EXPERT_DOWN_ALLREDUCE_OUTPUTS_PER_TILE
#error "NAIVERT_EXPERT_DOWN_ALLREDUCE_OUTPUTS_PER_TILE must come from the model's kernel geometry"
#endif

namespace naivert_replica_expert_down_allreduce {

using naivert::ExecutorTraits;
using naivert::StandardRoleLayout;

constexpr int kTotalExperts = naivert::geometry::kTotalExperts;
constexpr int kSelectedExperts = naivert::geometry::kSelectedExperts;
constexpr int kSharedExperts = naivert::geometry::kSharedExperts;
constexpr int kOutputSlots = naivert::geometry::kExpertSlots;
constexpr int kHidden = naivert::geometry::kHidden;
constexpr int kInter = NAIVERT_EXPERT_DOWN_ALLREDUCE_INTER;
constexpr int kOutputsPerTile = NAIVERT_EXPERT_DOWN_ALLREDUCE_OUTPUTS_PER_TILE;
#ifndef NAIVERT_EXPERT_DOWN_W8A8
#define NAIVERT_EXPERT_DOWN_W8A8 0
#endif
constexpr bool kW8A8 = NAIVERT_EXPERT_DOWN_W8A8 != 0;
#ifndef NAIVERT_EXPERT_DOWN_ALLREDUCE_SLOT_DEPTH
#define NAIVERT_EXPERT_DOWN_ALLREDUCE_SLOT_DEPTH 2
#endif
constexpr int kSlotDepth = NAIVERT_EXPERT_DOWN_ALLREDUCE_SLOT_DEPTH;
constexpr int kBlockThreads = StandardRoleLayout::kThreadCount;
constexpr int kConsumerThreads = StandardRoleLayout::kConsumerActiveThreads;
constexpr int kHelperThreads = StandardRoleLayout::kHelperThreads;

constexpr int kBarrierGroup0Offset = 0x3c00;
constexpr int kBarrierGroup1Offset = 0x3c80;
constexpr int kPayloadBase = 0x4000;
constexpr bool kMx = naivert::geometry::kExpertWeightsMx;   // E4m3 tile + ue8m0 [row][K block of 32]
constexpr int kMxScaleBytes = kOutputsPerTile * (kInter / 32);
constexpr int kPayloadStride = kOutputsPerTile * kInter + (kMx ? kMxScaleBytes : 0);
static_assert(kPayloadStride % 128 == 0, "ring slots stay 128-byte aligned");
constexpr int kSharedBytes =
    kPayloadBase + kSlotDepth * kPayloadStride;
// FP8 weight tiles use pack_down's chunk order; MX8 appends per-row K32 scales.
// Each tile is contiguous at (expert * kTiles + tile) * kWeightPaneBytes.
// W8A8 keeps the same tile extent, prepacked as two K128 UMMA SW128 panels.
constexpr int kWeightPaneBytes =
    kMx ? kOutputsPerTile * kInter + kMxScaleBytes : kOutputsPerTile * kInter;
static_assert(kWeightPaneBytes <= kPayloadStride, "a weight tile must fit its pane");
constexpr int kTiles = kHidden / kOutputsPerTile;
// pack_down's tile: kFullMTiles 16-row MMA tiles plus an optional 8-row tail
// (hidden 7168: 56 = 3 x 16 + 8, Naive N 64 = 4 x 16).
constexpr int kFullMTiles = kOutputsPerTile / 16;
constexpr int kTailRows = kOutputsPerTile - kFullMTiles * 16;
static_assert(kFullMTiles >= 1 && kFullMTiles <= 4 && (kTailRows == 0 || kTailRows == 8),
              "a down tile is up to four 16-row MMA tiles plus at most one 8-row tail");
static_assert(kHidden % kOutputsPerTile == 0, "tile width must divide hidden");
static_assert(kTiles <= 128, "expert_down tiles must fit one slab: 32 CTA records x 4 warps");
constexpr int kPartialRows = 64;
constexpr int kPartialOffset = 0x3000;
static_assert(!kW8A8 || (!kMx && kHidden == 4096 &&
                        kOutputsPerTile == 64 && kOutputSlots == 8 && kInter == 256),
              "native W8A8 down currently uses Naive N block128 geometry");

static_assert(kOutputsPerTile > 0,
              "expert_down_allreduce tile width must be positive");
static_assert(kOutputSlots == kSharedExperts + kSelectedExperts && kOutputSlots <= 32,
              "expert_down_allreduce consumes one up/gate row per shared and selected expert, one lane each");
static_assert(
    kBlockThreads == 288 || kBlockThreads == 384,
    "ExpertDown supports the production or compact CTA layout");
static_assert(kConsumerThreads == 256,
              "expert_down_allreduce follows the 256-consumer role layout");
static_assert(
    kHelperThreads == 32 || kHelperThreads == 128,
    "ExpertDown supports the production or compact helper layout");
static_assert(kPartialOffset + kPartialRows * int(sizeof(float)) <=
                  kBarrierGroup0Offset,
              "expert_down_allreduce partial scratch overlaps barriers");
static_assert(
    kPartialOffset + 8 * kPartialRows * int(sizeof(float)) +
            ((kOutputsPerTile + 1) / 2) *
                int(sizeof(std::uint32_t)) <=
        kBarrierGroup0Offset,
    "expert_down_allreduce staged mailbox payload overlaps barriers");
using ExpertDownAllreduceTraits =
    ExecutorTraits<kSlotDepth>;

}  // namespace naivert_replica_expert_down_allreduce
