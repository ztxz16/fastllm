#pragma once

#include "naivert/common/executor_traits.hpp"
#include "naivert/common/model_geometry.cuh"
#include "naivert/common/role_layout.hpp"
#include "naivert/common/expert_activation.hpp"

#ifndef NAIVERT_RMSNORM_HEAD_PROJ_OUTPUTS_PER_TILE
#define NAIVERT_RMSNORM_HEAD_PROJ_OUTPUTS_PER_TILE 16
#endif

namespace naivert_replica_rmsnorm_head_proj {

using naivert::ExecutorTraits;
using naivert::StandardRoleLayout;

constexpr int kHidden = naivert::geometry::kHidden;
constexpr int kVocabShard = naivert::geometry::kVocabShard;
constexpr int kOutputsPerTile = NAIVERT_RMSNORM_HEAD_PROJ_OUTPUTS_PER_TILE;
constexpr int kWeightPage = 1024;
constexpr int kWeightPages = kHidden / kWeightPage;
constexpr int kOutputTiles = kVocabShard / kOutputsPerTile;
constexpr int kPackedWeightRows = kOutputTiles * kWeightPages;
constexpr int kGridCtas = naivert::geometry::kSmCount;
constexpr int kRmsBootstrapCtas = 16;
constexpr int kSlotDepth = 4;
constexpr int kBlockThreads = StandardRoleLayout::kThreadCount;
constexpr int kConsumerThreads = StandardRoleLayout::kConsumerActiveThreads;
constexpr int kHelperThreads = StandardRoleLayout::kHelperThreads;

constexpr int kBarrierGroup0Offset = 0x3d00;
constexpr int kBarrierGroup1Offset = 0x3d80;
constexpr int kMetadataOffset = 0x400;
constexpr int kScratchOffset = 0x4000;
constexpr int kHiddenPageBytes =
    kWeightPage * int(sizeof(__nv_bfloat16));
constexpr int kWeightPaneOffset = 0x1000;
constexpr int kWeightPaneBytes =
    kOutputsPerTile * kWeightPage * int(sizeof(__nv_bfloat16));
constexpr int kScratchBytes = 0xa000;
constexpr int kReductionOffset =
    kWeightPaneOffset + kWeightPaneBytes + 0x100;
constexpr int kNormalizedOffset =
    kScratchOffset + kSlotDepth * kScratchBytes;
constexpr int kNormalizedBytes = kHidden * int(sizeof(__nv_bfloat16));
constexpr int kBootstrapReductionOffset =
    kNormalizedOffset + kNormalizedBytes;
constexpr int kSharedBytes = 0x30000;

static_assert(kHidden % kWeightPage == 0,
              "rmsnorm_head_proj hidden must divide into 1024-wide pages");
static_assert(kVocabShard % kOutputsPerTile == 0,
              "rmsnorm_head_proj output tiles must divide vocab");
static_assert(kHiddenPageBytes <= kWeightPaneOffset,
              "staged hidden page must fit before weight pane");
static_assert(kWeightPaneOffset + kWeightPaneBytes <= kReductionOffset,
              "staged weight page must fit before reduction scratch");
static_assert(kReductionOffset + 8 * int(sizeof(float)) <= kScratchBytes,
              "RMS reduction scratch must fit in each pane");
static_assert(kBootstrapReductionOffset + 8 * int(sizeof(float)) <=
                  kSharedBytes,
              "normalized hidden and bootstrap reduction must fit");
static_assert(kBlockThreads == 384,
              "rmsnorm_head_proj uses the SASS-observed 384-thread CTA");
static_assert(kConsumerThreads == 256,
              "rmsnorm_head_proj follows the 256-consumer role layout");
static_assert(kHelperThreads == 128,
              "rmsnorm_head_proj follows the 128-helper role layout");

using RMSNormHeadProjTraits =
    ExecutorTraits<kSlotDepth>;

// rows8 (verify-8): the rows body's panes carry eight hidden pages (16 KiB) plus the
// 32 KiB weight page, four of them after the control block. The rows8 build
// reads packed weights directly: four 16 KiB hidden panes, 4 KiB control,
// and a separate 4 KiB FP32 reduction buffer use 72 KiB in total.
constexpr int kRows8SharedBytes = 0x12000;   // The tensor head's map for every eight-row build
static_assert(kRows8SharedBytes <= 0x37000,
              "head rows8 shared layout must fit the SM100 CTA limit");


}  // namespace naivert_replica_rmsnorm_head_proj
