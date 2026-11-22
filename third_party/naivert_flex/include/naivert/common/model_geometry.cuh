#pragma once

// Model and hardware geometry that every TP8 kernel build receives from the
// model layer (naivert/models/<model>/kernel_geometry.py provides the -D flags
// from ModelSpec and the toolchain). There are deliberately no defaults: a
// build that forgets them fails here instead of silently compiling one
// model's numbers into another model's kernels.

#ifndef NAIVERT_TP_RANKS
#error "NAIVERT_TP_RANKS must be defined by the kernel build manifest"
#endif
#ifndef NAIVERT_HIDDEN
#error "NAIVERT_HIDDEN must be defined by the kernel build manifest"
#endif
#ifndef NAIVERT_SM_COUNT
#error "NAIVERT_SM_COUNT must be defined by the kernel build manifest"
#endif
#ifndef NAIVERT_ROUTED_EXPERTS
#error "NAIVERT_ROUTED_EXPERTS must be defined by the kernel build manifest"
#endif
#ifndef NAIVERT_SHARED_EXPERTS
#error "NAIVERT_SHARED_EXPERTS must be defined by the kernel build manifest"
#endif
#ifndef NAIVERT_SELECTED_EXPERTS
#error "NAIVERT_SELECTED_EXPERTS must be defined by the kernel build manifest"
#endif
#ifndef NAIVERT_ROUTE_KEPT_GROUPS
#error "NAIVERT_ROUTE_KEPT_GROUPS must be defined by the kernel build manifest"
#endif
#ifndef NAIVERT_DENSE_INTER
#error "NAIVERT_DENSE_INTER must be defined by the kernel build manifest"
#endif
#ifndef NAIVERT_VOCAB_SHARD
#error "NAIVERT_VOCAB_SHARD must be defined by the kernel build manifest"
#endif
#ifndef NAIVERT_ROUTE_SCALE
#error "NAIVERT_ROUTE_SCALE must be defined by the kernel build manifest"
#endif
#ifndef NAIVERT_NORM_EPS
#error "NAIVERT_NORM_EPS must be defined by the kernel build manifest"
#endif
#ifndef NAIVERT_EXPERT_WEIGHT_FORMAT
#error "NAIVERT_EXPERT_WEIGHT_FORMAT must be defined by the kernel build manifest"
#endif

namespace naivert::geometry {

constexpr int kTpRanks = NAIVERT_TP_RANKS;   // tensor-parallel ranks (model spec)
constexpr int kHidden = NAIVERT_HIDDEN;      // residual width (model spec)
constexpr int kSmCount = NAIVERT_SM_COUNT;   // CTAs per persistent launch (hardware)

static_assert(kTpRanks > 1 && kTpRanks <= 8, "the mailbox lane layout holds at most eight ranks");
static_assert(kHidden > 0 && kHidden % 64 == 0, "hidden width must be a multiple of 64");
static_assert(kSmCount > 0);

// MoE and head geometry (model spec). The packed expert tables store the
// shared experts first: a shared slot addresses packed expert `slot`, a
// routed slot addresses `selected index + kSharedExperts`. One token feeds
// kExpertSlots up/gate rows into expert_down.
constexpr int kRoutedExperts = NAIVERT_ROUTED_EXPERTS;
constexpr int kSharedExperts = NAIVERT_SHARED_EXPERTS;
constexpr int kSelectedExperts = NAIVERT_SELECTED_EXPERTS;
constexpr int kTotalExperts = kRoutedExperts + kSharedExperts;
constexpr int kExpertSlots = kSelectedExperts + kSharedExperts;
// The router scores its experts as eight warp-owned groups of 32; this many
// groups survive group selection (8 = the model routes without a group limit).
constexpr int kRouteKeptGroups = NAIVERT_ROUTE_KEPT_GROUPS;
constexpr int kDenseInter = NAIVERT_DENSE_INTER;   // dense MLP intermediate per rank
constexpr int kVocabShard = NAIVERT_VOCAB_SHARD;   // lm_head columns per rank
// The selected experts' renormalised sigmoid weights are scaled by this
// (some models use routed_scaling_factor 2.5, Naive N's 1.0).
constexpr float kRouteScale = float(NAIVERT_ROUTE_SCALE);
constexpr float kNormEps = float(NAIVERT_NORM_EPS);   // RMSNorm epsilon of the model
constexpr int kExpertWeightFormat = NAIVERT_EXPERT_WEIGHT_FORMAT;
constexpr bool kExpertWeightsMx = kExpertWeightFormat == 2;
static_assert(kExpertWeightFormat == 0 || kExpertWeightFormat == 2,
              "NAIVERT_EXPERT_WEIGHT_FORMAT is 0 (fp8 block-128) or 2 (mxfp8)");

static_assert(kRoutedExperts > 0 && kSelectedExperts > 0 && kSelectedExperts <= kRoutedExperts);
static_assert(kSharedExperts == 0 || kSharedExperts == 1, "the expert kernels stage at most one shared expert");
static_assert(kRouteKeptGroups >= 1 && kRouteKeptGroups <= 8, "the router owns eight expert groups");
static_assert(kDenseInter > 0 && kVocabShard > 0);
static_assert(kRouteScale > 0.0f, "the route scale must be positive");
static_assert(kNormEps > 0.0f, "the norm epsilon must be positive");

}  // namespace naivert::geometry
