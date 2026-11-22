#pragma once

// FastLLM adaptation: retain only geometry consumed by the vendored draft
// kernels. The upstream MoE/router geometry is outside this subset.
// See third_party/naivert_draft/UPSTREAM.json for the original source hash.

#ifndef NAIVERT_TP_RANKS
#error "NAIVERT_TP_RANKS must be defined by the kernel build manifest"
#endif
#ifndef NAIVERT_HIDDEN
#error "NAIVERT_HIDDEN must be defined by the kernel build manifest"
#endif
#ifndef NAIVERT_SM_COUNT
#error "NAIVERT_SM_COUNT must be defined by the kernel build manifest"
#endif
#ifndef NAIVERT_VOCAB_SHARD
#error "NAIVERT_VOCAB_SHARD must be defined by the kernel build manifest"
#endif
#ifndef NAIVERT_NORM_EPS
#error "NAIVERT_NORM_EPS must be defined by the kernel build manifest"
#endif

namespace naivert::geometry {
constexpr int kTpRanks = NAIVERT_TP_RANKS;
constexpr int kHidden = NAIVERT_HIDDEN;
constexpr int kSmCount = NAIVERT_SM_COUNT;
constexpr int kVocabShard = NAIVERT_VOCAB_SHARD;
constexpr float kNormEps = float(NAIVERT_NORM_EPS);

static_assert(kTpRanks > 1 && kTpRanks <= 8, "the mailbox lane layout holds at most eight ranks");
static_assert(kHidden > 0 && kHidden % 64 == 0, "hidden width must be a multiple of 64");
static_assert(kSmCount > 0 && kVocabShard > 0 && kNormEps > 0.0f);
}  // namespace naivert::geometry
