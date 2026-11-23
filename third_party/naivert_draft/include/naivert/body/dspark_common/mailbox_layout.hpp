#pragma once

// Modified for FastLLM fused draft execution. See the parent vendor NOTICE
// and UPSTREAM.json for adaptation details and original source hashes.
#include "naivert/common/tp8_mailbox.cuh"

namespace fastllm_draft_ops {
// D1/D3/D4 reuse row slabs with different generation salts. D6 and the
// block-boundary rendezvous own disjoint slabs. New claimants belong here.
inline constexpr int kRowSlabs = 8;
inline constexpr int kMarkovLatentSlab = 8;
inline constexpr int kMarkovCandSlab = 9;
inline constexpr int kRendezvousSlabIndex = 10;
inline constexpr int kMailboxSlabsPerRank = 11;
inline constexpr int kNonRowClaims[] = {
    kMarkovLatentSlab, kMarkovCandSlab, kRendezvousSlabIndex};
constexpr bool mailbox_claims_valid() {
  for (int claim : kNonRowClaims)
    if (claim < kRowSlabs || claim >= kMailboxSlabsPerRank) return false;
  for (int slab = 0; slab < kMailboxSlabsPerRank; ++slab) {
    int owners = slab < kRowSlabs ? 1 : 0;
    for (int claim : kNonRowClaims) owners += claim == slab;
    if (owners != 1) return false;
  }
  return true;
}
static_assert(mailbox_claims_valid(), "draft mailbox claims overlap or leave a hole");
}  // namespace fastllm_draft_ops
