#pragma once

#include "naivert/common/tp8_mailbox.cuh"

namespace naivert {

// Publish the local pair before calling. Independent inbox loads can overlap;
// both tags must match before a prefetched payload is used. The caller retains
// its existing wait/watchdog policy for a pair that was not ready.
template <class Layout, class Fetch>
__device__ __forceinline__ void accumulate_prefetched_rank_order(
    Bf16x4Acc& acc, const TileMailbox<Layout>& mailbox,
    std::uint32_t local01, std::uint32_t local23, int task, int qword, Fetch fallback) {
  TaggedPair peek[kMailboxRanks - 1];
#pragma unroll
  for (int slot = 0; slot < kMailboxRanks - 1; ++slot) {
    peek[slot] = load_tagged_pair<true>(mailbox.local_word(slot, task, qword));
  }
  accumulate_rank_order(acc, mailbox.rank, local01, local23, [&](int slot) {
    // Constant indices keep these values in registers.
    TaggedPair pair = peek[0];
#pragma unroll
    for (int i = 1; i < kMailboxRanks - 1; ++i) {
      if (slot == i) pair = peek[i];
    }
    if (tagged_tag(pair.lo) == mailbox.generation &&
        tagged_tag(pair.hi) == mailbox.generation) {
      return PayloadPair{tagged_payload(pair.lo), tagged_payload(pair.hi)};
    }
    return fallback(slot);
  });
}

}  // namespace naivert
