#pragma once

// Modified for FastLLM fused draft execution. See the parent vendor NOTICE
// and UPSTREAM.json for adaptation details and original source hashes.
#include <cuda/ptx>

#include <cstddef>
#include <cstdint>

#include "naivert/common/tp8_mailbox.cuh"
#include "naivert/body/dspark_common/mailbox_layout.hpp"

namespace fastllm_draft_ops {

// Which slab this is, and every check that the choice does not collide with anyone else's, live
// in dspark_mailbox_slabs.cuh. They are not repeated here: a second attempt at this had the ledger
// in THIS file, derived each index from the one above it, and asserted the total against a
// constant defined from that same total -- `X == X`, which catches nothing. The header explains
// what shape does work and which edit trips which assertion.
constexpr std::size_t kRendezvousSlab =
    std::size_t(kRendezvousSlabIndex) * naivert::kMailboxSlabStride;

// Each rank has seven peer lanes, indexed 0..6; it does not publish to itself.
constexpr std::size_t kRendezvousLanes = naivert::kMailboxRanks - 1;

// The seven lanes fit the slab they claim. Exactly, as it happens (7 * 0x10000 == 0x70000), so
// this has no slack and a wider peer stride would be caught here rather than by corrupting slab 11.
// With the ledger above, this is the only remaining way to leave the slab, so no separate
// "does the last lane lie inside the allocation" bound is needed -- that one was implied by the
// slab-fits-allocation assertion anyway, and two assertions where one bites read as two checks.
static_assert(kRendezvousLanes * naivert::kMailboxPeerStride <= naivert::kMailboxSlabStride,
              "the rendezvous lanes must fit inside one mailbox slab");

// Site id for the watchdog. The payload sites use 2..26 (the salt table); 1 is outside it, so a
// tripped flag distinguishes "the rendezvous stalled" from "site N's all-reduce stalled".
constexpr unsigned kSiteRendezvous = 1u;

// A peer's announcement lane. Same addressing the payload lanes use -- `mailbox_source_slot` into
// `kMailboxPeerStride` -- so the seven lanes fill the slab exactly. No phase: the value is
// monotone, so there is no second buffer to alternate with.
__device__ inline unsigned long long* rendezvous_word(const std::uint8_t* base, int source_slot) {
  return reinterpret_cast<unsigned long long*>(
      const_cast<std::uint8_t*>(base) + kRendezvousSlab +
      std::size_t(source_slot) * naivert::kMailboxPeerStride);
}

// "Has `announced` reached `k`?", written so a 32-bit counter may wrap. `announced - k` is the
// signed distance, correct for any skew below 2^31 blocks; plain `>=` would stall for four
// billion blocks at the wrap. The counter is `replay_generation`, which the harness zeroes in the
// same CALL as the mailbox -- two `torch.zeros`, not one allocation (run_naive_n0p5_dspark.py
// `alloc_kernel_state`), but neither is reachable without the other -- so block 0 compares 0
// against a zeroed slab and the rendezvous is correctly a no-op there. `bump_epoch` can raise
// the counter without the mailbox moving, which only makes this WAIT rather than pass -- the safe
// direction.
__device__ inline bool rendezvous_reached(unsigned announced, unsigned k) {
  return static_cast<int>(announced - k) >= 0;
}

// Step 1: tell every peer which block this rank has entered. ONE thread of ONE CTA calls this.
//
// WHAT THE `__threadfence_system()` HERE DOES AND DOES NOT DO. It orders THIS THREAD's prior
// accesses ahead of the announcement, and nothing else -- it is executed by one thread of CTA 0.
// What the waiting rank actually needs is stronger: that block N's site-12 reads, done by every
// thread of all 128 CTAs, are complete before N + 1 is announced. That ordering comes ENTIRELY
// from the D4 -> D1 kernel boundary, which retires every CTA of D4 before any CTA of D1 starts.
// This fence does not supply it and cannot.
//
// So the fence is a local correctness requirement (keep the store from being hoisted above this
// thread's own earlier work), not the cross-CTA guarantee. IF D4 AND D1 ARE FUSED INTO ONE
// LAUNCH, the kernel boundary disappears and this becomes unsound: the announce would then have
// to sit after a GRID-WIDE barrier (`grid_barrier(args.barrier)`, as the epoch bump at the bottom
// of D1 already does), so that all 128 CTAs' site-12 reads are known finished. Fusing those two
// launches without moving the announce would remove the required ordering guarantee.
__device__ inline void rendezvous_announce(const std::uint8_t* const* peers, int rank, unsigned k) {
  __threadfence_system();
#pragma unroll
  for (int peer_slot = 0; peer_slot < naivert::kMailboxRanks - 1; ++peer_slot) {
    const int peer = naivert::mailbox_peer_rank(peer_slot, rank);
    naivert::store_tagged(rendezvous_word(peers[peer], naivert::mailbox_source_slot(rank, peer)),
                          static_cast<unsigned long long>(k));
  }
}

// Step 2: wait until all seven peers have announced block `k` or later. EVERY CTA calls this (one
// thread each, then `__syncthreads()` at the call site) -- no grid barrier is needed because the
// condition is monotone: once true it stays true, so CTAs may observe it independently.
//
// The caller selects a watchdog timeout; limit_ns == 0 leaves the wait unbounded.
// Poll once before reading %globaltimer. A peer that has already announced its
// generation needs no timer read or retry loop. Only the slow path checks elapsed time.
__device__ inline void rendezvous_wait(const std::uint8_t* local_base, int rank, unsigned k,
                                       const naivert::MailboxWatchdog& watchdog) {
  bool all_reached = true;
#pragma unroll
  for (int peer_slot = 0; peer_slot < naivert::kMailboxRanks - 1; ++peer_slot) {
    const int peer = naivert::mailbox_peer_rank(peer_slot, rank);
    const unsigned long long announced = naivert::load_tagged<true>(
        rendezvous_word(local_base, naivert::mailbox_source_slot(peer, rank)));
    all_reached &= rendezvous_reached(static_cast<unsigned>(announced), k);
  }
  if (all_reached) {
    cuda::ptx::fence(cuda::ptx::sem_acquire, cuda::ptx::scope_sys);        // fast path: not one clock read
    return;
  }

  const bool bounded = watchdog.limit_ns != 0ull;
  const unsigned long long start = bounded ? naivert::mailbox_globaltimer_ns() : 0ull;
  unsigned spins = 0u;
#pragma unroll 1
  for (int peer_slot = 0; peer_slot < naivert::kMailboxRanks - 1; ++peer_slot) {
    const int peer = naivert::mailbox_peer_rank(peer_slot, rank);
    const unsigned long long* w =
        rendezvous_word(local_base, naivert::mailbox_source_slot(peer, rank));
    for (;;) {
      const unsigned long long announced = naivert::load_tagged<true>(w);
      if (rendezvous_reached(static_cast<unsigned>(announced), k)) {
        break;
      }
      if (bounded && ((++spins & naivert::kMailboxWatchdogSpinMask) == 0u) &&
          naivert::mailbox_globaltimer_ns() - start > watchdog.limit_ns) {
        naivert::mailbox_watchdog_trip(watchdog, kSiteRendezvous);
        // Fence even on the give-up path. The flag says this rank's boundary is unguarded and the
        // launch's numbers are void, but the rank still runs to its grid barrier and still
        // publishes -- so the acquire side has to hold on the way out too, or the failure mode
        // stops being "a reported wrong answer" and becomes "an unordered one".
        cuda::ptx::fence(cuda::ptx::sem_acquire, cuda::ptx::scope_sys);
        return;
      }
    }
  }
  // Acquire side: nothing this rank publishes afterwards may be hoisted above the wait, or the
  // rendezvous would order nothing.
  cuda::ptx::fence(cuda::ptx::sem_acquire, cuda::ptx::scope_sys);
}

}  // namespace fastllm_draft_ops
