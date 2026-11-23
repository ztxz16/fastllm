#pragma once

// The TP8 mailbox: tagged 64-bit words on peer-mapped global memory.
//
// Each word carries {tag:32 | payload:32}.  A writer stores its payload with
// the tag of the current step into every peer's mailbox; a reader polls its
// own mailbox until the tag matches, so the data is its own ready flag and
// one system-scope store publishes both (NCCL's LL protocol, ThunderKittens'
// barrier_t signal/wait).  Loads and stores are relaxed at system scope: the
// tag check is the synchronization, and the mailbox slot is never reused
// within a step.  ld.acquire at system scope serves the plain flag words the
// sampler entries poll.

#include "naivert/common/model_geometry.cuh"

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

namespace naivert {

struct TaggedPair {
  unsigned long long lo;
  unsigned long long hi;
};

__device__ inline unsigned long long tagged_word(unsigned tag,
                                                 std::uint32_t payload) {
  return (static_cast<unsigned long long>(tag) << 32) |
         static_cast<unsigned long long>(payload);
}

__device__ inline unsigned tagged_tag(unsigned long long word) {
  return static_cast<unsigned>(word >> 32);
}

__device__ inline std::uint32_t tagged_payload(unsigned long long word) {
  return static_cast<std::uint32_t>(word & 0xffffffffu);
}

__device__ inline void store_tagged(unsigned long long* ptr,
                                    unsigned long long word) {
  asm volatile("st.relaxed.sys.global.b64 [%0], %1;"
               :
               : "l"(ptr), "l"(word)
               : "memory");
}

// Same 16-byte alignment as load_tagged_pair. The two elements are not
// assumed to be one atomic 128-bit transaction: receivers check both tags.
__device__ inline void store_tagged_pair(unsigned long long* ptr,
                                         unsigned long long lo, unsigned long long hi) {
  asm volatile("st.relaxed.sys.global.v2.b64 [%0], {%1, %2};"
               : : "l"(ptr), "l"(lo), "l"(hi) : "memory");
}

// kNoAllocate: poll without allocating the line in L1 (the word is written
// by another GPU; a cached copy would only be re-validated anyway).
template <bool kNoAllocate>
__device__ inline unsigned long long load_tagged(
    const unsigned long long* ptr) {
  unsigned long long word;
  if constexpr (kNoAllocate) {
    asm volatile("ld.relaxed.sys.global.L1::no_allocate.u64 %0, [%1];"
                 : "=l"(word)
                 : "l"(ptr)
                 : "memory");
  } else {
    asm volatile("ld.relaxed.sys.global.b64 %0, [%1];"
                 : "=l"(word)
                 : "l"(ptr)
                 : "memory");
  }
  return word;
}

template <bool kNoAllocate>
__device__ inline TaggedPair load_tagged_pair(
    const unsigned long long* ptr) {
  TaggedPair pair;
  if constexpr (kNoAllocate) {
    asm volatile(
        "ld.relaxed.sys.global.L1::no_allocate.v2.u64 "
        "{%0, %1}, [%2];"
        : "=l"(pair.lo), "=l"(pair.hi)
        : "l"(ptr)
        : "memory");
  } else {
    asm volatile("ld.relaxed.sys.global.v2.u64 {%0, %1}, [%2];"
                 : "=l"(pair.lo), "=l"(pair.hi)
                 : "l"(ptr)
                 : "memory");
  }
  return pair;
}

// Spin until the word carries `tag`; returns it.  The retry is tight on
// purpose: the peers publish within a few hundred nanoseconds of each other
// and a sleep magnifies the skew.
template <bool kNoAllocate>
__device__ inline unsigned long long wait_tagged(
    const unsigned long long* ptr, unsigned tag) {
  unsigned long long word;
  do {
    word = load_tagged<kNoAllocate>(ptr);
  } while (tagged_tag(word) != tag);
  return word;
}

template <bool kNoAllocate>
__device__ inline TaggedPair wait_tagged_pair(const unsigned long long* ptr,
                                              unsigned tag) {
  TaggedPair pair;
  do {
    pair = load_tagged_pair<kNoAllocate>(ptr);
  } while (tagged_tag(pair.lo) != tag || tagged_tag(pair.hi) != tag);
  return pair;
}

// ---------------------------------------------------------------------------------------------
// Bounded variants: the same spin, with a deadline and a device-side error flag.
//
// Why they exist.  The tag protocol's failure mode is not a stale read -- `wait_tagged*` match the
// FULL generation, so a word that lands early does not get mistaken for the one a peer is waiting
// on, it DESTROYS it and that peer spins forever.  Inside a captured graph that is an unkillable
// hang rather than an error (dspark_fc_ctx.cu, the "THE HOST-SIDE STEP BARRIER'S THREE JOBS"
// block -- cited by name because the line numbers this used to give went stale within one stage).  A bound turns any such hang, whatever
// caused it, into one reported error.
//
// Why they are SEPARATE functions rather than a parameter on the ones above.  Those are shared
// with the shipped bodies (unproj_o, down, expert_down); adding to their signature would perturb
// kernels that are already gated.  Nothing above this comment changes, so their codegen does not.
//
// The clock is `%globaltimer` and NOT `clock64()`: clock64 is a per-SM CYCLE counter, unsynchronised
// across SMs and re-scaled by any clock change, so a cycle budget is not a wall-clock bound.
// `%globaltimer` is GPU-uniform nanoseconds (libcu++'s chrono uses it for the same reason).
// Its resolution is ~1 us while the comment above notes peers publish within a few HUNDRED ns of
// each other, so the deadline is sampled once every `kMailboxWatchdogSpinMask + 1` spins and never
// on the fast path -- the tight retry the protocol depends on is preserved literally.

__device__ inline unsigned long long mailbox_globaltimer_ns() {
  unsigned long long value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

// `limit_ns == 0` disables the bound, which is what every caller that has not opted in gets.
struct MailboxWatchdog {
  unsigned long long limit_ns;
  unsigned* flag;
};

// A tripped flag is always non-zero even when the site id is 0.
constexpr unsigned kMailboxWatchdogTripped = 0x80000000u;
constexpr unsigned kMailboxWatchdogSpinMask = 0x3ffu;

// First writer wins.  The FIRST site to time out is the diagnostic one; every later trip is
// usually its downstream victim, and letting those overwrite it would report the symptom.
__device__ inline void mailbox_watchdog_trip(const MailboxWatchdog& watchdog,
                                             unsigned site) {
  if (watchdog.flag == nullptr) {
    return;
  }
  atomicCAS_system(watchdog.flag, 0u, site | kMailboxWatchdogTripped);
}

template <bool kNoAllocate>
__device__ inline TaggedPair wait_tagged_pair_bounded(
    const unsigned long long* ptr, unsigned tag,
    const MailboxWatchdog& watchdog, unsigned site) {
  TaggedPair pair = load_tagged_pair<kNoAllocate>(ptr);
  if (tagged_tag(pair.lo) == tag && tagged_tag(pair.hi) == tag) {
    return pair;                       // fast path: not one clock read
  }
  const bool bounded = watchdog.limit_ns != 0ull;
  const unsigned long long start = bounded ? mailbox_globaltimer_ns() : 0ull;
  unsigned spins = 0u;
  for (;;) {
    pair = load_tagged_pair<kNoAllocate>(ptr);
    if (tagged_tag(pair.lo) == tag && tagged_tag(pair.hi) == tag) {
      return pair;
    }
    if (bounded && ((++spins & kMailboxWatchdogSpinMask) == 0u) &&
        mailbox_globaltimer_ns() - start > watchdog.limit_ns) {
      mailbox_watchdog_trip(watchdog, site);
      // The returned payload is garbage by construction; the flag is what says so, and the
      // caller's contract is that a tripped flag invalidates the whole launch.  Returning is
      // what lets every rank reach its grid barrier and the graph complete, which is the entire
      // point: a completed graph with a flag set can be reported, a hung one cannot.
      return pair;
    }
  }
}

__device__ inline unsigned load_acquire_sys(const unsigned* ptr) {
  unsigned value;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(value)
               : "l"(ptr)
               : "memory");
  return value;
}

// ---------------------------------------------------------------------------------------------
// Tile mailbox: the TP8 all-reduce transport shared by unproj_o, down and expert_down.
//
// Every rank owns one allocation that all peers can write over NVLink. Inside a destination's
// allocation a word is addressed by
//     base_offset (per query row or slab) -> source lane (7, one per other rank) -> phase (2)
//     -> Layout::tile_offset(task, qword)
// The generation tag is (replay_epoch << 8) | layer_tag, its parity selects the phase, so
// consecutive steps of one layer alternate buffers and a stale word is rejected by its tag.
// Traffic is push-only: a producer stores into the destination's memory; the consumer polls its
// own memory. Every rank reduces the whole tile itself (one-shot exchange).
//
// The accumulation order is part of each body's contract (its goldens pin the rounding), so it
// stays with the body: two recipes below reproduce the orders in use.
// ---------------------------------------------------------------------------------------------

constexpr int kMailboxRanks = geometry::kTpRanks;
constexpr std::size_t kMailboxSlabStride = 0x70000;   // one slab per query row / CTA group
constexpr std::size_t kMailboxPeerStride = 0x10000;   // one lane per source rank (7 in use)
constexpr std::size_t kMailboxPhaseStride = 0x8000;   // two phases, selected by tag parity
constexpr std::size_t kMailboxCtaStride = 0x400;      // four warp records per CTA record
constexpr std::size_t kMailboxWarpStride = 0x100;     // 32 tagged qwords

__device__ inline unsigned mailbox_generation(unsigned replay_epoch,
                                              unsigned layer_tag) {
  // The per-layer tag stays in the low byte (it runs 2..125), the epoch above it: a smaller
  // shift once aliased layer L of epoch e + 1 with layer L + 4 of epoch e.
  return (replay_epoch << 8) | (layer_tag & 0xffu);
}

__device__ inline unsigned mailbox_phase(unsigned generation) {
  return generation & 1u;
}

// Lane of `src_rank`'s payload inside `dst_rank`'s allocation (the destination's own rank has
// no lane).
__device__ inline int mailbox_source_slot(int src_rank, int dst_rank) {
  return src_rank - (src_rank > dst_rank ? 1 : 0);
}

// Rank behind peer lane `peer_slot` as seen from `rank`.
__device__ inline int mailbox_peer_rank(int peer_slot, int rank) {
  return peer_slot + (peer_slot >= rank ? 1 : 0);
}

struct PayloadPair {
  std::uint32_t lo;
  std::uint32_t hi;
};

template <class Layout>
struct TileMailbox {
  const std::uint8_t* const* peers;  // every rank's mailbox base, this rank's included
  const std::uint8_t* local_base;    // peers[rank], read once
  int rank;
  unsigned generation;
  unsigned phase;
  std::size_t base_offset;

  __device__ static TileMailbox make(const std::uint8_t* const* peers, int rank,
                                     unsigned generation,
                                     std::size_t base_offset) {
    return {peers, peers[rank], rank, generation, mailbox_phase(generation),
            base_offset};
  }
  __device__ unsigned long long* word(const std::uint8_t* base, int source_slot,
                                      int task, int qword) const {
    return reinterpret_cast<unsigned long long*>(
        const_cast<std::uint8_t*>(base) + base_offset +
        std::size_t(source_slot) * kMailboxPeerStride +
        std::size_t(phase) * kMailboxPhaseStride +
        Layout::tile_offset(task, qword));
  }
  // Where this rank publishes one qword of `task` for `dst_rank`.
  __device__ unsigned long long* remote_word(int dst_rank, int task,
                                             int qword) const {
    return word(peers[dst_rank], mailbox_source_slot(rank, dst_rank), task,
                qword);
  }
  // Where this rank polls the payload lane `source_slot` published for it.
  __device__ unsigned long long* local_word(int source_slot, int task,
                                            int qword) const {
    return word(local_base, source_slot, task, qword);
  }
  // Publish one qword to all seven peers.
  __device__ void publish(int task, int qword, std::uint32_t payload) const {
    const unsigned long long tagged = tagged_word(generation, payload);
#pragma unroll
    for (int peer_slot = 0; peer_slot < kMailboxRanks - 1; ++peer_slot) {
      store_tagged(remote_word(mailbox_peer_rank(peer_slot, rank), task, qword),
                   tagged);
    }
  }
  __device__ void publish_pair(int task, int qword, std::uint32_t lo,
                               std::uint32_t hi) const {
    const auto a = tagged_word(generation, lo);
    const auto b = tagged_word(generation, hi);
#pragma unroll
    for (int peer_slot = 0; peer_slot < kMailboxRanks - 1; ++peer_slot) {
      store_tagged_pair(remote_word(mailbox_peer_rank(peer_slot, rank), task, qword), a, b);
    }
  }
  template <bool kNoAllocate>
  __device__ PayloadPair wait_pair(int source_slot, int task,
                                   int qword) const {
    const TaggedPair pair = wait_tagged_pair<kNoAllocate>(
        local_word(source_slot, task, qword), generation);
    return {tagged_payload(pair.lo), tagged_payload(pair.hi)};
  }
  // Same read, with a deadline.  `site` is the all-reduce site's salt, so a tripped flag names
  // WHICH rendezvous stalled rather than only that one did -- with twenty-five live sites per
  // draft block that difference is the whole diagnostic value.
  template <bool kNoAllocate>
  __device__ PayloadPair wait_pair_bounded(int source_slot, int task, int qword,
                                           const MailboxWatchdog& watchdog,
                                           unsigned site) const {
    const TaggedPair pair = wait_tagged_pair_bounded<kNoAllocate>(
        local_word(source_slot, task, qword), generation, watchdog, site);
    return {tagged_payload(pair.lo), tagged_payload(pair.hi)};
  }
};

__device__ inline std::uint32_t pack_bf16x2(float lo, float hi) {
  const __nv_bfloat162 packed = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const std::uint32_t*>(&packed);
}

__device__ inline float payload_lane(std::uint32_t payload, int lane) {
  const __nv_bfloat162 packed =
      *reinterpret_cast<const __nv_bfloat162*>(&payload);
  return lane == 0 ? __low2float(packed) : __high2float(packed);
}

// Four fp32 accumulators over two bf16x2 payload words.
struct Bf16x4Acc {
  float v0;
  float v1;
  float v2;
  float v3;
  __device__ static Bf16x4Acc zero() { return {0.0f, 0.0f, 0.0f, 0.0f}; }
  __device__ static Bf16x4Acc of(std::uint32_t p01, std::uint32_t p23) {
    return {payload_lane(p01, 0), payload_lane(p01, 1), payload_lane(p23, 0),
            payload_lane(p23, 1)};
  }
  __device__ void add(std::uint32_t p01, std::uint32_t p23) {
    v0 += payload_lane(p01, 0);
    v1 += payload_lane(p01, 1);
    v2 += payload_lane(p23, 0);
    v3 += payload_lane(p23, 1);
  }
  __device__ unsigned long long packed() const {
    return static_cast<unsigned long long>(pack_bf16x2(v0, v1)) |
           (static_cast<unsigned long long>(pack_bf16x2(v2, v3)) << 32);
  }
};

// Recipe A: all eight partials in rank order, the local pair standing in for this rank's slot.
template <class Fetch>
__device__ inline void accumulate_rank_order(Bf16x4Acc& acc, int rank,
                                             std::uint32_t local01,
                                             std::uint32_t local23,
                                             Fetch fetch) {
#pragma unroll
  for (int src = 0; src < kMailboxRanks; ++src) {
    std::uint32_t p01 = local01;
    std::uint32_t p23 = local23;
    if (src != rank) {
      const PayloadPair pair = fetch(mailbox_source_slot(src, rank));
      p01 = pair.lo;
      p23 = pair.hi;
    }
    acc.add(p01, p23);
  }
}

// Recipe B: the seven peer lanes in lane order (the caller has already folded in the local
// pair, and whatever else its order puts first).
template <class Fetch>
__device__ inline void accumulate_peers_in_slot_order(Bf16x4Acc& acc,
                                                      Fetch fetch) {
#pragma unroll
  for (int peer_slot = 0; peer_slot < kMailboxRanks - 1; ++peer_slot) {
    const PayloadPair pair = fetch(peer_slot);
    acc.add(pair.lo, pair.hi);
  }
}

// Poll variant that counts loads, for the wait-metric kernel.
template <bool kNoAllocate>
__device__ inline TaggedPair wait_tagged_pair_counted(
    const unsigned long long* ptr, unsigned tag, unsigned& polls) {
  TaggedPair pair;
  do {
    polls += 1u;
    pair = load_tagged_pair<kNoAllocate>(ptr);
  } while (tagged_tag(pair.lo) != tag || tagged_tag(pair.hi) != tag);
  return pair;
}

}  // namespace naivert
