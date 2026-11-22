#pragma once

// Stable per-layer record read by the ExpertDown all-reduce kernels at replay time, and the
// mailbox geometry those kernels and their host writers share.
//
// One record per (layer, replay slot, query row) lives in device memory; its address is baked
// into the layer's constant-memory descriptor, so graphs replay with fixed parameters while the
// generation tag and (optionally) the replay epoch are read from this record.

#include "naivert/body/expert_down_allreduce/expert_down_allreduce_args.hpp"
#include "naivert/body/expert_down_allreduce/expert_down_allreduce_traits.cuh"
#include "naivert/common/tp8_mailbox.cuh"

#include <cstddef>
#include <cstdint>

namespace naivert_replica_expert_down_allreduce {

constexpr std::uint32_t kExpertDownDomainMagic = 0x4E414D44u;  // "NAMD"
constexpr std::uint32_t kExpertDownDomainVersion = 4u;

constexpr int kExpertDownDomainRanks = naivert::geometry::kTpRanks;

// Mailbox geometry lives in naivert/common/tp8_mailbox.cuh; these aliases keep the names the
// entry and host writer use. BF16 expert-down places four 64-output warp tiles
// per CTA record: 16 records x 4 warps x 32 qwords x 2 BF16 = 4096 outputs.
constexpr std::size_t kExpertDownMailboxSlabStride = naivert::kMailboxSlabStride;
// W8A8: one slab for FP32 expert reduce-scatter, one for BF16 mixture all-gather.
constexpr std::size_t kExpertDownMailboxRowBytes =
    kExpertDownMailboxSlabStride * (kW8A8 ? 2 : 1);
constexpr std::size_t kExpertDownMailboxPeerStride = naivert::kMailboxPeerStride;
constexpr std::size_t kExpertDownMailboxPhaseStride = naivert::kMailboxPhaseStride;
constexpr std::size_t kExpertDownMailboxCtaStride = naivert::kMailboxCtaStride;
constexpr std::size_t kExpertDownMailboxWarpStride = naivert::kMailboxWarpStride;
constexpr int kExpertDownMailboxWarpsPerCta =
    int(naivert::kMailboxCtaStride / naivert::kMailboxWarpStride);
constexpr int kExpertDownMailboxQwordsPerWarp = kOutputsPerTile / 2;
constexpr std::uint32_t kExpertDownMailboxSlab = 0u;

constexpr int kExpertDownOutputsPerTile = kOutputsPerTile;
constexpr int kExpertDownLaunchCtas = naivert::geometry::kSmCount;
// Reserve 48 slots for decode and 48 for each of two verify units. Keep
// headroom within the 64-KiB constant-memory budget.
constexpr int kExpertDownDescriptorCapacity = 192;

// Per-layer kernel arguments, published once into constant memory. Keeping the record in
// memory lets the kernel pass it to the body by reference instead of rebuilding it per thread.
struct alignas(16) ExpertDownDescriptor {
  ExpertDownAllreduceArgs args;
};

// Written by the profiled kernel when wait_metrics_enabled: one entry per (replay, CTA).
struct alignas(8) ExpertDownWaitMetric {
  std::uint32_t peer_wait_ns;
  std::uint32_t poll_load_count;
};

struct alignas(16) ExpertDownRuntimeDomain {
  std::uint32_t magic;
  std::uint32_t version;
  std::uint32_t rank;
  std::uint32_t nranks;
  // Ordinal of the replay whose wait metrics are recorded (0 disables recording).
  std::uint32_t replay_count;
  // Per-layer generation tag; its low bit selects the mailbox phase.
  std::uint32_t native_mailbox_tag;
  std::uint32_t native_mailbox_slab;
  // Optional replay-epoch word (uint64) read at replay time; 0 = tag-only generation.
  std::uint64_t step_block_ptr;
  // Experiment knob: delay each publish by this many ns (0 = off).
  std::uint32_t rank_skew_ns;
  std::uint32_t wait_metrics_enabled;
  std::uint32_t wait_metrics_replay_capacity;
  std::uint32_t wait_metrics_cta_capacity;
  std::uint64_t wait_metrics_base_ptr;
  // Destination mailbox base of every rank, copied here before capture so the kernel never
  // reads peer metadata: the tagged store is its only remote operation.
  std::uint64_t peer_mailbox_base_ptrs[kExpertDownDomainRanks];
};

}  // namespace naivert_replica_expert_down_allreduce
