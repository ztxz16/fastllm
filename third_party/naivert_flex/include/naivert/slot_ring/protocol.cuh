#pragma once

// Barriers of the slot ring on CUTLASS: the transaction-slot pipeline between one CTA's
// helper (producer) warps and its consumer warps, and the named barriers.
//
// One ring of SlotDepth panes.  Each pane has two mbarriers:
//   full  (helper -> consumers): kProducerArrivals arrivals (one elected lane
//         per helper warp) plus the transaction bytes the helper declares with
//         producer_expect_tx before it issues its async copies.
//   empty (consumers -> helper): kConsumerArrivals arrivals (every consumer
//         thread).  Empties start complete at phase 0 so the helper's first
//         pass over the ring does not wait (CUTLASS instead starts its producer
//         at phase 1: cutlass::make_producer_start_state).
//
// The vocabulary is cutlass::PipelineTransactionAsync's: producer_acquire /
// producer_expect_tx / producer_commit and consumer_wait / consumer_release,
// with TransactionState playing cutlass::PipelineState (slot = index, phase).
// The design notes map every primitive to CUTLASS,
// ThunderKittens and PTX.

#include <cstdint>

#include <cutlass/arch/barrier.h>

#include "../common/role_layout.hpp"
#include "state.cuh"

#if defined(__CUDACC__)
#include <cuda/ptx>
#endif

namespace naivert {

using Barrier = cutlass::arch::ClusterTransactionBarrier;

// elect.sync with a caller-supplied mask (cute::elect_one_sync is the
// full-mask form; bodies call this one under __activemask()).
__device__ inline bool elect_sync_compat(unsigned mask) {
#if defined(__CUDA_ARCH__)
  // CUDA 12.9 lacks the C++ wrapper; emit the same SM90+ PTX instruction.
  unsigned elected;
  asm volatile("{ .reg .pred p; elect.sync _|p, %1; selp.b32 %0, 1, 0, p; }"
               : "=r"(elected) : "r"(mask));
  return elected != 0;
#else
  (void)mask;
  return true;
#endif
}

// mbarrier.arrive with a count: pre-completes one phase of a barrier.
__device__ inline void barrier_arrive_count(std::uint64_t* barrier,
                                            std::uint32_t count) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  (void)cuda::ptx::mbarrier_arrive(barrier, count);
#else
  (void)barrier;
  (void)count;
#endif
}

// Hardware named-barrier ids the protocol reserves (BAR0 is __syncthreads).
// Bodies own the other ids (unproj ping-pong on 1/2, expert-select tokens
// and batches on 5..12, mla on 7) and go through named_barrier_sync.
struct SlotRingNamedBarriers {
  static constexpr unsigned kConsumerGroup = 4;
  static constexpr unsigned kConsumerAUtility = 5;
  static constexpr unsigned kHelperUtility = 6;
};

// bar.sync id, count -- cutlass::arch::NamedBarrier with the hardware id
// (its uint32_t overload would offset user ids past CUTLASS's reserved ones).
__device__ inline void named_barrier_sync(unsigned count, unsigned id) {
  cutlass::arch::NamedBarrier::sync(
      count, static_cast<cutlass::arch::ReservedNamedBarriers>(id));
}

// The consumer-group barrier (BAR4).  Precondition: called from a
// consumer-gated path, so exactly kConsumerActiveThreads arrive.
__device__ inline void consumer_group_named_sync() {
  named_barrier_sync(StandardRoleLayout::kConsumerActiveThreads,
                     SlotRingNamedBarriers::kConsumerGroup);
}

template <int SlotDepth>
struct SlotProtocol {
  static constexpr std::uint32_t kProducerArrivals = 4u;    // one per helper warp
  static constexpr std::uint32_t kConsumerArrivals = 256u;  // every consumer thread

  __device__ static std::uint64_t* slot_barrier(std::uint64_t* group,
                                                int slot) {
    return group + slot;
  }

  __device__ static void init_transaction_slot(SlotRingState& st, int slot) {
    Barrier::init(st.full_barriers + slot, kProducerArrivals);
    Barrier::init(st.empty_barriers + slot, kConsumerArrivals);
    barrier_arrive_count(st.empty_barriers + slot, kConsumerArrivals);
  }

  __device__ static void init_transaction_groups(SlotRingState& st) {
#pragma unroll
    for (int slot = 0; slot < SlotDepth; ++slot) {
      init_transaction_slot(st, slot);
    }
  }

  // ── consumer side ──────────────────────────────────────────────────────

  __device__ static bool consumer_try_wait(const SlotRingState& st) {
    return Barrier::try_wait(st.full_barriers + st.tx.slot,
                             std::uint32_t(st.tx.phase & 1));
  }

  __device__ static void consumer_wait(const SlotRingState& st) {
    while (!consumer_try_wait(st)) {
    }
  }

  __device__ static void consumer_release(const SlotRingState& st) {
    if (st.role == ThreadRole::kConsumerA ||
        st.role == ThreadRole::kConsumerB) {
      Barrier::arrive(st.empty_barriers + st.tx.slot);
    }
  }

  // ── producer (helper) side ─────────────────────────────────────────────

  __device__ static bool producer_try_acquire(const SlotRingState& st) {
    return Barrier::try_wait(st.empty_barriers + st.tx.slot,
                             std::uint32_t(st.tx.phase & 1));
  }

  __device__ static void producer_acquire(const SlotRingState& st) {
    while (!producer_try_acquire(st)) {
    }
  }

  // Declare the bytes the slot's async copies will land (no arrival).
  __device__ static void producer_expect_tx(const SlotRingState& st,
                                            std::uint32_t tx_bytes) {
    Barrier::expect_transaction(st.full_barriers + st.tx.slot, tx_bytes);
  }

  // Count this thread's outstanding cp.async copies on the full barrier.
  __device__ static void producer_cp_async_commit(const SlotRingState& st) {
    cutlass::arch::cpasync_barrier_arrive(st.full_barriers + st.tx.slot);
  }

  // One arrival per helper warp.
  __device__ static void producer_commit(const SlotRingState& st) {
    if (st.role == ThreadRole::kHelper && elect_sync_compat(__activemask())) {
      Barrier::arrive(st.full_barriers + st.tx.slot);
    }
  }

  // ── ring position ──────────────────────────────────────────────────────

  __device__ static void advance(SlotRingState& st) {
    st.tx.iteration += 1;
    st.tx.slot += 1;
    if (st.tx.slot == SlotDepth) {
      st.tx.slot = 0;
      st.tx.phase ^= 1;
    }
  }

  __device__ static void prepare_same_slot_reuse(SlotRingState& st) {
    prepare_same_slot_reuse(st, SlotDepth);
  }

  __device__ static void prepare_same_slot_reuse(
      SlotRingState& st, int steady_helper_slot_depth) {
    st.tx.iteration += steady_helper_slot_depth;
    st.tx.phase ^= 1;
  }
};

}  // namespace naivert
