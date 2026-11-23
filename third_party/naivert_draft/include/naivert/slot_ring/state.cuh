#pragma once

// The runtime view one thread has of the slot ring: its role, the ring
// position (cutlass::PipelineState's index and phase), and the shared-memory
// regions the body bound in bind_shared_layout.

#include <cstdint>

#include "../common/cuda_compat.hpp"

namespace naivert {

enum class ThreadRole {
  kConsumerA,
  kConsumerB,
  kHelper,
};

// Ring position: slot is the pane index, phase the parity of the current
// pass over the ring.
struct TransactionState {
  int slot;
  int iteration;
  int phase;
};

// What a body says about the launch: how many CTAs take part and
// how many tiles exist.  A CTA whose rank is beyond either is inactive.
struct SlotRingWorkArgs {
  int total_tiles;
  int active_ctas;
};

struct SlotRingState {
  int tid;
  int lane_id;
  int warp_id;
  int cta_rank;
  ThreadRole role;
  bool active;
  bool elected;

  TransactionState tx;

  unsigned char* smem_base;
  unsigned char* control_base;
  std::uint64_t* full_barriers;   // helper -> consumers, one per slot
  std::uint64_t* empty_barriers;  // consumers -> helper, one per slot
  unsigned char* early_base;
  unsigned char* deep_base;
  unsigned char* scratch_base;

  int early_stride;
  int deep_stride;

  // Body-owned values that must remain live across transport-loop iterations.
  // Native head-packed MLA keeps one 32-float distributed O fragment per
  // consumer lane; bodies that use only the first pair compile the tail away.
  float body_acc[32];

  __device__ unsigned char* early_slot_ptr() const {
    return early_base + tx.slot * early_stride;
  }

  __device__ unsigned char* deep_slot_ptr() const {
    return deep_base + tx.slot * deep_stride;
  }
};

}  // namespace naivert
