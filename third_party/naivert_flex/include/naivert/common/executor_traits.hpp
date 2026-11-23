#pragma once

// What a body's control flow reads from its traits: the depth of its
// transaction-slot ring.
//
// The native template symbol carried more -- its shape was
// <Schedule, SlotDepth, LogicalSmemBytes, ClusterCount, Variant, KeComputeType>
// and the replica mirrored it -- but nothing in this tree ever read the other
// five.  Shared-memory size reaches the launch through the op descriptor, and
// the compute-type enum had no consumer at all.  The native shape is recorded
// in the per-op args headers next to the rest of the symbol evidence.

namespace naivert {

template <int SlotDepth_>
struct ExecutorTraits {
  static constexpr int kSlotDepth = SlotDepth_;
};

}  // namespace naivert
