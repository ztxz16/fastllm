#pragma once
#include "naivert/common/block128_activation.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cutlass/float8.h>
#include <cutlass/arch/barrier.h>

#include <cstdint>

#include "naivert/body/expert_down_allreduce/expert_down_allreduce_args.hpp"
#include "naivert/body/expert_down_allreduce/expert_down_allreduce_traits.cuh"
#include "naivert/common/tp8_mailbox.cuh"
#include "naivert/entry/expert_down_runtime_domain.hpp"
#include "naivert/pdl/grid_dependency.cuh"
#include "naivert/slot_ring/state.cuh"
#include "naivert/slot_ring/protocol.cuh"

#ifndef NAIVERT_EXPERT_DOWN_ALLREDUCE_NATIVE_PARALLEL_PEER_POLL
#define NAIVERT_EXPERT_DOWN_ALLREDUCE_NATIVE_PARALLEL_PEER_POLL 0
#endif

namespace naivert_replica_expert_down_allreduce {

namespace slot_ring = naivert;

template <bool kProfileWaitPoll, bool kWaitForPrimary = false,
          bool kParallelW8A8Epilogue = false, bool kKeepTmem = false, bool kPublishOnly = false>
struct ExpertDownAllreduceBodyT;
// Production instantiations: one-row / multi-row target bodies and the wait-metric variant.
using ExpertDownAllreduceBody = ExpertDownAllreduceBodyT<false, false>;
// Decode benefits from parallel paired TMEM readers; verify retains serial readers.
using ExpertDownAllreduceDecodeBody = ExpertDownAllreduceBodyT<false, false, true>;
using ExpertDownAllreducePdlSecondaryBody = ExpertDownAllreduceBodyT<false, true>;
using ExpertDownAllreduceProfiledBody = ExpertDownAllreduceBodyT<true, false, true>;

template <bool kProfileWaitPoll, bool kWaitForPrimary, bool kParallelW8A8Epilogue, bool kKeepTmem, bool kPublishOnly>
struct ExpertDownAllreduceBodyT {
  // Four 32-output experts fill M128 in decode. Multirow retains two
  // 64-output experts. Both use the existing 32-KiB payload allocation.
  static constexpr bool kNarrowDecode = kW8A8 && kParallelW8A8Epilogue;
  static constexpr int kSlotDepth = kNarrowDecode ? 4 :
      naivert_replica_expert_down_allreduce::kSlotDepth;
  using Slot = slot_ring::SlotProtocol<kSlotDepth>;
  static constexpr int kComputeOutputs = kNarrowDecode ? 32 : kOutputsPerTile;
  static constexpr int kW8A8GroupExperts = kNarrowDecode ? 4 : 2;
  static constexpr int kW8A8PanelBytes = kComputeOutputs * 128;
  static constexpr int kW8A8ExpertBytes = 2 * kW8A8PanelBytes;
  static constexpr int kW8A8GroupPanelBytes = kW8A8GroupExperts * kW8A8PanelBytes;
  static_assert(!kW8A8 || kSlotDepth == kW8A8GroupExperts);
  static_assert(!kW8A8 || kW8A8GroupExperts * kComputeOutputs == 128);
  static_assert(!kW8A8 || kSlotDepth * kW8A8ExpertBytes == 32768);
  static_assert(!kW8A8 || kSlotDepth <= slot_ring::StandardRoleLayout::kControlInitThreads);

  struct NativeComputeMetadata {
    int expert[kOutputSlots];
    float route[kOutputSlots];
    float scale[kOutputSlots][2][2];
  };

  // One output tile per CTA. Helpers stream its expert weights while the
  // consumers run the existing local projection and mailbox epilogue.
  __device__ static void run(const ExpertDownAllreduceArgs& args,
                             unsigned char* shared) {
    slot_ring::SlotRingState st{};
    st.tid = int(threadIdx.x);
    st.lane_id = slot_ring::StandardRoleLayout::lane_id(st.tid);
    st.warp_id = slot_ring::StandardRoleLayout::warp_id(st.tid);
    st.cta_rank = int(blockIdx.x);
    st.active = true;
    st.smem_base = shared;
    st.role = slot_ring::StandardRoleLayout::is_consumer_a(st.tid)
                  ? slot_ring::ThreadRole::kConsumerA
                  : slot_ring::StandardRoleLayout::is_consumer_b(st.tid)
                        ? slot_ring::ThreadRole::kConsumerB
                        : slot_ring::ThreadRole::kHelper;
    bind_shared_layout(st, args, shared);
    int active = capped_active_ctas(args);
    if (active <= 0) active = int(gridDim.x);
    if (active > int(gridDim.x)) active = int(gridDim.x);
    if (effective_total_tiles(args) < active) active = effective_total_tiles(args);
    st.active = st.active && st.cta_rank < active;
    if (st.tid < kSlotDepth) Slot::init_transaction_slot(st, st.tid);
    cutlass::arch::fence_view_async_shared();
    asm volatile("bar.sync 0;" ::: "memory");
    if (!st.active) return;
    st.elected = st.role == slot_ring::ThreadRole::kHelper &&
                 st.warp_id == slot_ring::StandardRoleLayout::kHelperBaseWarp &&
                 slot_ring::elect_sync_compat(0xffffffffu);
    if (st.role == slot_ring::ThreadRole::kHelper) {
      const auto outer_tx = st.tx;
      st.tx.slot = 0;
      st.tx.iteration = 0;
      const int task = st.cta_rank < capped_active_ctas(args) && st.tx.iteration == 0
                           ? st.cta_rank : -1;
      helper_stage_output_tile(st, args, task);
      st.tx.slot = outer_tx.slot;
      st.tx.iteration = outer_tx.iteration;
    }
    while (true) {
      if (st.role == slot_ring::ThreadRole::kConsumerA ||
          st.role == slot_ring::ThreadRole::kConsumerB) {
        const int task = st.cta_rank < capped_active_ctas(args) && st.tx.iteration == 0
                             ? st.cta_rank : -1;
        if (task < 0) break;
        naivert::pdl::wait_for_primary<kWaitForPrimary>();
        consume_output_tile(st, args, task);
      } else {
        break;
      }
      Slot::advance(st);
    }
    // Helpers exit independently; the consumer epilogue completes publication
    // and reduction. Do not add a full-CTA join after that sequence.
  }

  __device__ static void bind_shared_layout(slot_ring::SlotRingState& st,
                                            const ExpertDownAllreduceArgs&,
                                            unsigned char* smem_raw) {
    unsigned char* base = smem_raw;
    st.smem_base = smem_raw;
    st.control_base = base + kBarrierGroup0Offset;
    st.full_barriers =
        reinterpret_cast<std::uint64_t*>(base + kBarrierGroup0Offset);
    st.empty_barriers =
        reinterpret_cast<std::uint64_t*>(base + kBarrierGroup1Offset);
    st.early_base = base + kPayloadBase;
    st.deep_base = base + kPayloadBase;
    st.scratch_base = base;
    st.early_stride = kW8A8 ? kW8A8ExpertBytes : kPayloadStride;
    st.deep_stride = kW8A8 ? kW8A8ExpertBytes : kPayloadStride;
  }

  __host__ __device__ static int pane_storage_offset(int local_col, int k) {
    int row = local_col;
    int k_block = k >> 5;
    int k_in_block = k & 31;
    int phys_row = 0;
    int phys_col = 0;
    if (row < kFullMTiles * 16) {
      int chunk = row >> 4;
      int row_in_chunk = row & 15;
      int a = row_in_chunk >> 3;
      int b = row_in_chunk & 7;
      int d = k_in_block >> 4;
      int e = (k_in_block >> 2) & 3;
      int f = k_in_block & 3;
      int out0 = b * 4 + e;
      int out1 = d * 2 + a;
      phys_row = chunk * 16 + k_block * 2 + (out0 >> 4);
      phys_col = ((out0 & 15) << 4) + (out1 << 2) + f;
    } else {
      int row8 = row - kFullMTiles * 16;
      int d = k_in_block >> 4;
      int e = (k_in_block >> 2) & 3;
      int f = k_in_block & 3;
      phys_row = kFullMTiles * 16 + k_block;
      phys_col = row8 * 32 + e * 8 + d * 4 + f;
    }
    return phys_row * kInter + phys_col;
  }

 public:
  __device__ static float* half1_partial(slot_ring::SlotRingState& st) {
    return reinterpret_cast<float*>(st.scratch_base + kPartialOffset);
  }

  __device__ static void set_inner_ring_slot(slot_ring::SlotRingState& st,
                                             int expert_slot) {
    int phase = expert_slot / kSlotDepth;
    st.tx.slot = expert_slot - phase * kSlotDepth;
    st.tx.iteration = expert_slot;
    st.tx.phase = phase;
    st.tx.phase = phase;
    st.tx.phase = phase;
  }

  __device__ static void helper_stage_output_tile(
      slot_ring::SlotRingState& st, const ExpertDownAllreduceArgs& args,
      int output_tile) {
    if (output_tile < 0) {
      return;
    }
    slot_ring::TransactionState outer_tx = st.tx;
    for (int expert_slot = 0; expert_slot < kOutputSlots; ++expert_slot) {
      if (expert_slot == kSharedExperts) {
        // Shared slots have fixed weight addresses and can be staged while
        // the primary is still running. Routed slots read sel_indices, so
        // acquire the primary before computing any of their weight addresses.
        naivert::pdl::wait_for_primary<kWaitForPrimary>();
      }
      set_inner_ring_slot(st, expert_slot);
      Slot::producer_acquire(st);
      stage_inner_weight_pane(st, args, output_tile, expert_slot,
                              st.deep_slot_ptr());
      if (st.elected) {
        naivert::barrier_arrive_count(
            Slot::slot_barrier(st.full_barriers, st.tx.slot),
            Slot::kProducerArrivals);
      }
      if constexpr (kW8A8 && !kParallelW8A8Epilogue) {
        if (st.elected && expert_slot >= kSharedExperts &&
            expert_slot + kW8A8GroupExperts < kOutputSlots) {
          const int future_slot = expert_slot + kW8A8GroupExperts;
          const int expert = logical_expert(args, future_slot);
          const auto* source = args.moe_w_down +
              (expert * kHidden + output_tile * kComputeOutputs) * kInter;
          asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                       :: "l"(source), "r"(std::uint32_t(kW8A8ExpertBytes)));
        }
      }
    }
    st.tx = outer_tx;
  }

  __device__ static void consume_output_tile(slot_ring::SlotRingState& st,
                                             const ExpertDownAllreduceArgs& args,
                                             int output_tile) {
    if constexpr (kW8A8) {
      consume_output_tile_w8a8(st, args, output_tile);
      return;
    }
    slot_ring::TransactionState outer_tx = st.tx;
    int lane = st.lane_id;
    int warp = st.warp_id;
    int group = lane >> 2;
    int tid4 = lane & 3;
    int row_warp = warp & 3;
    int k_half = warp >> 2;
    int local_base = row_warp * 16;
    int row0_local = local_base + group;
    int row8_local = row0_local + 8;
    int out0 = output_tile * kOutputsPerTile + row0_local;
    int out8 = output_tile * kOutputsPerTile + row8_local;
    bool compute_warp = warp < 8;
    auto* shared_up_gate =
        reinterpret_cast<__nv_bfloat16*>(st.scratch_base);
    constexpr int kUpGateVectorCount =
        kOutputSlots * kInter * int(sizeof(__nv_bfloat16)) /
        int(sizeof(uint4));
    if (compute_warp) {
      for (int vector_index = st.tid; vector_index < kUpGateVectorCount;
           vector_index += kConsumerThreads) {
        const uint4 value =
            reinterpret_cast<const uint4*>(args.up_gate)[vector_index];
        reinterpret_cast<uint4*>(shared_up_gate)[vector_index] = value;
      }
    }
    auto* shared_metadata = reinterpret_cast<NativeComputeMetadata*>(
        st.scratch_base + kOutputSlots * kInter *
                              int(sizeof(__nv_bfloat16)));
    if (compute_warp && st.tid < kOutputSlots) {
      const int expert_slot = st.tid;
      const int expert = logical_expert(args, expert_slot);
      const int output_base = output_tile * kOutputsPerTile;
      const int hidden_block0 = output_base / 128;
      const int hidden_block1 =
          (output_base + kOutputsPerTile - 1) / 128;
      shared_metadata->expert[expert_slot] = expert;
      shared_metadata->route[expert_slot] =
          route_weight(args, expert_slot);
      if constexpr (!(kMx)) {   // MX8 tiles carry their scales in the pane
#pragma unroll
      for (int block_slot = 0; block_slot < 2; ++block_slot) {
        const int hidden_block =
            block_slot == 0 ? hidden_block0 : hidden_block1;
#pragma unroll
        for (int k_group = 0; k_group < 2; ++k_group) {
          shared_metadata->scale[expert_slot][block_slot][k_group] =
              load_scale(args, expert, hidden_block * 128, k_group);
        }
      }
      } else {
        (void)hidden_block1;
      }
    }
    naivert::named_barrier_sync(256u,1u);
    float warp_partial0 = 0.0f;
    float warp_partial1 = 0.0f;
    float warp_partial2 = 0.0f;
    float warp_partial3 = 0.0f;
    float warp_partial4 = 0.0f;
    float warp_partial5 = 0.0f;
    float warp_partial6 = 0.0f;
    float warp_partial7 = 0.0f;

    for (int expert_slot = 0; expert_slot < kOutputSlots; ++expert_slot) {
      set_inner_ring_slot(st, expert_slot);
      Slot::consumer_wait(st);
      if (compute_warp) {
        accumulate_inner_expert_tile_native_k_split(
            args, output_tile, expert_slot, st.deep_slot_ptr(),
            shared_up_gate + expert_slot * kInter,
            warp, group, tid4,
            shared_metadata,
            warp_partial0, warp_partial1, warp_partial2, warp_partial3,
            warp_partial4, warp_partial5, warp_partial6, warp_partial7);
      }
      Slot::consumer_release(st);
    }

    float* partial = half1_partial(st);
    // Native Li1 partitions K=256 across the eight consumer warps.  Each warp
    // owns one K=32 slice and materializes all 56 output-row partials.  The
    // first 56 consumer threads then perform the native-style cross-warp sum.
    if (compute_warp && tid4 == 0) {
      const int warp_partial_base = warp * kPartialRows;
      const int row16 = group;
      const int row24 = 16 + group;
      const int row32 = 32 + group;
      const int row40 = 48 + group;
      partial[warp_partial_base + row16] = warp_partial0;
      partial[warp_partial_base + row16 + 8] = warp_partial1;
      partial[warp_partial_base + row24] = warp_partial2;
      partial[warp_partial_base + row24 + 8] = warp_partial3;
      partial[warp_partial_base + row32] = warp_partial4;
      partial[warp_partial_base + row32 + 8] = warp_partial5;
      if (row40 < kOutputsPerTile) {
        partial[warp_partial_base + row40] = warp_partial6;
      }
      if (row40 + 8 < kOutputsPerTile) {
        partial[warp_partial_base + row40 + 8] = warp_partial7;
      }
    }
    naivert::named_barrier_sync(256u,1u);
    if (compute_warp && st.tid < kOutputsPerTile) {
      float sum = partial[st.tid];
#pragma unroll
      for (int source_warp = 1; source_warp < 8; ++source_warp) {
        sum += partial[source_warp * kPartialRows + st.tid];
      }
      partial[st.tid] = sum;
    }
    naivert::named_barrier_sync(256u,1u);
    float local_delta0 = 0.0f;
    float local_delta8 = 0.0f;
    if (compute_warp && k_half == 0 && tid4 == 0) {
      if (row0_local < kOutputsPerTile) {
        local_delta0 = partial[row0_local];
      }
      if (row8_local < kOutputsPerTile) {
        local_delta8 = partial[row8_local];
      }
    }
          if (args.stable_domain != nullptr) {
            auto* staged_payload =
                reinterpret_cast<std::uint32_t*>(
                    st.scratch_base + kPartialOffset +
                    8 *
                    kPartialRows * int(sizeof(float)));
            // Materialize the 28 qwords before the rank-local residual read and
            // exp_out write. After the consumer sync, 196 threads fan them out
            // one store each, keeping local output traffic off the remote-publish
            // critical path.
            if (compute_warp && k_half == 0) {
              const float adjacent_delta0 =
                  __shfl_down_sync(0xffffffffu, local_delta0, 4);
              const float adjacent_delta8 =
                  __shfl_down_sync(0xffffffffu, local_delta8, 4);
              std::uint32_t owner_payload0 = 0u;
              std::uint32_t owner_payload8 = 0u;
              if (tid4 == 0 && (group & 1) == 0) {
                const bool valid0 =
                    row0_local + 1 < kOutputsPerTile;
                const bool valid8 =
                    row8_local + 1 < kOutputsPerTile;
                if (row0_local + 1 < kOutputsPerTile) {
                  owner_payload0 =
                      naivert::pack_bf16x2(local_delta0, adjacent_delta0);
                  staged_payload[row0_local >> 1] = owner_payload0;
                }
                if (row8_local + 1 < kOutputsPerTile) {
                  owner_payload8 =
                      naivert::pack_bf16x2(local_delta8, adjacent_delta8);
                  staged_payload[row8_local >> 1] = owner_payload8;
                }
              }
            }
            naivert::named_barrier_sync(256u,1u);
                        publish_native_warp0_from_staged(
                            st, args, output_tile, staged_payload);
                        if constexpr (kProfileWaitPoll) {
                          reduce_native_mailbox_tile_pair_from_staged_profiled(
                              st, args, output_tile, staged_payload);
                        } else {
              #if NAIVERT_EXPERT_DOWN_ALLREDUCE_NATIVE_PARALLEL_PEER_POLL
                          reduce_native_mailbox_tile_pair_from_staged_parallel(
                              st, args, output_tile, staged_payload);
              #else
                          reduce_native_mailbox_tile_pair_from_staged(
                              st, args, output_tile, staged_payload);
              #endif
                        }
                        // Native Li1 uses BAR4 after the warp-0 mailbox reduction; helper
                        // warps are not participants and can finish/exit independently.
                        naivert::named_barrier_sync(256u,1u);

          }

    // The production warp-0 epilogue writes exp_out four elements at a time
    // while initializing its reduction accumulators. Other backends retain
    // the original per-row store and its finalize() visibility barrier.

    st.tx = outer_tx;
  }

#include "naivert/body/expert_down_allreduce/expert_down_w8a8.cuh"

  static constexpr int kTpRanks = naivert::geometry::kTpRanks;

  __device__ static int effective_total_tiles(
      const ExpertDownAllreduceArgs& args) {
    constexpr int kNativeTiles =
        (kHidden + kComputeOutputs - 1) / kComputeOutputs;
        return kNativeTiles;

  }

  __device__ static int capped_active_ctas(
      const ExpertDownAllreduceArgs& args) {
    return (kHidden + kComputeOutputs - 1) / kComputeOutputs;

  }

  __device__ static unsigned step_gen(const ExpertDownAllreduceArgs& args) {
    const ExpertDownRuntimeDomain* domain = args.stable_domain;
    unsigned tag = domain->native_mailbox_tag;
    if (domain->step_block_ptr != 0ull) {
      const auto* replay_iteration =
          reinterpret_cast<const std::uint64_t*>(
              domain->step_block_ptr);
      // Epoch above the per-layer tag byte; see unproj_o_allreduce and
      // down_allreduce::step_gen.
      tag = (static_cast<unsigned>(*replay_iteration) << 8) |
            (tag & 0xffu);
    }
    return tag;
  }

  // Four output warp tiles per CTA record: task/4 selects the CTA record,
  // task%4 the warp record, kOutputsPerTile / 2 tagged qwords each.
  struct MailboxLayout {
    __host__ __device__ static constexpr std::size_t tile_offset(
        int task, int qword) {
      return std::size_t(task / kExpertDownMailboxWarpsPerCta) *
                 naivert::kMailboxCtaStride +
             std::size_t(task % kExpertDownMailboxWarpsPerCta) *
                 naivert::kMailboxWarpStride +
             std::size_t(qword) * sizeof(unsigned long long);
    }
  };
  static_assert(MailboxLayout::tile_offset(1, 0) == 0x100);
  static_assert(MailboxLayout::tile_offset(4, 0) == 0x400);
  static_assert(MailboxLayout::tile_offset(127, 27) == 0x7fd8);
  static_assert(MailboxLayout::tile_offset(127, 27) + sizeof(unsigned long long) <=
                naivert::kMailboxPhaseStride);

  __device__ static naivert::TileMailbox<MailboxLayout> tile_mailbox(
      const ExpertDownAllreduceArgs& args) {
    const ExpertDownRuntimeDomain* domain = args.stable_domain;
    return naivert::TileMailbox<MailboxLayout>::make(
        reinterpret_cast<const std::uint8_t* const*>(
            domain->peer_mailbox_base_ptrs),
        args.rank, step_gen(args),
        std::size_t(domain->native_mailbox_slab) *
            naivert::kMailboxSlabStride);
  }

  __device__ static unsigned long long globaltimer_ns() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
  }

  __device__ static void apply_publish_rank_skew(
      const ExpertDownAllreduceArgs& args) {
    const ExpertDownRuntimeDomain* domain = args.stable_domain;
    if (domain == nullptr || domain->rank_skew_ns == 0u) {
      return;
    }
    const unsigned long long deadline =
        globaltimer_ns() + domain->rank_skew_ns;
    while (globaltimer_ns() < deadline) {
      asm volatile("nanosleep.u32 8;" ::: "memory");
    }
  }


  __device__ static void store_mailbox_tagged_payload_nonnull(
      unsigned long long* ptr, unsigned long long word) {
    naivert::store_tagged(ptr, word);
  }

  __device__ static void publish_native_warp0_from_staged(
      slot_ring::SlotRingState& st, const ExpertDownAllreduceArgs& args,
      int task, const std::uint32_t* staged_payload) {
    if (st.tid >= kExpertDownMailboxQwordsPerWarp ||
        staged_payload == nullptr) {
      return;
    }
    apply_publish_rank_skew(args);
    tile_mailbox(args).publish(task, st.tid, staged_payload[st.tid]);
  }

  __device__ static void reduce_native_mailbox_tile_pair_from_staged(
      slot_ring::SlotRingState& st, const ExpertDownAllreduceArgs& args, int task,
      const std::uint32_t* staged_payload) {
    constexpr int kQwordPairs = kExpertDownMailboxQwordsPerWarp / 2;
    if (st.tid >= kQwordPairs || staged_payload == nullptr) {
      return;
    }
    // Sequential poll. Order pinned by the golden: residual, then the eight
    // partials in rank order with the local pair in this rank's slot.
    const int qword = st.tid * 2;
    const int elem = task * kOutputsPerTile + qword * 2;
    const unsigned long long residual_word =
        *reinterpret_cast<const unsigned long long*>(args.residual + elem);
    naivert::Bf16x4Acc acc = naivert::Bf16x4Acc::of(
        static_cast<std::uint32_t>(residual_word),
        static_cast<std::uint32_t>(residual_word >> 32));
    // exp_out carries the residual alone until the reduced value lands.
    *reinterpret_cast<unsigned long long*>(args.exp_out + elem) = acc.packed();
    const auto mailbox = tile_mailbox(args);
    naivert::accumulate_rank_order(
        acc, args.rank, staged_payload[qword], staged_payload[qword + 1],
        [&](int source_slot) {
          return mailbox.wait_pair<false>(source_slot, task, qword);
        });
    auto* result = args.tp_result != nullptr ? args.tp_result : args.exp_out;
    *reinterpret_cast<unsigned long long*>(result + elem) = acc.packed();
  }

  __device__ static void
  reduce_native_mailbox_tile_pair_from_staged_parallel(
      slot_ring::SlotRingState& st, const ExpertDownAllreduceArgs& args, int task,
      const std::uint32_t* staged_payload) {
    constexpr int kQwords = kExpertDownMailboxQwordsPerWarp;
    constexpr int kQwordPairs = kQwords / 2;
    constexpr int kRemotePeers = kTpRanks - 1;
    constexpr int kPollItems = kQwords * kRemotePeers;
    if (staged_payload == nullptr) {
      return;
    }
    // 7 x 28 lanes poll one peer qword each into shared memory, then 14 lanes
    // reduce pairs. Order pinned by the golden: residual, local, then the
    // peer lanes.
    auto* peer_payload = reinterpret_cast<std::uint32_t*>(
        const_cast<std::uint32_t*>(staged_payload) + kQwords);
    const auto mailbox = tile_mailbox(args);
    if (st.tid < kPollItems) {
      const int peer_slot = st.tid / kQwords;
      const int qword = st.tid - peer_slot * kQwords;
      peer_payload[st.tid] = naivert::tagged_payload(naivert::wait_tagged<false>(
          mailbox.local_word(peer_slot, task, qword), mailbox.generation));
    }
    naivert::named_barrier_sync(256u,1u);
    if (st.tid >= kQwordPairs) {
      return;
    }
    const int qword = st.tid * 2;
    const int elem = task * kOutputsPerTile + qword * 2;
    const unsigned long long residual_word =
        *reinterpret_cast<const unsigned long long*>(args.residual + elem);
    naivert::Bf16x4Acc acc = naivert::Bf16x4Acc::of(
        static_cast<std::uint32_t>(residual_word),
        static_cast<std::uint32_t>(residual_word >> 32));
    acc.add(staged_payload[qword], staged_payload[qword + 1]);
    naivert::accumulate_peers_in_slot_order(acc, [&](int peer_slot) {
      return naivert::PayloadPair{peer_payload[peer_slot * kQwords + qword],
                                  peer_payload[peer_slot * kQwords + qword + 1]};
    });
    auto* result = args.tp_result != nullptr ? args.tp_result : args.exp_out;
    *reinterpret_cast<unsigned long long*>(result + elem) = acc.packed();
  }

  __device__ static void reduce_native_mailbox_tile_pair_from_staged_profiled(
      slot_ring::SlotRingState& st, const ExpertDownAllreduceArgs& args, int task,
      const std::uint32_t* staged_payload) {
    constexpr int kQwordPairs =
        kExpertDownMailboxQwordsPerWarp / 2;
    if (st.tid >= 32 || staged_payload == nullptr) {
      return;
    }
    const bool active = st.tid < kQwordPairs;
    unsigned poll_loads = 0u;
    unsigned long long wait_ns = 0ull;
    if (active) {
      const int qword = st.tid * 2;
      const int elem = task * kOutputsPerTile + qword * 2;
      const unsigned long long residual_word =
          *reinterpret_cast<const unsigned long long*>(args.residual + elem);
      naivert::Bf16x4Acc acc = naivert::Bf16x4Acc::of(
          static_cast<std::uint32_t>(residual_word),
          static_cast<std::uint32_t>(residual_word >> 32));
      *reinterpret_cast<unsigned long long*>(args.exp_out + elem) =
          acc.packed();
      const auto mailbox = tile_mailbox(args);
      const unsigned long long wait_start = globaltimer_ns();
      naivert::accumulate_rank_order(
          acc, args.rank, staged_payload[qword], staged_payload[qword + 1],
          [&](int source_slot) {
            const naivert::TaggedPair pair =
                naivert::wait_tagged_pair_counted<false>(
                    mailbox.local_word(source_slot, task, qword),
                    mailbox.generation, poll_loads);
            return naivert::PayloadPair{naivert::tagged_payload(pair.lo),
                                        naivert::tagged_payload(pair.hi)};
          });
      wait_ns = globaltimer_ns() - wait_start;
      auto* result =
          args.tp_result != nullptr ? args.tp_result : args.exp_out;
      *reinterpret_cast<unsigned long long*>(result + elem) = acc.packed();
    }
    __syncwarp();
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      const unsigned long long other_wait =
          __shfl_down_sync(0xffffffffu, wait_ns, offset);
      const unsigned other_polls =
          __shfl_down_sync(0xffffffffu, poll_loads, offset);
      wait_ns = wait_ns > other_wait ? wait_ns : other_wait;
      poll_loads += other_polls;
    }
    const ExpertDownRuntimeDomain* domain = args.stable_domain;
    if (st.tid == 0 && domain->wait_metrics_enabled != 0u &&
        domain->wait_metrics_base_ptr != 0ull &&
        domain->replay_count > 0u &&
        st.cta_rank >= 0 &&
        unsigned(st.cta_rank) < domain->wait_metrics_cta_capacity) {
      const unsigned replay_ordinal = domain->replay_count - 1u;
      if (replay_ordinal < domain->wait_metrics_replay_capacity) {
        auto* metrics = reinterpret_cast<ExpertDownWaitMetric*>(
            domain->wait_metrics_base_ptr);
        const std::size_t index =
            std::size_t(replay_ordinal) *
                domain->wait_metrics_cta_capacity +
            unsigned(st.cta_rank);
        ExpertDownWaitMetric metric{};
        metric.peer_wait_ns =
            wait_ns > 0xffffffffull
                ? 0xffffffffu
                : static_cast<std::uint32_t>(wait_ns);
        metric.poll_load_count = poll_loads;
        metrics[index] = metric;
      }
    }
  }

  __device__ static void stage_inner_weight_pane(
      slot_ring::SlotRingState& st, const ExpertDownAllreduceArgs& args,
      int output_tile, int expert_slot, unsigned char* pane) {
    int helper_tid = st.tid - slot_ring::StandardRoleLayout::kHelperBaseThread;
    int expert = logical_expert(args, expert_slot);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        const int packed_tile = kW8A8 ? output_tile * kComputeOutputs / kOutputsPerTile : output_tile;
        const int packed_half = kW8A8 ? (output_tile * kComputeOutputs % kOutputsPerTile) * 128 : 0;
        const auto* src_bytes = reinterpret_cast<const unsigned char*>(
            args.moe_w_down +
            ((kMx) ? (expert * kTiles + packed_tile) * kWeightPaneBytes
                  : (expert * kHidden + packed_tile * kOutputsPerTile) *
                        kInter));
        if (st.elected) {
          Slot::producer_expect_tx(st, std::uint32_t(kW8A8 ? kW8A8ExpertBytes : kWeightPaneBytes));
          if constexpr (kW8A8) {
            static_assert(!kW8A8 || kWeightPaneBytes == 16384,
                          "global weight packing retains two 8-KiB panels");
#pragma unroll
            for (int kg = 0; kg < 2; ++kg) {
              cuda::ptx::cp_async_bulk(
                  cuda::ptx::space_shared_t{}, cuda::ptx::space_global_t{},
                  st.deep_base + kg * kW8A8GroupPanelBytes +
                      (expert_slot % kW8A8GroupExperts) * kW8A8PanelBytes,
                  src_bytes + kg * 8192 + packed_half, std::uint32_t(kW8A8PanelBytes),
                  Slot::slot_barrier(st.full_barriers, st.tx.slot));
            }
          } else {
          cuda::ptx::cp_async_bulk(
              cuda::ptx::space_shared_t{}, cuda::ptx::space_global_t{}, pane,
              src_bytes, std::uint32_t(kWeightPaneBytes),
              Slot::slot_barrier(st.full_barriers, st.tx.slot));
          }
        }
        return;

#endif
    for (int off = helper_tid * int(sizeof(std::uint32_t));
         off < kWeightPaneBytes;
         off += kHelperThreads * int(sizeof(std::uint32_t))) {
      *reinterpret_cast<std::uint32_t*>(pane + off) = 0u;
    }
  }

  __device__ static int logical_expert(const ExpertDownAllreduceArgs& args,
                                       int slot) {
    if (slot < kSharedExperts) {
      return slot;
    }
    return args.sel_indices[slot - kSharedExperts] + kSharedExperts;
  }

  __device__ static float route_weight(const ExpertDownAllreduceArgs& args,
                                       int slot) {
    if (slot < kSharedExperts) {
      return 1.0f;
    }
    return args.sel_probs[slot - kSharedExperts];
  }

  __device__ static float load_scale(const ExpertDownAllreduceArgs& args,
                                     int expert, int hidden_col,
                                     int k_group) {

    // pack_down's layout: scale_in [expert, hidden/128, inter/128] -> each block
    // scale repeated 16 times -> reshape(expert, tiles, kOutputsPerTile / 4 live
    // entries) -> padded to 16 per tile -> [expert, tiles * 8, 2]. Any repeat lane
    // is equivalent, so use repeat lane 0 and invert the reshape/pad.
    int hidden_block;
    int group;
        hidden_block = hidden_col / 128;
        group = k_group;

    constexpr int kLiveScalesPerTile = kOutputsPerTile / 4;
    int repeated_linear = (hidden_block * 16) * 2 + group;
    int tile_row = repeated_linear / kLiveScalesPerTile;
    int tile_col = repeated_linear - tile_row * kLiveScalesPerTile;
    int padded_linear = tile_row * 16 + tile_col;
    int idx = expert * (kTiles * 16) + padded_linear;
    return naivert::expert_weight_scale(args.moe_w_down_scale[idx]);
  }

  __device__ static std::uint32_t convert_fp8_pair_as_bf16x2(
      std::uint16_t raw) {
    std::uint32_t out;
    asm volatile("{cvt.rn.bf16x2.e4m3x2 %0, %1;}\n"
                 : "=r"(out)
                 : "h"(raw));
    return out;
  }

  __device__ static std::uint32_t load_bf16_pair(
      const __nv_bfloat16* ptr) {
    // bf16 activations feed the bf16 HMMA directly: no fp16 widening, so
    // no 65,504 ceiling.
    return *reinterpret_cast<const std::uint32_t*>(ptr);
  }

  __device__ __forceinline__ static void
  accumulate_inner_expert_tile_native_k_split(
      const ExpertDownAllreduceArgs& args, int output_tile, int expert_slot,
      const unsigned char* pane, const __nv_bfloat16* vec_base, int warp,
      int group, int tid4, const NativeComputeMetadata* metadata,
      float& out0_acc, float& out1_acc, float& out2_acc, float& out3_acc,
      float& out4_acc, float& out5_acc, float& out6_acc, float& out7_acc) {
    // One warp consumes a contiguous K=32 slice and reuses each B operand for
    // four independent M=16 fragments.  This is the Li1 SASS shape: eight
    // HMMA.16816 instructions per expert slot, with four independent
    // accumulator groups and a later eight-warp partial reduction.
    const int k_base = warp * 32;
    const int lane_k4 = k_base + tid4 * 4;
    // The native pane swizzle stores, in one aligned 16-byte run:
    //   x: row M,   K{0..3}
    //   y: row M+8, K{0..3}
    //   z: row M,   K{16..19}
    //   w: row M+8, K{16..19}
    // Low/high halves therefore form two legal, K-permuted HMMA fragments.
    // Permuting the matching B values preserves the dot product while replacing
    // eight scalar shared loads with one conflict-free LDS.128.
    const std::uint32_t b_low01 =
        load_bf16_pair(vec_base + lane_k4);
    const std::uint32_t b_low23 =
        load_bf16_pair(vec_base + lane_k4 + 16);
    const std::uint32_t b_high01 =
        load_bf16_pair(vec_base + lane_k4 + 2);
    const std::uint32_t b_high23 =
        load_bf16_pair(vec_base + lane_k4 + 18);
    float d[4][4] = {};
#pragma unroll
    for (int m_tile = 0; m_tile < kFullMTiles; ++m_tile) {
      const int row0_local = m_tile * 16 + group;
      const auto packed = *reinterpret_cast<const uint4*>(
          pane + pane_storage_offset(row0_local, lane_k4));
      mma_16816_f32_bf16(
          convert_fp8_pair_as_bf16x2(std::uint16_t(packed.x)),
          convert_fp8_pair_as_bf16x2(std::uint16_t(packed.y)),
          convert_fp8_pair_as_bf16x2(std::uint16_t(packed.z)),
          convert_fp8_pair_as_bf16x2(std::uint16_t(packed.w)), b_low01,
          b_low23, d[m_tile][0], d[m_tile][1], d[m_tile][2],
          d[m_tile][3]);
      mma_16816_f32_bf16(
          convert_fp8_pair_as_bf16x2(std::uint16_t(packed.x >> 16)),
          convert_fp8_pair_as_bf16x2(std::uint16_t(packed.y >> 16)),
          convert_fp8_pair_as_bf16x2(std::uint16_t(packed.z >> 16)),
          convert_fp8_pair_as_bf16x2(std::uint16_t(packed.w >> 16)), b_high01,
          b_high23, d[m_tile][0], d[m_tile][1], d[m_tile][2],
          d[m_tile][3]);

    }

    if constexpr (kTailRows == 8) {
      // The final physical chunk has only rows 48..55. Its row-major-within-K32
      // tail places the four pairs owned by a lane in one aligned eight-byte
      // run. Load it once and unpack from registers.
      const int tail_row = kFullMTiles * 16 + group;
      const unsigned long long tail_packed =
          *reinterpret_cast<const unsigned long long*>(
              pane + pane_storage_offset(tail_row, lane_k4));
      mma_16816_f32_bf16(
          convert_fp8_pair_as_bf16x2(std::uint16_t(tail_packed)),
          0u,
          convert_fp8_pair_as_bf16x2(std::uint16_t(tail_packed >> 32)),
          0u, b_low01, b_low23, d[3][0], d[3][1], d[3][2], d[3][3]);
      mma_16816_f32_bf16(
          convert_fp8_pair_as_bf16x2(std::uint16_t(tail_packed >> 16)),
          0u,
          convert_fp8_pair_as_bf16x2(std::uint16_t(tail_packed >> 48)),
          0u, b_high01, b_high23, d[3][0], d[3][1], d[3][2], d[3][3]);
    }

    if (tid4 == 0) {
      const float route = metadata->route[expert_slot];
      const int k_group = warp >> 2;
      const int row0 = group;
      const int row2 = 16 + group;
      const int row4 = 32 + group;
      const int row6 = 48 + group;
      const int output_base = output_tile * kOutputsPerTile;
      const int hidden_block0 = output_base / 128;
      // MX8: one ue8m0 scale per row at K block `warp`. FP8: the 128 x 128 block scale from the metadata.
      const unsigned char* pane_scales_mx = pane + kOutputsPerTile * kInter;
#define NAIVERT_NATIVE_ROW_SCALE(row)                                      \
  (kMx ? naivert::mx::ue8m0_to_float(pane_scales_mx[(row) * (kInter / 32) + warp]) \
        : metadata->scale[expert_slot]                                    \
                        [((output_base + (row)) / 128) != hidden_block0   \
                             ? 1                                          \
                             : 0][k_group])
      out0_acc +=
          d[0][0] * (NAIVERT_NATIVE_ROW_SCALE(row0) * route);
      out1_acc +=
          d[0][2] * (NAIVERT_NATIVE_ROW_SCALE(row0 + 8) * route);
      out2_acc +=
          d[1][0] * (NAIVERT_NATIVE_ROW_SCALE(row2) * route);
      out3_acc +=
          d[1][2] * (NAIVERT_NATIVE_ROW_SCALE(row2 + 8) * route);
      out4_acc +=
          d[2][0] * (NAIVERT_NATIVE_ROW_SCALE(row4) * route);
      out5_acc +=
          d[2][2] * (NAIVERT_NATIVE_ROW_SCALE(row4 + 8) * route);
      if (row6 < kOutputsPerTile) {
        out6_acc +=
            d[3][0] * (NAIVERT_NATIVE_ROW_SCALE(row6) * route);
      }
      if (row6 + 8 < kOutputsPerTile) {
        out7_acc +=
            d[3][2] * (NAIVERT_NATIVE_ROW_SCALE(row6 + 8) * route);
      }
#undef NAIVERT_NATIVE_ROW_SCALE
    }
  }

  __device__ static void mma_16816_f32_bf16(std::uint32_t a0,
                                           std::uint32_t a1,
                                           std::uint32_t a2,
                                           std::uint32_t a3,
                                           std::uint32_t b0,
                                           std::uint32_t b1, float& c0,
                                           float& c1, float& c2, float& c3) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    (void)a0;
    (void)a1;
    (void)a2;
    (void)a3;
    (void)b0;
    (void)b1;
    (void)c0;
    (void)c1;
    (void)c2;
    (void)c3;
#endif
  }

};

}  // namespace naivert_replica_expert_down_allreduce
