#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdint>

#include <cuda/ptx>

#include "naivert/pdl/grid_dependency.cuh"
#include "naivert/body/expert_select_up_gate_silu/expert_select_up_gate_silu_args.hpp"
#include "naivert/body/expert_select_up_gate_silu/expert_select_up_gate_silu_traits.cuh"
#include "naivert/common/role_layout.hpp"
#include "naivert/slot_ring/state.cuh"
#include "naivert/slot_ring/protocol.cuh"

namespace naivert_replica_expert_select_up_gate_silu {

namespace slot_ring = naivert;

struct ExpertSelectUpGateSiluBody;

}  // namespace naivert_replica_expert_select_up_gate_silu

namespace naivert_replica_expert_select_up_gate_silu {

struct ExpertSelectUpGateSiluBody {
  using Slot = slot_ring::SlotProtocol<kSlotDepth>;
  // Pages the helpers stage before the loop; each is refilled four iterations later.
  static constexpr int kBootstrapSlots = 4;

  __device__ static slot_ring::SlotRingWorkArgs work_args(
      const ExpertSelectUpGateSiluArgs& args) {
    return args.progress;
  }

  __device__ static void bind_shared_layout(
      slot_ring::SlotRingState& st, const ExpertSelectUpGateSiluArgs&,
      unsigned char* smem_raw) {
    unsigned char* base = smem_raw;
    st.smem_base = smem_raw;
    st.control_base = base + kBarrierGroup0Offset;
    st.full_barriers =
        reinterpret_cast<std::uint64_t*>(base + kBarrierGroup0Offset);
    st.empty_barriers =
        reinterpret_cast<std::uint64_t*>(base + kBarrierGroup1Offset);
    st.early_base = base + kMetadataOffset;
    st.deep_base = base + kPaneBaseOffset;
    st.scratch_base = base + kActivationScratchOffset;
    st.early_stride = 0;
    st.deep_stride = kPaneStride;
  }

  __device__ static void init_control_slab(
      slot_ring::SlotRingState& st, const ExpertSelectUpGateSiluArgs&) {
    if (st.tid < kSlotDepth) {
      Slot::init_transaction_slot(st, st.tid);
    }
    if (st.tid == kSlotDepth) {
      slot_ring::Barrier::init(private_barrier(st), 0x100u);
    }
  }

  __device__ static void dispatch(
      slot_ring::SlotRingState& st, const ExpertSelectUpGateSiluArgs& args) {
    if (st.cta_rank >= args.total_tiles) {
      st.active = false;
      return;
    }
    if (st.tid == 0 || st.tid == slot_ring::StandardRoleLayout::kHelperBaseThread) {
      auto* meta = reinterpret_cast<std::uint64_t*>(st.early_base);
      int slot = tile_slot(st, args);
      int group = tile_group(st, args);
      std::uintptr_t dst = reinterpret_cast<std::uintptr_t>(
          args.y + slot * args.y_slot_stride + group * args.outputs_per_group);
      meta[0] = static_cast<std::uint64_t>(dst);
    }
  }

  __device__ static int slot_task(
      slot_ring::SlotRingState& st, const ExpertSelectUpGateSiluArgs& args, int) {
    if (st.cta_rank >= args.total_tiles) {
      return -1;
    }
    if (is_consumer(st)) {
      return st.tx.iteration < args.pages ? st.tx.iteration : -1;
    }
    int page_count = helper_page_count(st, args);
    if (st.tx.iteration >= page_count) {
      return -1;
    }
    return st.tx.iteration;
  }

  __device__ static void consumer_bootstrap(
      slot_ring::SlotRingState& st, const ExpertSelectUpGateSiluArgs& args) {
    if (!is_consumer(st)) {
      return;
    }
    int token = token_index(st, args);
    int consumer_tid = st.warp_id * 32 + st.lane_id;
    auto* dst16 = reinterpret_cast<uint4*>(bootstrap_x_ptr(st));
    const auto* src16 =
        reinterpret_cast<const uint4*>(
            args.x + token * args.x_stride);
#pragma unroll
    for (int idx = consumer_tid; idx < kHidden / 8;
         idx += kConsumerThreads) {
      __pipeline_memcpy_async(dst16 + idx, src16 + idx, 16);
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    // Quantization redistributes K blocks across warps, so each reader needs
    // every consumer's async copy complete, not just its own pieces.
    consumer_sync(st);
    // Routed slots need the selection before their weights can be addressed;
    // shared slots (fixed weights) skip it, except the tile that publishes the
    // selection for the rest of the layer.
    const bool computes_route =
        tile_slot(st, args) >= kSharedExperts || local_tile(st, args) == 0;
    if (computes_route) {
      compute_route(st, args, token);
      slot_ring::Barrier::arrive(private_barrier(st));
    } else {
      __pipeline_wait_prior(0);
      convert_activation_fp8(st, 0, kConsumerThreads);
      consumer_sync(st);
    }
  }

  __device__ static void convert_activation_fp8(
      slot_ring::SlotRingState& st, int first_tid, int worker_count) {
    int consumer_tid = st.warp_id * 32 + st.lane_id;
    if (consumer_tid < first_tid) {
      return;
    }
    const int worker_tid = consumer_tid - first_tid;
    const auto* x_bf16 = bootstrap_x_ptr(st);
    auto* x_fp8 = activation_fp8_ptr(st);
    if constexpr (naivert::kExpertActivationFp32) {
      naivert::quantize_block128_row(x_bf16, x_fp8, activation_scales(),
                                    kHidden, worker_tid, worker_count);
      return;
    }
#pragma unroll
    for (int idx = worker_tid; idx < kHidden / 8;
         idx += worker_count) {
      const int base = idx * 8;
#pragma unroll
      for (int item = 0; item < 8; ++item) {
        x_fp8[base + item] = __nv_fp8_e4m3(
            __bfloat162float(x_bf16[base + item]));
      }
    }
  }

  // The kernel's control flow: role split, layout binding, control-slab and
  // barrier init on the control lanes, async-proxy fences, BAR0
  // (__syncthreads), dispatch, a four-page helper bootstrap alongside the
  // consumer bootstrap (no rendezvous between them), then the transport
  // loop: consumers process one projection page per task, helpers refill the
  // page they staged four iterations ago.  kPdl: the consumers wait for the
  // primary grid (the router) before reading x or the scores; the helpers
  // stage the immutable slot-zero expert pages meanwhile.
  template <bool kPdl>
  __device__ static void run(const ExpertSelectUpGateSiluArgs& args,
                             unsigned char* shared) {
    slot_ring::SlotRingState st{};
    st.tid = int(threadIdx.x);
    st.lane_id = slot_ring::StandardRoleLayout::lane_id(st.tid);
    st.warp_id = slot_ring::StandardRoleLayout::warp_id(st.tid);
    st.cta_rank = int(blockIdx.x);
    st.active = true;
    st.elected = false;
    st.smem_base = shared;
    st.control_base = nullptr;
    st.full_barriers = nullptr;
    st.empty_barriers = nullptr;
    st.early_base = nullptr;
    st.deep_base = nullptr;
    st.scratch_base = nullptr;
    st.early_stride = 0;
    st.deep_stride = 0;
    st.body_acc[0] = 0.0f;
    st.body_acc[1] = 0.0f;
    st.tx = {0, 0, 0};
    if (slot_ring::StandardRoleLayout::is_consumer_a(st.tid)) {
      st.role = slot_ring::ThreadRole::kConsumerA;
    } else if (slot_ring::StandardRoleLayout::is_consumer_b(st.tid)) {
      st.role = slot_ring::ThreadRole::kConsumerB;
    } else {
      st.role = slot_ring::ThreadRole::kHelper;
    }

    bind_shared_layout(st, args, shared);
    {
      const slot_ring::SlotRingWorkArgs work = work_args(args);
      int active_ctas =
          work.active_ctas > 0 ? work.active_ctas : int(gridDim.x);
      if (active_ctas > int(gridDim.x)) {
        active_ctas = int(gridDim.x);
      }
      if (work.total_tiles < active_ctas) {
        active_ctas = work.total_tiles;
      }
      st.active = st.active && st.cta_rank < active_ctas;
    }
    if (slot_ring::StandardRoleLayout::is_control_init_lane(st.tid)) {
      init_control_slab(st, args);
    }
    cutlass::arch::fence_view_async_shared();
    __threadfence();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
    __syncthreads();  // BAR0
    if (!st.active) {
      return;
    }
    dispatch(st, args);

    st.elected = st.role == slot_ring::ThreadRole::kHelper &&
                 st.warp_id == slot_ring::StandardRoleLayout::kHelperBaseWarp &&
                 slot_ring::elect_sync_compat(0xffffffffu);
    if (st.role == slot_ring::ThreadRole::kHelper) {
#pragma unroll
      for (int s = 0; s < kBootstrapSlots; ++s) {
        const int old_slot = st.tx.slot;
        const int old_iteration = st.tx.iteration;
        st.tx.slot = s;
        st.tx.iteration = s;
        helper_stage(st, args, slot_task(st, args, s), true);
        st.tx.slot = old_slot;
        st.tx.iteration = old_iteration;
      }
    }
    if (is_consumer(st)) {
      if constexpr (kPdl) {
        naivert::pdl::wait_for_primary<true>();
      }
      consumer_bootstrap(st, args);
    }

    while (true) {
      const int slot = st.tx.slot;
      if (is_consumer(st)) {
        const int task = slot_task(st, args, slot);
        if (task < 0) {
          break;
        }
        process_projection_page(st, args, task);
      } else {
        slot_ring::SlotRingState refill_st = st;
        Slot::prepare_same_slot_reuse(refill_st, kBootstrapSlots);
        const int task = slot_task(refill_st, args, slot);
        if (task < 0) {
          break;
        }
        helper_stage(refill_st, args, task, true);
      }
      Slot::advance(st);
    }
  }

 private:
  __device__ static bool is_consumer(const slot_ring::SlotRingState& st) {
    return st.role == slot_ring::ThreadRole::kConsumerA ||
           st.role == slot_ring::ThreadRole::kConsumerB;
  }

  __device__ static __nv_bfloat16* bootstrap_x_ptr(
      const slot_ring::SlotRingState&) {
    __shared__ __align__(128) __nv_bfloat16 bootstrap_x[kHidden];
    return bootstrap_x;
  }

  __device__ static std::uint64_t* metadata64(const slot_ring::SlotRingState& st) {
    return reinterpret_cast<std::uint64_t*>(st.early_base);
  }

  __device__ static __nv_fp8_e4m3* activation_fp8_ptr(
      const slot_ring::SlotRingState& st, int token = 0) {
    return reinterpret_cast<__nv_fp8_e4m3*>(st.scratch_base + 0x5000) +
           token * kHidden;
  }

  __device__ static float* activation_scales() {
    __shared__ float values[kHidden / 128];
    return values;
  }

  __device__ static float* output_accum_ptr(const slot_ring::SlotRingState& st) {
    return reinterpret_cast<float*>(st.scratch_base + 0x7000);
  }

  __device__ static float* route_sigmoid_ptr(const slot_ring::SlotRingState& st) {
    return reinterpret_cast<float*>(st.scratch_base + 0x4000);
  }

  __device__ static float* route_select_ptr(const slot_ring::SlotRingState& st) {
    return reinterpret_cast<float*>(st.scratch_base + 0x4400);
  }

  __device__ static int* route_selected_ptr(const slot_ring::SlotRingState& st) {
    return reinterpret_cast<int*>(st.scratch_base + 0x4800);
  }

  __device__ static float* route_group_score_ptr(const slot_ring::SlotRingState& st) {
    return reinterpret_cast<float*>(st.scratch_base + 0x4880);
  }

  __device__ static int* route_group_keep_ptr(const slot_ring::SlotRingState& st) {
    return reinterpret_cast<int*>(st.scratch_base + 0x48A0);
  }

  __device__ static float* route_warp_score_ptr(const slot_ring::SlotRingState& st) {
    return reinterpret_cast<float*>(st.scratch_base + 0x48C0);
  }

  __device__ static int* route_warp_index_ptr(const slot_ring::SlotRingState& st) {
    return reinterpret_cast<int*>(st.scratch_base + 0x48E0);
  }

  __device__ static int* route_reuse_ptr(const slot_ring::SlotRingState& st) {
    return reinterpret_cast<int*>(st.scratch_base + 0x4900);
  }

  __device__ static int* x_reuse_ptr(const slot_ring::SlotRingState& st) {
    return reinterpret_cast<int*>(st.scratch_base + 0x4904);
  }

  __device__ static int* score_reuse_ptr(const slot_ring::SlotRingState& st) {
    return reinterpret_cast<int*>(st.scratch_base + 0x4908);
  }

  __device__ static int* token_x_reuse_ptr(const slot_ring::SlotRingState& st) {
    return reinterpret_cast<int*>(st.scratch_base + 0x490C);
  }

  __device__ static std::uint64_t* private_barrier(const slot_ring::SlotRingState& st) {
    return reinterpret_cast<std::uint64_t*>(
        st.smem_base + kPrivateSyncOffset);
  }

  __device__ static void consumer_sync(slot_ring::SlotRingState& st) {
    if (is_consumer(st)) {
      slot_ring::consumer_group_named_sync();  // BAR4
    }
  }

  __device__ static int tile_slot(
      const slot_ring::SlotRingState& st, const ExpertSelectUpGateSiluArgs& args) {
    return local_tile(st, args) / args.output_groups;
  }

  __device__ static int tile_group(
      const slot_ring::SlotRingState& st, const ExpertSelectUpGateSiluArgs& args) {
    int slot = tile_slot(st, args);
    return local_tile(st, args) - slot * args.output_groups;
  }

  __device__ static int tiles_per_token(const ExpertSelectUpGateSiluArgs& args) {
    return args.output_slots * args.output_groups;
  }

  __device__ static int token_index(
      const slot_ring::SlotRingState& st, const ExpertSelectUpGateSiluArgs& args) {
    (void)st;
    (void)args;
    return 0;
  }

  __device__ static int local_tile(
      const slot_ring::SlotRingState& st, const ExpertSelectUpGateSiluArgs& args) {
    (void)args;
    return st.cta_rank;
  }

  __device__ static bool route_reuses_weights(
      const slot_ring::SlotRingState& st, const ExpertSelectUpGateSiluArgs& args) {
    (void)st;
    (void)args;
    return true;
  }

  __device__ static bool output_reuses_tokens(
      const slot_ring::SlotRingState& st, const ExpertSelectUpGateSiluArgs& args) {
    (void)st;
    return args.seq_len > 1;
  }

  __device__ static int helper_page_count(
      const slot_ring::SlotRingState& st, const ExpertSelectUpGateSiluArgs& args) {
    return route_reuses_weights(st, args) ? args.pages : args.seq_len * args.pages;
  }

  __device__ static float fast_exp2(float v) {
    float out;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(v));
    return out;
  }

  __device__ static float route_score(
      const slot_ring::SlotRingState& st, const ExpertSelectUpGateSiluArgs& args,
      int token, int expert) {
    (void)st;
    return args.scores[token * args.routed_experts + expert];
  }

  __device__ static void compute_route(
      slot_ring::SlotRingState& st, const ExpertSelectUpGateSiluArgs& args,
      int token) {
    int consumer_tid = st.warp_id * 32 + st.lane_id;
    auto* route_scores = route_sigmoid_ptr(st);
    auto* select = route_select_ptr(st);
    float s = route_score(st, args, token, consumer_tid);
    float sig = s;
    float select_value = sig + args.bias[consumer_tid];
    route_scores[consumer_tid] = sig;
    select[consumer_tid] = select_value;

    // Each consumer warp owns one 32-expert group.  Find its top two with
    // shuffles instead of having one lane serially reload all 32 shared
    // values.  The lower expert index wins ties, matching the old scan.
    auto* group_scores = route_group_score_ptr(st);
    float group_best0 = select_value;
    int group_best0_index = consumer_tid;
    warp_reduce_max_pair(group_best0, group_best0_index);
    const int first_index =
        __shfl_sync(0xffffffffu, group_best0_index, 0);
    float group_best1 =
        consumer_tid == first_index ? -3.402823466e38f : select_value;
    int group_best1_index = consumer_tid;
    warp_reduce_max_pair(group_best1, group_best1_index);
    if (st.lane_id == 0) {
      group_scores[st.warp_id] = group_best0 + group_best1;
    }
    consumer_sync(st);

    auto* selected = route_selected_ptr(st) + token * kSelectedExperts;
    if (st.warp_id == 0) {
      // Keep the best kRouteKeptGroups groups entirely inside warp 0 (every
      // group when the model routes without a group limit).
      float candidate_group =
          st.lane_id < 8 ? group_scores[st.lane_id]
                         : -3.402823466e38f;
      int candidate_group_index = st.lane_id;
      unsigned int kept_group_mask = 0;
      if constexpr (kRouteKeptGroups >= 8) {
        (void)candidate_group;
        (void)candidate_group_index;
        kept_group_mask = 0xffu;
      } else {
#pragma unroll
        for (int pick = 0; pick < kRouteKeptGroups; ++pick) {
          float best_group = candidate_group;
          int best_group_index = candidate_group_index;
          warp_reduce_max_pair(best_group, best_group_index);
          const int winner =
              __shfl_sync(0xffffffffu, best_group_index, 0);
          if (st.lane_id == 0) {
            kept_group_mask |= 1u << winner;
          }
          if (st.lane_id == winner) {
            candidate_group = -3.402823466e38f;
          }
        }
      }
      kept_group_mask =
          __shfl_sync(0xffffffffu, kept_group_mask, 0);

      // Lane L owns experts L, L+32, ... L+224.  Select the global top eight
      // from the retained groups with warp reductions only.
      float lane_scores[8];
#pragma unroll
      for (int group = 0; group < 8; ++group) {
        const int expert = group * 32 + st.lane_id;
        lane_scores[group] =
            (kept_group_mask & (1u << group))
                ? select[expert]
                : -3.402823466e38f;
      }
#pragma unroll
      for (int pick = 0; pick < 8; ++pick) {
        float lane_best = -3.402823466e38f;
        int lane_best_index = st.lane_id;
#pragma unroll
        for (int group = 0; group < 8; ++group) {
          const int expert = group * 32 + st.lane_id;
          const float value = lane_scores[group];
          if (value > lane_best ||
              (value == lane_best && expert < lane_best_index)) {
            lane_best = value;
            lane_best_index = expert;
          }
        }
        warp_reduce_max_pair(lane_best, lane_best_index);
        const int winner =
            __shfl_sync(0xffffffffu, lane_best_index, 0);
        if (st.lane_id == 0) {
          selected[pick] = winner;
        }
#pragma unroll
        for (int group = 0; group < 8; ++group) {
          if (group * 32 + st.lane_id == winner) {
            lane_scores[group] = -3.402823466e38f;
          }
        }
      }
    }

    __pipeline_wait_prior(0);
    convert_activation_fp8(st, 32, kConsumerThreads - 32);
    consumer_sync(st);

    if (consumer_tid == 0) {
      float prob_sum = 0.0f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        prob_sum += route_scores[selected[i]];
      }
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        int expert = selected[i];
        if (local_tile(st, args) == 0) {
          float prob = route_scores[expert] / (prob_sum + (naivert::kExpertActivationFp32 ? 1e-20f : 0.0f)) * args.route_scale;
          if (args.seq_len == 1) {
            args.expert_indices[i] = expert;
            args.expert_probs[i] = prob;
            continue;
          }
#pragma unroll
          for (int out_token = 0; out_token < 4; ++out_token) {
            if (out_token < args.seq_len) {
              args.expert_indices[out_token * args.selected_experts + i] = expert;
              args.expert_probs[out_token * args.selected_experts + i] = prob;
            }
          }
        }
      }
    }
  }

  __device__ static void publish_route_reuse(
      slot_ring::SlotRingState& st, const ExpertSelectUpGateSiluArgs& args) {
    int consumer_tid = st.warp_id * 32 + st.lane_id;
    if (consumer_tid == 0) {
      *route_reuse_ptr(st) = 1;
    }
    consumer_sync(st);
  }

  __device__ static void copy_route_from_first(
      slot_ring::SlotRingState& st, const ExpertSelectUpGateSiluArgs& args,
      int token) {
    int consumer_tid = st.warp_id * 32 + st.lane_id;
    auto* selected = route_selected_ptr(st);
    if (consumer_tid < kSelectedExperts) {
      int expert = selected[consumer_tid];
      selected[token * kSelectedExperts + consumer_tid] = expert;
      if (local_tile(st, args) == 0) {
        args.expert_indices[token * args.selected_experts + consumer_tid] = expert;
        args.expert_probs[token * args.selected_experts + consumer_tid] =
            args.expert_probs[consumer_tid];
      }
    }
    consumer_sync(st);
  }

  __device__ static void warp_reduce_max_pair(float& score, int& index) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
      float other_score = __shfl_down_sync(0xffffffffu, score, delta);
      int other_index = __shfl_down_sync(0xffffffffu, index, delta);
      if (other_score > score ||
          (other_score == score && other_index < index)) {
        score = other_score;
        index = other_index;
      }
    }
  }

  __device__ static const __nv_fp8_e4m3* packed_page_ptr(
      const slot_ring::SlotRingState& st, const ExpertSelectUpGateSiluArgs& args,
      int task) {
    int slot = tile_slot(st, args);
    int group = tile_group(st, args);
    int token = token_index(st, args);
    int page = route_reuses_weights(st, args) ? task : task - token * kPages;
    // A shared slot addresses its packed expert directly; a routed slot
    // follows the selection, offset past the shared experts.
    int expert = slot;
    if (slot >= kSharedExperts) {
      expert = route_selected_ptr(st)[token * kSelectedExperts + slot - kSharedExperts] +
               kSharedExperts;
    }
    return args.packed_weights + expert * args.packed_expert_stride +
           (group * args.pages + page) * args.packed_page_stride;
  }

  __device__ static naivert::ExpertActivation* output_tile_ptr(
      const slot_ring::SlotRingState& st, const ExpertSelectUpGateSiluArgs& args,
      int token) {
    int slot = tile_slot(st, args);
    int group = tile_group(st, args);
    return args.y + token * args.output_slots * args.y_slot_stride +
           slot * args.y_slot_stride + group * args.outputs_per_group;
  }

  __device__ static void helper_stage(
      slot_ring::SlotRingState& st, const ExpertSelectUpGateSiluArgs& args, int page,
      bool wait_for_release) {
    if (st.role != slot_ring::ThreadRole::kHelper || page < 0) {
      return;
    }
    if (wait_for_release) {
      Slot::producer_acquire(st);
    }
    if (tile_slot(st, args) >= kSharedExperts && page == 0) {
      while (!slot_ring::Barrier::try_wait(private_barrier(st), 0)) {
#if defined(__CUDA_ARCH__)
        __nanosleep(32);
#endif
      }
    }
    void* dst = st.deep_slot_ptr();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if (st.elected) {
      const void* src = packed_page_ptr(st, args, page);
      Slot::producer_expect_tx(st, std::uint32_t(kPaneVisibleBytes));
      cp_async_bulk_evict_first(
          dst, src, std::uint32_t(kPaneVisibleBytes),
          Slot::slot_barrier(st.full_barriers, st.tx.slot));
    }
#else
    const void* src = packed_page_ptr(st, args, page);
    int helper_tid = st.tid - slot_ring::StandardRoleLayout::kHelperBaseThread;
    auto* dst4 = reinterpret_cast<uint4*>(dst);
    const auto* src4 = reinterpret_cast<const uint4*>(src);
    for (int idx = helper_tid; idx < kPaneVisibleBytes / int(sizeof(uint4));
         idx += kHelperThreads) {
      dst4[idx] = src4[idx];
    }
#endif
    Slot::producer_commit(st);
  }

  __device__ static __forceinline__ void
  cp_async_bulk_evict_first(
      void* destination, const void* source, std::uint32_t bytes,
      std::uint64_t* barrier) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const std::uint32_t shared_destination =
        static_cast<std::uint32_t>(
            __cvta_generic_to_shared(destination));
    const std::uint64_t global_source =
        __cvta_generic_to_global(source);
    const std::uint32_t shared_barrier =
        static_cast<std::uint32_t>(
            __cvta_generic_to_shared(barrier));
    asm volatile(
        "{\n"
        "  .reg .b64 policy;\n"
        "  createpolicy.fractional.L2::evict_first.b64 "
        "policy, 1.0;\n"
        "  cp.async.bulk.shared::cta.global."
        "mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1], %2, [%3], policy;\n"
        "}\n"
        :
        : "r"(shared_destination), "l"(global_source),
          "r"(bytes), "r"(shared_barrier)
        : "memory");
#else
    (void)destination;
    (void)source;
    (void)bytes;
    (void)barrier;
#endif
  }

  __device__ static void initialize_projection_state(
      slot_ring::SlotRingState&, const ExpertSelectUpGateSiluArgs&) {}

  __device__ static void finish_projection_token(
      slot_ring::SlotRingState& st, const ExpertSelectUpGateSiluArgs& args, int token,
      float d0, float d1, float d2, float d3) {
    auto* accum = output_accum_ptr(st);
    if (st.warp_id <= 7 && (st.lane_id & 3) == 0) {
      const int row = st.lane_id >> 2;
      accum[st.warp_id * kOutputsPerGroup + row] = d0;
      accum[st.warp_id * kOutputsPerGroup + row + 8] = d2;
    }
    consumer_sync(st);
    auto* dst = output_tile_ptr(st, args, token);
    if (st.warp_id == 0 && st.lane_id < kOutputsPerGroup) {
      float gate = 0.0f;
      float up = 0.0f;
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        gate += accum[part * kOutputsPerGroup + st.lane_id];
        up += accum[(4 + part) * kOutputsPerGroup + st.lane_id];
      }
      float sig = 1.0f / (1.0f + fast_exp2(-gate * 1.4426950408889634f));
      dst[st.lane_id] = naivert::expert_activation_from_float(gate * sig * up);
    }
    if (token + 1 < args.seq_len) {
      consumer_sync(st);
    }
  }

  __device__ static void finish_projection_reused_tokens(
      slot_ring::SlotRingState& st, const ExpertSelectUpGateSiluArgs& args, float d0,
      float d1, float d2, float d3) {
    auto* accum = output_accum_ptr(st);
    if (st.warp_id <= 7 && (st.lane_id & 3) == 0) {
      const int row = st.lane_id >> 2;
      accum[st.warp_id * kOutputsPerGroup + row] = d0;
      accum[st.warp_id * kOutputsPerGroup + row + 8] = d2;
    }
    consumer_sync(st);
    if (st.warp_id == 0 && st.lane_id < kOutputsPerGroup) {
      float gate = 0.0f;
      float up = 0.0f;
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        gate += accum[part * kOutputsPerGroup + st.lane_id];
        up += accum[(4 + part) * kOutputsPerGroup + st.lane_id];
      }
      float sig = 1.0f / (1.0f + fast_exp2(-gate * 1.4426950408889634f));
      naivert::ExpertActivation value = naivert::expert_activation_from_float(gate * sig * up);
      if (args.seq_len == 1) {
        output_tile_ptr(st, args, 0)[st.lane_id] = value;
        return;
      }
#pragma unroll
      for (int token = 0; token < 4; ++token) {
        if (token < args.seq_len) {
          output_tile_ptr(st, args, token)[st.lane_id] = value;
        }
      }
    }
  }

  __device__ static void mma_m16n8k32_e4m3(
      float& d0, float& d1, float& d2, float& d3, std::uint32_t a0,
      std::uint32_t a1, std::uint32_t a2, std::uint32_t a3,
      std::uint32_t b0, std::uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }

  __device__ static void accumulate_projection_page_fp8_mma(
      slot_ring::SlotRingState& st, int token, int page, float& d0, float& d1,
      float& d2, float& d3) {
    if (st.warp_id > 7) {
      return;
    }
    const auto* x_fp8 = activation_fp8_ptr(st, token);
    const auto* packed =
        reinterpret_cast<const __nv_fp8_e4m3*>(st.deep_slot_ptr());
    const auto* scales = reinterpret_cast<const naivert::ExpertWeightScale*>(
        packed + (kPackedRows * kPageK));
    const auto* scale_bytes = reinterpret_cast<const unsigned char*>(scales);
    const int branch = st.warp_id >> 2;
    const int k_part = st.warp_id & 3;
    const int lane = st.lane_id;
    const int lane_group = lane >> 2;
    const int lane_quad = lane & 3;
    const int page_offset = page * kPageK;
    const auto* branch_packed =
        packed + branch * kOutputsPerGroup * kPageK;
    if constexpr (naivert::kExpertActivationFp32) {
      // Use the rows8 arithmetic tree. Each branch's four warps own
      // (K half, scale-group parity), preserving each stream across pages.
      const int half = st.warp_id & 1;
      const int parity = (st.warp_id >> 1) & 1;
#pragma unroll
      for (int scale_group = parity; scale_group < 8; scale_group += 2) {
        float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
#pragma unroll
        for (int part = 0; part < 2; ++part) {
          const int block = scale_group * 4 + half + part * 2;
          const int offset = block * 512 + lane_group * 64 + lane_quad * 16;
          const uint4 a = *reinterpret_cast<const uint4*>(branch_packed + offset);
          const int k = page_offset + block * 32 + lane_quad * 4;
          const unsigned b0 = *reinterpret_cast<const unsigned*>(x_fp8 + k);
          const unsigned b1 = *reinterpret_cast<const unsigned*>(x_fp8 + k + 16);
          mma_m16n8k32_e4m3(p0, p1, p2, p3, a.x, a.y, a.z, a.w, b0, b1);
        }
        float scale = naivert::expert_weight_scale(scales[branch * 8 + scale_group]);
        scale *= activation_scales()[page * 8 + scale_group];
        d0 += p0 * scale; d1 += p1 * scale;
        d2 += p2 * scale; d3 += p3 * scale;
      }
      return;
    }
#pragma nounroll
    for (int block = k_part; block < kPageK / 32; block += 4) {
      uint2 a_lo;
      uint2 a_hi;
      const int a_offset = block * 512 + lane_group * 64 + lane_quad * 16;
      a_lo = *reinterpret_cast<const uint2*>(branch_packed + a_offset);
      a_hi = *reinterpret_cast<const uint2*>(branch_packed + a_offset + 8);

      const int k0 = page_offset + block * 32 + lane_quad * 4;
      const int k1 = k0 + 16;
      std::uint32_t b0 =
          *reinterpret_cast<const std::uint32_t*>(x_fp8 + k0);
      std::uint32_t b1 =
          *reinterpret_cast<const std::uint32_t*>(x_fp8 + k1);
      float p0 = 0.0f;
      float p1 = 0.0f;
      float p2 = 0.0f;
      float p3 = 0.0f;
      mma_m16n8k32_e4m3(p0, p1, p2, p3, a_lo.x, a_lo.y, a_hi.x, a_hi.y, b0,
                        b1);
      if constexpr (kBlockScales) {
        // One scale per (row, K block) (MX8: ue8m0): rows lane_group (d0, d1)
        // and lane_group + 8 (d2, d3) of this branch.
        const int scale_base = (branch * (kPageK / 32) + block) * kOutputsPerGroup;
        const float scale_lo = page_block_scale(scale_bytes, scale_base + lane_group);
        const float scale_hi = page_block_scale(scale_bytes, scale_base + lane_group + 8);
        d0 += p0 * scale_lo;
        d1 += p1 * scale_lo;
        d2 += p2 * scale_hi;
        d3 += p3 * scale_hi;
      } else {
        float scale = naivert::expert_weight_scale(scales[branch * 8 + (block >> 2)]);
        if constexpr (naivert::kExpertActivationFp32)
          scale *= activation_scales()[page * 8 + (block >> 2)];
        d0 += p0 * scale;
        d1 += p1 * scale;
        d2 += p2 * scale;
        d3 += p3 * scale;
      }
    }
  }

  __device__ static void finish_projection_from_shared(
      slot_ring::SlotRingState& st, const ExpertSelectUpGateSiluArgs& args) {
    auto* accum = output_accum_ptr(st);
    if (st.warp_id == 0 && st.lane_id < kOutputsPerGroup) {
      float gate = 0.0f;
      float up = 0.0f;
      if constexpr (naivert::kExpertActivationFp32) {
        const int row = st.lane_id;
        gate = (accum[row] + accum[2 * kOutputsPerGroup + row]) +
               (accum[kOutputsPerGroup + row] + accum[3 * kOutputsPerGroup + row]);
        up = (accum[4 * kOutputsPerGroup + row] + accum[6 * kOutputsPerGroup + row]) +
             (accum[5 * kOutputsPerGroup + row] + accum[7 * kOutputsPerGroup + row]);
      } else {
#pragma unroll
        for (int part = 0; part < 4; ++part) {
          gate += accum[part * kOutputsPerGroup + st.lane_id];
          up += accum[(4 + part) * kOutputsPerGroup + st.lane_id];
        }
      }
      float sig = 1.0f / (1.0f + fast_exp2(-gate * 1.4426950408889634f));
      naivert::ExpertActivation value = naivert::expert_activation_from_float(gate * sig * up);
#pragma unroll
      for (int token = 0; token < 4; ++token) {
        if (token < args.seq_len) {
          output_tile_ptr(st, args, token)[st.lane_id] = value;
        }
      }
    }
  }

  __device__ static void accumulate_page_into_shared(
      slot_ring::SlotRingState& st, int page, float d0, float d2) {
    auto* accum = output_accum_ptr(st);
    if (st.warp_id <= 7 && (st.lane_id & 3) == 0) {
      const int row = st.lane_id >> 2;
      float* gate = accum + st.warp_id * kOutputsPerGroup + row;
      float* up = accum + st.warp_id * kOutputsPerGroup + row + 8;
      if (page == 0 || naivert::kExpertActivationFp32) {
        *gate = d0;
        *up = d2;
      } else {
        *gate += d0;
        *up += d2;
      }
    }
  }

  __device__ static void process_projection_page(
      slot_ring::SlotRingState& st, const ExpertSelectUpGateSiluArgs& args, int page) {
    if (page == 0) {
      initialize_projection_state(st, args);
    }
    Slot::consumer_wait(st);
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    if constexpr (naivert::kExpertActivationFp32) {
      if (page != 0) {
        const float* accum = output_accum_ptr(st) + st.warp_id * kOutputsPerGroup;
        d0 = d1 = accum[st.lane_id >> 2];
        d2 = d3 = accum[(st.lane_id >> 2) + 8];
      }
    }
    accumulate_projection_page_fp8_mma(st, 0, page, d0, d1, d2, d3);
    Slot::consumer_release(st);
    accumulate_page_into_shared(st, page, d0, d2);
    if constexpr (naivert::kExpertActivationFp32) __syncwarp();
    if (page + 1 == args.pages) {
      consumer_sync(st);
      finish_projection_from_shared(st, args);
    }
  }
};

}  // namespace naivert_replica_expert_select_up_gate_silu
