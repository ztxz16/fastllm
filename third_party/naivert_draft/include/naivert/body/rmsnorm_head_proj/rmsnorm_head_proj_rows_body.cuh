#pragma once

#include <cuda_bf16.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_runtime.h>

#include <cstdint>

#include <cuda/ptx>

#include "naivert/body/execute_top_p/sampler_mailbox.hpp"
#include "naivert/body/rmsnorm_head_proj/rmsnorm_head_proj_args.hpp"
#include "naivert/body/rmsnorm_head_proj/rmsnorm_head_proj_traits.cuh"
#include "naivert/pdl/grid_dependency.cuh"
#include "naivert/slot_ring/protocol.cuh"
#include "naivert/body/attn_common/rows_gemv_mma.cuh"

namespace naivert_replica_rmsnorm_head_proj {

// Multi-row head: hidden-only panes and packed global BF16 weights.
// Eight warps split each K1024 page into K128 slices; partials and logits are FP32.
template <int kRows_>
struct RMSNormHeadProjRowsBody {
  using Trans64 = naivert::Barrier;

  static constexpr int kRows = kRows_;
  static constexpr int kRmsProducerCtas = 32;
  static constexpr int kRmsProducerGroups = kRmsProducerCtas / kRows;
  static constexpr int kConsumerThreads = 256;
  static constexpr int kHelperBaseThread = 256;
  static constexpr int kHelperWarps = 4;
  static constexpr int kBarrierOffset = 0x0000;
  static constexpr int kReleaseBarrierOffset = 0x0040;
  static constexpr int kRmsPartialOffset = 0x0100;
  static constexpr int kInvRmsOffset = 0x0180;
  static constexpr int kCtaWinnerOffset = 0x0200;
  static constexpr int kMmaPartialOffset = 0x1000;
  static constexpr int kPaneBaseOffset = 0x2000;
  static constexpr int kPaneHiddenBytes =
      kRows * kWeightPage * int(sizeof(__nv_bfloat16));
  static constexpr int kPaneStride = kPaneHiddenBytes;
  static constexpr int kStageBytes = kPaneStride;
  static constexpr int kSharedBytesUsed = kPaneBaseOffset + kSlotDepth * kPaneStride;
  static constexpr int kWorkspaceRowStride = 64;
  static constexpr int kSamplerInboxRowStride = 128;

  static_assert(kRows == 8, "head projection uses eight rows");
  static_assert(kRmsProducerGroups * kRows == kRmsProducerCtas);
  static_assert(kOutputsPerTile == 16, "tensor head uses M16 tiles");
  static_assert(kCtaWinnerOffset + kRows * (kConsumerThreads / 32) * 8 <= kPaneBaseOffset);
  static_assert(kSharedBytesUsed <= kRows8SharedBytes,
                "head panes must fit the variant's shared memory");
  static_assert(kMmaPartialOffset + 8 * 128 * int(sizeof(float)) <= kPaneBaseOffset,
                "MMA split partials must not overlap the hidden panes");

  __device__ static __forceinline__ std::uint64_t* barriers(
      unsigned char* shared) {
    return reinterpret_cast<std::uint64_t*>(
        shared + kBarrierOffset);
  }

  __device__ static __forceinline__ std::uint64_t* release_barriers(
      unsigned char* shared) {
    return reinterpret_cast<std::uint64_t*>(
        shared + kReleaseBarrierOffset);
  }

  __device__ static __forceinline__ float* rms_partials(
      unsigned char* shared) {
    return reinterpret_cast<float*>(shared + kRmsPartialOffset);
  }

  __device__ static __forceinline__ float* inv_rms(
      unsigned char* shared) {
    return reinterpret_cast<float*>(shared + kInvRmsOffset);
  }

  __device__ static __forceinline__ unsigned long long* cta_winners(
      unsigned char* shared) {
    return reinterpret_cast<unsigned long long*>(
        shared + kCtaWinnerOffset);
  }

  __device__ static __forceinline__ std::uint8_t* row_workspace(
      const RMSNormHeadProjArgs& args, int row) {
    return args.workspace_base + row * kWorkspaceRowStride;
  }

  __device__ static __forceinline__ unsigned char* pane(
      unsigned char* shared, int slot) {
    return shared + kPaneBaseOffset + slot * kPaneStride;
  }

  __device__ static __forceinline__ void consumer_sync() {
    naivert::consumer_group_named_sync();
  }

  __device__ static __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
      value += __shfl_down_sync(0xffffffffu, value, delta);
    }
    return value;
  }

  __device__ static __forceinline__ float fast_rsqrt(float value) {
    float out;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(value));
    return out;
  }

  __device__ static __forceinline__ unsigned ordered_float_key(
      float value) {
    const unsigned bits = __float_as_uint(value);
    const unsigned flip =
        (static_cast<unsigned>(static_cast<int>(bits) >> 31) |
         0x80000000u);
    return bits ^ flip;
  }

  __device__ static __forceinline__ unsigned long long packed_candidate(
      float score, int local_index) {
    return
        (static_cast<unsigned long long>(ordered_float_key(score)) << 32) |
        static_cast<unsigned>(
            0xffffffffu - static_cast<unsigned>(local_index));
  }

  __device__ static __forceinline__ __nv_bfloat162 bf16x2_from_bits(
      std::uint32_t bits) {
    return *reinterpret_cast<const __nv_bfloat162*>(&bits);
  }

  __device__ static __forceinline__ void cp_async_bulk_evict_first(
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

  template <bool kWaitForPrimary>
  __device__ static void compute_inv_rms_and_hidden(
      const RMSNormHeadProjArgs& args, unsigned char* shared) {
    const int tid = int(threadIdx.x);
    if (tid >= kConsumerThreads) {
      return;
    }
    // Helpers stream the input panes; exp_out must be ready before normalization.
    naivert::pdl::wait_for_primary<kWaitForPrimary>();
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int producer_row = int(blockIdx.x) & (kRows - 1);
    const int producer_group = int(blockIdx.x) / kRows;
    if (int(blockIdx.x) < kRmsProducerCtas) {
      float sum = 0.0f;
      for (int k = tid; k < kHidden; k += kConsumerThreads) {
        const float x =
            __bfloat162float(
                args.exp_out[producer_row * kHidden + k]);
        sum = fmaf(x, x, sum);
      }
      sum = warp_sum(sum);
      if (lane == 0) {
        rms_partials(shared)[warp] = sum;
      }
      consumer_sync();
      if (tid == 0) {
        float total = 0.0f;
#pragma unroll
        for (int source_warp = 0; source_warp < 8; ++source_warp) {
          total += rms_partials(shared)[source_warp];
        }
        inv_rms(shared)[0] =
            fast_rsqrt(
                total / static_cast<float>(kHidden) + naivert::geometry::kNormEps);
      }
      consumer_sync();
      for (int k = producer_group * kConsumerThreads + tid;
           k < kHidden;
           k += kRmsProducerGroups * kConsumerThreads) {
        const float gamma = args.final_norm[k];
        const float x = __bfloat162float(
            args.exp_out[producer_row * kHidden + k]);
        args.last_hidden_states[
            producer_row * kHidden + k] =
            __float2bfloat16(x * gamma * inv_rms(shared)[0]);
      }
      consumer_sync();
    }
    auto* counters =
        reinterpret_cast<unsigned int*>(args.workspace_base);
    if (int(blockIdx.x) < kRmsProducerCtas && tid == 0) {
      __threadfence();
      atomicAdd(counters, 1u);
    }
    if (tid == 0) {
      const auto* ready =
          reinterpret_cast<const volatile unsigned int*>(counters);
      while (*ready != unsigned(kRmsProducerCtas)) {
      }
      __threadfence();
    }
    consumer_sync();
  }

  __device__ static void helper_pages(
      const RMSNormHeadProjArgs& args, unsigned char* shared) {
    const int tid = int(threadIdx.x);
    if (tid < kHelperBaseThread) {
      return;
    }
    const int helper = (tid - kHelperBaseThread) >> 5;
    const int lane = tid & 31;
    if (helper >= kHelperWarps) {
      return;
    }
    std::uint64_t* barrier = barriers(shared) + helper;
    for (int stage = helper;; stage += kHelperWarps) {
      const int wave = stage / kWeightPages;
      const int page = stage - wave * kWeightPages;
      const int output_tile = int(blockIdx.x) + wave * kGridCtas;
      if (output_tile >= kOutputTiles) {
        break;
      }
      const int sequence = stage >> 2;
      while (!Trans64::try_wait(
          release_barriers(shared) + helper,
          std::uint32_t(sequence & 1))) {
      }
      unsigned char* destination = pane(shared, helper);
      if (lane == 0) {
        const auto* source = args.lm_head +
            (output_tile * kWeightPages + page) * kOutputsPerTile * kWeightPage;
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                     :: "l"(source), "r"(std::uint32_t(kWeightPaneBytes)));
      }
      if (stage == helper) {
        const auto* ready =
            reinterpret_cast<const volatile unsigned int*>(
                args.workspace_base);
        while (*ready != unsigned(kRmsProducerCtas)) {
        }
        __threadfence();
        __syncwarp();
      }
      auto* hidden_destination =
          reinterpret_cast<uint4*>(destination);
#pragma unroll
      for (int row = 0; row < kRows; ++row) {
        const auto* hidden_source =
            reinterpret_cast<const uint4*>(
                args.last_hidden_states +
                row * kHidden + page * kWeightPage);
#pragma unroll
        for (int chunk = lane;
             chunk < kHiddenPageBytes / int(sizeof(uint4));
             chunk += 32) {
          __pipeline_memcpy_async(
              hidden_destination +
                  row * (kHiddenPageBytes / int(sizeof(uint4))) +
                  chunk,
              hidden_source + chunk,
              int(sizeof(uint4)));
        }
      }
      __pipeline_commit();
      // Join every lane's hidden copies to full; weights are read directly.
      // Close registration only after every lane has attached its copies.
      cutlass::arch::cpasync_barrier_arrive(barrier);
      __syncwarp();
      if (lane == 0) {
        Trans64::arrive(barrier);
      }
    }
  }

  __device__ static __forceinline__ void wait_stage(
      unsigned char* shared, int stage) {
    const int slot = stage & 3;
    const int sequence = stage >> 2;
    while (!Trans64::try_wait(
        barriers(shared) + slot,
        std::uint32_t(sequence & 1))) {
    }
  }

  __device__ static __forceinline__ void release_stage(
      unsigned char* shared, int stage) {
    // Each reader releases its own shared-memory accesses, as in the
    // protocol's consumer_release. The producer waits for all 256 readers.
    Trans64::arrive(release_barriers(shared) + (stage & 3));
  }

  // Reuse the BF16 MMA primitive with packed global weight pages and four
  // hidden-only panes. Eight warps split each K1024 page into K128 slices.
  // Short K32 FP32 partials feed a balanced reduction across chunks/splits.
  // The hidden panes retain the consumer wait/release protocol.
  __device__ static void project_mma(
      const RMSNormHeadProjArgs& args, unsigned char* shared) {
    const int tid = int(threadIdx.x);
    if (tid >= kConsumerThreads) return;
    const int warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, q = lane & 3;
    auto* partials = reinterpret_cast<float*>(shared + kMmaPartialOffset);
    for (int wave = 0;; ++wave) {
      const int tile = int(blockIdx.x) + wave * kGridCtas;
      if (tile >= kOutputTiles) break;
      float chunks[4][4] = {};
#pragma unroll 1
      for (int page = 0; page < kWeightPages; ++page) {
        const int stage = wave * kWeightPages + page;
        wait_stage(shared, stage);
        const auto* hidden = reinterpret_cast<const __nv_bfloat16*>(pane(shared, stage & 3));
        const auto* weight = args.lm_head +
            (tile * kWeightPages + page) * kOutputsPerTile * kWeightPage;
#pragma unroll
        for (int chunk = 0; chunk < 4; ++chunk) {
          const int k = warp * 128 + chunk * 32 + q * 8;
          const uint4 a0 = *reinterpret_cast<const uint4*>(weight + g * kWeightPage + k);
          const uint4 a1 = *reinterpret_cast<const uint4*>(weight + (g + 8) * kWeightPage + k);
          const uint4 b = *reinterpret_cast<const uint4*>(hidden + g * kWeightPage + k);
          float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
          naivert::attn_rows::mma_bf16_16816(c, a0.x, a1.x, a0.y, a1.y, b.x, b.y);
          naivert::attn_rows::mma_bf16_16816(c, a0.z, a1.z, a0.w, a1.w, b.z, b.w);
#pragma unroll
          for (int i = 0; i < 4; ++i) chunks[chunk][i] += c[i];
        }
        release_stage(shared, stage);
      }
      float acc[4];
#pragma unroll
      for (int i = 0; i < 4; ++i)
        acc[i] = (chunks[0][i] + chunks[1][i]) + (chunks[2][i] + chunks[3][i]);
      float* mine = partials + warp * 128;
      mine[g * 8 + 2 * q] = acc[0];
      mine[g * 8 + 2 * q + 1] = acc[1];
      mine[(g + 8) * 8 + 2 * q] = acc[2];
      mine[(g + 8) * 8 + 2 * q + 1] = acc[3];
      consumer_sync();
      float sum = 0.0f;
      if (tid < 128) {
        const float a = partials[tid] + partials[128 + tid];
        const float b = partials[256 + tid] + partials[384 + tid];
        const float c = partials[512 + tid] + partials[640 + tid];
        const float d = partials[768 + tid] + partials[896 + tid];
        sum = (a + b) + (c + d);
        args.logits_out[(tid % 8) * kVocabShard + tile * 16 + tid / 8] = sum;
      }
      consumer_sync();
      if (tid < 64) {
        const int row = tid % 8, output = tid / 8;
        const int out0 = tile * 16 + output;
        const float sum1 = args.logits_out[row * kVocabShard + out0 + 8];
        const auto a = packed_candidate(sum, out0);
        const auto b = packed_candidate(sum1, out0 + 8);
        auto* winner = cta_winners(shared) + row * 8 + output;
        const auto best = a > b ? a : b;
        *winner = best > *winner ? best : *winner;
      }
      // Complete every partial reader before the next wave reuses scratch.
      consumer_sync();
    }
  }

  __device__ static __forceinline__ unsigned long long* shard_top1_for(
      std::uint8_t* workspace) {
    return reinterpret_cast<unsigned long long*>(
        workspace + sizeof(unsigned int) * 2);
  }

  // One warp publishes four rows (8 destinations each); rows8 calls it twice.
  __device__ static void publish_top1_inbox_warp(
      const RMSNormHeadProjArgs& args, int row_base) {
    using SamplerPeerState =
        naivert_replica_execute_top_p::SamplerPeerState;
    constexpr int kTpRanks =
        naivert_replica_execute_top_p::kSamplerTpRanks;
    if (args.tp_rank < 0 || args.tp_rank >= kTpRanks) {
      return;
    }
    const int lane = int(threadIdx.x);
    const int row = row_base + lane / kTpRanks;
    const int destination = lane - (lane / kTpRanks) * kTpRanks;
    const unsigned valid_peers = __ballot_sync(
        0xffffffffu,
        args.sampler_inbox_peers[destination] != nullptr);
    if (valid_peers != 0xffffffffu) {
      return;
    }
    auto* workspace = row_workspace(args, row);
    auto* counters = reinterpret_cast<unsigned int*>(workspace);
    unsigned generation = 0u;
    if (destination == 0) {
      generation = atomicAdd(counters + 5, 1u) + 1u;
      if (generation == 0u) {
        generation = 1u;
      }
    }
    generation = __shfl_sync(
        0xffffffffu, generation, (lane / kTpRanks) * kTpRanks);
    const unsigned long long packed = *shard_top1_for(workspace);
    auto* inbox = reinterpret_cast<SamplerPeerState*>(
        const_cast<std::uint8_t*>(
            args.sampler_inbox_peers[destination] +
            row * kSamplerInboxRowStride));
    inbox->top1_inbox_packed[args.tp_rank] = packed;
    __threadfence_system();
    atomicExch_system(
        &inbox->top1_inbox_generation[args.tp_rank], generation);
  }

  __device__ static void finalize(
      const RMSNormHeadProjArgs& args, unsigned char* shared) {
    const int tid = int(threadIdx.x);
    if (tid >= kConsumerThreads) {
      return;
    }
    consumer_sync();
    auto* counters =
        reinterpret_cast<unsigned int*>(args.workspace_base);
    auto* winners = cta_winners(shared);
    // Eight lanes per row fold the warp winners; rows 4..7 (rows8) live in warp 1.
    if (tid < kRows * 8) {
      const int row = tid >> 3;
      const int row_lane = tid & 7;
      unsigned long long cta_best =
          winners[row * (kConsumerThreads / 32) + row_lane];
#pragma unroll
      for (int delta = 4; delta > 0; delta >>= 1) {
        const unsigned long long candidate = __shfl_down_sync(
            0xffffffffu, cta_best, delta, 8);
        if (candidate > cta_best) {
          cta_best = candidate;
        }
      }
      if (row_lane == 0) {
        atomicMax(
            shard_top1_for(row_workspace(args, row)),
            cta_best);
      }
    }
    if constexpr (kRows > 4) {
      consumer_sync();
    }
    if (tid >= 32) {
      return;
    }
    __syncwarp();
    unsigned is_last = 0u;
    if (tid == 0) {
      __threadfence();
      is_last =
          atomicAdd(counters + 1, 1u) == kGridCtas - 1;
    }
    is_last = __shfl_sync(0xffffffffu, is_last, 0);
    if (is_last == 0u) {
      return;
    }
#pragma unroll
    for (int row_base = 0; row_base < kRows; row_base += 4) {
      publish_top1_inbox_warp(args, row_base);
    }
    __syncwarp();
    if (tid < kRows) {
      *shard_top1_for(row_workspace(args, tid)) = 0ull;
    }
    if (tid == 0) {
      counters[0] = 0u;
      counters[1] = 0u;
      __threadfence();
    }
  }

  template <bool kWaitForPrimary = false>
  __device__ static void run(
      const RMSNormHeadProjArgs& args, unsigned char* shared) {
    if (args.exp_out == nullptr || args.final_norm == nullptr ||
        args.lm_head == nullptr || args.logits_out == nullptr ||
        args.last_hidden_states == nullptr ||
        args.workspace_base == nullptr || args.dim != kHidden ||
        args.vocab != kVocabShard ||
        int(blockIdx.x) >= kGridCtas) {
      return;
    }
    const int tid = int(threadIdx.x);
    if (tid < kSlotDepth) {
      Trans64::init(barriers(shared) + tid, 1u);
      Trans64::init(
          release_barriers(shared) + tid, kConsumerThreads);
      naivert::barrier_arrive_count(
          release_barriers(shared) + tid, kConsumerThreads);
    }
    if (tid < kRows * (kConsumerThreads / 32)) {
      cta_winners(shared)[tid] = 0ull;
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
    __syncthreads();

    if (tid >= kHelperBaseThread) {
      helper_pages(args, shared);
    } else {
      compute_inv_rms_and_hidden<kWaitForPrimary>(args, shared);
      project_mma(args, shared);
    }
    finalize(args, shared);
  }
};

using RMSNormHeadProjRows8Body = RMSNormHeadProjRowsBody<8>;

}  // namespace naivert_replica_rmsnorm_head_proj
