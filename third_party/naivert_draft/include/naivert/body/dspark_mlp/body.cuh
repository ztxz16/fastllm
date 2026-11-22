#pragma once

// Modified for FastLLM fused draft execution. See the parent vendor NOTICE
// and UPSTREAM.json for adaptation details and original source hashes.

#include "naivert/body/dspark_mlp/args.hpp"
#include "naivert/body/attn_common/rows_gemv_mma.cuh"
#include "naivert/body/rmsnorm_head_proj/rmsnorm_head_proj_rows_body.cuh"
#include "naivert/common/prefetched_rank_sum.cuh"
#include "naivert/pdl/grid_dependency.cuh"
#include <cuda/atomic>
#include <cuda/ptx>

namespace fastllm_draft_ops {
inline constexpr bool kExportMlpDiagnostics = NAIVERT_DSPARK_DIAGNOSTICS != 0;

// Adapted D4 with bounded request tiles. Each request retains its BF16
// boundaries, rank-ordered fold, mailbox layout and generation protocol.
constexpr int kMlpWindow = MlpGeometry::Rows;
constexpr int kMlpHidden = MlpGeometry::Hidden;
constexpr int kMlpInterR = MlpGeometry::IntermediatePerRank;
constexpr int kMlpCtas = 128;
constexpr int kMlpInputStride = kMlpHidden;
constexpr int kMlpInputRotation = 32;
static_assert((kMlpHidden & (kMlpHidden - 1)) == 0 && kMlpInputRotation % 8 == 0);
__host__ __device__ constexpr int mlp_input_index(int row, int column) {
  return row * kMlpHidden + ((column + row * kMlpInputRotation) & (kMlpHidden - 1));
}
constexpr int kMlpDownStride = kMlpInterR + 32;
constexpr int kMlpThreads = 320;
constexpr int kMlpWarps = kMlpThreads / 32;
constexpr int kColsPerCtaD4 = kMlpHidden / kMlpCtas;
constexpr int kPairsD4 = kColsPerCtaD4 / 4;
constexpr int kDownStageStride = kColsPerCtaD4 + 4;
constexpr unsigned kSaltD4Base = 4;
constexpr int kGuSlots = 2;
constexpr unsigned kArrivalUnit = 1u << kMlpWindow;
constexpr unsigned kNormReadyMask = kArrivalUnit - 1u;
static_assert(kMlpWindow == 8 && kMlpCtas < (1u << (32 - kMlpWindow)));
constexpr int kDownWeightBytes = kColsPerCtaD4 * kMlpInterR * 2;
constexpr int kGammaBytes = kMlpHidden * sizeof(float);
constexpr unsigned kWeightWatchdogSite = 0xD4u;
using WeightTransport = naivert_replica_rmsnorm_head_proj::RMSNormHeadProjRows8Body;
static_assert(2*kMlpInterR/16 <= kMlpCtas);
static_assert(naivert::kMailboxRanks == MlpGeometry::Ranks);
static_assert(kMlpCtas < NAIVERT_SM_COUNT);
static_assert(kColsPerCtaD4 == 32);
static_assert(kMlpCtas * naivert::kMailboxWarpStride <= naivert::kMailboxPhaseStride);
static_assert(kColsPerCtaD4 / 2 * sizeof(std::uint64_t) <= naivert::kMailboxWarpStride);
struct MlpLayout {
  __host__ __device__ static constexpr std::size_t tile_offset(int task, int qword) {
    return std::size_t(task) * naivert::kMailboxWarpStride +
           std::size_t(qword) * sizeof(unsigned long long);
  }
};

// Physical gate/up rows are interleaved, then packed into K32 pages.
using PackedMlpRow = naivert::attn_rows::K32PackedRow;


// The shared helper owns the arrival/generation synchronization protocol.
__device__ inline void mlp_grid_barrier(unsigned* state) {
  __syncthreads();
  if (threadIdx.x == 0) {
    cuda::atomic_ref<unsigned, cuda::thread_scope_device> arrivals(state[0]);
    cuda::atomic_ref<unsigned, cuda::thread_scope_device> generation(state[1]);
    const unsigned gen = generation.load(cuda::memory_order_relaxed);
    const unsigned old = arrivals.fetch_add(kArrivalUnit, cuda::memory_order_acq_rel);
    if ((old >> kMlpWindow) == unsigned(gridDim.x) - 1u) {
      arrivals.store(0u, cuda::memory_order_relaxed);
      generation.fetch_add(1u, cuda::memory_order_release);
    } else {
      while (generation.load(cuda::memory_order_relaxed) == gen) {}
    }
    cuda::ptx::fence(cuda::ptx::sem_acquire, cuda::ptx::scope_gpu);
  }
  __syncthreads();
}

__device__ inline float mlp_bf16_at(const __nv_bfloat16* v, int i) {
  return __bfloat162float(v[i]);
}

__device__ bool wait_weight(std::uint64_t* barrier, unsigned phase,
                         const naivert::MailboxWatchdog& watchdog) {
  if (naivert::Barrier::try_wait(barrier, phase)) return true;
  const bool bounded = watchdog.limit_ns != 0ull;
  const auto start = bounded ? naivert::mailbox_globaltimer_ns() : 0ull;
  unsigned spins = 0;
  for (;;) {
    if (naivert::Barrier::try_wait(barrier, phase)) return true;
    if (bounded && ((++spins & naivert::kMailboxWatchdogSpinMask) == 0u) &&
        naivert::mailbox_globaltimer_ns() - start > watchdog.limit_ns) {
      naivert::mailbox_watchdog_trip(watchdog, kWeightWatchdogSite);
      // The caller must stop advancing this pane. Arriving on an incomplete
      // phase again can underflow its arrival count and raise a GPU exception.
      return false;
    }
  }
}

// Each launch handles a bounded row tile. Requests keep independent scratch,
// epochs and mailboxes; larger batches iterate these tiles on the host.
template<int Chunks = 1> struct MlpRowsBody {
  static_assert(Chunks == 1 || Chunks == 2 || Chunks == 4);
  static constexpr int kMlpWindow = 8 * Chunks;
  static constexpr int kDownAccumFloats = 8 * kColsPerCtaD4 * kMlpWindow;
  static constexpr int kMlpBaseSmem = Chunks == 1 ?
      kMlpWindow * kMlpInputStride * 2 + (kMlpWindow * 16 + kMlpWarps * 128 * Chunks) * 4 :
      kMlpWindow * kMlpDownStride * 2 + (kDownAccumFloats + kMlpWindow * kDownStageStride) * 4;
  // Batched normalization reuses resid_out, leaving room for the same two
  // 2048-K weight pages as the scalar kernel.
  static constexpr int kGuPageK = 2048;
  static constexpr int kGuPages = kMlpHidden / kGuPageK;
  static constexpr int kGuPageBytes = 16 * kGuPageK * 2;
  static constexpr int kWeightBytes = kGuSlots * kGuPageBytes;
  static constexpr int kReadyOffset = kMlpBaseSmem;
  static constexpr int kEmptyOffset = kReadyOffset + kGuSlots * sizeof(std::uint64_t);
  static constexpr int kFailedOffset = kEmptyOffset + kGuSlots * sizeof(std::uint64_t);
  static constexpr int kWeightOffset = (kFailedOffset + sizeof(unsigned) + 127) & ~127;
  static constexpr int kDownReadyOffset = (kFailedOffset + sizeof(unsigned) + 7) & ~7;
  static constexpr int kGammaOffset = kWeightOffset + kWeightBytes;
  static constexpr int kGammaReadyOffset = kGammaOffset + kGammaBytes;
  static constexpr int kResidualOffset = (kGammaReadyOffset + sizeof(std::uint64_t) + 15) & ~15;
  static constexpr int kMlpSmem = (kResidualOffset + kMlpWindow * kColsPerCtaD4 * 2 + 127) & ~127;
  static_assert(kMlpSmem <= 232448);
  static_assert(kDownReadyOffset + sizeof(std::uint64_t) <= kWeightOffset);
  static_assert(kDownWeightBytes <= kWeightBytes);
  static_assert(kMlpWindow * kMlpDownStride * 2 +
      (kDownAccumFloats + kMlpWindow * kDownStageStride) * 4 <= kMlpWindow * kMlpInputStride * 2);
#include "naivert/body/dspark_mlp/pipeline.cuh"

#include "naivert/body/attn_common/prefetch_l2_bulk.cuh"
  static constexpr int kThreads = kMlpThreads;
  static constexpr int kCtas = kMlpCtas;
  static constexpr int kSharedBytes = kMlpSmem;

  __device__ static void run(const MlpArgs& args, unsigned char* memory, const MlpArgs* second = nullptr) {
    if (threadIdx.x == 0) {
      for (int slot = 0; slot < kGuSlots; ++slot) {
        naivert::Barrier::init(reinterpret_cast<std::uint64_t*>(memory+kReadyOffset) + slot,1u);
        naivert::Barrier::init(reinterpret_cast<std::uint64_t*>(memory+kEmptyOffset) + slot,8u);
      }
      naivert::Barrier::init(reinterpret_cast<std::uint64_t*>(memory+kDownReadyOffset),1u);
      naivert::Barrier::init(reinterpret_cast<std::uint64_t*>(memory+kGammaReadyOffset),1u);
      *reinterpret_cast<unsigned*>(memory+kFailedOffset)=0u;
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    // Join barrier initialization before any asynchronous producer starts.
    __syncthreads();

    auto* sx = reinterpret_cast<__nv_bfloat16*>(memory);
    auto* sh = sx;
    auto* down_partials = reinterpret_cast<float*>(sx + kMlpWindow * kMlpDownStride);
    auto* stage = down_partials + kDownAccumFloats;
    auto* gu = reinterpret_cast<float*>(sx + (Chunks == 1 ? kMlpWindow * kMlpInputStride : 0));
    auto* partials = gu + kMlpWindow * 16;
    const int tid = threadIdx.x, cta = blockIdx.x;
    const int col0 = cta * kColsPerCtaD4;
    const bool has_gate = cta < 2*kMlpInterR/16;
    auto* ready = reinterpret_cast<std::uint64_t*>(memory+kReadyOffset);
    auto* failed = reinterpret_cast<unsigned*>(memory+kFailedOffset);
    auto* weight = reinterpret_cast<__nv_bfloat16*>(memory+kWeightOffset);
    auto* residual = reinterpret_cast<__nv_bfloat16*>(memory + kResidualOffset);
    // Programmatic dependent launch: warm this CTA's weights (gate/up tile, down tile, gamma)
    // into L2 while the previous D3 drains, then wait for its completion before the first
    // hid_a read. Cache hints only; the bulk copies below are unchanged. No-ops without an edge.
    if (tid == 0) {
      if (has_gate) {
        const int row = cta * PackedMlpRow::Rows;
        const auto* source = (row < kMlpInterR ? args.gate_w : args.up_w) +
            std::size_t(row % kMlpInterR) * kMlpHidden;
        prefetch_l2_bulk(source, kGuPages * kGuPageBytes);
        prefetch_l2_bulk(args.post_ln, kGammaBytes);
      }
      prefetch_l2_bulk(args.down_w + std::size_t(col0) * kMlpInterR, kDownWeightBytes);
    }
    naivert::pdl::wait_for_primary<true>();
    naivert::pdl::trigger_dependents<true>();
    if (tid < kMlpWindow * kColsPerCtaD4 / 8) {
      const int row = tid / (kColsPerCtaD4 / 8), vector = tid % (kColsPerCtaD4 / 8);
      __pipeline_memcpy_async(reinterpret_cast<uint4*>(residual) + tid,
          reinterpret_cast<const uint4*>((row < 8 ? args : second[row/8-1]).hid_a + (row % 8) * kMlpHidden + col0) + vector, 16);
      __pipeline_commit();
    }
    auto* gamma = reinterpret_cast<float*>(memory + kGammaOffset);
    auto* gamma_ready = reinterpret_cast<std::uint64_t*>(memory + kGammaReadyOffset);
    if (tid == 288 && (Chunks == 1 ? has_gate : cta < kMlpWindow)) {
      naivert::Barrier::expect_transaction(gamma_ready,kGammaBytes);
      WeightTransport::cp_async_bulk_evict_first(gamma,args.post_ln,kGammaBytes,gamma_ready);
      naivert::Barrier::arrive(gamma_ready);
    }
    const naivert::MailboxWatchdog watchdog{args.watchdog_limit_ns,args.watchdog_flag};
    if constexpr (Chunks > 1) {
      // hid_a and resid_out are distinct prepared lane buffers.
      // One CTA owns each normalized row. Reuse the eventual output: GU has
      // retired every read before the existing post-SiLU grid join releases down.
      if (cta < kMlpWindow && tid < 256)
        mlp_normalize_row(cta < 8 ? args : second[cta/8-1], cta % 8, gamma, gamma_ready, partials);
    }
    if (has_gate) {
      if constexpr (Chunks == 1) {
        if (tid < 256) mlp_normalize_local(args, gamma, sx, gamma_ready);
      }
      auto* empty = reinterpret_cast<std::uint64_t*>(memory+kEmptyOffset);
      mlp_gate_pages(args, sx, weight, ready, empty, failed, partials, gu, second);
    }
    auto* down_ready = reinterpret_cast<std::uint64_t*>(memory+kDownReadyOffset);
    // Successful GU consumption has crossed the helper's final CTA barrier.
    // A failed GU DMA must never have its destination reused.
    if (tid == 288 && !*failed) {
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      naivert::Barrier::expect_transaction(down_ready,kDownWeightBytes);
      WeightTransport::cp_async_bulk_evict_first(weight,args.down_w + std::size_t(col0)*kMlpInterR,
                                                kDownWeightBytes,down_ready);
      naivert::Barrier::arrive(down_ready);
    }
    if (cta < 2 * kMlpInterR / 16 && tid < kMlpWindow * 8) {
      const int row = tid / 8, col = tid % 8;
      const float g = gu[row * 16 + col * 2];
      const float u = gu[row * 16 + col * 2 + 1];
      float value = float(__float2bfloat16_rn(g / (1.0f + expf(-g)))) * u;
      (row < 8 ? args : second[row/8-1]).mlp_h[std::size_t(row % 8) * kMlpInterR + cta * 8 + col] = __float2bfloat16_rn(value);
    }
    mlp_grid_barrier(args.barrier);
    if constexpr (Chunks > 1) {
      // The grid join retired all normalized-output readers and reset the first
      // request's arrival word. Clear every remaining lane before any next launch.
      if (cta == 0 && tid < Chunks-1)
        cuda::atomic_ref<unsigned, cuda::thread_scope_device>(second[tid].barrier[0]).store(0u, cuda::memory_order_release);
    }
    // A failed GU DMA retains weight-arena ownership. The ended GU/partials
    // arena is disjoint and can hold fallback output.
    if (*failed) stage = gu;
    // Phase two reuses the input/GU arena after all GU consumers have joined.
    if (!*failed) for (int i = tid; i < kMlpWindow * kMlpInterR / 8; i += kMlpThreads) {
      reinterpret_cast<uint4*>(sh + (i / (kMlpInterR / 8)) * kMlpDownStride)[i % (kMlpInterR / 8)] =
          reinterpret_cast<const uint4*>((i < kMlpInterR ? args : second[i/kMlpInterR-1]).mlp_h)[i % (8 * kMlpInterR / 8)];
    }
    if (tid == 0 && !*failed && !wait_weight(down_ready,0u,watchdog)) *failed=1u;
    __syncthreads();
    if (*failed) {
      for (int i = tid; i < kMlpWindow*kColsPerCtaD4; i += kMlpThreads) stage[(i/kColsPerCtaD4)*kDownStageStride+i%kColsPerCtaD4]=0.0f;
      __syncthreads();
    } else {
      mlp_down_tile(sh, weight, down_partials, stage);
    }
    if (tid < kMlpWindow * kColsPerCtaD4 / 8) __pipeline_wait_prior(0);
    __syncthreads();
    // --- publish this rank's partial (bf16, exactly what the stand-in's `down_partial` holds)
    for (int item = tid; item < kMlpWindow * kColsPerCtaD4; item += kMlpThreads) {
      const int m = item / kColsPerCtaD4, col = item % kColsPerCtaD4;
      const __nv_bfloat16 v = __float2bfloat16_rn(stage[m * kDownStageStride + col]);
      if constexpr (kExportMlpDiagnostics)
        if (args.watchdog_flag == nullptr ||
            !(*reinterpret_cast<volatile unsigned*>(args.watchdog_flag) & naivert::kMailboxWatchdogTripped))
          (m < 8 ? args : second[m/8-1]).down_partial[std::size_t(m % 8) * kMlpHidden + col0 + col] = v;
      stage[m * kDownStageStride + col] = __bfloat162float(v);   // the reduce sums the ROUNDED partials
    }
    __syncthreads();

    // Publish every request before waiting for any TP result. Each request
    // keeps its original mailbox layout, generation and rank accumulation order.
    if constexpr (Chunks > 1) {
#pragma unroll
      for (int chunk = 0; chunk < Chunks; ++chunk) {
        const auto &request = chunk == 0 ? args : second[chunk-1];
        if (tid / 32 < 8 && tid % 32 < kPairsD4) {
          const int row = tid / 32, pair = tid % 32;
          const auto mailbox = naivert::TileMailbox<MlpLayout>::make(request.mailbox_peers, request.rank,
              naivert::mailbox_generation(static_cast<unsigned>(*request.replay_generation), request.generation_salt),
              std::size_t(row) * naivert::kMailboxSlabStride);
          const float* v = stage + (chunk * 8 + row) * kDownStageStride + pair * 4;
          mailbox.publish_pair(cta, pair * 2, naivert::pack_bf16x2(v[0],v[1]), naivert::pack_bf16x2(v[2],v[3]));
        }
      }
    }
    // Fold each request in the same order on every rank. Mailbox slabs
    // remain eight rows per request, independent of the number of row tiles.
#pragma unroll
    for (int chunk = 0; chunk < Chunks; ++chunk) {
      const auto &request = chunk == 0 ? args : second[chunk-1];
      const unsigned gen = naivert::mailbox_generation(
          static_cast<unsigned>(*request.replay_generation), request.generation_salt);

      // One consumer warp owns each row; active lanes fold tagged qword pairs.
      if (tid / 32 < 8 && tid % 32 < kPairsD4) {
        const int m = tid / 32;
        const int pair = tid % 32;
        const int qword = pair * 2;
        const auto mailbox = naivert::TileMailbox<MlpLayout>::make(
            request.mailbox_peers, request.rank, gen, std::size_t(m) * naivert::kMailboxSlabStride);
        const float* v = stage + (chunk * 8 + m) * kDownStageStride + pair * 4;
        const std::uint32_t p01 = naivert::pack_bf16x2(v[0], v[1]);
        const std::uint32_t p23 = naivert::pack_bf16x2(v[2], v[3]);
        if constexpr (Chunks == 1) mailbox.publish_pair(cta, qword, p01, p23);

        const naivert::MailboxWatchdog watchdog{request.watchdog_limit_ns, request.watchdog_flag};
        naivert::Bf16x4Acc red = naivert::Bf16x4Acc::zero();
        naivert::accumulate_prefetched_rank_order(red, mailbox, p01, p23, cta, qword, [&](int slot) {
          return mailbox.wait_pair_bounded<true>(slot, cta, qword, watchdog,
                                                 request.generation_salt);
        });

        // Round once, then add the residual and round again -- the stand-in stores the reduced sum
        // into a bf16 `resid[L+1]` and only then does `add_(hid_a)`, so both roundings are real.
        const float r[4] = {red.v0, red.v1, red.v2, red.v3};
    #pragma unroll
        for (int i = 0; i < 4; ++i) {
          const std::size_t idx = std::size_t(m) * kMlpHidden + col0 + pair * 4 + i;
          const float reduced = __bfloat162float(__float2bfloat16_rn(r[i]));
          request.resid_out[idx] = __float2bfloat16_rn(reduced + mlp_bf16_at(residual, (chunk * 8 + m) * kColsPerCtaD4 + pair * 4 + i));
        }
      }
    }
  }
};
using MlpBody = MlpRowsBody<1>;
constexpr int kMlpSmem = MlpBody::kSharedBytes;
}  // namespace fastllm_draft_ops
