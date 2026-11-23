#pragma once

// Modified for FastLLM fused draft execution. See the parent vendor NOTICE
// and UPSTREAM.json for adaptation details and original source hashes.
#include "naivert/body/dspark_head_markov/args.hpp"
#include "naivert/body/dspark_sampling/random.cuh"
#include "naivert/body/dspark_common/temperature.cuh"
#include "naivert/common/resident_grid_barrier.cuh"
#include "naivert/body/dspark_common/diagnostics.hpp"
// D6: seven serial Markov steps in one persistent launch. W1 is replicated
// on every rank, so the latent is a local row lookup. W2 and logits remain
// vocabulary-sharded; candidate maxima still cross all eight ranks.
//
// Only one former masked-rank contribution was nonzero. Adding zero to the
// selected BF16 row reproduces that reduction, including signed-zero behavior.
// lat_r remains the masked rank-local diagnostic; latent/lat_trace contain
// the selected row on every rank. Initialization uses the existing rows MMA
// primitive to precompute all possible BF16 biases. Each recurrent step reads
// its table row; rounding, ties and the seven candidate exchanges remain.
//
// Candidate tags 20..26 alternate phase buffers. CTA partials also alternate
// two scratch slots. Before a CTA reuses slot j at step j+2, it must cross
// barrier A at j+1, where every local CTA has already finished reading j.
// Likewise, publishing candidate j+1 requires all local CTAs to cross A at
// j+1, after consuming every peer's candidate j. A peer cannot publish j+2
// until it consumes j+1 from all ranks, so no unread j candidate is replaced.
// D1's block-boundary rendezvous remains necessary across whole draft blocks.
// Former latent tags 13..19 and their slab remain reserved in the public layout.
// The fixed grid remains resident; all candidate waits retain their watchdog.
#include "naivert/common/bf16_fma.cuh"
#include "naivert/body/attn_common/rows_gemv_mma.cuh"

#include "naivert/common/tp8_mailbox.cuh"
#include "naivert/body/dspark_common/mailbox_layout.hpp"

#include <cuda_bf16.h>
#include <cuda/ptx>
#include <cuda/atomic>
#include <cuda_runtime_api.h>

#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace fastllm_draft_ops {

constexpr int kMkRows = 7;            // GAMMA: seven proposal rows, seven serial steps
constexpr int kMkVocab = 19072;       // VOCAB_SHARD = 152576 / 8
constexpr int kMkRank = 256;          // MARKOV_RANK, the latent width
constexpr int kMkTilesPerCta = 8;
constexpr int kMkCtas = kMkVocab / (16 * kMkTilesPerCta);
static_assert(kMkVocab % (16 * kMkTilesPerCta) == 0);
constexpr int kMkScratchElements = 2 * kMkCtas;
constexpr int kMkWarps = kMkTilesPerCta;
constexpr int kMkThreads = kMkWarps * 32;
constexpr int kMkPairs = kMkRank / 4;              // 64 threads load the latent
constexpr unsigned kSaltD6Latent = 13;             // reserved former latent tags 13..19
constexpr unsigned kSaltD6Cand = 20;               // .. 26, one per step
// D6's two slabs are minted in dspark_mailbox_slabs.cuh, not here. They used to be two literals in
// this file -- and this file did not include the ledger, so "the ledger accounts for every slab"
// was a claim about a ledger that two of the eleven slabs had never been entered in. The ledger
// asserts they are distinct, clear of the row slabs, clear of the rendezvous, and inside the
// allocation; none of that was checkable while the numbers lived here.
constexpr std::size_t kMkSlabCand = std::size_t(kMarkovCandSlab);

static_assert(kMkRank % 4 == 0, "the latent is carried as bf16x2 pairs, four values per task");
static_assert(kMkRank == 32 * 8, "one warp covers the latent with an eight-wide load per lane");
static_assert(kMkPairs <= kMkThreads, "one thread per four latent values");
static_assert(kMkCtas <= kMkThreads, "the CTA partials are reduced by one thread each");
// A CORRECTNESS requirement, unlike D5's. This grid barriers and spins on peers inside the launch:
// a CTA that has not been scheduled can neither arrive nor publish, and the whole grid hangs.
static_assert(kMkCtas <= 2 * NAIVERT_SM_COUNT, "at most two CTAs per SM; runtime residency check remains mandatory");
static_assert(kSaltD6Latent + kMkRows - 1 < kSaltD6Cand, "the two D6 tag runs must not overlap");
static_assert(kSaltD6Cand + kMkRows - 1 <= 125, "the tag lives in the generation's low byte");

struct MarkovLayout {
  __host__ __device__ static constexpr std::size_t tile_offset(int task, int qword) {
    return std::size_t(task) * naivert::kMailboxWarpStride +
           std::size_t(qword) * sizeof(unsigned long long);
  }
};


static_assert(sizeof(MarkovArgs) == 264,
              "MarkovArgs changed size; preserve the shared ABI");

// Sense-reversing arrive-and-wait across the whole grid; see dspark_fc_ctx.cu for the fencing
// argument (both directions have to be fenced by hand once `grid.sync()` is not used).
__device__ inline void markov_grid_barrier(unsigned* state) {
  naivert::resident_grid_barrier(state);
}

// Ties to the lower index, which is what `torch.max(dim=)` documents ("the indices of the first
// maximal value"). Written once and used for the warp fold, the CTA fold, the grid fold and
// the eight-rank fold, so all four tie-breaks are the same rule by construction.
__device__ inline void markov_take(float& bv, int& bi, float v, int i) {
  if (v > bv || (v == bv && i < bi)) {
    bv = v;
    bi = i;
  }
}

template<int Ctas, int Threads, bool Fused>
__device__ __forceinline__ void markov_steps(const MarkovArgs& args, const float* local_logits) {
  __shared__ float swv[(Threads / 32)];
  __shared__ int swi[(Threads / 32)];
  __shared__ int sprev;

  const int tid = int(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int cta = int(blockIdx.x);
  const unsigned epoch = static_cast<unsigned>(*args.replay_generation);

  // Every CTA on every rank starts from the same anchor and walks the same chain, so `prev` stays
  // grid-uniform and rank-uniform without ever being broadcast.
  int prev = min(max(int(args.block_ids[0]), 0), 8 * kMkVocab - 1);

  for (int j = 0; j < kMkRows; ++j) {
    const int scratch = (j & 1) * Ctas;
    const int owner = prev / kMkVocab;
    const bool own = owner == args.rank;

    // Replicated W1 preserves the latent diagnostics without a rank reduction.
    if (kExportDiagnostics && args.apply_markov) {
      if (tid < kMkPairs) {
        const auto* row = args.markov_w1 + std::size_t(prev) * kMkRank + tid * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float value = __bfloat162float(row[i]);
          const auto latent = __float2bfloat16_rn(__fadd_rn(0.0f, value));
          if (cta == 0) {
            args.lat_r[tid * 4 + i] = __float2bfloat16_rn(own ? value : 0.0f);
            args.lat_trace[std::size_t(j) * kMkRank + tid * 4 + i] = latent;
            args.latent[tid * 4 + i] = latent;
          }
        }
      }
      __syncthreads();
    }

    // ------------------------------------------- bias lookup and this CTA's partial argmax
    float bv = -INFINITY;
    int bi = INT_MAX;
    // Reuse the rows primitive with one latent query and sixteen weight
    // columns per warp. Emit adjacent vocabulary columns across adjacent
    // threads, so the required bias/logit traces are coalesced too.
    auto emit = [&](int n, float bias_value) {
      float slog;
      if constexpr (Fused) {
        const int wave = (n / 16) / Ctas;
        slog = local_logits[j * 144 + wave * 16 + n % 16];
      } else {
        slog = args.logits[std::size_t(j) * kMkVocab + n];
      }
      if (args.apply_markov) {
        const __nv_bfloat16 bias = __float2bfloat16_rn(bias_value);
        slog += __bfloat162float(bias);
        if constexpr (kExportDiagnostics) {
          args.bias_trace[std::size_t(j) * kMkVocab + n] = bias;
          args.step_bias[n] = bias;
        }
      }
      if constexpr (kExportDiagnostics) {
        args.step_logits[n] = slog;
      }
      if (kExportDiagnostics || args.sampling_state)
        args.logit_trace[std::size_t(j) * kMkVocab + n] = slog;
      const float score = args.sampling_state
          ? naivert_dspark::tempered(slog, args.sampling_params ? args.sampling_params[4] : naivert_dspark::kInvTemperature)
                + sampling_gumbel(args.sampling_state, j, args.rank * kMkVocab + n)
          : slog;
      markov_take(bv, bi, score, n);
    };
    if constexpr (Fused) {
      if (tid < 144) {
        const int n = (cta + (tid / 16) * Ctas) * 16 + tid % 16;
        if (n < kMkVocab)
          emit(n, args.apply_markov ? __bfloat162float(args.bias_table[std::size_t(prev) * kMkVocab + n]) : 0.f);
      }
    } else {
    if (args.apply_markov) {
      const int n = cta * kMkTilesPerCta * 16 + tid;
      if (tid < kMkTilesPerCta * 16)
        emit(n, __bfloat162float(args.bias_table[std::size_t(prev) * kMkVocab + n]));
    } else {
      const int n = cta * kMkTilesPerCta * 16 + tid;
      if (tid < kMkTilesPerCta * 16) emit(n, 0.0f);
    }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const float other_value = __shfl_xor_sync(0xffffffffu, bv, off);
      const int other_index = __shfl_xor_sync(0xffffffffu, bi, off);
      markov_take(bv, bi, other_value, other_index);
    }
    if (lane == 0) {
      swv[warp] = bv;
      swi[warp] = bi;
    }
    __syncthreads();
    if (tid == 0) {
      float cv = swv[0];
      int ci = swi[0];
      for (int w = 1; w < (Threads / 32); ++w) {
        markov_take(cv, ci, swv[w], swi[w]);
      }
      args.cta_val[scratch + cta] = cv;
      args.cta_idx[scratch + cta] = ci;
    }

    // ------------------------------------------------- barrier A: all Ctas partials are written
    markov_grid_barrier(args.barrier);

    // Every CTA folds all Ctas partials itself. This small array is cheaper than another rendezvous, and
    // it leaves the shard winner in every CTA without a broadcast.
    {
      float cv = -INFINITY;
      int ci = INT_MAX;
      if (tid < Ctas) {
        cv = args.cta_val[scratch + tid];
        ci = args.cta_idx[scratch + tid];
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, cv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, ci, o);
        markov_take(cv, ci, ov, oi);
      }
      if (lane == 0) {
        swv[warp] = cv;
        swi[warp] = ci;
      }
    }
    __syncthreads();

    // ------------------------------------------------------- rendezvous 2: the candidate max
    if (tid == 0) {
      float cv = swv[0];
      int ci = swi[0];
      for (int w = 1; w < (Threads / 32); ++w) {
        markov_take(cv, ci, swv[w], swi[w]);
      }
      const int gidx = ci + args.rank * kMkVocab;   // the stand-in's `+= shard_base[rank]`
      const unsigned gen = naivert::mailbox_generation(epoch, kSaltD6Cand + unsigned(j));
      const auto mailbox = naivert::TileMailbox<MarkovLayout>::make(
          args.mailbox_peers, args.rank, gen, kMkSlabCand * naivert::kMailboxSlabStride);
      const std::uint32_t mine_v = __float_as_uint(cv);
      const std::uint32_t mine_i = static_cast<std::uint32_t>(gidx);
      if (cta == 0) {
        mailbox.publish_pair(0, 0, mine_v, mine_i);
      }
      // The candidate is (fp32 value, vocabulary index) -- 64 bits, which is exactly one tagged
      // pair. Nothing here is bf16: rounding a logit to publish it would change the winner.
      float best = -INFINITY;
      int bestg = INT_MAX;
#pragma unroll
      for (int src = 0; src < 8; ++src) {
        float v = cv;
        int g = gidx;
        if (src != args.rank) {
          const naivert::MailboxWatchdog watchdog{args.watchdog_limit_ns,
                                                  args.watchdog_flag};
          const naivert::PayloadPair pr = mailbox.wait_pair_bounded<true>(
              naivert::mailbox_source_slot(src, args.rank), 0, 0, watchdog,
              kSaltD6Cand + unsigned(j));
          v = __uint_as_float(pr.lo);
          g = int(pr.hi);
        }
        if (kExportDiagnostics && cta == 0) {
          args.cand_val[src] = v;
          args.cand_idx[src] = g;
        }
        // Rank order 0..7 with strictly-greater, so a tie goes to the lowest rank -- which is what
        // `torch.max(cand_val, dim=0)` returns.
        if (v > best) {
          best = v;
          bestg = g;
        }
      }
      // A timed-out candidate wait returns garbage and sets the watchdog.
      // Clamp both the next full-table lookup and exported embedding index:
      // the invalid result must be reported, not turned into an out-of-bounds read.
      bestg = min(max(bestg, 0), 8 * kMkVocab - 1);
      sprev = bestg;
      if (cta == 0) {
        args.draft_tokens[j] = bestg;
        args.block_ids[j + 1] = bestg;   // row j + 1 now carries its draft
        args.prev_token[0] = bestg;
      }
    }
    __syncthreads();
    prev = sprev;

    // Alternating scratch is protected by the next step's barrier A.
  }
}

struct MarkovBody {
  __device__ static void run(const MarkovArgs& args) {
    markov_steps<kMkCtas, kMkThreads, false>(args, nullptr);
  }
};
} // namespace fastllm_draft_ops
