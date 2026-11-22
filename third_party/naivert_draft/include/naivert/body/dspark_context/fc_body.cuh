#pragma once

// Modified for FastLLM fused draft execution. See the parent vendor NOTICE
// and UPSTREAM.json for adaptation details and original source hashes.
#include <cutlass/arch/memory_sm80.h>
#include "naivert/body/dspark_context/args.hpp"
#include "naivert/body/dspark_context/geometry.cuh"
#include "naivert/common/resident_grid_barrier.cuh"
#include "naivert/body/dspark_common/diagnostics.hpp"
// D1: existing BF16 rows MMA, followed by the original TP8
// rank-order reduction, masked RMSNorm and replay-generation rendezvous.
// Each CTA owns 32 output columns (two MMA tiles). D12 uses all 384 threads
// for peer exchange; 64 reducers preserve rank order. All threads join barriers.
// Projection, transport and RMSNorm reduction orders are preserved.
#include "naivert/body/attn_common/strided_rows_mma.cuh"

#include "naivert/common/tp8_mailbox.cuh"
#include "naivert/common/prefetched_rank_sum.cuh"
#include "naivert/body/dspark_common/rendezvous.cuh"

#include <cuda_bf16.h>
#include <cuda/ptx>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fastllm_draft_ops {

constexpr int kRanks = naivert::geometry::kTpRanks;
constexpr int kCtas = 128;                    // < SM count, so every CTA is resident: both the
                                              // mailbox spin and the grid barrier require it
constexpr int kColsPerCta = kHidden / kCtas;  // 32
constexpr int kShellWarps = kThreads / 32;
constexpr int kQwords = kColsPerCta / 2;      // 16 mailbox qwords per (row, CTA)
constexpr int kPairs = kQwords / 2;           // 8 qword pairs, one thread each

// All-reduce sites alternate mailbox phases in execution order:
//   D1: 2; D3: 3,5,7,9,11; D4: 4,6,8,10,12;
//   D6 candidate: 20..26. Former latent tags 13..19 are reserved.
// D1/D3/D4 use row slabs 0..7; D6 uses slab 9 (8 is reserved). D2/D5 do not reduce.
// Within a block, alternating phases tolerate one site of rank skew. At a block
// boundary, D4(12)->D1(2) and D6(26)->D6(20) reuse a phase.
// D1's monotone rendezvous waits until every rank has finished the previous block
// before any new mailbox publication, covering both reuse hazards.
//
// Every CTA reads replay_generation at entry. One thread increments it after the
// grid barrier; D3/D4/D6 only read it. D1 must execute equally often on all ranks.
// Generation tags prevent stale matches at a repeated site; they do not protect
// a peer that still needs the previous contents of a reused mailbox address.
// All waits therefore retain their watchdog deadline and device error flag, which
// the runtime checks after draining. watchdog_limit_ns==0 explicitly disables it.
constexpr std::uint32_t kSaltD1 = 2;

static_assert(kColsPerCta == 32, "each mailbox tile covers 32 columns");
static_assert(std::size_t(kCtas) * naivert::kMailboxWarpStride <= naivert::kMailboxPhaseStride,
              "one row's CTA records must fit one phase of one peer lane");

// A row of the block gets its own mailbox slab; inside it a CTA's record is one warp stride.
struct FcCtxLayout {
  __host__ __device__ static constexpr std::size_t tile_offset(int task, int qword) {
    return std::size_t(task) * naivert::kMailboxWarpStride +
           std::size_t(qword) * sizeof(unsigned long long);
  }
};

// FcCtxArgs is shared with bindings through args.hpp. Pin its
// original ABI so the frozen standalone D1 launcher remains compatible.
// 9 pointers (72) + mailbox_peers[8] (64) + replay_generation (8) + salt/rank (8) + the two
// watchdog fields (16).
static_assert(sizeof(FcCtxArgs) == 176,
              "FcCtxArgs changed size -- check the shared D12 parameter contract");

// ------------------------------------------------------------------ grid barrier
// Sense-reversing arrive-and-wait across the whole grid. Every CTA is resident (kCtas < SM count),
// so the spin cannot deadlock. `generation` is never reset, which is what makes the barrier safe
// across graph replays: a replay just continues counting.

// The barrier and the mailbox spin BOTH assume every CTA is resident. If the grid ever exceeds
// what the device can hold, a CTA that has not launched cannot arrive and the ones that did spin
// forever -- inside a captured graph that is an unkillable hang, not an error.
static_assert(kCtas < NAIVERT_SM_COUNT,
              "dspark_fc_ctx needs the whole grid resident; kCtas must stay below the SM count");

// FastLLM retains one row-major FC shard for both small resident kernels
// and wide GEMMs, without retaining another packed weight payload.
struct FcBeforeReduce { __device__ void operator()() const {} };
struct FcCtxBody {
  template <bool kConsumerNorm = false, bool PreparedInput = false, class BeforeReduce = FcBeforeReduce, bool Precomputed = false>
  __device__ static void run(unsigned char* smem_raw, const FcCtxArgs& args,
      BeforeReduce before_reduce = {}, int valid_rows = 8, float precomputed = 0.f) {
    constexpr int stride = kHidden + (kConsumerNorm ? 32 : 0);
    constexpr int packed_offset=4096;
    static_assert((kRanks-1)*kWindow*kPairs*sizeof(naivert::PayloadPair)<=packed_offset);
    static_assert(packed_offset+kWindow*kColsPerCta*sizeof(__nv_bfloat16)<=kWindow*kHidden*2);
    auto* sa = reinterpret_cast<__nv_bfloat16*>(smem_raw);          // [8][4096] staged hidden, 64 KiB
    auto* stage = reinterpret_cast<float*>(smem_raw + kWindow * stride * 2);  // [8][32] partials
    auto* sctx = stage + kWindow * kColsPerCta;                     // [8][32] reduced, masked
    auto* sred = sctx + kWindow * kColsPerCta;                      // [8] rsqrt scale

    auto* mma_partials = sred + kWindow;
    const int tid = int(threadIdx.x);
    const int cta = int(blockIdx.x);
    const int col0 = cta * kColsPerCta;

// --- block-boundary rendezvous, announce side (dspark_common/rendezvous.cuh)
    // Announced HERE and waited on below, so the NVLink round trip overlaps the whole GEMV instead
    // of being paid in series. `k` is the same single read of the counter every mailbox site below
    // uses -- see the salt table's "D1 READS the counter ONCE".
    const unsigned k = static_cast<unsigned>(*args.replay_generation);
    if (!Precomputed && cta == 0 && tid == 0) {
      rendezvous_announce(args.mailbox_peers, args.rank, k);
    }

    if constexpr (!Precomputed) {
      // --- stage this rank's eight hidden rows (every warp reads all of them, 8x reuse)
      {
        const auto* src = reinterpret_cast<const uint4*>(args.target_hidden);
        auto* dst = reinterpret_cast<uint4*>(sa);
        const int total = kWindow * kHidden / 8;      // 16 B per element
        if constexpr (kConsumerNorm) {
          int pending = 0;
          for (int i = tid; i < total; i += kThreads) {
            const int destination = (i / (kHidden / 8)) * (stride / 8) + i % (kHidden / 8);
            if (!PreparedInput || i < valid_rows * (kHidden / 8))
              cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(dst + destination, src + i);
            else dst[destination] = make_uint4(0,0,0,0);
            if (++pending == 8) {
              cutlass::arch::cp_async_fence();
              pending = 0;
            }
          }
          cutlass::arch::cp_async_fence();
          cutlass::arch::cp_async_wait<0>();
        } else {
          for (int i = tid; i < total; i += kThreads) {
            dst[(i / (kHidden / 8)) * (stride / 8) + i % (kHidden / 8)] = src[i];
          }
        }
      }
      __syncthreads();

      naivert::attn_rows::project_strided_rows_mma<kWindow, kHidden, 4, 2, kShellWarps, stride>(
          0, kHidden / 16,
          [&](int column) {
            return args.fc + std::size_t(column) * kHidden;
          },
          sa, mma_partials,
          [&](int column, int row, float value) {
            stage[row * kColsPerCta + column - col0] = value;
          });

    } // projection is supplied by the two-request producer when precomputed.
    constexpr int pairs = kColsPerCta / 2;
    constexpr int items = kWindow * pairs;
    constexpr int exact_offset = 8192;
    static_assert((kRanks - 1) * items * sizeof(naivert::PayloadPair) <= exact_offset);
    static_assert(pairs * 2 * sizeof(unsigned long long) <= naivert::kMailboxWarpStride);
    auto* exact = reinterpret_cast<float*>(smem_raw + exact_offset);
    auto* gathered = reinterpret_cast<naivert::PayloadPair*>(smem_raw);
    if (tid < kWindow * kColsPerCta) exact[tid] = Precomputed ? precomputed : stage[tid];
    __syncthreads();
    before_reduce();
    const naivert::MailboxWatchdog watchdog{args.watchdog_limit_ns,args.watchdog_flag};
    if (tid == 0) rendezvous_wait(args.mailbox_peers[args.rank], args.rank, k, watchdog);
    __syncthreads();
    const unsigned gen = naivert::mailbox_generation(k,args.generation_salt);
    for (int job=tid; job<(kRanks-1)*items; job+=kThreads) {
      const int peer=job/items,item=job%items,row=item/pairs,pair=item%pairs;
      const uint2 payload=*reinterpret_cast<const uint2*>(exact+item*2);
      const auto mailbox=naivert::TileMailbox<FcCtxLayout>::make(
          args.mailbox_peers,args.rank,gen,std::size_t(row)*naivert::kMailboxSlabStride);
      auto* target=mailbox.remote_word(naivert::mailbox_peer_rank(peer,args.rank),cta,pair*2);
      naivert::store_tagged_pair(target,naivert::tagged_word(gen,payload.x),naivert::tagged_word(gen,payload.y));
    }
    for (int job=tid;job<(kRanks-1)*items;job+=kThreads) {
      const int slot=job/items,item=job%items,row=item/pairs,pair=item%pairs;
      const auto mailbox=naivert::TileMailbox<FcCtxLayout>::make(
          args.mailbox_peers,args.rank,gen,std::size_t(row)*naivert::kMailboxSlabStride);
      gathered[job]=mailbox.wait_pair_bounded<true>(slot,cta,pair*2,watchdog,args.generation_salt);
    }
    __syncthreads();
    if (tid<items) {
      const int row=tid/pairs,pair=tid%pairs;
      const uint2 own=*reinterpret_cast<const uint2*>(exact+tid*2);
      float sum[2]={0.f,0.f};
#pragma unroll
      for (int peer=0;peer<kRanks;++peer) {
        const auto value=peer==args.rank ? naivert::PayloadPair{own.x,own.y} :
            gathered[naivert::mailbox_source_slot(peer,args.rank)*items+tid];
        sum[0]=__fadd_rn(sum[0],__uint_as_float(value.lo));
        sum[1]=__fadd_rn(sum[1],__uint_as_float(value.hi));
      }
      const bool valid=PreparedInput ? row<valid_rows : float(args.ctx_valid[row])!=0.f;
      float ssq=0.f;
#pragma unroll
      for (int i=0;i<2;++i) {
        const auto value=__float2bfloat16_rn(valid?sum[i]:0.f);
        args.ctx_sum[std::size_t(row)*kHidden+col0+pair*2+i]=value;
        const float x=float(value); sctx[row*kColsPerCta+pair*2+i]=x; ssq+=x*x;
      }
#pragma unroll
      for (int off=8;off;off>>=1) ssq+=__shfl_xor_sync(0xffffffffu,ssq,off);
      if (!pair) args.ssq[row*kCtas+cta]=ssq;
    }

    if constexpr (kConsumerNorm) asm volatile("fence.proxy.async;" ::: "memory");
    grid_barrier(args.barrier);

    if constexpr (!kConsumerNorm) {
    // One output warp owns one row's ordered SSQ fold and normalization.
    // Lane l loads columns 4*l..4*l+3. The running sum visits lanes 0..31,
    // retaining exactly the original scalar c=0..127 addition order.
    static_assert(kCtas == 32 * 4 && kColsPerCta == 32);
    const int row = tid >> 5, lane = tid & 31;
    if (row < kWindow) {
      const float4 pieces = reinterpret_cast<const float4*>(args.ssq + row * kCtas)[lane];
      float sum = 0.0f;
#pragma unroll 1
      for (int owner = 0; owner < 32; ++owner) {
        if (lane == owner) {
          sum = __fadd_rn(sum, pieces.x);
          sum = __fadd_rn(sum, pieces.y);
          sum = __fadd_rn(sum, pieces.z);
          sum = __fadd_rn(sum, pieces.w);
        }
        sum = __shfl_sync(0xffffffffu, sum, owner);
      }
      const float inverse_rms = rsqrtf(sum / float(kHidden) + args.rms_epsilon);
      const int n = col0 + lane;
      const bool valid = PreparedInput ? row<valid_rows : __bfloat162float(args.ctx_valid[row]) != 0.0f;
      const float f = (sctx[row * kColsPerCta + lane] * inverse_rms) * args.hidden_norm[n];
      args.ctx[std::size_t(row) * kHidden + n] = __float2bfloat16_rn(valid ? f : 0.0f);
    }

    }

    // Every CTA has read the counter (that happened before the barrier), so one thread may advance
    // it for the next execution.
    if (cta == 0 && tid == 0) {
      atomicAdd(reinterpret_cast<unsigned*>(args.replay_generation), 1u);
    }
  }

};

}  // namespace fastllm_draft_ops
