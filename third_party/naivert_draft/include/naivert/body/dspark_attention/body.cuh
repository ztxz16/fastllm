#pragma once

// Modified for FastLLM fused draft execution. See the parent vendor NOTICE
// and UPSTREAM.json for adaptation details and original source hashes.
#include "naivert/body/dspark_attention/args.hpp"
#include "naivert/body/attn_common/strided_rows_mma.cuh"
#include "naivert/body/dspark_common/diagnostics.hpp"
// Eight-row Q/K/V, attention, O projection, TP fold and residual.
// Preserve the reference arithmetic and internal synchronization. The entry
// supplies the fixed resident grid; this body does not use the consumer/helper role split.
#include "naivert/common/prefetched_rank_sum.cuh"

#include "naivert/common/bulk_input_copy.cuh"
#include "naivert/pdl/grid_dependency.cuh"

#include "naivert/common/bf16_fma.cuh"

#include "naivert/common/tp8_mailbox.cuh"
#include "naivert/body/dspark_common/mailbox_layout.hpp"

#include <cuda_bf16.h>
#include <cuda/atomic>
#include <cuda/ptx>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include "naivert/body/rmsnorm_head_proj/rmsnorm_head_proj_rows_body.cuh"

namespace fastllm_draft_ops {

constexpr int kAttnWindow = 8;                          // WINDOW
// This one slab index per block row is the whole of D3's mailbox footprint, so the window
// IS the row-slab count. Asserted, not assumed: growing this alone used to compile green
// and send D3's o_proj publishes straight over D6's slabs and the rendezvous.
static_assert(kAttnWindow == kRowSlabs, "D3's window must be the ledger's row-slab count");
constexpr int kAttnHidden = 4096;                       // D_HIDDEN
constexpr int kAttnHeadDim = 128;                       // D_HEAD_DIM
constexpr int kAttnQHeads = 4;                          // Q_HEADS_R = 32 / 8
constexpr int kAttnQRows = kAttnQHeads * kAttnHeadDim;  // Q_ROWS_R  = 512
constexpr int kAttnKvRows = kAttnHeadDim;               // KV_ROWS_R = 128, one kv head per rank
constexpr int kAttnCtas = 128;                          // < SM count: barrier and mailbox need it
constexpr int kAttnWarps = 8;
constexpr int kAttnWorkThreads = kAttnWarps * 32;
constexpr int kAttnThreads = 4 * kAttnHeadDim;           // 512: one thread per P*V stream/dimension
constexpr int kAttnThreadGroups = kAttnThreads / kAttnWorkThreads;
constexpr int kRowsPerGroup = kAttnWindow / kAttnThreadGroups;
static_assert(kAttnThreads % kAttnWorkThreads == 0 && kAttnWindow % kAttnThreadGroups == 0);
constexpr float kAttnRmsEps = 1e-6f;                    // RMS_EPS

// Phase 1: 512 q + 128 k + 128 v columns, projected in 16-column MMA tiles.
constexpr int kProjCols = kAttnQRows + 2 * kAttnKvRows;      // 768
constexpr int kQkRaw = kAttnQRows + kAttnKvRows;             // 640: the q and k halves of a row
constexpr int kChunksK = kAttnHidden / 256;                  // 16 k-chunks of 256

// Phase 2: one selected Q head and fresh K, eight rows each, across 16 warps.
constexpr int kHeadTasks = 2 * kAttnWindow; // one selected Q head plus fresh K
constexpr int kAttnSplits = kAttnCtas / kAttnQHeads;
constexpr int kSplitTile = 16;
constexpr int kPvBatchKeys = 64;
constexpr int kScoreStride = kPvBatchKeys + 4;
static_assert(kAttnWindow * kPvBatchKeys <= kAttnThreads);
constexpr int kSplitRowStride = kAttnHeadDim + 2;
constexpr int kSplitScratch = kAttnCtas * kAttnWindow * kSplitRowStride;
constexpr int kReadyK = kAttnQHeads;
constexpr int kReadyV = kReadyK + 1;
constexpr int kReadyPv = kReadyV + 1;
constexpr int kReadyMerged = kReadyPv + kAttnQHeads;
constexpr int kReadyCount = kReadyMerged + 1;
constexpr int kReadyStride = 32;
constexpr unsigned kMergedHeadBits = 4;
constexpr unsigned kMergedHeadMask = (1u << kMergedHeadBits) - 1u;
static_assert(kAttnWindow <= kMergedHeadMask && kAttnQHeads * kMergedHeadBits <= 32);
static_assert(kAttnQHeads == 4 && kAttnWindow == 8);
constexpr int kReadyReset = kReadyCount;
constexpr int kReadyWords = (kReadyCount + 1) * kReadyStride;
static_assert(kSplitScratch * sizeof(float) % 128 == 0);

static_assert(kAttnSplits == 32 && kAttnCtas % kAttnQHeads == 0);
constexpr int kPerLane = kAttnHeadDim / 32;                           // 4 elements per lane

// phase 3: one CTA per (head, row).
constexpr int kAttnRowTasks = kAttnQHeads * kAttnWindow;              // 32

// Phase 4: two 16-column MMA tiles per CTA, followed by the original mailbox reduction.
constexpr int kColsPerCtaD3 = kAttnHidden / kAttnCtas;                // 32
constexpr int kQwordsD3 = kColsPerCtaD3 / 2;                          // 16
constexpr int kPairsD3 = kQwordsD3 / 2;                               // 8

// Shared memory: phase 1's raw input is staged in the main arena and every later
// phase aliases the same bytes. See the offsets in the kernel; the tail past this holds the two
// tiny fp32 scratch arrays, which is the layout dspark_mlp_rows8.cu arrived at for the same
// reason -- writing a later phase's structure into the middle of `sx` corrupts the GEMV's input.
// Offset adjacent query rows by four shared-memory banks for QK vector loads.
constexpr int kSharedQStride = kAttnQRows + 8;
constexpr int kSharedKStride = kAttnKvRows + 8;
constexpr int kNormQueryStride = kAttnHidden + 8;
constexpr int kNormQueryBytes = kAttnWindow * kNormQueryStride * int(sizeof(__nv_bfloat16));
constexpr int kAttnMain = kAttnWindow * kAttnHidden * 2;              // 65536
constexpr int kMaxSplitKeys = kAttnMain / int(sizeof(float)) / kAttnSplits;
static_assert((kMaxSplitKeys + kAttnWindow) * kAttnWindow * sizeof(float) <=
              kAttnWindow * (kQkRaw + kSharedQStride) * sizeof(__nv_bfloat16),
              "local scores must not overwrite the fresh normalized keys");
constexpr int kAttnTailFloats = 2 * kAttnWindow;                      // sred[8], swarp[8]
constexpr int kPvTileRows = 256;
constexpr int kPvTileOffset = kAttnMain + kAttnTailFloats * int(sizeof(float));
constexpr int kAttnSmem = kPvTileOffset + kNormQueryBytes;
static_assert(kNormQueryBytes >= kPvTileRows * kAttnHeadDim * sizeof(__nv_bfloat16));
static_assert((kAttnThreads / 32) * 128 * sizeof(float) <= kAttnMain);

static_assert(kAttnQHeads % kAttnThreadGroups == 0, "split whole independent heads");
static_assert(kAttnWindow * kAttnWarps * sizeof(float) <=
                  kPvTileRows * kAttnHeadDim * sizeof(__nv_bfloat16),
              "norm partials fit in the unused V tile");
static_assert(kPerLane * 32 == kAttnHeadDim, "the rope partner is lane l^16 only if 32*4 == 128");
static_assert(kAttnRowTasks <= kAttnCtas, "phase 3 must be a subset of the launched grid");
static_assert(kColsPerCtaD3 == 32, "two 16-column MMA tiles per CTA");
static_assert((kAttnThreads / 32) * 128 * sizeof(float) <=
              kPvTileRows * kAttnHeadDim * sizeof(__nv_bfloat16), "MMA scratch fits the idle V tile");
static_assert(std::size_t(kAttnCtas) * naivert::kMailboxWarpStride <= naivert::kMailboxPhaseStride,
              "one row's CTA records must fit one phase of one peer lane");
// Same residency requirement as D1, D2 and D4: a CTA that has not launched cannot arrive at the
// barrier, and the ones that did would spin forever -- inside a captured graph, an unkillable hang.
// The four phases hand-partition one dynamic allocation, so bound every phase's high-water mark
// here rather than in a comment: D4 already paid for one hand-checked partition (gate3.md §5.7).
// phase 1  sx    [8][4096] bf16, then MMA partials                      = 65536 = kAttnMain
//          normalized queries [8][4104] bf16 occupy the separate V arena
// phase 2  sraw  [8][640] + sq [8][520] + sk [8][136] bf16             = 20736
// phase 3  CTA-local scores, online softmax/PV and split merge; all fit the main/V arenas
// phase 4  sh    [8][520] bf16 + stage [8][32] fp32                    =  9344
// `sred` / `swarp` live in the 64-byte tail PAST kAttnMain and never overlap any of the above.
// The separate V arena holds normalized queries in phase 1 and MMA partials
// in phases 2 and 4. CTA barriers protect local reuse; readiness counters
// guard the global handoffs between producers and consumers.
static_assert(std::size_t(kAttnWindow) * kAttnHidden * sizeof(__nv_bfloat16) == kAttnMain,
              "phase 1's sx is exactly the main arena");
static_assert(std::size_t(kAttnWindow) * (kQkRaw + kSharedQStride + kSharedKStride) *
                      sizeof(__nv_bfloat16) <= kAttnMain,
              "phase 2's sraw + sq + sk must fit the main arena");
static_assert(std::size_t(kAttnWindow) * kSharedQStride * sizeof(__nv_bfloat16) +
                      std::size_t(kAttnWindow) * 32 * sizeof(float) <= kAttnMain,
              "phase 4's sh + stage must fit the main arena");
static_assert(kAttnMain + kAttnTailFloats * sizeof(float) <= kAttnSmem,
              "the scalar tail must fit past the main arena");

static_assert(kAttnCtas < NAIVERT_SM_COUNT,
              "dspark_attn_rows8 needs the whole grid resident; kAttnCtas must stay below the SMs");

// D3's layer salts interleave with D4. D1 owns the epoch bump and the
// block-boundary rendezvous; its source documents the canonical live table.
constexpr std::uint32_t kSaltD3Base = 3;

// A row of the block gets its own mailbox slab; inside it a CTA's record is one warp stride --
// the same layout D1 and D4 use, because the sites are told apart by the TAG, not the address.
struct AttnLayout {
  __host__ __device__ static constexpr std::size_t tile_offset(int task, int qword) {
    return std::size_t(task) * naivert::kMailboxWarpStride +
           std::size_t(qword) * sizeof(unsigned long long);
  }
};

// Per-head producer readiness follows the existing MLA release/acquire protocol.
// The generation-qualified control tail is disjoint from numerical split records.
__device__ inline unsigned* attn_ready_words(const AttnArgs& args) {
  return reinterpret_cast<unsigned*>(args.split_scratch + kSplitScratch);
}
__device__ inline unsigned attn_ready_count(const AttnArgs& args, int index) {
  cuda::atomic_ref<unsigned, cuda::thread_scope_device> ready(
      attn_ready_words(args)[index * kReadyStride]);
  return ready.load(cuda::memory_order_relaxed);
}
// Leader-only acquisition; callers provide any required CTA barrier.
template<class Predicate>
__device__ inline void attn_acquire_ready(const AttnArgs& args, Predicate complete, unsigned site) {
  const auto start = naivert::mailbox_globaltimer_ns();
  while (!complete()) {
    if (args.watchdog_limit_ns &&
        naivert::mailbox_globaltimer_ns() - start >= args.watchdog_limit_ns) {
      if (args.watchdog_flag) atomicCAS(args.watchdog_flag, 0u, site);
      break;
    }
  }
  cuda::ptx::fence(cuda::ptx::sem_acquire, cuda::ptx::scope_gpu);
}
template<class Predicate>
__device__ inline void attn_wait_ready(const AttnArgs& args, Predicate complete, unsigned site) {
  if (threadIdx.x == 0) attn_acquire_ready(args, complete, site);
  __syncthreads();
}
template<bool kAcquireReset = false, bool kAlreadySynchronized = false>
__device__ inline void attn_publish_ready(const AttnArgs& args, int index, unsigned increment = 1u) {
  if constexpr (!kAlreadySynchronized) __syncthreads();
  if (threadIdx.x == 0) {
    if constexpr (kAcquireReset) {
      const unsigned epoch = naivert::mailbox_generation(
          static_cast<unsigned>(*args.replay_generation), args.generation_salt);
      attn_acquire_ready(args, [&] {
        return attn_ready_count(args, kReadyReset) == epoch;
      }, 0xD300u + args.generation_salt);
    }
    if (index >= 0) {
      cuda::atomic_ref<unsigned, cuda::thread_scope_device> ready(
          attn_ready_words(args)[index * kReadyStride]);
      ready.fetch_add(increment, cuda::memory_order_release);
    }
  }
}

__device__ inline float attn_bf16_at(const __nv_bfloat16* v, int i) {
  return __bfloat162float(v[i]);
}

// After PV publication the main score arena is dead. Keep O weights away
// from phase4 query/stage and the separate V-arena MMA partials.
constexpr int kOWeightOffset = 16384;
constexpr int kOWeightBytes = kColsPerCtaD3 * kAttnQRows * sizeof(__nv_bfloat16);
constexpr int kOReadyOffset = kOWeightOffset + kOWeightBytes;
constexpr int kOFailedOffset = kOReadyOffset + sizeof(std::uint64_t);
static_assert(kOFailedOffset + sizeof(unsigned) <= kAttnMain);
static_assert(kOWeightOffset >= kAttnWindow * kSharedQStride * 2 + kAttnWindow * 32 * 4);
using OWeightTransport = naivert_replica_rmsnorm_head_proj::RMSNormHeadProjRows8Body;
__device__ bool wait_o_weight(std::uint64_t* ready, const AttnArgs& args) {
  if (naivert::Barrier::try_wait(ready, 0u)) return true;
  const bool bounded = args.watchdog_limit_ns != 0ull;
  const auto start = bounded ? naivert::mailbox_globaltimer_ns() : 0ull;
  unsigned spins = 0;
  for (;;) {
    if (naivert::Barrier::try_wait(ready, 0u)) return true;
    if (bounded && ((++spins & naivert::kMailboxWatchdogSpinMask) == 0u) &&
        naivert::mailbox_globaltimer_ns() - start > args.watchdog_limit_ns) {
      naivert::mailbox_watchdog_trip(
          naivert::MailboxWatchdog{args.watchdog_limit_ns, args.watchdog_flag},
          0xD350u + args.generation_salt);
      return false;
    }
  }
}

// A batched launch gives each request a resident subgrid. Per-request ready
// words and mailbox slabs are unchanged; no global barrier spans requests.
// The spare word following the reset generation belongs to the same
// per-request ready arena. It publishes eight normalized rows without a new
// allocation. hid_a is scratch until every QKV reader has published its result.
__device__ inline unsigned* attn_normalized_word(const AttnArgs& a) {
  return attn_ready_words(a) + kReadyReset * kReadyStride + 1;
}
template<int Requests, class Args>
__device__ void attn_project_requests(const Args* requests, unsigned char* memory) {
  const int tid=threadIdx.x, cta=blockIdx.x;
  if (cta==0 && tid==0) for (int request=0;request<Requests;++request) {
    const auto &a=requests[request];
    for(int index=0;index<kReadyCount;++index) attn_ready_words(a)[index*kReadyStride]=0u;
    *attn_normalized_word(a)=0u;
    cuda::atomic_ref<unsigned,cuda::thread_scope_device>(attn_ready_words(a)[kReadyReset*kReadyStride]).store(
        naivert::mailbox_generation(unsigned(*a.replay_generation),a.generation_salt),cuda::memory_order_release);
  }
  if(cta<Requests*8) {
    const auto &a=requests[cta/8]; const int row=cta%8;
    const unsigned epoch=naivert::mailbox_generation(unsigned(*a.replay_generation),a.generation_salt);
    attn_wait_ready(a,[&]{return attn_ready_count(a,kReadyReset)==epoch;},0xD340u+a.generation_salt);
    float *partial=reinterpret_cast<float*>(memory);
    if(tid<256) {
      float sum=0.f;
      for(int c=0;c<16;++c){float v=__bfloat162float(a.resid_in[row*4096+c*256+tid]);sum+=v*v;}
      for(int off=16;off;off>>=1)sum+=__shfl_xor_sync(0xffffffffu,sum,off);
      if(tid%32==0)partial[tid/32]=sum;
    }
    __syncthreads();
    if(tid==0){float sum=0;for(int i=0;i<8;++i)sum+=partial[i];partial[0]=rsqrtf(sum/4096.f+a.rms_epsilon);}
    __syncthreads();
    const int k=tid*8;
    uint4 raw=*reinterpret_cast<const uint4*>(a.resid_in+row*4096+k),out;
    const unsigned words[4]={raw.x,raw.y,raw.z,raw.w};
    auto *packed=reinterpret_cast<unsigned*>(&out);
    for(int i=0;i<4;++i){
      float2 v=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(words+i));
      auto n=__floats2bfloat162_rn(float(__float2bfloat16_rn(v.x*partial[0]))*a.input_ln[k+2*i],
                                float(__float2bfloat16_rn(v.y*partial[0]))*a.input_ln[k+2*i+1]);
      packed[i]=*reinterpret_cast<unsigned*>(&n);
    }
    *reinterpret_cast<uint4*>(a.hid_a+row*4096+k)=out;
    __syncthreads();
    if(tid==0)cuda::atomic_ref<unsigned,cuda::thread_scope_device>(*attn_normalized_word(a)).fetch_or(1u<<row,cuda::memory_order_release);
  }
  if(cta<kProjCols/16) {
    for(int request=0;request<Requests;++request) {
      const auto &a=requests[request];
      const unsigned epoch=naivert::mailbox_generation(unsigned(*a.replay_generation),a.generation_salt);
      attn_wait_ready(a,[&]{return attn_ready_count(a,kReadyReset)==epoch;},0xD340u+a.generation_salt);
      attn_wait_ready(a,[&]{return cuda::atomic_ref<unsigned,cuda::thread_scope_device>(*attn_normalized_word(a)).load(cuda::memory_order_relaxed)==255u;},0xD350u+a.generation_salt);
    }
    const auto &a=requests[0];
    naivert::attn_rows::project_strided_rows_mma<8*Requests,4096,16,1,16>(0,kProjCols/16,
      [&](int column){
        using Row=naivert::attn_rows::K32PackedRow;
        const auto *weight=column<512?a.q_w:column<640?a.k_w:a.v_w;
        const int local=column<512?column:column<640?column-512:column-640;
        return Row{weight+std::size_t(local/16)*(16*4096)+(local%16)*32};
      },a.hid_a,reinterpret_cast<float*>(memory),
      [&](int column,int row,float value){
        const auto &out=requests[row/8];auto v=__float2bfloat16_rn(value);
        if(column<640)out.qk_raw[(row%8)*640+column]=v;
        else out.v_blk[(row%8)*128+column-640]=v;
      },-1,[&](int row,int k){return __ldcg(reinterpret_cast<const uint4*>(requests[row/8].hid_a+(row%8)*4096+k));});
    for(int request=0;request<Requests;++request)attn_publish_ready<true,true>(requests[request],cta/8);
  }
}

template<int Ctas = kAttnCtas, int Requests = 1> struct AttnRowsBody {
  static_assert(Ctas == 64 || Ctas == 128 || (Ctas == 32 && Requests == 4));
  static_assert(Requests == 1 || Requests == 2 || Requests == 4);
  static constexpr int kOutputCtas = Requests == 1 ? Ctas : 128;
  static constexpr int kAttnSplits = Ctas / kAttnQHeads;
  static constexpr int kColsPerCtaD3 = kAttnHidden / kOutputCtas;
  static constexpr int kPairsD3 = kColsPerCtaD3 / 4;
  static constexpr int kOWeightOffset = Ctas == 128 ? fastllm_draft_ops::kOWeightOffset : kAttnSmem;
  static constexpr int kOWeightBytes = kColsPerCtaD3 * kAttnQRows * sizeof(__nv_bfloat16);
  static constexpr int kOReadyOffset = kOWeightOffset + kOWeightBytes;
  static constexpr int kOFailedOffset = kOReadyOffset + sizeof(std::uint64_t);
  static constexpr int kSharedBytes = Ctas == 128 ? kAttnSmem : ((kOFailedOffset + sizeof(unsigned) + 127) & ~127);
  // The score arena retains its existing capacity even with fewer splits.
  static constexpr int kMaxHistory = kMaxSplitKeys * kAttnSplits;
  static_assert(kSharedBytes <= 232448);
  static_assert(kColsPerCtaD3 / 2 * sizeof(std::uint64_t) <= naivert::kMailboxWarpStride);

#include "naivert/body/attn_common/prefetch_l2_bulk.cuh"
  template<bool Sliding, class Args>
  __device__ static void run(const Args& args, unsigned char* smem_raw, const Args* requests = nullptr) {
    // Retain the original collective initialization barrier.
    __syncthreads();
    auto* mma_partials = reinterpret_cast<float*>(smem_raw + kPvTileOffset);
    auto* tail = reinterpret_cast<float*>(smem_raw + kAttnMain);
    float* sred = tail;                       // [8] the row norm's scale

    const int tid = int(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int projection_row0 = (tid / kAttnWorkThreads) * kRowsPerGroup;
    const int cta = Requests == 1 ? int(blockIdx.x) : int(blockIdx.x) % Ctas;
    const int output_cta = Requests == 1 ? cta : int(blockIdx.x);
    const int cta_head = cta / kAttnSplits;
    const int ncache = args.ncache;
    const int total = args.total_cols;
    // Prefetch exactly this producer's packed16-column tile while the
    // existing input staging and normalization execute. This is a cache hint;
    // original global loads and all readiness/CTA/TP8 ordering remain.
    if (tid == 0 && output_cta < kProjCols / 16) {
      const int column = output_cta * 16;
      const auto* weight = column < kAttnQRows ? args.q_w :
                           column < kQkRaw ? args.k_w : args.v_w;
      const int local = column < kAttnQRows ? column :
                        column < kQkRaw ? column - kAttnQRows : column - kQkRaw;
      prefetch_l2_bulk(weight + std::size_t(local) * kAttnHidden,
                       16 * kAttnHidden * sizeof(__nv_bfloat16));
    }

    // Programmatic dependent launch: everything above touches only this rank's weights and
    // this CTA's shared memory. From here on the previous draft kernel's outputs (resid_in,
    // ready words, mailboxes) are read or written, so wait for its full completion first, then
    // let the next kernel's CTAs launch into the free SMs and run their own prologue. Without a
    // programmatic edge both calls are no-ops.
    naivert::pdl::wait_for_primary<true>();
    naivert::pdl::trigger_dependents<true>();

    // Begin publishes positions as well as hidden rows. Read both only after
    // the dependency join, including first use and replays with new lengths.
    int history_begin = 0, history_count = min(int(args.blk_pos[0]), args.window_left);
    float rms_epsilon = kAttnRmsEps;
    if constexpr (Sliding) {
      const auto start = args.blk_pos[0];
      history_begin = 0;
      history_count = min(int(start), args.window_left);
      rms_epsilon = args.rms_epsilon;
    }
    auto cached_position = [&](int local) {
      if constexpr (Sliding) return history_begin + min(local, max(0, history_count - 1));
      else return local;
    };
    auto score_value = [&](float value, int row, int key, bool fresh) {
      if constexpr (Sliding) {
        const auto key_pos = fresh ? args.blk_pos[0] + key : args.blk_pos[0] - history_count + key;
        const auto delta = args.blk_pos[row] - key_pos;
        const bool valid = (fresh ? key < args.valid_rows : key < history_count) && delta <= args.window_left;
        return valid ? float(__float2bfloat16_rn(float(__float2bfloat16_rn(value)) * args.attn_scale)) : -INFINITY;
      } else {
        return __fadd_rn(__fmul_rn(value, args.attn_scale), args.key_add[fresh ? ncache + key : key]);
      }
    };
    // ================================================================= phase 1: norm + projections
    if constexpr(Requests > 1) attn_project_requests<Requests>(requests,smem_raw);
    else {
      auto* sx = reinterpret_cast<__nv_bfloat16*>(smem_raw);          // [8][4096]
      auto* norm_queries = reinterpret_cast<__nv_bfloat16*>(smem_raw + kPvTileOffset);
      const unsigned ready_epoch = naivert::mailbox_generation(
          static_cast<unsigned>(*args.replay_generation), args.generation_salt);
      if (cta == 0 && tid == 0) {
#pragma unroll
        for (int index = 0; index < kReadyCount; ++index)
          attn_ready_words(args)[index * kReadyStride] = 0u;
        cuda::atomic_ref<unsigned, cuda::thread_scope_device> reset(
            attn_ready_words(args)[kReadyReset * kReadyStride]);
        reset.store(ready_epoch, cuda::memory_order_release);
      }
      naivert::bulk_input_copy<kAttnWindow * kAttnHidden * 2>(sx, args.resid_in);

      // Two independent rows share each iteration. Each retains the original 256-thread
      // reduction and eight-warp summation order. Phase3's V tile is free scratch here.
      auto* norm_warp = reinterpret_cast<float*>(smem_raw + kPvTileOffset);
      const int local_tid = tid % kAttnWorkThreads;
      for (int row = 0; row < kRowsPerGroup; ++row) {
        const int m = projection_row0 + row;
        float sum = 0.0f;
        for (int c = 0; c < kChunksK; ++c) {
          const float v = attn_bf16_at(sx, m * kAttnHidden + c * 256 + local_tid);
          sum += v * v;
        }
  #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
          sum += __shfl_xor_sync(0xffffffffu, sum, off);
        }
        // Each row owns its eight warp partials until every row is ready.
        if (lane == 0) norm_warp[m * kAttnWarps + local_tid / 32] = sum;
      }
      __syncthreads();
      if (tid < kAttnWindow) {
        float sum_warps = 0.0f;
        for (int w = 0; w < kAttnWarps; ++w) {
          sum_warps += norm_warp[tid * kAttnWarps + w];
        }
        sred[tid] = rsqrtf(sum_warps / float(kAttnHidden) + rms_epsilon);
      }
      __syncthreads();
      // ONE rounding, at the store. Two separate multiplies, left to right; there is no add here,
      // so nothing can contract.
      static_assert(kAttnThreads == kAttnHidden / 8,
                    "each thread owns the same gamma vector across all rows");
      const int k = tid * 8;
      // Reuse normalize_rows' float4 gamma layout when aligned, retaining
      // this D3 entry point's existing four-byte alignment contract.
      float4 g_lo, g_hi;
      if ((reinterpret_cast<std::uintptr_t>(args.input_ln) & 15u) == 0u) {
        g_lo = __ldg(reinterpret_cast<const float4*>(args.input_ln + k));
        g_hi = __ldg(reinterpret_cast<const float4*>(args.input_ln + k + 4));
      } else {
        g_lo = make_float4(args.input_ln[k], args.input_ln[k+1],
                           args.input_ln[k+2], args.input_ln[k+3]);
        g_hi = make_float4(args.input_ln[k+4], args.input_ln[k+5],
                           args.input_ln[k+6], args.input_ln[k+7]);
      }
      const float gamma[8] = {g_lo.x,g_lo.y,g_lo.z,g_lo.w,g_hi.x,g_hi.y,g_hi.z,g_hi.w};
      auto* sx4 = reinterpret_cast<uint4*>(sx);
      for (int piece = tid; piece < kAttnWindow * kAttnHidden / 8; piece += kAttnThreads) {
        const int m = piece / (kAttnHidden / 8);
        const uint4 input = sx4[piece];
        const unsigned words[4] = {input.x, input.y, input.z, input.w};
        uint4 output;
        auto* out_words = reinterpret_cast<unsigned*>(&output);

#pragma unroll
        for (int w = 0; w < 4; ++w) {
          const float2 value = __bfloat1622float2(
              *reinterpret_cast<const __nv_bfloat162*>(words + w));
          // The two multiplies and BF16 rounding retain their original order.
          const __nv_bfloat162 normalized = __floats2bfloat162_rn(
              float(__float2bfloat16_rn(value.x * sred[m])) * gamma[2 * w],
              float(__float2bfloat16_rn(value.y * sred[m])) * gamma[2 * w + 1]);
          out_words[w] = *reinterpret_cast<const unsigned*>(&normalized);
        }
        *reinterpret_cast<uint4*>(norm_queries + m * kNormQueryStride + k) = output;
        if constexpr (Sliding && kExportDiagnostics) {
          if (cta == 0) reinterpret_cast<uint4*>(args.x_norm)[piece] = output;
        }
      }
      __syncthreads();

      // All raw input reads finished at the preceding CTA barrier. Reuse
      // that input arena for MMA partials; normalized queries live in V scratch.
      naivert::attn_rows::project_strided_rows_mma<kAttnWindow, kAttnHidden, 16, 1, kAttnThreads / 32, kNormQueryStride>(
          0, kProjCols / 16,
          [&](int column) {
            using Row = naivert::attn_rows::K32PackedRow;
            const auto* weight = column < kAttnQRows ? args.q_w :
                                 column < kQkRaw ? args.k_w : args.v_w;
            const int local = column < kAttnQRows ? column :
                              column < kQkRaw ? column - kAttnQRows : column - kQkRaw;
            return Row{weight + std::size_t(local / Row::Rows) * (Row::Rows * kAttnHidden) +
                       (local % Row::Rows) * Row::Page};
          }, norm_queries, reinterpret_cast<float*>(smem_raw),
          [&](int column, int row, float value) {
            const auto out = __float2bfloat16_rn(value);
            if (column < kQkRaw) args.qk_raw[std::size_t(row) * kQkRaw + column] = out;
            else args.v_blk[std::size_t(row) * kAttnKvRows + column - kQkRaw] = out;
          });
    }

    constexpr int kProjectionTilesPerHead = kAttnHeadDim / 16;
    const int projection_owner = cta < kProjCols / 16 ? cta / kProjectionTilesPerHead : -1;
    // QKV producers return from project_rows_mma's final CTA barrier after
    // all emit stores. Nonproducers have no raw output to publish.
    if constexpr(Requests == 1) attn_publish_ready<true, true>(args, projection_owner);
    else attn_publish_ready<true,true>(args,-1); // every consumer must acquire this replay reset
    attn_wait_ready(args, [&] {
      // Observe all three monotone counters before combining their predicates.
      // This permits independent loads instead of short-circuit load branches.
      const bool q_ready = attn_ready_count(args, cta_head) == kProjectionTilesPerHead;
      const bool k_ready = attn_ready_count(args, kReadyK) == kProjectionTilesPerHead;
      const bool v_ready = attn_ready_count(args, kReadyV) == kProjectionTilesPerHead;
      return q_ready & k_ready & v_ready;
    }, 0xD310u + args.generation_salt);

    // ================================================ phase 2: head norm + RoPE, then the scores
    auto* sq = reinterpret_cast<__nv_bfloat16*>(smem_raw + kAttnWindow * kQkRaw * 2);   // [8][520]
    auto* sk = sq + kAttnWindow * kSharedQStride;                                      // [8][136]
    // The acquired Q/K producers are immutable for this phase. Each head
    // warp loads its own four BF16 values directly, with no shared staging.

    // Distribute the 16 independent head tasks over all 16 warps. A task is 128 elements held four to a lane, which is what makes
    // RoPE's element e <-> e + 64 partner exactly lane l <-> l ^ 16.
  #pragma unroll
    for (int task = warp; task < kHeadTasks;
         task += kAttnThreads / 32) {
      const bool is_k = task >= kAttnWindow;
      const int m = is_k ? task - kAttnWindow : task;
      const int head = is_k ? 0 : cta_head;
      const __nv_bfloat16* src = args.qk_raw + std::size_t(m) * kQkRaw +
                                 (is_k ? kAttnQRows : head * kAttnHeadDim);
      const float* gamma = is_k ? args.k_norm : args.q_norm;
      __nv_bfloat16* dst = is_k ? (sk + std::size_t(m) * kSharedKStride)
                                : (sq + std::size_t(m) * kSharedQStride + head * kAttnHeadDim);

      const int e0 = lane * kPerLane;
      static_assert(kPerLane == 4);
      const uint2 raw = *reinterpret_cast<const uint2*>(src + e0);
      const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
      const float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
      const float v[kPerLane] = {lo.x, lo.y, hi.x, hi.y};
      float ssq = 0.0f;
  #pragma unroll
      for (int j = 0; j < kPerLane; ++j) {
        ssq = fmaf(v[j], v[j], ssq);          // bf16 x bf16 is exact in fp32, so this is a + b*b
      }
  #pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        ssq += __shfl_xor_sync(0xffffffffu, ssq, off);
      }
      const float scale = rsqrtf(ssq / float(kAttnHeadDim) + rms_epsilon);
      // Round once here: `_head_rmsnorm` writes bf16 and `_rope_dev` reads that bf16 back, so RoPE's
      // input -- and its half-split partner -- are the ROUNDED values, not the fp32 ones.
      float f[kPerLane];
  #pragma unroll
      for (int j = 0; j < kPerLane; ++j) {
        f[j] = __bfloat162float(__float2bfloat16_rn(float(__float2bfloat16_rn(v[j] * scale)) * gamma[e0 + j]));
      }
      const std::size_t rp = std::size_t(m) * kAttnHeadDim + e0;
  #pragma unroll
      for (int j = 0; j < kPerLane; ++j) {
        const float partner = __shfl_xor_sync(0xffffffffu, f[j], 16);
        const float rot = (lane < 16) ? -partner : partner;
        // Not fmaf: the stand-in multiplies twice and adds once, and contraction would move a ulp.
        const float roped = __fadd_rn(float(__float2bfloat16_rn(__fmul_rn(f[j], args.rope_cos[rp + j]))),
                                      float(__float2bfloat16_rn(__fmul_rn(rot, args.rope_sin[rp + j]))));
        dst[e0 + j] = __float2bfloat16_rn(roped);
      }
    }
    __syncthreads();

    // Each head's first split publishes its query diagnostics. Every CTA
    // computes only this head's eight Q norms and the eight fresh K norms.
    if (kExportDiagnostics && cta % kAttnSplits == 0) {
      for (int i = tid; i < kAttnWindow * kAttnHeadDim; i += kAttnThreads) {
        const int row = i / kAttnHeadDim, dim = i % kAttnHeadDim;
        args.q_out[row * kAttnQRows + cta_head * kAttnHeadDim + dim] =
            sq[row * kSharedQStride + cta_head * kAttnHeadDim + dim];
      }
    }
    if (cta == 0) { // Preserve speculative KV output for the FastLLM caller.
      for (int i = tid; i < kAttnWindow * kAttnHeadDim; i += kAttnThreads) {
        args.k_blk[i] = sk[(i / kAttnHeadDim) * kSharedKStride + i % kAttnHeadDim];
      }
    }

    // The padded query rows must not alias the MMA partials: a faster
    // warp can publish its partials while another is still reading queries.
    constexpr int kQueryOffset = kPvTileOffset + (kAttnThreads / 32) * 128 * sizeof(float);
    static_assert(kQueryOffset + kAttnWindow * kSharedKStride * sizeof(__nv_bfloat16) <= kAttnSmem);
    auto* queries = reinterpret_cast<__nv_bfloat16*>(smem_raw + kQueryOffset);
    if (tid < kAttnWindow * kAttnHeadDim / 8) {
      const int row = tid / (kAttnHeadDim / 8), k = tid % (kAttnHeadDim / 8) * 8;
      *reinterpret_cast<uint4*>(queries + row * kSharedKStride + k) =
          *reinterpret_cast<const uint4*>(sq + row * kSharedQStride + cta_head * kAttnHeadDim + k);
    }
    auto* local_scores = reinterpret_cast<float*>(smem_raw);
    auto* tile_scores = reinterpret_cast<float*>(smem_raw + 32768);
    float* running_max = tile_scores + kAttnWindow * kScoreStride;
    float* running_den = running_max + kAttnWindow;
    float* rescale = running_den + kAttnWindow;
    if (tid < kAttnWindow) {
      running_max[tid] = -INFINITY;
      running_den[tid] = 0.0f;
    }
    __syncthreads();

    const int key_split = cta % kAttnSplits;
    constexpr int kTilesPerBatch = 8;
    constexpr int kKeyStride = kAttnSplits * kSplitTile;
    const int key_tiles = (ncache - key_split * kSplitTile + kKeyStride - 1) / kKeyStride;
    // The fresh keys are already normalized/rotated in this CTA's shared memory.
    // Repeat them in the unused second half of M; only eight key rows are stored.
    if (key_split == 0 && warp == 0) {
      const int key = lane >> 2;
      const int query_pair = lane & 3;
      float dot[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int k0 = 0; k0 < kAttnHeadDim; k0 += 32) {
        const int k = k0 + query_pair * 8;
        const uint4 a = *reinterpret_cast<const uint4*>(sk + key * kSharedKStride + k);
        const uint4 b = *reinterpret_cast<const uint4*>(queries + key * kSharedKStride + k);
        naivert::attn_rows::mma_bf16_16816(dot, a.x, a.x, a.y, a.y, b.x, b.y);
        naivert::attn_rows::mma_bf16_16816(dot, a.z, a.z, a.w, a.w, b.z, b.w);
      }
      local_scores[(kMaxSplitKeys + key) * kAttnWindow + 2 * query_pair] =
          score_value(dot[0], 2 * query_pair, key, true);
      local_scores[(kMaxSplitKeys + key) * kAttnWindow + 2 * query_pair + 1] =
          score_value(dot[1], 2 * query_pair + 1, key, true);
    }

    // Cache MMA's existing final CTA barrier publishes the fresh scores too.
    // Split zero always has a cache tile (bindings require ncache > 0).
    // Empty nonzero splits produce neither cached nor fresh scores.
    for (int batch = 0; batch < key_tiles; batch += kTilesPerBatch) {
      const int tiles_in_batch = min(kTilesPerBatch, key_tiles - batch);
      auto column = [&](int local) {
        return key_split * kSplitTile + (batch + local / kSplitTile) * kKeyStride + local % kSplitTile;
      };
      // Cache scores do not alias queries or fresh K; direct output is safe.
      naivert::attn_rows::project_strided_rows_mma<kAttnWindow, kAttnHeadDim, 1, kTilesPerBatch, kAttnThreads / 32, kSharedKStride, true>(
          0, tiles_in_batch,
          [&](int local) {
            const int key = min(column(local), ncache - 1);
            return args.kv_k + cached_position(key) * kAttnHeadDim;
          }, queries, mma_partials,
          [&](int local, int row, float value) {
            const int key = column(local);
            if (key < ncache)
              local_scores[(batch * kSplitTile + local) * kAttnWindow + row] =
                  score_value(value, row, key, false);
          }, 0);
    }

    // Correctness-only export. Online attention consumes the CTA-local copy.
    if constexpr (kExportDiagnostics) {
      const int tiles = (ncache + kAttnSplits * kSplitTile - 1 - key_split * kSplitTile) /
                        (kAttnSplits * kSplitTile);
      const int owned_keys = tiles * kSplitTile;
      for (int item = tid; item < owned_keys * kAttnWindow; item += kAttnThreads) {
        const int row = item / owned_keys, local = item % owned_keys;
        const int c = key_split * kSplitTile + (local / kSplitTile) * kAttnSplits * kSplitTile + local % kSplitTile;
        if (c < ncache) args.scores[(std::size_t(cta_head) * kAttnWindow + row) * total + c] =
            local_scores[local * kAttnWindow + row];
      }
      if (key_split == 0 && tid < kAttnWindow * kAttnWindow) {
        const int row = tid / kAttnWindow, key = tid % kAttnWindow;
        args.scores[(std::size_t(cta_head) * kAttnWindow + row) * total + ncache + key] =
            local_scores[(kMaxSplitKeys + key) * kAttnWindow + row];
      }
    }

    float accum[kAttnWindow] = {};
    const int split = cta % kAttnSplits;
    const int stream = tid / kAttnHeadDim;
    const int dimension = tid % kAttnHeadDim;
    const int live = Sliding ? history_count : min(int(args.blk_pos[0]), ncache);
    auto* values = reinterpret_cast<__nv_bfloat16*>(smem_raw + kPvTileOffset);

    // Flatten this split's live cache tiles, followed by its fresh keys.
    // Clip the final cache tile to the allocated cache extent, retaining
    // masked columns inside a live tile just as the original loop does.
    constexpr int cache_stride = kAttnSplits * kSplitTile;
    const int first_cache = split * kSplitTile;
    const int live_tiles = live > first_cache
        ? (live - first_cache + cache_stride - 1) / cache_stride : 0;
    const int cache_count = live_tiles == 0 ? 0 : (live_tiles - 1) * kSplitTile +
        min(kSplitTile, ncache - (first_cache + (live_tiles - 1) * cache_stride));
    const int pv_count = cache_count + (split == 0 ? kAttnWindow : 0);
    for (int first = 0; first < pv_count; first += kPvBatchKeys) {
      const int count = min(kPvBatchKeys, pv_count - first);
      if (tid < kAttnWindow * kPvBatchKeys) {
        const int row = tid % kAttnWindow, column = tid / kAttnWindow;
        const int key = first + column;
        float score = -INFINITY;
        if (column < count) {
          const bool cached = key < cache_count;
          const int physical = cached
              ? first_cache + (key / kSplitTile) * cache_stride + key % kSplitTile
              : ncache + key - cache_count;
          const int local = cached ? key : kMaxSplitKeys + key - cache_count;
          if constexpr (Sliding) {
            score = local_scores[local * kAttnWindow + row];
          } else if (args.key_add[physical] == 0.0f) {
            score = local_scores[local * kAttnWindow + row];
          }
        }
        tile_scores[row * kScoreStride + column] = score;
      }
      __syncthreads();
      if (warp < kAttnWindow) {
        const float score0 = tile_scores[warp * kScoreStride + lane];
        const float score1 = tile_scores[warp * kScoreStride + lane + 32];
        float maximum = fmaxf(score0, score1);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
          maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, off));
        const float old_max = running_max[warp];
        maximum = fmaxf(old_max, maximum);
        const float alpha = old_max == -INFINITY ? 0.0f : expf(old_max - maximum);
        const float probability0 = score0 == -INFINITY ? 0.0f : expf(score0 - maximum);
        const float probability1 = score1 == -INFINITY ? 0.0f : expf(score1 - maximum);
        const float denominator = naivert::attn_rows::warp_sum_f32(probability0 + probability1);
        tile_scores[warp * kScoreStride + lane] = probability0;
        tile_scores[warp * kScoreStride + lane + 32] = probability1;
        if (lane == 0) {
          running_max[warp] = maximum;
          running_den[warp] = __fadd_rn(__fmul_rn(running_den[warp], alpha), denominator);
          rescale[warp] = alpha;
        }
      }
      for (int piece = tid; piece < count * kAttnHeadDim / 8; piece += kAttnThreads) {
        const int key = first + piece / (kAttnHeadDim / 8);
        const int dimension8 = (piece % (kAttnHeadDim / 8)) * 8;
        const bool cached = key < cache_count;
        const int physical = cached
            ? first_cache + (key / kSplitTile) * cache_stride + key % kSplitTile
            : key - cache_count;
        const auto* source = cached
            ? args.kv_v + cached_position(physical) * kAttnHeadDim
            : args.v_blk + physical * kAttnHeadDim;
        reinterpret_cast<uint4*>(values)[piece] =
            *reinterpret_cast<const uint4*>(source + dimension8);
      }
      __syncthreads();
#pragma unroll
      for (int row = 0; row < kAttnWindow; ++row) accum[row] *= rescale[row];
      for (int key = stream; key < count; key += 4) {
        const float value = attn_bf16_at(values, key * kAttnHeadDim + dimension);
#pragma unroll
        for (int row = 0; row < kAttnWindow; ++row)
          accum[row] = fmaf(tile_scores[row * kScoreStride + key], value, accum[row]);
      }
      __syncthreads();
    }
    auto* sums = reinterpret_cast<float*>(values);
#pragma unroll
    for (int row = 0; row < kAttnWindow; ++row)
      sums[row * kAttnThreads + tid] = accum[row];
    __syncthreads();
    if (tid < kAttnHeadDim) {
#pragma unroll
      for (int row = 0; row < kAttnWindow; ++row) {
        const float* src = sums + row * kAttnThreads + tid;
        args.split_scratch[(cta * kAttnWindow + row) * kSplitRowStride + tid] =
            __fadd_rn(__fadd_rn(__fadd_rn(src[0], src[kAttnHeadDim]),
                                         src[2 * kAttnHeadDim]), src[3 * kAttnHeadDim]);
      }
    }
    if (tid < kAttnWindow) {
      float* out = args.split_scratch + (cta * kAttnWindow + tid) * kSplitRowStride;
      out[kAttnHeadDim] = running_max[tid];
      out[kAttnHeadDim + 1] = running_den[tid];
    }
    attn_publish_ready(args, kReadyPv + cta_head);
    // The publication barrier ended all generic accesses to the old scores.
    auto* o_ready = reinterpret_cast<std::uint64_t*>(smem_raw + kOReadyOffset);
    auto* o_failed = reinterpret_cast<unsigned*>(smem_raw + kOFailedOffset);
    auto* o_weight = reinterpret_cast<__nv_bfloat16*>(smem_raw + kOWeightOffset);
    if (tid == 0) {
      naivert::Barrier::init(o_ready, 1u);
      *o_failed = 0u;
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      naivert::Barrier::expect_transaction(o_ready, kOWeightBytes);
      OWeightTransport::cp_async_bulk_evict_first(o_weight,
          args.o_w + std::size_t(output_cta * kColsPerCtaD3) * kAttnQRows, kOWeightBytes, o_ready);
      naivert::Barrier::arrive(o_ready);
    }


    // Merge independently normalized splits for one (head, query) per CTA.
    if (cta < kAttnRowTasks) {
      const int head = cta / kAttnWindow, row = cta % kAttnWindow;
      attn_wait_ready(args, [&] {
        return attn_ready_count(args, kReadyPv + head) == kAttnSplits;
      }, 0xD320u + args.generation_salt);
      // Each of the four output warps computes the identical split weights.
      // Keep both XOR reduction trees and ascending split FMA order unchanged.
      if (tid < kAttnHeadDim) {
        const int lane = tid & 31;
        const float* partial = args.split_scratch +
            ((head * kAttnSplits + lane) * kAttnWindow + row) * kSplitRowStride;
        const float local_max = lane < kAttnSplits ? partial[kAttnHeadDim] : -INFINITY;
        float maximum = local_max;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
          maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, off));
        const float scale = local_max == -INFINITY ? 0.0f : expf(local_max - maximum);
        float denominator = lane < kAttnSplits ? partial[kAttnHeadDim + 1] * scale : 0.f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
          denominator += __shfl_xor_sync(0xffffffffu, denominator, off);
        denominator = __shfl_sync(0xffffffffu, denominator, 0);
        const float* base = args.split_scratch +
            (head * kAttnSplits * kAttnWindow + row) * kSplitRowStride + tid;
        float value = 0.0f;
        for (int first = 0; first < kAttnSplits; first += 4) {
          float values[4];
#pragma unroll
          for (int s = 0; s < 4; ++s)
            values[s] = base[(first + s) * kAttnWindow * kSplitRowStride];
#pragma unroll
          for (int s = 0; s < 4; ++s)
            value = fmaf(values[s], __shfl_sync(0xffffffffu, scale, first + s), value);
        }
        args.attn_out[(row * kAttnQHeads + head) * kAttnHeadDim + tid] =
            __float2bfloat16_rn(__fdiv_rn(value, denominator));
      }
    }
    // One nibble per head: eight row producers fit without carrying into
    // another head. Each release RMW publishes the same row payload as before.
    const unsigned merged_increment = cta < kAttnRowTasks
        ? 1u << ((cta / kAttnWindow) * kMergedHeadBits) : 0u;
    attn_publish_ready(args, cta < kAttnRowTasks ? kReadyMerged : -1, merged_increment);

    // ================================================ phase 4: o_proj + all-reduce + residual
    auto* stage = reinterpret_cast<float*>(smem_raw + (Requests == 1 ? kAttnWindow * kSharedQStride * 2 : 32768));  // [8][32]
    if (tid == 0 && !wait_o_weight(o_ready, args)) *o_failed = 1u;
    __syncthreads();

    const int col0 = output_cta * kColsPerCtaD3;
    if (*o_failed == 0u) {
      // Each of the two output tiles has four K splits, one per head. Only
      // that warp's leader acquires its eight merged query rows. Warp sync
      // passes this acquisition to the other lanes before their global loads.
      if constexpr(Requests > 1) {
      if (warp < kColsPerCtaD3 / 16 * 4) {
        const int head=warp%4;
        for(int request=0;request<Requests;++request)if(lane==0)
          attn_acquire_ready(requests[request],[&]{return ((attn_ready_count(requests[request],kReadyMerged)>>(head*kMergedHeadBits))&kMergedHeadMask)==8;},0xD330u+args.generation_salt);
        __syncwarp();
      }
      naivert::attn_rows::project_strided_rows_mma<8*Requests,512,4,kColsPerCtaD3/16,16,512,false,true>(
        0,4096/16,[&](int column){
          using Row=naivert::attn_rows::K32PackedRow;int local=column-col0;
          return Row{o_weight+std::size_t(local/16)*16*512+(local%16)*32};
        },args.attn_out,mma_partials,[&](int column,int row,float value){stage[row*kColsPerCtaD3+column-col0]=value;},
        -1,[&](int row,int k){return __ldcg(reinterpret_cast<const uint4*>(requests[row/8].attn_out+(row%8)*512+k));});
      } else {
      if (warp < kColsPerCtaD3 / 16 * 4) {
        const int head = warp % kAttnQHeads;
        if (lane == 0) attn_acquire_ready(args, [&] {
          return ((attn_ready_count(args, kReadyMerged) >> (head * kMergedHeadBits)) &
                  kMergedHeadMask) == kAttnWindow;
        }, 0xD330u + args.generation_salt);
        __syncwarp();
      }
    naivert::attn_rows::project_strided_rows_mma<kAttnWindow, kAttnQRows, 4, kColsPerCtaD3 / 16, kAttnThreads / 32, kAttnQRows, false, true>(
        0, kAttnHidden / 16,
        [&](int column) {
          using Row = naivert::attn_rows::K32PackedRow;
          const int local = column - col0;
          return Row{o_weight + std::size_t(local / Row::Rows) * Row::Rows * kAttnQRows +
                     (local % Row::Rows) * Row::Page};
        },
        args.attn_out, mma_partials,
        [&](int column, int row, float value) {
          stage[row * kColsPerCtaD3 + column - col0] = value;
        });
      }
    } else {
      // Do not consume or overwrite an unfinished DMA destination.
      for (int i = tid; i < Requests * kAttnWindow * kColsPerCtaD3; i += kAttnThreads)
        stage[i] = 0.0f;
      __syncthreads();
    }

    if constexpr(Requests > 1) {
      for(int i=tid;i<Requests*8*kColsPerCtaD3;i+=kAttnThreads)
        stage[i]=__bfloat162float(__float2bfloat16_rn(stage[i]));
      __syncthreads();
      // Publish all requests before waiting for any peer. Mailbox layouts and
      // rank accumulation order remain the same as the 128-CTA scalar path.
      for(int request=0;request<Requests;++request)if(tid<8*kPairsD3) {
        const auto &a=requests[request];const int row=tid/kPairsD3,pair=tid%kPairsD3;
        const auto mailbox=naivert::TileMailbox<AttnLayout>::make(a.mailbox_peers,a.rank,
          naivert::mailbox_generation(unsigned(*a.replay_generation),a.generation_salt),std::size_t(row)*naivert::kMailboxSlabStride);
        const float *v=stage+(request*8+row)*kColsPerCtaD3+pair*4;
        mailbox.publish_pair(output_cta,pair*2,naivert::pack_bf16x2(v[0],v[1]),naivert::pack_bf16x2(v[2],v[3]));
      }
      for(int request=0;request<Requests;++request)if(tid<8*kPairsD3) {
        const auto &a=requests[request];const int row=tid/kPairsD3,pair=tid%kPairsD3;
        const auto mailbox=naivert::TileMailbox<AttnLayout>::make(a.mailbox_peers,a.rank,
          naivert::mailbox_generation(unsigned(*a.replay_generation),a.generation_salt),std::size_t(row)*naivert::kMailboxSlabStride);
        const float *v=stage+(request*8+row)*kColsPerCtaD3+pair*4;
        auto red=naivert::Bf16x4Acc::zero();
        naivert::accumulate_prefetched_rank_order(red,mailbox,naivert::pack_bf16x2(v[0],v[1]),naivert::pack_bf16x2(v[2],v[3]),output_cta,pair*2,[&](int slot){
          return mailbox.wait_pair_bounded<true>(slot,output_cta,pair*2,naivert::MailboxWatchdog{a.watchdog_limit_ns,a.watchdog_flag},a.generation_salt);
        });
        const float reduced[4]={red.v0,red.v1,red.v2,red.v3};
        for(int i=0;i<4;++i){int at=row*4096+col0+pair*4+i;
          a.hid_a[at]=__float2bfloat16_rn(__bfloat162float(__float2bfloat16_rn(reduced[i]))+__bfloat162float(a.resid_in[at]));}
      }
    } else {
    // Publish this rank's partial (bf16, exactly what the stand-in's `o_partial` holds).
    if (tid < kAttnWindow * kColsPerCtaD3) {
      const int m = tid / kColsPerCtaD3, col = tid % kColsPerCtaD3;
      const __nv_bfloat16 v = __float2bfloat16_rn(stage[m * kColsPerCtaD3 + col]);
      if constexpr (kExportDiagnostics)
        args.o_partial[std::size_t(m) * kAttnHidden + col0 + col] = v;
      stage[m * kColsPerCtaD3 + col] = __bfloat162float(v);   // the reduce sums the ROUNDED partials
    }
    __syncthreads();

    const unsigned gen = naivert::mailbox_generation(
        static_cast<unsigned>(*args.replay_generation), args.generation_salt);

    if (tid < kAttnWindow * kPairsD3) {
      const int m = tid / kPairsD3;
      const int pair = tid % kPairsD3;
      const int qword = pair * 2;
      const auto mailbox = naivert::TileMailbox<AttnLayout>::make(
          args.mailbox_peers, args.rank, gen, std::size_t(m) * naivert::kMailboxSlabStride);
      const float* v = stage + m * kColsPerCtaD3 + pair * 4;
      const std::uint32_t p01 = naivert::pack_bf16x2(v[0], v[1]);
      const std::uint32_t p23 = naivert::pack_bf16x2(v[2], v[3]);
      mailbox.publish_pair(cta, qword, p01, p23);

      const naivert::MailboxWatchdog watchdog{args.watchdog_limit_ns, args.watchdog_flag};
      naivert::Bf16x4Acc red = naivert::Bf16x4Acc::zero();
      naivert::accumulate_prefetched_rank_order(red, mailbox, p01, p23, cta, qword, [&](int slot) {
        return mailbox.wait_pair_bounded<true>(slot, cta, qword, watchdog,
                                               args.generation_salt);
      });

      // Round once, then add the residual and round again -- the stand-in reduces into a bf16
      // `hid_a` and only then does `add_(resid[L])`, so both roundings are real.
      const float r[4] = {red.v0, red.v1, red.v2, red.v3};
  #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const std::size_t idx = std::size_t(m) * kAttnHidden + col0 + pair * 4 + i;
        const float reduced = __bfloat162float(__float2bfloat16_rn(r[i]));
        args.hid_a[idx] = __float2bfloat16_rn(reduced + attn_bf16_at(args.resid_in, int(idx)));
      }
    }
    }
  }
};
using AttnBody = AttnRowsBody<>;
}  // namespace fastllm_draft_ops
