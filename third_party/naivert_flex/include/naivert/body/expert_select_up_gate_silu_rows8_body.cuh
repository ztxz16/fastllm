#pragma once
// Eight-row expert_select_up_gate_silu: one launch for the verify rows.
//
// Expert-major work: route all tokens, build the ascending list of distinct
// selected experts, then distribute (expert, output group) items across CTAs.
// Two items stay in flight per CTA, with four panes indexed by item slot and
// page parity. Weight pages are reused across tokens that selected the expert.
// The epilogue writes y[token][slot] only for those tokens.

#include "naivert/common/prepared_mlp_input_layout.hpp"
#include <cute/arch/mma_sm80.hpp>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>

#include <cuda/ptx>

#include "naivert/body/expert_select_up_gate_silu/expert_select_up_gate_silu_args.hpp"
#include "naivert/body/expert_select_up_gate_silu/expert_select_up_gate_silu_traits.cuh"
#include "naivert/pdl/grid_dependency.cuh"
#include "naivert/slot_ring/protocol.cuh"

namespace naivert_mtp_expert_select_rows8 {

namespace es = naivert_replica_expert_select_up_gate_silu;
using es::ExpertSelectUpGateSiluArgs;

constexpr int kRows8Rows = 8;
constexpr int kRows8Ctas = naivert::geometry::kSmCount;
constexpr int kRows8ConsumerThreads = 256;
constexpr int kRows8HelperBaseThread = 256;
constexpr int kRows8HelperWarps = 4;
constexpr int kRows8Experts = es::kRoutedExperts;
constexpr int kRows8Selected = es::kSelectedExperts;
constexpr int kRows8KeptGroups = es::kRouteKeptGroups;
constexpr int kRows8Pages = es::kPages;
constexpr int kRows8Groups = es::kOutputGroups;
constexpr int kRows8Outputs = es::kOutputsPerGroup;
constexpr int kRows8PackedRows = es::kPackedRows;
constexpr int kRows8PageK = es::kPageK;
constexpr int kRows8Hidden = es::kHidden;
constexpr int kRows8MaxDistinct = kRows8Rows * kRows8Selected;
static_assert(es::kSharedExperts == 0, "the eight-row body routes every slot (no shared expert)");
static_assert(kRows8Experts == 256 && kRows8Selected == 8, "warp routing: 8 experts per lane");
static_assert(kRows8Pages == 4, "two pages per parity helper");
static_assert(kRows8Outputs == 16 && kRows8PackedRows == 32, "m16n8k32 fragments");
static_assert(kRows8Hidden % 16 == 0);

// Shared map: barriers and routing scratch below the panes, four panes, fp8 activations after.
constexpr int kRows8BarrierOffset = 0x0000;              // pane full [4]
constexpr int kRows8ReleaseBarrierOffset = 0x0040;       // pane free [4]
constexpr int kRows8ActivationBarrierOffset = 0x0080;    // x rows staged [2]
constexpr int kRows8RouteBarrierOffset = 0x00c0;         // scores + bias staged
constexpr int kRows8RouteReleaseBarrierOffset = 0x00e0;  // routes + item list ready
constexpr int kRows8SelectedOffset = 0x0100;             // int[8 tokens][8 slots]
constexpr int kRows8DistinctOffset = 0x0200;             // int[64]
constexpr int kRows8MaskOffset = 0x0300;                 // unsigned[8]
constexpr int kRows8CountOffset = 0x0340;                // int: distinct experts
constexpr int kRows8BiasOffset = 0x0400;                 // float[256]
constexpr int kRows8SigmoidOffset = 0x0800;              // float[8][256]: scores, then sigmoid
constexpr int kRows8PartialOffset = 0x2800;              // unprepared float[2 slots][4 warps][16][8]
// Prepared item partials occupy [0x800,0x2800). Their former storage holds
// immutable byte [expert][token] selected-slot bitsets (zero if absent).
constexpr int kRows8RoutingSlotOffset = kRows8PartialOffset;
static_assert(kRows8RoutingSlotOffset + kRows8Experts * kRows8Rows
              <= kRows8PartialOffset + 2 * 4 * 16 * 8 * sizeof(float));
// Prepared W8A8 reuses completed routing scratch for eight partial planes per item.
static_assert(kRows8SigmoidOffset + 2 * 8 * 16 * 8 * sizeof(float) == kRows8PartialOffset);
constexpr int kRows8PaneBaseOffset = 0x4000;
constexpr int kRows8PaneStride = es::kPaneStride;   // 0x8020; 0x8400 for mx pages
constexpr int kRows8ActivationOffset = kRows8PaneBaseOffset + 4 * kRows8PaneStride;
constexpr int kRows8ActivationStride = kRows8Hidden + 16;   // token rows 16 B apart mod 128: no bank conflicts on B
constexpr int kRows8ActivationBytes = kRows8Rows * kRows8ActivationStride;
constexpr int kRows8SharedBytes = kRows8ActivationOffset + kRows8ActivationBytes;
static_assert(kRows8PartialOffset + 2 * 4 * 16 * 8 * 4 <= kRows8PaneBaseOffset);
static_assert(kRows8ActivationOffset == (es::kMx ? 0x25000 : (naivert::kExpertActivationFp32 ? 0x24100 : 0x24080)));
static_assert(4 * kRows8Hidden * 2 <= kRows8PaneStride, "four bf16 rows stage in one pane");
static_assert(kRows8SharedBytes <= 232448, "opt-in shared memory");

struct ExpertSelectUpGateSiluRows8Body {
  using Trans64 = naivert::Barrier;

  __device__ static __forceinline__ unsigned decode_fp8_pair(unsigned value) {
    const __half2_raw pair = __nv_cvt_fp8x2_to_halfraw2(
        static_cast<__nv_fp8x2_storage_t>(value), __NV_E4M3);
    unsigned bits;
    memcpy(&bits, &pair, sizeof(bits));
    return bits;
  }

  __device__ static __forceinline__ void mma_with_decoded_activation(
      float& p0, float& p1, float& p2, float& p3, const uint4& a,
      unsigned b0lo, unsigned b1lo, unsigned b0hi, unsigned b1hi) {
    float q0, q1, q2, q3;
    using Mma = cute::SM80_16x8x16_F32F16F16F32_TN;
    Mma::fma(q0, q1, q2, q3, decode_fp8_pair(a.x), decode_fp8_pair(a.y),
             decode_fp8_pair(a.z), decode_fp8_pair(a.w), b0lo, b1lo, 0.f, 0.f, 0.f, 0.f);
    Mma::fma(q0, q1, q2, q3, decode_fp8_pair(a.x >> 16), decode_fp8_pair(a.y >> 16),
             decode_fp8_pair(a.z >> 16), decode_fp8_pair(a.w >> 16), b0hi, b1hi, q0, q1, q2, q3);
    // Match the SM103a compiler lowering: complete each K32 dot, then add it to C.
    p0 = __fadd_rn(p0, q0); p1 = __fadd_rn(p1, q1);
    p2 = __fadd_rn(p2, q2); p3 = __fadd_rn(p3, q3);
  }

  __device__ static __forceinline__ float fast_exp2(float value) {
    float result;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
    return result;
  }

  __device__ static __forceinline__ void warp_reduce_max_pair(float& score, int& index) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
      float other_score = __shfl_down_sync(0xffffffffu, score, delta);
      int other_index = __shfl_down_sync(0xffffffffu, index, delta);
      if (other_score > score || (other_score == score && other_index < index)) {
        score = other_score;
        index = other_index;
      }
    }
  }

  __device__ static __forceinline__ void consumer_sync() {
    naivert::consumer_group_named_sync();
  }

  // The four consumer warps of one item slot.
  __device__ static __forceinline__ void slot_sync(int item_slot) {
    naivert::named_barrier_sync(128u, unsigned(item_slot + 9));
  }

  __device__ static __forceinline__ void mma_m16n8k32_e4m3(
      float& d0, float& d1, float& d2, float& d3, std::uint32_t a0, std::uint32_t a1,
      std::uint32_t a2, std::uint32_t a3, std::uint32_t b0, std::uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }

  __device__ static __forceinline__ void cp_async_bulk_weight_evict_first(
      void* destination, const void* source, std::uint32_t bytes, std::uint64_t* barrier) {
    const std::uint32_t destination_shared =
        std::uint32_t(__cvta_generic_to_shared(destination));
    const std::uint32_t barrier_shared = std::uint32_t(__cvta_generic_to_shared(barrier));
    std::uint64_t cache_policy;
    asm volatile("createpolicy.fractional.L2::evict_first.L2::evict_unchanged.b64 %0, 1.0;"
                 : "=l"(cache_policy));
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1], %2, [%3], %4;"
        :
        : "r"(destination_shared), "l"(source), "r"(bytes), "r"(barrier_shared), "l"(cache_policy)
        : "memory");
  }

  __device__ static __forceinline__ std::uint64_t* barriers(unsigned char* shared) {
    return reinterpret_cast<std::uint64_t*>(shared + kRows8BarrierOffset);
  }
  __device__ static __forceinline__ std::uint64_t* release_barriers(unsigned char* shared) {
    return reinterpret_cast<std::uint64_t*>(shared + kRows8ReleaseBarrierOffset);
  }
  __device__ static __forceinline__ std::uint64_t* activation_barriers(unsigned char* shared) {
    return reinterpret_cast<std::uint64_t*>(shared + kRows8ActivationBarrierOffset);
  }
  __device__ static __forceinline__ std::uint64_t* route_barrier(unsigned char* shared) {
    return reinterpret_cast<std::uint64_t*>(shared + kRows8RouteBarrierOffset);
  }
  __device__ static __forceinline__ std::uint64_t* route_release_barrier(unsigned char* shared) {
    return reinterpret_cast<std::uint64_t*>(shared + kRows8RouteReleaseBarrierOffset);
  }
  __device__ static __forceinline__ int* selected(unsigned char* shared) {
    return reinterpret_cast<int*>(shared + kRows8SelectedOffset);
  }
  __device__ static __forceinline__ int* distinct(unsigned char* shared) {
    return reinterpret_cast<int*>(shared + kRows8DistinctOffset);
  }
  __device__ static __forceinline__ unsigned* expert_mask(unsigned char* shared) {
    return reinterpret_cast<unsigned*>(shared + kRows8MaskOffset);
  }
  __device__ static __forceinline__ int* distinct_count(unsigned char* shared) {
    return reinterpret_cast<int*>(shared + kRows8CountOffset);
  }
  __device__ static __forceinline__ float* bias_scratch(unsigned char* shared) {
    return reinterpret_cast<float*>(shared + kRows8BiasOffset);
  }
  __device__ static __forceinline__ float* sigmoid(unsigned char* shared, int token) {
    return reinterpret_cast<float*>(shared + kRows8SigmoidOffset) + token * kRows8Experts;
  }
  __device__ static __forceinline__ float* partial(unsigned char* shared, int item_slot) {
    return reinterpret_cast<float*>(shared + kRows8PartialOffset) + item_slot * (4 * 16 * 8);
  }
  __device__ static __forceinline__ unsigned char* pane(unsigned char* shared, int index) {
    return shared + kRows8PaneBaseOffset + index * kRows8PaneStride;
  }
  __device__ static __forceinline__ __nv_fp8_e4m3* activation(unsigned char* shared, int token) {
    return reinterpret_cast<__nv_fp8_e4m3*>(shared + kRows8ActivationOffset +
                                            token * kRows8ActivationStride);
  }

  // Work item i = (distinct expert i / 16, group i % 16); CTA c takes items c, c + G, ... and
  // pairs them into slots: the k-th item of slot s is item index 2k + s of the CTA's sequence.
  __device__ static __forceinline__ int item_of(int local, int item_slot) {
    return int(blockIdx.x) + (2 * local + item_slot) * int(gridDim.x);
  }

  // Helper 0 stages the scores and bias, helpers 2 / 3 four bf16 x rows each into panes 2 / 3.
  template <bool kPrepared = false>
  __device__ static void stage_inputs(const ExpertSelectUpGateSiluArgs& args,
                                      unsigned char* shared,
                                      const unsigned char* prepared = nullptr) {
    constexpr std::uint32_t kScoresBytes = std::uint32_t(kRows8Rows * kRows8Experts * 4);
    constexpr std::uint32_t kBiasBytes = std::uint32_t(kRows8Experts * 4);
    constexpr std::uint32_t kRowsBytes = std::uint32_t(4 * kRows8Hidden * 2);
    const int tid = int(threadIdx.x);
    if (tid >= kRows8HelperBaseThread) {
      const int helper = (tid - kRows8HelperBaseThread) >> 5;
      // Prepared helpers can arrive here independently after consumer-only work.
      // A fixed owner remains unique even when the warp is temporarily divergent.
      if (kPrepared ? ((tid & 31) != 0) : !naivert::elect_sync_compat(__activemask())) {
        return;
      }
      if (helper == 0) {
        auto* routing_barrier = route_barrier(shared);
        Trans64::arrive_and_expect_tx(routing_barrier, kScoresBytes + kBiasBytes);
        if (args.seq_len == 8) {
        cuda::ptx::cp_async_bulk(cuda::ptx::space_shared_t{}, cuda::ptx::space_global_t{},
                                 sigmoid(shared, 0), args.scores, kScoresBytes, routing_barrier);
        } else {
          for (int row=0;row<8;++row) {
            int src=row < args.seq_len ? row : args.seq_len-1;
            cuda::ptx::cp_async_bulk(cuda::ptx::space_shared_t{}, cuda::ptx::space_global_t{},
                sigmoid(shared,row),args.scores+src*kRows8Experts,
                std::uint32_t(kRows8Experts*4),routing_barrier);
          }
        }
        cuda::ptx::cp_async_bulk(cuda::ptx::space_shared_t{}, cuda::ptx::space_global_t{},
                                 bias_scratch(shared), args.bias, kBiasBytes, routing_barrier);
        if constexpr (kPrepared) {
          auto* ready = activation_barriers(shared);
          Trans64::arrive_and_expect_tx(ready, naivert::prepared_mlp::kBytes);
          cuda::ptx::cp_async_bulk(cuda::ptx::space_shared_t{}, cuda::ptx::space_global_t{},
              activation(shared,0), prepared, naivert::prepared_mlp::kScaleOffset, ready);
          cuda::ptx::cp_async_bulk(cuda::ptx::space_shared_t{}, cuda::ptx::space_global_t{},
              shared+naivert::prepared_mlp::kScaleScratch, prepared+naivert::prepared_mlp::kScaleOffset,
              naivert::prepared_mlp::kScaleBytes, ready);
        }
      } else if (helper >= 2 && !kPrepared) {
        const int half = helper - 2;
        auto* barrier = activation_barriers(shared) + half;
        Trans64::arrive_and_expect_tx(barrier, kRowsBytes);
        if (args.seq_len == 8) {
        cuda::ptx::cp_async_bulk(cuda::ptx::space_shared_t{}, cuda::ptx::space_global_t{},
                                 pane(shared, helper), args.x + half * 4 * kRows8Hidden, kRowsBytes,
                                 barrier);
        } else {
          for (int row=0;row<4;++row) {
            int src=half*4+row;
            src=src < args.seq_len ? src : args.seq_len-1;
            cuda::ptx::cp_async_bulk(cuda::ptx::space_shared_t{}, cuda::ptx::space_global_t{},
                pane(shared,helper)+row*kRows8Hidden*2,args.x+src*kRows8Hidden,
                std::uint32_t(kRows8Hidden*2),barrier);
          }
        }
      }
      return;
    }
    {
      while (!Trans64::try_wait(route_barrier(shared), 0u)) {
      }
    }
    consumer_sync();
  }

  // The warp's best (score, expert) as the original comparator orders it (score descending,
  // expert ascending) with two hardware reductions: the local best is never NaN (it starts at
  // -FLT_MAX and a NaN never wins a comparison), so an ordered-integer key of the score keeps the
  // float order including infinities; signed zero is canonicalised because +0 and -0 tie.
  __device__ static __forceinline__ int warp_max_expert(float score, int index) {
    const unsigned bits = __float_as_uint(score == 0.0f ? 0.0f : score);
    const unsigned key = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
    const unsigned best_key = __reduce_max_sync(0xffffffffu, key);
    const unsigned eligible = key == best_key ? unsigned(index) : 0xffffffffu;
    return int(__reduce_min_sync(0xffffffffu, eligible));
  }

  // One warp routes one token: sigmoid + bias, group-limited top-8 (the routing rules
  // and tie-breaks), probabilities; CTA 0 publishes the expert ids and probabilities.
  __device__ static void compute_routes_warp(const ExpertSelectUpGateSiluArgs& args,
                                             unsigned char* shared, int token) {
    const int lane = int(threadIdx.x) & 31;
    float* sig = sigmoid(shared, token);
    const float* bias = bias_scratch(shared);
    float select_score[8];
    float probability[8];
#pragma unroll
    for (int item = 0; item < 8; ++item) {
      const int expert = lane + item * 32;
      const float score = sig[expert];
      probability[item] = score;
      select_score[item] = probability[item] + bias[expert];
    }
    __syncwarp();
#pragma unroll
    for (int item = 0; item < 8; ++item) {
      sig[lane + item * 32] = probability[item];
    }
    // With every group kept (Naive N: kRouteKeptGroups == 8) the group ranking cannot change
    // the result, unless a score is non-finite (a top-2 sum of -inf / NaN makes the original walk
    // skip groups): the warp-uniform guard keeps the original ranking for those inputs.
    static_assert(kRows8KeptGroups == 8, "all-groups routing specialization");
    bool normal_group_range = true;
#pragma unroll
    for (int item = 0; item < 8; ++item) {
      normal_group_range = normal_group_range && (select_score[item] > -0x1p126f);
    }
    const bool keep_all = __all_sync(0xffffffffu, normal_group_range);
    unsigned keep = 0xffu;
    if (!keep_all) {
      // Group scores: the top-2 sum of each group of 32 (group = item: expert = lane + 32 item).
      float group_score[8];
#pragma unroll
      for (int item = 0; item < 8; ++item) {
        float best0 = select_score[item];
        int best0_index = lane + item * 32;
        warp_reduce_max_pair(best0, best0_index);
        const int chosen0 = __shfl_sync(0xffffffffu, best0_index, 0);
        float best1 = (lane + item * 32) == chosen0 ? -3.402823466e38f : select_score[item];
        int best1_index = lane + item * 32;
        warp_reduce_max_pair(best1, best1_index);
        group_score[item] = __shfl_sync(0xffffffffu, best0, 0) + __shfl_sync(0xffffffffu, best1, 0);
      }
      keep = 0u;
#pragma unroll
      for (int pick = 0; pick < kRows8KeptGroups; ++pick) {
        int best_group = 0;
        float best_value = -3.402823466e38f;
#pragma unroll
        for (int group = 0; group < 8; ++group) {
          const float value = group_score[group];
          if (!((keep >> group) & 1u) &&
              (value > best_value || (value == best_value && group < best_group))) {
            best_value = value;
            best_group = group;
          }
        }
        keep |= 1u << best_group;
      }
    }
    int* token_selected = selected(shared) + token * kRows8Selected;
    unsigned used_mask = 0u;
#pragma unroll
    for (int pick = 0; pick < kRows8Selected; ++pick) {
      float best_value = -3.402823466e38f;
      int best_expert = 0;
#pragma unroll
      for (int item = 0; item < 8; ++item) {
        const int expert = lane + item * 32;
        if (!((used_mask >> item) & 1u) && ((keep >> item) & 1u)) {
          const float value = select_score[item];
          if (value > best_value || (value == best_value && expert < best_expert)) {
            best_value = value;
            best_expert = expert;
          }
        }
      }
      const int chosen = warp_max_expert(best_value, best_expert);
      if (lane == 0) {
        token_selected[pick] = chosen;
      }
#pragma unroll
      for (int item = 0; item < 8; ++item) {
        if (lane + item * 32 == chosen) {
          used_mask |= 1u << item;
        }
      }
    }
    __syncwarp();
    if (lane < kRows8Selected && int(blockIdx.x) == 0) {
      float sum = 0.0f;
#pragma unroll
      for (int item = 0; item < kRows8Selected; ++item) {
        sum += sig[token_selected[item]];
      }
      const int expert = token_selected[lane];
      args.expert_indices[token * args.selected_experts + lane] = expert;
      args.expert_probs[token * args.selected_experts + lane] =
          sig[expert] / (sum + (naivert::kExpertActivationFp32 ? 1e-20f : 0.0f)) * args.route_scale;
    }
  }

  // The ascending list of distinct selected experts (every CTA builds the same list).
  template <bool kPrepared = false>
  __device__ static void build_item_list(unsigned char* shared) {
    const int tid = int(threadIdx.x);
    if constexpr (kPrepared) {
      // One expert's eight slots per consumer, before the existing first join.
      reinterpret_cast<unsigned long long*>(shared + kRows8RoutingSlotOffset)[tid] = 0ull;
    }
    unsigned* mask = expert_mask(shared);
    if (tid < 8) {
      mask[tid] = 0u;
    }
    consumer_sync();
    if (tid < kRows8MaxDistinct) {
      const int expert = selected(shared)[tid];
      atomicOr(mask + (expert >> 5), 1u << (expert & 31));
      if constexpr (kPrepared) {
        // Repeated experts in non-finite routing set multiple bits safely.
        atomicOr(reinterpret_cast<unsigned*>(shared + kRows8RoutingSlotOffset)
                     + expert * 2 + (tid >> 5), 1u << (tid & 31));
      }
    }
    consumer_sync();
    {
      const int expert = tid;   // 256 consumer threads, one per expert
      const int word = expert >> 5;
      const unsigned bits = mask[word];
      int rank = 0;
#pragma unroll
      for (int w = 0; w < 8; ++w) {
        rank += w < word ? __popc(mask[w]) : 0;
      }
      if ((bits >> (expert & 31)) & 1u) {
        rank += __popc(bits & ((1u << (expert & 31)) - 1u));
        distinct(shared)[rank] = expert;
      }
      if (tid == 0) {
        int total = 0;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
          total += __popc(mask[w]);
        }
        *distinct_count(shared) = total;
      }
    }
    consumer_sync();
  }

  __device__ static float* activation_scales(int token) {
    // Adjacent token scales used by an MMA fragment occupy different banks.
    __shared__ float values[kRows8Rows][kRows8Hidden / 128 + 1];
    return values[token];
  }

  // bf16 rows in panes 2 / 3 -> fp8 activations (all consumer warps), then panes 2 / 3 are freed.
  __device__ static void convert_activations(unsigned char* shared) {
    const int tid = int(threadIdx.x);
    {
      while (!Trans64::try_wait(activation_barriers(shared), 0u)) {
      }
      while (!Trans64::try_wait(activation_barriers(shared) + 1, 0u)) {
      }
    }
    consumer_sync();
    if constexpr (naivert::kExpertActivationFp32) {
      const int token = tid >> 5;
      const auto* source = reinterpret_cast<const __nv_bfloat16*>(pane(shared, 2 + (token >> 2)))
                           + (token & 3) * kRows8Hidden;
      naivert::quantize_block128_row(source, activation(shared, token), activation_scales(token),
                                    kRows8Hidden, tid & 31, 32);
    } else {
    constexpr int kPairs = kRows8Hidden / 2;
    for (int index = tid; index < kRows8Rows * kPairs; index += kRows8ConsumerThreads) {
      const int token = index / kPairs;
      const int pair = index - token * kPairs;
      const auto* source = reinterpret_cast<const __nv_bfloat162*>(
                               pane(shared, 2 + (token >> 2))) +
                           (token & 3) * kPairs;
      reinterpret_cast<__nv_fp8x2_e4m3*>(activation(shared, token))[pair] =
          __nv_fp8x2_e4m3(source[pair]);
    }
    }
    consumer_sync();

  }

  // W8A8 preparation only. Consumers quantize the first half of their token;
  // each helper quantizes the second half of two tokens while routing runs.
  __device__ static void convert_activations_with_helpers(unsigned char* shared) {
    const int warp = int(threadIdx.x) >> 5, lane = int(threadIdx.x) & 31;
    while (!Trans64::try_wait(activation_barriers(shared), 0u)) {}
    while (!Trans64::try_wait(activation_barriers(shared) + 1, 0u)) {}
    constexpr int kHalf = kRows8Hidden / 2;
    static_assert(kHalf % 128 == 0 && kRows8Rows == 2 * kRows8HelperWarps);
    const bool helper = warp >= kRows8ConsumerThreads / 32;
    const int first_token = helper ? 2 * (warp - kRows8ConsumerThreads / 32) : warp;
    const int first_k = helper ? kHalf : 0;
    for (int local = 0; local < (helper ? 2 : 1); ++local) {
      const int token = first_token + local;
      const auto* source = reinterpret_cast<const __nv_bfloat16*>(pane(shared, 2 + (token >> 2)))
                           + (token & 3) * kRows8Hidden + first_k;
      naivert::quantize_block128_row(source, activation(shared, token) + first_k,
                                    activation_scales(token) + first_k / 128, kHalf, lane, 32);
    }
  }

  template <bool kPrepared = false>
  __device__ static void helper_pages(const ExpertSelectUpGateSiluArgs& args,
                                      unsigned char* shared) {
    constexpr int kPaneBytes = es::kPaneVisibleBytes;
    const int tid = int(threadIdx.x);
    if (tid < kRows8HelperBaseThread) {
      return;
    }
    const int helper = (tid - kRows8HelperBaseThread) >> 5;
    if (helper >= kRows8HelperWarps ||
        (kPrepared ? ((tid & 31) != 0) : !naivert::elect_sync_compat(__activemask()))) {
      return;
    }
    const int item_slot = helper >> 1;
    const int page_parity = helper & 1;
    std::uint64_t* barrier = barriers(shared) + helper;
    while (!Trans64::try_wait(route_release_barrier(shared), 0u)) {
    }
    const int items = *distinct_count(shared) * kRows8Groups;
#pragma unroll 1
    for (int local = 0;; ++local) {
      const int item = item_of(local, item_slot);
      if (item >= items) {
        break;
      }
      const int expert = distinct(shared)[item >> 4];
      const int group = item & 15;
      const auto* source_base = args.packed_weights + expert * args.packed_expert_stride +
                                group * args.pages * args.packed_page_stride;
#pragma unroll
      for (int page = page_parity; page < kRows8Pages; page += 2) {
        const int sequence = local * (kRows8Pages / 2) + (page >> 1);
        while (!Trans64::try_wait(release_barriers(shared) + helper,
                                  std::uint32_t(sequence & 1))) {
        }
        Trans64::arrive_and_expect_tx(barrier, std::uint32_t(kPaneBytes));
        cp_async_bulk_weight_evict_first(pane(shared, helper),
                                         source_base + page * args.packed_page_stride,
                                         std::uint32_t(kPaneBytes), barrier);
      }
    }
  }

  __device__ static __forceinline__ void wait_weight_page(unsigned char* shared, int item_slot,
                                                          int local, int page) {
    const int pane_slot = item_slot * 2 + (page & 1);
    const int sequence = local * (kRows8Pages / 2) + (page >> 1);
    while (!Trans64::try_wait(barriers(shared) + pane_slot, std::uint32_t(sequence & 1))) {
    }
  }

  __device__ static __forceinline__ void release_weight_page(unsigned char* shared,
                                                             int item_slot, int page) {
    const int pane_slot = item_slot * 2 + (page & 1);
    Trans64::arrive(release_barriers(shared) + pane_slot);
  }

  // One page of one item: this warp's (branch, K part) over 16 outputs x 8 tokens.
  __device__ static void accumulate_page(unsigned char* shared, int item_slot, int page,
                                         float (&accum)[8]) {
    const int tid = int(threadIdx.x);
    const int warp = tid >> 5;
    const int token_warp = warp & 3;
    const int branch = token_warp >> 1;
    const int k_part = token_warp & 1;
    const int pane_slot = item_slot * 2 + (page & 1);
    const int lane = tid & 31;
    const int lane_group = lane >> 2;
    const int lane_quad = lane & 3;
    const auto* weight_pane = reinterpret_cast<const __nv_fp8_e4m3*>(pane(shared, pane_slot));
    const auto* scales =
        reinterpret_cast<const naivert::ExpertWeightScale*>(weight_pane + kRows8PackedRows * kRows8PageK);
    const auto* branch_packed = weight_pane + branch * kRows8Outputs * kRows8PageK;
    // B column n = token n: this lane's column is lane_group.
    const __nv_fp8_e4m3* act = activation(shared, lane_group) + page * kRows8PageK + lane_quad * 4;
    if constexpr (es::kMx) {
      // Fp8 pages with one ue8m0 scale per (row, K=32 block), applied per block;
      // rows lane_group (p0, p1) and lane_group + 8 (p2, p3).
      const auto* scale_bytes = reinterpret_cast<const unsigned char*>(scales);
#pragma unroll
      for (int scale_group = 0; scale_group < 8; ++scale_group) {
#pragma unroll
        for (int scale_part = 0; scale_part < 2; ++scale_part) {
          const int block = scale_group * 4 + k_part + scale_part * 2;
          const std::uint32_t b0 = *reinterpret_cast<const std::uint32_t*>(act + block * 32);
          const std::uint32_t b1 = *reinterpret_cast<const std::uint32_t*>(act + block * 32 + 16);
          const uint4 a = *reinterpret_cast<const uint4*>(branch_packed + block * 512 +
                                                          lane_group * 64 + lane_quad * 16);
          float p0 = 0.0f;
          float p1 = 0.0f;
          float p2 = 0.0f;
          float p3 = 0.0f;
          mma_m16n8k32_e4m3(p0, p1, p2, p3, a.x, a.y, a.z, a.w, b0, b1);
          const int scale_base = (branch * (kRows8PageK / 32) + block) * kRows8Outputs;
          const float scale_lo = naivert::mx::ue8m0_to_float(scale_bytes[scale_base + lane_group]);
          const float scale_hi = naivert::mx::ue8m0_to_float(scale_bytes[scale_base + lane_group + 8]);
          const int stream = (scale_group & 1) * 4;
          accum[stream + 0] += p0 * scale_lo;
          accum[stream + 1] += p1 * scale_lo;
          accum[stream + 2] += p2 * scale_hi;
          accum[stream + 3] += p3 * scale_hi;
        }
      }
      return;
    }
#pragma unroll
    for (int scale_group = 0; scale_group < 8; ++scale_group) {
      const float scale = naivert::expert_weight_scale(scales[branch * 8 + scale_group]);
      float p0 = 0.0f;
      float p1 = 0.0f;
      float p2 = 0.0f;
      float p3 = 0.0f;
#pragma unroll
      for (int scale_part = 0; scale_part < 2; ++scale_part) {
        const int block = scale_group * 4 + k_part + scale_part * 2;
        const std::uint32_t b0 = *reinterpret_cast<const std::uint32_t*>(act + block * 32);
        const std::uint32_t b1 = *reinterpret_cast<const std::uint32_t*>(act + block * 32 + 16);
        const uint4 a = *reinterpret_cast<const uint4*>(branch_packed + block * 512 +
                                                        lane_group * 64 + lane_quad * 16);
        mma_m16n8k32_e4m3(p0, p1, p2, p3, a.x, a.y, a.z, a.w, b0, b1);
      }
      float scale0 = scale, scale1 = scale;
      if constexpr (naivert::kExpertActivationFp32) {
        scale0 *= activation_scales(lane_quad * 2)[page * 8 + scale_group];
        scale1 *= activation_scales(lane_quad * 2 + 1)[page * 8 + scale_group];
      }
      const int stream = (scale_group & 1) * 4;
      accum[stream + 0] += p0 * scale0;
      accum[stream + 1] += p1 * scale1;
      accum[stream + 2] += p2 * scale0;
      accum[stream + 3] += p3 * scale1;
    }
  }

  // K parts summed through shared, silu(gate) x up, y[token][slot][group] for the tokens that
  // selected this expert (slot = the expert's position in the token's selection).
  __device__ static void finish_item(const ExpertSelectUpGateSiluArgs& args,
                                     unsigned char* shared, int item_slot, int expert,
                                     int group, const float (&accum)[8]) {
    const int tid = int(threadIdx.x);
    const int warp = tid >> 5;
    const int token_warp = warp & 3;
    const int lane = tid & 31;
    const int lane_group = lane >> 2;
    const int lane_quad = lane & 3;
    float* values = partial(shared, item_slot);
    {
      float* mine = values + token_warp * 128;   // [16 outputs][8 tokens]
      mine[lane_group * 8 + lane_quad * 2] = accum[0] + accum[4];
      mine[lane_group * 8 + lane_quad * 2 + 1] = accum[1] + accum[5];
      mine[(lane_group + 8) * 8 + lane_quad * 2] = accum[2] + accum[6];
      mine[(lane_group + 8) * 8 + lane_quad * 2 + 1] = accum[3] + accum[7];
    }
    slot_sync(item_slot);
    {
      const int local = tid - item_slot * 128;   // 0..127
      const int output = local >> 3;
      const int token = local & 7;
      const int index = output * 8 + token;
      const float gate = values[index] + values[128 + index];
      const float up = values[256 + index] + values[384 + index];
      const float sigmoid_gate = 1.0f / (1.0f + fast_exp2(-gate * 1.4426950408889634f));
      const int* token_selected = selected(shared) + token * kRows8Selected;
      int slot = -1;
#pragma unroll
      for (int s = 0; s < kRows8Selected; ++s) {
        slot = token_selected[s] == expert ? s : slot;
      }
      if (slot >= 0) {
        args.y[token * args.output_slots * args.y_slot_stride + slot * args.y_slot_stride +
               group * args.outputs_per_group + output] =
            naivert::expert_activation_from_float(gate * sigmoid_gate * up);
      }
    }
    slot_sync(item_slot);
  }

  // Keep four compute warps per item. Each owns one original K part and
  // scale-group parity, and reuses its decoded activation across gate and up.
  __device__ static void accumulate_page_prepared(unsigned char* shared, int item_slot,
                                                  int page, float (&accum)[8]) {
    const int tid = int(threadIdx.x), lane = tid & 31;
    const int k_part = (tid >> 5) & 1;
    const int parity = ((tid >> 5) >> 1) & 1;
    const int lane_group = lane >> 2, lane_quad = lane & 3;
    const auto* weight = reinterpret_cast<const __nv_fp8_e4m3*>(
        pane(shared, item_slot * 2 + (page & 1)));
    const auto* scales = reinterpret_cast<const naivert::ExpertWeightScale*>(
        weight + kRows8PackedRows * kRows8PageK);
    const auto* act = activation(shared, lane_group) + page * kRows8PageK + lane_quad * 4;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      const int sg = 2 * g + parity;
      float p[2][4] = {};
#pragma unroll
      for (int sp = 0; sp < 2; ++sp) {
        const int block = sg * 4 + k_part + sp * 2;
        const unsigned b0 = *reinterpret_cast<const unsigned*>(act + block * 32);
        const unsigned b1 = *reinterpret_cast<const unsigned*>(act + block * 32 + 16);
        const unsigned b0lo = decode_fp8_pair(b0), b1lo = decode_fp8_pair(b1);
        const unsigned b0hi = decode_fp8_pair(b0 >> 16), b1hi = decode_fp8_pair(b1 >> 16);
#pragma unroll
        for (int branch = 0; branch < 2; ++branch) {
          const uint4 a = *reinterpret_cast<const uint4*>(weight + branch * kRows8Outputs * kRows8PageK +
              block * 512 + lane_group * 64 + lane_quad * 16);
          mma_with_decoded_activation(p[branch][0], p[branch][1], p[branch][2], p[branch][3],
                                      a, b0lo, b1lo, b0hi, b1hi);
        }
      }
      const auto* input_scales = reinterpret_cast<const float*>(
          shared + naivert::prepared_mlp::kScaleScratch) + (page * 8 + sg) * naivert::prepared_mlp::kScaleBlockStride;
      const float as0 = input_scales[lane_quad * 2];
      const float as1 = input_scales[lane_quad * 2 + 1];
#pragma unroll
      for (int branch = 0; branch < 2; ++branch) {
        const float ws = naivert::expert_weight_scale(scales[branch * 8 + sg]);
        const float scale0 = ws * as0, scale1 = ws * as1;
        const int stream = branch * 4;
        accum[stream + 0] += p[branch][0] * scale0;
        accum[stream + 1] += p[branch][1] * scale1;
        accum[stream + 2] += p[branch][2] * scale0;
        accum[stream + 3] += p[branch][3] * scale1;
      }
    }
  }

  __device__ static void finish_item_prepared(const ExpertSelectUpGateSiluArgs& args,
                                             unsigned char* shared, int item_slot, int expert,
                                             int group, const float (&accum)[8]) {
    const int tid = int(threadIdx.x), lane = tid & 31, k_part = (tid >> 5) & 1;
    const int parity = ((tid >> 5) >> 1) & 1;
    const int lane_group = lane >> 2, lane_quad = lane & 3;
    // Every consumer acquired route_release before the item loop. Routing has
    // finished reading [0x800, 0x2800); reuse it as [item][branch][K part][parity][128].
    // The two existing slot joins publish these partials and protect their reuse.
    float* values = reinterpret_cast<float*>(shared + kRows8SigmoidOffset) + item_slot * 8 * 128;
#pragma unroll
    for (int branch = 0; branch < 2; ++branch) {
      float* mine = values + ((branch * 2 + k_part) * 2 + parity) * 128;
      const int base = branch * 4;
      mine[lane_group * 8 + lane_quad * 2] = accum[base];
      mine[lane_group * 8 + lane_quad * 2 + 1] = accum[base + 1];
      mine[(lane_group + 8) * 8 + lane_quad * 2] = accum[base + 2];
      mine[(lane_group + 8) * 8 + lane_quad * 2 + 1] = accum[base + 3];
    }
    slot_sync(item_slot);
    const int local = tid - item_slot * 128;
    const int output = local >> 3, token = local & 7, index = output * 8 + token;
    // Preserve parity addition before K-part addition, including FP32 rounding.
    const float g0 = values[index] + values[128 + index];
    const float g1 = values[256 + index] + values[384 + index];
    const float u0 = values[512 + index] + values[640 + index];
    const float u1 = values[768 + index] + values[896 + index];
    const float gate = g0 + g1, up = u0 + u1;
    const float sigmoid_gate = 1.0f / (1.0f + fast_exp2(-gate * 1.4426950408889634f));
    // The item-list joins and route_release published this immutable lookup.
    // Highest set bit preserves the original last-occurrence search; zero
    // yields -1 and keeps the existing absent-token guard.
    const unsigned bits = shared[kRows8RoutingSlotOffset + expert * 8 + token];
    const int slot = 31 - __clz(bits);
    if (slot >= 0) {
      args.y[token * args.output_slots * args.y_slot_stride + slot * args.y_slot_stride +
             group * args.outputs_per_group + output] =
          naivert::expert_activation_from_float(gate * sigmoid_gate * up);
    }
    slot_sync(item_slot);
  }

  template <bool kWaitForPrimary = false, bool kPrepared = false>
  __device__ static void run(const ExpertSelectUpGateSiluArgs& args, unsigned char* shared,
                             const unsigned char* prepared = nullptr) {
    static_assert(!kPrepared || (naivert::kExpertActivationFp32 && !es::kMx));
    static_assert(naivert::prepared_mlp::kScaleScratch >= kRows8PartialOffset+2*4*16*8*4);
    static_assert(naivert::prepared_mlp::kScaleScratch+naivert::prepared_mlp::kScaleBytes <= kRows8PaneBaseOffset);
    static_assert(naivert::prepared_mlp::kScaleOffset == kRows8ActivationBytes);
    if (int(blockIdx.x) >= kRows8Ctas) {
      return;
    }
    const int tid = int(threadIdx.x);
    if (tid < 4) {
      Trans64::init(barriers(shared) + tid, 1u);
      Trans64::init(release_barriers(shared) + tid, 128u);
      naivert::barrier_arrive_count(release_barriers(shared) + tid, 128u);
      if (tid < 2) {
        // Helpers wait for route_release before reusing the staged input panes.
        Trans64::init(activation_barriers(shared) + tid, 1u);
      }
    }
    if (tid == 0) {
      Trans64::init(route_barrier(shared), 1u);
      Trans64::init(route_release_barrier(shared), kRows8ConsumerThreads +
                     ((naivert::kExpertActivationFp32 && !kPrepared) ? 32 * kRows8HelperWarps : 0));
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
    __syncthreads();

    naivert::pdl::wait_for_primary<kWaitForPrimary>();
    stage_inputs<kPrepared>(args, shared, prepared);
    if constexpr (kPrepared) {
      if (tid >= kRows8HelperBaseThread) {
        helper_pages<true>(args,shared);
        return;
      }
      compute_routes_warp(args, shared, tid >> 5);
      consumer_sync();
      build_item_list<kPrepared>(shared);
      while (!Trans64::try_wait(activation_barriers(shared),0u)) {}
      Trans64::arrive(route_release_barrier(shared));
      while (!Trans64::try_wait(route_release_barrier(shared),0u)) {}
    } else if constexpr (naivert::kExpertActivationFp32) {
      if (tid < kRows8HelperBaseThread) {
        compute_routes_warp(args, shared, tid >> 5);
        consumer_sync();
        build_item_list<kPrepared>(shared);
      }
      // Helpers start once x is ready, independently of consumer routing.
      convert_activations_with_helpers(shared);
      Trans64::arrive(route_release_barrier(shared));
      if (tid >= kRows8HelperBaseThread) {
        helper_pages(args, shared);
        return;
      }
      // This acquire covers the helpers' FP8/scale writes as well as routes.
      while (!Trans64::try_wait(route_release_barrier(shared), 0u)) {}
    } else {
      if (tid >= kRows8HelperBaseThread) {
        helper_pages(args, shared);
        return;
      }
      compute_routes_warp(args, shared, tid >> 5);
      consumer_sync();
      build_item_list<kPrepared>(shared);
      convert_activations(shared);
      // All readers publish completion before a helper reuses an input pane.
      Trans64::arrive(route_release_barrier(shared));
    }

    const int items = *distinct_count(shared) * kRows8Groups;
    const int item_slot = (tid >> 5) >> 2;
    float accum[8];
#pragma unroll 1
    for (int local = 0;; ++local) {
      const int item = item_of(local, item_slot);
      if (item >= items) {
        break;
      }
      const int expert = distinct(shared)[item >> 4];
      const int group = item & 15;
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        accum[index] = 0.0f;
      }
#pragma unroll 1
      for (int page = 0; page < kRows8Pages; ++page) {
        wait_weight_page(shared, item_slot, local, page);
        if constexpr (kPrepared) {
          accumulate_page_prepared(shared, item_slot, page, accum);
        } else {
          accumulate_page(shared, item_slot, page, accum);
        }
        release_weight_page(shared, item_slot, page);
      }
      if constexpr (kPrepared) {
        finish_item_prepared(args, shared, item_slot, expert, group, accum);
      } else {
        finish_item(args, shared, item_slot, expert, group, accum);
      }
    }
  }
};

}  // namespace naivert_mtp_expert_select_rows8
