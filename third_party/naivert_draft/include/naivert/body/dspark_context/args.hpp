#pragma once

// Modified for FastLLM fused draft execution. See the parent vendor NOTICE
// and UPSTREAM.json for adaptation details and original source hashes.
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
namespace fastllm_draft_ops {
struct FcCtxArgs {
  const __nv_bfloat16* target_hidden;  // [8, 4096]  this rank's slice, already masked
  const __nv_bfloat16* fc;             // [tile16][K/32][row16][K32], original BF16 elements
  const float* hidden_norm;            // [4096] fp32 gamma
  const __nv_bfloat16* ctx_valid;      // [8, 1] 1.0 / 0.0
  __nv_bfloat16* fc_partial;           // [8, 4096] out: this rank's partial, before the reduce
  __nv_bfloat16* ctx_sum;              // [8, 4096] out: reduced and masked, before the norm
  __nv_bfloat16* ctx;                  // [8, 4096] out
  float* ssq;                          // [8, 128] scratch: per-CTA partial sum of squares
  unsigned* barrier;                    // [2] {arrivals, generation}, zeroed once at setup
  const std::uint8_t* mailbox_peers[8];
  // Device-side execution counter, read by every CTA at the start and bumped by ONE thread after
  // the grid barrier. It is not a host counter: the mailbox tag must be different on every
  // execution (a repeat would let a rank match a peer's STALE word and reduce last block's
  // numbers), and a captured graph replays without the host getting a say. Bumping it inside the
  // kernel makes "different every execution" a property of the kernel rather than of whoever
  // remembered to advance a counter before the replay. Every rank runs D1 the same number of
  // times, so the eight counters stay in lockstep, which is what makes the tags match.
  std::int32_t* replay_generation;
  std::uint32_t generation_salt;          // this all-reduce site's tag; see the salt table below
  int rank;
  // Stage 4's bounded spin (see the lockstep paragraph above).  `watchdog_limit_ns == 0` is the
  // unbounded behaviour every pre-Stage-4 caller gets, so a harness that does not set these two
  // runs the exact spin it ran before.  `watchdog_flag` is this rank's own word, not a peer's:
  // each rank reports its OWN stall, and the host reads eight of them.
  unsigned long long watchdog_limit_ns;
  unsigned* watchdog_flag;
  float rms_epsilon;
};
struct CtxKvArgs {
  const void* const* cache_table;
  int table_requests, table_request;
  const __nv_bfloat16* ctx;            // [8, 4096], already all-reduced by D1
  const __nv_bfloat16* k_w[5];   // [128, 4096] row-major, one row per output column
  const __nv_bfloat16* v_w[5];
  const float* k_norm[5];        // [128] fp32 gamma
  const float* rope_cos;               // [max_kv, 128] fp32
  const float* rope_sin;
  const std::int64_t* ctx_pos;         // [8] RoPE position per ctx row
  const std::int64_t* kv_dst;          // [8] destination cache row; padded rows -> the trash row
  __nv_bfloat16* kv_raw;               // [5][2][8][128] scratch handed from phase 1 to phase 2
  __nv_bfloat16* kv_k;                 // [5, max_seq_len + 1, 1, 128]
  __nv_bfloat16* kv_v;
  unsigned* barrier;
  int cache_rows;                      // max_seq_len + 1, the stride between layers of kv_k/kv_v
  float rms_epsilon;
};
static_assert(sizeof(FcCtxArgs) == 176);
static_assert(sizeof(CtxKvArgs) == 216);
struct FcCtxKvArgs { FcCtxArgs fc; CtxKvArgs kv; float* kv_partials; };
static_assert(sizeof(FcCtxKvArgs) == 400);
}  // namespace fastllm_draft_ops
