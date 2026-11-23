#pragma once

// Modified for FastLLM fused draft execution. See the parent vendor NOTICE
// and UPSTREAM.json for adaptation details and original source hashes.
#include <cuda_bf16.h>
#include <cstdint>

namespace fastllm_draft_ops {
struct AttnArgs {
  const __nv_bfloat16* resid_in;   // [8, 4096] resid[L], this layer's input AND its residual
  const float* input_ln;           // [4096] fp32 gamma
  const __nv_bfloat16* q_w;        // [32,128,16,32] K32-packed Q weight
  const __nv_bfloat16* k_w;        // [8,128,16,32] K32-packed weight
  const __nv_bfloat16* v_w;        // [8,128,16,32] K32-packed weight
  const float* q_norm;             // [128] fp32 gamma, inside the head
  const float* k_norm;             // [128]
  const __nv_bfloat16* o_w;        // [tile16][K/32][row16][K32], 4096 x 512 BF16
  const float* rope_cos;           // [max_kv, 128] fp32
  const float* rope_sin;
  const std::int64_t* blk_pos;     // [8] RoPE position per block row
  const float* key_add;            // [total_cols] 0 / -MASK_NEG
  const __nv_bfloat16* kv_k;       // [5, cache_rows, 1, 128]
  const __nv_bfloat16* kv_v;
  __nv_bfloat16* qk_raw;           // [8, 640] out: raw q|k projections, phase 1 -> phase 2
  __nv_bfloat16* q_out;            // [8, 4, 128] out: normed + roped q (CTA 0 publishes)
  __nv_bfloat16* k_blk;            // [8, 1, 128] out: normed + roped k (CTA 0 publishes)
  __nv_bfloat16* v_blk;            // [8, 1, 128] out: raw v, phase 1 -> phase 3
  float* scores;                   // [4,8,total_cols] diagnostic score export only
  __nv_bfloat16* attn_out;         // [8, 4, 128] out: phase 3 -> phase 4
  __nv_bfloat16* o_partial;        // [8, 4096] out: this rank's partial, before the reduce
  __nv_bfloat16* hid_a;            // [8, 4096] out
  unsigned* barrier;               // [2] retained for the caller ABI; unused by D3
  const std::uint8_t* mailbox_peers[8];
  const std::int32_t* replay_generation;   // read only: D1 owns the bump (see the salt table)
  std::uint32_t generation_salt;
  float attn_scale;                // float(1 / sqrt(128)), computed on the host in double
  int layer;
  int cache_rows;                  // kv_k / kv_v row stride between layers (max_seq_len + 1)
  int ncache;                      // cache columns = max_seq_len; the block's eight follow
  int total_cols;                  // ncache + 8 = max_kv
  int padded_cols;                 // total_cols rounded up to a multiple of 256
  int rank;
  // Nonzero bounds readiness, weight and peer waits; caller checks the flag.
  unsigned long long watchdog_limit_ns;
  unsigned* watchdog_flag;
  float* split_scratch;
};
static_assert(sizeof(AttnArgs) == 312, "legacy D3 launch ABI must remain stable");

struct SwaAttnArgs : AttnArgs {
  int valid_rows;
  int window_left;
  float rms_epsilon;
  __nv_bfloat16* x_norm;  // diagnostics only; [8,4096]
};
static_assert(sizeof(SwaAttnArgs) == 336, "SWA D3 launch ABI");

}  // namespace fastllm_draft_ops
