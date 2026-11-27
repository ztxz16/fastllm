#pragma once

// Modified for FastLLM fused draft execution. See the parent vendor NOTICE
// and UPSTREAM.json for adaptation details and original source hashes.
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <cstdint>
namespace fastllm_draft_ops {
constexpr int kHeadVocab = 19072;
struct HeadGrid {
  int ctas;
  int rowStride;
  static constexpr int rowPadding = 2;

  static constexpr HeadGrid forCtas(int ctas) {
    return {ctas, ((kHeadVocab + 16 * ctas - 1) / (16 * ctas)) * 16 + rowPadding};
  }
};
struct HeadArgs {
  const __nv_bfloat16* resid_in;
  const float* norm_gamma;
  const __nv_bfloat16* lm_head;  // head_rows8_weight_shape(), packed during weight initialization
  __nv_bfloat16* norm_out;
  float* logits;
  unsigned long long watchdog_limit_ns;
  unsigned* watchdog_flag;
  float rms_epsilon;
  int first_logit_row;
};
struct MarkovArgs {
  const __nv_bfloat16* markov_w1;   // [152576, 256], full replicated lookup table
  const __nv_bfloat16* markov_w2;   // [19072, 256] row-major, this rank's vocabulary shard
  const float* logits;              // [7, 19072] fp32, D5's output on this rank
  std::int64_t* block_ids;          // [8] in: [0] is the anchor; out: [j + 1] = draft token j
  std::int64_t* draft_tokens;       // [7] out
  std::int64_t* prev_token;         // [1] out: the last step's token
  __nv_bfloat16* lat_r;             // [1, 256] out: THIS rank's masked contribution, last step
  __nv_bfloat16* latent;            // [256]    out: the reduced latent, last step
  __nv_bfloat16* step_bias;         // [19072]  out: last step
  float* step_logits;               // [19072]  out: last step
  float* cand_val;                  // [8] out: every rank's candidate value, last step
  std::int64_t* cand_idx;           // [8] out: every rank's candidate index, last step
  __nv_bfloat16* lat_trace;         // [7, 256]   out: every step's latent
  __nv_bfloat16* bias_trace;        // [7, 19072] out: every step's bias
  float* logit_trace;               // [7, 19072] corrected logits; production p/q input when sampling
  float* cta_val;                   // [2, kMkCtas] alternating partial argmax
  std::int32_t* cta_idx;            // [2, kMkCtas] scratch
  const std::uint8_t* mailbox_peers[8];
  unsigned* barrier;                // {arrivals, generation}
  const std::int32_t* replay_generation;   // read only: D1 owns the bump (see the salt table)
  int rank;
  int apply_markov;                 // 0 = plain argmax over D5's logits (the --no-markov ablation)
  // Stage 4's bounded spin -- see the lockstep paragraph in dspark_fc_ctx.cu and
  // evidence/gate4.md §4.5.  0 keeps the unbounded spin this kernel shipped with.
  unsigned long long watchdog_limit_ns;
  unsigned* watchdog_flag;
  const __nv_bfloat16* bias_table;
  const std::int64_t* sampling_state;  // optional [seed, block]; nullptr = greedy
  const float* sampling_params;        // optional DraftBuffers.sampling_params; [4] = 1/T for the draw (nullptr: compile-time T)
};
static_assert(sizeof(HeadArgs) == 64);
static_assert(sizeof(MarkovArgs) == 264);
}
