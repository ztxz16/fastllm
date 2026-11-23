#pragma once

// Modified for FastLLM fused draft execution. See the parent vendor NOTICE
// and UPSTREAM.json for adaptation details and original source hashes.
#include <cuda_bf16.h>
#include <cstdint>
#ifndef NAIVERT_DSPARK_DIAGNOSTICS
#define NAIVERT_DSPARK_DIAGNOSTICS 0
#endif
static_assert(NAIVERT_DSPARK_DIAGNOSTICS == 0 || NAIVERT_DSPARK_DIAGNOSTICS == 1);
namespace fastllm_draft_ops {
struct MlpGeometry {
  static constexpr int Rows = 8;
  static constexpr int Hidden = 4096;
  static constexpr int IntermediatePerRank = 512;
  static constexpr int Ranks = 8;
};
struct MlpArgs {
  const __nv_bfloat16* hid_a;      // [8, 4096] attention output + residual, this layer's input
  const float* post_ln;            // [4096] fp32 gamma
  const __nv_bfloat16* gate_w;     // first half of interleaved gate/up, packed into K32 pages
  const __nv_bfloat16* up_w;       // second half of interleaved gate/up, packed into K32 pages
  const __nv_bfloat16* down_w;     // [4096, 512] packed K32 row tiles
  __nv_bfloat16* mlp_h;            // [8, 512]  out: the SiLU'd intermediate, phase 1 -> phase 2
  __nv_bfloat16* down_partial;     // [8,4096] diagnostic down partial only
  __nv_bfloat16* resid_out;        // [8, 4096] out: resid[L + 1]
  unsigned* barrier;               // [2] {shifted arrivals, generation}; arrivals settle to 0; generation advances
  const std::uint8_t* mailbox_peers[8];
  const std::int32_t* replay_generation;   // read only: D1 owns the bump (see the salt table)
  std::uint32_t generation_salt;
  int rank;
  // A nonzero nanosecond bound reports stalled mailbox/weight waits.
  unsigned long long watchdog_limit_ns;
  unsigned* watchdog_flag;
  float rms_epsilon;
};
static_assert(sizeof(MlpArgs) == 176);
}
