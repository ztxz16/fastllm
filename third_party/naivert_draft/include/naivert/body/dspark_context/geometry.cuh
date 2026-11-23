#pragma once

// Modified for FastLLM fused draft execution. See the parent vendor NOTICE
// and UPSTREAM.json for adaptation details and original source hashes.
#include "naivert/common/resident_grid_barrier.cuh"
namespace fastllm_draft_ops {
constexpr int kWindow = 8;
constexpr int kHidden = 4096;
constexpr int kThreads = 384;
__device__ inline void grid_barrier(unsigned* state) {
  naivert::resident_grid_barrier(state);
}
}
