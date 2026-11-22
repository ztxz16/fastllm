#pragma once

// Modified for FastLLM fused draft execution. See the parent vendor NOTICE
// and UPSTREAM.json for adaptation details and original source hashes.
#include <curand_kernel.h>
#include <cstdint>

namespace fastllm_draft_ops {
// Philox domains: 0 draft Gumbel, 1 accept, 2 bonus, 3 prefill anchor.
// The request owns seed; accept owns block counter. No host RNG on replay.
__device__ inline float sampling_uniform(const std::int64_t* state, unsigned domain,
                                         unsigned row, unsigned token = 0) {
  curandStatePhilox4_32_10_t rng;
  curand_init(static_cast<unsigned long long>(state[0]),
              (static_cast<unsigned long long>(domain) << 32) | token,
              static_cast<unsigned long long>(state[1]) * 8 + row, &rng);
  const double u = (double(curand(&rng)) + 0.5) * 0x1p-32;
  return fminf(float(u), 0x1.fffffep-1f);
}
__device__ inline float sampling_gumbel(const std::int64_t* state, unsigned row,
                                       unsigned token) {
  return -logf(-logf(sampling_uniform(state, 0, row, token)));
}
}
