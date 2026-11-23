#pragma once
// Sampling temperature T for the speculative round, represented as 1/T. Serving supplies it
// in the per-request parameter block. DSPARK_INV_TEMPERATURE is the fallback for callers
// without that block; extension builds can set it from NAIVERT_DSPARK_INV_TEMPERATURE.
// The same inverse temperature is applied to
//  * the draft's Gumbel draw in D56: argmax(step_logits / T + g), so a proposal is a sample of softmax(step_logits / T);
//  * the target sampler's logit loads: p = top-p(softmax(target_logits / T)), q = softmax(step_logits / T),
// which is exactly the pair the classic p/q rejection test needs. T = 1 is the default and bitwise the old path.
#ifndef DSPARK_INV_TEMPERATURE
#define DSPARK_INV_TEMPERATURE 1.0f
#endif
namespace naivert_dspark {
constexpr float kInvTemperature = DSPARK_INV_TEMPERATURE;
__device__ __forceinline__ float tempered(float logit) {
  if constexpr (kInvTemperature == 1.0f) return logit;
  else return logit * kInvTemperature;
}
// Runtime form: ``inv`` is 1/T from the request's sampling parameter block (index 4). inv == 1
// returns the logit itself, bitwise the compile-time default.
__device__ __forceinline__ float tempered(float logit, float inv) {
  return inv == 1.0f ? logit : logit * inv;
}
}  // namespace naivert_dspark
