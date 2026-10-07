#pragma once
#include <cstdint>

// Cumulative hybrid decode/verifier counters attributed to one CUDA device.
// values[8]: layer calls, all routes, resident routes before dispatch,
// nonresident routes, GPU execution routes, CPU execution routes,
// resident GPU execution routes, prefetched experts.
// Includes CPU-assigned routes; excludes prefetch lookups from residency and
// route totals. Does not include prefill, Qwen4 expert parallel or pure-GPU mode.
// Call between requests, after host inference has finished. Synchronizes the
// device. Take snapshot differences; reading never resets scheduler or cache.
extern "C" bool fastllm_moe_cuda_cache_route_stats(int device, uint64_t *values);
// GLM whole-expert distribution: device count, local slots, occupied local slots,
// local payload bytes, cached routes computed locally, uploaded expert bytes.
// Uses the same snapshot rules as route_stats.
extern "C" bool fastllm_moe_cuda_cache_ep_stats(int device, uint64_t *values);
