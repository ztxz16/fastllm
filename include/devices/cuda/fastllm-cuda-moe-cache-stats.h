#pragma once
#include <cstdint>

// Cumulative single-token hybrid decode counters on one CUDA device.
// values[8]: layer calls, all routes, resident routes before dispatch,
// nonresident routes, GPU execution routes, CPU execution routes,
// resident GPU execution routes, prefetched experts.
// Includes CPU-assigned routes; excludes prefetch lookups from residency and
// route totals. Does not include prefill, batched verify or pure-GPU mode.
// Call between requests, after host inference has finished. Synchronizes the
// device. Take snapshot differences; reading never resets scheduler or cache.
extern "C" bool fastllm_moe_cuda_cache_route_stats(int device, uint64_t *values);
