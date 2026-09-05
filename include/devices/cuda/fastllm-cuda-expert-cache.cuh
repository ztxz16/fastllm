#pragma once

#include <cuda_runtime.h>
#include <climits>
#include <cstdint>

namespace fastllm {
namespace cuda {

// Payload-independent, GPU-resident LRU metadata. All updates of one view must
// be ordered on a CUDA stream (or by graph dependencies). Initialize maps to -1
// and ages/step/statistics to zero before first use. Query/output storage belongs
// to the caller, as do the expert records and their quantization-specific layout.
struct ExpertCacheView {
    int32_t *keyToSlot;
    int32_t *slotKeys;
    unsigned long long *lastUsed;
    unsigned long long *step;
    unsigned long long *hitCount;
    unsigned long long *missCount;
    int slots;
};

namespace expert_cache_detail {

struct Candidate {
    unsigned long long age;
    int slot;
};

__device__ __forceinline__ Candidate Earlier(Candidate a, Candidate b) {
    return b.age < a.age || (b.age == a.age && b.slot < a.slot) ? b : a;
}

__device__ __forceinline__ Candidate WarpMin(Candidate value) {
    for (int delta = 16; delta; delta >>= 1) {
        Candidate other = {
            __shfl_down_sync(0xffffffffu, value.age, delta),
            __shfl_down_sync(0xffffffffu, value.slot, delta)};
        if ((threadIdx.x & 31) + delta < 32) value = Earlier(value, other);
    }
    return value;
}

template<int Threads>
__device__ __forceinline__ Candidate BlockMin(Candidate value,
                                             Candidate *warps) {
    value = WarpMin(value);
    if ((threadIdx.x & 31) == 0) warps[threadIdx.x / 32] = value;
    __syncthreads();
    if (threadIdx.x < 32) {
        value = threadIdx.x < Threads / 32
            ? warps[threadIdx.x] : Candidate{ULLONG_MAX, INT_MAX};
        value = WarpMin(value);
        if (threadIdx.x == 0) warps[0] = value;
    }
    __syncthreads();
    return warps[0];
}

// CachedItems == 0 scans metadata in tiles instead of imposing a maximum cache
// size or allocating an array proportional to capacity in shared memory.
template<int Threads, int CachedItems, int MaxQueries>
__global__ void LruEnsureKernel(
        ExpertCacheView cache, const int32_t *indices, int keyBase,
        int numExperts, int queries, int32_t *routeSlots,
        int32_t *missingExperts, int32_t *missingSlots, int32_t *numMissing) {
    __shared__ int keys[MaxQueries];
    __shared__ int resident[MaxQueries];
    __shared__ int firstMissing[MaxQueries];
    __shared__ int missingRank[MaxQueries];
    __shared__ int incoming[MaxQueries];
    __shared__ int victims[MaxQueries];
    __shared__ int misses;
    __shared__ unsigned long long tick;
    __shared__ Candidate warps[Threads / 32];
    const int tid = threadIdx.x;
    if (tid == 0) {
        tick = *cache.step + 1;
        *cache.step = tick;
    }
    for (int q = tid; q < queries; q += Threads) {
        int expert = indices[q];
        int key = expert >= 0 && expert < numExperts ? keyBase + expert : -1;
        keys[q] = key;
        int slot = key >= 0 ? cache.keyToSlot[key] : -1;
        resident[q] = slot >= 0 && slot < cache.slots && cache.slotKeys[slot] == key
            ? slot : -1;
    }
    __syncthreads();
    for (int q = tid; q < queries; q += Threads) {
        bool first = keys[q] >= 0;
        for (int p = 0; p < q; ++p) first &= keys[p] != keys[q];
        firstMissing[q] = first && resident[q] < 0;
        // Dedup hit stores as well as copies; no concurrent idempotent writes.
        if (first && resident[q] >= 0) cache.lastUsed[resident[q]] = tick;
        routeSlots[q] = resident[q];
    }
    __syncthreads();
    if (tid == 0) {
        int valid = 0;
        misses = 0;
        for (int q = 0; q < queries; ++q) {
            misses += firstMissing[q];
            valid += keys[q] >= 0;
        }
        *numMissing = misses;
        if (cache.hitCount) *cache.hitCount += valid - misses;
        if (cache.missCount) *cache.missCount += misses;
    }
    for (int q = tid; q < queries; q += Threads) {
        int rank = 0;
        for (int p = 0; p < queries; ++p)
            rank += firstMissing[p] && keys[p] < keys[q];
        missingRank[q] = rank;
        if (firstMissing[q]) incoming[rank] = keys[q];
    }
    __syncthreads();
    if (misses == 0) return;

    unsigned long long ages[CachedItems > 0 ? CachedItems : 1];
    if constexpr (CachedItems > 0) {
        #pragma unroll
        for (int i = 0; i < CachedItems; ++i) {
            int slot = tid + i * Threads;
            unsigned long long age = slot < cache.slots ? cache.lastUsed[slot] : ULLONG_MAX;
            ages[i] = age == tick ? ULLONG_MAX : age;
        }
    }
    for (int m = 0; m < misses; ++m) {
        Candidate best{ULLONG_MAX, INT_MAX};
        if constexpr (CachedItems > 0) {
            #pragma unroll
            for (int i = 0; i < CachedItems; ++i) {
                int slot = tid + i * Threads;
                if (slot < cache.slots) best = Earlier(best, Candidate{ages[i], slot});
            }
        } else {
            for (int slot = tid; slot < cache.slots; slot += Threads) {
                unsigned long long age = cache.lastUsed[slot];
                best = Earlier(best, Candidate{age == tick ? ULLONG_MAX : age, slot});
            }
        }
        best = BlockMin<Threads>(best, warps);
        if (tid == 0) {
            victims[m] = best.slot;
            if constexpr (CachedItems == 0) cache.lastUsed[best.slot] = tick;
        }
        if constexpr (CachedItems > 0) {
            #pragma unroll
            for (int i = 0; i < CachedItems; ++i)
                if (tid + i * Threads == best.slot) ages[i] = ULLONG_MAX;
        }
        __syncthreads();
    }
    for (int m = tid; m < misses; m += Threads) {
        int slot = victims[m];
        int oldKey = cache.slotKeys[slot];
        if (oldKey >= 0) cache.keyToSlot[oldKey] = -1;
        int key = incoming[m];
        cache.slotKeys[slot] = key;
        cache.keyToSlot[key] = slot;
        cache.lastUsed[slot] = tick;
        missingExperts[m] = key - keyBase;
        missingSlots[m] = slot;
    }
    __syncthreads();
    for (int q = tid; q < queries; q += Threads)
        if (keys[q] >= 0 && resident[q] < 0)
            routeSlots[q] = victims[missingRank[q]];
}

template<int Threads, int MaxQueries>
inline void Launch(ExpertCacheView cache, const int32_t *indices,
                   int keyBase, int numExperts, int queries,
                   int32_t *routes, int32_t *experts, int32_t *slots,
                   int32_t *missing, cudaStream_t stream) {
    if (cache.slots <= Threads * 16)
        LruEnsureKernel<Threads, 16, MaxQueries><<<1, Threads, 0, stream>>>(
            cache, indices, keyBase, numExperts, queries, routes, experts, slots, missing);
    else
        LruEnsureKernel<Threads, 0, MaxQueries><<<1, Threads, 0, stream>>>(
            cache, indices, keyBase, numExperts, queries, routes, experts, slots, missing);
}
} // namespace expert_cache_detail

// Select once during cache creation. Dispatch depends on capacity and the
// device's thread limit, never on a model name, data precision, or SM number.
inline int ExpertCacheThreads(int slots, int maxThreadsPerBlock) {
    int threads = slots <= 2048 ? 128 : 256;
    while (threads > maxThreadsPerBlock && threads > 32) threads /= 2;
    return threads;
}

// Caller guarantees 0 <= keyBase, keyBase + numExperts <= map allocation,
// queries <= slots, and output buffers with at least queries entries.
// Larger query batches can instantiate another MaxQueries without changing the
// metadata representation. Invalid expert IDs map to -1; duplicates copy once.
template<int MaxQueries = 64>
inline bool EnsureExpertCache(ExpertCacheView cache, const int32_t *indices,
                              int keyBase, int numExperts, int queries,
                              int32_t *routes, int32_t *experts, int32_t *slots,
                              int32_t *missing, int threads, cudaStream_t stream) {
    if (queries < 1 || queries > MaxQueries || cache.slots < queries ||
        keyBase < 0 || numExperts < 1 || numExperts > INT_MAX - keyBase) return false;
    #define FASTLLM_LRU_LAUNCH(T) \
        expert_cache_detail::Launch<T, MaxQueries>(cache, indices, keyBase, numExperts, \
                                                  queries, routes, experts, slots, missing, stream)
    switch (threads) {
        case 32: FASTLLM_LRU_LAUNCH(32); break;
        case 64: FASTLLM_LRU_LAUNCH(64); break;
        case 128: FASTLLM_LRU_LAUNCH(128); break;
        case 256: FASTLLM_LRU_LAUNCH(256); break;
        default: return false;
    }
    #undef FASTLLM_LRU_LAUNCH
    return true;
}
} // namespace cuda
} // namespace fastllm
