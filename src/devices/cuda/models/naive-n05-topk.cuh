#pragma once
#include <cuda_runtime.h>

namespace naive_topk {
// Canonicalize signed zero while retaining the existing Inf/NaN bit ordering.
__device__ inline unsigned OrderedScoreBits(float score) {
    unsigned bits = score == 0.0f ? 0u : __float_as_uint(score);
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}
} // namespace naive_topk

#if !defined(FASTLLM_CUDA_LEGACY_ONLY) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800)
#define FASTLLM_NAIVE_COOPERATIVE_TOPK
#include <cooperative_groups.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>

namespace naive_topk {
constexpr int kTopK = 2048, kCapacity = 4096;
constexpr int kBlocks = 64, kThreads = 256, kBuckets = 256;
constexpr int kPositionBits = 18, kMaxKeys = 1 << kPositionBits, kMinKeys = 8192;
constexpr unsigned kPositionMask = kMaxKeys - 1;

struct State {
    unsigned prefix, maximum;
    int rank, candidates, selected, uniform, done;
};
// Candidate storage comes first so its eight-byte alignment is preserved.
struct Workspace {
    unsigned long long candidates[kCapacity];
    int partials[kBlocks * (kBuckets + 1)], ties[kBlocks];
    State state;
};
__device__ inline unsigned OrderedScoreBits(unsigned bits) {
    return bits;
}
// Fixed K/capacity specialization. Equal scores retain input order.
// A cooperative launch is required: all CTAs must be simultaneously resident.
template <typename Score>
__global__ void Select(const Score *scores, int n, int *partials, int *ties, State *state,
                       unsigned long long *out) {
    constexpr int T = kThreads;
    using Scan = cub::BlockScan<int, T>;
    __shared__ int hist[kBuckets];
    __shared__ unsigned warpMax[kThreads / 32];
    __shared__ typename Scan::TempStorage scan;
    auto grid = cooperative_groups::this_grid();
    int t = threadIdx.x, lane = t % 32;
    int chunk = ((n + int(gridDim.x) * T - 1) / (int(gridDim.x) * T)) * T;
    int begin = blockIdx.x * chunk, end = min(n, begin + chunk);
    if (blockIdx.x == 0 && t == 0) {
        state->selected = 0;
        state->uniform = 0;
    }
    unsigned reference = OrderedScoreBits(scores[0]);
    unsigned prefix = 0, mask = 0;
    int rank = kTopK, candidates = n;
    for (int shift = 24; shift >= 0; shift -= 8) {
        hist[t] = 0;
        __syncthreads();
        unsigned maximum = 0;
        bool different = false;
        for (int base = begin; base < end; base += T) {
            int i = base + t;
            unsigned bits = i < end ? OrderedScoreBits(scores[i]) : 0;
            if (shift == 24) {
                maximum = max(maximum, bits);
                different |= i < end && bits != reference;
            }
            int bucket =
                i < end && (bits & mask) == prefix ? int((bits >> shift) & (kBuckets - 1)) : kBuckets;
            unsigned peers = __match_any_sync(0xffffffff, bucket);
            if (bucket < kBuckets && lane == __ffs(peers) - 1)
                atomicAdd(hist + bucket, __popc(peers));
        }
        __syncthreads();
        partials[blockIdx.x * kBuckets + t] = hist[t];
        if (shift == 24) {
            int changed = __syncthreads_or(different);
            unsigned wm = __reduce_max_sync(0xffffffff, maximum);
            if (lane == 0)
                warpMax[t / 32] = wm;
            __syncthreads();
            if (t == 0) {
                unsigned m = 0;
                for (int w = 0; w < kThreads / 32; ++w)
                    m = max(m, warpMax[w]);
                partials[gridDim.x * kBuckets + blockIdx.x] = m;
                ties[blockIdx.x] = changed;
            }
        }
        grid.sync();
        if (blockIdx.x == 0) {
            if (shift == 24) {
                int changed = __syncthreads_or(t < gridDim.x && ties[t]);
                unsigned m = t < gridDim.x ? (unsigned)partials[gridDim.x * kBuckets + t] : 0;
                unsigned wm = __reduce_max_sync(0xffffffff, m);
                if (lane == 0)
                    warpMax[t / 32] = wm;
                __syncthreads();
                if (t == 0) {
                    unsigned hi = 0;
                    for (int w = 0; w < kThreads / 32; ++w)
                        hi = max(hi, warpMax[w]);
                    state->maximum = hi;
                    if (!changed) {
                        state->uniform = 1;
                        state->selected = kTopK;
                    }
                }
            }
            int bucket = kBuckets - 1 - t, total = 0;
            for (int b = 0; b < gridDim.x; ++b)
                total += partials[b * kBuckets + bucket];
            int before;
            Scan(scan).ExclusiveSum(total, before);
            if (before < rank && before + total >= rank) {
                state->prefix = prefix | (unsigned(bucket) << shift);
                state->rank = rank - before;
                state->candidates = kTopK - (rank - before) + total;
            }
            __syncthreads();
            if (t == 0)
                state->done = state->candidates <= kCapacity || shift == 0 || state->prefix == state->maximum;
            __syncthreads();
            if (state->done && state->candidates > kCapacity) {
                int equal = t < gridDim.x
                                ? partials[t * kBuckets + ((state->prefix >> shift) & (kBuckets - 1))]
                                : 0,
                    offset;
                Scan(scan).ExclusiveSum(equal, offset);
                if (t < gridDim.x)
                    ties[t] = offset;
            }
        }
        grid.sync();
        if (state->uniform)
            return;
        prefix = state->prefix;
        rank = state->rank;
        candidates = state->candidates;
        mask |= (kBuckets - 1u) << shift;
        if (state->done)
            break;
    }
    int tiesBefore = candidates > kCapacity ? ties[blockIdx.x] : 0;
    for (int base = begin; base < end; base += T) {
        int i = base + t;
        unsigned bits = i < end ? OrderedScoreBits(scores[i]) : 0;
        bool take = i < end && bits >= prefix;
        if (candidates > kCapacity) {
            int equal = i < end && bits == prefix, before, total;
            Scan(scan).ExclusiveSum(equal, before, total);
            __syncthreads();
            take = i < end && (bits > prefix || (equal && tiesBefore + before < rank));
            tiesBefore += total;
        }
        unsigned ballot = __ballot_sync(0xffffffff, take);
        if (ballot) {
            int leader = __ffs(ballot) - 1, offset = 0;
            if (lane == leader)
                offset = atomicAdd(&state->selected, __popc(ballot));
            offset = __shfl_sync(0xffffffff, offset, leader);
            // Subtracting the common lower bound loses no information. Eighteen bits
            // encode positions for n<=262144; larger score keys win, then lower indices.
            if (take)
                out[offset + __popc(ballot & ((1u << lane) - 1))] =
                    ((unsigned long long)(bits - prefix) << kPositionBits) | (kPositionMask - (unsigned)i);
        }
    }
}
__global__ void Sort(const unsigned long long *in, const State *state, int *out) {
    if (state->uniform) {
        for (int i = threadIdx.x; i < kTopK; i += blockDim.x)
            out[i] = i;
        return;
    }
    constexpr int T = kThreads, Items = kCapacity / T;
    using BlockSort = cub::BlockRadixSort<unsigned long long, T, Items, cub::NullType, 6>;
    __shared__ typename BlockSort::TempStorage storage;
    unsigned long long keys[Items];
    int n = state->selected;
#pragma unroll
    for (int i = 0; i < Items; ++i) {
        int j = threadIdx.x * Items + i;
        keys[i] = j < n ? in[j] : 0;
    }
    BlockSort(storage).SortDescending(keys, 0, kPositionBits + 32 - __clz(state->maximum - state->prefix));
#pragma unroll
    for (int i = 0; i < Items; ++i) {
        int j = threadIdx.x * Items + i;
        if (j < kTopK)
            out[j] = kPositionMask - ((unsigned)keys[i] & kPositionMask);
    }
}
} // namespace naive_topk
#endif
