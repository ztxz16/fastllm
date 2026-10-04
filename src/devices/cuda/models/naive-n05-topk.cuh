#pragma once
#include <cuda_runtime.h>

#include <cooperative_groups.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>

namespace naive_topk {
// Canonicalize signed zero while retaining the existing Inf/NaN bit ordering.
__device__ inline unsigned OrderedScoreBits(float score) {
    unsigned bits = score == 0.0f ? 0u : __float_as_uint(score);
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}
constexpr int kTopK = 2048, kCapacity = 4096;
constexpr int kMaxBlocks = 64, kThreads = 256, kBuckets = 256;
constexpr int kMaxKeys = 262144, kMinKeys = 8192;

struct State {
    unsigned prefix, maximum;
    int rank, candidates, selected, uniform, done;
};
struct alignas(8) Candidates {
    unsigned keys[kCapacity];
    int positions[kCapacity];
};
struct Workspace {
    Candidates candidates;
    // The histogram tail is reused for block maxima, then candidate counts.
    int partials[kMaxBlocks * (kBuckets + 1)], ties[kMaxBlocks];
    State state;
};
__device__ inline unsigned WarpMax(unsigned value) {
    for (int offset = 16; offset; offset >>= 1)
        value = max(value, __shfl_down_sync(0xffffffff, value, offset));
    return __shfl_sync(0xffffffff, value, 0);
}
__device__ inline unsigned OrderedScoreBits(unsigned bits) {
    return bits;
}
// Fixed K/capacity specialization. Equal scores retain input order.
// A cooperative launch is required: all CTAs must be simultaneously resident.
template <typename Score>
__global__ void Select(const Score *scores, int n, int *partials, int *ties, State *state,
                       Candidates *out) {
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
    int rank = kTopK, candidates = n, greaterCount = 0, blockCount = 0;
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
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
            unsigned peers = __match_any_sync(0xffffffff, bucket);
            if (bucket < kBuckets && lane == __ffs(peers) - 1)
                atomicAdd(hist + bucket, __popc(peers));
#else
            if (bucket < kBuckets) atomicAdd(hist + bucket, 1);
#endif
        }
        __syncthreads();
        partials[blockIdx.x * kBuckets + t] = hist[t];
        if (shift == 24) {
            int changed = __syncthreads_or(different);
            unsigned wm = WarpMax(maximum);
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
                unsigned wm = WarpMax(m);
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
        // Reuse the histogram to count this block's winners. Higher buckets
        // from earlier passes are already included in greaterCount.
        int bucket = (prefix >> shift) & (kBuckets - 1), unused, greater;
        Scan(scan).ExclusiveSum(t > bucket ? hist[t] : 0, unused, greater);
        __syncthreads();
        greaterCount += greater;
        if (state->done) {
            int equal = hist[bucket];
            if (candidates > kCapacity)
                equal = min(equal, max(0, rank - ties[blockIdx.x]));
            blockCount = greaterCount + equal;
            break;
        }
    }
    // Preserve input order during compaction. A stable 32-bit key/value radix
    // sort can then preserve ties without sorting an additional position key.
    auto *outKeys = out->keys;
    auto *outPositions = out->positions;
    auto *counts = partials + gridDim.x * kBuckets;
    if (t == 0) counts[blockIdx.x] = blockCount;
    grid.sync();
    int offset = 0;
    for (int b = 0; b < blockIdx.x; ++b) offset += counts[b];
    if (t == 0 && blockIdx.x == gridDim.x - 1)
        state->selected = offset + blockCount;
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
        int before, total;
        Scan(scan).ExclusiveSum(int(take), before, total);
        __syncthreads();
        if (take) {
            outKeys[offset + before] = bits - prefix;
            outPositions[offset + before] = i;
        }
        offset += total;
    }
}
__global__ void Sort(const Candidates *in, const State *state, int *out) {
    if (state->uniform) {
        for (int i = threadIdx.x; i < kTopK; i += blockDim.x)
            out[i] = i;
        return;
    }
    constexpr int T = kThreads, Items = kCapacity / T;
    using BlockSort = cub::BlockRadixSort<unsigned, T, Items, int, 6>;
    __shared__ typename BlockSort::TempStorage storage;
    unsigned keys[Items];
    int positions[Items];
    int n = state->selected;
    const auto *inKeys = in->keys;
    const auto *inPositions = in->positions;
#pragma unroll
    for (int i = 0; i < Items; ++i) {
        int j = threadIdx.x * Items + i;
        keys[i] = j < n ? inKeys[j] : 0;
        positions[i] = j < n ? inPositions[j] : -1;
    }
    // Valid zero keys precede all padding: stability retains the real items.
    int bits = max(1, 32 - __clz(state->maximum - state->prefix));
    BlockSort(storage).SortDescending(keys, positions, 0, bits);
#pragma unroll
    for (int i = 0; i < Items; ++i) {
        int j = threadIdx.x * Items + i;
        if (j < kTopK) out[j] = positions[i];
    }
}
} // namespace naive_topk
