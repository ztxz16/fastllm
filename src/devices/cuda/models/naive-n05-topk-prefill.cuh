#pragma once
#include "naive-n05-topk.cuh"

namespace naive_topk {
// One CTA per live row. Keep the 8K-16K score row in registers across the
// radix passes. Compact in input order so the final stable sort needs only
// score bits, including when many positions tie at the cutoff.
__global__ void SelectSortPrefill16K(const float *scores, int keys, int past,
                                     int *indices, int partRows, int gap) {
    constexpr int threads = 512, items = 32, capacity = 3072, top = 2048;
    using Scan = cub::BlockScan<int, threads>;
    using Sort = cub::BlockRadixSort<unsigned, threads, capacity / threads, int, 4>;
    __shared__ union {
        struct {
            typename Scan::TempStorage scan;
            unsigned histogram[256];
            int counts[threads], positions[capacity];
        } select;
        typename Sort::TempStorage sort;
    } temp;
    auto &histogram = temp.select.histogram;
    auto &counts = temp.select.counts;
    auto &positions = temp.select.positions;
    __shared__ unsigned prefix, maximum;
    __shared__ int rank, candidates;
    const int t = threadIdx.x, row = blockIdx.x;
    int count = past + row + 1 + (partRows && row >= partRows ? gap : 0);
    scores += size_t(row) * keys;
    unsigned values[items], high = 0;
    if (!t) { prefix = 0; rank = top; maximum = 0; }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < items; ++j) {
        int col = j * threads + t;
        values[j] = col < count ? OrderedScoreBits(scores[col]) : 0;
        high = max(high, values[j]);
    }
    high = WarpMax(high);
    if (!(t % 32)) atomicMax(&maximum, high);
    unsigned mask = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        if (t < 256) histogram[t] = 0;
        const unsigned wanted = prefix;
        const int target = rank;
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < items; ++j) {
            unsigned v = values[j];
            int bucket = j * threads + t < count && (v & mask) == wanted ? int((v >> shift) & 255) : 256;
#if __CUDA_ARCH__ >= 700
            unsigned peers = __match_any_sync(0xffffffff, bucket);
            if (bucket < 256 && t % 32 == __ffs(peers) - 1) atomicAdd(histogram + bucket, __popc(peers));
#else
            if (bucket < 256) atomicAdd(histogram + bucket, 1);
#endif
        }
        __syncthreads();
        int n = t < 256 ? histogram[255 - t] : 0, before;
        Scan(temp.select.scan).ExclusiveSum(n, before);
        if (before < target && target <= before + n) {
            prefix = wanted | (unsigned(255 - t) << shift);
            rank = target - before;
            candidates = top - target + before + n;
        }
        __syncthreads();
        mask |= 255u << shift;
        if (candidates <= capacity) break;
    }

    const unsigned cutoff = prefix;
    const bool trimTies = candidates > capacity;
    if (trimTies) {
        // Fully unrolling the rare tie scan with CUDA 12.9 on Blackwell can
        // produce misaligned shared accesses in the subsequent compaction.
        #pragma unroll 1
        for (int j = 0; j < items; ++j) {
            unsigned equal = __ballot_sync(0xffffffff, j * threads + t < count && values[j] == cutoff);
            if (!(t % 32)) counts[j * (threads / 32) + t / 32] = __popc(equal);
        }
        __syncthreads();
        int before;
        Scan(temp.select.scan).ExclusiveSum(counts[t], before);
        __syncthreads();
        counts[t] = before;
        __syncthreads();
    }
    unsigned takeBits = 0;
    #pragma unroll
    for (int j = 0; j < items; ++j) {
        bool live = j * threads + t < count;
        bool take = live && values[j] >= cutoff;
        if (trimTies) {
            unsigned equal = __ballot_sync(0xffffffff, live && values[j] == cutoff);
            if (values[j] == cutoff)
                take &= counts[j * (threads / 32) + t / 32] + __popc(equal & ((1u << (t % 32)) - 1)) < rank;
        }
        takeBits |= unsigned(take) << j;
    }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < items; ++j) {
        unsigned take = __ballot_sync(0xffffffff, (takeBits >> j) & 1u);
        if (!(t % 32)) counts[j * (threads / 32) + t / 32] = __popc(take);
    }
    __syncthreads();
    int before;
    Scan(temp.select.scan).ExclusiveSum(counts[t], before);
    __syncthreads();
    counts[t] = before;
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < items; ++j) {
        unsigned take = __ballot_sync(0xffffffff, (takeBits >> j) & 1u);
        if ((takeBits >> j) & 1u)
            positions[counts[j * (threads / 32) + t / 32] + __popc(take & ((1u << (t % 32)) - 1))] = j * threads + t;
    }
    __syncthreads();
    constexpr int perThread = capacity / threads;
    unsigned sortKeys[perThread];
    int ids[perThread];
    #pragma unroll
    for (int i = 0; i < perThread; ++i) {
        int col = t * perThread + i;
        ids[i] = col < (trimTies ? top : candidates) ? positions[col] : -1;
        sortKeys[i] = ids[i] >= 0 ? OrderedScoreBits(scores[ids[i]]) - cutoff : 0;
    }
    // Valid zero scores precede padding in blocked input order.
    __syncthreads();
    Sort(temp.sort).SortDescendingBlockedToStriped(sortKeys, ids, 0, max(1, 32 - __clz(maximum - cutoff)));
    #pragma unroll
    for (int i = 0; i < top / threads; ++i)
        indices[size_t(row) * top + i * threads + t] = ids[i];
}
}
