#pragma once
#include "naive-n05-topk.cuh"
#include <cstdint>

namespace naive_topk {
__device__ inline unsigned long long TopKOrder(unsigned scoreBits, int index) {
    return ((unsigned long long)scoreBits << 32) | (0xffffffffu - (unsigned)index);
}

// Select a bounded superset of the top K before sorting it. A radix prefix can
// stop as soon as at most capacity entries remain; no possible winner is lost.
// If equal scores straddle a full candidate buffer, retain the lowest indices.
template <bool Sampled, int SampleItems = 4, int Threads = 256>
__device__ __forceinline__ void SelectTopKCompactRow(const float *input, unsigned long long *selected,
                                                     int *offsets, int queries, int count, int topK,
                                                     int capacity, int row = blockIdx.x) {
    constexpr int threads = Threads, warps = threads / 32;
    using Scan = cub::BlockScan<int, threads>;
    __shared__ typename Scan::TempStorage scan;
    using SampleSort = cub::BlockRadixSort<unsigned, threads, SampleItems>;
    __shared__ union {
        int histogram[warps][256];
        unsigned long long compact[Threads >= 512 ? 4096 : 1];
        typename SampleSort::TempStorage sampleSort;
    } temporary;
    auto &histogram = temporary.histogram;
    __shared__ unsigned prefix;
    __shared__ int rank, candidateCount, selectedCount, tiesSeen;
    int t = threadIdx.x, lane = t % 32, warp = t / 32;
    int keep = min(count, topK);
    unsigned long long *output = selected + (size_t)row * capacity;
    if (t == 0) {
        offsets[row] = row * capacity;
        prefix = 0;
        rank = keep;
        selectedCount = 0;
        tiesSeen = 0;
    }
    __syncthreads();
    if (count <= capacity) {
        if (t == 0)
            offsets[queries + row] = row * capacity + count;
        for (int col = t; col < count; col += threads)
            output[col] = TopKOrder(OrderedScoreBits(input[col]), col);
        return;
    }
    // Estimate a threshold cheaply, but never use an approximate TopK.
    // The full scan must prove that the retained superset fits and has >= K
    // entries; otherwise discard it and run the original exact radix search.
    if constexpr (Sampled)
        if ((int64_t)count >= 4LL * capacity && keep >= 256) {
            constexpr int sampleCount = threads * SampleItems;
            unsigned sampled[SampleItems];
#pragma unroll
            for (int i = 0; i < SampleItems; ++i) {
                int col = (int)((int64_t)(t * SampleItems + i) * count / sampleCount);
                sampled[i] = OrderedScoreBits(input[col]);
            }
            // Only estimate a threshold here. Round down after sorting the
            // high bits; the full scan still proves a bounded exact superset.
            constexpr int lowBit = SampleItems > 4 || Threads > 256 ? 8 : 16;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
            constexpr bool histogramSample = Threads >= 512;
#else
            constexpr bool histogramSample = Threads == 1024;
#endif
            if constexpr (histogramSample) {
                // Only the sampled kth score is needed, not a sorted sample. Three
                // radix histograms find exactly the same rounded threshold. Keep
                // all sample positions and the full-scan candidate proof.
                const int sampleRank =
                    max(0, min(sampleCount - 1,
                               (int)(((int64_t)keep + capacity) * sampleCount / (2LL * count)) - 1));
                if (!t) {
                    prefix = 0;
                    rank = sampleRank + 1;
                }
                __syncthreads();
                unsigned sampleMask = 0;
#pragma unroll
                for (int shift = 24; shift >= lowBit; shift -= 8) {
                    const unsigned wanted = prefix;
                    const int target = rank;
                    for (int i = t; i < 256; i += threads)
                        ((int *)histogram)[i] = 0;
                    __syncthreads();
#pragma unroll
                    for (int i = 0; i < SampleItems; ++i) {
                        unsigned bits = sampled[i];
                        int bucket = (bits & sampleMask) == wanted ? int((bits >> shift) & 255) : 256;
#if __CUDA_ARCH__ >= 700
                        unsigned peers = __match_any_sync(0xffffffff, bucket);
                        if (bucket < 256 && lane == __ffs(peers) - 1)
                            atomicAdd(&histogram[0][bucket], __popc(peers));
#else
                        if (bucket < 256)
                            atomicAdd(&histogram[0][bucket], 1);
#endif
                    }
                    __syncthreads();
                    const int bucket = 255 - t;
                    const int n = t < 256 ? histogram[0][bucket] : 0;
                    int before;
                    Scan(scan).ExclusiveSum(n, before);
                    if (before < target && before + n >= target) {
                        prefix = wanted | (unsigned(bucket) << shift);
                        rank = target - before;
                    }
                    __syncthreads();
                    sampleMask |= 255u << shift;
                }
            } else {
                SampleSort(temporary.sampleSort).SortDescending(sampled, lowBit, 32);
                const int sampleRank =
                    max(0, min(sampleCount - 1,
                               (int)(((int64_t)keep + capacity) * sampleCount / (2LL * count)) - 1));
                if (t == sampleRank / SampleItems)
                    prefix = sampled[sampleRank % SampleItems] & (0xffffffffu << lowBit);
                __syncthreads();
            }
            unsigned threshold = prefix;
            unsigned long long *sampledOutput = output;
            if constexpr (Threads >= 512)
                if (capacity <= 4096)
                    sampledOutput = temporary.compact;
            // One shared reservation per warp and tile, preserving every
            // candidate and its original index for the exact final sort.
            constexpr int Items = 8;
            for (int base = 0; base < count; base += threads * Items) {
                unsigned bits[Items], ballots[Items];
                int total = 0;
                if constexpr (Threads == 1024) {
                    const int first = base + t * Items;
                    if (first + Items <= count && !(reinterpret_cast<uintptr_t>(input) & 15)) {
                        const float4 a = *reinterpret_cast<const float4 *>(input + first);
                        const float4 b = *reinterpret_cast<const float4 *>(input + first + 4);
                        bits[0] = OrderedScoreBits(a.x);
                        bits[1] = OrderedScoreBits(a.y);
                        bits[2] = OrderedScoreBits(a.z);
                        bits[3] = OrderedScoreBits(a.w);
                        bits[4] = OrderedScoreBits(b.x);
                        bits[5] = OrderedScoreBits(b.y);
                        bits[6] = OrderedScoreBits(b.z);
                        bits[7] = OrderedScoreBits(b.w);
                    } else {
#pragma unroll
                        for (int i = 0; i < Items; ++i)
                            bits[i] = first + i < count ? OrderedScoreBits(input[first + i]) : 0;
                    }
#pragma unroll
                    for (int i = 0; i < Items; ++i) {
                        ballots[i] = __ballot_sync(0xffffffff, first + i < count && bits[i] >= threshold);
                        total += __popc(ballots[i]);
                    }
                } else {
#pragma unroll
                    for (int i = 0; i < Items; ++i) {
                        int col = base + t + i * threads;
                        bits[i] = col < count ? OrderedScoreBits(input[col]) : 0;
                        ballots[i] = __ballot_sync(0xffffffff, col < count && bits[i] >= threshold);
                        total += __popc(ballots[i]);
                    }
                }
                if (total) {
                    int offset = lane == 0 ? atomicAdd(&selectedCount, total) : 0;
                    offset = __shfl_sync(0xffffffff, offset, 0);
#pragma unroll
                    for (int i = 0; i < Items; ++i) {
                        int pos = offset + __popc(ballots[i] & ((1u << lane) - 1));
                        if ((ballots[i] & (1u << lane)) && pos < capacity)
                            sampledOutput[pos] = TopKOrder(
                                bits[i], base + (Threads == 1024 ? t * Items + i : t + i * threads));
                        offset += __popc(ballots[i]);
                    }
                }
            }
            __syncthreads();
            if (selectedCount >= keep && selectedCount <= capacity) {
                // Coalesce the sparse warp reservations into one contiguous write.
                if constexpr (Threads >= 512)
                    if (capacity <= 4096)
                        for (int i = t; i < selectedCount; i += threads)
                            output[i] = temporary.compact[i];
                if (t == 0)
                    offsets[queries + row] = row * capacity + selectedCount;
                return;
            }
            if (t == 0) {
                prefix = 0;
                rank = keep;
                selectedCount = 0;
                tiesSeen = 0;
            }
            __syncthreads();
        }
    unsigned mask = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        unsigned wanted = prefix;
        int target = rank;
        for (int i = t; i < warps * 256; i += threads)
            ((int *)histogram)[i] = 0;
        __syncthreads();
        for (int base = 0; base < count; base += threads) {
            int col = base + t;
            unsigned bits = col < count ? OrderedScoreBits(input[col]) : 0;
            int bucket = col < count && (bits & mask) == wanted ? int((bits >> shift) & 255) : 256;
// Aggregate identical buckets within each warp before its histogram
// update, including the common case where all score exponents match.
#if __CUDA_ARCH__ >= 700
            unsigned peers = __match_any_sync(0xffffffff, bucket);
            if (bucket < 256 && lane == __ffs(peers) - 1)
                atomicAdd(&histogram[warp][bucket], __popc(peers));
#else
            if (bucket < 256)
                atomicAdd(&histogram[warp][bucket], 1);
#endif
        }
        __syncthreads();
        int bucket = 255 - t, n = 0;
#pragma unroll
        for (int w = 0; w < warps; ++w)
            n += t < 256 ? histogram[w][bucket] : 0;
        int before;
        Scan(scan).ExclusiveSum(n, before);
        if (before < target && before + n >= target) {
            prefix = wanted | ((unsigned)bucket << shift);
            rank = target - before;
            candidateCount = keep - rank + n;
        }
        __syncthreads();
        mask |= 255u << shift;
        if (candidateCount <= capacity)
            break;
    }
    unsigned threshold = prefix;
    int tiesWanted = rank;
    if (candidateCount <= capacity) {
        if (t == 0)
            offsets[queries + row] = row * capacity + candidateCount;
        for (int base = 0; base < count; base += threads) {
            int col = base + t;
            unsigned bits = col < count ? OrderedScoreBits(input[col]) : 0;
            bool take = col < count && bits >= threshold;
            unsigned ballot = __ballot_sync(0xffffffff, take);
            if (ballot) {
                int outputBase = 0, leader = __ffs(ballot) - 1;
                if (lane == leader)
                    outputBase = atomicAdd(&selectedCount, __popc(ballot));
                outputBase = __shfl_sync(0xffffffff, outputBase, leader);
                if (take)
                    output[outputBase + __popc(ballot & ((1u << lane) - 1))] = TopKOrder(bits, col);
            }
        }
    } else {
        if (t == 0)
            offsets[queries + row] = row * capacity + keep;
        for (int base = 0; base < count; base += threads) {
            int col = base + t;
            unsigned bits = col < count ? OrderedScoreBits(input[col]) : 0;
            int equal = col < count && bits == threshold, equalBefore, equalTotal;
            Scan(scan).ExclusiveSum(equal, equalBefore, equalTotal);
            __syncthreads();
            int take = col < count && (bits > threshold || (equal && tiesSeen + equalBefore < tiesWanted));
            int before, total;
            Scan(scan).ExclusiveSum(take, before, total);
            __syncthreads();
            if (take)
                output[selectedCount + before] = TopKOrder(bits, col);
            __syncthreads();
            if (t == 0) {
                selectedCount += total;
                tiesSeen += equalTotal;
            }
            __syncthreads();
        }
    }
}

template <bool Sampled, int SampleItems = 4, int Threads = 256>
__global__ void SelectTopKCompact(const float *scores, unsigned long long *selected, int *offsets,
                                  int queries, int keys, int queryStart, int topK, int capacity,
                                  int partRows = 0, int gap = 0) {
    int row = blockIdx.x;
    int count = queryStart + row + 1 + (partRows && row >= partRows ? gap : 0);
    SelectTopKCompactRow<Sampled, SampleItems, Threads>(scores + (size_t)row * keys, selected, offsets,
                                                        queries, count, topK, capacity);
}
} // namespace naive_topk
