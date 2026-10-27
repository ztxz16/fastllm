#pragma once

// Exact ordered-u32 selection. Input order resolves every score tie. This is
// Hopper specialization: scoring, stable sort and context parallelism are unchanged.
namespace naive_indexer_wide_select {
using namespace naive_indexer_batch;
template<class BatchType = Batch>
__global__ __launch_bounds__(1024) void SelectWide(
#if CUDART_VERSION >= 12010
        const __grid_constant__
#endif
        BatchType batch) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    constexpr int Threads=1024, RadixBits=11, CompactItems=8;
    using namespace naive_topk;
    constexpr int Buckets = 1 << RadixBits;
    constexpr int HistItems = Buckets / Threads;
    static_assert(Buckets >= Threads && Buckets % Threads == 0);
    using Scan = cub::BlockScan<int, Threads>;
    __shared__ typename Scan::TempStorage scan;
    __shared__ int hist[Buckets];
    __shared__ unsigned maxima[Threads / 32];
    const Row row = TopKRow(batch, blockIdx.x);
    auto *state = &row.workspace->state;
    auto *out = &row.workspace->candidates;
    const int t = threadIdx.x, lane = t & 31, n = row.Count();
    const int top = min(n, kTopK);
    unsigned prefix = 0, mask = 0;
    int rank = top, candidates = n;
    // The first histogram also collects the uniform flag and maximum.
    for (int pass = 0; pass < (32 + RadixBits - 1) / RadixBits; ++pass) {
        const int shift = max(0, 32 - RadixBits * (pass + 1));
        const unsigned digitMask = (1u << min(RadixBits, 32 - RadixBits * pass)) - 1;
        for (int b = t; b < Buckets; b += Threads) hist[b] = 0;
        __syncthreads();
        unsigned maximum = 0, reference = n ? row.scores[0] : 0;
        bool different = false;
        for (int base = 0; base < n; base += Threads) {
            const int i = base + t;
            const unsigned bits = i < n ? row.scores[i] : 0;
            if (!pass) {
                maximum = max(maximum, bits);
                different |= i < n && bits != reference;
            }
            const int bucket = i < n && (bits & mask) == prefix
                ? int((bits >> shift) & digitMask) : Buckets;
            const unsigned peers = __match_any_sync(0xffffffff, bucket);
            if (bucket < Buckets && lane == __ffs(peers) - 1)
                atomicAdd(hist + bucket, __popc(peers));
        }
        if (!pass) {
            const int changed = __syncthreads_or(different);
            maximum = WarpMax(maximum);
            if (!lane) maxima[t / 32] = maximum;
            __syncthreads();
            if (!t) {
                maximum = 0;
                for (int w = 0; w < Threads / 32; ++w)
                    maximum = max(maximum, maxima[w]);
                state->maximum = maximum;
                state->uniform = !changed;
                state->selected = top;
                state->prefix = 0;
                state->done = 0;
            }
            __syncthreads();
            if (!changed) return;
        } else __syncthreads();
        int totals[HistItems], before[HistItems];
#pragma unroll
        for (int j = 0; j < HistItems; ++j)
            totals[j] = hist[Buckets - 1 - (t * HistItems + j)];
        Scan(scan).ExclusiveSum(totals, before);
#pragma unroll
        for (int j = 0; j < HistItems; ++j) {
            if (before[j] < rank && before[j] + totals[j] >= rank) {
                const int bucket = Buckets - 1 - (t * HistItems + j);
                state->prefix = prefix | (unsigned(bucket) << shift);
                state->rank = rank - before[j];
                state->candidates = top - (rank - before[j]) + totals[j];
            }
        }
        __syncthreads();
        prefix = state->prefix;
        rank = state->rank;
        candidates = state->candidates;
        mask |= digitMask << shift;
        if (candidates <= kCapacity || shift == 0 || prefix == state->maximum) break;
    }

    // A single scan covers Threads*CompactItems positions instead of Threads.
    // Each thread owns consecutive input positions, so compaction is stable.
    int offset = 0, tiesBefore = 0;
    for (int base = 0; base < n; base += Threads * CompactItems) {
        unsigned bits[CompactItems];
        bool equal[CompactItems], take[CompactItems];
        int equalCount = 0;
#pragma unroll
        for (int j = 0; j < CompactItems; ++j) {
            const int i = base + t * CompactItems + j;
            bits[j] = i < n ? row.scores[i] : 0;
            equal[j] = i < n && bits[j] == prefix;
            equalCount += int(equal[j]);
            take[j] = i < n && bits[j] >= prefix;
        }
        if (candidates > kCapacity) {
            int before, total;
            Scan(scan).ExclusiveSum(equalCount, before, total);
            __syncthreads();
            int equalOffset = tiesBefore + before;
#pragma unroll
            for (int j = 0; j < CompactItems; ++j) {
                const int i = base + t * CompactItems + j;
                take[j] = i < n && (bits[j] > prefix || (equal[j] && equalOffset < rank));
                equalOffset += int(equal[j]);
            }
            tiesBefore += total;
        }
        int count = 0, before, total;
#pragma unroll
        for (int j = 0; j < CompactItems; ++j) count += int(take[j]);
        Scan(scan).ExclusiveSum(count, before, total);
        __syncthreads();
        int target = offset + before;
#pragma unroll
        for (int j = 0; j < CompactItems; ++j) {
            if (take[j]) {
                out->keys[target] = bits[j] - prefix;
                out->positions[target] = base + t * CompactItems + j;
                ++target;
            }
        }
        offset += total;
    }
    if (!t) { state->selected = offset; state->done = 1; }
#endif
}
} // namespace naive_indexer_wide_select
