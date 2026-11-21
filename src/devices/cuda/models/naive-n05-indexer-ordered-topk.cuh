#pragma once
namespace naive_indexer_ordered_topk {
using namespace naive_indexer_batch;
using naive_topk::kTopK;

constexpr int Threads = 256, Buckets = 2048;
// Five mantissa bits per exponent, starting at biased exponent 112.
// The end buckets also receive values outside this range.
constexpr int HalfBuckets = Buckets / 2, FirstBin = 112 * 32, BinShift = 18;
template<int Items> using RadixSort = cub::BlockRadixSort<unsigned, Threads, Items, int, 5>;
union SortStorage {
    RadixSort<1>::TempStorage one;
    RadixSort<2>::TempStorage two;
    RadixSort<4>::TempStorage four;
    RadixSort<8>::TempStorage eight;
};
struct Candidates {
    unsigned keys[2 * kTopK];
    int positions[2 * kTopK];
};
constexpr size_t SharedBytes = sizeof(Candidates) + sizeof(SortStorage);
static_assert(SharedBytes >= naive_indexer_sortpack::SortBytes<256, 5, true, false>());

// A monotone projection of ordered FP32 bits. Saturation only widens the two
// end buckets. All selection, ordering and ties use the original 32-bit keys.
__device__ int Bucket(unsigned key) {
    const bool positive = key >> 31;
    const unsigned magnitude = positive ? key ^ 0x80000000u : ~key & 0x7fffffffu;
    const int coarse = min(HalfBuckets - 1, max(0, int(magnitude >> BinShift) - FirstBin));
    return positive ? HalfBuckets + coarse : HalfBuckets - 1 - coarse;
}
__device__ unsigned LowerBound(int bucket) {
    if (bucket >= HalfBuckets)
        return 0x80000000u | (bucket == HalfBuckets ? 0u : unsigned(FirstBin + bucket - HalfBuckets) << BinShift);
    return bucket == 0 ? 0u : 0x80000000u - (unsigned(FirstBin + HalfBuckets - bucket) << BinShift);
}
__device__ uint4 LoadFour(const unsigned *scores, int i, int n, bool aligned) {
    if (aligned && i + 3 < n) return *reinterpret_cast<const uint4 *>(scores + i);
    return make_uint4(i < n ? scores[i] : 0, i + 1 < n ? scores[i + 1] : 0,
                      i + 2 < n ? scores[i + 2] : 0, i + 3 < n ? scores[i + 3] : 0);
}

template<int Items>
__device__ void WriteRun(const Candidates &candidates, int start, int count, int keep,
                         int bits, SortStorage &storage, int *output) {
    using Sort = RadixSort<Items>;
    unsigned keys[Items];
    int positions[Items];
#pragma unroll
    for (int j = 0; j < Items; ++j) {
        const int i = threadIdx.x * Items + j;
        keys[j] = i < count ? candidates.keys[start + i] : 0;
        positions[j] = i < count ? candidates.positions[start + i] : -1;
    }
    Sort(*reinterpret_cast<typename Sort::TempStorage *>(&storage)).SortDescendingBlockedToStriped(
        keys, positions, 0, bits);
#pragma unroll
    for (int j = 0; j < Items; ++j) {
        const int i = threadIdx.x + j * Threads;
        if (i < keep) output[i] = positions[j];
    }
}

// One CTA owns one complete logical score row. Items above the cutoff bucket
// precede every item inside it, so the two stable sorts write disjoint ranks.
__global__ void OrderedTopK(const __grid_constant__ LocalTopKBatch batch) {
    const Row row = TopKRow(batch, blockIdx.x);
    const int t = threadIdx.x, n = row.Count(), top = min(n, kTopK);
    if (n <= 0) {
        for (int i = t; i < kTopK; i += Threads) row.indices[i] = -1;
        return;
    }
    extern __shared__ __align__(16) unsigned char shared_storage[];
    auto &candidates = *reinterpret_cast<Candidates *>(shared_storage);
    auto &sort = *reinterpret_cast<SortStorage *>(shared_storage + sizeof(Candidates));
    using Scan = cub::BlockScan<unsigned, Threads>;
    __shared__ typename Scan::TempStorage scan;
    __shared__ unsigned hist[Buckets], maximum;
    __shared__ int cutoff, higher, boundary;
    if (!t) maximum = 0;
    for (int i = t; i < Buckets; i += Threads) hist[i] = 0;
    __syncthreads();
    const bool aligned = (reinterpret_cast<uintptr_t>(row.scores) & 15) == 0;
    unsigned localMax = 0;
    const int vectorEnd = aligned ? n - n % 4 : 0;
    if (aligned) {
        for (int i = t * 4; i < vectorEnd; i += Threads * 4) {
            const auto v = *reinterpret_cast<const uint4 *>(row.scores + i);
            const unsigned keys[] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                atomicAdd(hist + Bucket(keys[j]), 1);
                localMax = max(localMax, keys[j]);
            }
        }
    }
    for (int j = vectorEnd + t; j < n; j += Threads) {
        const unsigned key = row.scores[j];
        atomicAdd(hist + Bucket(key), 1);
        localMax = max(localMax, key);
    }
    localMax = naive_topk::WarpMax(localMax);
    if (!(t & 31)) atomicMax(&maximum, localMax);
    __syncthreads();
    constexpr int Items = Buckets / Threads;
    unsigned counts[Items], before[Items];
#pragma unroll
    for (int j = 0; j < Items; ++j) counts[j] = hist[Buckets - 1 - (t * Items + j)];
    Scan(scan).ExclusiveSum(counts, before);
#pragma unroll
    for (int j = 0; j < Items; ++j) if (before[j] < top && before[j] + counts[j] >= top) {
        cutoff = Buckets - 1 - (t * Items + j);
        higher = before[j];
        boundary = counts[j];
    }
    __syncthreads();
    if (boundary > kTopK) {
        SelectRow(row);
        __syncthreads();
        naive_indexer_sortpack::SortRow<256, 5, true, false>(row);
        return;
    }

    const unsigned low = LowerBound(cutoff);
    const unsigned last = cutoff == Buckets - 1 ? ~0u : LowerBound(cutoff + 1) - 1;
    // Each half of the scan counts at most kTopK elements. Position-ordered
    // compaction makes equal keys stable, including at the final cutoff.
    unsigned offset = 0;
    for (int tile = 0; tile < n; tile += Threads * Items) {
        const int first = tile + t * Items;
        unsigned keys[Items], count = 0;
        bool above[Items], take[Items];
#pragma unroll
        for (int j = 0; j < Items; j += 4) {
            const uint4 v = LoadFour(row.scores, first + j, n, aligned);
            keys[j] = v.x; keys[j + 1] = v.y; keys[j + 2] = v.z; keys[j + 3] = v.w;
        }
#pragma unroll
        for (int j = 0; j < Items; ++j) {
            take[j] = first + j < n && keys[j] >= low;
            above[j] = take[j] && keys[j] > last;
            count += above[j] ? 1u << 16 : unsigned(take[j]);
        }
        unsigned cursor, total;
        Scan(scan).ExclusiveSum(count, cursor, total);
        __syncthreads();
        cursor += offset;
#pragma unroll
        for (int j = 0; j < Items; ++j) if (take[j]) {
            const int dest = above[j] ? int(cursor >> 16) : kTopK + int(cursor & 0xffffu);
            candidates.keys[dest] = keys[j] - (above[j] ? last + 1 : low);
            candidates.positions[dest] = first + j;
            cursor += above[j] ? 1u << 16 : 1u;
        }
        offset += total;
    }
    __syncthreads();
    if (higher) WriteRun<8>(candidates, 0, higher, higher, max(1, 32 - __clz(maximum - (last + 1))), sort, row.indices);
    __syncthreads();
    const int bits = max(1, 32 - __clz(min(last, maximum) - low));
    const int keep = top - higher;
    int *output = row.indices + higher;
    if (boundary <= Threads) WriteRun<1>(candidates, kTopK, boundary, keep, bits, sort, output);
    else if (boundary <= 2 * Threads) WriteRun<2>(candidates, kTopK, boundary, keep, bits, sort, output);
    else if (boundary <= 4 * Threads) WriteRun<4>(candidates, kTopK, boundary, keep, bits, sort, output);
    else WriteRun<8>(candidates, kTopK, boundary, keep, bits, sort, output);
    for (int i = top + t; i < kTopK; i += Threads) row.indices[i] = -1;
}
} // namespace naive_indexer_ordered_topk
