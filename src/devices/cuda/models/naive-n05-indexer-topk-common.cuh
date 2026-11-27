#pragma once
#include <cub/block/block_radix_sort.cuh>

namespace naive_indexer_ordered_topk {
constexpr int Threads = 256, Buckets = 2048;
// Five mantissa bits per exponent, starting at biased exponent 112.
// The end buckets also receive values outside this range.
constexpr int HalfBuckets = Buckets / 2, FirstBin = 112 * 32, BinShift = 18;
template <int Items> using RadixSort = cub::BlockRadixSort<unsigned, Threads, Items, int, 5>;
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
    return make_uint4(i < n ? scores[i] : 0, i + 1 < n ? scores[i + 1] : 0, i + 2 < n ? scores[i + 2] : 0,
                      i + 3 < n ? scores[i + 3] : 0);
}

template <int Items, class Candidates, class SortStorage>
__device__ void WriteRun(const Candidates &candidates, int start, int count, int keep, int bits,
                         SortStorage &storage, int *output) {
    using Sort = RadixSort<Items>;
    unsigned keys[Items];
    int positions[Items];
#pragma unroll
    for (int j = 0; j < Items; ++j) {
        const int i = threadIdx.x * Items + j;
        keys[j] = i < count ? candidates.keys[start + i] : 0;
        positions[j] = i < count ? candidates.positions[start + i] : -1;
    }
    Sort(*reinterpret_cast<typename Sort::TempStorage *>(&storage))
        .SortDescendingBlockedToStriped(keys, positions, 0, bits);
#pragma unroll
    for (int j = 0; j < Items; ++j) {
        const int i = threadIdx.x + j * Threads;
        if (i < keep) output[i] = positions[j];
    }
}

} // namespace naive_indexer_ordered_topk
