#pragma once
#include "naive-n05-topk.cuh"

namespace naive_topk {
// Hopper benefits from fewer radix passes; retain CUB's original setting
// on other architectures where the larger digit's resource cost is unmeasured.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
constexpr int kCompactTopKRadixBits = 6;
#else
constexpr int kCompactTopKRadixBits = 4;
#endif
template<int Items, int Threads = 256>
using CompactTopKSort = cub::BlockRadixSort<unsigned long long, Threads, Items,
    cub::NullType, kCompactTopKRadixBits>;

// Keep all score bits but compress the inverse index to its actual width.
// Skipping constant index bits reduces radix passes without changing score
// order or lower-index tie breaking, including signed zero and NaNs.
template<int Items, bool RawScores = false, int Threads = 256>
__device__ void SortTopKCompactRow(const unsigned long long *input, const int *ends,
        int *indices, int queries, int stride, int topK, int keyCount,
        typename CompactTopKSort<Items, Threads>::TempStorage &storage, int outputRow = -1) {
    using Sort = CompactTopKSort<Items, Threads>;
    const int row = blockIdx.x, count = ends[queries + row] - row * stride;
    const int indexBits = 32 - __clz((unsigned)max(1, keyCount - 1));
    const unsigned mask = 0xffffffffu >> (32 - indexBits);
    unsigned long long keys[Items];
    #pragma unroll
    for (int i = 0; i < Items; ++i) {
        int col = threadIdx.x + i * Threads;
        unsigned long long key = col < count ? input[(size_t)row * stride + col] : 0;
        if constexpr (RawScores)
            key = (static_cast<unsigned long long>(OrderedScoreBits(__uint_as_float(unsigned(key >> 32)))) << 32) |
                  unsigned(key);
        unsigned index = 0xffffffffu - (unsigned)key;
        keys[i] = col < count ? ((key >> 32) << indexBits) | (mask - index) : 0;
    }
    int scoreBits = 32;
    if constexpr (RawScores) {
        // Refined candidates normally share the sign/exponent prefix. Find
        // exactly which score bits differ, so radix sort can skip constant
        // passes without rounding scores or changing index-based tie breaks.
        const unsigned firstScore = count ? OrderedScoreBits(__uint_as_float(
            unsigned(input[(size_t)row * stride] >> 32))) : 0;
        unsigned different = 0;
        #pragma unroll
        for (int i = 0; i < Items; ++i) if (threadIdx.x + i * Threads < count)
            different |= unsigned(keys[i] >> indexBits) ^ firstScore;
        for (int delta = 16; delta; delta >>= 1)
            different |= __shfl_xor_sync(0xffffffff, different, delta);
        unsigned *scratch = reinterpret_cast<unsigned *>(&storage);
        if (!(threadIdx.x % 32)) scratch[threadIdx.x / 32] = different;
        __syncthreads();
        different = scratch[threadIdx.x % (Threads / 32)];
        for (int delta = Threads / 64; delta; delta >>= 1)
            different |= __shfl_xor_sync(0xffffffff, different, delta);
        // Invalid slots have zero low bits, so they still sort after every
        // retained key, even when all valid scores are identical.
        scoreBits -= __clz(different);
        __syncthreads();
    }
    Sort(storage).SortDescendingBlockedToStriped(keys, 0, scoreBits + indexBits);
    #pragma unroll
    for (int i = 0; i < Items; ++i) {
        int col = threadIdx.x + i * Threads;
        if (col < topK)
            indices[(size_t)(outputRow < 0 ? row : outputRow) * topK + col] = col < count
                ? (int)(mask - ((unsigned)keys[i] & mask)) : -1;
    }
}

// Candidate selection still has 4096 slots. Sort only the populated range;
// shrinking selection itself would add radix passes on long or tied rows.
template<bool RawScores = false, int Threads = 256>
__global__ void SortTopKCompact(const unsigned long long *input, const int *ends,
        int *indices, int queries, int stride, int topK, int keyCount) {
    static_assert(Threads == 128 || Threads == 256, "Unsupported compact sort geometry");
    union Storage {
        typename CompactTopKSort<2048 / Threads, Threads>::TempStorage small;
        typename CompactTopKSort<3072 / Threads, Threads>::TempStorage medium;
        typename CompactTopKSort<4096 / Threads, Threads>::TempStorage large;
    };
    __shared__ Storage storage;
    const int count = ends[queries + blockIdx.x] - blockIdx.x * stride;
    if (count <= 2048)
        SortTopKCompactRow<2048 / Threads, RawScores, Threads>(input, ends, indices, queries, stride, topK, keyCount, storage.small);
    else if (count <= 3072)
        SortTopKCompactRow<3072 / Threads, RawScores, Threads>(input, ends, indices, queries, stride, topK, keyCount, storage.medium);
    else
        SortTopKCompactRow<4096 / Threads, RawScores, Threads>(input, ends, indices, queries, stride, topK, keyCount, storage.large);
}

} // namespace naive_topk
