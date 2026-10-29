#pragma once
namespace merge_chunk_screen {
using namespace naive_indexer_batch;
using Key = unsigned long long;
constexpr int K = naive_topk::kTopK;

// Each worker finds one output interval, then performs an ordinary sorted
// merge. Keys retain both ordered score bits and inverse logical positions.
template<int Items, bool Final>
__device__ __forceinline__ void MergeInterval(
        const Key *a, const Key *b, int na, int nb, int rank,
        Key *pairOut, int *indices) {
    int lo = max(0, rank - nb), hi = min(rank, na);
    if (rank < na + nb) {
        while (lo < hi) {
            const int i = (lo + hi) / 2, j = rank - i;
            if (i < na && j > 0 && a[i] > b[j - 1]) lo = i + 1;
            else hi = i;
        }
    }
    int ia = lo, ib = rank - lo;
    Key left = ia < na ? a[ia] : 0;
    Key right = ib < nb ? b[ib] : 0;
#pragma unroll
    for (int item = 0; item < Items; ++item) {
        const Key value = max(left, right);
        if constexpr (Final) indices[rank + item] = value ? int(~unsigned(value)) : -1;
        else pairOut[rank + item] = value;
        // Do not read beyond either list, including wholly empty intervals.
        if (left >= right) {
            ++ia;
            left = ia < na ? a[ia] : 0;
        } else {
            ++ib;
            right = ib < nb ? b[ib] : 0;
        }
    }
}

template<int Threads, int Items>
__global__ __launch_bounds__(Threads) void MergeFourChunked(Batch batch) {
    static_assert(K % Items == 0);
    const Row &row = batch.rows[blockIdx.x];
    extern __shared__ __align__(16) Key storage[];
    Key *pairs = storage + 4 * K;
    int count[4];
#pragma unroll
    for (int rank = 0; rank < 4; ++rank)
        count[rank] = min(K, row.LocalCount(rank));
    for (int i = threadIdx.x; i < 4 * K; i += Threads) {
        const int owner = i / K, local = i % K;
        merge_screen::GlobalList list{
            row.merged + ((size_t)owner * row.mergedRows + row.mergedRow) * K * 2};
        storage[i] = local < count[owner] ? list[local] : 0;
    }
    __syncthreads();
    for (int chunk = threadIdx.x; chunk < 2 * K / Items; chunk += Threads) {
        const int pair = chunk / (K / Items), rank = chunk % (K / Items) * Items;
        const int owner = pair * 2;
        MergeInterval<Items, false>(storage + owner * K, storage + (owner + 1) * K,
            count[owner], count[owner + 1], rank, pairs + pair * K, nullptr);
    }
    __syncthreads();
    const int na = min(K, count[0] + count[1]), nb = min(K, count[2] + count[3]);
    for (int chunk = threadIdx.x; chunk < K / Items; chunk += Threads)
        MergeInterval<Items, true>(pairs, pairs + K, na, nb, chunk * Items, nullptr, row.indices);
}
} // namespace merge_chunk_screen
