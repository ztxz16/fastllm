#pragma once
namespace merge_chunk_screen {
using namespace naive_indexer_batch;
using Key = unsigned long long;
constexpr int K = naive_topk::kTopK;

struct CompactRow { const int *live; int *indices; };
constexpr int CompactLimit = (ArgumentBytes - 160) / sizeof(CompactRow);
struct CompactBatch {
    CompactRow rows[CompactLimit];
    const unsigned *merged;
    int total, first, page;
    int ranks = 4;
};
static_assert(sizeof(CompactBatch) + 128 <= ArgumentBytes);
__device__ __forceinline__ Row MergeRow(const Batch &batch, int index) {
    return batch.rows[index];
}
__device__ __forceinline__ Row MergeRow(const CompactBatch &batch, int index) {
    Row row{};
    row.live = batch.rows[index].live; row.indices = batch.rows[index].indices;
    row.merged = batch.merged; row.mergedRows = batch.total; row.mergedRow = batch.first + index;
    row.cpRanks = batch.ranks; row.contextPageLen = batch.page;
    return row;
}

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

template<int Threads, int Items, class Rows = Batch, int Ranks = 4>
__global__ __launch_bounds__(Threads) void MergeChunked(Rows batch) {
    static_assert(K % Items == 0 && (Ranks == 2 || Ranks == 4 || Ranks == 8));
    const Row row = MergeRow(batch, blockIdx.x);
    extern __shared__ __align__(16) Key storage[];
    Key *input = storage, *output = storage + Ranks * K;
    int count[Ranks];
#pragma unroll
    for (int rank = 0; rank < Ranks; ++rank)
        count[rank] = min(K, row.LocalCount(rank));
    for (int i = threadIdx.x; i < Ranks * K; i += Threads) {
        const int owner = i / K, local = i % K;
        merge_screen::GlobalList list{
            row.merged + ((size_t)owner * row.mergedRows + row.mergedRow) * K * 2};
        input[i] = local < count[owner] ? list[local] : 0;
    }
    __syncthreads();
    // Merge a tree of sorted lists, retaining only the first K entries at
    // each level. Ping-pong storage is bounded by Ranks + Ranks/2 lists.
#pragma unroll
    for (int lists = Ranks; lists > 2; lists /= 2) {
        for (int chunk = threadIdx.x; chunk < lists / 2 * K / Items; chunk += Threads) {
            const int pair = chunk / (K / Items), rank = chunk % (K / Items) * Items;
            const int owner = pair * 2;
            MergeInterval<Items, false>(input + owner * K, input + (owner + 1) * K,
                count[owner], count[owner + 1], rank, output + pair * K, nullptr);
        }
        __syncthreads();
        for (int pair = 0; pair < lists / 2; ++pair)
            count[pair] = min(K, count[pair * 2] + count[pair * 2 + 1]);
        Key *previous = input; input = output; output = previous;
    }
    for (int chunk = threadIdx.x; chunk < K / Items; chunk += Threads)
        MergeInterval<Items, true>(input, input + K, count[0], count[1], chunk * Items, nullptr, row.indices);
}
} // namespace merge_chunk_screen
