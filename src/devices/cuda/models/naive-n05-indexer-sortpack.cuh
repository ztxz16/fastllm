#pragma once
namespace naive_indexer_sortpack {
using namespace naive_indexer_batch;
template<int Threads, int RadixBits, bool Striped, bool Pack = true, int Capacity = naive_topk::kCapacity>
__device__ void SortRow(const Row &r) {
    using namespace naive_topk;
    const State &state = r.workspace->state;
    auto &candidate = r.workspace->candidates;
    if (state.uniform) {
        for (int j = threadIdx.x; j < kTopK; j += Threads) {
            const int position = j < state.selected ? j : -1;
            r.indices[j] = position;
            if constexpr (Pack) {
                candidate.keys[j] = position >= 0 ? r.scores[position] : 0;
                candidate.positions[j] = position >= 0 ? r.LogicalPosition(position) : -1;
            }
        }
        return;
    }
    constexpr int Items = Capacity / Threads;
    using BlockSort = cub::BlockRadixSort<unsigned, Threads, Items, int, RadixBits>;
    extern __shared__ __align__(16) unsigned char shared_storage[];
    auto &storage = *reinterpret_cast<typename BlockSort::TempStorage *>(shared_storage);
    unsigned keys[Items];
    int positions[Items];
    // Keep the original logical input order so equal keys retain position order.
#pragma unroll
    for (int i = 0; i < Items; ++i) {
        const int j = threadIdx.x * Items + i;
        keys[i] = j < state.selected ? candidate.keys[j] : 0;
        positions[i] = j < state.selected ? candidate.positions[j] : -1;
    }
    const int bits = max(1, 32 - __clz(state.maximum - state.prefix));
    if constexpr (Striped)
        BlockSort(storage).SortDescendingBlockedToStriped(keys, positions, 0, bits);
    else
        BlockSort(storage).SortDescending(keys, positions, 0, bits);
    // The sort has consumed all original candidate data. Each output has one
    // writer, and exchange copies remain deferred until every batch is packed.
#pragma unroll
    for (int i = 0; i < Items; ++i) {
        const int j = Striped ? threadIdx.x + i * Threads : threadIdx.x * Items + i;
        if (j < kTopK) {
            const int position = positions[i];
            r.indices[j] = position;
            if constexpr (Pack) {
                candidate.keys[j] = position >= 0 ? r.scores[position] : 0;
                candidate.positions[j] = position >= 0 ? r.LogicalPosition(position) : -1;
            }
        }
    }
}

template<int Threads, int RadixBits, bool Striped, bool Pack = true, class BatchType = Batch>
__global__ void SortPack(
#if CUDART_VERSION >= 12010
        const __grid_constant__
#endif
        BatchType batch) {
    const Row r = TopKRow(batch, blockIdx.x);
    SortRow<Threads, RadixBits, Striped, Pack>(r);
}
// Keep exact score/position order while avoiding a separate selection launch.
// Smaller candidate storage requires a finer threshold; use it only for short rows.
template<int Capacity = naive_topk::kTopK>
__global__ void SelectSort(
#if CUDART_VERSION >= 12010
        const __grid_constant__
#endif
        Batch batch) {
    const Row r = batch.rows[blockIdx.x];
    SelectRow<false, Capacity>(r);
    __syncthreads();
    SortRow<256, 5, true, false, Capacity>(r);
}

template<int Threads, int RadixBits, bool Striped, bool Pack = true, int Capacity = naive_topk::kCapacity>
constexpr size_t SortBytes() {
    using Sort = cub::BlockRadixSort<unsigned, Threads, Capacity / Threads, int, RadixBits>;
    return sizeof(typename Sort::TempStorage);
}

}
