namespace merge_screen {
using namespace naive_indexer_batch;
using Key = unsigned long long;
constexpr int K = naive_topk::kTopK;

struct GlobalList {
    const unsigned *data;
    __device__ Key operator[](int i) const {
        const uint2 item = reinterpret_cast<const uint2 *>(data)[i];
        return (Key(item.x) << 32) | Key(~item.y);
    }
};
struct SharedList {
    const Key *data;
    __device__ Key operator[](int i) const { return data[i]; }
};

// Both input lists are descending in (ordered score, inverse logical position).
// Logical positions are unique across context shards, so valid keys are unique.
template<class A, class B>
__device__ __forceinline__ Key MergedAt(A a, B b, int na, int nb, int rank) {
    if (rank >= na + nb) return 0;
    int lo = max(0, rank - nb), hi = min(rank, na);
    while (lo < hi) {
        const int i = (lo + hi) / 2, j = rank - i;
        if (i < na && j > 0 && a[i] > b[j - 1]) lo = i + 1;
        else hi = i;
    }
    const int j = rank - lo;
    const Key left = lo < na ? a[lo] : 0;
    const Key right = j < nb ? b[j] : 0;
    return max(left, right);
}

template<int Threads, bool SharedInput>
__global__ __launch_bounds__(Threads) void MergeFour(Batch batch) {
    const Row &row = batch.rows[blockIdx.x];
    extern __shared__ __align__(16) Key storage[];
    Key *pairs = storage + (SharedInput ? 4 * K : 0);
    int count[4];
#pragma unroll
    for (int rank = 0; rank < 4; ++rank)
        count[rank] = min(K, row.LocalCount(rank));
    if constexpr (SharedInput) {
        for (int i = threadIdx.x; i < 4 * K; i += Threads) {
            const int owner = i / K, local = i % K;
            GlobalList list{row.merged + ((size_t)owner * row.mergedRows + row.mergedRow) * K * 2};
            storage[i] = local < count[owner] ? list[local] : 0;
        }
        __syncthreads();
    }
    // Two independent sorted merges produce at most K entries each.
    for (int i = threadIdx.x; i < 2 * K; i += Threads) {
        const int pair = i / K, rank = i % K, owner = pair * 2;
        Key result;
        if constexpr (SharedInput) {
            result = MergedAt(SharedList{storage + owner * K}, SharedList{storage + (owner + 1) * K},
                              count[owner], count[owner + 1], rank);
        } else {
            GlobalList a{row.merged + ((size_t)owner * row.mergedRows + row.mergedRow) * K * 2};
            GlobalList b{row.merged + ((size_t)(owner + 1) * row.mergedRows + row.mergedRow) * K * 2};
            result = MergedAt(a, b, count[owner], count[owner + 1], rank);
        }
        pairs[i] = result;
    }
    __syncthreads();
    const int na = min(K, count[0] + count[1]), nb = min(K, count[2] + count[3]);
    for (int i = threadIdx.x; i < K; i += Threads) {
        const Key value = MergedAt(SharedList{pairs}, SharedList{pairs + K}, na, nb, i);
        // A valid nonnegative position has a nonzero inverted low word,
        // including when its ordered score word is zero.
        row.indices[i] = value ? int(~unsigned(value)) : -1;
    }
}

} // namespace merge_screen
