#pragma once
#include "naive-n05-dsa-mma.cuh"
#include "naive-n05-topk.cuh"

namespace naive_indexer_batch {
using BF16 = __nv_bfloat16;
struct Row {
    const BF16 *key;
    const int *live;
    BF16 *query;
    float *scale;
    unsigned *scores;
    naive_topk::Workspace *workspace;
    int *indices;
    int sequence, stride;
};
// CUDA 12.1 raised the kernel-argument limit from 4 KiB to 32,764 bytes.
#if CUDART_VERSION >= 12010
constexpr int BatchLimit = 256;
#else
constexpr int BatchLimit = 32;
#endif
struct Batch { Row rows[BatchLimit]; };

// Descriptors travel in the launch arguments and remain fixed in a graph.
// Only device lengths change during replay; all scratch belongs to the caller.
__global__ void Quantize(const BF16 *query, Batch batch) {
    const Row &row = batch.rows[blockIdx.y];
    naive_dsa_mma::QuantizeIndexerRow(query + (size_t)row.sequence *
        naive_dsa_mma::kHeads * naive_dsa_mma::kIndexerDim,
        row.query, row.scale, naive_dsa_mma::kIndexerDim, 0, false);
}
struct ScoreOutput {
    unsigned *bits;
    __device__ void Store(int index, float value) const {
        bits[index] = naive_topk::OrderedScoreBits(value);
    }
};
__global__ void Scores(const BF16 *weights, Batch batch) {
    const Row &row = batch.rows[blockIdx.y];
    int n = *row.live;
    if (blockIdx.x * naive_dsa_mma::kKeys >= n) return;
    naive_dsa_mma::IndexerDecodeScoresRow(row.query, row.key, row.scale,
        weights + (size_t)row.sequence * naive_dsa_mma::kHeads,
        ScoreOutput{row.scores}, row.stride, n, n - 1);
}

// A request owns one CTA instead of a cooperative grid. Independent requests
// fill the GPU together. Refine a radix threshold until at most 4096 candidates
// remain, then compact in position order for the existing stable block sort.
__global__ void Select(Batch batch) {
    using namespace naive_topk;
    const Row &row = batch.rows[blockIdx.x];
    const unsigned *scores = row.scores;
    auto *out = &row.workspace->candidates;
    auto *state = &row.workspace->state;
    const int t = threadIdx.x, lane = t & 31, n = *row.live;
    using Scan = cub::BlockScan<int, kThreads>;
    __shared__ typename Scan::TempStorage scan;
    __shared__ int hist[kBuckets];
    __shared__ unsigned maxima[kThreads / 32];
    unsigned maximum = 0, reference = n ? scores[0] : 0;
    bool different = false;
    for (int i = t; i < n; i += kThreads) {
        unsigned bits = scores[i];
        maximum = max(maximum, bits);
        different |= bits != reference;
    }
    int changed = __syncthreads_or(different);
    maximum = WarpMax(maximum);
    if (lane == 0) maxima[t / 32] = maximum;
    __syncthreads();
    if (t == 0) {
        maximum = 0;
        for (int w = 0; w < kThreads / 32; ++w) maximum = max(maximum, maxima[w]);
        state->maximum = maximum;
        state->uniform = !changed;
        state->selected = min(n, kTopK);
    }
    __syncthreads();
    if (!changed) return;
    unsigned prefix = 0, mask = 0;
    int rank = min(n, kTopK), candidates = n;
    for (int shift = 24; shift >= 0; shift -= 8) {
        hist[t] = 0;
        __syncthreads();
        for (int base = 0; base < n; base += kThreads) {
            int i = base + t;
            unsigned bits = i < n ? scores[i] : 0;
            int bucket = i < n && (bits & mask) == prefix ? int((bits >> shift) & (kBuckets - 1)) : kBuckets;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
            unsigned peers = __match_any_sync(0xffffffff, bucket);
            if (bucket < kBuckets && lane == __ffs(peers) - 1) atomicAdd(hist + bucket, __popc(peers));
#else
            if (bucket < kBuckets) atomicAdd(hist + bucket, 1);
#endif
        }
        __syncthreads();
        int bucket = kBuckets - 1 - t, total = hist[bucket], before;
        Scan(scan).ExclusiveSum(total, before);
        if (before < rank && before + total >= rank) {
            state->prefix = prefix | (unsigned(bucket) << shift);
            state->rank = rank - before;
            state->candidates = min(n, kTopK) - (rank - before) + total;
        }
        __syncthreads();
        prefix = state->prefix;
        rank = state->rank;
        candidates = state->candidates;
        mask |= (kBuckets - 1u) << shift;
        if (candidates <= kCapacity || shift == 0 || prefix == state->maximum) break;
    }
    int offset = 0, tiesBefore = 0;
    for (int base = 0; base < n; base += kThreads) {
        int i = base + t;
        unsigned bits = i < n ? scores[i] : 0;
        bool take = i < n && bits >= prefix;
        if (candidates > kCapacity) {
            int equal = i < n && bits == prefix, before, total;
            Scan(scan).ExclusiveSum(equal, before, total);
            __syncthreads();
            take = i < n && (bits > prefix || (equal && tiesBefore + before < rank));
            tiesBefore += total;
        }
        int before, total;
        Scan(scan).ExclusiveSum(int(take), before, total);
        __syncthreads();
        if (take) {
            out->keys[offset + before] = bits - prefix;
            out->positions[offset + before] = i;
        }
        offset += total;
    }
    if (t == 0) state->selected = offset;
}
__global__ void Sort(Batch batch) {
    const Row &row = batch.rows[blockIdx.x];
    naive_topk::SortRow(&row.workspace->candidates, &row.workspace->state, row.indices);
}
} // namespace naive_indexer_batch
