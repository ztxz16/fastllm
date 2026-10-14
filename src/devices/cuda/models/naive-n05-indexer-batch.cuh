#pragma once
#include "naive-n05-dsa-mma.cuh"
#include "naive-n05-topk.cuh"

namespace naive_indexer_batch {
using BF16 = __nv_bfloat16;
struct Row {
    FastllmNaiveKV key;
    const int *live;
    BF16 *query;
    float *scale;
    unsigned *scores;
    naive_topk::Workspace *workspace;
    int *indices;
    int sequence, stride;
    const void *const *cache;
};
// CUDA 12.1 raised the kernel-argument limit from 4 KiB to 32,764 bytes.
#if CUDART_VERSION >= 12010
constexpr int BatchLimit = 256;
#else
constexpr int BatchLimit = 32;
#endif
struct Batch { Row rows[BatchLimit]; };

// Descriptors travel in the launch arguments and remain fixed in a graph.
// Device lengths and optional KV bindings change without rebinding scratch.
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
    auto key = row.key;
    key.Bind(row.cache);
    naive_dsa_mma::IndexerDecodeScoresRow(row.query, key, row.scale,
        weights + (size_t)row.sequence * naive_dsa_mma::kHeads,
        ScoreOutput{row.scores}, row.stride, n, n - 1);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template<bool CacheL1>
__device__ __forceinline__ void LoadScoreVector(void *destination, const void *source, bool valid) {
    unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(destination));
    int bytes = valid ? 16 : 0;
    if constexpr (CacheL1)
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
            :: "r"(address), "l"(source), "r"(bytes) : "memory");
    else
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
            :: "r"(address), "l"(source), "r"(bytes) : "memory");
}
template<bool CacheL1>
__device__ __forceinline__ void LoadScoreVector(void *destination, FastllmNaiveKVRow source, bool valid) {
    if (source.fp8) *(uint4 *)destination = valid ? source.Load8() : make_uint4(0,0,0,0);
    else LoadScoreVector<CacheL1>(destination, source.data, valid);
}

#endif

// K occupies the MMA row dimension: each warp produces sixteen key scores.
// Cache the reused Q in L1, while streaming K directly to shared memory.
// The caller guarantees 16-byte alignment and retains Scores as the fallback.
__global__ void HopperScores(const BF16 *weights, Batch batch) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace naive_dsa_mma;
    constexpr int Threads = 128;
    Row r = batch.rows[blockIdx.y];
    r.key.Bind(r.cache);
    const int n = *r.live, first = blockIdx.x * kKeys;
    if (first >= n) return;
    __shared__ __align__(16) BF16 qs[kHeads][kIndexerDim + 8], ks[kKeys][kIndexerDim + 8];
    __shared__ float scales[kHeads], weightsF[kHeads];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    for (int i = t; i < kHeads * (kIndexerDim / 8); i += Threads) {
        int h = i / (kIndexerDim / 8), d = i % (kIndexerDim / 8) * 8;
        LoadScoreVector<true>(&qs[h][d], r.query + h * kIndexerDim + d, true);
    }
    // Tiles aligned to a divisor of the page length need only one page base.
    // Keep per-row translation for arbitrary page lengths (e.g. 96 tokens).
    const bool onePage = !r.key.pages || r.key.pageLen % kKeys == 0;
    const auto tile = onePage ? r.key.Row(first, r.stride) : FastllmNaiveKVRow{};
    for (int i = t; i < kKeys * (kIndexerDim / 8); i += Threads) {
        int key = i / (kIndexerDim / 8), d = i % (kIndexerDim / 8) * 8;
        const auto base = onePage ? tile + (size_t)key * r.stride
                                  : r.key.Row(min(first + key, n - 1), r.stride);
        LoadScoreVector<false>(&ks[key][d], base + r.stride - kIndexerDim + d, first + key < n);
    }
    if (t < kHeads) {
        scales[t] = r.scale[t];
        weightsF[t] = (float)weights[r.sequence * kHeads + t];
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    float dot[2][4] = {};
#pragma unroll
    for (int d = 0; d < kIndexerDim; d += 16) {
        uint32_t a[4];
        Load4(a, &ks[warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8][d + (lane >> 4) * 8]);
#pragma unroll
        for (int group = 0; group < 2; ++group) {
            uint32_t b[2];
            Load2(b, &qs[group * 8 + (lane & 7)][d + ((lane >> 3) & 1) * 8]);
            Mma(dot[group], a, b);
        }
    }
    // Each lane scales only its own head values, before exchanging them.
#pragma unroll
    for (int group = 0; group < 2; ++group)
#pragma unroll
        for (int i = 0; i < 4; ++i)
            dot[group][i] = fmaxf(dot[group][i] * scales[group * 8 + (lane & 3) * 2 + i % 2], 0.f);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        float sum = 0.f;
#pragma unroll
        for (int group = 0; group < 2; ++group) {
            const int h = group * 8 + (lane & 3) * 2;
            float w0 = weightsF[h], w1 = weightsF[h + 1];
            // Pass the accumulator through each pair of heads in ascending
            // order. This halves the shuffles without reassociating the sum.
#pragma unroll
            for (int part = 0; part < 4; ++part) {
                float value = fmaf(dot[group][half * 2], w0, sum);
                value = fmaf(dot[group][half * 2 + 1], w1, value);
                sum = __shfl_sync(0xffffffff, value, (lane & ~3) + part);
            }
        }
        int key = first + warp * 16 + (lane >> 2) + half * 8;
        if ((lane & 3) == 0 && key < n) r.scores[key] = naive_topk::OrderedScoreBits(sum);
    }
#endif
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
