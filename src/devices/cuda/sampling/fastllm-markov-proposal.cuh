#pragma once

#include "topk.cuh"
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <curand_kernel.h>

namespace fastllm_markov {

// Retain the distribution for rejection while measuring its mass in parallel.
// Sampling then needs to read just the selected tile, including for unfiltered rows.
constexpr int kGraphDrawTile = 4096;
__global__ void FastllmRetainProposalTiles(const float *probs, float *q, int stride, int vocab, double *mass, const int *done) {
    if (done[blockIdx.y]) return;
    const int row = blockIdx.y, begin = blockIdx.x * kGraphDrawTile;
    double sum = 0;
    for (int i = begin + threadIdx.x; i < min(begin + kGraphDrawTile, vocab); i += blockDim.x) {
        float value = probs[size_t(row) * vocab + i];
        q[size_t(row) * stride + i] = value;
        sum += value;
    }
    using Reduce = cub::BlockReduce<double, 256>;
    __shared__ typename Reduce::TempStorage storage;
    sum = Reduce(storage).Sum(sum);
    if (!threadIdx.x) mass[size_t(row) * gridDim.x + blockIdx.x] = sum;
}

__global__ void FastllmDrawProposalTile(const float *q, int qStride, const double *mass, int tiles, int vocab,
                                        const uint64_t *seed, int step, int *tokens, int tokenStride, const int *done) {
    if (done[blockIdx.x]) return;
    const int row = blockIdx.x;
    __shared__ int first;
    __shared__ double target;
    if (!threadIdx.x) {
        const double *weights = mass + size_t(row) * tiles;
        double total = 0;
        for (int i = 0; i < tiles; ++i) total += weights[i];
        curandStatePhilox4_32_10_t state;
        curand_init(*seed, row, step, &state);
        double draw = fmin(double(curand_uniform(&state)), 1.0 - 0x1p-33) * total;
        int tile = 0;
        double prefix = 0;
        while (tile + 1 < tiles && prefix + weights[tile] < draw) prefix += weights[tile++];
        first = tile * kGraphDrawTile;
        target = draw - prefix;
        tokens[size_t(row) * tokenStride] = 0;
    }
    __syncthreads();
    constexpr int perLane = kGraphDrawTile / 256;
    const int begin = first + threadIdx.x * perLane;
    double sum = 0;
    for (int i = begin; i < min(begin + perLane, vocab); ++i) sum += q[size_t(row) * qStride + i];
    using Scan = cub::BlockScan<double, 256>;
    __shared__ typename Scan::TempStorage storage;
    double prefix;
    Scan(storage).ExclusiveSum(sum, prefix);
    if (prefix < target && target <= prefix + sum) {
        for (int i = begin; i < min(begin + perLane, vocab); ++i) {
            prefix += q[size_t(row) * qStride + i];
            if (prefix >= target) {
                tokens[size_t(row) * tokenStride] = i;
                break;
            }
        }
    }
}

// Pack the finite support after stable top-k masking. Rows with a broader
// support continue through the dense normalizer in the same captured graph.
constexpr int kCompactGraphDraw = FastllmFoldSelection::maxCandidates;
template<bool Descending>
__device__ __forceinline__ unsigned long long SortGraphDrawKeys(unsigned long long key, unsigned long long *shared) {
    const unsigned lane = threadIdx.x;
#pragma unroll
    for (unsigned width = 2; width <= kCompactGraphDraw; width *= 2) {
#pragma unroll
        for (unsigned stride = width / 2; stride; stride /= 2) {
            unsigned long long other;
            if (stride >= 32) {
                shared[lane] = key;
                __syncthreads();
                other = shared[lane ^ stride];
                __syncthreads();
            } else {
                other = __shfl_xor_sync(0xffffffffu, key, stride);
            }
            const bool lower = ((lane & width) == 0) ^ ((lane & stride) != 0) ^ Descending;
            key = lower ? min(key, other) : max(key, other);
        }
    }
    return key;
}
// Preserve both activation-precision additions and the final softcap rounding.
template<class T> struct MarkovProposalScores {
    const T *base, *adjust, *bias;
    int baseStride, vocab;
    float cap;
    __device__ T raw(int row, int id) const {
        if (!adjust) return base[size_t(row) * baseStride + id];
        if constexpr (std::is_same_v<T, float>)
            return (base[size_t(row) * baseStride + id] + adjust[size_t(row) * vocab + id]) + bias[id];
        const T correction = T(float(adjust[size_t(row) * vocab + id]) + float(bias[id]));
        return T(float(base[size_t(row) * baseStride + id]) + float(correction));
    }
    __device__ T softcap(T value) const { return adjust ? T(cap * tanhf(float(value) / cap)) : value; }
    __device__ T operator()(int row, int id) const { return softcap(raw(row, id)); }
};

// FlashInfer supplies the radix selection and its resident CTA group barrier. Keep
// each tile in shared memory through collection; only unresolved rows need dense logits.
constexpr int kGraphRadixThreads = 512;
template<class T>
__global__ void SelectGraphDrawSupport(MarkovProposalScores<T> scores, const float *temperatures, const int *topK, int *counts,
                                       float2 *support, float *pivots, float *q, int stride, int vocab, int rows,
                                       int chunk, int blocksPerRow, flashinfer::sampling::RadixRowState *states,
                                       const int *previous = nullptr, int supportStride = 0) {
    using namespace flashinfer::sampling;
    using Traits = RadixTopKTraits<T>;
    using Key = typename Traits::OrderedType;
    extern __shared__ uint32_t storage[];
    auto *ordered = reinterpret_cast<Key *>(storage);
    __shared__ uint32_t histogram[256], suffix[256], scalars[2];
    const int group = blockIdx.x / blocksPerRow, tile = blockIdx.x % blocksPerRow;
    const int begin = tile * chunk, length = min(chunk, vocab - begin);
    int phase = 0, selection = 0;
    for (int row = group; row < rows; row += gridDim.x / blocksPerRow) {
        const int k = topK[row];
        if (k >= vocab) {
            if (!tile && !threadIdx.x) pivots[row] = -INFINITY;
            continue;
        }
        const float inv = 1.f / temperatures[row];
        for (int i = threadIdx.x; i < length; i += blockDim.x)
            ordered[i] = Traits::ToOrdered(scores.raw(row, begin + i));
        __syncthreads();
        uint32_t prefix = 0, remaining = k;
#pragma unroll
        for (uint32_t digit = 0; digit < sizeof(T); ++digit)
            RadixSelectOneRound<kGraphRadixThreads, false>(ordered, length, histogram, suffix, scalars, states + group,
                prefix, remaining, digit, selection, phase, blocksPerRow, tile, threadIdx.x, &prefix, &remaining);
        ++selection;
        const float cutoff = float(scores.softcap(Traits::FromOrdered(Key(prefix)))) * inv;
        if (!tile && !threadIdx.x) pivots[row] = cutoff;
        if (k <= kCompactGraphDraw) {
            const int *ids = previous ? previous + size_t(row) * supportStride : nullptr;
            const bool clearAll = !ids || ids[0] < -1;
            if (!clearAll && tile == 0 && threadIdx.x < kCompactGraphDraw && ids[threadIdx.x] >= 0)
                q[size_t(row) * stride + ids[threadIdx.x]] = 0.f;
            // A raw score below the pivot can round onto its softcap plateau. Probe a
            // lower key: only prune below it if its final score is strictly smaller.
            const uint32_t lowest = Traits::ToOrdered(Traits::NegInf());
            const uint32_t probe = prefix - min(prefix - lowest, uint32_t(warpSize));
            const uint32_t firstEligible = float(scores.softcap(Traits::FromOrdered(Key(probe)))) * inv < cutoff
                ? probe + 1 : lowest;
            for (int i = threadIdx.x; i < length; i += blockDim.x) {
                if (clearAll) q[size_t(row) * stride + begin + i] = 0.f;
                if (ordered[i] >= firstEligible) {
                    const float score = float(scores.softcap(Traits::FromOrdered(ordered[i]))) * inv;
                    if (!(score >= cutoff)) continue;
                    int dst = atomicAdd(counts + row, 1);
                    if (dst < kCompactGraphDraw)
                        support[size_t(row) * kCompactGraphDraw + dst] = make_float2(score, __int_as_float(begin + i));
                    if (!isfinite(score)) atomicMax(counts + row, kCompactGraphDraw + 1);
                }
            }
        }
        __syncthreads();
    }
}

template<class T>
__global__ void MaterializeGraphDrawMask(MarkovProposalScores<T> scores, const float *temperatures, const float *pivots,
                                         const int *done, float *masked, int vocab) {
    const int row = blockIdx.y;
    if (done[row]) return;
    const int begin = blockIdx.x * kGraphDrawTile;
    const float inv = 1.f / temperatures[row], pivot = pivots[row];
    for (int i = begin + threadIdx.x; i < min(begin + kGraphDrawTile, vocab); i += blockDim.x) {
        size_t pos = size_t(row) * vocab + i;
        float x = float(scores(row, i)) * inv;
        masked[pos] = x >= pivot ? x : -INFINITY;
    }
}

template<bool Round = false>
__global__ void SampleGraphDrawSupport(const float *scratch, const int *counts, const int *topK, const float *topP,
                                       const uint64_t *seed, int step, float *q, int stride, int *tokens,
                                       int tokenStride, int *done, int *fallbackK, int vocab, bool first = false,
                                       int *previous = nullptr, int supportStride = 0) {
    constexpr int N = kCompactGraphDraw;
    const int row = blockIdx.x, lane = threadIdx.x, k = topK[row];
    if (previous) {
        previous += size_t(row) * supportStride;
        if (!lane) previous[0] = -2; // Dense repair may populate the whole row.
    }
    if (!lane) {
        if constexpr (Round) {
            if (first) *done = 1;
            *tokens = 0; // A provisional step must always leave a valid embedding index.
        } else {
            done[row] = 0;
            fallbackK[row] = k;
        }
    }
    const int count = counts[row];
    if (k < 1 || k > count || count > N) {
        if constexpr (Round) if (!lane) *done = 0;
        return;
    }
    using Scan = cub::BlockScan<double, N>;
    __shared__ union { unsigned long long sort[N]; Scan::TempStorage scan; } tmp;
    __shared__ float maximum, cutoff;
    __shared__ double draw;
    unsigned long long key = 0;
    if (lane < count) {
        const auto x = reinterpret_cast<const float2 *>(scratch)[size_t(row) * N + lane];
        unsigned bits = __float_as_uint(x.x == 0.f ? 0.f : x.x);
        bits ^= int(bits) < 0 ? ~0u : 0x80000000u;
        key = (static_cast<unsigned long long>(bits) << 32) | ~__float_as_uint(x.y);
    }
    key = SortGraphDrawKeys<true>(key, tmp.sort);
    if constexpr (Round) {
        // Drop extra radix-bucket entries before the token-ID prefix scan, so
        // its FP64 reduction sees exactly the same inputs as the full selection.
        __shared__ unsigned kth;
        if (lane == k-1) kth = unsigned(key >> 32);
        __syncthreads();
        if ((key >> 32) < kth) key = 0;
    }
    unsigned bits = key >> 32;
    float score = __uint_as_float(bits ^ ((bits >> 31) ? 0x80000000u : ~0u));
    const int id = ~unsigned(key);
    if (!lane) maximum = score;
    __syncthreads();
    const float weight = lane < k ? expf(score - maximum) : 0.f;
    double prefix, total;
    Scan(tmp.scan).ExclusiveSum(double(weight), prefix, total);
    const double threshold = double(topP[row]) * total;
    if (!lane) cutoff = 0.f;
    __syncthreads();
    if (topP[row] < 1.f && prefix < threshold && threshold <= prefix + weight) cutoff = weight;
    __syncthreads();
    // Use the dense implementation on ambiguous FP32 nucleus boundaries.
    if (topP[row] < 1.f && __syncthreads_or(lane < k && weight > 0.f &&
            (fabs((prefix + weight) - threshold) <= 2.e-6 * total ||
             (weight != cutoff && fabsf(weight - cutoff) <= 2.e-6f * cutoff)))) {
        if constexpr (Round) if (!lane) *done = 0;
        return;
    }
    const double kept = weight >= cutoff ? weight : 0.f;
    Scan(tmp.scan).ExclusiveSum(kept, prefix, total);
    const float probability = float(kept / total);
    if (previous) previous[lane] = probability > 0.f ? id : -1;
    if (probability > 0.f) q[size_t(row) * stride + id] = probability;
    // The ordinary Philox draw traverses token IDs in ascending order.
    key = (static_cast<unsigned long long>(unsigned(id)) << 32) | __float_as_uint(probability);
    __syncthreads();
    key = SortGraphDrawKeys<false>(key, tmp.sort);
    __syncthreads();
    const float mass = __uint_as_float(unsigned(key));
    Scan(tmp.scan).ExclusiveSum(double(mass), prefix, total);
    if (!lane) {
        curandStatePhilox4_32_10_t state;
        curand_init(*seed, row, step, &state);
        draw = fmin(double(curand_uniform(&state)), 1.0 - 0x1p-33) * total;
        if constexpr (!Round) {
            done[row] = 1;
            fallbackK[row] = vocab;
        }
    }
    __syncthreads();
    if (prefix < draw && draw <= prefix + mass) tokens[size_t(row) * tokenStride] = int(key >> 32);
}

template<class T>
__global__ void ExpandGraphDrawScores(MarkovProposalScores<T> scores, float *expanded, size_t n) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) expanded[i] = float(scores(i / scores.vocab, i % scores.vocab));
}
} // namespace fastllm_markov
