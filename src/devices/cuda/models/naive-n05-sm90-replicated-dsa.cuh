#pragma once
#include "naive-n05-sm90-replicated-dsa.h"

#include "devices/cuda/fastllm-mxfp8-activation.cuh"
#include <cuda/atomic>
#include <cooperative_groups.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <unordered_map>

namespace {
#include "naive-n05-topk.cuh"
#include "naive-n05-indexer-topk-common.cuh"
#include "naive-n05-dsa-split.cuh"
namespace naive_sm90_dsa {
// The last attention CTA owns complete 32-channel groups. Quantize its
// rounded BF16 vector while it is in registers for the output projection.
struct PackOutput {
    static constexpr bool Packed = true;
    uint8_t *packed;
    float *scales;
    __device__ __forceinline__ void operator()(__nv_bfloat16 *out, size_t i, __nv_bfloat16 value) const {
        out[i] = value;
        const float x = float(value);
        float maximum = fmaxf(1.e-4f, fabsf(x));
#pragma unroll
        for (int delta = 16; delta; delta >>= 1)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, delta));
        const float scale = __uint_as_float(fastllm_mxfp8::ScaleExponent(maximum) << 23);
        if (!(threadIdx.x & 31)) scales[i / 32] = scale;
        packed[i] = __nv_fp8_e4m3(x / scale).__x;
    }

    // Each lane owns four adjacent output channels. Load them together
    // from each split and preserve the ordered FP32 reduction.
    __device__ void Finish(__nv_bfloat16 *out, size_t outputHead, const float *part, const float *stats,
                           const float *sink, int active) const {
        using namespace naive_dsa_split;
        const int lane = threadIdx.x % 32, head = threadIdx.x / 32;
        const float2 stat = lane < active ? reinterpret_cast<const float2 *>(stats)[lane * Heads + head]
                                    : make_float2(-INFINITY, 0.f);
        float maximum = stat.x;
#pragma unroll
        for (int d = 16; d; d /= 2) maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, d));
        maximum = __shfl_sync(0xffffffffu, maximum, 0);
        if (sink) maximum = fmaxf(maximum, *sink);
        const float correction = lane < active && maximum != -INFINITY ? expf(stat.x - maximum) : 0.f;
        float denominator = 0.f;
        float4 value{};
#pragma unroll
        for (int k = 0; k < Splits; ++k) {
            const float factor = __shfl_sync(0xffffffffu, correction, k);
            const float sum = __shfl_sync(0xffffffffu, stat.y, k);
            if (k < active) {
                denominator = fmaf(sum, factor, denominator);
                const float4 a = reinterpret_cast<const float4 *>(part + (k * Heads + head) * Values)[lane];
                value.x = fmaf(a.x, factor, value.x);
                value.y = fmaf(a.y, factor, value.y);
                value.z = fmaf(a.z, factor, value.z);
                value.w = fmaf(a.w, factor, value.w);
            }
        }
        if (sink) denominator += expf(*sink - maximum);
        const auto xy = denominator > 0.f ? __floats2bfloat162_rn(value.x / denominator, value.y / denominator)
                                          : __float2bfloat162_rn(0.f);
        const auto zw = denominator > 0.f ? __floats2bfloat162_rn(value.z / denominator, value.w / denominator)
                                          : __float2bfloat162_rn(0.f);
        const size_t col = outputHead * Values + lane * 4;
        reinterpret_cast<__nv_bfloat162 *>(out + col)[0] = xy;
        reinterpret_cast<__nv_bfloat162 *>(out + col)[1] = zw;
        const float2 a = __bfloat1622float2(xy), b = __bfloat1622float2(zw);
        float magnitude = fmaxf(1.e-4f, fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(b.x), fabsf(b.y))));
#pragma unroll
        for (int d = 4; d; d /= 2)
            magnitude = fmaxf(magnitude, __shfl_xor_sync(0xffffffffu, magnitude, d, 8));
        const unsigned exponent = fastllm_mxfp8::ScaleExponent(magnitude);
        const float scale = __uint_as_float(exponent << 23);
        const auto x = __nv_cvt_float2_to_fp8x2(make_float2(a.x / scale, a.y / scale), __NV_SATFINITE, __NV_E4M3);
        const auto y = __nv_cvt_float2_to_fp8x2(make_float2(b.x / scale, b.y / scale), __NV_SATFINITE, __NV_E4M3);
        *reinterpret_cast<uint32_t *>(packed + col) = uint32_t(x) | uint32_t(y) << 16;
        if (!(lane & 7)) scales[col / 32] = scale;
    }
};
struct Verify {
    PackOutput pack;
    FastllmNaiveDSARow binding[8];
    int rows;
    const __nv_bfloat16 *query;
    const float *sink;
    __nv_bfloat16 *output;
    unsigned *scores;
    const __nv_bfloat16 *index;
    const float *scales;
    naive_topk::Workspace *select;
    float *attention;
    unsigned *completion;
    int *selected;
};

// Two warp groups cover the 128 Indexer heads of an eight-row verification.
// Keep their Q tile resident while the CTA traverses successive pages.
__device__ void ScoreTiles(const Verify &v) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    using namespace naive_indexer;
    constexpr int M = 8 * 16;
    extern __shared__ __align__(1024) unsigned char tile_storage[];
    BF *sq = reinterpret_cast<BF *>(tile_storage), *sk = sq + M * 128;
    float *partial = reinterpret_cast<float *>(sk + 2 * 64 * 128);
    float *sc = partial + M * 64, *weights = sc + M;
    const int t = threadIdx.x, lane = t % 32, warp = t / 32;
    const int length = lane < v.rows ? max(0, min(v.binding[lane].capacity, *v.binding[lane].liveKeys)) : 0;
    const int longest = __reduce_max_sync(0xffffffffu, length);
    const int firstLength = __shfl_sync(0xffffffffu, length, t / 64);
    const int secondLength = __shfl_sync(0xffffffffu, length, 4 + t / 64);
    if (blockIdx.x * 64 >= longest) return;
    for (int i = t; i < M * 16; i += 256) {
        const int head = i / 16, d = i % 16 * 8;
        const BF *src = v.index + min(head / 16, v.rows - 1) * 2192 + head % 16 * 128 + d;
        naive_dsa_decode::Copy<true>(sq + offset(head, d, M), src, head / 16 < v.rows);
    }
    if (t < M) {
        sc[t] = t / 16 < v.rows ? v.scales[t] : 0.f;
        weights[t] = t / 16 < v.rows ? float(v.index[t / 16 * 2192 + 2176 + t % 16]) : 0.f;
    }
    const auto r = FastllmNaiveResolveDSARow(v.binding[0]);
    // Keep Q resident and stage the next page while WGMMA consumes the current one.
    auto loadKeys = [&](int first, BF *dst) {
        const bool fullPageTile = longest - first >= 64 && r.pageLen > 0 && r.pageLen % 64 == 0;
        const int base = fullPageTile ? FastllmNaivePhysicalRow(r, first) : 0;
        for (int i = t; i < 64 * 16; i += 256) {
            const int token = first + i / 16, d = i % 16 * 8;
            const int physical = fullPageTile ? base + i / 16
                                              : FastllmNaivePhysicalRow(r, min(token, longest - 1));
            const BF *src = static_cast<const BF *>(r.key) + size_t(physical) * r.keyStride + 192 + d;
            naive_dsa_decode::Copy<true>(dst + offset(i / 16, d, 64), src, token < longest);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    const int stride = gridDim.x * 64;
    int phase = 0;
    loadKeys(blockIdx.x * 64, sk);
    for (int first = blockIdx.x * 64; first < longest; first += stride, phase ^= 1) {
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        const bool nextTile = longest - first > stride;
        if (nextTile) loadKeys(first + stride, sk + (phase ^ 1) * 64 * 128);
        asm volatile("fence.proxy.async.shared::cta; wgmma.fence.sync.aligned;" ::: "memory");
        float dot[32]{};
#pragma unroll
        for (int d = 0; d < 128; d += 16)
            mma(dot, descriptor(sq + offset(t / 128 * 64, d, M)), descriptor(sk + phase * 64 * 128 + offset(0, d, 64)));
        asm volatile("wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;" ::: "memory");
#pragma unroll
        for (int n = 0; n < 8; ++n)
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int m = warp * 16 + lane / 4 + half * 8, col = n * 8 + (lane % 4) * 2;
                *reinterpret_cast<float2 *>(partial + m * 64 + (col ^ ((m & 7) * 8))) =
                    make_float2(dot[n * 4 + half * 2], dot[n * 4 + half * 2 + 1]);
            }
        __syncthreads();
        for (int i = t; i < v.rows * 64; i += 256) {
            const int row = i / 64, col = i % 64;
            if (first + col >= (i < 256 ? firstLength : secondLength)) continue;
            float sum = 0;
#pragma unroll
            for (int h = 0; h < 16; ++h) {
                const int m = row * 16 + h;
                sum = fmaf(fmaxf(partial[m * 64 + (col ^ ((m & 7) * 8))] * sc[m], 0.f), weights[m], sum);
            }
            v.scores[size_t(row) * r.capacity + first + col] = naive_topk::OrderedScoreBits(sum);
        }
        if (!nextTile) break;
        __syncthreads();
    }
#endif
}

enum CutoffState { Bounded, NeedsRadix, ExactCutoff };

// A coarse bucket can contain more candidates than the stable sort can hold.
// Refine its exact FP32 cutoff in parallel, then reserve the remaining equal
// scores by their logical position. Scratch stays bounded at every length.
__device__ void SelectHeavy(const Verify &v) {
    // Radix refinement uses 256 bins per partition, so more resident CTAs
    // can share a long row than in the coarse, 2048-bin selection.
    constexpr int MaxParts = sizeof(naive_topk::Workspace::ties) / (2 * sizeof(int));
    static_assert(MaxParts * 256 <= sizeof(naive_topk::Workspace::partials) / sizeof(int));
    const int parts = min(MaxParts, int(gridDim.x) / v.rows);
    const int query = blockIdx.x / parts, part = blockIdx.x % parts;
    const int row = min(query, v.rows - 1);
    const auto &desc = v.binding[row];
    const int n = max(0, min(desc.capacity, *desc.liveKeys)), top = min(n, naive_topk::kTopK);
    const int chunk = (n + parts * 32 - 1) / (parts * 32) * 32;
    const int begin = part * chunk, end = min(n, begin + chunk);
    const unsigned *scores = v.scores + size_t(row) * desc.capacity;
    auto &w = v.select[row];
    constexpr int Threads = 256;
    using Scan = cub::BlockScan<unsigned, Threads>;
    __shared__ typename Scan::TempStorage scan;
    extern __shared__ __align__(16) unsigned char refine_storage[];
    auto *hist = reinterpret_cast<unsigned *>(refine_storage);
    const bool assigned = query < v.rows;
    const int t = threadIdx.x, lane = t & 31;
    auto grid = cooperative_groups::this_grid();
    // Only the overflowing coarse bucket needs further radix digits. Scores
    // in higher buckets have already consumed their share of the target rank.
    const unsigned low = w.state.prefix;
    const bool needsRadix = assigned && w.state.done == NeedsRadix;
    const int coarseBin = needsRadix ? w.state.candidates : 0;
    const unsigned upper = coarseBin + 1 < naive_indexer_ordered_topk::Buckets
                               ? naive_indexer_ordered_topk::LowerBound(coarseBin + 1) - 1
                               : ~0u;
    const unsigned span = needsRadix ? min(upper, w.state.maximum) - low : 0u;
    const int firstShift = (31 - __clz(max(1u, span))) & ~7;
    for (int shift = 24; shift >= 0; shift -= 8) {
        bool active = false;
        for (int row = 0; row < v.rows; ++row) active |= v.select[row].state.done == NeedsRadix;
        if (!active) break;
        const bool refining = needsRadix && w.state.done == NeedsRadix && shift <= firstShift;
        const unsigned prefix = shift >= firstShift ? 0u : w.state.prefix - low;
        const unsigned mask = shift >= firstShift ? 0u : ~0u << (shift + 8);
        if (refining) {
            hist[t] = 0;
            __syncthreads();
            for (int base = begin; base < end; base += Threads) {
                const int i = base + t;
                const unsigned key = i < end ? scores[i] - low : 0;
                const unsigned bin = i < end && key <= span && (key & mask) == prefix ? (key >> shift) & 255u : 256u;
                const unsigned peers = __match_any_sync(0xffffffffu, bin);
                if (bin < 256 && lane == __ffs(peers) - 1) atomicAdd(hist + bin, __popc(peers));
            }
            __syncthreads();
            w.partials[part * 256 + t] = hist[t];
        }
        grid.sync();
        if (refining && !part) {
            const unsigned bin = 255 - t;
            unsigned count = 0, before;
            for (int p = 0; p < parts; ++p) count += w.partials[p * 256 + bin];
            const int need = w.state.rank;
            Scan(scan).ExclusiveSum(count, before);
            if (before < need && before + count >= need) {
                const unsigned next = low + (prefix | (bin << shift));
                w.state.prefix = next;
                w.state.rank = need - before;
                if (shift == 0 || next == w.state.maximum) w.state.done = ExactCutoff;
            }
        }
        grid.sync();
    }
    const bool heavy = assigned && w.state.done == ExactCutoff;
    if (heavy) {
        const unsigned cut = w.state.prefix;
        unsigned higher = 0, equal = 0;
        for (int i = begin + t; i < end; i += Threads) {
            const unsigned key = scores[i];
            higher += key > cut;
            equal += key == cut;
        }
        higher = __reduce_add_sync(0xffffffffu, higher);
        equal = __reduce_add_sync(0xffffffffu, equal);
        if (!lane) {
            hist[t / 32] = higher;
            hist[8 + t / 32] = equal;
        }
        __syncthreads();
        if (!t) {
            unsigned h = 0, e = 0;
            for (int warp = 0; warp < 8; ++warp) {
                h += hist[warp];
                e += hist[8 + warp];
            }
            w.ties[part] = h;
            w.ties[parts + part] = e;
        }
    }
    grid.sync();
    if (heavy) {
        const unsigned cut = w.state.prefix;
        const int keepEqual = w.state.rank, higherTotal = top - keepEqual;
        unsigned h = 0, e = 0;
        for (int p = 0; p < part; ++p) {
            h += w.ties[p];
            e += w.ties[parts + p];
        }
        using PairScan = cub::BlockScan<unsigned long long, Threads>;
        __shared__ typename PairScan::TempStorage pairs;
        for (int base = begin; base < end && (higherTotal || e < keepEqual); base += Threads * 4) {
            const int first = base + t * 4;
            const uint4 load =
                naive_indexer_ordered_topk::LoadFour(scores, first, end, (uintptr_t(scores) & 15) == 0);
            const unsigned key[] = {load.x, load.y, load.z, load.w};
            unsigned long long counts = 0, prior, total;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                if (first + j < end) counts += key[j] > cut ? 1ull << 32 : unsigned(key[j] == cut);
            PairScan(pairs).ExclusiveSum(counts, prior, total);
            __syncthreads();
            unsigned highPos = h + unsigned(prior >> 32), eqPos = e + unsigned(prior);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (first + j >= end) continue;
                unsigned dest;
                if (key[j] > cut)
                    dest = highPos++;
                else if (key[j] == cut && eqPos < keepEqual)
                    dest = higherTotal + eqPos++;
                else
                    continue;
                w.candidates.keys[dest] = key[j] - cut;
                w.candidates.positions[dest] = first + j;
            }
            h += unsigned(total >> 32);
            e += unsigned(total);
        }
    }
}

constexpr size_t ScoreBytes = (128 + 2 * 64) * 128 * sizeof(__nv_bfloat16) + (128 * 64 + 2 * 128) * sizeof(float);
constexpr size_t Shared = std::max(sizeof(naive_dsa_split::Storage), ScoreBytes);

struct ScoreDigit {
    int shift;
    __device__ unsigned Digit(unsigned key) const { return (key >> shift) & 255u; }
};

// Warp-striped input retains logical token order on ties. Exchange each score
// together with its position, once per radix pass.
template <int Items>
__device__ void WriteSortedPositions(const naive_topk::Candidates &candidates, int count, int bits,
                                     unsigned char *storage, int *indices) {
    constexpr int Threads = naive_indexer_ordered_topk::Threads;
    using Rank = cub::BlockRadixRankMatch<Threads, 8, true>;
    struct SortStorage {
        typename Rank::TempStorage rank;
        uint2 pairs[Threads * Items];
    };
    static_assert(sizeof(SortStorage) <= Shared);
    auto &shared = *reinterpret_cast<SortStorage *>(storage);
    const int t = threadIdx.x, first = (t / 32) * (32 * Items) + t % 32;
    unsigned keys[Items];
    int positions[Items], ranks[Items];
#pragma unroll
    for (int j = 0; j < Items; ++j) {
        const int i = first + 32 * j;
        keys[j] = i < count ? candidates.keys[i] : 0;
        positions[j] = i < count ? candidates.positions[i] : -1;
    }
    for (int shift = 0; shift < bits; shift += 8) {
        Rank(shared.rank).RankKeys(keys, ranks, ScoreDigit{shift});
#pragma unroll
        for (int j = 0; j < Items; ++j)
            shared.pairs[ranks[j]] = make_uint2(keys[j], static_cast<unsigned>(positions[j]));
        __syncthreads();
        if (shift + 8 >= bits) break;
#pragma unroll
        for (int j = 0; j < Items; ++j) {
            const uint2 pair = shared.pairs[first + j * 32];
            keys[j] = pair.x;
            positions[j] = static_cast<int>(pair.y);
        }
        __syncthreads();
    }
    for (int i = t; i < naive_topk::kTopK; i += Threads)
        indices[i] = static_cast<int>(shared.pairs[i].y);
}

// Each partition owns a contiguous interval. The histogram picks a monotone
// lower bound only; the final stable sort uses the original FP32 score bits.
__device__ void SelectPartitioned(const Verify &v) {
    using namespace naive_indexer_ordered_topk;
    using Scan = cub::BlockScan<unsigned, Threads>;
    constexpr int MaxParts = 8, Groups = Buckets / Threads, Items = Buckets / Threads;
    static_assert(Buckets % Threads == 0 && Items % 4 == 0 && Groups <= 32);
    static_assert(MaxParts * (Buckets + Groups) <= sizeof(naive_topk::Workspace::partials) / sizeof(int));
    __shared__ typename Scan::TempStorage scan;
    extern __shared__ __align__(16) unsigned char select_storage[];
    auto *hist = reinterpret_cast<unsigned *>(select_storage);
    const int t = threadIdx.x, partitions = min(MaxParts, int(gridDim.x) / v.rows);
    const int query = blockIdx.x / partitions, part = blockIdx.x % partitions;
    const bool assigned = query < v.rows;
    const auto desc = v.binding[min(query, v.rows - 1)];
    const int n = max(0, min(desc.capacity, *desc.liveKeys));
    const int chunk = (n + partitions * 32 - 1) / (partitions * 32) * 32;
    const int begin = part * chunk, end = min(n, begin + chunk);
    auto *w = v.select + min(query, v.rows - 1);
    auto *scores = v.scores + size_t(min(query, v.rows - 1)) * desc.capacity;
    constexpr int CacheOffset = Buckets + Threads / 32;
    const bool cacheScores = chunk <= int(Shared / sizeof(unsigned)) - CacheOffset;
    auto *cached = hist + CacheOffset;
    auto *coarse = w->partials + MaxParts * Buckets;
    if (assigned) {
        for (int i = t; i < Buckets; i += Threads) hist[i] = 0;
        __syncthreads();
        unsigned largest = 0;
        for (int i = begin + t; i < end; i += Threads) {
            const unsigned key = scores[i];
            if (cacheScores) cached[i - begin] = key;
            largest = max(largest, key);
            atomicAdd(hist + Bucket(key), 1u);
        }
        largest = naive_topk::WarpMax(largest);
        if (!(t & 31)) hist[Buckets + t / 32] = largest;
        __syncthreads();
        if (!t) {
            unsigned m = 0;
            for (int i = 0; i < Threads / 32; ++i) m = max(m, hist[Buckets + i]);
            reinterpret_cast<unsigned *>(w->ties)[part] = m;
        }
        // Each entry counts the scores at this bucket or higher. Keeping every
        // 256th entry separately lets all partitions locate the cutoff in two reads.
        unsigned counts[Items], suffix[Items];
        const int firstBin = Buckets - (t + 1) * Items;
#pragma unroll
        for (int j = 0; j < Items; ++j) counts[j] = hist[firstBin + Items - 1 - j];
        Scan(scan).InclusiveSum(counts, suffix);
        auto *dest = reinterpret_cast<uint4 *>(w->partials + part * Buckets + firstBin);
#pragma unroll
        for (int j = 0; j < Items; j += 4)
            dest[(Items - 4 - j) / 4] = make_uint4(suffix[j + 3], suffix[j + 2], suffix[j + 1], suffix[j]);
        if (firstBin % Threads == 0) coarse[part * Groups + firstBin / Threads] = suffix[Items - 1];
    }
    auto grid = cooperative_groups::this_grid();
    grid.sync();
    const int top = min(n, naive_topk::kTopK), lane = t % 32;
    __shared__ unsigned threshold, selected, partitionOffset;
    if (assigned) {
        int groupCount = 0;
        if (lane < Groups)
            for (int p = 0; p < partitions; ++p) groupCount += coarse[p * Groups + lane];
        const unsigned groups = __ballot_sync(0xffffffffu, lane < Groups && groupCount >= top);
        const int group = top ? __popc(groups) - 1 : 0;
        const int bin = group * Threads + t;
        int count = 0;
        for (int p = 0; p < partitions; ++p) count += w->partials[p * Buckets + bin];
        const int last = top ? __syncthreads_count(count >= top) - 1 : 0;
        const int cut = group * Threads + last;
        if (t == last) {
            threshold = top ? LowerBound(cut) : 0;
            selected = top ? count : 0;
        }
        // The same suffix counts give each contiguous partition its stable output offset.
        unsigned prefix = lane < part ? w->partials[lane * Buckets + cut] : 0;
        prefix = __reduce_add_sync(0xffffffffu, prefix);
        if (!t) partitionOffset = prefix;
        unsigned largest = lane < partitions ? reinterpret_cast<unsigned *>(w->ties)[lane] : 0;
        largest = __reduce_max_sync(0xffffffffu, largest);
        __syncthreads();
        if (!part && !t) {
            w->state.prefix = threshold;
            w->state.selected = selected;
            w->state.maximum = largest;
            w->state.done = selected <= naive_topk::kCapacity ? Bounded
                            : threshold == largest            ? ExactCutoff
                                                              : NeedsRadix;
            w->state.rank = top;
            if (w->state.done == NeedsRadix) {
                if (cut + 1 < Buckets)
                    for (int p = 0; p < partitions; ++p) w->state.rank -= w->partials[p * Buckets + cut + 1];
                w->state.candidates = cut;
            }
        }
    }
    if (assigned && selected <= naive_topk::kCapacity) {
        int target = partitionOffset;
        const unsigned low = threshold;
        for (int base = begin; base < end; base += Threads * 4) {
            const int first = base + t * 4;
            const uint4 keys = cacheScores ? LoadFour(cached, first - begin, end - begin, true)
                                           : LoadFour(scores, first, end, (uintptr_t(scores) & 15) == 0);
            const unsigned key[4] = {keys.x, keys.y, keys.z, keys.w};
            unsigned count = 0, cursor, batch;
#pragma unroll
            for (int j = 0; j < 4; ++j) count += first + j < end && key[j] >= low;
            Scan(scan).ExclusiveSum(count, cursor, batch);
            __syncthreads();
#pragma unroll
            for (int j = 0; j < 4; ++j)
                if (first + j < end && key[j] >= low) {
                    w->candidates.keys[target + cursor] = key[j] - low;
                    w->candidates.positions[target + cursor++] = first + j;
                }
            target += batch;
        }
    }
    grid.sync();
    bool heavy = false;
    for (int row = 0; row < v.rows; ++row) heavy |= v.select[row].state.done != Bounded;
    if (heavy) {
        SelectHeavy(v);
        grid.sync();
    }
    if (assigned && !part) {
        if (w->state.done == ExactCutoff) {
            if (!t) w->state.selected = min(n, naive_topk::kTopK);
            __syncthreads();
        }
        const int bits = max(1, 32 - __clz(w->state.maximum - w->state.prefix));
        int *indices = const_cast<int *>(desc.indices);
        if (w->state.selected <= naive_topk::kTopK)
            WriteSortedPositions<8>(w->candidates, w->state.selected, bits, select_storage, indices);
        else if (w->state.selected <= Threads * 10)
            WriteSortedPositions<10>(w->candidates, w->state.selected, bits, select_storage, indices);
        else
            WriteSortedPositions<16>(w->candidates, w->state.selected, bits, select_storage, indices);
    }
}

__global__ __launch_bounds__(256, 1) void SelectAttention(const __grid_constant__ Verify v) {
    const int index = blockIdx.x;
    if (v.binding[0].indices) {
        ScoreTiles(v);
        cooperative_groups::this_grid().sync();
        SelectPartitioned(v);
    } else {
        const int width = v.binding[0].capacity;
        for (int i = index * blockDim.x + threadIdx.x; i < v.rows * width; i += gridDim.x * blockDim.x)
            v.selected[i] = i % width < *v.binding[i / width].liveKeys ? i % width : -1;
    }
    if (index < v.rows && !threadIdx.x) v.completion[index] = 0;
    cooperative_groups::this_grid().sync();
    for (int task = index; task < v.rows * naive_dsa_split::Splits; task += gridDim.x) {
        const int row = task / naive_dsa_split::Splits;
        naive_dsa_split::Attend(v.query, v.binding[row], v.sink, v.output, v.attention, v.completion, 8, 1, row, 0,
                                task % naive_dsa_split::Splits, v.pack);
        __syncthreads();
    }
}
static_assert(sizeof(naive_topk::Workspace) % alignof(uint4) == 0);
static_assert(offsetof(naive_topk::Workspace, partials) % alignof(uint4) == 0);

struct ScratchLayout {
    size_t candidates, completion, bytes;
    ScratchLayout(int rows, int capacity) {
        const size_t index = size_t(rows) * (2192 * sizeof(__nv_bfloat16) + 16 * sizeof(float));
        const size_t scores = capacity > 2048 ? size_t(rows) * capacity * sizeof(unsigned) : 0;
        candidates = (index + scores + alignof(uint4) - 1) / alignof(uint4) * alignof(uint4);
        const size_t selected = candidates + (scores ? rows * sizeof(naive_topk::Workspace) : 0);
        const size_t attention = size_t(rows) * naive_dsa_split::Scratch * sizeof(float);
        completion = std::max(selected, attention);
        bytes = completion + rows * sizeof(unsigned);
    }
};
} // namespace naive_sm90_dsa
} // namespace

bool FastllmSm90DsaSelectConfig(int device, int rows, int capacity, int &grid, size_t &bytes) {
    using namespace naive_sm90_dsa;
    if (rows < 1 || rows > 8 || capacity < 1) return false;
    static thread_local std::unordered_map<int, int> resident;
    auto it = resident.find(device);
    if (it == resident.end()) {
        if (FastllmCudaGraphIsCapturingFast()) return false;
        int blocks = 0, sm = 0;
        if (cudaFuncSetAttribute(SelectAttention, cudaFuncAttributeMaxDynamicSharedMemorySize, Shared) !=
                cudaSuccess ||
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, SelectAttention, 256, Shared) != cudaSuccess ||
            cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        it = resident.emplace(device, sm * blocks).first;
    }
    // A long Indexer scan can use resident CTAs beyond the 16 attention splits per row.
    const int scoreTiles = capacity > 2048 ? 1 + (capacity - 1) / 64 : 0;
    grid = std::min(std::max(rows * naive_dsa_split::Splits, scoreTiles), it->second);
    // The attention partials can overwrite all Indexer scratch after selection.
    bytes = ScratchLayout(rows, capacity).bytes;
    return grid >= rows && bytes / sizeof(float) + size_t(rows) * (1152 + 8 * 2048) < INT_MAX;
}

cudaError_t FastllmSm90DsaSelectAttention(const std::vector<FastllmNaiveDSARow> &rows, int grid, unsigned *scores,
                                          const __nv_bfloat16 *query, const float *sink, __nv_bfloat16 *out,
                                          float *scratch, int *selected, const __nv_bfloat16 *index,
                                          const float *scales, uint8_t *packed, float *outScales) {
    using namespace naive_sm90_dsa;
    Verify v{};
    v.rows = int(rows.size());
    const int cap = rows.front().capacity;
    const ScratchLayout layout(v.rows, cap);
    auto *base = reinterpret_cast<unsigned char *>(scratch);
    v.select = cap > 2048 ? reinterpret_cast<naive_topk::Workspace *>(base + layout.candidates) : nullptr;
    v.scores = scores;
    v.pack = {packed, outScales};
    v.index = index;
    v.scales = scales;
    v.attention = scratch;
    v.completion = reinterpret_cast<unsigned *>(base + layout.completion);
    v.selected = selected;
    v.query = query;
    v.output = out;
    v.sink = sink;
    for (int row = 0; row < v.rows; ++row) {
        v.binding[row] = rows[row];
        v.binding[row].indices = cap > 2048 ? selected + row * 2048 : nullptr;
        v.binding[row].selectedCount = cap > 2048 ? 2048 : 0;
    }
    void *args[] = {&v};
    return cudaLaunchCooperativeKernel(reinterpret_cast<const void *>(SelectAttention), grid, 256, args, Shared,
                                       cudaStreamPerThread);
}
