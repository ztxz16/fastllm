#pragma once
#include "naive-n05-dsa-mma.cuh"
#include "naive-n05-topk.cuh"
#include <type_traits>

// Keep launch stubs in SM70/SM75-only builds too; large arguments require SM70.
// Tensor Core bodies and runtime dispatch still require SM80.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
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
    int cpRank = 0, cpRanks = 1;
    int contextPageLen = 0;
    __device__ int LocalCount(int rank) const {
        int localPage = contextPageLen / cpRanks;
        return (*live / contextPageLen) * localPage + min(localPage, max(0, *live % contextPageLen - rank * localPage));
    }
    __device__ int LogicalPosition(int position) const {
        if (!contextPageLen) return Begin() + position;
        int localPage = contextPageLen / cpRanks;
        return position / localPage * contextPageLen + cpRank * localPage + position % localPage;
    }
    const unsigned *merged = nullptr;
    int mergedRows = 0, mergedRow = 0;
    __device__ int Chunk() const {
        return ((*live + cpRanks * 64 - 1) / (cpRanks * 64)) * 64;
    }
    __device__ int Begin() const { return contextPageLen ? 0 : cpRanks > 1 ? cpRank * Chunk() : 0; }
    __device__ int MergeWidth() const { return min(Chunk(), naive_topk::kTopK); }
    __device__ int Count() const {
        int n = *live;
        if (contextPageLen && !merged) return LocalCount(cpRank);
        if (merged) {
            if (!n) return 0;
            int chunk = Chunk();
            return (n / chunk) * min(chunk, naive_topk::kTopK) + min(n % chunk, naive_topk::kTopK);
        }
        return cpRanks > 1 ? max(0, min(n - Begin(), Chunk())) : n;
    }
    template<bool Merge> __device__ unsigned Score(int i, int width) const {
        if constexpr (Merge) return merged[((size_t)(i / width) * mergedRows + mergedRow) *
            naive_topk::kTopK * 2 + (i % width) * 2];
        else return scores[i];
    }
    template<bool Merge> __device__ int Position(int i, int width) const {
        if constexpr (Merge) return (int)merged[((size_t)(i / width) * mergedRows + mergedRow) *
            naive_topk::kTopK * 2 + (i % width) * 2 + 1];
        else return i;
    }
};
// CUDA 12.1 raised the kernel-argument limit from 4 KiB to 32,764 bytes.
#if CUDART_VERSION >= 12010
constexpr int ArgumentBytes = 32764;
#else
constexpr int ArgumentBytes = 4096;
#endif
// Leave room for the other arguments and their alignment.
constexpr int BatchLimit = (ArgumentBytes - 128) / sizeof(Row);
struct Batch { Row rows[BatchLimit]; };

__device__ __forceinline__ Row TopKRow(const Batch &batch, int i) {
    return batch.rows[i];
}
#if CUDART_VERSION >= 12010
struct LocalTopKRow {
    const int *live;
    unsigned *scores;
    naive_topk::Workspace *workspace;
    int *indices;
};
constexpr int kLocalTopKLimit = 512;
struct LocalTopKBatch {
    int rank, ranks, page;
    LocalTopKRow rows[kLocalTopKLimit];
};
static_assert(sizeof(LocalTopKRow) == 32);
static_assert(sizeof(LocalTopKBatch) + 128 <= ArgumentBytes);
__device__ __forceinline__ Row TopKRow(const LocalTopKBatch &batch, int i) {
    const auto &x = batch.rows[i];
    Row row{};
    row.live=x.live; row.scores=x.scores;
    row.workspace=x.workspace; row.indices=x.indices;
    row.cpRank=batch.rank; row.cpRanks=batch.ranks; row.contextPageLen=batch.page;
    return row;
}
#endif


// Long mixed-context batches used to launch one CTA for every key tile of the
// longest row. Keep enough CTAs to fill Hopper, then let each CTA process a
// strided sequence of tiles. Scores and TopK order remain unchanged.
// Descriptors travel in the launch arguments and remain fixed in a graph.
// Device lengths and optional KV bindings change without rebinding scratch.
__global__ void Quantize(const BF16 *query, Batch batch) {
    const Row &row = batch.rows[blockIdx.y];
    if (*row.live <= 0) return;
    naive_dsa_mma::QuantizeIndexerRow(query + (size_t)row.sequence *
        naive_dsa_mma::kHeads * naive_dsa_mma::kIndexerDim,
        row.query, row.scale, naive_dsa_mma::kIndexerDim, 0, false);
}
struct ScoreOutput {
    unsigned *bits;
    int first = 0;
    __device__ void Store(int index, float value) const {
        bits[index - first] = naive_topk::OrderedScoreBits(value);
    }
};
__global__ void Scores(const BF16 *weights, Batch batch) {
    const Row &row = batch.rows[blockIdx.y];
    const int count = row.Count(), first = row.Begin(), end = first + count;
    if (end <= first) return;
    auto key = row.key;
    key.Bind(row.cache);
    for (int offset = 0;; ++offset) {
        const int tile = (offset * (int)gridDim.x + blockIdx.x) * naive_dsa_mma::kKeys;
        if (first + tile >= end) break;
        naive_dsa_mma::IndexerDecodeScoresRow(row.query, key, row.scale,
            weights + (size_t)row.sequence * naive_dsa_mma::kHeads,
            ScoreOutput{row.scores, first}, row.stride, end, end - 1, first, offset);
        // The next tile reuses the helper's shared-memory staging area.
        __syncthreads();
    }
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
#endif

__device__ __forceinline__ void MmaFp8(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
#if CUDART_VERSION >= 12040 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

// K occupies the MMA row dimension: each warp produces sixteen key scores.
// Cache the reused Q in L1, while streaming K directly to shared memory.
// The caller guarantees 16-byte alignment and retains Scores as the fallback.
template<bool FP8, bool NativeFp8 = false>
__global__ void HopperScores(const BF16 *weights, Batch batch) {
    static_assert(!NativeFp8 || FP8, "Native FP8 MMA requires FP8 keys.");
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // Older-toolkit builds and older PTX images retain the BF16 MMA path,
    // including when that PTX is JIT-compiled on a newer GPU.
#if CUDART_VERSION >= 12040 && __CUDA_ARCH__ >= 890
    constexpr bool UseNativeFp8 = NativeFp8;
#else
    constexpr bool UseNativeFp8 = false;
#endif
    using namespace naive_dsa_mma;
    constexpr int Threads = 128;
    Row r = batch.rows[blockIdx.y];
    r.key.Bind(r.cache);
    const int begin = r.Begin(), n = begin + r.Count();
    const int tileStride = (int)gridDim.x * kKeys;
    if (begin + blockIdx.x * kKeys >= n) return;
    using Tile = std::conditional_t<UseNativeFp8, unsigned char, BF16>;
    constexpr int Padding = UseNativeFp8 ? 16 : 8, Step = UseNativeFp8 ? 32 : 16;
    __shared__ __align__(16) Tile qs[kHeads][kIndexerDim + Padding], ks[kKeys][kIndexerDim + Padding];
    __shared__ float scales[kHeads], weightsF[kHeads];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    if constexpr (UseNativeFp8) {
        // Quantize already stores exact E4M3 values in BF16. Repacking those
        // values adds no quantization; keep their original FP32 scales below.
        for (int i = t; i < kHeads * kIndexerDim; i += Threads)
            qs[i / kIndexerDim][i % kIndexerDim] =
                __nv_cvt_float_to_fp8((float)r.query[i], __NV_SATFINITE, __NV_E4M3);
    } else for (int i = t; i < kHeads * (kIndexerDim / 8); i += Threads) {
        int h = i / (kIndexerDim / 8), d = i % (kIndexerDim / 8) * 8;
        LoadScoreVector<true>(&qs[h][d], r.query + h * kIndexerDim + d, true);
    }
    if (t < kHeads) {
        scales[t] = r.scale[t];
        weightsF[t] = (float)weights[r.sequence * kHeads + t];
    }
    asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    for (int first = begin + blockIdx.x * kKeys; first < n; first += tileStride) {
        // Tiles aligned to a divisor of the page length need only one page base.
        // Keep per-row translation for arbitrary page lengths (e.g. 96 tokens).
        const bool onePage = !r.key.pages || r.key.pageLen % kKeys == 0;
        const auto tile = onePage ? r.key.IndexRow(first, r.stride) : FastllmNaiveKVRow{};
        const int byteStride = r.stride * (r.key.fp8 ? 1 : 2);
        if constexpr (UseNativeFp8) {
            for (int i = t; i < kKeys * (kIndexerDim / 16); i += Threads) {
                int key = i / (kIndexerDim / 16), d = i % (kIndexerDim / 16) * 16;
                const auto base = onePage ? FastllmNaiveKVRow((const char *)tile.data + (size_t)key * byteStride, true)
                                          : r.key.IndexRow(min(first + key, n - 1), r.stride);
                LoadScoreVector<false>(&ks[key][d], (const unsigned char *)base.data + d, first + key < n);
            }
        } else if constexpr (FP8) {
            __shared__ __align__(16) unsigned char packed[kKeys][kIndexerDim];
            for (int i = t; i < kKeys * (kIndexerDim / 16); i += Threads) {
                int key = i / (kIndexerDim / 16), d = i % (kIndexerDim / 16) * 16;
                const auto base = onePage ? FastllmNaiveKVRow((const char *)tile.data + (size_t)key * byteStride, FP8)
                                          : r.key.IndexRow(min(first + key, n - 1), r.stride);
                LoadScoreVector<false>(&packed[key][d],
                    (const unsigned char *)base.data + d, first + key < n);
            }
            asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
            __syncthreads();
            for (int i = t; i < kKeys * (kIndexerDim / 8); i += Threads) {
                int key = i / (kIndexerDim / 8), d = i % (kIndexerDim / 8) * 8;
                *(uint4 *)&ks[key][d] = FastllmNaiveKVRow(&packed[key][d], true).Load8();
            }
        } else {
            for (int i = t; i < kKeys * (kIndexerDim / 8); i += Threads) {
                int key = i / (kIndexerDim / 8), d = i % (kIndexerDim / 8) * 8;
                const auto base = onePage ? FastllmNaiveKVRow((const char *)tile.data + (size_t)key * byteStride, FP8)
                                          : r.key.IndexRow(min(first + key, n - 1), r.stride);
                LoadScoreVector<false>(&ks[key][d], (base + d).data, first + key < n);
            }
        }
        if constexpr (!FP8 || UseNativeFp8)
            asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        float dot[2][4] = {};
#pragma unroll
        for (int d = 0; d < kIndexerDim; d += Step) {
            uint32_t a[4];
            Load4(a, &ks[warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8][d + (lane >> 4) * (Step / 2)]);
#pragma unroll
            for (int group = 0; group < 2; ++group) {
                uint32_t b[2];
                Load2(b, &qs[group * 8 + (lane & 7)][d + ((lane >> 3) & 1) * (Step / 2)]);
                if constexpr (UseNativeFp8) MmaFp8(dot[group], a, b);
                else Mma(dot[group], a, b);
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
            if ((lane & 3) == 0 && key < n) r.scores[key - begin] = naive_topk::OrderedScoreBits(sum * r.key.IndexScale(key, r.stride));
        }
        // Do not let the next tile overwrite shared K while another warp is
        // still finishing the current MMA result.
        __syncthreads();
    }
#endif
}

// A request owns one CTA instead of a cooperative grid. Independent requests
// fill the GPU together. Refine a radix threshold until at most 4096 candidates
// remain, then compact in position order for the existing stable block sort.
template<bool Merge = false>
__global__ void Select(Batch batch) {
    using namespace naive_topk;
    const Row &row = batch.rows[blockIdx.x];
    const int mergeWidth = row.MergeWidth();
    auto *out = &row.workspace->candidates;
    auto *state = &row.workspace->state;
    const int t = threadIdx.x, lane = t & 31, n = row.Count();
    using Scan = cub::BlockScan<int, kThreads>;
    __shared__ typename Scan::TempStorage scan;
    __shared__ int hist[kBuckets];
    __shared__ unsigned maxima[kThreads / 32];
    unsigned maximum = 0, reference = n ? row.Score<Merge>(0, mergeWidth) : 0;
    bool different = false;
    for (int i = t; i < n; i += kThreads) {
        unsigned bits = row.Score<Merge>(i, mergeWidth);
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
            unsigned bits = i < n ? row.Score<Merge>(i, mergeWidth) : 0;
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
        unsigned bits = i < n ? row.Score<Merge>(i, mergeWidth) : 0;
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
            out->positions[offset + before] = row.Position<Merge>(i, mergeWidth);
        }
        offset += total;
    }
    if (t == 0) state->selected = offset;
}
// Multi-CTA radix selection. Kernel boundaries provide the grid barriers, so
// independent rows do not need one cooperative launch per request. All partial
// histograms, counts and candidates reuse the existing TopK workspace.
template<bool Merge = false>
__global__ void SplitHistogram(Batch batch, int shift) {
    using namespace naive_topk;
    const Row &r=batch.rows[blockIdx.y];
    auto *w=r.workspace; int n=r.Count(), t=threadIdx.x, mergeWidth=r.MergeWidth();
    if(shift!=24 && w->state.done) return;
    __shared__ int histogram[kBuckets];
    __shared__ unsigned maxima[kThreads/32];
    histogram[t]=0; __syncthreads();
    int chunk=((n+int(gridDim.x)*kThreads-1)/(int(gridDim.x)*kThreads))*kThreads;
    int begin=blockIdx.x*chunk,end=min(n,begin+chunk);
    unsigned mask=shift==24?0u:~0u<<(shift+8),prefix=shift==24?0u:w->state.prefix;
    unsigned maximum=0,reference=n?r.Score<Merge>(0,mergeWidth):0;bool different=false;
    for(int base=begin;base<end;base+=kThreads) {
        int i=base+t;unsigned bits=i<end?r.Score<Merge>(i,mergeWidth):0;
        if(shift==24) {maximum=max(maximum,bits);different|=i<end&&bits!=reference;}
        int bucket=i<end&&(bits&mask)==prefix?int((bits>>shift)&255):256;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        unsigned peers=__match_any_sync(0xffffffff,bucket);
        if(bucket<256&&(t&31)==__ffs(peers)-1)atomicAdd(histogram+bucket,__popc(peers));
#else
        if(bucket<256)atomicAdd(histogram+bucket,1);
#endif
    }
    __syncthreads();w->partials[blockIdx.x*kBuckets+t]=histogram[t];
    if(shift==24) {
        int changed=__syncthreads_or(different);maximum=WarpMax(maximum);
        if(!(t&31))maxima[t/32]=maximum;__syncthreads();
        if(!t) {
            unsigned m=0;for(int i=0;i<kThreads/32;++i)m=max(m,maxima[i]);
            w->partials[gridDim.x*kBuckets+blockIdx.x]=m;
            w->ties[blockIdx.x]=changed;
        }
    }
}
template<bool Merge = false>
__global__ void SplitThreshold(Batch batch,int blocks,int shift) {
    using namespace naive_topk;
    const Row &r=batch.rows[blockIdx.x];auto *w=r.workspace;auto *s=&w->state;
    int n=r.Count(),t=threadIdx.x,topK=min(n,kTopK);
    if(shift!=24&&s->done)return;
    if(!n) {if(!t){s->uniform=1;s->selected=0;s->done=1;}return;}
    using Scan=cub::BlockScan<int,kThreads>;
    __shared__ typename Scan::TempStorage scan;
    __shared__ unsigned maxima[kThreads/32];
    if(shift==24) {
        int changed=__syncthreads_or(t<blocks&&w->ties[t]);
        unsigned m=t<blocks?(unsigned)w->partials[blocks*kBuckets+t]:0;m=WarpMax(m);
        if(!(t&31))maxima[t/32]=m;__syncthreads();
        if(!t) {
            m=0;for(int i=0;i<kThreads/32;++i)m=max(m,maxima[i]);
            s->maximum=m;s->uniform=!changed;s->selected=topK;s->done=!changed;
        }
        __syncthreads();if(!changed)return;
    }
    int rank=shift==24?topK:s->rank,bucket=255-t,total=0;
    unsigned prefix=shift==24?0u:s->prefix;
    for(int b=0;b<blocks;++b)total+=w->partials[b*kBuckets+bucket];
    int before;Scan(scan).ExclusiveSum(total,before);__syncthreads();
    if(before<rank&&before+total>=rank) {
        s->prefix=prefix|(unsigned(bucket)<<shift);s->rank=rank-before;
        s->candidates=topK-(rank-before)+total;
    }
    __syncthreads();
    if(!t)s->done=s->candidates<=kCapacity||shift==0||s->prefix==s->maximum;
    __syncthreads();
    bucket=(s->prefix>>shift)&255;
    int greater=t<blocks&&shift!=24?w->partials[blocks*kBuckets+t]:0;
    int equal=0;
    if(t<blocks) {
        for(int b=bucket+1;b<kBuckets;++b)greater+=w->partials[t*kBuckets+b];
        equal=w->partials[t*kBuckets+bucket];
    }
    if(!s->done) {if(t<blocks)w->partials[blocks*kBuckets+t]=greater;return;}
    int tieOffset;Scan(scan).ExclusiveSum(equal,tieOffset);__syncthreads();
    int count=greater+(s->candidates>kCapacity?min(equal,max(0,s->rank-tieOffset)):equal);
    if(t<blocks){w->ties[t]=tieOffset;w->partials[blocks*kBuckets+t]=count;}
    int unused,selected;Scan(scan).ExclusiveSum(count,unused,selected);
    if(!t)s->selected=selected;
}
template<bool Merge = false>
__global__ void SplitCompact(Batch batch) {
    using namespace naive_topk;
    const Row&r=batch.rows[blockIdx.y];auto*w=r.workspace;auto*s=&w->state;
    if(s->uniform)return;
    using Scan=cub::BlockScan<int,kThreads>;__shared__ typename Scan::TempStorage scan;
    int n=r.Count(),t=threadIdx.x,mergeWidth=r.MergeWidth();
    int chunk=((n+int(gridDim.x)*kThreads-1)/(int(gridDim.x)*kThreads))*kThreads;
    int begin=blockIdx.x*chunk,end=min(n,begin+chunk),offset=0;
    for(int b=0;b<int(blockIdx.x);++b)offset+=w->partials[gridDim.x*kBuckets+b];
    int tiesBefore=w->ties[blockIdx.x];unsigned prefix=s->prefix;
    for(int base=begin;base<end;base+=kThreads) {
        int i=base+t;unsigned bits=i<end?r.Score<Merge>(i,mergeWidth):0;bool take=i<end&&bits>=prefix;
        if(s->candidates>kCapacity) {
            int equal=i<end&&bits==prefix,before,total;
            Scan(scan).ExclusiveSum(equal,before,total);__syncthreads();
            take=i<end&&(bits>prefix||(equal&&tiesBefore+before<s->rank));tiesBefore+=total;
        }
        int before,total;Scan(scan).ExclusiveSum(int(take),before,total);__syncthreads();
        if(take){w->candidates.keys[offset+before]=bits-prefix;w->candidates.positions[offset+before]=r.Position<Merge>(i,mergeWidth);}
        offset+=total;
    }
}

template<bool Merge = false>
__global__ void Sort(Batch batch) {
    const Row &row = batch.rows[blockIdx.x];
    if (Merge && row.workspace->state.uniform) {
        int width = row.MergeWidth(), n = row.workspace->state.selected;
        for (int i = threadIdx.x; i < naive_topk::kTopK; i += blockDim.x)
            row.indices[i] = i < n ? row.Position<Merge>(i, width) : -1;
    } else naive_topk::SortRow(&row.workspace->candidates, &row.workspace->state, row.indices);
}
// Preserve the selected scores before the shared score arena is reused for
// exchange. No KV copies or additional per-request score allocations.
__global__ void PackLocal(Batch batch) {
    const Row &r = batch.rows[blockIdx.x];
    for (int i = threadIdx.x; i < naive_topk::kTopK; i += blockDim.x) {
        int position = r.indices[i];
        r.workspace->candidates.keys[i] = position >= 0 ? r.scores[position] : 0;
        r.workspace->candidates.positions[i] = position >= 0 ? r.LogicalPosition(position) : -1;
    }
}
template<class BatchType = Batch>
__global__ void CopyPacked(
#if CUDART_VERSION >= 12010
        const __grid_constant__
#endif
        BatchType batch, unsigned *exchange, int rows, int first) {
    const Row r = TopKRow(batch, blockIdx.x);
    unsigned *out = exchange + ((size_t)r.cpRank * rows + first + blockIdx.x) * naive_topk::kTopK * 2;
    for (int i = threadIdx.x; i < naive_topk::kTopK; i += blockDim.x) {
        out[i * 2] = r.workspace->candidates.keys[i];
        out[i * 2 + 1] = r.workspace->candidates.positions[i];
    }
}

__global__ void ClearContextIndices(Batch batch) {
    const Row &r = batch.rows[blockIdx.x];
    for (int i = threadIdx.x; i < naive_topk::kTopK; i += blockDim.x) r.indices[i] = -1;
}
// Every local list is sorted by (score descending, logical position ascending).
// Binary-search the other lists to obtain each candidate's exact global rank.
// The unique logical position breaks ties across interleaved context shards.
__global__ void MergeContextCandidates(Batch batch) {
    const Row &r = batch.rows[blockIdx.y];
    constexpr int K = naive_topk::kTopK;
    int i = blockIdx.x * blockDim.x + threadIdx.x, owner = i / K, local = i % K;
    if (owner >= r.cpRanks || local >= min(K, r.LocalCount(owner))) return;
    const unsigned *source = r.merged + ((size_t)owner * r.mergedRows + r.mergedRow) * K * 2;
    unsigned score = source[local * 2]; int position = source[local * 2 + 1];
    int global = local;
    for (int peer = 0; peer < r.cpRanks; ++peer) if (peer != owner) {
        const unsigned *list = r.merged + ((size_t)peer * r.mergedRows + r.mergedRow) * K * 2;
        int lo = 0, hi = min(K, r.LocalCount(peer));
        while (lo < hi) {
            int mid = (lo + hi) / 2; unsigned other = list[mid * 2];
            int otherPosition = list[mid * 2 + 1];
            if (other > score || (other == score && otherPosition < position)) lo = mid + 1;
            else hi = mid;
        }
        global += lo;
        if (global >= K) return;
    }
    r.indices[global] = position;
}

} // namespace naive_indexer_batch
#endif
