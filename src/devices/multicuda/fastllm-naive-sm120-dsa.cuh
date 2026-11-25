#pragma once
namespace sm120_verify {
constexpr size_t DsaSelectionBytes = 8 * 2048 * sizeof(int);
constexpr size_t DsaListBytes = 8 * 8 * 2048 * sizeof(unsigned long long);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200 &&                                                       \
    (defined(__CUDA_ARCH_SPECIFIC__) || defined(__CUDA_ARCH_FAMILY_SPECIFIC__))

// Split K in the same contiguous intervals as Lt. Stage the next 16 KiB of weights for all
// active splits together; the two k16 instructions in each k32 step retain their original order.
template <int Splits, bool Early = false>
__device__ __forceinline__ void ProjectIndexPipelined(const Args &a, int tile, uint8_t *shared,
                                                      float (&part)[16][128]) {
    constexpr int Chunk = 512, span = Chunk / Splits, partition = 4096 / Splits;
    const int t = threadIdx.x, warp = t / 32, lane = t % 32, g = lane / 4, quad = t % 4;
    const BF *matrix = tile < 128 ? a.iqWeight : tile < 136 ? a.ikWeight : a.iwWeight;
    int outputTile = tile < 128 ? tile : tile < 136 ? tile - 128 : 0;
    const BF *weight = matrix + outputTile * 16 * 4096;
    auto *stage = reinterpret_cast<BF *>(shared);
    // Stage the same slice of each K partition, so all split warps can work on each panel.
    auto copy = [&](int panel) {
        for (int i = t; i < 1024; i += Threads) {
            int row = i / 64, k = i % 64 * 8;
            const auto *src = weight + row * 4096 + k / span * partition + panel * span + k % span;
            auto dst = static_cast<unsigned>(
                __cvta_generic_to_shared(stage + panel % 2 * 8192 + row * Chunk + (k ^ (row % 8 * 8))));
            asm volatile("cp.async.cg.shared.global [%0],[%1],16;" ::"r"(dst), "l"(src) : "memory");
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    float acc[4]{};
    copy(0);
    for (int panel = 0; panel < 8; ++panel) {
        if (panel < 7) {
            copy(panel + 1);
            asm volatile("cp.async.wait_group 1;" ::: "memory");
        } else
            asm volatile("cp.async.wait_group 0;" ::: "memory");
        if constexpr (Early) { if (panel == 0) fastllm::cuda_pdl::Wait(); }
        __syncthreads();
        if (warp < Splits) {
#pragma unroll
            for (int off = 0; off < span; off += 32) {
                auto load = [&](int half) {
                    int r = lane % 16, k = warp * span + off + half * 16 + lane / 16 * 8;
                    unsigned addr =
                        __cvta_generic_to_shared(stage + panel % 2 * 8192 + r * Chunk + (k ^ (r % 8 * 8)));
                    uint4 v;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                                 : "r"(addr));
                    return v;
                };
                uint4 lo = load(0), hi = load(1);
                const auto *x = reinterpret_cast<const unsigned *>(a.input + g * 4096 + warp * partition +
                                                                   panel * span + off + quad * 2);
                fastllm_swa_mma::Step(acc, make_uint4(lo.x, lo.z, hi.x, hi.z),
                                      make_uint4(lo.y, lo.w, hi.y, hi.w),
                                      make_uint4(x[0], x[4], x[8], x[12]));
            }
        }
        __syncthreads();
    }
    StoreIndexProjection(a, tile, 0, acc, part);
}

// One head per warp. All eight verification queries share the same 16 keys.
// K16 keeps the accumulation boundaries of the BF16 reference for these E4M3 inputs.
struct IndexScoreTile {
    float2 dots[16][16][4]; // head, key, query pair
    int counts[8];
};

__device__ void ScoreVerifyIndex(const Args &a, IndexScoreTile &tile) {
    const auto &first = a.batch.rows[0];
    for (int row = 1; row < 8; ++row) {
        if (a.batch.rows[row].cache != first.cache || a.batch.rows[row].pageLen != first.pageLen) {
            ScoreIndex(a);
            return;
        }
    }
    const int t = threadIdx.x, head = t / 32, lane = t % 32, group = lane / 4, pair = lane % 4;
    if (t < 8) {
        const auto &r = a.batch.rows[t];
        tile.counts[t] = ShardCount(max(0, min(r.capacity, *r.liveKeys)), r.pageLen, a.rank);
    }
    __syncthreads();
    int keys = 0;
#pragma unroll
    for (int row = 0; row < 8; ++row)
        keys = max(keys, tile.counts[row]);

    const int step = first.pageLen / 8;
    auto key = [&](int i) {
        const int physical = static_cast<const int *>(first.cache[6])[i / step] * step + i % step;
        return static_cast<const uint8_t *>(first.cache[4]) +
               size_t(physical) * (a.scaledIndex ? FastllmNaiveIndexerBytes : 128);
    };
    const auto *query = reinterpret_cast<const unsigned *>(a.iqBytes + (group * 16 + head) * 128) + pair;
    for (int base = blockIdx.x * 16; base < keys; base += gridDim.x * 16) {
        const auto *lo = reinterpret_cast<const unsigned *>(key(min(base + group, keys - 1))) + pair;
        const auto *hi = reinterpret_cast<const unsigned *>(key(min(base + group + 8, keys - 1))) + pair;
        float dot[4]{};
#pragma unroll
        for (int k = 0; k < 32; k += 4) {
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32 "
                         "{%0,%1,%2,%3},{%4,%5},{%6},{%0,%1,%2,%3};"
                         : "+f"(dot[0]), "+f"(dot[1]), "+f"(dot[2]), "+f"(dot[3])
                         : "r"(lo[k]), "r"(hi[k]), "r"(query[k]));
        }
        tile.dots[head][group][pair] = make_float2(dot[0], dot[1]);
        tile.dots[head][group + 8][pair] = make_float2(dot[2], dot[3]);
        __syncthreads();
        const int row = t % 8, token = base + t / 8;
        if (t < 128 && token < tile.counts[row]) {
            float sum = 0.f;
#pragma unroll
            for (int h = 0; h < 16; ++h) {
                const float d = reinterpret_cast<const float *>(tile.dots[h][t / 8])[row];
                sum = fmaf(fmaxf(d * a.iqScale[row * 16 + h], 0.f),
                           float(a.indexRaw[row * 2192 + 2176 + h]), sum);
            }
            const float scale = a.scaledIndex ? *reinterpret_cast<const float *>(key(token) + 128) : 1.f;
            const float score = sum * scale;
            const unsigned bits = score == 0.f ? 0 : __float_as_uint(score);
            a.indexScores[size_t(row) * a.shardWidth + token] = bits >> 31 ? ~bits : bits ^ 0x80000000u;
        }
        __syncthreads();
    }
}

// Sort disjoint tiles across the cooperative grid. A discarded item already
// has 2048 better items in the same subtree, including the position tie-break.
template<bool LiveTiles = false>
__device__ unsigned long long *ShardTopK(const Args &a, IndexSortStorage &sort, cooperative_groups::grid_group grid, int &lastSpan) {
    constexpr int K = 2048, warps = Threads / 32;
    const int tiles = a.shardWidth / K;
    auto *source = a.lists;
    auto nkeys = [&](int row) {
        const auto &r = a.batch.rows[row];
        return ShardCount(max(0, min(r.capacity, *r.liveKeys)), r.pageLen, a.rank);
    };
    // The lists keep their allocated row pitch even when the live prefix needs fewer merge levels.
    int liveTiles = tiles;
    if constexpr (LiveTiles) {
        int count = 0;
        for (int row = 0; row < a.rows; ++row)
            count = max(count, nkeys(row));
        liveTiles = max(1, (count + K - 1) / K);
        if (liveTiles == 1) return source;
    }
    for (int task = blockIdx.x; task < a.rows * liveTiles; task += gridDim.x) {
        const int row = task / liveTiles, first = (task % liveTiles) * K;
        const int count = nkeys(row), remaining = count - first;
        auto *output = source + size_t(row) * a.shardWidth + first;
        if (remaining > 0) {
            if (LiveTiles && remaining <= Threads)
                SortIndexTile<1>(a, row, count, first, output, sort.shortList);
            else if (LiveTiles && remaining <= 2 * Threads)
                SortIndexTile<2>(a, row, count, first, output, sort.fullList);
            else
                SortIndexTile<4>(a, row, count, first, output, sort.topK);
        }
        __syncthreads();
    }
    grid.sync();
    if (liveTiles == 1)
        return source;
    auto *dest = source + size_t(a.rows) * a.shardWidth;
    for (int span = 1; span < liveTiles; span *= 2) {
        if constexpr (LiveTiles) {
            if (2 * span >= liveTiles) {
                lastSpan = span;
                return source;
            }
        }
        const int pairs = (liveTiles + 2 * span - 1) / (2 * span);
        for (int work = blockIdx.x * warps + threadIdx.x / 32; work < a.rows * pairs * (K / 32);
             work += gridDim.x * warps) {
            const int pair = work / (K / 32) % pairs, row = work / (pairs * (K / 32));
            const int first = work % (K / 32) * 32, remain = nkeys(row) - pair * 2 * span * K;
            const int na = min(K, max(0, remain)), nb = min(K, max(0, remain - span * K));
            if (first < na + nb) {
                const auto *left = source + size_t(row) * a.shardWidth + 2 * pair * K;
                // An unpaired last tile has no right-hand storage.
                const auto value = MergeIndexWarp<LiveTiles>(left, na, nb ? left + K : left, nb, first);
                dest[size_t(row) * a.shardWidth + pair * K + first + threadIdx.x % 32] = value;
            }
        }
        grid.sync();
        auto *previous = source;
        source = dest;
        dest = previous;
    }
    return source;
}

// The full-kernel fallback replicates sorted lists to every peer.
__device__ void PushIndexRow(const Args &a, IndexSortStorage &sort, const unsigned long long *top) {
    const int row = blockIdx.x, t = threadIdx.x;
    auto r = a.batch.rows[row];
    int n = ShardCount(max(0, min(r.capacity, *r.liveKeys)), r.pageLen, a.rank);
    const int width = top ? 2048 : 1024;
    if (!top)
        PublishIndexRow(a, row, n, sort);
    CustomArBarrierStart<8>(a.signals, a.signals.signals[a.rank], a.rank);
    const auto *src = top ? reinterpret_cast<const ulonglong2 *>(top + size_t(row) * a.shardWidth)
                          : reinterpret_cast<const ulonglong2 *>(a.projection) + row * 512;
    const int count = (min(n, width) + 1) / 2;
    for (int i = t; i < count * 8; i += Threads) {
        int rank = i / count, k = i % count;
        auto *dst = reinterpret_cast<ulonglong2 *>(static_cast<uint8_t *>(a.exchange.scratch[rank]) +
                                                   VerifyExchangeBytes);
        dst[(row * 8 + a.rank) * (width / 2) + k] = src[k];
    }
    __threadfence_system();
    CustomArBarrierPublish<8>(a.signals, a.signals.signals[a.rank], a.rank);
}

// Route the final local merge straight to the row owner. If there is only
// one tile, the stable block sort can write that peer's slot directly.
__device__ void SendSelectionRoot(const Args &a, IndexSortStorage &sort,
                                  const unsigned long long *roots, int span) {
    const int row = blockIdx.x, t = threadIdx.x;
    const auto &r = a.batch.rows[row];
    const int n = ShardCount(max(0, min(r.capacity, *r.liveKeys)), r.pageLen, a.rank);
    const int width = min(2048, a.shardWidth);
    auto *base = static_cast<uint8_t *>(a.exchange.scratch[row]) + VerifyExchangeBytes;
    auto *dest = reinterpret_cast<unsigned long long *>(base) + (row * 8 + a.rank) * width;
    if (n > 0 && !span) {
        if (n <= Threads)
            SortIndexTile<1, true>(a, row, n, 0, dest, sort.shortList);
        else if (n <= 2 * Threads)
            SortIndexTile<2, true>(a, row, n, 0, dest, sort.fullList);
        else
            SortIndexTile<4, true>(a, row, n, 0, dest, sort.topK);
    } else if (n > 0) {
        const auto *left = roots + size_t(row) * a.shardWidth;
        const int na = min(2048, n), nb = min(2048, max(0, n - span * 2048));
        const int count = min(2048, n);
        for (int first = t / 32 * 32; first < count; first += Threads) {
            const auto v = MergeIndexWarp<true>(left, na, nb ? left + 2048 : left, nb, first);
            if (first + t % 32 < count) dest[first + t % 32] = v;
        }
    }
    // Separate candidate and result mailboxes allow two rendezvous per replay.
    // Publishing the candidates also retires the preceding result reads.
    __threadfence_system();
    CustomArBarrierStart<8>(a.signals, a.signals.signals[a.rank], a.rank);
}
// CTA i completes a disjoint 256-position stripe for its query row. Publish
// it directly to all peers; the same CTA can then consume that stripe for all
// eight rows as soon as the peer CTAs have published theirs.
__device__ void FinishOwnedSelection(const Args &a) {
    constexpr int stripeSize = 256, vectors = stripeSize / 4;
    const int stripe = blockIdx.x, t = threadIdx.x;
    __shared__ int4 selected[vectors];
    if (t < stripeSize) {
        const auto &r = a.batch.rows[a.rank];
        const int live = max(0, min(r.capacity, *r.liveKeys));
        const int halfPage = r.pageLen / 2, full = (live / r.pageLen) * halfPage, tail = live % r.pageLen;
        const int na = min(2048, full + min(halfPage, tail));
        const int nb = min(2048, full + max(0, tail - halfPage));
        const auto *left = a.lists + size_t(a.rank) * 8192;
        const int first = stripe * stripeSize + (t / 32) * 32;
        const auto value = MergeIndexWarp<true>(left, na, left + 2048, nb, first);
        reinterpret_cast<int *>(selected)[t] = value ? int(~unsigned(value)) : -1;
    }
    __syncthreads();
    auto mailbox = [&](int rank, int row) {
        auto *base = static_cast<uint8_t *>(a.exchange.scratch[rank]) + VerifyExchangeBytes + DsaListBytes;
        return reinterpret_cast<int4 *>(base) + row * (2048 / 4) + stripe * vectors;
    };
    const int peer = t / vectors, offset = t % vectors;
    mailbox(peer, a.rank)[offset] = selected[offset];
    __threadfence_system();
    CustomArBarrierPublish<8>(a.signals, a.signals.signals[a.rank], a.rank);
    reinterpret_cast<int4 *>(a.selected + peer * 2048 + stripe * stripeSize)[offset] =
        mailbox(a.rank, peer)[offset];
}
#endif
enum class DsaStage { Full, IndexScores, Select, AttentionOutput };
template<DsaStage Stage = DsaStage::Full>
__global__ __launch_bounds__(Threads, 2) void DsaLayer(
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    const __grid_constant__
#else
    const
#endif
        Args a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200 &&                                                       \
    (defined(__CUDA_ARCH_SPECIFIC__) || defined(__CUDA_ARCH_FAMILY_SPECIFIC__))
    auto grid = cooperative_groups::this_grid();
    __shared__ union {
        VerifyProjectionStorage projection;
        IndexScoreTile indexScore;
        IndexSortStorage sort;
        struct {
            BF values[64 * 128];
            float probabilities[512];
        } pv;
        struct {
            float scratch[256], maximum, denominator;
        } softmax;
    } store;
    const int cta = blockIdx.x, t = threadIdx.x;
    auto &part = store.projection.part;
    auto *panel = store.projection.panel;
    if constexpr (Stage == DsaStage::Full || Stage == DsaStage::IndexScores) {
        if (!a.prepared.values) {
            if constexpr (Stage == DsaStage::IndexScores) fastllm::cuda_pdl::Wait();
            Pack(a.input, 4096, a);
        }
        if (cta < 128)
            ProjectIndexPipelined<8, Stage == DsaStage::IndexScores>(a, cta, panel, part);
        else if (cta < IndexProjectionTiles)
            ProjectIndexPipelined<16, Stage == DsaStage::IndexScores>(a, cta, panel, part);
        grid.sync();
        PrepareIndex(a, &part[0][0]);
        grid.sync();
        ScoreVerifyIndex(a, store.indexScore);
    }
    if constexpr (Stage == DsaStage::Full) grid.sync();
    if constexpr (Stage == DsaStage::Full || Stage == DsaStage::Select) {
        constexpr bool split = Stage == DsaStage::Select;
        int lastSpan = 0;
        const auto *top = a.shardWidth > 1024 ? ShardTopK<split>(a, store.sort, grid, lastSpan) : nullptr;
        // Eight CTAs sort and exchange candidates while 116 independent CTAs project attention QKV.
        if (cta < a.rows) {
            if constexpr (split) SendSelectionRoot(a, store.sort, top, lastSpan);
            else PushIndexRow(a, store.sort, top);
        } else if (cta < 124)
            ProjectVerifyQkv<false>(a, cta - 8, panel, part);
        grid.sync();
        auto *pairs = reinterpret_cast<unsigned long long *>(a.scores);
        auto *received = reinterpret_cast<const unsigned long long *>(
            static_cast<uint8_t *>(a.exchange.scratch[a.rank]) + VerifyExchangeBytes);
        MergeIndexPairs<1, split>(a, received, pairs);
        grid.sync();
        MergeIndexPairs<2, split>(a, pairs, a.lists);
        grid.sync();
        if constexpr (split) {
            if (cta < 8) FinishOwnedSelection(a);
        } else {
            MergeIndexPairs<4>(a, a.lists, nullptr);
        }
    }
    if constexpr (Stage == DsaStage::Full) grid.sync();
    if constexpr (Stage == DsaStage::Full || Stage == DsaStage::AttentionOutput) {
        Scores(a, store.pv.values);
        grid.sync();
        if (cta < 64) {
            float *s = a.scores + cta * 2048;
            naive_dsa_decode::Normalize(s, s, a.sink, cta % 8, 2048, store.softmax.scratch, store.softmax.maximum,
                                        store.softmax.denominator, t);
        }
        grid.sync();
        ValueTiles(a, store.pv.values, store.pv.probabilities);
        grid.sync();
        // Overlap the O-weight prefetch with the ordered 64-key PV partial reduction.
        if (cta < OutputBlocks)
            StageVerifyOutputWeight(a, panel);
        for (int i = cta * Threads + t; i < 8 * 1024; i += gridDim.x * Threads) {
            float sum = 0;
            for (int p = 0; p < 32; ++p)
                sum += a.partials[(i / 1024 * 32 + p) * 1024 + i % 1024];
            StoreOutput(a, i, sum);
        }
        grid.sync();
        FinishVerifyOutput(a, panel, part, grid);
    }
#else
    asm("trap;");
#endif
}
} // namespace sm120_verify
