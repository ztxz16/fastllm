// Included in dp_verify. The graph's Indexer score arena is free until O packing.
// All phases read request ownership and visible lengths from replay-time bindings.
constexpr int BatchPitch = 32768, ScoreGroups = 8;
constexpr int MaxBatchRows = FallbackElements / 2048;
struct ScoreLayout : Arena { static constexpr int kMaxSeqLen = BatchPitch; };
template<int Bits> using ScoreBody = fastllm_dsa_fused::Body<Bits != 16, false, ScoreLayout, ScoreGroups>;
using BatchScan = cub::BlockScan<int, Threads>;
struct SelectStorage {
    unsigned keys[BatchPitch], hist[256], answer[2], minimum[Threads / 32], maximum[Threads / 32];
    int ids[Threads * 6];
    union { BatchScan::TempStorage scan; ScoreSort::TempStorage sort; };
};
union BatchShared {
    ProjectionStorage<16> projection;
    ScoreBody<16>::SweepStorage sweep;
    ScoreBody<8>::SweepStorage fp8Sweep;
    SelectStorage select;
    struct {
        native::StorageFor<256, GroupHeads> tile;
        float partial[GroupHeads * (128 + 2)];
    } attention;
};

__device__ __forceinline__ void SelectBatchRow(const float *logits, int n, int *ids, SelectStorage &s) {
    const int t = threadIdx.x, lane = t % 32;
    const int k = min(n, 2048);
    constexpr int sortCapacity = Threads * 6;
    int candidates = n;
    if (n > sortCapacity) {
        unsigned lo = ~0u, hi = 0;
        for (int i = t; i < n; i += Threads) {
            const unsigned value = Core::ordered_float_key(logits[i]);
            s.keys[i] = value;
            lo = min(lo, value); hi = max(hi, value);
        }
        for (int d = 16; d; d >>= 1) {
            lo = min(lo, __shfl_down_sync(0xffffffff, lo, d));
            hi = max(hi, __shfl_down_sync(0xffffffff, hi, d));
        }
        if (!lane) { s.minimum[t / 32] = lo; s.maximum[t / 32] = hi; }
        __syncthreads();
        if (!t) {
            for (int w = 1; w < Threads / 32; ++w) {
                lo = min(lo, s.minimum[w]); hi = max(hi, s.maximum[w]);
            }
            s.minimum[0] = lo; s.maximum[0] = hi;
        }
        __syncthreads();
        const unsigned minimum = s.minimum[0], range = s.maximum[0] - minimum;
        if (!range) {
            for (int i = t; i < 2048; i += Threads) ids[i] = i;
            return;
        }
        unsigned prefix = 0, mask = 0;
        int rank = k;
        bool truncateTies = false;
        for (int shift = max(0, 32 - __clz(range) - 8); ; shift = max(0, shift - 8)) {
            if (t < 256) s.hist[t] = 0;
            __syncthreads();
            for (int base = 0; base < n; base += Threads) {
                const int i = base + t;
                const unsigned key = i < n ? s.keys[i] - minimum : 0;
                const unsigned digit = i < n && (key & mask) == prefix ? ((key >> shift) & 255) : 256;
                const unsigned peers = __match_any_sync(0xffffffffu, digit);
                if (digit < 256 && lane == __ffs(peers) - 1) atomicAdd(s.hist + digit, __popc(peers));
            }
            __syncthreads();
            if (t < 32) Core::topk_grid_find_bin_warp(s.hist, rank, lane, s.answer);
            __syncthreads();
            prefix |= s.answer[0] << shift;
            rank = s.answer[1];
            candidates = k - rank + s.hist[s.answer[0]];
            mask |= 255u << shift;
            __syncthreads();
            if (candidates <= sortCapacity) break;
            if (!shift) { candidates = k; truncateTies = true; break; }
        }
        prefix += minimum;
        // Stable compaction in token order preserves the lower-ID cutoff on ties.
        constexpr int Items = 8;
        int beforeTies = 0, beforeSelected = 0;
        for (int base = 0; base < n; base += Threads * Items) {
            unsigned keys[Items];
            int equal = 0;
#pragma unroll
            for (int j = 0; j < Items; ++j) {
                const int token = base + t * Items + j;
                keys[j] = token < n ? s.keys[token] : 0;
                equal += token < n && keys[j] == prefix;
            }
            int offset, total;
            int tie = 0, count = 0;
            if (truncateTies) {
                BatchScan(s.scan).ExclusiveSum(equal, offset, total);
                __syncthreads();
                tie = beforeTies + offset;
                beforeTies += total;
            }
            bool keep[Items];
#pragma unroll
            for (int j = 0; j < Items; ++j) {
                const bool valid = base + t * Items + j < n;
                keep[j] = valid && (keys[j] > prefix || (keys[j] == prefix && (!truncateTies || tie < rank)));
                tie += valid && keys[j] == prefix;
                count += keep[j];
            }
            BatchScan(s.scan).ExclusiveSum(count, offset, total);
            __syncthreads();
            int dst = beforeSelected + offset;
#pragma unroll
            for (int j = 0; j < Items; ++j)
                if (keep[j]) s.ids[dst++] = base + t * Items + j;
            beforeSelected += total;
        }
        __syncthreads();
    }
    unsigned keys[6]; int tokens[6];
#pragma unroll
    for (int i = 0; i < 6; ++i) {
        const int slot = t * 6 + i;
        const int token = slot < candidates ? (n > sortCapacity ? s.ids[slot] : slot) : -1;
        tokens[i] = token;
        keys[i] = token >= 0 ? (n > sortCapacity ? s.keys[token] : Core::ordered_float_key(logits[token])) : 0;
    }
    ScoreSort(s.sort).SortDescending(keys, tokens);
#pragma unroll
    for (int i = 0; i < 6; ++i)
        if (t * 6 + i < 2048) ids[t * 6 + i] = tokens[i];
}

template<int Bits> __device__ __forceinline__ FastllmNaiveDSARow BatchRow(const Input &a, const FastllmNaiveVerifyGroups &groups, int row) {
    auto r = a.core.rows[0];
    r.liveKeys += row;
    r.cache += groups.Owner(row) * (Bits == 4 ? 8 : 4);
    return r;
}

template<int Bits> __global__ __launch_bounds__(Threads, 1) void BatchedDpDSAVerify(
        const __grid_constant__ Input a, const __grid_constant__ FastllmNaiveVerifyGroups groups, float *scores, B *query) {
    using BatchScorer = ScoreBody<Bits>;
    extern __shared__ __align__(1024) unsigned char memory[];
    auto &s = *reinterpret_cast<BatchShared *>(memory);
    const int rows = a.packed.rows, t = threadIdx.x, lane = t % 32, warp = t / 32;
    const int block = blockIdx.x, blocks = gridDim.x;
    auto grid = cooperative_groups::this_grid();
    B *ik = a.core.output, *iq = ik + rows * 128, *iw = iq + rows * 2048;
    // Selection starts after all scorers finish. Its arena can hold one padded
    // tensor-core image per request, including a short final tile.
    B *images = reinterpret_cast<B *>(a.selected);
    float *scales = reinterpret_cast<float *>(images + size_t(groups.count) * 8 * 2048);
    // Two adjacent N8 groups share each loaded Indexer weight tile.
    const int pairs = (rows + 15) / 16;
    for (int task = block; task < pairs * ProjectionBlocks; task += blocks) {
        const int first = task % pairs * 16;
        ProjectIndexer(a, a.normed + first * 4096, min(16, rows - first), task / pairs,
                       ik + first * 128, iq + first * 2048, iw + first * 16, s.projection);
        __syncthreads();
    }
    for (int task = block; task < rows * 8; task += blocks)
        AppendMain<Bits>(a, FastllmNaiveResolveDSARow(BatchRow<Bits>(a, groups, task / 8)), task / 8, task % 8, query);
    grid.sync();
    for (int slot = block; slot < groups.count * 8; slot += blocks) {
        const int owner = slot / 8, local = slot % 8, row = groups.First(owner) + local;
        auto *image = images + owner * 8 * 2048;
        auto *scale = scales + owner * 8 * 16;
        if (local < groups.Size(owner) && a.core.rows[0].liveKeys[row] > 0) {
            auto r = FastllmNaiveResolveDSARow(BatchRow<Bits>(a, groups, row));
            AppendIndex<Bits>(a, r, row, local, ik, iq, image, scale);
        } else for (int h = warp; h < 16; h += Threads / 32) {
            const int head = local * 16 + h;
            for (int k = lane; k < 128; k += 32) {
                if constexpr (Bits == 16) image[Core::ImageOffset(head, k, 128) / 2] = B(0.f);
                else reinterpret_cast<uint8_t *>(image)[BatchScorer::ImageOffset(head, k, 128)] = 0;
            }
            if (!lane) scale[head] = 0;
        }
        __syncthreads();
    }
    grid.sync();
    const int requests = groups.count, group = block / BatchScorer::cta_count();
    __shared__ int lengths[9];
    for (int request = group; request < requests; request += ScoreGroups) {
        const int first = groups.First(request);
        auto args = a.core;
        args.activeRows = groups.Size(request);
        args.indexCache += request * (Bits == 4 ? 8 : 4);
        args.weights = iw + first * 16;
        if (t < 8) lengths[t] = t < args.activeRows ? max(0, min(args.rows[0].capacity, args.rows[0].liveKeys[first + t])) : 0;
        __syncthreads();
        if (!t) {
            int n = 0;
            for (int i = 0; i < 8; ++i) n = max(n, lengths[i]);
            lengths[8] = n;
        }
        __syncthreads();
        if (lengths[8]) {
            auto &sweep = [&]() -> typename BatchScorer::SweepStorage & {
                if constexpr (Bits == 16) return s.sweep;
                else return s.fp8Sweep;
            }();
            BatchScorer::compute_index_logits_tcgen05(args, reinterpret_cast<unsigned char *>(images + request * 8 * 2048),
                scales + request * 8 * 16, scores + size_t(first) * BatchPitch, lengths, sweep,
                request + ScoreGroups >= requests, false);
        }
        __syncthreads();
    }
    grid.sync();
    for (int row = block; row < rows; row += blocks) {
        const int n = max(0, min(a.core.rows[0].capacity, a.core.rows[0].liveKeys[row]));
        SelectBatchRow(scores + size_t(row) * BatchPitch, n, a.selected + row * 2048, s.select);
        if constexpr (Bits == 8) {
            // Logits are dead after selection. Use their first 2048 slots for
            // the same token-ascending attention order as the single-request path.
            __syncthreads();
            unsigned keys[6]; int tokens[6];
            #pragma unroll
            for (int i = 0; i < 6; ++i) {
                const int slot = t * 6 + i;
                tokens[i] = slot < min(n, 2048) ? a.selected[row * 2048 + slot] : -1;
                keys[i] = unsigned(tokens[i]);
            }
            ScoreSort(s.select.sort).Sort(keys, tokens, 0, 16);
            auto *members = reinterpret_cast<int *>(scores + size_t(row) * BatchPitch);
            #pragma unroll
            for (int i = 0; i < 6; ++i)
                if (t * 6 + i < 2048) members[t * 6 + i] = tokens[i];
        }
        __syncthreads();
    }
    grid.sync();
    if constexpr (Bits == 8) {
        // Small FP8 batches expose too few independent KV heads to fill the grid.
        // Distribute the eight context splits, then fold their states in split order.
        auto *partial = reinterpret_cast<float *>(a.selected + rows * 2048);
        for (int tile = block; tile < rows * Groups * splits; tile += blocks) {
            const int work = tile / splits, row = work / Groups;
            auto r = BatchRow<Bits>(a, groups, row);
            r.indices = reinterpret_cast<const int *>(scores + size_t(row) * BatchPitch);
            native::Attend<true, native::Phase::Produce, splits, 128, 256, true, false,
                           native::NoOutputEpilogue, GroupHeads, true, Bits>(
                query, r, a.core.sink, a.core.output, partial + size_t(tile) * GroupHeads * 130, nullptr,
                Heads, KvHeads, row, work % Groups, tile % splits, s.attention.tile);
            __syncthreads();
        }
        grid.sync();
        for (int tile = block * (Threads / 32) + warp; tile < rows * Heads * segments;
             tile += blocks * (Threads / 32)) {
            const int h = tile / segments, row = h / Heads;
            FinishAttention<true>(row, h, h, tile % segments, a.core.rows[0].liveKeys[row] > 0,
                                  partial, a.core.sink, a.core.output, a.packed);
        }
        return;
    }
    // Each CTA owns a KV head and its sixteen query heads. Keep its eight
    // PV fragments private until the split maxima are known. No grid barrier
    // or global publication is required between sparse attention and O packing.
    float *parts = reinterpret_cast<float *>(a.selected + rows * 2048) +
                   size_t(block) * splits * GroupHeads * (128 + 2);
    for (int work = block; work < rows * Groups; work += blocks) {
        const int row = work / Groups, group = work % Groups;
        auto r = BatchRow<Bits>(a, groups, row);
        r.indices = a.selected + row * 2048;
        if (*r.liveKeys <= 0) {
            for (int i = t; i < GroupHeads * 128 / 8; i += Threads) {
                const int column = group * GroupHeads * 128 + i * 8;
                reinterpret_cast<uint4 *>(a.core.output + size_t(row) * Heads * 128)[column / 8] = {};
                fastllm_mxfp8::StoreZero8(a.packed, row, column);
            }
            continue;
        }
        float2 state[2] = {make_float2(-INFINITY, 0), make_float2(-INFINITY, 0)};
        for (int k = 0; k < splits; ++k) {
            native::Attend<true, native::Phase::Produce, splits, 128, 256, true, false,
                           native::NoOutputEpilogue, GroupHeads, true, Bits>(
                query, r, a.core.sink, a.core.output, s.attention.partial, nullptr,
                Heads, KvHeads, row, group, k, s.attention.tile);
            __syncthreads();
            if (warp < 8) {
                for (int half = 0; half < 2; ++half) {
                    const int head = warp + half * 8;
                    reinterpret_cast<float4 *>(parts)[(k * GroupHeads + head) * 32 + lane] =
                        reinterpret_cast<float4 *>(s.attention.partial)[head * 32 + lane];
                    if (lane == k)
                        state[half] = reinterpret_cast<float2 *>(s.attention.partial + GroupHeads * 128)[head];
                }
            }
            __syncthreads();
        }
        if (warp < 8) {
            for (int half = 0; half < 2; ++half) {
                const int head = warp + half * 8, layerHead = group * GroupHeads + head;
                float maximum = state[half].x, sum = state[half].y;
                for (int d = 16; d; d >>= 1) maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, d));
                maximum = __shfl_sync(0xffffffff, maximum, 0);
                float sink = a.core.sink ? a.core.sink[layerHead] : -INFINITY;
                maximum = fmaxf(maximum, sink);
                const float correction = maximum == -INFINITY ? 0.f : __expf(state[half].x - maximum);
                float values[4] = {}, denom = 0.f;
#pragma unroll
                for (int k = 0; k < splits; ++k) {
                    const float c = __shfl_sync(0xffffffff, correction, k);
                    denom = fmaf(__shfl_sync(0xffffffff, sum, k), c, denom);
                    float4 fragment = reinterpret_cast<float4 *>(parts)[(k * GroupHeads + head) * 32 + lane];
                    values[0] = fmaf(fragment.x, c, values[0]); values[1] = fmaf(fragment.y, c, values[1]);
                    values[2] = fmaf(fragment.z, c, values[2]); values[3] = fmaf(fragment.w, c, values[3]);
                }
                if (a.core.sink) denom += __expf(sink - maximum);
                B rounded[4];
                for (int j = 0; j < 4; ++j)
                    rounded[j] = __float2bfloat16(denom > 0 ? values[j] / denom : 0.f);
                const int column = layerHead * 128 + lane * 4;
                *reinterpret_cast<uint2 *>(a.core.output + size_t(row) * Heads * 128 + column) =
                    *reinterpret_cast<const uint2 *>(rounded);
                fastllm_mxfp8::StoreGroup(a.packed, row, column, rounded);
            }
        }
        __syncthreads();
    }
}
