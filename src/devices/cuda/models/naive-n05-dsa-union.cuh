// Included by the Blackwell prefill translation unit after the FMHA types.
namespace blackwell_dsa {
constexpr int UnionKeys = 2048, UnionCapacity = UnionRows * UnionKeys;
constexpr int UnionWords = UnionCapacity / 32;

// Compact positions and logical masks share the same workspace.
// Longer staged histories borrow one extra logical bitmap per query row.
constexpr int MaxBitmapWords = 8192;
constexpr int ValueLoadWarp = 15;
struct UnionKV {
    const int *positions, *lengths;
    const uint32_t *bits;
    FastllmNaiveKV key, value;
    int keyStride, kvHeads, positionStride, maskStride;
    bool staged;
    CUtensorMap maps[2];
};

// One CTA owns a group. Load and validate each TopK ID exactly once, then reuse
// the registers to construct both the sorted union and its per-query memberships.
__global__ __launch_bounds__(512) void BuildUnion(const int *indices, int *positions, int *lengths,
        uint32_t *bits, TileSpan *spans, int rows, int past, int keep, int windowWords, bool adaptive,
        int positionStride, int maskStride) {
    constexpr int threads = 512, perThread = UnionCapacity / threads;
    using Scan = cub::BlockScan<int, threads>;
    __shared__ typename Scan::TempStorage scan;
    __shared__ int low, high;
    extern __shared__ uint32_t memory[];
    uint32_t *present = memory;
    int *offset = reinterpret_cast<int *>(present + windowWords);
    uint32_t *members = present + 2 * windowWords;
    const int t = threadIdx.x, group = blockIdx.x, first = group * UnionRows;
    const int n = min(UnionRows, rows - first);
    if (!t) { low = INT_MAX; high = -1; }
    __syncthreads();
    int held[perThread], lo = INT_MAX, hi = -1;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < perThread; ++i) {
        int entry = i * threads + t, row = entry / UnionKeys;
        const int col = entry % UnionKeys;
        int k = row < n && col < keep ? indices[size_t(first + row) * keep + col] : -1;
        k = k <= past + first + row ? k : -1;
        held[i] = k;
        if (k >= 0) { lo = min(lo,k); hi = max(hi,k); }
    }
    for (int step = 16; step; step /= 2) {
        lo = min(lo,__shfl_down_sync(0xffffffff,lo,step));
        hi = max(hi,__shfl_down_sync(0xffffffff,hi,step));
    }
    if (!(t % 32)) { atomicMin(&low,lo); atomicMax(&high,hi); }
    __syncthreads();
    const int firstKey = high < 0 ? 0 : low / 128 * 128;
    const bool single = int64_t(high) < int64_t(firstKey) + 32 * windowWords;
    int memberStride = UnionWords, total = 0;
    if (!single) {
        for (int i = t; i < n * memberStride; i += threads) members[i] = 0;
        __syncthreads();
    }
    // A window bounds shared storage independently of the logical context.
    // Even an empty group executes once to initialize its output mask.
    for (int64_t start = firstKey; start <= max(firstKey,high); start += 32 * windowWords) {
        const int64_t limit = min(int64_t(high) + 1, start + 32 * windowWords);
        const int count = int((limit - start + 31) / 32);
        for (int i = t; i < count; i += threads) present[i] = 0;
        __syncthreads();
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < perThread; ++i) {
            if (held[i] >= start && held[i] < limit) {
                const int k = held[i] - start;
                atomicOr(present + k / 32,1u << (k % 32));
            }
        }
        __syncthreads();
        const int perLane = (count + threads - 1) / threads;
        const int begin = t * perLane, end = min(count,begin + perLane);
        int num = 0, before, unique;
        for (int i = begin; i < end; ++i) num += __popc(present[i]);
        Scan(scan).ExclusiveSum(num, before, unique);
        before += total;
        total += unique;
        const int tiles = max(1,(count + 3) / 4);
        if (adaptive && total * 3 >= tiles * 256) {
            if (!t) { lengths[group] = -1; spans[group] = {int(start / 128),tiles}; }
            return;
        }
        for (int i = begin; i < end; ++i) {
            offset[i] = before;
            before += __popc(present[i]);
        }
        if (single) {
            memberStride = max(1, (total + 127) / 128) * 4;
            for (int i = t; i < n * memberStride; i += threads) members[i] = 0;
        }
        __syncthreads();
        int *sorted = positions + size_t(group) * positionStride;
        for (int w=t; w<count; w+=threads) {
            int slot = offset[w];
            for (uint32_t v = present[w]; v; v &= v-1)
                sorted[slot++] = start + w * 32 + __ffs(v) - 1;
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < perThread; ++i) {
            if (held[i] >= start && held[i] < limit) {
                const int k = held[i] - start, w = k / 32;
                const int slot = offset[w] + __popc(present[w] & ((1u << (k % 32)) - 1));
                int row = (i * threads + t) / UnionKeys;
                atomicOr(members + row * memberStride + slot / 32,1u << (slot % 32));
            }
        }
        __syncthreads();
    }
    const int outWords = max(1, (total + 127) / 128) * 4;
    if (!t) lengths[group] = total;
    int row = t / 32, lane = t % 32;
    if (row < n) {
        uint32_t any = 0;
        for (int w = lane; w < outWords; w += 32) {
            uint32_t v = members[row * memberStride + w];
            bits[size_t(first + row) * maskStride + w] = v; any |= v;
        }
        int valid = __any_sync(0xffffffff,any != 0);
        if (!lane) bits[size_t(first + row) * maskStride + maskStride - 1] = valid;
    }
}

// A paired score selector already owns one exact logical bitmap per query.
// OR the 16 rows, scan each word once, then remap row membership into the
// compact order. No ordered TopK list or second dense-mask build is needed.
__global__ void BuildBitmapUnion(const uint32_t *input, int stride, int *positions,
        int *lengths, uint32_t *bits, TileSpan *spans, int rows,
        int positionStride, int maskStride, bool adaptive) {
    constexpr int threads = 512;
    using Scan = cub::BlockScan<int, threads>;
    __shared__ typename Scan::TempStorage storage;
    __shared__ int low, high;
    const int t = threadIdx.x, group = blockIdx.x, first = group * UnionRows;
    const int live = min(UnionRows, rows - first);
    if (!t) { low = stride - 1; high = 0; }
    __syncthreads();
    input += size_t(first) * stride;
    bits += size_t(first) * maskStride;
    positions += size_t(group) * positionStride;
    unsigned united = 0;
    if (t < stride - 1)
        for (int r = 0; r < live; ++r) united |= input[r * stride + t];
    int offset, total;
    Scan(storage).ExclusiveSum(__popc(united), offset, total);
    // One atomic per nonempty warp, independent of the number of live keys.
    unsigned occupied = __ballot_sync(0xffffffff, united != 0);
    if (!(t % 32) && occupied) {
        atomicMin(&low, t + __ffs(occupied) - 1);
        atomicMax(&high, t + 31 - __clz(occupied));
    }
    if (t < live) bits[t * maskStride + maskStride - 1] = input[t * stride + stride - 1];
    __syncthreads();
    const int tiles = max(1, high / 4 - low / 4 + 1);
    if (adaptive && total * 3 >= tiles * 256) {
        if (!t) { lengths[group] = -1; spans[group] = {low / 4, tiles}; }
        return;
    }
    if (!t) lengths[group] = total;
    for (unsigned remaining = united; remaining; remaining &= remaining - 1)
        positions[offset++] = t * 32 + __ffs(remaining) - 1;
    __syncthreads();
    const int padded = max(128, (total + 127) / 128 * 128);
    for (int col = t; col - int(t % 32) < padded; col += threads) {
        int key = col < total ? positions[col] : 0;
        for (int r = 0; r < live; ++r) {
            bool selected = col < total && (input[r * stride + key / 32] & (1u << (key % 32)));
            unsigned word = __ballot_sync(0xffffffff, selected);
            if (!(t % 32)) bits[r * maskStride + col / 32] = word;
        }
    }
}

struct UnionMask {
    template<class Coord, class Tile, class P>
    CUTLASS_DEVICE int get_trip_count(const Coord &c, const Tile &, const P &p) {
        return max(1, (get<4>(p).lengths[int(get<0>(c))] + 127) / 128);
    }
    template<class C, class T, class P>
    CUTLASS_DEVICE int get_unmasked_trip_count(const C &, const T &, const P &) { return 0; }
    template<class C, class T, class P>
    CUTLASS_DEVICE int get_masked_trip_count(const C &c, const T &t, const P &p) { return get_trip_count(c,t,p); }
    template<class Acc, class Coords, class P>
    CUTLASS_DEVICE void apply_mask(Acc &a, const Coords &coords, const P &p) {
        int q = get<0>(coords(0));
        // The builder zero-pads the last 128 columns. Each softmax thread
        // owns this whole row fragment, so fetch its four mask words together.
        static_assert(decltype(size(a))::value == 128);
        uint4 masks = make_uint4(0,0,0,0);
        if (q < get<0>(p)) {
            int col = get<1>(coords(0));
            masks = *reinterpret_cast<const uint4 *>(get<4>(p).bits + size_t(q / Group) * get<4>(p).maskStride + col / 32);
        }
        if ((masks.x & masks.y & masks.z & masks.w) == ~0u) return;
        unsigned words[] = {masks.x, masks.y, masks.z, masks.w};
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 128; ++i)
            if (!(words[i / 32] & (1u << (i % 32)))) a(i) = -INFINITY;
    }
};

using UnionBase = Sm100FmhaFwdMainloopTmaWarpspecialized<Element, float, float,
    Shape<_256,_128,_192>, Shape<_256,_128,_128>, StrideQ, StrideK, StrideV, UnionMask, Shape<_2,_1,_1>, 1, 0, true, true>;
struct UnionMainloop : UnionBase {
    template<bool Value, class Smem, class Pipeline, class State>
    CUTLASS_DEVICE void gather(const UnionKV &u, int group, int head, int block, int live,
            Smem smem, Pipeline &pipe, State &state) {
        constexpr int dim = Value ? ValueDim : QKDim;
        auto srcLayout = [] {
            if constexpr (Value)
                return typename CollectiveMmaPV::TiledMma{}.get_slice(0).partition_B(
                    make_tensor((int*)nullptr, make_layout(make_shape(_128{}, _128{})))).layout();
            else
                return typename CollectiveMmaQK::TiledMma{}.get_slice(0).partition_B(
                    make_tensor((int*)nullptr, make_layout(make_shape(_128{}, _192{}), LayoutRight{}))).layout();
        }();
        auto inverse = right_inverse(srcLayout);
        static_assert(size(inverse) == 128 * dim);
        pipe.producer_acquire(state);
        auto dst = smem(_, _, _, state.index());
        const int lane = threadIdx.x % 32;
        if (u.staged) {
            int row = lane * 4, slot = 128 * block + row;
            const int *ids = u.positions + size_t(group) * u.positionStride;
            int4 pos;
            if (slot + 4 <= live) pos = *reinterpret_cast<const int4 *>(ids + slot);
            else pos = make_int4(slot < live ? ids[slot] : 0, slot+1 < live ? ids[slot+1] : 0,
                                 slot+2 < live ? ids[slot+2] : 0, 0);
            auto bar = cast_smem_ptr_to_uint(pipe.producer_get_barrier(state));
            CUTLASS_PRAGMA_UNROLL
            for (int col = 0; col < dim; col += 64) {
                auto addr = cast_smem_ptr_to_uint(&dst(inverse(row * dim + col))) & ~127u;
                asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes "
                    "[%0], [%1, {%2,%3,%4,%5,%6}], [%7];" ::
                    "r"(addr), "l"(&u.maps[Value ? 1 : 0]), "r"(head * dim + col),
                    "r"(pos.x), "r"(pos.y), "r"(pos.z), "r"(pos.w), "r"(bar) : "memory");
            }
            ++state;
            return;
        }
        for (int row = lane / 8; row < 128; row += 4) {
            int slot = block * 128 + row;
            int pos = slot < live ? u.positions[size_t(group) * u.positionStride + slot] : 0;
            auto data = Value ? u.value.Row(pos, u.kvHeads * ValueDim) : u.key.Row(pos, u.keyStride);
            CUTLASS_PRAGMA_UNROLL
            for (int col = (lane % 8) * 8; col < dim; col += 64) {
                *(uint4*)&dst(inverse(row * dim + col)) = (data + head * dim + col).Load8();
            }
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncwarp();
        if (!lane)
            cutlass::arch::ClusterTransactionBarrier::complete_transaction(
                pipe.producer_get_barrier(state), 0, Value ? TransactionBytesLoadV : TransactionBytesLoadK);
        ++state;
    }

    template<class C, class P, class PP>
    CUTLASS_DEVICE void load(const C &c, const P &p, const Params &params, const PP &original,
            TensorStorage &storage, PipelineQ &pq, typename PipelineQ::PipelineState &sq,
            PipelineK &pk, typename PipelineK::PipelineState &sk,
            PipelineV &pv, typename PipelineV::PipelineState &sv) {
        const auto &u = get<4>(original);
        int group = get<0>(c), head = get<2,0>(c), live = u.lengths[group];
        const auto &ld = params.load;
        auto mq = ld.tma_load_Q.get_tma_tensor(ld.layout_Q.shape());
        auto gq = get_local_tile_tensor(mq, select<0,2>(TileShapeQK{}), head, 0, get<0>(p));
        auto smQ = make_tensor(make_smem_ptr(storage.smem_q.data()), SmemLayoutQ{});
        auto g = typename CollectiveMmaQK::TiledMma{}.get_slice(0).partition_A(gq);
        auto partitions = tma_partition(ld.tma_load_Q, _0{}, Layout<_1>{}, group_modes<0,3>(smQ), group_modes<0,3>(g));
        auto loadQ = [&](int half) {
            pq.producer_acquire(sq);
            if (elect_one_sync()) copy(ld.tma_load_Q.with(*pq.producer_get_barrier(sq), 0),
                get<0>(partitions)(_, 2 * group + half), get<1>(partitions)(_, sq.index()));
            ++sq;
        };
        auto smK = make_tensor(make_smem_ptr(storage.smem_k.data()), SmemLayoutK{});
        auto smV = make_tensor(make_smem_ptr(storage.smem_v.data()), SmemLayoutV{});
        // The second producer advances V independently of the Q/K pipeline.
        if (threadIdx.x / 32 == ValueLoadWarp) {
            const int tiles = max(1, (live + 127) / 128);
            for (int i = 0; i < tiles; ++i)
                gather<true>(u, group, head, i, live, smV, pv, sv);
        } else {
            loadQ(0);
            gather<false>(u, group, head, 0, live, smK, pk, sk);
            loadQ(1);
            for (int i = 1; i * 128 < live; ++i)
                gather<false>(u, group, head, i, live, smK, pk, sk);
        }
    }
};
}
