// Included by the Blackwell prefill translation unit after the FMHA types.
namespace blackwell_dsa {
constexpr int UnionRows = 16, UnionKeys = 2048, UnionCapacity = UnionRows * UnionKeys;
constexpr int UnionWords = UnionCapacity / 32, UnionStride = UnionWords + 1;

// Compact positions and logical masks share the same workspace.
constexpr int DenseWords = UnionCapacity / UnionRows;
constexpr int ValueLoadWarp = 15;
struct UnionKV {
    const int *positions, *lengths;
    const uint32_t *bits;
    FastllmNaiveKV key, value;
    int keyStride, kvHeads;
    bool staged;
    CUtensorMap maps[2];
};

// Compact the selected logical positions in ascending order. The bitmap window
// is bounded independently of context length. Only the per-row membership
// and union positions survive this CTA.
__global__ void BuildUnion(const int *ids, int rows, int keep, int past, int keys,
        int *positions, int *lengths, uint32_t *bits, bool adaptive) {
    using Scan = cub::BlockScan<int, 256>;
    extern __shared__ uint32_t members[][UnionWords];
    __shared__ typename Scan::TempStorage scan;
    __shared__ uint32_t present[DenseWords];
    __shared__ int offsets[DenseWords], total, low, high;
    const int group = blockIdx.x, t = threadIdx.x;
    const int first = group * UnionRows, n = min(UnionRows, rows - first);
    const int end = past + first + n;
    const bool singleWindow = end <= 32 * DenseWords;
    if (!t) {
        total = 0;
        low = singleWindow ? 0 : keys;
        high = singleWindow ? end - 1 : -1;
    }
    for (int i = t; i < UnionRows * UnionWords; i += blockDim.x)
        members[i / UnionWords][i % UnionWords] = 0;
    __syncthreads();
    // Only longer prefixes need a pass to find the selected range.
    if (!singleWindow) {
        int lo = keys, hi = -1;
        for (int i = t; i < n * keep; i += blockDim.x) {
            int row = i / keep, id = ids[size_t(first) * keep + i];
            if (id >= 0 && id < keys && id <= past + first + row) {
                lo = min(lo, id); hi = max(hi, id);
            }
        }
        for (int d = 16; d; d /= 2) {
            lo = min(lo, __shfl_down_sync(0xffffffff, lo, d));
            hi = max(hi, __shfl_down_sync(0xffffffff, hi, d));
        }
        if (!(t & 31)) { atomicMin(&low, lo); atomicMax(&high, hi); }
        __syncthreads();
    }
    for (int64_t begin = low; begin <= high; begin += 32 * DenseWords) {
        int span = min(int64_t(DenseWords * 32), int64_t(high) + 1 - begin);
        int nw = (span + 31) / 32;
        for (int w = t; w < nw; w += blockDim.x) present[w] = 0;
        __syncthreads();
        for (int i = t; i < n * keep; i += blockDim.x) {
            int id = ids[size_t(first) * keep + i], row = i / keep;
            int64_t j = int64_t(id) - begin;
            if (j >= 0 && j < span && id <= past + first + row)
                atomicOr(present + (j >> 5), 1u << (j & 31));
        }
        __syncthreads();
        // The contiguous kernel visits the causal prefix. Compare its span with
        // the exact union size before materializing any compact positions.
        if (adaptive && singleWindow) {
            int pop = 0, before, unique;
            for (int w = t; w < nw; w += blockDim.x) pop += __popc(present[w]);
            Scan(scan).ExclusiveSum(pop, before, unique);
            if (unique * 3 >= end) {
                if (!t) lengths[group] = -1;
                return;
            }
            __syncthreads();
        }
        for (int base = 0; base < nw; base += 256) {
            int w = base + t;
            unsigned mask = w < nw ? present[w] : 0;
            int before, size;
            Scan(scan).ExclusiveSum(__popc(mask), before, size);
            int slot = total + before;
            if (w < nw) {
                offsets[w] = slot;
                while (mask) {
                    positions[size_t(group) * UnionCapacity + slot++] = int(begin + w * 32 + __ffs(mask) - 1);
                    mask &= mask - 1;
                }
            }
            __syncthreads();
            if (!t) total += size;
            __syncthreads();
        }
        for (int i = t; i < n * keep; i += blockDim.x) {
            int id = ids[size_t(first) * keep + i], row = i / keep;
            int64_t j = int64_t(id) - begin;
            if (j >= 0 && j < span && id <= past + first + row) {
                int slot = offsets[j >> 5] + __popc(present[j >> 5] & ((1u << (j & 31)) - 1));
                atomicOr(&members[row][slot >> 5], 1u << (slot & 31));
            }
        }
        __syncthreads();
    }
    if (!t) lengths[group] = total;
    const int words = (total + 127) / 128 * 4;
    const int row = t / 16, lane = t % 16;
    if (row < n) {
        unsigned valid = 0;
        for (int w = lane; w < words; w += 16) {
            uint32_t v = members[row][w];
            bits[size_t(first + row) * UnionStride + w] = v;
            valid |= v;
        }
        unsigned lanes = (t & 16) ? 0xffff0000u : 0xffffu;
        int any = __any_sync(lanes, valid != 0);
        if (!lane) bits[size_t(first + row) * UnionStride + UnionWords] = any;
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
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(a); i += 32) {
            int slot = get<1>(coords(i));
            unsigned mask = q < get<0>(p) && slot < get<4>(p).lengths[q / (UnionRows * Group)]
                ? get<4>(p).bits[size_t(q / Group) * UnionStride + slot / 32] : 0;
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < 32; ++j) if (!(mask & (1u << j))) a(i + j) = -INFINITY;
        }
    }
};

using UnionBase = Sm100FmhaFwdMainloopTmaWarpspecialized<Element, float, float,
    Shape<_256,_128,_192>, Shape<_256,_128,_128>, StrideQ, StrideK, StrideV, UnionMask, Shape<_2,_1,_1>, 0, 1, true>;
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
            const int *ids = u.positions + size_t(group) * UnionCapacity;
            int pos[4];
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < 4; ++j) pos[j] = slot + j < live ? ids[slot+j] : 0;
            auto bar = cast_smem_ptr_to_uint(pipe.producer_get_barrier(state));
            CUTLASS_PRAGMA_UNROLL
            for (int col = 0; col < dim; col += 64) {
                auto addr = cast_smem_ptr_to_uint(&dst(inverse(row * dim + col))) & ~127u;
                asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes "
                    "[%0], [%1, {%2,%3,%4,%5,%6}], [%7];" ::
                    "r"(addr), "l"(&u.maps[Value ? 1 : 0]), "r"(head * dim + col),
                    "r"(pos[0]), "r"(pos[1]), "r"(pos[2]), "r"(pos[3]), "r"(bar) : "memory");
            }
            ++state;
            return;
        }
        for (int row = lane / 8; row < 128; row += 4) {
            int slot = block * 128 + row;
            int pos = slot < live ? u.positions[size_t(group) * UnionCapacity + slot] : 0;
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
