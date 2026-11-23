// Blackwell DSA prefill. Fold the 16 GQA heads sharing K/V into the MMA M
// dimension. Each row's TopK bitmap masks logits before online softmax and PV.
#include "devices/cuda/naive-n05-indexer-attention.cuh"
#include "devices/cuda/naive-n05-swa-prefill.cuh"
#include "devices/cuda/fastllm-cuda.cuh"
#include "utils/utils.h"
#include "devices/cuda/naive-n05-kv.cuh"
#include "naive-n05-decode-blackwell.cuh"
#include "attention/blackwell/collective/fmha_fusion.hpp"

// Indexer scores are consumed as a set by fused attention. Select the exact
// kth ordered float, then take tied positions in increasing logical KV order.
#include <cub/block/block_scan.cuh>
#include <cuda.h>
#include <numeric>
namespace blackwell_dsa {
template<int Threads> __global__ void SelectScoreMask(const float *scores, uint32_t *bits, int keys,
                                int past, int keep, int stride) {
    using Scan = cub::BlockScan<int, Threads>;
    __shared__ typename Scan::TempStorage scan;
    __shared__ unsigned histogram[256], equalWords[Threads], greaterWords[Threads], cutoff;
    __shared__ int remaining;
    const int t = threadIdx.x, row = blockIdx.x;
    const int count = min(keys, past + row + 1), words = stride - 1;
    uint32_t *dest = bits + size_t(row) * stride;
    if (count <= keep) {
        if (t < words) {
            const int valid = max(0, min(32, count - t * 32));
            dest[t] = valid == 32 ? ~0u : (1u << valid) - 1;
        }
        if (!t) dest[words] = count != 0;
        return;
    }
    unsigned values[32];
    const int stripes = (count + Threads - 1) / Threads;
    for (int j = 0; j < stripes; ++j) {
        int i = j * Threads + t;
        float x = i < count ? scores[size_t(row) * keys + i] : 0.f;
        unsigned u = x == 0.f ? 0 : __float_as_uint(x);
        values[j] = (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
    }
    if (!t) { cutoff = 0; remaining = keep; }
    __syncthreads();
    unsigned tested = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        const unsigned prefix = cutoff;
        const int rank = remaining;
        if (t < 256) histogram[t] = 0;
        __syncthreads();
        for (int j = 0; j < stripes; ++j) {
            unsigned u = values[j];
            int bucket = j * Threads + t < count && (u & tested) == prefix ? ((u >> shift) & 255u) : 256;
            unsigned peers = __match_any_sync(0xffffffff, bucket);
            if (bucket < 256 && (t % 32) == __ffs(peers)-1) atomicAdd(histogram + bucket, __popc(peers));
        }
        __syncthreads();
        const int size = t < 256 ? histogram[255-t] : 0; int before;
        Scan(scan).ExclusiveSum(size, before);
        if (before < rank && rank <= before + size) {
            cutoff = prefix | (unsigned(255-t) << shift); remaining = rank - before;
        }
        tested |= 255u << shift;
        __syncthreads();
    }
    for (int j = 0; j < (words + Threads/32 - 1)/(Threads/32); ++j) {
        const bool valid = j * Threads + t < count;
        unsigned u = valid ? values[j] : 0;
        unsigned eq = __ballot_sync(0xffffffff, valid && u == cutoff);
        unsigned gt = __ballot_sync(0xffffffff, valid && u > cutoff);
        int word = j * (Threads/32) + t / 32;
        if (!(t % 32)) { equalWords[word] = eq; greaterWords[word] = gt; }
    }
    __syncthreads();
    unsigned eq = t < words ? equalWords[t] : 0;
    int before; Scan(scan).ExclusiveSum(__popc(eq), before);
    if (t < words) {
        int take = remaining - before;
        if (take <= 0) eq = 0;
        else if (take < __popc(eq)) {
            unsigned rest=eq;
            for (int i=0;i<take;++i) rest &= rest-1;
            eq ^= rest;
        }
        dest[t] = greaterWords[t] | eq;
    }
    if (!t) dest[words] = 1;
}

struct Length {
    struct Offsets {
        int length;
        CUTE_HOST_DEVICE int operator[](int batch) const { return batch * length; }
    } segment_offsets;
};
constexpr int UnionRows = 16;
constexpr int DenseRows = -1, DirectRows = -2;
struct TileSpan { int first, count; };
struct Membership {
    const uint32_t *bits;
    int stride, past;
    const TileSpan *spans;
    CUTLASS_DEVICE int physical_tile(int group, int index) const {
        return spans ? spans[group].first + index : index;
    }
};
}
namespace cutlass::fmha::collective {
template<> struct is_variable_length<blackwell_dsa::Length> : std::true_type {};
}
namespace cute {
template<> struct is_integral<blackwell_dsa::Length> : true_type {};
}
#include "attention/blackwell/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"
#include "attention/blackwell/collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp"
#include "attention/blackwell/kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp"
#include "attention/blackwell/device/fmha.hpp"

namespace blackwell_dsa {
using namespace cute;
using namespace cutlass::fmha::collective;
constexpr int Group = 16, QKDim = 192, ValueDim = 128;
using BF16 = __nv_bfloat16;
using Element = cutlass::bfloat16_t;

struct TopKMask {
    template<class Coord, class Problem>
    CUTLASS_DEVICE int kv_tile(int index, const Coord &c, const Problem &p) const {
        return get<4>(p).physical_tile(int(get<0>(c)), index);
    }
    template<class Coord, class Tile, class Problem>
    CUTLASS_DEVICE int get_trip_count(const Coord &coord, const Tile &tile, const Problem &p) {
        if (auto spans = get<4>(p).spans) return spans[int(get<0>(coord))].count;
        int end = get<4>(p).past + (int(get<0>(coord)) + 1) * int(get<0>(tile)) / Group;
        return ceil_div(min(get<1>(p), end), int(get<1>(tile)));
    }
    template<class Coord, class Tile, class Problem>
    CUTLASS_DEVICE int get_unmasked_trip_count(const Coord &, const Tile &, const Problem &) { return 0; }
    template<class Coord, class Tile, class Problem>
    CUTLASS_DEVICE int get_masked_trip_count(const Coord &c, const Tile &t, const Problem &p) {
        return get_trip_count(c, t, p);
    }
    template<class Accumulator, class Coordinates, class Problem>
    CUTLASS_DEVICE void apply_mask(Accumulator &acc, const Coordinates &coords, const Problem &p) {
        const auto &mask = get<4>(p);
        // Each TMEM fragment holds consecutive keys from one packed query
        // head. One bitmap load masks 32 scores (the FA4 vector mask contract).
        int q = get<0>(coords(0));
        int tile = get<1>(coords(0)) / 128;
        int offset = (mask.physical_tile(q / (UnionRows * Group), tile) - tile) * 128;
        CUTLASS_PRAGMA_UNROLL
        for (int base = 0; base < size(acc); base += 32) {
            int first = offset + get<1>(coords(base));
            uint32_t word = q < get<0>(p) && first < get<1>(p)
                ? mask.bits[(q / Group) * mask.stride + first / 32] : 0;
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < 32; ++j)
                if (!(word & (1u << j))) acc(base + j) = -INFINITY;
        }
    }
};

struct TileScheduler {
    enum class Work { All, Compact, Dense };
    struct Arguments {
        int tiles, heads;
        const int *lengths;
        Work workType;
    };
    struct Params {
        int tiles, heads, work, grid;
        const int *lengths;
        Work workType;
    };
    Params params;
    int index;
    CUTLASS_DEVICE TileScheduler(const Params &p) : params(p), index(blockIdx.x) { skip(); }
    CUTLASS_DEVICE void skip() {
        if (params.workType == Work::All) return;
        while (index < params.work) {
            const int tile = params.tiles - 1 - index / params.heads;
            const int length = params.lengths[tile];
            if (params.workType == Work::Dense ? length == DenseRows : length >= 0) break;
            index += params.grid;
        }
    }
    static Params to_underlying_arguments(const Arguments &args, cutlass::KernelHardwareInfo hw) {
        int work = args.tiles * args.heads;
        return {args.tiles, args.heads, work, std::min(work, hw.sm_count), args.lengths, args.workType};
    }
    static dim3 get_grid_shape(const Params &p) { return dim3(p.grid); }
    CUTLASS_DEVICE bool is_valid() const { return index < params.work; }
    CUTLASS_DEVICE auto get_block_coord() const {
        // Later query tiles traverse more causal keys. Issue them first so the
        // last partial wave contains the least work across all KV heads.
        return make_coord(params.tiles - 1 - index / params.heads, _0{}, make_coord(index % params.heads, 0));
    }
    CUTLASS_DEVICE TileScheduler &operator++() { index += params.grid; skip(); return *this; }
};

using Problem = tuple<Length, Length, int, tuple<tuple<int, int>, int>, Membership>;
using StrideQ = tuple<int, _1, tuple<int, int>>;
using StrideK = tuple<int, _1, tuple<_0, int>>;
using StrideV = tuple<_1, int, tuple<_0, int>>;
using Mainloop = Sm100FmhaFwdMainloopTmaWarpspecialized<Element, float, float,
    Shape<_256,_128,_192>, Shape<_256,_128,_128>, StrideQ, StrideK, StrideV, TopKMask, Shape<_2,_1,_1>, 1, 0, true, true>;
using Epilogue = Sm100FmhaFwdEpilogueTmaWarpspecialized<Element, float, typename Mainloop::TileShapePV>;
using Kernel = cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized<Problem, Mainloop, Epilogue, TileScheduler>;
static_assert(Kernel::SharedStorageSize <= 232448);

} // namespace blackwell_dsa
#include "naive-n05-dsa-union.cuh"
namespace blackwell_dsa {
using UnionProblem = tuple<Length, Length, int, tuple<tuple<int, int>, int>, UnionKV>;
struct UnionSchedule : cutlass::fmha::kernel::Sm100FmhaCtxKernelWarpspecializedSchedule {
    static constexpr int NumWarpsLoad = 2;
    CUTLASS_HOST_DEVICE static constexpr WarpRole warp_idx_to_WarpRole(int warp) {
        return warp == ValueLoadWarp ? WarpRole::Load : Sm100FmhaCtxKernelWarpspecializedSchedule::warp_idx_to_WarpRole(warp);
    }
};
using UnionKernel = cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized<
    UnionProblem, UnionMainloop, Epilogue, TileScheduler, UnionSchedule>;

template<class ActiveKernel, class MaskData>
cudaError_t Attention(BF16 *q, BF16 *k, BF16 *v, BF16 *out, MaskData mask,
        int rows, int keys, int kvHeads, int sms, cudaStream_t stream,
        const int *lengths = nullptr, TileScheduler::Work work = TileScheduler::Work::All) {
    int m = rows * Group;
    auto p = make_tuple(Length{{m}}, Length{{keys}}, QKDim, make_tuple(make_tuple(1, kvHeads), 1),
                           mask);
    auto lq = make_layout(make_shape(m, QKDim, make_shape(1, kvHeads)),
                          StrideQ{kvHeads * QKDim, _1{}, {QKDim, QKDim}});
    auto lk = make_layout(make_shape(keys, QKDim, make_shape(1, kvHeads)),
                          StrideK{kvHeads * QKDim, _1{}, {_0{}, QKDim}});
    auto lv = make_layout(make_shape(ValueDim, keys, make_shape(1, kvHeads)),
                          StrideV{_1{}, kvHeads * ValueDim, {_0{}, ValueDim}});
    auto lo = make_layout(make_shape(m, ValueDim, make_shape(make_shape(1, kvHeads), 2 * m)),
        make_stride(kvHeads * ValueDim, _1{}, make_stride(make_stride(ValueDim, ValueDim), kvHeads * ValueDim)));
    auto ll = make_layout(make_shape(m, make_shape(1, kvHeads)), make_stride(kvHeads, make_stride(_1{}, 1)));
    cutlass::KernelHardwareInfo hw{};
    hw.sm_count = sms;
    // The ragged epilogue adds m to its segment coordinate. Packed Q precedes
    // packed O in the same allocation and spans at least this prefix.
    Element *outputBase = (Element *)out - size_t(m) * kvHeads * ValueDim;
    using ActiveOp = cutlass::fmha::device::FMHA<ActiveKernel>;
    typename ActiveOp::Arguments args{p, {(Element *)q, lq, (Element *)k, lk, (Element *)v, lv,
        1.f / sqrtf(float(QKDim)), 1.f, 1.f, 1.f, 1.f}, {outputBase, lo, nullptr, ll, m},
        {(m + 255) / 256, kvHeads, lengths, work}, hw};
    auto params = ActiveKernel::to_underlying_arguments(args, nullptr);
    return ActiveOp::run(params, stream, false) == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorLaunchFailure;
}

__global__ void GatherKV(FastllmNaiveKV key, FastllmNaiveKV value, BF16 *k, BF16 *v,
        int kvHeads, int keyStride, int firstHead = 0, int stripeHeads = 0) {
    int row = blockIdx.x;
    if (!stripeHeads) stripeHeads = kvHeads;
    int keyWidth = stripeHeads * QKDim, valueWidth = stripeHeads * ValueDim;
    auto keyRow = key.Row(row, keyStride) + firstHead * QKDim;
    auto valueRow = value.Row(row, kvHeads * ValueDim) + firstHead * ValueDim;
    for (int i = threadIdx.x * 8; i < keyWidth; i += blockDim.x * 8)
        *(uint4 *)(k + size_t(row) * keyWidth + i) = (keyRow + i).Load8();
    for (int i = threadIdx.x * 8; i < valueWidth; i += blockDim.x * 8)
        *(uint4 *)(v + size_t(row) * valueWidth + i) = (valueRow + i).Load8();
}

__global__ void BuildTopKMask(const int *indices, uint32_t *bits, int keys, int count, int past,
        int words, int stride, uint32_t *valid, int flagStride, const int *lengths = nullptr) {
    if (lengths && lengths[blockIdx.x / UnionRows] != DenseRows) return;
    extern __shared__ uint32_t mask[];
    int row = blockIdx.x;
    for (int i = threadIdx.x; i < words; i += blockDim.x) mask[i] = 0;
    __syncthreads();
    bool any = false;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        int key = indices[size_t(row) * count + i];
        if (key >= 0 && key < keys && key <= past + row) {
            atomicOr(mask + key / 32, 1u << (key % 32));
            any = true;
        }
    }
    int nonempty = __syncthreads_or(any);
    uint32_t *dst = bits + row * stride;
    for (int i = threadIdx.x; i < words; i += blockDim.x) dst[i] = mask[i];
    if (!threadIdx.x) valid[size_t(row) * flagStride] = nonempty;
}

// A whole query tile can omit leading and trailing KV tiles that no row selected.
// Keep the original per-row bitmap for exact masking inside this interval.
__global__ void FindTileSpans(const uint32_t *bits, int stride, int rows, TileSpan *spans) {
    __shared__ int lo, hi;
    if (!threadIdx.x) { lo = INT_MAX; hi = -1; }
    __syncthreads();
    const int first = blockIdx.x * UnionRows, n = min(UnionRows, rows - first);
    const int words = stride - 1;
    int low = INT_MAX, high = -1;
    for (int i = threadIdx.x; i < n * words; i += blockDim.x) {
        const int word = i % words;
        if (bits[size_t(first + i / words) * stride + word]) {
            low = min(low, word / 4);
            high = max(high, word / 4);
        }
    }
    for (int step = 16; step; step /= 2) {
        low = min(low, __shfl_down_sync(0xffffffff, low, step));
        high = max(high, __shfl_down_sync(0xffffffff, high, step));
    }
    if (!(threadIdx.x % 32)) { atomicMin(&lo, low); atomicMax(&hi, high); }
    __syncthreads();
    if (!threadIdx.x) spans[blockIdx.x] = hi < 0 ? TileSpan{0,1} : TileSpan{lo,hi-lo+1};
}

// One CTA attends two adjacent KV heads of a single query. The union
// builder publishes the decision on the same stream for all sixteen rows.
__global__ void __launch_bounds__(256, 2) DirectPrefill(const BF16 *q, FastllmNaiveDSARow base,
        const int *routes, BF16 *out, int heads) {
    const int row = blockIdx.x;
    if (routes[row / UnionRows] != DirectRows) return;
    base.indices += size_t(row) * base.selectedCount;
    base.length += row;
    blackwell_decode::AttendRow<false, 2, true>(q, base, nullptr, out, heads, nullptr, nullptr);
}

// Model layout [row, KV head, group, dim]; FMHA layout [row, group, KV head, dim].
template<bool Output>
__global__ void ReorderHeads(const BF16 *src, BF16 *dst, const uint32_t *bits, int stride, int kvHeads, int firstHead = 0, const int *routes = nullptr) {
    if (routes && routes[blockIdx.x / UnionRows] == DirectRows) return;
    constexpr int dim = Output ? ValueDim : QKDim;
    int row = blockIdx.x, kv = blockIdx.y;
    bool valid = !Output || bits[row * stride + stride - 1] != 0;
    for (int i = threadIdx.x; i < Group * dim / 8; i += blockDim.x) {
        int group = i / (dim / 8), col = i % (dim / 8);
        size_t native = ((size_t(row) * kvHeads + kv + firstHead) * Group + group) * (dim / 8) + col;
        size_t packed = ((size_t(row) * Group + group) * gridDim.y + kv) * (dim / 8) + col;
        ((uint4 *)dst)[Output ? native : packed] = valid
            ? ((const uint4 *)src)[Output ? packed : native] : make_uint4(0,0,0,0);
    }
}

} // namespace blackwell_dsa

bool FastllmCudaHasBlackwellIndexerAttention() {
    const int arch = FastllmCudaRuntimeArch();
#ifdef FASTLLM_DSA_ARCH100
    if (arch == 100) return true;
#endif
#ifdef FASTLLM_DSA_ARCH103
    if (arch == 103) return true;
#endif
    return false;
}

namespace blackwell_dsa {
struct Window {
    const float *sink;
    int past;
};
struct WindowMask {
    template<class C, class P>
    CUTLASS_DEVICE int kv_tile(int tile, const C &coord, const P &p) const {
        return tile + max(0, int(get<0>(coord)) * 256 + get<4>(p).past - 127) / 128;
    }
    template<class C, class T, class P>
    CUTLASS_DEVICE int get_trip_count(const C &coord, const T &, const P &p) {
        int end = min(int(get<1>(p)), (int(get<0>(coord)) + 1) * 256 + get<4>(p).past);
        return (end + 127) / 128 - kv_tile(0, coord, p);
    }
    template<class C, class T, class P>
    CUTLASS_DEVICE int get_unmasked_trip_count(const C &, const T &, const P &) { return 0; }
    template<class C, class T, class P>
    CUTLASS_DEVICE int get_masked_trip_count(const C &c, const T &t, const P &p) { return get_trip_count(c,t,p); }
    template<class Acc, class Coords, class P>
    CUTLASS_DEVICE void apply_mask(Acc &a, const Coords &c, const P &p) {
        int row = get<0>(c(0)), end = row + get<4>(p).past;
        int first = max(0, row / 256 * 256 + get<4>(p).past - 127) / 128 * 128 + get<1>(c(0));
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(a); ++i)
            if (first + i > end || first + i < end - 127) a(i) = -INFINITY;
    }
    template<class C, class P>
    CUTLASS_DEVICE void init_softmax(const C &coord, const P &p, float scale, float &maximum, float &sum) const {
        if (auto sink = get<4>(p).sink) {
            maximum = sink[int(get<2,0>(coord))] / scale;
            sum = 1;
        }
    }
};
using WindowProblem = tuple<Length, Length, int, tuple<tuple<int,int>,int>, Window>;
using WindowLoop = Sm100FmhaFwdMainloopTmaWarpspecialized<Element, float, float, Shape<_256,_128,_192>,
    Shape<_256,_128,_128>, StrideQ, StrideK, StrideV, WindowMask, Shape<_2,_1,_1>, 1, 0, true>;
using WindowEpilogue = Sm100FmhaFwdEpilogueTmaWarpspecialized<Element, float, typename WindowLoop::TileShapePV, true>;
using WindowKernel = cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized<
    WindowProblem, WindowLoop, WindowEpilogue, TileScheduler>;
}

bool FastllmCudaTryBlackwellSwa(const FastllmNaiveStagedSwa &item, const float *sink, int heads, int kvHeads) {
    using namespace blackwell_dsa;
    if (item.rows < 1024 || kvHeads <= 0 || heads != kvHeads * 8 || item.past < 0 ||
        int64_t(item.past) + item.rows != item.keys || !FastllmCudaHasBlackwellIndexerAttention() || FastllmCudaGraphIsCapturingFast())
        return false;
    const int rows = item.rows, keys = item.keys, group = heads / kvHeads;
    auto h = make_shape(group, kvHeads);
    auto p = make_tuple(Length{{rows}}, Length{{keys}}, QKDim, make_tuple(h,1), Window{sink,item.past});
    auto lq = make_layout(make_shape(rows,QKDim,h), StrideQ{heads*QKDim,_1{},{QKDim,group*QKDim}});
    auto lk = make_layout(make_shape(keys,QKDim,h), StrideK{item.stride,_1{},{_0{},QKDim}});
    auto lv = make_layout(make_shape(ValueDim,keys,h), StrideV{_1{},kvHeads*ValueDim,{_0{},ValueDim}});
    auto lo = make_layout(make_shape(rows,ValueDim,make_shape(h,1)),
        make_stride(heads*ValueDim,_1{},make_stride(make_stride(ValueDim,group*ValueDim),heads*ValueDim)));
    auto ll = make_layout(make_shape(rows,make_shape(heads,1)),make_stride(heads,make_stride(_1{},1)));
    cutlass::KernelHardwareInfo hw{};
    if (cudaDeviceGetAttribute(&hw.sm_count, cudaDevAttrMultiProcessorCount, FastllmCudaGetDevice()) != cudaSuccess)
        return false;
    using Op = cutlass::fmha::device::FMHA<WindowKernel>;
    typename Op::Arguments args{p, {(Element *)item.query,lq,(Element *)item.key,lk,(Element *)item.value,lv,
        1.f/sqrtf(float(QKDim)),1.f,1.f,1.f,1.f}, {(Element *)item.output,lo,nullptr,ll,rows},
        {(rows+255)/256,heads,nullptr,TileScheduler::Work::All},hw};
    auto attr = cudaFuncSetAttribute(cutlass::device_kernel<WindowKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     WindowKernel::SharedStorageSize);
    if (attr != cudaSuccess) { cudaGetLastError(); return false; }
    auto params = WindowKernel::to_underlying_arguments(args, nullptr);
    auto result = Op::run(params, cudaStreamPerThread, false);
    fastllm::AssertInFastLLM(result == cutlass::Status::kSuccess, "Naive Blackwell SWA prefill launch failed.");
    return true;
}

static bool BlackwellAttention(const FastllmNaivePrefillAttentionItem &item,
        const fastllm::Data &sink, int heads, int kvHeads, int dim, int valueDim,
        const fastllm::Data &scratch, const fastllm::Data *scores) {
    using namespace fastllm;
    using namespace blackwell_dsa;
    if (FastllmCudaGraphIsCapturingFast() || !FastllmCudaHasBlackwellIndexerAttention() || !sink.dims.empty() || kvHeads <= 0 || kvHeads > 64 ||
        heads != kvHeads * Group || dim != QKDim || valueDim != ValueDim || item.past < 0) return false;
    const auto &q = *item.query, &k = *item.key, &v = *item.value, &ids = *item.indices;
    if (q.dims.size() != 3 || q.dims[0] != 1 || q.dims[2] != heads * dim ||
        k.dims.size() != 3 || k.dims[0] != 1 || k.dims[2] < kvHeads * dim || k.dims[2] % 8 ||
        v.dims.size() != 3 || v.dims[0] != 1 || v.dims[1] != k.dims[1] || v.dims[2] != kvHeads * valueDim ||
        q.dataType != BFLOAT16 || k.dataType != v.dataType || (k.dataType != FP8_E4M3 && k.dataType != BFLOAT16))
        return false;
    int rows = q.dims[1], keys = k.dims[1];
    // Scanning logical tiles is efficient while the causal span is comparable
    // to TopK. Longer histories traverse the union of selected positions.
    if (rows < 128 || int64_t(item.past) + rows > keys || ids.dataType != INT32 || ids.dims.size() != 2 ||
        ids.dims[0] != rows || ids.dims[1] <= 0) return false;
    const bool sparse = keys > 16384 || keys > int64_t(ids.dims[1]) * 8 ||
                        (ids.dims[1] == UnionKeys && keys > (scores ? 8192 : 3 * UnionKeys));
    if (sparse && (ids.dims[1] > UnionKeys || (scores && (keys <= 8192 || ids.dims[1] != UnionKeys)))) return false;
    const int device = FastllmCudaGetDevice();
    auto resident = [device](const Data &d) {
        return d.dataDevice == DataDevice::CUDA && d.dataDeviceIds.size() == 1 && d.dataDeviceIds[0] == device &&
               d.cudaData && (uintptr_t(d.cudaData) % 16 == 0);
    };
    if (!resident(q) || !resident(k) || !resident(v) || !resident(ids)) return false;
    if (!resident(scratch) || (scratch.dataType != BFLOAT16 && scratch.dataType != FLOAT32) ||
        scratch.isPagedKVCache || (&scratch == item.output)) return false;
    // Never grow the caller's storage. A view lends only its logical slice.
    size_t bytes = scratch.isFake || scratch.cudaDataBorrowed ? scratch.GetBytes() : scratch.expansionBytes;
    bytes = std::min(bytes, size_t(256) << 20);
    int maskStride = (keys - 1) / 32 + 2;
    if (scores && (keys > 16384 || (keys > 8192 && !sparse) || maskStride > ids.dims[1] ||
        scores->dataType != FLOAT32 || scores->dims != std::vector<int>({rows, keys}) ||
        !resident(*scores) || ids.cudaData == scores->cudaData)) return false;
    const size_t kvPerHead = size_t(keys) * (dim + valueDim);
    int sms = 0;
    auto status = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    if (status != cudaSuccess) return false;
    const int words = (keys - 1) / 32 + 1;
    // A union contains at most one entry per logical key. Pad positions
    // to a full MMA tile and preserve 16-byte alignment of membership rows.
    const int positionStride = std::min<int64_t>(UnionCapacity, ((int64_t(keys) + 127) / 128) * 128);
    const int unionStride = positionStride / 32 + 4;
    const int denseStride = positionStride / UnionRows;
    const bool externalBitmap = sparse && words > denseStride && words <= MaxBitmapWords;
    // Span storage is per tile. Two words per row also cover alignment of each metadata plane.
    constexpr size_t metadataPadding = sizeof(TileSpan) + 2 * sizeof(uint32_t);
    int stripeHeads = kvHeads;
    size_t rowBytes = size_t(heads) * (dim + valueDim) * sizeof(BF16) + (scores ? 0 : maskStride * sizeof(uint32_t));
    if (sparse) {
        // Independent KV heads can share the same staging allocation in turns.
        const size_t metadata = sizeof(int) * (positionStride / UnionRows + unionStride + 1 + (externalBitmap ? words : 0));
        const size_t perHead = Group * (dim + valueDim) * sizeof(BF16);
        const size_t reserve = 128 * metadata;
        int fit = std::min<size_t>(kvHeads, bytes > reserve ?
            (bytes - reserve) / (kvPerHead * sizeof(BF16) + 128 * perHead) : 0);
        // Equal stripes share a tensor map, including when the KV head count is odd.
        while (fit && kvHeads % fit) --fit;
        // Leave room for at least one wave of query tiles when a full-head
        // KV stage would force short, underfilled attention launches.
        if (fit > 1 && rows >= 1024) {
            int best = fit;
            for (int h = fit; h > 0; --h) {
                if (kvHeads % h) continue;
                size_t waveRows = std::min<size_t>(rows, (sms + h - 1) / h * Group);
                if (kvPerHead * h * sizeof(BF16) + waveRows * (metadata + perHead * h + metadataPadding) <= bytes) {
                    best = h; break;
                }
            }
            fit = best;
        }
        stripeHeads = fit;
        rowBytes = metadata + perHead * (fit ? fit : kvHeads);
        if (!fit && externalBitmap) rowBytes -= words * sizeof(uint32_t);
    }
    rowBytes += metadataPadding;
    const size_t kvElems = kvPerHead * stripeHeads;
    if (bytes <= kvElems * sizeof(BF16)) return false;
    // Budget full metadata tiles even when the last one has fewer live rows.
    // Keeping this tile in the same launch avoids a tiny second forward.
    const int storageRows = sparse ? int(std::min<int64_t>(INT_MAX, (int64_t(rows) + UnionRows - 1) / UnionRows * UnionRows)) : rows;
    int chunk = std::min<size_t>(storageRows, (bytes - kvElems * sizeof(BF16)) / rowBytes);
    if (chunk < (sparse ? UnionRows : 128)) return false;
    if (chunk < rows || sparse) chunk -= chunk % Group;
    const int activeHeads = stripeHeads ? stripeHeads : kvHeads;
    if (chunk < rows && sparse) {
        // Cut each long-query chunk at a full wave of resident FMHA CTAs.
        const int wave = sms / std::gcd(sms, activeHeads) * Group;
        if (chunk >= wave) chunk = chunk / wave * wave;
    }
    const int packedHeads = activeHeads * Group;
    size_t qElems = size_t(chunk) * packedHeads * dim, outElems = size_t(chunk) * packedHeads * valueDim;
    size_t used = kvElems * sizeof(BF16) + chunk * rowBytes;
    uintptr_t start = uintptr_t(scratch.cudaData);
    auto overlaps = [start, used](const void *ptr, size_t count) {
        if (!ptr || !count) return false;
        uintptr_t addr = uintptr_t(ptr);
        return addr < start ? start - addr < count : addr - start < used;
    };
    auto overlapsKV = [&](const Data &cache) {
        if (!cache.isPagedKVCache) return overlaps(cache.cudaData, cache.GetBytes());
        return cache.pageLen <= 0 || !cache.pagedKVCacheData || !resident(*cache.pagedKVCacheData) ||
            overlaps(cache.cudaData, (1 + (keys - 1) / cache.pageLen) * sizeof(int)) ||
            overlaps(cache.pagedKVCacheData->cudaData, cache.pagedKVCacheData->GetBytes());
    };
    if (overlaps(q.cudaData, q.GetBytes()) || overlapsKV(k) || overlapsKV(v) ||
        overlaps(ids.cudaData, ids.GetBytes()) || overlaps(item.output->cudaData,
            size_t(rows) * heads * valueDim * sizeof(BF16)) ||
        (uintptr_t(item.output->cudaData) % 16)) return false;
    if (sparse) {
        status = cudaFuncSetAttribute(BuildUnion, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (2 * MaxBitmapWords + UnionRows * UnionWords) * sizeof(uint32_t));
        if (status != cudaSuccess) { cudaGetLastError(); return false; }
        status = cudaFuncSetAttribute(cutlass::device_kernel<UnionKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      UnionKernel::SharedStorageSize);
        if (status != cudaSuccess) { cudaGetLastError(); return false; }
    }
    status = cudaFuncSetAttribute(cutlass::device_kernel<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  Kernel::SharedStorageSize);
    if (status != cudaSuccess) { cudaGetLastError(); return false; }
    auto &output = *item.output;
    output.dataType = BFLOAT16;
    output.Resize({1, rows, heads * valueDim});
    output.ToDevice(DataDevice::CUDA, {device}, false);
    output.Allocate(false);
    BF16 *key = (BF16 *)scratch.cudaData, *value = key + size_t(keys) * stripeHeads * dim;
    BF16 *query = key + kvElems, *out = query + qElems;
    cudaStream_t stream = cudaStreamPerThread;
    auto *spans = reinterpret_cast<TileSpan *>(out + outElems);
    auto *metadata = reinterpret_cast<uint32_t *>((uintptr_t(spans + (chunk + UnionRows - 1) / UnionRows) + 15) & ~uintptr_t(15));
    if (sparse) {
        auto *positions = reinterpret_cast<int *>(metadata);
        int groups = chunk / UnionRows;
        auto *lengths = positions + size_t(groups) * positionStride;
        auto *bits = (uint32_t *)(lengths + ((groups + 3) & ~3));
        auto *logical = externalBitmap ? bits + size_t(chunk) * unionStride : reinterpret_cast<uint32_t *>(positions);
        UnionKV u{positions, lengths, bits, FastllmNaiveKV(k), FastllmNaiveKV(v), k.dims[2], kvHeads, positionStride, unionStride, kvElems != 0, {}};
        if (u.staged) {
            for (int i = 0; i < 2; ++i) {
                const int width = stripeHeads * (i ? ValueDim : QKDim);
                cuuint64_t shape[] = {cuuint64_t(width), cuuint64_t(keys)}, strides[] = {cuuint64_t(width * sizeof(BF16))};
                cuuint32_t box[] = {64,1}, steps[] = {1,1};
                if (cuTensorMapEncodeTiled(&u.maps[i], CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, i ? value : key,
                        shape, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
                    return false;
            }
        }
        // The private index buffer survives staging, even when the score
        // matrix and the KV workspace are the same allocation.
        auto *rowMask = static_cast<uint32_t *>(ids.cudaData);
        if (scores) SelectScoreMask<512><<<rows,512,0,stream>>>(static_cast<const float *>(scores->cudaData),
            rowMask, keys, item.past, ids.dims[1], maskStride);
        bool direct = !scores && keys > 16384 && k.dataType == FP8_E4M3 && kvHeads % 2 == 0 &&
            ids.dims[1] == UnionKeys && k.dims[2] % 16 == 0 && u.staged && words <= MaxBitmapWords &&
            k.isPagedKVCache == v.isPagedKVCache && (!k.isPagedKVCache || k.pageLen == v.pageLen);
        constexpr int directBytes = blackwell_decode::Layout<2>::Bytes;
        if (direct && cudaFuncSetAttribute(DirectPrefill, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                          directBytes) != cudaSuccess) {
            cudaGetLastError();
            direct = false;
        }
        for (int head = 0; head < kvHeads; head += activeHeads) {
            if (u.staged) GatherKV<<<keys,128,0,stream>>>(u.key, u.value, key, value, kvHeads, k.dims[2],
                                                         head, activeHeads);
            for (int first = 0; first < rows; first += chunk) {
                int count = std::min(chunk, rows - first);
                // A dense group reuses its compact-position storage for the logical bitmap.
                const bool adaptive = u.staged && words <= MaxBitmapWords;
                const auto *selected = (const int*)ids.cudaData + size_t(first) * ids.dims[1];
                const int windowWords = std::min(words, MaxBitmapWords);
                const size_t shared = (2 * windowWords + UnionRows * UnionWords) * sizeof(uint32_t);
                const auto *chunkMask = rowMask + size_t(first) * maskStride;
                if (scores)
                    BuildBitmapUnion<<<(count + UnionRows - 1) / UnionRows,512,0,stream>>>(chunkMask,
                        maskStride, positions, lengths, bits, spans, count, positionStride, unionStride, adaptive);
                else
                    BuildUnion<<<(count + UnionRows - 1) / UnionRows,512,shared,stream>>>(selected, positions, lengths,
                    bits, spans, count, item.past + first, ids.dims[1], windowWords, adaptive, direct, positionStride, unionStride);
                if (direct && head == 0) {
                    FastllmNaiveDSARow row{};
                    row.key = u.key.base; row.value = u.value.base; row.indices = selected;
                    row.keyPages = u.key.pages; row.valuePages = u.value.pages; row.pageLen = u.key.pageLen;
                    row.fp8 = true; row.keyStride = u.keyStride; row.selectedCount = UnionKeys;
                    row.length = item.past + first + 1; row.capacity = keys;
                    DirectPrefill<<<dim3(count, kvHeads / 2), 256, directBytes, stream>>>(
                        (const BF16*)q.cudaData + size_t(first) * heads * dim, row, lengths,
                        (BF16*)output.cudaData + size_t(first) * heads * valueDim, heads);
                }
                dim3 grid(count, activeHeads);
                ReorderHeads<false><<<grid,128,0,stream>>>((const BF16*)q.cudaData + size_t(first) * heads * dim,
                    query, nullptr, 0, kvHeads, head, direct ? lengths : nullptr);
                status = Attention<UnionKernel>(query, query, query, out, u, count, 1, activeHeads, sms, stream,
                    lengths, adaptive ? TileScheduler::Work::Compact : TileScheduler::Work::All);
                AssertInFastLLM(status == cudaSuccess, std::string("DSA compact prefill: ") + cudaGetErrorString(status));
                if (adaptive) {
                    // Bound the descriptor to this chunk, including a partial final group.
                    const int causalKeys = item.past + first + count;
                    const int stride = scores ? maskStride : (externalBitmap ? words : denseStride);
                    if (!scores) BuildTopKMask<<<count,256,words * sizeof(uint32_t),stream>>>(selected, logical,
                        keys, ids.dims[1], item.past + first, words, stride, bits + unionStride - 1, unionStride, lengths);
                    Membership mask{scores ? chunkMask : logical, stride, item.past + first, spans};
                    status = Attention<Kernel>(query, key, value, out, mask, count, causalKeys, activeHeads, sms, stream,
                                               lengths, TileScheduler::Work::Dense);
                    AssertInFastLLM(status == cudaSuccess, std::string("DSA dense prefill: ") + cudaGetErrorString(status));
                }
                ReorderHeads<true><<<grid,128,0,stream>>>(out, (BF16*)item.output->cudaData + size_t(first) * heads * valueDim,
                    bits, unionStride, kvHeads, head, direct ? lengths : nullptr);
            }
        }
        status = cudaGetLastError();
        AssertInFastLLM(status == cudaSuccess, std::string("DSA compact prefill: ") + cudaGetErrorString(status));
        return true;
    }
    auto *bitmap = scores ? (uint32_t *)ids.cudaData : reinterpret_cast<uint32_t *>(metadata);
    if (scores) {
        // The loan may hold the scores. Finish selection before staging KV in it.
        SelectScoreMask<256><<<rows, 256, 0, stream>>>((const float *)scores->cudaData,
            bitmap, keys, item.past, ids.dims[1], maskStride);
    }
    GatherKV<<<keys,128,0,stream>>>(FastllmNaiveKV(k), FastllmNaiveKV(v), key, value, kvHeads, k.dims[2]);
    for (int first = 0; first < rows; first += chunk) {
        int count = std::min(chunk, rows - first);
        auto *mask = scores ? bitmap + size_t(first) * maskStride : bitmap;
        if (!scores)
            BuildTopKMask<<<count,256,(maskStride - 1) * sizeof(uint32_t),stream>>>(
                (const int *)ids.cudaData + size_t(first) * ids.dims[1], mask, keys, ids.dims[1], item.past + first, maskStride - 1, maskStride, mask + maskStride - 1, maskStride);
        dim3 grid(count, kvHeads);
        ReorderHeads<false><<<grid,128,0,stream>>>((const BF16 *)q.cudaData + size_t(first) * heads * dim,
            query, nullptr, 0, kvHeads);
        const bool trim = keys > 2 * int64_t(ids.dims[1]);
        if (trim) FindTileSpans<<<(count + UnionRows - 1) / UnionRows,128,0,stream>>>(mask, maskStride, count, spans);
        status = Attention<Kernel>(query, key, value, out, Membership{mask, maskStride, item.past + first, trim ? spans : nullptr}, count, keys, kvHeads, sms, stream);
        AssertInFastLLM(status == cudaSuccess, std::string("DSA Blackwell prefill: ") + cudaGetErrorString(status));
        ReorderHeads<true><<<grid,128,0,stream>>>(out, (BF16 *)item.output->cudaData + size_t(first) * heads * valueDim,
            mask, maskStride, kvHeads);
    }
    status = cudaGetLastError();
    AssertInFastLLM(status == cudaSuccess, std::string("DSA Blackwell prefill: ") + cudaGetErrorString(status));
    return true;
}

bool FastllmCudaTryNaiveIndexerAttention(const FastllmNaivePrefillAttentionItem &item,
        const fastllm::Data &sink, int heads, int kvHeads, int dim, int valueDim,
        const fastllm::Data &scratch) {
    return BlackwellAttention(item, sink, heads, kvHeads, dim, valueDim, scratch, nullptr);
}
bool FastllmCudaTryBlackwellScoreAttention(const FastllmNaivePrefillAttentionItem &item,
        const fastllm::Data &sink, int heads, int kvHeads, int dim, int valueDim,
        const fastllm::Data &scratch, const fastllm::Data &scores) {
    return BlackwellAttention(item, sink, heads, kvHeads, dim, valueDim, scratch, &scores);
}

namespace blackwell_decode {
DecodeKernel PairedDecodeKernel() { return &Decode<false, 2>; }
}
