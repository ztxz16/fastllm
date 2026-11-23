// Blackwell DSA prefill. Fold the 16 GQA heads sharing K/V into the MMA M
// dimension. Each row's TopK bitmap masks logits before online softmax and PV.
#include "devices/cuda/naive-n05-indexer-attention.cuh"
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
__global__ void SelectScoreMask(const float *scores, uint32_t *bits, int keys,
                                int past, int keep, int stride) {
    using Scan = cub::BlockScan<int, 256>;
    __shared__ typename Scan::TempStorage scan;
    __shared__ unsigned histogram[256], equalWords[256], greaterWords[256], cutoff;
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
    const int stripes = (count + 255) / 256;
    for (int j = 0; j < stripes; ++j) {
        int i = j * 256 + t;
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
        histogram[t] = 0;
        __syncthreads();
        for (int j = 0; j < stripes; ++j) {
            unsigned u = values[j];
            int bucket = j * 256 + t < count && (u & tested) == prefix ? ((u >> shift) & 255u) : 256;
            unsigned peers = __match_any_sync(0xffffffff, bucket);
            if (bucket < 256 && (t % 32) == __ffs(peers)-1) atomicAdd(histogram + bucket, __popc(peers));
        }
        __syncthreads();
        const int size = histogram[255-t]; int before;
        Scan(scan).ExclusiveSum(size, before);
        if (before < rank && rank <= before + size) {
            cutoff = prefix | (unsigned(255-t) << shift); remaining = rank - before;
        }
        tested |= 255u << shift;
        __syncthreads();
    }
    for (int j = 0; j < (words+7)/8; ++j) {
        const bool valid = j * 256 + t < count;
        unsigned u = valid ? values[j] : 0;
        unsigned eq = __ballot_sync(0xffffffff, valid && u == cutoff);
        unsigned gt = __ballot_sync(0xffffffff, valid && u > cutoff);
        int word = j * 8 + t / 32;
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
struct Membership {
    const uint32_t *bits;
    int stride, past;
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
    template<class Coord, class Tile, class Problem>
    CUTLASS_DEVICE int get_trip_count(const Coord &coord, const Tile &tile, const Problem &p) {
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
        CUTLASS_PRAGMA_UNROLL
        for (int base = 0; base < size(acc); base += 32) {
            int first = get<1>(coords(base));
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
            if ((params.lengths[tile] < 0) == (params.workType == Work::Dense)) break;
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
    Shape<_256,_128,_192>, Shape<_256,_128,_128>, StrideQ, StrideK, StrideV, TopKMask, Shape<_2,_1,_1>, 1, 0, true>;
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
    if (lengths && lengths[blockIdx.x / UnionRows] >= 0) return;
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

// Model layout [row, KV head, group, dim]; FMHA layout [row, group, KV head, dim].
template<bool Output, bool Pack = false>
__global__ void ReorderHeads(const BF16 *src, BF16 *dst, const uint32_t *bits, int stride, int kvHeads, int firstHead = 0,
        FastllmMxfp8Activation projection = {}, int firstRow = 0) {
    constexpr int dim = Output ? ValueDim : QKDim;
    int row = blockIdx.x, kv = blockIdx.y;
    bool valid = !Output || bits[row * stride + stride - 1] != 0;
    for (int i = threadIdx.x; i < Group * dim / 8; i += blockDim.x) {
        int group = i / (dim / 8), col = i % (dim / 8);
        size_t native = ((size_t(row) * kvHeads + kv + firstHead) * Group + group) * (dim / 8) + col;
        size_t packed = ((size_t(row) * Group + group) * gridDim.y + kv) * (dim / 8) + col;
        uint4 value = valid ? ((const uint4 *)src)[Output ? packed : native] : make_uint4(0,0,0,0);
        ((uint4 *)dst)[Output ? native : packed] = value;
        if constexpr (Pack) {
            const int column = (((kv + firstHead) * Group + group) * dim) + col * 8;
            fastllm_mxfp8::Store8(projection, firstRow + row, column, value);
        }
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
    const bool sparse = keys > 16384 || keys > int64_t(ids.dims[1]) * 8;
    if (sparse && (scores || ids.dims[1] > UnionKeys)) return false;
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
    if (scores && (keys > 8192 || maskStride > ids.dims[1] ||
        scores->dataType != FLOAT32 || scores->dims != std::vector<int>({rows, keys}) ||
        !resident(*scores) || ids.cudaData == scores->cudaData)) return false;
    const size_t kvPerHead = size_t(keys) * (dim + valueDim);
    int stripeHeads = kvHeads;
    size_t rowBytes = size_t(heads) * (dim + valueDim) * sizeof(BF16) + (scores ? 0 : maskStride * sizeof(uint32_t));
    if (sparse) {
        // Independent KV heads can share the same staging allocation in turns.
        const size_t metadata = sizeof(int) * (UnionCapacity / UnionRows + UnionStride + 1);
        const size_t perHead = Group * (dim + valueDim) * sizeof(BF16);
        const size_t reserve = 128 * metadata;
        int fit = std::min<size_t>(kvHeads, bytes > reserve ?
            (bytes - reserve) / (kvPerHead * sizeof(BF16) + 128 * perHead) : 0);
        // Equal stripes share a tensor map, including when the KV head count is odd.
        while (fit && kvHeads % fit) --fit;
        stripeHeads = fit;
        rowBytes = metadata + perHead * (fit ? fit : kvHeads);
    }
    const size_t kvElems = kvPerHead * stripeHeads;
    if (bytes <= kvElems * sizeof(BF16)) return false;
    int chunk = std::min<size_t>(rows, (bytes - kvElems * sizeof(BF16)) / rowBytes);
    if (chunk < (sparse ? UnionRows : 128)) return false;
    if (chunk < rows || sparse) chunk -= chunk % Group;
    int sms = 0;
    auto status = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    if (status != cudaSuccess) return false;
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
                                      UnionRows * UnionWords * sizeof(uint32_t));
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
    const bool pack = item.packed.Accepts(rows, heads * valueDim);
    auto reorderOutput = pack ? ReorderHeads<true, true> : ReorderHeads<true>;
    if (sparse) {
        auto *positions = (int *)(out + outElems);
        int groups = chunk / UnionRows;
        auto *lengths = positions + size_t(groups) * UnionCapacity;
        auto *bits = (uint32_t *)(lengths + groups);
        UnionKV u{positions, lengths, bits, FastllmNaiveKV(k), FastllmNaiveKV(v), k.dims[2], kvHeads, kvElems != 0, {}};
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
        for (int head = 0; head < kvHeads; head += activeHeads) {
            if (u.staged) GatherKV<<<keys,128,0,stream>>>(u.key, u.value, key, value, kvHeads, k.dims[2],
                                                         head, activeHeads);
            for (int first = 0; first < rows; first += chunk) {
                int count = std::min(chunk, rows - first);
                // A dense group reuses its compact-position storage for the logical bitmap.
                const bool adaptive = u.staged && item.past + first < DenseWords * 32;
                BuildUnion<<<(count + UnionRows - 1) / UnionRows, 256, UnionRows * UnionWords * sizeof(uint32_t), stream>>>(
                    (const int*)ids.cudaData + size_t(first) * ids.dims[1], count, ids.dims[1], item.past + first, keys,
                    positions, lengths, bits, adaptive);
                dim3 grid(count, activeHeads);
                ReorderHeads<false><<<grid,128,0,stream>>>((const BF16*)q.cudaData + size_t(first) * heads * dim,
                    query, nullptr, 0, kvHeads, head);
                status = Attention<UnionKernel>(query, query, query, out, u, count, 1, activeHeads, sms, stream,
                    lengths, adaptive ? TileScheduler::Work::Compact : TileScheduler::Work::All);
                AssertInFastLLM(status == cudaSuccess, std::string("DSA compact prefill: ") + cudaGetErrorString(status));
                if (adaptive) {
                    // Bound the descriptor to this chunk, including a partial final group.
                    const int causalKeys = item.past + first + count;
                    const int words = (std::min(keys, DenseWords * 32) + 31) / 32;
                    BuildTopKMask<<<count,256,words * sizeof(uint32_t),stream>>>(
                        (const int*)ids.cudaData + size_t(first) * ids.dims[1], (uint32_t *)positions,
                        keys, ids.dims[1], item.past + first, words, DenseWords, bits + UnionWords, UnionStride, lengths);
                    Membership mask{reinterpret_cast<const uint32_t *>(positions), DenseWords, item.past + first};
                    status = Attention<Kernel>(query, key, value, out, mask, count, causalKeys, activeHeads, sms, stream,
                                               lengths, TileScheduler::Work::Dense);
                    AssertInFastLLM(status == cudaSuccess, std::string("DSA dense prefill: ") + cudaGetErrorString(status));
                }
                reorderOutput<<<grid,128,0,stream>>>(out, (BF16*)item.output->cudaData + size_t(first) * heads * valueDim,
                    bits, UnionStride, kvHeads, head, item.packed.slice, first);
            }
        }
        status = cudaGetLastError();
        AssertInFastLLM(status == cudaSuccess, std::string("DSA compact prefill: ") + cudaGetErrorString(status));
        if (pack) item.packed.Complete();
        return true;
    }
    auto *bitmap = scores ? (uint32_t *)ids.cudaData : (uint32_t *)(out + outElems);
    if (scores) {
        // The loan may hold the scores. Finish selection before staging KV in it.
        SelectScoreMask<<<rows, 256, 0, stream>>>((const float *)scores->cudaData,
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
        status = Attention<Kernel>(query, key, value, out, Membership{mask, maskStride, item.past + first}, count, keys, kvHeads, sms, stream);
        AssertInFastLLM(status == cudaSuccess, std::string("DSA Blackwell prefill: ") + cudaGetErrorString(status));
        reorderOutput<<<grid,128,0,stream>>>(out, (BF16 *)item.output->cudaData + size_t(first) * heads * valueDim,
            mask, maskStride, kvHeads, 0, item.packed.slice, first);
    }
    status = cudaGetLastError();
    AssertInFastLLM(status == cudaSuccess, std::string("DSA Blackwell prefill: ") + cudaGetErrorString(status));
    if (pack) item.packed.Complete();
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
DecodeKernel PairedDecodeKernel(bool pack) { return pack ? &Decode<false, 2, true> : &Decode<false, 2>; }
}
