// Inkling's additive relative-position logits and learned sink with the in-tree
// FlashInfer paged-decode implementation. Each query sees the paged prefix followed
// by the uncommitted suffix, including during a multi-token chunk crossing wrap.
struct AttentionRow {
    const float *q;
    const void *key, *value;
    PageView pages;
    const float *relative, *sink;
    float *output;
    const DecodeVisit *visit;
    int past, first, heads, kvHeads, width, distances, window, floor;
    float alpha;
    int sinkStride = heads, gateStride = heads * width;
};

// Keep each query's split boundaries and reduction order identical to decode.
// Adjacent CTAs process consecutive queries against the same physical KV pages.
template<int D>
__device__ void selectSplitQuery(AttentionRow &a, float *&partial, int row, int splits) {
    a.visit += row;
    a.q += size_t(row) * a.heads * D;
    a.relative += size_t(row) * a.heads * a.distances;
    partial += size_t(row) * splits * a.heads * (D + 1);
}

template<class T> struct KVSegment {
    PageView pages;
    const T *suffix;
    int first, past, stride;
    bool valueSide;
    __device__ const T *operator+(size_t element) const {
        return at(first + int(element / stride), element % stride);
    }
    __device__ const T *at(int token, unsigned channel) const {
        if (suffix && token >= past) return suffix + size_t(token - past) * stride + channel;
        const T *base = static_cast<const T *>(valueSide ? pages.value : pages.key);
        return base + pages.offset(token, valueSide, stride) + channel;
    }
};

template<class T> struct KVView {
    // FlashInfer only needs to map a virtual page and head to an element
    // offset. One virtual page represents one chronological KV token.
    struct UnitPage {
        __device__ operator unsigned() const { return 1; }
        __device__ void divmod(unsigned n, unsigned &q, unsigned &r) const { q = n; r = 0; }
    } page_size;
    int batch_size = 1, length, group, width;
    const int *indptr, *rope_pos_offset = nullptr;
    KVSegment<T> k_data, v_data;
    __device__ unsigned get_length(unsigned) const { return length; }
    __device__ size_t protective_get_kv_offset(unsigned token, unsigned head, unsigned, unsigned, int) const {
        return token < length ? size_t(token) * k_data.stride + size_t(head / group) * width : 0;
    }
};

template<class T> struct RelativeParams {
    using DTypeQ = float;
    using DTypeKV = T;
    using DTypeO = float;
    using IdType = int;
    const float *q, *relative, *sink;
    float *o, *lse = nullptr;
    KVView<T> paged_kv;
    const bool *block_valid_mask = nullptr;
    bool partition_kv = false;
    const int *request_indices, *kv_tile_indices, *kv_chunk_size_ptr;
    unsigned num_qo_heads, q_stride_n = 0, q_stride_h, padded_batch_size = 1;
    int distances;
    float tau, scale;
};

struct RelativeAttention : flashinfer::AttentionVariantBase {
    static constexpr float sm_scale_log2 = 1.4426950408889634f;
    template<class P> __device__ RelativeAttention(const P &, unsigned, void *) {}
    template<class P, class T>
    __device__ T LogitsTransform(const P &p, T qk, unsigned, unsigned, unsigned t, unsigned h, unsigned) const {
        unsigned distance = p.paged_kv.length - 1u - t;
        float bias = distance < p.distances ? p.relative[size_t(h) * p.distances + distance] : 0.f;
        return (qk * p.scale + bias) * p.tau;
    }
    template<class P, class T, class M>
    __device__ T OutputTransform(const P &p, T o, unsigned, unsigned, unsigned h, M &m, float &d, float) const {
        if (p.partition_kv) return o / d;
        float s = p.sink[h] * sm_scale_log2, mx = fmaxf(m, s);
        float a = exp2f(m - mx);
        return o * a / (d * a + exp2f(s - mx));
    }
};

template<int D, int G, bool Split, class T>
__global__ void relativeAttention(AttentionRow a, float *partial) {
    constexpr int vector = 16 / sizeof(T), lanes = D / vector, groups = Split ? 1 : 128 / lanes;
    const int past = a.visit ? a.visit->past : a.past;
    const int row = a.first + blockIdx.x, pos = past + row;
    const int begin = a.window ? max(0, pos + 1 - a.window) : 0;
    const int length = pos - begin + 1;
    const int split = Split ? blockIdx.z : 0;
    const int chunk = Split ? max(16, (length + int(gridDim.z) - 1) / int(gridDim.z)) : length;
    if (Split && split >= (length + chunk - 1) / chunk) return;
    const int pointers[] = {0, length};
    RelativeParams<T> p{};
    p.q = a.q + size_t(row) * a.heads * D;
    if constexpr (!Split) p.o = a.output + size_t(row) * a.heads * D;
    p.relative = a.relative + size_t(blockIdx.x) * a.heads * a.distances;
    if constexpr (!Split) p.sink = a.sink + size_t(row) * a.heads;
    p.num_qo_heads = a.heads;
    p.q_stride_h = D;
    p.scale = rsqrtf(float(a.width));
    p.distances = a.distances;
    p.tau = a.window ? 1.f : 1.f + a.alpha * logf(fmaxf(1.f, (pos + 1.f) / a.floor));
    p.request_indices = pointers;
    p.kv_tile_indices = &split;
    p.kv_chunk_size_ptr = &chunk;
    if constexpr (Split) {
        p.partition_kv = true;
        p.o = partial + size_t(split) * a.heads * D;
        p.lse = partial + size_t(gridDim.z) * a.heads * D + split * a.heads;
    }
    auto &kv = p.paged_kv;
    kv.indptr = pointers;
    kv.length = pointers[1];
    kv.group = a.heads / (a.kvHeads * G);
    kv.width = D;
    kv.k_data = {a.visit ? a.visit->pages : a.pages, static_cast<const T *>(a.key), begin, past, a.kvHeads * D, false};
    kv.v_data = {a.visit ? a.visit->pages : a.pages, static_cast<const T *>(a.value), begin, past, a.kvHeads * D, true};
    extern __shared__ uint8_t attentionStorage[];
    flashinfer::BatchDecodeWithPagedKVCacheDevice<flashinfer::PosEncodingMode::kNone,
        2, 2, vector, lanes, G, groups, RelativeAttention>(p, attentionStorage, 0, blockIdx.y);
}

template<class T, bool ToShared = false>
__device__ void loadRelativeKV(PageView pages, int token, int stride, int channel, uint4 &k, uint4 &v) {
    const int page = pages.shift >= 0 ? token >> pages.shift : token / pages.pageLen;
    const int2 ids = pages.ids[page - pages.first];
    const size_t offset = size_t(token - page * pages.pageLen) * stride + channel;
    const size_t pageStride = size_t(pages.pageLen) * stride;
    const auto *key = reinterpret_cast<const uint4 *>(static_cast<const T *>(pages.key) + ids.x * pageStride + offset);
    const auto *value = reinterpret_cast<const uint4 *>(static_cast<const T *>(pages.value) + ids.y * pageStride + offset);
    if constexpr (ToShared) {
        using namespace flashinfer::cp_async;
        load_128b<PrefetchMode::kNoPrefetch>(&k, key);
        load_128b<PrefetchMode::kNoPrefetch>(&v, value);
    } else {
        k = *key;
        v = *value;
    }
}

// Score independent KV rows in parallel and consume V in chronological
// order. A bounded window fits in one tile. Global splits stream tiles and
// carry the same FP32 softmax state across page and tile boundaries.
constexpr int windowTileRows = 48;
template<int D, int G> constexpr int windowThreads = D * G > 256 ? D * G : 256;

// Preserve the same online softmax and value accumulation order for a single
// decode row and a window tile shared by several verification rows.
template <int G, bool Prefetch, bool ClampTail, size_t channelsPerLane, int D, class T>
__device__ __forceinline__ void accumulateRelativeValues(flashinfer::state_t<channelsPerLane> &state,
                                                         const float *scores, const T (*values)[D], int channel,
                                                         int n) {
    constexpr int tile = 2 * G;
    for (int base = 0; base < n; base += tile) {
        flashinfer::vec_t<float, channelsPerLane> prefetched[tile];
        if constexpr (Prefetch) {
#pragma unroll
            for (int t = 0; t < tile; ++t) prefetched[t].cast_load(&values[base + t][channel]);
        }
        float maximum = state.m;
#pragma unroll
        for (int t = 0; t < tile; ++t) maximum = fmaxf(maximum, scores[base + t]);
        const float rescale = flashinfer::math::ptx_exp2(state.m - maximum);
        state.m = maximum;
        state.d *= rescale;
#pragma unroll
        for (int c = 0; c < channelsPerLane; ++c) state.o[c] = __fmul_rn(state.o[c], rescale);
#pragma unroll
        for (int t = 0; t < tile; ++t) {
            const float w = flashinfer::math::ptx_exp2(scores[base + t] - maximum);
            state.d += w;
            flashinfer::vec_t<float, channelsPerLane> value;
            if constexpr (!Prefetch)
                value.cast_load(&values[ClampTail ? min(base + t, n - 1) : base + t][channel]);
            else
                value = prefetched[t];
#pragma unroll
            for (int c = 0; c < channelsPerLane; ++c) state.o[c] += w * value[c];
        }
    }
}

template<int D, int G, class T, bool SingleTile, bool Batched = false>
__global__ void tiledRelative(AttentionRow a, float *partial) {
    if constexpr (Batched) selectSplitQuery<D>(a, partial, blockIdx.x, gridDim.z);
    constexpr int vec = 16 / sizeof(T), lanes = D / vec, tile = 2 * G;
    constexpr int threads = SingleTile ? windowThreads<D,G> : 256, rowsPerPass = threads / lanes;
    // Global KV loads use two passes per stripe. Keep complete online-softmax
    // groups so splitting a stripe never changes FP32 accumulation order.
    constexpr int relativeTileRows = SingleTile ? windowTileRows : (2 * rowsPerPass / tile) * tile;
    constexpr int passes = (relativeTileRows + rowsPerPass - 1) / rowsPerPass;
    constexpr int channelsPerLane = SingleTile ? 4 : 2;
    static_assert(relativeTileRows % tile == 0 && threads >= G * D / channelsPerLane);
    __shared__ __align__(16) T values[relativeTileRows][D];
    __shared__ float scores[G][relativeTileRows];
    const int tid = threadIdx.x, split = blockIdx.z, head = blockIdx.y * G;
    const int past = a.visit->past, length = (SingleTile || a.window) ? min(past + 1, a.window) : past + 1;
    const int splits = gridDim.z, chunk = max(16, (length + splits - 1) / splits);
    const int lo = split * chunk, count = min(chunk, length - lo);
    if (count <= 0) return;
    const int lane = tid % lanes;
    const int kh = head / (a.heads / a.kvHeads);
    const PageView pages = a.visit->pages;
    const float scale = rsqrtf(float(a.width));
    flashinfer::vec_t<float,vec> query[G];
#pragma unroll
    for (int h = 0; h < G; ++h)
        query[h].cast_load(a.q + (head + h) * D + lane * vec);

    const float tau = (SingleTile || a.window) ? 1.f : 1.f + a.alpha * logf(fmaxf(1.f, (past + 1.f) / a.floor));
    const int h = tid / (D / channelsPerLane), channel = tid % (D / channelsPerLane) * channelsPerLane;
    flashinfer::state_t<channelsPerLane> state;
    for (int offset = 0; offset < count; offset += relativeTileRows) {
        const int n = SingleTile ? count : min(relativeTileRows, count - offset);
        const int padded = (n + tile - 1) / tile * tile;
        const int first = past + 1 - length + lo + offset;
        // Issue independent page reads before QK. The key vectors stay in
        // registers; each value vector is shared by the grouped query heads.
        uint4 keys[passes];
#pragma unroll
        for (int pass = 0; pass < passes; ++pass) {
            const int t = tid / lanes + pass * rowsPerPass;
            uint4 k{}, v{};
            if (t < n) loadRelativeKV<T>(pages, first + t, a.kvHeads * D, kh * D + lane * vec, k, v);
            keys[pass] = k;
            if (t < padded) *reinterpret_cast<uint4 *>(&values[t][lane * vec]) = v;
        }
#pragma unroll
        for (int pass = 0; pass < passes; ++pass) {
            const int t = tid / lanes + pass * rowsPerPass;
            constexpr int rowsPerWarp = 32 / lanes;
            if (t / rowsPerWarp * rowsPerWarp >= padded) continue;
            flashinfer::vec_t<float,vec> key;
            key.cast_load(reinterpret_cast<const T *>(&keys[pass]));
#pragma unroll
            for (int h = 0; h < G; ++h) {
                float dot = 0.f;
#pragma unroll
                for (int c = 0; c < vec; ++c) dot += query[h][c] * key[c];
#pragma unroll
                for (int delta = lanes / 2; delta; delta /= 2)
                    dot += __shfl_xor_sync(0xffffffffu, dot, delta);
                if (!lane && t < padded) {
                    const unsigned distance = length - 1u - lo - offset - t;
                    const float bias = distance < a.distances ? a.relative[size_t(head + h) * a.distances + distance] : 0.f;
                    scores[h][t] = t < n ? ((dot * scale + bias) * tau) * RelativeAttention::sm_scale_log2 : -flashinfer::math::inf;
                }
            }
        }
        __syncthreads();
        if (tid < G * D / channelsPerLane) {
            accumulateRelativeValues<G, !SingleTile, false>(state, scores[h], values, channel, n);
        }
        if constexpr (SingleTile) break;
        if (offset + relativeTileRows < count) __syncthreads();
    }
    if (tid >= G * D / channelsPerLane) return;
    const size_t row = size_t(head + h) * splits + split;
#pragma unroll
    for (int c = 0; c < channelsPerLane; ++c) state.o[c] /= state.d;
    state.o.store(partial + row * D + channel);
    if (!channel) partial[size_t(splits) * a.heads * D + row] = state.get_lse();
}

#include "attention-verify.cuh"

// Merge normalized partial outputs using their base-2 log normalizers.
// The learned sink is a zero-valued token and contributes to the denominator once.
template<int D, class Act, bool HeadMajor = false, bool Batched = false>
__global__ void mergeRelative(AttentionRow a, const float *partial, int splits,
                              const Act *sinks, const Act *gate, Act *output) {
    if constexpr (Batched) {
        const int row = blockIdx.z;
        a.visit += row;
        partial += size_t(row) * splits * a.heads * (D + 1);
        sinks += size_t(row) * a.sinkStride;
        if (gate) gate += size_t(row) * a.gateStride;
        output += size_t(row) * a.heads * D;
    }

    const int past = a.visit ? a.visit->past : a.past;
    const int length = a.window ? min(past + 1, a.window) : past + 1;
    const int chunk = max(16, (length + splits - 1) / splits);
    const int active = (length + chunk - 1) / chunk;
    extern __shared__ float weights[];
    const int head = blockIdx.x, lane = threadIdx.x;
    const size_t stride = size_t(a.heads) * D;
    const int splitStride = HeadMajor ? 1 : a.heads;
    const size_t firstRow = HeadMajor ? size_t(head) * splits : head;
    const float *lse = partial + stride * splits + firstRow;
    // The first warp finds the maximum and evaluates exp2 in parallel. Sum in
    // the original split order: even a one-ulp change can alter expert routing.
    if (lane < 32) {
        float m = float(sinks[head]) * RelativeAttention::sm_scale_log2;
        const float sink = m;
        for (int s = lane; s < active; s += 32) m = fmaxf(m, lse[s * splitStride]);
        for (int delta = 16; delta; delta /= 2)
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, delta));
        for (int s = lane; s < active; s += 32) weights[s] = exp2f(lse[s * splitStride] - m);
        __syncwarp();
        float sum = exp2f(sink - m);
        if (!lane)
            for (int s = 0; s < active; ++s) sum += weights[s];
        sum = __shfl_sync(0xffffffffu, sum, 0);
        for (int s = lane; s < active; s += 32) weights[s] /= sum;
    }
    __syncthreads();
    float out = 0.f;
    for (int s = 0; s < active; ++s)
        out += weights[s] * partial[(firstRow + size_t(s) * splitStride) * D + lane];
    const int channel = head * D + lane;
    if (gate) out *= 1.f / (1.f + expf(-float(gate[channel])));
    output[channel] = Act(out);
}

// Up to one warp of splits: prefetch the partials, broadcast normalizers
// with shuffles, and accumulate in the same split order as mergeRelative.
template<int D, class Act, bool HeadMajor = false, bool Batched = false>
__global__ void mergeRelativeWarp(AttentionRow a, const float *partial, int splits,
                                  const Act *sinks, const Act *gate, Act *output) {
    if constexpr (Batched) {
        const int row = blockIdx.z;
        a.visit += row;
        partial += size_t(row) * splits * a.heads * (D + 1);
        sinks += size_t(row) * a.sinkStride;
        if (gate) gate += size_t(row) * a.gateStride;
        output += size_t(row) * a.heads * D;
    }

    const int head = blockIdx.x, lane = threadIdx.x % 32;
    const int channel = head * D + blockIdx.y * blockDim.x + threadIdx.x;
    const int past = a.visit ? a.visit->past : a.past;
    const int length = a.window ? min(past + 1, a.window) : past + 1;
    const int chunk = max(16, (length + splits - 1) / splits);
    const int active = (length + chunk - 1) / chunk;
    const size_t stride = size_t(a.heads) * D;
    const int splitStride = HeadMajor ? 1 : a.heads;
    const size_t firstRow = HeadMajor ? size_t(head) * splits : head;
    float values[32];
#pragma unroll
    for (int s = 0; s < 32; ++s)
        values[s] = s < active ? partial[(firstRow + size_t(s) * splitStride) * D + (channel - head * D)] : 0.f;
    const float logW = lane < active ? partial[splits * stride + firstRow + lane * splitStride] : -INFINITY;
    const float sink = float(sinks[head]) * RelativeAttention::sm_scale_log2;
    float maximum = fmaxf(sink, logW);
#pragma unroll
    for (int offset = 16; offset; offset /= 2)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset));
    const float w = exp2f(logW - maximum);
    float sum = exp2f(sink - maximum);
#pragma unroll
    for (int s = 0; s < 32; ++s) sum += __shfl_sync(0xffffffffu, w, s);
    const float normalized = w / sum;
    float out = 0.f;
#pragma unroll
    for (int s = 0; s < 32; ++s) out += values[s] * __shfl_sync(0xffffffffu, normalized, s);
    if (gate) out *= 1.f / (1.f + expf(-float(gate[channel])));
    output[channel] = Act(out);
}

template<int D, int G, bool Split, class T>
void launchRelativeKernel(AttentionRow a, int rows, float *partial, int splits) {
    constexpr int vector = 16 / sizeof(T), lanes = D / vector, groups = Split ? 1 : 128 / lanes;
    constexpr int tile = 2 * G * groups;
    constexpr size_t storage = 2 * 2 * tile * D * sizeof(T) + tile * lanes * sizeof(size_t);
    relativeAttention<D,G,Split,T><<<dim3(rows, a.heads / G, splits), dim3(lanes,G,groups),
                                   storage, cudaStreamPerThread>>>(a, partial);
}

template<int D, bool HeadMajor, int Channels = 64, class Act>
void launchRelativeMerge(AttentionRow a, const Matrix &scratch, const Act *sink, const Act *gate, Act *out, int rows = 1) {
    const int splits = scratch.rows;
    if (splits <= 32) {
        static_assert(D % Channels == 0);
        auto kernel = rows > 1 ? mergeRelativeWarp<D,Act,HeadMajor,true> : mergeRelativeWarp<D,Act,HeadMajor>;
        kernel<<<dim3(a.heads,D/Channels,rows),Channels,0,cudaStreamPerThread>>>(a, scratch.p, splits, sink, gate, out);
    } else {
        auto kernel = rows > 1 ? mergeRelative<D,Act,HeadMajor,true> : mergeRelative<D,Act,HeadMajor>;
        kernel<<<dim3(a.heads,1,rows),D,splits*sizeof(float),cudaStreamPerThread>>>(a, scratch.p, splits, sink, gate, out);
    }
}

#include "attention-sm120.cuh"

template<int D, class T, class Act>
void launchSplitRelative(AttentionRow a, const Matrix &scratch, const Act *sink, const Act *gate, Act *out,
                         int sm120Splits = 0, int rows = 1, int chain = 1) {
    const int splits = scratch.rows;
    const bool grouped = a.heads == 3 * a.kvHeads;
    if (a.visit && !a.key && !a.value) {
        if (a.window > 0 && a.window <= int64_t(windowTileRows) * splits) {
            if constexpr (D == 128 && std::is_same_v<T, half>) {
                if (grouped && chain > 1) {
                    constexpr int tile = 4;
                    const int blocks = (rows / chain) * ((chain + tile - 1) / tile);
                    verifyWindowTile<tile><<<dim3(blocks, a.kvHeads, splits), tile * 3 * 32, 0, cudaStreamPerThread>>>(
                        a, scratch.p, chain);
                    launchRelativeMerge<D,true>(a, scratch, sink, gate, out, rows);
                    return;
                }
            }
            if (grouped)
                (rows > 1 ? tiledRelative<D,3,T,true,true> : tiledRelative<D,3,T,true>)<<<dim3(rows,a.kvHeads,splits),windowThreads<D,3>,0,cudaStreamPerThread>>>(a, scratch.p);
            else
                (rows > 1 ? tiledRelative<D,1,T,true,true> : tiledRelative<D,1,T,true>)<<<dim3(rows,a.heads,splits),windowThreads<D,1>,0,cudaStreamPerThread>>>(a, scratch.p);
            launchRelativeMerge<D,true>(a, scratch, sink, gate, out, rows);
            return;
        }
        if constexpr (D == 128 && std::is_same_v<T, half>) {
            if (!a.window && sm120Splits > 0 && sm120Splits <= splits && grouped && a.width == D) {
                const int tiles = (3 * chain + 3) / 4;
                if (tiles < chain)
                    globalRelativeSm120<true><<<dim3((rows / chain) * tiles, a.kvHeads, sm120Splits), 128, 0,
                                                cudaStreamPerThread>>>(a, scratch.p, chain);
                else
                    globalRelativeSm120<false><<<dim3(rows, a.kvHeads, sm120Splits), 128, 0,
                                                 cudaStreamPerThread>>>(a, scratch.p);
                launchRelativeMerge<D,true,32>(a, scratch.slice(0, sm120Splits), sink, gate, out, rows);
                return;
            }
        }
        // Long windows and global attention both consume stripes in chronological order.
        if (grouped)
            (rows > 1 ? tiledRelative<D,3,T,false,true> : tiledRelative<D,3,T,false>)<<<dim3(rows,a.kvHeads,splits),256,0,cudaStreamPerThread>>>(a, scratch.p);
        else
            (rows > 1 ? tiledRelative<D,1,T,false,true> : tiledRelative<D,1,T,false>)<<<dim3(rows,a.heads,splits),256,0,cudaStreamPerThread>>>(a, scratch.p);
        launchRelativeMerge<D,true,32>(a, scratch, sink, gate, out, rows);
        return;
    } else if (grouped) {
        launchRelativeKernel<D,3,true,T>(a, 1, scratch.p, splits);
    } else {
        launchRelativeKernel<D,1,true,T>(a, 1, scratch.p, splits);
    }
    launchRelativeMerge<D,false>(a, scratch, sink, gate, out);
}

template<int D, class T>
void launchRelative(AttentionRow a, int rows, bool single, const Matrix &scratch) {
    if (single) launchSplitRelative<D, T>(a, scratch, a.sink, static_cast<const float *>(nullptr), a.output);
    else launchRelativeKernel<D,1,false,T>(a, rows, nullptr, 1);
}
