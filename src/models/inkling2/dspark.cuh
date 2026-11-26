// The embedded draft uses the Inkling2 rnn_hidden architecture. The target
// feature convention and Markov recurrence follow the official Inkling2 model.
// Request-owned handles borrow pages from the same manager as target attention.
struct DraftKV {
    Data key, value;
    VisitCache cache;
};
struct DraftView {
    PageView pages;
    float *context;
    int past, begin, window = 0;
};
struct DraftInput {
    int anchor, topK, steps;
    float temperature, topP;
};
// The verifier's output contains the exact next anchor and committed length.
// Draft-only K/V lives in graph scratch. Rejected or cancelled lookahead
// therefore cannot publish tokens, histories or writes to request pages.
__global__ void advanceDraft(const int *result, DraftInput *inputs, DraftView *views,
                             int batch, int chain, int layers) {
    const int row = blockIdx.x, accepted = result[batch * chain + row];
    if (accepted < 0 || accepted >= chain) return;
    if (!threadIdx.x) inputs[row].anchor = result[row * chain + accepted];
    for (int l = threadIdx.x; l < layers; l += blockDim.x) {
        auto &v = views[l * batch + row];
        v.past += accepted + 1;
        v.begin = v.window ? max(0, v.past - v.window) : 0;
    }
}
// A request may be cancelled while a queued draft still reads its private KV.
struct DraftFence {
    cudaEvent_t event = nullptr;
    int gpu;
    DraftFence() {
        check(cudaGetDevice(&gpu));
        check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
    ~DraftFence() {
        int old = gpu;
        cudaGetDevice(&old);
        if (old != gpu) cudaSetDevice(gpu);
        cudaEventDestroy(event);
        if (old != gpu) cudaSetDevice(old);
    }
};
struct DraftState {
    std::shared_ptr<DraftFence> reader;
    std::vector<std::unique_ptr<DraftKV>> kv;
    Matrix recurrentContext;
    double roundMs = 0;
    std::deque<std::pair<int, int>> pending;
    int rounds = 0, proposed = 0, accepted = 0;
};

template <class T> __global__ void draftHistory(const T *input, float *out, int stride, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width)
        out[size_t(blockIdx.y) * width + col] = float(input[size_t(blockIdx.y) * stride + col]);
}

struct Verification {
    std::vector<std::function<void(int)>> writes;
    template <class T>
    void history(Matrix &dest, const Matrix &old, Rows<T> storage, const T *input, int stride, int width, int total,
                 int depth) {
        if (!depth)
            return;
        writes.emplace_back([&dest, old, storage, input, stride, width, total, depth](int prefix) {
            if (prefix == total)
                return;
            Matrix next(depth, width);
            int keep = std::max(0, depth - prefix);
            if (keep)
                check(cudaMemcpyAsync(next.p, old.p + size_t(prefix) * width, size_t(keep) * width * sizeof(float),
                                      cudaMemcpyDeviceToDevice, cudaStreamPerThread));
            dim3 grid((width + 255) / 256, depth - keep);
            draftHistory<<<grid, 256, 0, cudaStreamPerThread>>>(input + size_t(std::max(0, prefix - depth)) * stride,
                                                                next.p + size_t(keep) * width, stride, width);
            dest = std::move(next);
        });
    }
};

template <class T>
__global__ void draftMix(const float *const *taps, const float *alpha, float *out, int count, int width, int offset) {
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= width)
        return;
    int row = blockIdx.x;
    float v = 0;
    for (int j = 0; j < count; ++j)
        v += alpha[j] * float(T(taps[j][size_t(row + offset) * width + i]));
    out[size_t(row) * width + i] = v;
}

template<int Warps> using DraftFlashTraits = flashinfer::KernelTraits<flashinfer::MaskMode::kNone, 16 * Warps, 1, 2, 8, 8, Warps, 1,
    flashinfer::PosEncodingMode::kNone, half, half, half, float, int,
    flashinfer::DefaultAttention<false, false, false, false>>;

struct DraftPages : PrefillPages<half, 128> {
    int begin;
    __device__ size_t protective_get_kv_offset(unsigned token, unsigned head, unsigned, unsigned col, int) const {
        return (size_t(begin + (token < length ? token : 0)) << 32) | (head * 128 + col);
    }
};
struct DraftFlashParams : flashinfer::BatchPrefillPagedParams<half, half, half, int> {
    DraftPages paged_kv;
    const DraftView *views;
    int block, maxSplits;
    const half *keySuffix, *valueSuffix;
    __host__ __device__ unsigned get_qo_len(unsigned) const { return block; }
    __host__ __device__ unsigned get_kv_len(unsigned) const { return paged_kv.length; }
};

__device__ inline int draftChunk(int length, int splits) { return max(128, (length + splits - 1) / splits); }

// Each request has its own logical position, window and page table. All requests
// share one launch, including when they span different numbers of physical pages.
template<class Traits>
__global__ __launch_bounds__(Traits::NUM_THREADS) void draftGraphAttention(DraftFlashParams p) {
    cuda_launch::DependencyStart();
    int tiles = (p.block * int(p.group_size) + Traits::CTA_TILE_Q - 1) / Traits::CTA_TILE_Q;
    int b = blockIdx.x / tiles, tile = blockIdx.x % tiles, split = blockIdx.y, zero = 0;
    auto v = p.views[b];
    int length = v.past - v.begin + p.block, chunk = draftChunk(length, p.maxSplits);
    if (split * chunk >= length) return;
    p.q += size_t(b) * p.block * p.q_stride_n;
    size_t offset = size_t(b) * p.block * p.maxSplits * p.num_qo_heads;
    p.o += offset * 128;
    p.lse += offset;
    p.paged_kv.length = length;
    p.paged_kv.begin = v.begin;
    int keys[] = {0, length};
    p.paged_kv.indptr = keys;
    const int stride = int(gridDim.z) * 128;
    const size_t row = size_t(b) * p.block * stride;
    p.paged_kv.k_data = {v.pages, p.keySuffix ? p.keySuffix + row : nullptr, 0, v.past, stride, false};
    p.paged_kv.v_data = {v.pages, p.valueSuffix ? p.valueSuffix + row : nullptr, 0, v.past, stride, true};
    p.request_indices = p.q_indptr = p.o_indptr = &zero;
    p.qo_tile_indices = &tile;
    p.kv_tile_indices = &split;
    p.kv_chunk_size_ptr = &chunk;
    extern __shared__ uint8_t smem[];
    auto &storage = *reinterpret_cast<typename Traits::SharedStoragePaged *>(smem);
    // Keep the logical partitions and FP16 partials independent of the launch grid.
    for (; split * chunk < length; split += gridDim.y) {
        flashinfer::BatchPrefillWithPagedKVCacheDevice<Traits>(p, storage, threadIdx, 0, blockIdx.z, gridDim.z);
        if ((split + gridDim.y) * chunk < length) __syncthreads();
    }
}

// Portable paged fallback for devices without the tensor-core prefill kernel.
__global__ void draftAttentionPartial(DraftFlashParams p, const DraftView *views, int kvHeads) {
    cuda_launch::DependencyStart();
    int row = blockIdx.x, b = row / p.block, head = blockIdx.z, col = threadIdx.x;
    auto v = views[b];
    int len = v.past - v.begin + p.block, chunk = draftChunk(len, gridDim.y);
    int first = blockIdx.y * chunk;
    if (first >= len) return;
    const int stride = kvHeads * 128, kv = head / int(p.group_size);
    float q = float(p.q[(size_t(row) * p.num_qo_heads + head) * 128 + col]);
    __shared__ float temp[128];
    float m = -INFINITY, sum = 0.f, out = 0.f;
    for (int i = first; i < min(len, first + chunk); ++i) {
        auto token = v.begin + i;
        float key = float(v.pages.row<half>(token, false, stride)[kv * 128 + col]);
        float score = blockSum(q * key, temp) * p.sm_scale, next = fmaxf(score, m);
        float a = expf(m - next), w = expf(score - next);
        out = out * a +
              w * float(v.pages.row<half>(token, true, stride)[kv * 128 + col]);
        sum = sum * a + w;
        m = next;
    }
    int splits = (len + chunk - 1) / chunk;
    size_t idx = ((size_t(b) * gridDim.y * p.block + (row % p.block) * splits + blockIdx.y) * p.num_qo_heads + head);
    p.o[idx * 128 + col] = half(out / sum);
    if (!col) p.lse[idx] = (m + logf(sum)) * flashinfer::math::log2e;
}

template <class T>
__global__ void mergeDraftAttention(DraftFlashParams p, const DraftView *views, T *out, int maxSplits) {
    cuda_launch::DependencyStart();
    int row = blockIdx.x, b = row / p.block, head = blockIdx.y, col = threadIdx.x;
    int len = views[b].past - views[b].begin + p.block;
    int splits = (len - 1) / draftChunk(len, maxSplits) + 1;
    size_t base = (size_t(b) * p.block * maxSplits + (row % p.block) * splits) * p.num_qo_heads + head;
    float maximum = -INFINITY, total = 0.f, value = 0.f;
    for (int s = 0; s < splits; ++s) maximum = fmaxf(maximum, p.lse[base + s * p.num_qo_heads]);
    for (int s = 0; s < splits; ++s) {
        auto i = base + s * p.num_qo_heads;
        float w = exp2f(p.lse[i] - maximum);
        total += w;
        value += w * float(p.o[i * 128 + col]);
    }
    out[(size_t(row) * p.num_qo_heads + head) * 128 + col] = T(float(half(value / total)));
}

// Side 0 produces token-major queries. Sides 1 and 2 append into request KV.
template <class T, int Side>
__device__ __forceinline__ void draftHeadBody(const T *src, half *query, const float *norm, int heads, int dim, float eps, float theta,
                          int block, DraftView single, const DraftView *views, const int *accepted, int head, half *suffix = nullptr) {
    int row = blockIdx.x, b = row / block, pos = row % block, col = threadIdx.x;
    if (accepted && pos > accepted[b]) return;
    auto view = views ? views[b] : single;
    int token = view.past + pos;
    const T *in = src + (size_t(row) * heads + head) * dim;
    float x = float(in[col]);
    if (norm) {
        __shared__ float tmp[threads];
        float inv = rsqrtf(blockSum(x * x, tmp) / dim + eps);
        int paired = (col + dim / 2) % dim;
        float y = float(T(float(in[paired]) * inv * norm[paired]));
        x = float(T(x * inv * norm[col]));
        float si, co;
        sincosf(token * powf(theta, -2.f * (col % (dim / 2)) / dim), &si, &co);
        x = float(T(x * co + (col < dim / 2 ? -y : y) * si));
    }
    if constexpr (Side == 0)
        query[(size_t(row) * heads + head) * dim + col] = half(x);
    else {
        if (suffix) {
            suffix[(size_t(row) * heads + head) * dim + col] = half(x);
            return;
        }
        view.pages.row<half>(token, Side == 2, heads * dim)[head * dim + col] = half(x);
    }
}

template<class T, int Side>
__global__ void draftHead(const T *src, half *query, const float *norm, int heads, int dim, float eps, float theta,
                          int block, DraftView single, const DraftView *views, const int *accepted = nullptr) {
    draftHeadBody<T, Side>(src, query, norm, heads, dim, eps, theta, block, single, views, accepted, blockIdx.y);
}

template<class T> __global__ void draftPrepareQkv(const T *q, const T *k, const T *v, half *query,
        const float *qn, const float *kn, int heads, int kvHeads, int dim, float eps, float theta,
        int block, const DraftView *views, half *keySuffix, half *valueSuffix) {
    cuda_launch::DependencyStart();
    const int head = blockIdx.y;
    if (head < heads)
        draftHeadBody<T, 0>(q, query, qn, heads, dim, eps, theta, block, {}, views, nullptr, head);
    else if (head < heads + kvHeads)
        draftHeadBody<T, 1>(k, nullptr, kn, kvHeads, dim, eps, theta, block, {}, views, nullptr, head - heads, keySuffix);
    else
        draftHeadBody<T, 2>(v, nullptr, nullptr, kvHeads, dim, eps, theta, block, {}, views, nullptr, head - heads - kvHeads, valueSuffix);
}

template<class T, bool Add> __global__ void draftResidualNorm(T *x, const T *delta, const float *weights,
                                                             T *out, int width, float eps) {
    cuda_launch::DependencyStart();
    const size_t first = size_t(blockIdx.x) * width;
    __shared__ float scratch[threads];
    float squares = 0.f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float v = float(x[first + i]);
        if constexpr (Add) {
            const T sum = T(v + float(delta[first + i]));
            x[first + i] = sum;
            v = float(sum);
        }
        squares += v * v;
    }
    const float scale = rsqrtf(blockSum(squares, scratch) / width + eps);
    for (int i = threadIdx.x; i < width; i += blockDim.x)
        out[first + i] = T(float(x[first + i]) * scale * weights[i]);
}

template <class T> __global__ void draftBlockInput(const T *anchor, const float *mask, T *out, int block, int h) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (i < h) out[size_t(row) * h + i] = row % block ? T(mask[i]) : anchor[size_t(row / block) * h + i];
}
template <class T>
__global__ void draftMarkovInput(T *state, const T *embed, const int *tokens, const T *hidden, const DraftView *views,
                                 int step, int steps, int rank, int h) {
    cuda_launch::DependencyStart();
    int b = blockIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
    state += size_t(b) * (2 * rank + 2 * h);
    if (col < rank) {
        if (!step) state[col] = T(0.f);
        state[rank + col] = embed[size_t(tokens[b * (steps + 1) + step]) * rank + col];
    }
    if (col < h) {
        state[2 * rank + col] = hidden[(size_t(b) * steps + step) * h + col];
        if (!step) state[2 * rank + h + col] = T(views[b].context[col]);
    }
}
__global__ void storeDraftContext(const float *source, const DraftView *views, const int *accepted, int chain, int h) {
    int b = blockIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < h) views[b].context[col] = source[(size_t(b) * chain + accepted[b]) * h + col];
}
template <class T> __global__ void draftAdd(T *dest, const T *add, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        dest[i] = T(float(dest[i]) + float(add[i]));
}
template <class T> __global__ void draftSwiGLU(T *gate, const T *up, int size) {
    cuda_launch::DependencyStart();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float v = float(gate[i]);
        gate[i] = T(v / (1.f + expf(-v)) * float(up[i]));
    }
}
template<class T> __device__ void draftRecurOutput(float gate, float candidate, float out, T previous,
                                                   T &state, T &latent) {
    float g = float(T(gate)), c = float(T(candidate)), o = float(T(out));
    g = float(T(1.f / (1.f + expf(-g))));
    state = T(float(T(g * float(previous))) + float(T(float(T(1.f - g)) * float(T(tanhf(c))))));
    latent = T(tanhf(o));
}
template <class T>
__global__ void draftRecur(const float *raw, const float *bias, T *state, T *latent, int rank, int pitch) {
    cuda_launch::DependencyStart();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rank)
        return;
    raw += size_t(blockIdx.y) * 3 * rank;
    state += size_t(blockIdx.y) * pitch;
    latent += size_t(blockIdx.y) * rank;
    draftRecurOutput(raw[i] + bias[i], raw[i + rank] + bias[i + rank], raw[i + 2 * rank] + bias[i + 2 * rank],
                     state[i], state[i], latent[i]);
}

template<class T> __global__ void draftMarkovContext(const DraftView *views, T *out, int h) {
    cuda_launch::DependencyStart();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < h) out[size_t(blockIdx.y) * h + i] = T(views[blockIdx.y].context[i]);
}

// Only the previous state and token embedding vary serially. The wider
// hidden and context projections have already been computed for the window.
template<class T> __global__ void draftMarkovStep(const T *weights, const T *embed, const int *tokens,
        const float *hidden, const float *context, const float *bias, const T *previous,
        T *next, T *latent, int rank, int pitch, int steps, int step) {
    cuda_launch::DependencyStart();
    int i = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (i >= rank) return;
    const int lane = threadIdx.x % 32, b = blockIdx.y;
    const T *word = embed + size_t(tokens[size_t(b) * (steps + 1) + step]) * rank;
    previous += size_t(b) * rank;
    const T *row = weights + size_t(i) * pitch;
    float dot[3]{};
    for (int k = lane; k < rank; k += 32) {
        const float v = step ? float(previous[k]) : 0.f, w = float(word[k]);
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const T *a = row + size_t(j) * rank * pitch;
            dot[j] = fmaf(v, float(a[k]), dot[j]);
            dot[j] = fmaf(w, float(a[rank+k]), dot[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        for (int d = 16; d; d /= 2) dot[j] += __shfl_down_sync(0xffffffffu, dot[j], d);
        int col = j * rank + i;
        if (!lane) dot[j] += hidden[(size_t(b) * steps + step) * (3 * rank) + col]
                            + context[size_t(b) * (3 * rank) + col] + bias[col];
    }
    if (!lane) draftRecurOutput(dot[0], dot[1], dot[2], step ? previous[i] : T(0.f),
                                next[size_t(b)*rank+i], latent[size_t(b)*rank+i]);
}
// One chronological panel per (layer visit, request, convolution) holds the old
// history followed by every candidate input. Accepting k inputs advances the
// history by k rows, including across physical KV page and window boundaries.
template<bool Commit>
__global__ void verifyHistories(const DecodeVisit *source, DecodeVisit *rows, float *panels,
                                const int *accepted, int batch, int chain, int depth, int kv, int hidden) {
    const int visit = blockIdx.x, side = blockIdx.y, width = side < 2 ? kv : hidden;
    const int pitch = depth + chain;
    const size_t channel = side < 2 ? side * kv : 2 * kv + (side - 2) * hidden;
    float *base = panels + (size_t(visit) * 2 * (kv + hidden) + channel) * pitch;
    const DecodeVisit original = source[visit];
    float *history = original.history[side];
    if constexpr (Commit) {
        const int prefix = accepted[visit % batch] + 1;
        for (int i = threadIdx.x; i < depth * width; i += blockDim.x)
            history[i] = base[size_t(prefix) * width + i];
    } else {
        for (int i = threadIdx.x; i < depth * width; i += blockDim.x) base[i] = history[i];
        if (side == 0) {
            for (int r = threadIdx.x; r < chain; r += blockDim.x) {
                auto d = original;
                d.past += min(r, original.verifyRemaining - 1);
                d.verifyRemaining -= r;
                for (int j = 0; j < 4; ++j) {
                    const int offset = j < 2 ? j * kv : 2 * kv + (j - 2) * hidden;
                    d.history[j] = base + size_t(offset) * pitch + size_t(r) * (j < 2 ? kv : hidden);
                }
                rows[size_t(visit) * chain + r] = d;
            }
        }
    }
}

__global__ void stageDraftInputs(const DraftInput *inputs, float *anchors, int *tokens, float *t, int *k, float *p,
                                 int *live, int batch, int chain, int vocab) {
    int b = blockIdx.x;
    const auto x = inputs[b];
    if (!threadIdx.x) {
        live[b] = x.steps;
        anchors[b] = float(x.anchor);
        tokens[b * chain] = x.anchor;
        t[b] = x.temperature;
        k[b] = min(vocab, x.topK);
        p[b] = x.topP;
    }
    for (int r = threadIdx.x; r < chain; r += blockDim.x) {
        int i = batch + b * chain + r;
        t[i] = x.temperature;
        k[i] = min(vocab, x.topK);
        p[i] = x.topP;
    }
}

// Match the activation-precision correction, addition and softcap before
// exchanging the vocabulary shards. Padding is excluded by the full-vocab sampler.
template<class T> __global__ void mergeDraftScores(const T *base, int stride, T *adjust, const T *bias,
                                                   int vocab, size_t count, float cap) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int col = i % vocab;
    const T correction = T(float(adjust[i]) + float(bias[col]));
    const T score = T(float(base[(i / vocab) * stride + col]) + float(correction));
    adjust[i] = T(cap * tanhf(float(score) / cap));
}
