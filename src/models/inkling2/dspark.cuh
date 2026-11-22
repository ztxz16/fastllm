// The embedded draft uses the Inkling2 rnn_hidden architecture. The target
// feature convention and Markov recurrence follow the official Inkling2 model.
struct DraftKV {
    Buffer<half> key, value;
    int first = 0, capacity = 0;
    void reserve(int begin, int past, int end, int heads, int dim, int window, int block) {
        if (key.p && begin >= first && end - first <= capacity)
            return;
        int kept = std::max(0, past - begin);
        int grown = std::max(32, capacity * 2);
        if (window)
            grown = std::min(grown, 2 * window + block);
        int next = std::max(end - begin, grown);
        Buffer<half> k(size_t(heads) * next * dim), v(size_t(heads) * next * dim);
        if (kept) {
            if (!key.p || begin < first || past - first > capacity)
                throw std::runtime_error("Missing Inkling draft KV prefix");
            for (auto side : {std::make_pair(k.p, key.p), std::make_pair(v.p, value.p)})
                check(cudaMemcpy2DAsync(side.first, size_t(next) * dim * sizeof(half),
                                        side.second + size_t(begin - first) * dim,
                                        size_t(capacity) * dim * sizeof(half), size_t(kept) * dim * sizeof(half), heads,
                                        cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        }
        key = std::move(k);
        value = std::move(v);
        capacity = next;
        first = begin;
    }
};

struct DraftState {
    std::vector<DraftKV> kv;
    Matrix recurrentContext;
    Data samplerWorkspace;
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

// Uploaded before replay. A cache can grow or slide without invalidating the graph.
struct DraftGraphKV {
    half *key, *value;
    int past, first, capacity, begin;
};

using DraftFlashParams = flashinfer::SinglePrefillParams<half, half, half>;
using DraftFlashTraits = flashinfer::KernelTraits<flashinfer::MaskMode::kNone, 64, 1, 2, 8, 8, 4, 1,
    flashinfer::PosEncodingMode::kNone, half, half, half, float, int,
    flashinfer::DefaultAttention<false, false, false, false>>;

// The cache view is already cropped to the visible window. Every query sees
// the complete live draft block, including positions to its right.
__global__ __launch_bounds__(128) void draftGraphAttention(DraftFlashParams p, const DraftGraphKV *view, int *indptr) {
    extern __shared__ uint8_t smem[];
    const auto v = *view;
    p.kv_len = v.past - v.begin + p.qo_len;
    const int splits = min(int(gridDim.y), 1 + (int(p.kv_len) - 1) / 128);
    if (!blockIdx.x && !blockIdx.y && !blockIdx.z && threadIdx.x == 0 && threadIdx.z == 0)
        for (unsigned i = threadIdx.y; i <= p.qo_len; i += blockDim.y) indptr[i] = i * splits;
    if (int(blockIdx.y) >= splits) return;
    p.k = v.key + size_t(v.begin - v.first) * 128;
    p.v = v.value + size_t(v.begin - v.first) * 128;
    p.k_stride_h = p.v_stride_h = v.capacity * 128;
    p.partition_kv = splits > 1;
    auto &storage = *reinterpret_cast<DraftFlashTraits::SharedStorageSingle *>(smem);
    flashinfer::SinglePrefillWithKVCacheDevice<DraftFlashTraits>(p, storage, threadIdx, blockIdx.x,
        blockIdx.y, blockIdx.z, splits, gridDim.z);
}

// Transpose to [head, position, channel] while applying per-head RMSNorm
// and split-half RoPE. A null norm denotes V, which only needs the transpose.
template <class T, int Side = -1>
__global__ void draftHead(const T *src, half *dst, const float *norm, int heads, int dim, int past, int first,
                          int capacity, float eps, float theta, const DraftGraphKV *view = nullptr) {
    if constexpr (Side >= 0) {
        past = view->past;
        if constexpr (Side == 0) first = past;
        else {
            first = view->first;
            capacity = view->capacity;
            dst = Side == 1 ? view->key : view->value;
        }
    }
    int row = blockIdx.x, head = blockIdx.y, col = threadIdx.x;
    const T *in = src + (size_t(row) * heads + head) * dim;
    float x = float(in[col]);
    if (norm) {
        __shared__ float tmp[threads];
        float inv = rsqrtf(blockSum(x * x, tmp) / dim + eps);
        int paired = (col + dim / 2) % dim;
        float y = float(T(float(in[paired]) * inv * norm[paired]));
        x = float(T(x * inv * norm[col]));
        float si, co;
        sincosf((past + row) * powf(theta, -2.f * (col % (dim / 2)) / dim), &si, &co);
        x = float(T(x * co + (col < dim / 2 ? -y : y) * si));
    }
    dst[(size_t(head) * capacity + past + row - first) * dim + col] = half(x);
}
template <class T> __global__ void draftUnpack(const half *in, T *out, int rows, int heads, int dim) {
    int row = blockIdx.x, head = blockIdx.y, col = threadIdx.x;
    out[(size_t(row) * heads + head) * dim + col] = T(float(in[(size_t(head) * rows + row) * dim + col]));
}
template <class T> __global__ void draftAdd(T *dest, const T *add, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        dest[i] = T(float(dest[i]) + float(add[i]));
}
template <class T> __global__ void draftSwiGLU(T *gate, const T *up, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float v = float(gate[i]);
        gate[i] = T(v / (1.f + expf(-v)) * float(up[i]));
    }
}
template <class T> __global__ void draftTokenInput(T *input, const T *embedding, const int *token, int rank) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < rank)
        input[rank + col] = embedding[size_t(*token) * rank + col];
}
template <class T> __global__ void draftRecur(const float *raw, const float *bias, T *state, T *latent, int rank) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rank)
        return;
    float g = float(T(raw[i] + bias[i])), c = float(T(raw[i + rank] + bias[i + rank]));
    float o = float(T(raw[i + 2 * rank] + bias[i + 2 * rank]));
    g = float(T(1.f / (1.f + expf(-g))));
    state[i] = T(float(T(g * float(state[i]))) + float(T(float(T(1.f - g)) * float(T(tanhf(c))))));
    latent[i] = T(tanhf(o));
}
template <class T>
__global__ void draftScores(const T *base, const T *adjust, const T *bias, float *out, int vocab, float cap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < vocab)
        out[i] = float(T(cap * tanhf(float(T(float(base[i]) + float(T(float(adjust[i]) + float(bias[i]))))) / cap)));
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
