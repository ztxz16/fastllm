template<class T> struct ProjectionView {
    const T *data;
    int stride;
    __device__ float read(int row, int col) const { return float(data[size_t(row) * stride + col]); }
};
template<class T> struct PrefillParts { ProjectionView<T> items[8]; };
template<class Act> struct RelativeInputs {
    PrefillParts<Act> sources;
    const float *qs, *rs, *kw, *vw, *table;
    int heads, kvHeads, dim, rel, extent, taps;
    float qClip, rClip, eps;
};

__device__ float headScale(float v, float clip, int width, float eps, float *scratch) {
    const float sum = blockSum(v * v, scratch);
    return clip > 0 ? fminf(1.f, clip / (sqrtf(sum) + 1e-6f)) : rsqrtf(sum / width + eps);
}

template<class T, bool Timeline = false, bool PackQuery = false, class Act>
__global__ void prepareRelative(const DecodeVisit *visit, RelativeInputs<Act> in, int tiles, float *q, float *bias, int chain = 1) {
    int row = 0, priorRows = 0;
    if constexpr (Timeline) {
        row = blockIdx.y;
        priorRows = row % chain;
        visit += row;
        q += size_t(row) * in.heads * in.dim;
        bias += size_t(row) * in.heads * in.extent;
    }
    __shared__ float scratch[threads], relative[threads];
    const int col = threadIdx.x, task = blockIdx.x;
    if (task < in.heads) {
        int i = task * in.dim + col;
        float x = col < in.dim ? in.sources.items[0].read(row, i) : 0.f;
        float w = col < in.dim ? in.qs[i] : 0.f;
        float scale = headScale(in.qClip > 0 ? x*w : x, in.qClip, in.dim, in.eps, scratch);
        if (col < in.dim) {
            const float value = x * scale * w;
            if constexpr (PackQuery) {
                half *packed = reinterpret_cast<half *>(q) + 2 * task * in.dim + col;
                packed[0] = half(value);
                packed[in.dim] = half(value - float(packed[0]));
            } else q[i] = value;
        }
        return;
    }
    int idx = task - in.heads;
    if (idx < in.heads * tiles) {
        const int head = idx / tiles, dist = idx % tiles * threads + col;
        if (idx % tiles * threads > visit->past) return;
        int i = head * in.rel + col;
        float x = col < in.rel ? in.sources.items[1].read(row, i) : 0.f;
        float w = col < in.rel ? in.rs[i] : 0.f;
        float scale = headScale(in.rClip > 0 ? x*w : x, in.rClip, in.rel, in.eps, scratch);
        relative[col] = x * scale * w;
        __syncthreads();
        if (dist < in.extent && dist <= visit->past) {
            float sum = 0;
            for (int j = 0; j < in.rel; ++j) sum += relative[j] * in.table[j * in.extent + dist];
            bias[head * in.extent + dist] = sum;
        }
        return;
    }
    idx -= in.heads * tiles;
    const int which = idx / in.kvHeads, channel = idx % in.kvHeads * in.dim + col;
    const int width = in.kvHeads * in.dim;
    const auto input = in.sources.items[which ? 6 : 4], dynamic = in.sources.items[which ? 7 : 5];
    float x = col < in.dim ? updateConvChannel<Timeline>(input.data + size_t(row) * input.stride, visit, which,
        which ? in.vw : in.kw, dynamic.data ? dynamic.data + size_t(row) * dynamic.stride : nullptr,
        width, in.taps, in.kvHeads, channel, priorRows, input.stride) : 0.f;
    float scale = headScale(x, which ? 0.f : in.qClip, in.dim, in.eps, scratch);
    if (col < in.dim && (!Timeline || visit->verifyRemaining > 0)) {
        auto dst = static_cast<T *>(which ? visit->pages.value : visit->pages.key);
        dst[visit->pages.offset(visit->past, which, width) + channel] = T(x * scale);
    }
}

struct PrefillPreparation {
    const float *qs, *rs, *conv[2];
    int heads, kvHeads, dim, rel, taps;
    float qClip, rClip, eps;
};

// One warp owns a head. Preserve the original FP32 reduction order before
// converting normalized channels into persistent cache or projection types.
__device__ float prefillScale(const float (&v)[4], float clip, int width, float eps, bool rmsKV) {
    float sum;
    if (rmsKV) {
        sum = v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3];
    } else {
        // blockSum first stores all squares, then sums lanes 64 and 32 apart.
        sum = (__fmul_rn(v[0], v[0]) + __fmul_rn(v[2], v[2])) +
              (__fmul_rn(v[1], v[1]) + __fmul_rn(v[3], v[3]));
    }
    for (int step = 16; step; step /= 2) sum += __shfl_down_sync(0xffffffff, sum, step);
    sum = __shfl_sync(0xffffffff, sum, 0);
    return clip > 0 ? fminf(1.f, clip / (sqrtf(sum) + 1e-6f)) : rsqrtf(sum / width + eps);
}

template<bool Compact, class T, class KV, class R = float> __global__ void preparePrefill(PrefillParts<T> p, PrefillPreparation cfg,
        int offset, int length, const float *kh, const float *vh, float *nextK, float *nextV,
        KV *q, R *r, float *sink, KV *k, KV *v) {
    const unsigned rowIndex = Compact ? blockIdx.y + blockIdx.z * gridDim.y : blockIdx.x;
    if (rowIndex >= unsigned(length)) return;
    const int row = rowIndex, token = offset + row, lane = threadIdx.x % 32;
    if constexpr (Compact) cfg.dim = 128;
    int task = (Compact ? blockIdx.x : blockIdx.y) * (blockDim.x / 32) + threadIdx.x / 32;
    if (task >= 2 * (cfg.heads + cfg.kvHeads)) return;
    if (task < 2 * cfg.heads) {
        const bool rel = task >= cfg.heads;
        const int head = rel ? task - cfg.heads : task, width = rel ? cfg.rel : cfg.dim;
        const float clip = rel ? cfg.rClip : cfg.qClip;
        float x[4]{}, weight[4]{}, scaled[4]{};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int col = lane + 32 * j, channel = head * width + col;
            if (col < width) {
                x[j] = p.items[rel ? 1 : 0].read(token, channel);
                weight[j] = (rel ? cfg.rs : cfg.qs)[channel];
                scaled[j] = clip > 0 ? x[j] * weight[j] : x[j];
            }
        }
        const float factor = prefillScale(scaled, clip, width, cfg.eps, false);
        const size_t base = (size_t(token) * cfg.heads + head) * width;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int col = lane + 32 * j;
            if (col < width) {
                const float value = x[j] * factor * weight[j];
                if (rel) r[base + col] = R(value);
                else if constexpr (Compact) q[base + col] = KV(value);
                else {
                    const KV high(value);
                    q[2 * base + col] = high;
                    q[2 * base + width + col] = KV(value - float(high));
                }
            }
        }
        if (!rel && lane == 0) sink[size_t(token) * cfg.heads + head] = p.items[3].read(token, head);
        return;
    }
    task -= 2 * cfg.heads;
    const int side = task / cfg.kvHeads, head = task % cfg.kvHeads;
    const int width = cfg.kvHeads * cfg.dim;
    const auto x = p.items[side ? 6 : 4], dynamic = p.items[side ? 7 : 5];
    const float *history = side ? vh : kh, *weight = cfg.conv[side];
    const bool rms = side || cfg.qClip <= 0;
    float values[4]{};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int c = rms ? lane * 4 + j : lane + j * 32;
        if (c >= cfg.dim) continue;
        const int col = head * cfg.dim + c;
        const float raw = x.read(token, col);
        float value = raw;
        for (int t = 0; t < cfg.taps; ++t) {
            const float a = t <= row ? x.read(token - t, col) :
                (history ? history[(cfg.taps - 1 + row - t) * width + col] : 0.f);
            const float w = dynamic.data ? weight[t * width + col] + dynamic.read(token, head * cfg.taps + t)
                                         : weight[col * cfg.taps + cfg.taps - 1 - t];
            value += a * w;
        }
        values[j] = value * rsqrtf(1.f + cfg.taps * (dynamic.data ? 2 : 1));
        const int dest = cfg.taps - 1 - length + row;
        if (dest >= 0) (side ? nextV : nextK)[dest * width + col] = raw;
    }
    const float factor = prefillScale(values, rms ? 0.f : cfg.qClip, cfg.dim, cfg.eps, rms);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int c = rms ? lane * 4 + j : lane + j * 32;
        if (c < cfg.dim) (side ? v : k)[size_t(token) * width + head * cfg.dim + c] = KV(values[j] * factor);
    }
}

template<class T> __global__ void gatePrefill(const float *x, ProjectionView<T> gate, T *out, int width) {
    const int row = blockIdx.x, col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col < width) {
        const size_t i = size_t(row) * width + col;
        out[i] = T(x[i] * float(1.0 / (1.0 + expf(-gate.read(row, col)))));
    }
}
