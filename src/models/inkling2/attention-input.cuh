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
    const __nv_bfloat16 *windowTable = nullptr;
    float *batchGates = nullptr, *batchSinks = nullptr;
    bool splitBias = false;
};

template<bool FullBatch = false>
__device__ float headScale(float v, float clip, int width, float eps, float *scratch) {
    float sum;
    if constexpr (FullBatch) {
        static_assert(threads == 128);
        // Match blockSum's 128-leaf tree with two block barriers. The final
        // five additions are local to the first warp.
        const int lane = threadIdx.x;
        scratch[lane] = v * v;
        __syncthreads();
        if (lane < 32) {
            sum = (scratch[lane] + scratch[lane + 64]) + (scratch[lane + 32] + scratch[lane + 96]);
            for (int d = 16; d; d /= 2) sum += __shfl_down_sync(0xffffffff, sum, d);
            if (!lane) scratch[0] = sum;
        }
        __syncthreads();
        sum = scratch[0];
    } else sum = blockSum(v * v, scratch);
    return clip > 0 ? fminf(1.f, clip / (sqrtf(sum) + 1e-6f)) : rsqrtf(sum / width + eps);
}

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

#include "window-verify-input.cuh"

template<class T, bool Timeline = false, bool PackQuery = false, class Act, bool Pdl = false, int BiasHeads = 1, bool Compact = false, bool OrdinaryBatch = false>
__global__ void prepareRelative(const DecodeVisit *visit, RelativeInputs<Act> in, int tiles, float *q, float *bias, int chain = 1) {
#if CUDART_VERSION >= 12080 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    if constexpr (Pdl) cuda_launch::DependencyStart();
#endif
    int row = 0, priorRows = 0;
    int task = blockIdx.x;
    if constexpr (Timeline || OrdinaryBatch) {
        row = blockIdx.y;
        if constexpr (Compact) {
            static_assert(BiasHeads == 16 && PackQuery);
            const int pairs = (chain + 1) / 2, local = (row % pairs) * 2;
            const int first = (row / pairs) * chain + local;
            const int rowTasks = (in.heads + 3) / 4 + (in.heads + 15) / 16 * tiles;
            if (task >= 2 * rowTasks) {
                prepareWindowKV<2, T>(visit, in, first, local, min(2, chain - local), task - 2 * rowTasks);
                return;
            }
            const int offset = task / rowTasks;
            if (local + offset >= chain) return;
            row = first + offset;
            task %= rowTasks;
        }
        priorRows = row % chain;
        visit += row;
        q += size_t(row) * in.heads * in.dim / (Compact ? 2 : 1);
        bias += size_t(row) * in.heads * (OrdinaryBatch && !in.splitBias ? in.rel : in.extent) / (Compact ? 2 : 1);
    }
    __shared__ float scratch[threads], relative[threads];
    const int col = threadIdx.x;
    constexpr int headsPerBlock = BiasHeads > 1 ? threads / 32 : 1;
    constexpr int channels = BiasHeads > 1 ? 4 : 1;
    const int queryTasks = (in.heads + headsPerBlock - 1) / headsPerBlock;
    const int warp = BiasHeads > 1 ? col / 32 : 0, lane = BiasHeads > 1 ? col % 32 : col;
    auto normalize = [&](float (&v)[channels], float clip) {
        if constexpr (channels == 4) return prefillScale(v, clip, 128, in.eps, false);
        else return headScale(v[0], clip, in.dim, in.eps, scratch);
    };
    if (task < queryTasks) {
        const int head = task * headsPerBlock + warp;
        if (head >= in.heads) return;
        float values[channels], weights[channels], scaled[channels];
#pragma unroll
        for (int j = 0; j < channels; ++j) {
            const int channel = lane + 32*j, i = head * in.dim + channel;
            values[j] = channel < in.dim ? in.sources.items[0].read(row, i) : 0.f;
            weights[j] = channel < in.dim ? in.qs[i] : 0.f;
            scaled[j] = in.qClip > 0 ? values[j] * weights[j] : values[j];
        }
        float scale = normalize(scaled, in.qClip);
#pragma unroll
        for (int j = 0; j < channels; ++j) {
            const int channel = lane + 32*j;
            if (channel >= in.dim) continue;
            const float value = values[j] * scale * weights[j];
            if constexpr (OrdinaryBatch) {
                if (in.batchGates) in.batchGates[(size_t(row) * in.heads + head) * in.dim + channel] =
                    in.sources.items[2].read(row, head * in.dim + channel);
                if (in.batchSinks && channel == 0) in.batchSinks[size_t(row) * in.heads + head] = in.sources.items[3].read(row, head);
            }
            if constexpr (Compact) {
                reinterpret_cast<__nv_bfloat16 *>(q)[head * in.dim + channel] = __nv_bfloat16(value);
            } else if constexpr (PackQuery) {
                half *packed = reinterpret_cast<half *>(q) + 2 * head * in.dim + channel;
                packed[0] = half(value);
                packed[in.dim] = half(value - float(packed[0]));
            } else q[head * in.dim + channel] = value;
        }
        return;
    }
    int idx = task - queryTasks;
    const int biasTasks = (in.heads + BiasHeads - 1) / BiasHeads * tiles;
    if (idx < biasTasks) {
        if (OrdinaryBatch && !in.splitBias) {
            const int h = idx * 4 + warp;
            if (h < in.heads) {
                const int ch = h * 16 + lane;
                float x = lane < 16 ? in.sources.items[1].read(row, ch) : 0.f;
                float w = lane < 16 ? in.rs[ch] : 0.f;
                float v = x * w;
                float sum = v * v;
                for (int step = 8; step; step /= 2) sum += __shfl_down_sync(0xffffffffu, sum, step);
                float scale = fminf(1.f, in.rClip / (sqrtf(__shfl_sync(0xffffffffu, sum, 0)) + 1.e-6f));
                if (lane < 16) bias[h * 16 + lane] = x * scale * w;
            }
            return;
        }
        if constexpr (Compact) {
            if (visit->verifyRemaining > 0)
                projectWindowBias(in, row, idx / tiles * BiasHeads, idx % tiles, visit->past,
                                  reinterpret_cast<__nv_bfloat16 *>(bias));
            return;
        }
        if (in.rel == 16 && in.extent % 4 == 0) {
            const int firstHead = idx / tiles * BiasHeads, lane = col % 32;
            const int dist = ((idx % tiles) * threads + col) * 4;
            if ((idx % tiles) * threads * 4 > visit->past) return;
            float rel[BiasHeads];
            float4 dots[BiasHeads]{}, totals[OrdinaryBatch ? BiasHeads : 1]{};
#pragma unroll
            for (int h = 0; h < BiasHeads; ++h) {
                const int channel = (firstHead + h) * in.rel + lane;
                const bool valid = lane < 16 && firstHead + h < in.heads;
                float x = valid ? in.sources.items[1].read(row, channel) : 0.f;
                float w = valid ? in.rs[channel] : 0.f;
                float scaled = in.rClip > 0 ? x * w : x;
                float sum = scaled * scaled;
#pragma unroll
                for (int d = 8; d; d /= 2) sum += __shfl_down_sync(0xffffffffu, sum, d);
                sum = __shfl_sync(0xffffffffu, sum, 0);
                float norm = in.rClip > 0 ? fminf(1.f, in.rClip / (sqrtf(sum) + 1.e-6f)) : rsqrtf(sum / 16 + in.eps);
                rel[h] = x * norm * w;
            }
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                float4 table = dist < in.extent ? *reinterpret_cast<const float4 *>(in.table + j * in.extent + dist) : float4{};
#pragma unroll
                for (int h = 0; h < BiasHeads; ++h) {
                    float w = __shfl_sync(0xffffffffu, rel[h], j);
                    if constexpr (OrdinaryBatch) if (!(j % 4)) dots[h] = float4{};
#if CUDART_VERSION >= 12080 && __CUDA_ARCH__ >= 1000
                    if constexpr (BiasHeads > 1) {
                        float2 a = __ffma2_rn({w, w}, {table.x, table.y}, {dots[h].x, dots[h].y});
                        float2 b = __ffma2_rn({w, w}, {table.z, table.w}, {dots[h].z, dots[h].w});
                        dots[h] = make_float4(a.x, a.y, b.x, b.y);
                    } else
#endif
                    {
                        dots[h].x += w * table.x; dots[h].y += w * table.y;
                        dots[h].z += w * table.z; dots[h].w += w * table.w;
                    }
                    if constexpr (OrdinaryBatch) {
                        // cuBLAS' 768 x 1024 x 16 SIMT kernel accumulates four groups of four FMAs.
                        if (j % 4 == 3) {
                            totals[h].x = __fadd_rn(totals[h].x,dots[h].x); totals[h].y = __fadd_rn(totals[h].y,dots[h].y);
                            totals[h].z = __fadd_rn(totals[h].z,dots[h].z); totals[h].w = __fadd_rn(totals[h].w,dots[h].w);
                        }
                    }
                }
            }
#pragma unroll
            for (int h = 0; h < BiasHeads; ++h)
                if (firstHead + h < in.heads && dist < in.extent)
                    *reinterpret_cast<float4 *>(bias + (firstHead + h) * in.extent + dist) = OrdinaryBatch ? totals[h] : dots[h];
            return;
        }
        const int head = idx / tiles, dist = idx % tiles * threads + col;
        if (idx % tiles * threads > visit->past) return;
        int i = head * in.rel + col;
        float x = col < in.rel ? in.sources.items[1].read(row, i) : 0.f;
        float w = col < in.rel ? in.rs[i] : 0.f;
        float scale = headScale<(BiasHeads > 1)>(in.rClip > 0 ? x*w : x, in.rClip, in.rel, in.eps, scratch);
        relative[col] = x * scale * w;
        __syncthreads();
        if (dist < in.extent && dist <= visit->past) {
            float sum = 0;
            for (int j = 0; j < in.rel; ++j) sum += relative[j] * in.table[j * in.extent + dist];
            bias[head * in.extent + dist] = sum;
        }
        return;
    }
    idx = (idx - biasTasks) * headsPerBlock + warp;
    if (idx >= 2 * in.kvHeads) return;
    const int which = idx / in.kvHeads, base = (idx % in.kvHeads) * in.dim;
    const int width = in.kvHeads * in.dim;
    const auto input = in.sources.items[which ? 6 : 4], dynamic = in.sources.items[which ? 7 : 5];
    float values[channels];
#pragma unroll
    for (int j = 0; j < channels; ++j) {
        const int channel = lane + 32*j;
        values[j] = channel < in.dim ? updateConvChannel<Timeline>(input.data + size_t(row) * input.stride, visit, which,
            which ? in.vw : in.kw, dynamic.data ? dynamic.data + size_t(row) * dynamic.stride : nullptr,
            width, in.taps, in.kvHeads, base + channel, priorRows, input.stride) : 0.f;
    }
    float scale;
    if constexpr (OrdinaryBatch) {
        // The unfused value projection uses a warp RMS reduction over four channels per lane.
        if (which) {
            __shared__ float valuesByChannel[4][128];
#pragma unroll
            for (int j = 0; j < 4; ++j) valuesByChannel[warp][32*j+lane] = values[j];
            __syncwarp();
            float consecutive[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) consecutive[j] = valuesByChannel[warp][4*lane+j];
            scale = prefillScale(consecutive, 0.f, 128, in.eps, true);
        } else scale = prefillScale(values, in.qClip, 128, in.eps, false);
    } else scale = normalize(values, which ? 0.f : in.qClip);
    if (!Timeline || visit->verifyRemaining > 0) {
        auto dst = static_cast<T *>(which ? visit->pages.value : visit->pages.key) + visit->pages.offset(visit->past, which, width);
#pragma unroll
        for (int j = 0; j < channels; ++j)
            if (lane + 32*j < in.dim) dst[base + lane + 32*j] = T(values[j] * scale);
    }
}

struct PrefillPreparation {
    const float *qs, *rs, *conv[2];
    int heads, kvHeads, dim, rel, taps;
    float qClip, rClip, eps;
};

template<bool Compact, class T, class KV, class R = float> __global__ void preparePrefill(PrefillParts<T> p, PrefillPreparation cfg,
        int offset, int length, const float *kh, const float *vh, float *nextK, float *nextV,
        PrefillQuery<KV> *q, R *r, float *sink, KV *k, KV *v) {
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
                else if constexpr (Compact) q[base + col] = PrefillQuery<KV>(value);
                else {
                    const PrefillQuery<KV> high(value);
                    q[2 * base + col] = high;
                    q[2 * base + width + col] = PrefillQuery<KV>(value - float(high));
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
