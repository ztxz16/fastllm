// Controller fusion follows vLLM Inkling ops/loop.py (Apache-2.0).
// The loop controller consumes two independently normalized rows concatenated
// along their channels. Produce the Linear input directly, including for prefill.
struct LoopStream {
    const float *dest, *msg, *norm;
};
struct LoopStreams {
    LoopStream items[3]; // query, key, value (or a single MLP stream)
};
template <typename T> __global__ void PairRMSNorm(LoopStreams streams, T *packed, int rows, int width, float eps) {
    int packedRow = blockIdx.x / 2;
    const LoopStream &s = streams.items[packedRow / rows];
    const float *x = (blockIdx.x % 2 ? s.msg : s.dest) + size_t(packedRow % rows) * width;
    __shared__ float scratch[16];
    float sq = 0.f;
    if (width % 4 == 0) {
        for (int j = threadIdx.x; j < width / 4; j += blockDim.x)
            sq += cuda_attn_res::Square4(reinterpret_cast<const float4 *>(x)[j]);
    } else {
        for (int j = threadIdx.x; j < width; j += blockDim.x) sq += x[j] * x[j];
    }
    float inv = cuda_attn_res::RowRmsScale(sq, width, eps, scratch);
    for (int j = threadIdx.x; j < width; j += blockDim.x)
        packed[size_t(blockIdx.x) * width + j] = x[j] * inv;
}

template <class T, class Out> __global__ void BlendRMSNorm(LoopStreams streams, const float *gates, Out *out, int rows, int width, float eps) {
    const LoopStream &s = streams.items[blockIdx.x / rows];
    const size_t offset = size_t(blockIdx.x % rows) * width;
    out += size_t(blockIdx.x) * width;
    __shared__ float scratch[16];
    extern __shared__ __align__(16) float temp[];
    float *mixed = temp;
    if (size_t(width) * sizeof(float) > 16384) {
        if constexpr (std::is_same_v<Out, float>) mixed = out;
        else mixed = nullptr;
    }
    float sq = 0.f;
    gates += size_t(blockIdx.x) * 2 * width;
    auto blend = [&](int j) {
        size_t i = offset + j;
        return (1.f - tanhf(float(gates[width + j]))) * s.dest[i] + tanhf(float(gates[j])) * s.msg[i];
    };
    if (width % 4 == 0) {
        for (int j = threadIdx.x; j < width / 4; j += blockDim.x) {
            float4 v = make_float4(blend(4*j), blend(4*j+1), blend(4*j+2), blend(4*j+3));
            if (mixed) reinterpret_cast<float4 *>(mixed)[j] = v;
            sq += cuda_attn_res::Square4(v);
        }
    } else {
        for (int j = threadIdx.x; j < width; j += blockDim.x) {
            float v = blend(j);
            if (mixed) mixed[j] = v;
            sq += v*v;
        }
    }
    float inv = cuda_attn_res::RowRmsScale(sq, width, eps, scratch);
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        float v = mixed ? mixed[j] : blend(j);
        out[j] = Out(T(v * inv * s.norm[j]));
    }
}

// Two rows and two output channels per block. Pair normalization packs the
// eight successive inputs consumed by each lane into one aligned vector.
__global__ void controllerDown(const __nv_bfloat16 *input, const __nv_bfloat16 *weight, __nv_bfloat16 *output,
                               int rows, int columns) {
    constexpr int width = 4608;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row = blockIdx.y * 2 + warp / 2, col = blockIdx.x * 2 + warp % 2;
    if (row >= rows) return;
    const size_t index = size_t(blockIdx.z) * rows + row;
    input += index * width;
    weight += (size_t(blockIdx.z) * columns + col) * width;
    float sum = 0.f;
#pragma unroll
    for (int base = 0; base < width; base += 256) {
        flashinfer::vec_t<__nv_bfloat16, 8> values;
        values.load(input + base + lane * 8);
#pragma unroll
        for (int j = 0; j < 8; ++j)
            sum = fmaf(float(values[j]), float(weight[base + j * 32 + lane]), sum);
    }
#pragma unroll
    for (int step = 16; step; step /= 2) sum += __shfl_down_sync(0xffffffff, sum, step);
    if (!lane) output[index * columns + col] = __float2bfloat16_rn(sum);
}

__device__ __forceinline__ float controllerMix(float dest, float msg, float gain, float forget) {
    return fmaf(1.f - tanhf(forget), dest, __fmul_rn(tanhf(gain), msg));
}

// The three-matrix singleton controller uses BF16 tensor cores. Keep its
// K=16 accumulation boundaries and reuse each weight tile across queries.
__global__ void controllerQkvGates(const __nv_bfloat16 *hidden,
        const __nv_bfloat16 *weight, float *out, int rows, int width) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int rank = 64;
    const int lane = threadIdx.x % 32, first = blockIdx.y * 8;
    const int channel = blockIdx.x * 64 + (threadIdx.x / 32) * 16 + lane / 4;
    const int pair = (lane % 4) * 2, stream = blockIdx.z;
    const auto *input = hidden + (size_t(stream) * rows + min(first + lane / 4, rows - 1)) * rank + pair;
    weight += (size_t(stream) * 2 * width + channel) * rank + pair;
    float gates[2][4]{};
    auto load = [](const __nv_bfloat16 *p) { return *reinterpret_cast<const unsigned *>(p); };
#pragma unroll
    for (int k = 0; k < rank; k += 16) {
        const unsigned b0 = load(input + k), b1 = load(input + k + 8);
#pragma unroll
        for (int side = 0; side < 2; ++side) {
            const auto *w = weight + side * width * rank + k;
            auto &v = gates[side];
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                : "+f"(v[0]), "+f"(v[1]), "+f"(v[2]), "+f"(v[3])
                : "r"(load(w)), "r"(load(w + 8 * rank)), "r"(load(w + 8)),
                  "r"(load(w + 8 * rank + 8)), "r"(b0), "r"(b1));
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = first + pair + i % 2;
        if (row >= rows) continue;
        const size_t offset = (size_t(stream) * rows + row) * (2 * width) + channel + (i / 2) * 8;
        out[offset] = gates[0][i];
        out[offset + width] = gates[1][i];
    }
#endif
}

// The singleton BF16 GEMV reduces eight contiguous segments of the rank dimension.
// Reuse each pair of weight vectors across verification rows. Keep both gates
// in FP32 through the shared residual blend and RMS normalization.
template<int Tile>
__global__ void controllerMlpGates(const __nv_bfloat16 *hidden,
        const __nv_bfloat16 *weight, float *out, int rows, int width) {
    constexpr int rank = 64, lanes = 8;
    const int lane = threadIdx.x % lanes, channel = (blockIdx.x * blockDim.x + threadIdx.x) / lanes;
    const int first = blockIdx.y * Tile;
    if (channel >= width) return;
    float gain[Tile]{}, forget[Tile]{};
    using Pack = cuda_attn_res::Channels4<__nv_bfloat16>;
#pragma unroll
    for (int j = 0; j < rank / lanes; j += 4) {
        const int col = lane * (rank / lanes) + j;
        const auto a = *reinterpret_cast<const Pack *>(weight + size_t(channel) * rank + col);
        const auto b = *reinterpret_cast<const Pack *>(weight + size_t(width + channel) * rank + col);
#pragma unroll
        for (int r = 0; r < Tile; ++r) {
            if (first + r >= rows) break;
            const auto x = *reinterpret_cast<const Pack *>(hidden + size_t(first + r) * rank + col);
            auto dot = [&](float sum, Pack v) {
                sum = fmaf(float(x.x), float(v.x), sum);
                sum = fmaf(float(x.y), float(v.y), sum);
                sum = fmaf(float(x.z), float(v.z), sum);
                return fmaf(float(x.w), float(v.w), sum);
            };
            gain[r] = dot(gain[r], a);
            forget[r] = dot(forget[r], b);
        }
    }
    const unsigned mask = __activemask();
#pragma unroll
    for (int r = 0; r < Tile; ++r) {
        if (first + r >= rows) break;
#pragma unroll
        for (int step = lanes / 2; step; step /= 2) {
            gain[r] += __shfl_down_sync(mask, gain[r], step, lanes);
            forget[r] += __shfl_down_sync(mask, forget[r], step, lanes);
        }
        if (!lane) {
            const size_t i = size_t(first + r) * (2 * width) + channel;
            out[i] = gain[r];
            out[i + width] = forget[r];
        }
    }
}

// A controller row fits in registers on the packed path. Pair carries both
// sums through the same block barrier. Blend retains the mixed FP32 channels
// until the scale is available, including for narrowed outputs.
template <bool Pair, class T, class Out, int Width, bool PackDown = false>
__global__ void ControllerNormRows(FASTLLM_ROW_PARAMETER LoopStreams streams, const float *gates, Out *out, float eps) {
    constexpr int threads = 512, cols = Width / 4;
    constexpr int packs = (cols + threads - 1) / threads;
    const LoopStream &stream = streams.items[blockIdx.y];
    const size_t row = size_t(blockIdx.y) * gridDim.x + blockIdx.x;
    const size_t offset = size_t(blockIdx.x) * cols;
    const auto dest = reinterpret_cast<const float4 *>(stream.dest) + offset;
    const auto msg = reinterpret_cast<const float4 *>(stream.msg) + offset;
    float4 a[packs], b[packs];
    float sa = 0.f, sb = 0.f;
#pragma unroll
    for (int i = 0; i < packs; ++i) {
        const int j = threadIdx.x + i * threads;
        if (j < cols) {
            float4 x = dest[j], m = msg[j];
            if constexpr (Pair) {
                a[i] = x;
                b[i] = m;
                sb += cuda_attn_res::Square4(m);
            } else {
                auto g = reinterpret_cast<const float4 *>(gates) + row * 2 * cols + j;
                float4 gain = g[0], forget = g[cols];
                a[i] = make_float4(controllerMix(x.x, m.x, gain.x, forget.x), controllerMix(x.y, m.y, gain.y, forget.y),
                                   controllerMix(x.z, m.z, gain.z, forget.z), controllerMix(x.w, m.w, gain.w, forget.w));
            }
            sa += cuda_attn_res::Square4(a[i]);
        }
    }
    constexpr int warps = threads / 32;
    __shared__ float partial[Pair ? 2 : 1][warps];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    sa = cuda::WarpSum(sa);
    if constexpr (Pair) sb = cuda::WarpSum(sb);
    if (lane == 0) {
        partial[0][warp] = sa;
        if constexpr (Pair) partial[1][warp] = sb;
    }
    __syncthreads();
    auto scale = [&](int which) {
        return rsqrtf(cuda::WarpSum(lane < warps ? partial[which][lane] : 0.f) / Width + eps);
    };
    sa = scale(0);
    if constexpr (Pair) sb = scale(1);
    if constexpr (PackDown) {
        static_assert(Pair && Width % 256 == 0 && std::is_same_v<Out, __nv_bfloat16>);
        // Preserve the reference warp sums above. Adjacent lanes read adjacent
        // source channels, then store eight consecutive dot-product terms per lane.
        for (int v = threadIdx.x; v < 2 * Width / 8; v += threads) {
            const bool message = v >= Width / 8;
            const int slot = v % (Width / 8), start = slot / 32 * 256 + slot % 32;
            const auto *source = (message ? stream.msg : stream.dest) + size_t(blockIdx.x) * Width + start;
            flashinfer::vec_t<Out, 8> values;
#pragma unroll
            for (int j = 0; j < 8; ++j) values[j] = Out(source[32 * j] * (message ? sb : sa));
            values.store(out + row * 2 * Width + v * 8);
        }
        return;
    }
    auto y = reinterpret_cast<cuda_attn_res::Channels4<Out> *>(out) + row * (Pair ? 2 : 1) * cols;
#pragma unroll
    for (int i = 0; i < packs; ++i) {
        const int j = threadIdx.x + i * threads;
        if (j < cols) {
            float4 v = a[i];
            if constexpr (Pair) {
                y[j] = {Out(v.x * sa), Out(v.y * sa), Out(v.z * sa), Out(v.w * sa)};
                v = b[i];
                y[cols + j] = {Out(v.x * sb), Out(v.y * sb), Out(v.z * sb), Out(v.w * sb)};
            } else {
                float4 w = reinterpret_cast<const float4 *>(stream.norm)[j];
                y[j] = {Out(T(v.x * sa * w.x)), Out(T(v.y * sa * w.y)),
                        Out(T(v.z * sa * w.z)), Out(T(v.w * sa * w.w))};
            }
        }
    }
}
