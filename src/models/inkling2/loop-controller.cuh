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

// A controller row fits in registers on the packed path. Pair carries both
// sums through the same block barrier. Blend retains the mixed FP32 channels
// until the scale is available, including for narrowed outputs.
template <bool Pair, class T, class Out, int Width>
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
                auto blend = [](float d, float v, float f, float k) {
                    // Keep the original FP32 rounding before narrowing.
                    return fmaf(1.f - tanhf(f), d, __fmul_rn(tanhf(k), v));
                };
                a[i] = make_float4(blend(x.x, m.x, forget.x, gain.x), blend(x.y, m.y, forget.y, gain.y),
                                   blend(x.z, m.z, forget.z, gain.z), blend(x.w, m.w, forget.w, gain.w));
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
