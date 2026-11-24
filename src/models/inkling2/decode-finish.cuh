// Singleton decode epilogues. The residual and convolution histories remain in the
// same graph-owned locations as the unfused path. Fusion boundaries also used by
// Basalt's Inkling post_attn and moe_finish (MIT, third_party/basalt/LICENSE).
// https://github.com/jesdga95/basalt/blob/v0.5.6/src/models/inkling2/kernels/glue.cu
// The last three inputs are laid out oldest to newest. Preload them and the
// immutable coefficients while the preceding projection is still running.
struct DecodeConv4 {
    float4 weight;
    float oldest, middle, newest;

    __device__ void load(const float *weights, const float *history, int width, int c) {
        weight = reinterpret_cast<const float4 *>(weights)[c];
        oldest = history[c];
        middle = history[width + c];
        newest = history[2 * width + c];
    }
    __device__ float apply(float x, float *history, int width, int c, float scale) const {
        const float v = fmaf(oldest, weight.x, fmaf(middle, weight.y,
                        fmaf(newest, weight.z, fmaf(x, weight.w, x))));
        history[c] = middle;
        history[width + c] = newest;
        history[2 * width + c] = x;
        return v * scale;
    }
};

template<int Slots, class T, class Round, bool Timeline, class R>
__global__ void postAttentionMix(const T *projected, const DecodeVisit *visit, const float *weight, int taps, int width,
                                 const R *skip, R *residual, FASTLLM_ROW_PARAMETER cuda_attn_res::SourceList<R> sources,
                                 cuda_attn_res::Query query, float eps, bool normalizeValues, float *mixed, int chain) {
    constexpr int cols = 2304, threads = 512, perLane = (cols + threads - 1) / threads;
    __shared__ __align__(16) R latest[cols];
    const int row = blockIdx.x;
    visit += row;
    const size_t offset = size_t(row) * width;
    projected += offset;
    residual += offset;
    if (skip) skip += offset;
    float *history = visit->history[2];
    DecodeConv4 state[perLane];
    const float scale = rsqrtf(1.f + taps);
#pragma unroll
    for (int i = 0; i < perLane; ++i) {
        const int c = threadIdx.x + i * threads;
        if constexpr (!Timeline) if (c < cols) state[i].load(weight, history, cols, c);
    }
    cuda_launch::DependencyStart();
#pragma unroll
    for (int i = 0; i < perLane; ++i) {
        const int c = threadIdx.x + i * threads;
        if (c >= cols) break;
        float y;
        if constexpr (Timeline)
            y = updateConvChannel<true>(projected, visit, 2, weight, static_cast<const float *>(nullptr),
                                         cols, taps, 1, c, row % chain, cols);
        else
            y = state[i].apply(float(projected[c]), history, cols, c, scale);
        latest[c] = residual[c] = skip ? __fadd_rn(y, float(skip[c])) : y;
    }
    __syncthreads();
    // Preserve the unfused device rsqrt(width). Folding rsqrt(2304) at
    // compile time differs by one ULP and can cross an expert quantization boundary.
    cuda_attn_res::AttnResRegisterBody<Slots, Round, float, cols, (Slots > 4), void>(
        sources, query, 0, row, gridDim.x, width, eps, normalizeValues, mixed, latest);
}

__device__ __forceinline__ float sumRouteChannel(const __nv_bfloat16 *routes, int top, int width, int channel) {
    float sum = 0.f;
    for (int r = 0; r < top; ++r) sum += float(routes[size_t(r) * width + channel]);
    return sum;
}

template<class R> __global__ void finishExpertRoutes(const __nv_bfloat16 *routes, int top, const DecodeVisit *visit,
                                   const float *weight, int width, int taps, R *residual) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    DecodeConv4 state;
    float *history = visit->history[3];
    if (c < width) state.load(weight, history, width, c);
    cuda_launch::DependencyStart();
    if (c >= width) return;
    const float sum = sumRouteChannel(routes, top, width, c);
    const float y = state.apply(sum, history, width, c, rsqrtf(1.f + taps));
    residual[c] = __fadd_rn(y, residual[c]);
}

constexpr int verifyRouteRows = 8;
constexpr int verifyRouteChannels = 32;

// Each warp reduces one row. Keep the convolution inputs in shared memory so
// neighbouring rows never depend on history written by another CTA.
template<class R> __global__ void finishVerifyRoutes(const __nv_bfloat16 *routes, int top, const DecodeVisit *visits,
                                   const float *weight, int width, int taps, int chain, R *residual) {
    constexpr int depth = 3;
    __shared__ float sums[depth + verifyRouteRows][verifyRouteChannels];
    const int lane = threadIdx.x, warp = threadIdx.y;
    const int channel = blockIdx.x * verifyRouteChannels + lane;
    const int request = blockIdx.z * chain;
    const int tile = blockIdx.y * verifyRouteRows;
    const int row = tile + warp;
    const size_t stride = size_t(top) * width;
    cuda_launch::DependencyStart();
    if (channel < width) {
        const float *history = visits[request].history[3];
        if (warp < depth) {
            // Later tiles reduce just the three preceding rows from this request.
            sums[warp][lane] = tile == 0 ? history[warp * width + channel]
                : sumRouteChannel(routes + size_t(request + tile - depth + warp) * stride, top, width, channel);
        }
        if (row < chain)
            sums[depth + warp][lane] = sumRouteChannel(routes + size_t(request + row) * stride, top, width, channel);
    }
    __syncthreads();
    if (channel >= width || row >= chain) return;

    const float sum = sums[depth + warp][lane];
    const float4 w = reinterpret_cast<const float4 *>(weight)[channel];
    const float y = fmaf(sums[warp][lane], w.x, fmaf(sums[warp + 1][lane], w.y,
                    fmaf(sums[warp + 2][lane], w.z, fmaf(sum, w.w, sum))));
    visits[request + row].history[3][depth * width + channel] = sum;
    R &out = residual[size_t(request + row) * width + channel];
    out = __fadd_rn(float(out), __fmul_rn(y, rsqrtf(1.f + taps)));
}
