// The fixed rank-16 table is shared by all requests. Keep adjacent
// rank coefficients packed for a single Tensor Core operand load.
__global__ void packWindowTable(const float *source, __nv_bfloat16 *out, int extent) {
    const int i = int(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < 16 * extent) out[i] = __nv_bfloat16(source[(i % 16) * extent + i / 16]);
}

// One warp projects 16 heads. Store column key % extent so the attention
// kernel reads consecutive BF16 pairs even when the sliding window moves.
template<class Act>
__device__ __forceinline__ void projectWindowBias(const RelativeInputs<Act> &in, int row, int headBase,
                                                   int tile, int position, __nv_bfloat16 *out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(CUDA_NO_TENSOR_CORE)
    const int lane = threadIdx.x % 32;
    const int k = 2 * (lane % 4);
    unsigned a[4];
#pragma unroll
    for (int halfRow = 0; halfRow < 2; ++halfRow) {
        const int h = headBase + lane / 4 + 8 * halfRow;
        float val[4], w[4], ss = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int ch = h * 16 + k + 8 * (i / 2) + i % 2;
            val[i] = h < in.heads ? in.sources.items[1].read(row, ch) : 0.f;
            w[i] = h < in.heads ? in.rs[ch] : 0.f;
            float x = in.rClip > 0 ? val[i] * w[i] : val[i];
            ss += x * x;
        }
        ss += __shfl_xor_sync(0xffffffff, ss, 1);
        ss += __shfl_xor_sync(0xffffffff, ss, 2);
        float scale = in.rClip > 0 ? fminf(1.f, in.rClip / (sqrtf(ss) + 1.e-6f)) : rsqrtf(ss / 16 + in.eps);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            __nv_bfloat162 pair = __floats2bfloat162_rn(val[2*i] * scale * w[2*i], val[2*i+1] * scale * w[2*i+1]);
            a[halfRow+2*i] = reinterpret_cast<const unsigned &>(pair);
        }
    }
    const int base = tile * 256 + threadIdx.x / 32 * 64;
#pragma unroll
    for (int n = 0; n < 64; n += 16) {
        if (base + n >= in.extent) break;
        unsigned b[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int key = base + n + lane / 4 + 8 * (j / 2);
            const int dist = (position - key) & (in.extent - 1), kk = k + 8 * (j % 2);
            b[j] = *reinterpret_cast<const unsigned *>(in.windowTable + dist * 16 + kk);
        }
        float scores[8];
        flashinfer::mma::mma_sync_m16n16k16_row_col_f16f16f32<__nv_bfloat16, flashinfer::mma::MMAMode::kInit>(scores, a, b);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int head = headBase + lane / 4 + 8 * (j % 2);
            const int key = base + n + 2 * (lane % 4) + 8 * (j / 2);
            if (head < in.heads)
                *reinterpret_cast<__nv_bfloat162 *>(out + head * in.extent + key) =
                    __floats2bfloat162_rn(scores[2*j], scores[2*j+1]);
        }
    }
#endif
}

// The static four-tap convolution shares weights and preceding inputs across
// adjacent verification positions. Each visit still owns its saved history.
template<int Rows, class KV, class Act>
__device__ void prepareWindowKV(const DecodeVisit *visits, const RelativeInputs<Act> &in,
                                int first, int local, int count, int task) {
    const int lane = threadIdx.x % 32;
    const int idx = task * (threads / 32) + threadIdx.x / 32;
    if (idx >= 2 * in.kvHeads) return;
    const bool value = idx >= in.kvHeads;
    const int base = (idx % in.kvHeads) * 128, width = in.kvHeads * 128;
    const auto input = in.sources.items[value ? 6 : 4];
    const float *weights = value ? in.vw : in.kw;
    float convolved[Rows][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int channel = base + 32 * j + lane;
        const float4 w = reinterpret_cast<const float4 *>(weights)[channel];
        float previous[3];
#pragma unroll
        for (int t = 1; t <= 3; ++t)
            previous[t - 1] = t <= local ? input.read(first - t, channel) :
                visits[first].history[value][(3 - t) * width + channel];
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            if (r >= count) break;
            const float x = input.read(first + r, channel);
            float v = x + x * w.w;
            v += previous[0] * w.z;
            v += previous[1] * w.y;
            v += previous[2] * w.x;
            convolved[r][j] = v * rsqrtf(1.f + in.taps);
            visits[first + r].history[value][3 * width + channel] = x;
            previous[2] = previous[1];
            previous[1] = previous[0];
            previous[0] = x;
        }
    }
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
        if (r >= count) break;
        const auto &visit = visits[first + r];
        if (visit.verifyRemaining <= 0) continue;
        const float scale = prefillScale(convolved[r], value ? 0.f : in.qClip, 128, in.eps, false);
        auto dst = visit.pages.row<KV>(visit.past, value, width) + base;
#pragma unroll
        for (int j = 0; j < 4; ++j) dst[lane + 32 * j] = KV(convolved[r][j] * scale);
    }
}
