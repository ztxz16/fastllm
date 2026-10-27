// Included inside deepseekv41-kernels.cu's anonymous namespace, after the
// scalar KV decoders. Keep the unpacked FP32 values: rounding them to BF16
// here would change the scalar attention's arithmetic for packed KV rows.
struct V41Sm75QuantizedAttention {
    const __nv_bfloat16 *chunk, *ring, *compressed;
    const uint8_t *packedRing, *packedCompressed;
    const int32_t *indices;
    // windowCount is the maximum window length in this verification chunk.
    // Shorter early-token windows leave invalid padding before compressed KV.
    int window, start, capacity, windowCount, topWidth;
    int ringMode, ringRowBytes, compressedMode, compressedRowBytes;
};

template<bool MultiToken>
__global__ void V41Sm75QuantizedAttentionGather(V41Sm75QuantizedAttention p,
                                              float *rows, int32_t *valid) {
    const int token = MultiToken ? blockIdx.z : 0;
    const int count = p.windowCount + p.topWidth;
    const int windowCount = MultiToken ? min(p.window, p.start + token + 1) : p.windowCount;
    rows += size_t(token) * count * 512;
    valid += size_t(token) * count;
    const int c = blockIdx.x;
    const __nv_bfloat16 *row = nullptr;
    const uint8_t *packed = nullptr;
    int mode = 0;
    if (c < windowCount) {
        const int position = p.start + token - windowCount + 1 + c;
        // Earlier rows from this verification chunk remain unquantized and
        // are visible only up to this token, even across a ring-buffer wrap.
        if (position >= p.start) row = p.chunk + size_t(position - p.start) * 512;
        else if (p.packedRing) {
            packed = p.packedRing + size_t(position % p.window) * p.ringRowBytes;
            mode = p.ringMode;
        } else row = p.ring + size_t(position % p.window) * 512;
    } else if (c >= p.windowCount) {
        const int index = p.indices[size_t(token) * p.topWidth + c - p.windowCount];
        if (index >= 0 && index < p.capacity) {
            if (p.packedCompressed) {
                packed = p.packedCompressed + size_t(index) * p.compressedRowBytes;
                mode = p.compressedMode;
            } else row = p.compressed + size_t(index) * 512;
        }
    }
    const bool present = row || packed;
    if (threadIdx.x == 0) valid[c] = present;
    if (!present) return;
    for (int d = threadIdx.x; d < 512; d += blockDim.x)
        rows[size_t(c) * 512 + d] = packed ? V41LoadKvRow(packed, 512, d, mode)
                                         : __bfloat162float(row[d]);
}

// Four warps x four heads per CTA. Verification reuses each query for two
// candidates, retaining the scalar kernel's 64 FMAs and eight-lane reduction.
template<bool MultiToken, int CandidatesPerWarp = 1>
__global__ void __launch_bounds__(128)
V41Sm75QuantizedAttentionScores(const __nv_bfloat16 *q, const float *rows,
                                const int32_t *valid, int count, int heads,
                                float *scores) {
    const int token = MultiToken ? blockIdx.z : 0;
    q += size_t(token) * heads * 512;
    rows += size_t(token) * count * 512;
    valid += size_t(token) * count;
    scores += size_t(token) * heads * count;
    // Four padding floats keep each lane's 64-element segment float4 aligned.
    __shared__ __align__(16) float keys[4 * CandidatesPerWarp][544], queries[4][544];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int first = (blockIdx.x * 4 + warp) * CandidatesPerWarp;
    const int h = blockIdx.y * 4 + lane / 8;
    bool present[CandidatesPerWarp];
    #pragma unroll
    for (int j = 0; j < CandidatesPerWarp; ++j) {
        const int c = first + j;
        present[j] = c < count && valid[c];
        if (present[j]) {
            for (int d = lane; d < 512; d += 32)
                keys[warp * CandidatesPerWarp + j][d + (d / 64) * 4] = rows[size_t(c) * 512 + d];
        }
    }
    for (int i = threadIdx.x; i < 4 * 512; i += 128) {
        const int head = blockIdx.y * 4 + i / 512, d = i % 512;
        queries[i / 512][d + (d / 64) * 4] = head < heads
            ? __bfloat162float(q[size_t(head) * 512 + d]) : 0.f;
    }
    __syncthreads();
    const int segment = (lane % 8) * 68;
    float dot[CandidatesPerWarp] = {};
    #pragma unroll
    for (int d = 0; d < 64; d += 4) {
        const float4 a = *reinterpret_cast<const float4 *>(&queries[lane / 8][segment + d]);
        #pragma unroll
        for (int j = 0; j < CandidatesPerWarp; ++j) if (present[j]) {
            const float4 b = *reinterpret_cast<const float4 *>(&keys[warp * CandidatesPerWarp + j][segment + d]);
            dot[j] += a.x * b.x;
            dot[j] += a.y * b.y;
            dot[j] += a.z * b.z;
            dot[j] += a.w * b.w;
        }
    }
    #pragma unroll
    for (int j = 0; j < CandidatesPerWarp; ++j) {
        dot[j] += __shfl_xor_sync(0xffffffff, dot[j], 1);
        dot[j] += __shfl_xor_sync(0xffffffff, dot[j], 2);
        dot[j] += __shfl_xor_sync(0xffffffff, dot[j], 4);
        if (present[j] && lane % 8 == 0 && h < heads)
            scores[size_t(h) * count + first + j] = dot[j];
    }
}

// Compute the online-softmax coefficients once per head/token, rather than
// repeating exp/max/sum for all 512 value dimensions. Prefix maxima are exact;
// the denominator still follows the scalar kernel's ordered FP32 recurrence.
__global__ void __launch_bounds__(32)
V41Sm75QuantizedAttentionCoefficients(const int32_t *valid, const float *scores,
                                     const float *sink, int count, int heads, float scale,
                                     float2 *coefficients, float *inverseSums) {
    const int token = blockIdx.y;
    const int head = blockIdx.x, lane = threadIdx.x;
    const size_t offset = (size_t(token) * heads + head) * count;
    valid += size_t(token) * count;
    scores += offset;
    coefficients += offset;
    float mx = sink[head], sum = 1.f;
    for (int base = 0; base < count; base += 32) {
        const int c = base + lane;
        const bool present = c < count && valid[c];
        const unsigned mask = __ballot_sync(0xffffffff, present);
        const float score = present ? scores[c] * scale : -INFINITY;
        float inclusive = fmaxf(mx, score);
        #pragma unroll
        for (int delta = 1; delta < 32; delta *= 2) {
            const float previous = __shfl_up_sync(0xffffffff, inclusive, delta);
            if (lane >= delta) inclusive = fmaxf(inclusive, previous);
        }
        float previous = __shfl_up_sync(0xffffffff, inclusive, 1);
        if (lane == 0) previous = mx;
        const float alpha = present ? __expf(previous - inclusive) : 1.f;
        const float prob = present ? __expf(score - inclusive) : 0.f;
        if (present) coefficients[c] = make_float2(alpha, prob);
        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            const float a = __shfl_sync(0xffffffff, alpha, j);
            const float p = __shfl_sync(0xffffffff, prob, j);
            if (lane == 0 && (mask & (1u << j))) sum = sum * a + p;
        }
        mx = __shfl_sync(0xffffffff, inclusive, 31);
    }
    if (lane == 0) inverseSums[size_t(token) * heads + head] = 1.f / sum;
}

// Independent output dimensions can run in parallel without changing the
// online-softmax or PV accumulation order, including skipped/duplicate indices.
template<bool MultiToken, bool ReuseCoefficients = false, int HeadsPerBlock = 1,
         int CandidateTile = HeadsPerBlock == 1 ? 8 : 4>
__global__ void __launch_bounds__(64)
V41Sm75QuantizedAttentionValues(const float *rows, const int32_t *valid,
                                const float *scores, const float *sink,
                                int count, int heads, float scale, __nv_bfloat16 *output,
                                const float2 *coefficients = nullptr, const float *inverseSums = nullptr) {
    const int token = MultiToken ? blockIdx.z : 0;
    rows += size_t(token) * count * 512;
    valid += size_t(token) * count;
    scores += size_t(token) * heads * count;
    output += size_t(token) * heads * 512;
    static_assert(ReuseCoefficients || HeadsPerBlock == 1, "head reuse needs shared softmax coefficients");
    const int head = blockIdx.x * HeadsPerBlock, d = blockIdx.y * 64 + threadIdx.x;
    if constexpr (ReuseCoefficients) coefficients += (size_t(token) * heads + head) * count;
    // Multiple heads reuse each FP32 KV load during verification. Their
    // independent accumulators keep the original candidate/rounding order.
    constexpr int Tile = CandidateTile;
    float acc[HeadsPerBlock] = {}, mx = sink[head], sum = 1.f;
    for (int base = 0; base < count; base += Tile) {
        float sc[Tile], values[Tile];
        float2 coeff[HeadsPerBlock][Tile];
        bool present[Tile];
        #pragma unroll
        for (int j = 0; j < Tile; ++j) {
            const int c = base + j;
            present[j] = c < count && valid[c];
            if (present[j]) {
                if constexpr (ReuseCoefficients) {
                    #pragma unroll
                    for (int h = 0; h < HeadsPerBlock; ++h) if (head + h < heads)
                        coeff[h][j] = coefficients[size_t(h) * count + c];
                } else sc[j] = scores[size_t(head) * count + c];
                values[j] = rows[size_t(c) * 512 + d];
            }
        }
        #pragma unroll
        for (int j = 0; j < Tile; ++j) if (present[j]) {
            if constexpr (ReuseCoefficients) {
                #pragma unroll
                for (int h = 0; h < HeadsPerBlock; ++h) if (head + h < heads)
                    acc[h] = acc[h] * coeff[h][j].x + coeff[h][j].y * values[j];
            } else {
                const float s = sc[j] * scale;
                const float next = fmaxf(mx, s);
                const float alpha = __expf(mx - next), prob = __expf(s - next);
                sum = sum * alpha + prob;
                acc[0] = acc[0] * alpha + prob * values[j];
                mx = next;
            }
        }
    }
    #pragma unroll
    for (int h = 0; h < HeadsPerBlock; ++h) if (head + h < heads) {
        const float inv = ReuseCoefficients ? inverseSums[size_t(token) * heads + head + h] : 1.f / sum;
        output[size_t(head + h) * 512 + d] = __float2bfloat16_rn(acc[h] * inv);
    }
}
