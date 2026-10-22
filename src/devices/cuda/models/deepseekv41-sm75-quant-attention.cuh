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

// Four candidates x four heads per CTA. Each QK dot is computed once, using
// the scalar kernel's 64 ordered FMAs and the same eight-lane reduction.
template<bool MultiToken>
__global__ void __launch_bounds__(128)
V41Sm75QuantizedAttentionScores(const __nv_bfloat16 *q, const float *rows,
                                const int32_t *valid, int count, int heads,
                                float *scores) {
    const int token = MultiToken ? blockIdx.z : 0;
    q += size_t(token) * heads * 512;
    rows += size_t(token) * count * 512;
    valid += size_t(token) * count;
    scores += size_t(token) * heads * count;
    __shared__ float keys[4][520], queries[4][520];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int c = blockIdx.x * 4 + warp;
    const int h = blockIdx.y * 4 + lane / 8;
    const bool present = c < count && valid[c];
    if (present) {
        for (int d = lane; d < 512; d += 32)
            keys[warp][d + d / 64] = rows[size_t(c) * 512 + d];
    }
    for (int i = threadIdx.x; i < 4 * 512; i += 128) {
        const int head = blockIdx.y * 4 + i / 512, d = i % 512;
        queries[i / 512][d + d / 64] = head < heads
            ? __bfloat162float(q[size_t(head) * 512 + d]) : 0.f;
    }
    __syncthreads();
    if (!present) return;
    const int segment = (lane % 8) * 64;
    float dot = 0.f;
    #pragma unroll
    for (int d = 0; d < 64; ++d)
        dot += queries[lane / 8][segment + segment / 64 + d] *
               keys[warp][segment + segment / 64 + d];
    dot += __shfl_xor_sync(0xffffffff, dot, 1);
    dot += __shfl_xor_sync(0xffffffff, dot, 2);
    dot += __shfl_xor_sync(0xffffffff, dot, 4);
    if (lane % 8 == 0 && h < heads) scores[size_t(h) * count + c] = dot;
}

// Independent output dimensions can run in parallel without changing the
// online-softmax or PV accumulation order, including skipped/duplicate indices.
template<bool MultiToken>
__global__ void __launch_bounds__(32)
V41Sm75QuantizedAttentionValues(const float *rows, const int32_t *valid,
                                const float *scores, const float *sink,
                                int count, int heads, float scale, __nv_bfloat16 *output) {
    const int token = MultiToken ? blockIdx.z : 0;
    rows += size_t(token) * count * 512;
    valid += size_t(token) * count;
    scores += size_t(token) * heads * count;
    output += size_t(token) * heads * 512;
    const int head = blockIdx.x, d = blockIdx.y * 32 + threadIdx.x;
    constexpr int Tile = 4;
    float acc = 0.f, mx = sink[head], sum = 1.f;
    for (int base = 0; base < count; base += Tile) {
        float sc[Tile], values[Tile];
        bool present[Tile];
        #pragma unroll
        for (int j = 0; j < Tile; ++j) {
            const int c = base + j;
            present[j] = c < count && valid[c];
            if (present[j]) {
                sc[j] = scores[size_t(head) * count + c];
                values[j] = rows[size_t(c) * 512 + d];
            }
        }
        #pragma unroll
        for (int j = 0; j < Tile; ++j) if (present[j]) {
            const float s = sc[j] * scale;
            const float next = fmaxf(mx, s);
            const float alpha = __expf(mx - next), prob = __expf(s - next);
            sum = sum * alpha + prob;
            acc = acc * alpha + prob * values[j];
            mx = next;
        }
    }
    const float inv = 1.f / sum;
    output[size_t(head) * 512 + d] = __float2bfloat16_rn(acc * inv);
}
