#pragma once

// Two 16-head groups share each gathered KV tile. Query halves are cached in
// registers, then staged through the same shared memory as the score tile.
// Keep the existing two QK accumulators, 64-slot softmax (including holes),
// BF16 probabilities, PV accumulation order and final attention sink.
struct V41Sm120PrefillShared {
    __nv_bfloat16 kv[64][520];
    union {
        __nv_bfloat16 q[32][264];
        float scores[32][64];
    } stage;
    __nv_bfloat16 p[32][72];
    float maximum[32], sum[32], alpha[32];
    int valid[64];
};

__global__ void __launch_bounds__(512)
V41SparseAttentionSm120PrefillKernel(
        const __nv_bfloat16 *q, const __nv_bfloat16 *chunk,
        const __nv_bfloat16 *ring, const __nv_bfloat16 *compressed,
        const uint8_t *ring8, const uint8_t *compressed8,
        const int32_t *indices, const float *sink, int rows, int heads,
        int window, int capacity, int top, int start, float scale,
        __nv_bfloat16 *output, int ringMode, int ringBytes,
        int cmpMode, int cmpBytes, int vecUnpack) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ char v41Sm120PrefillRaw[];
    auto &sh = *reinterpret_cast<V41Sm120PrefillShared *>(v41Sm120PrefillRaw);
    const int token = blockIdx.x, batch = token / rows, pos = start + token % rows;
    const int h0 = blockIdx.y * 32, tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    constexpr int Warps = 16, Threads = Warps * 32, QueryItems = 1024 / Threads;
    constexpr int Dots = 16 / Warps, Values = 128 / Warps, Slice = 1024 / Warps;
    const int group = warp / (Warps / 2), nw = warp % (Warps / 2);
    const int slots = start == 0 ? min(rows, window) : window;
    const int first = start == 0 ? max(0, pos - window + 1) : pos - window + 1;
    const int count = slots + (indices ? top : 0);
    const int32_t *selected = indices ? indices + size_t(token) * top : nullptr;

    float4 query[2][QueryItems];
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
        #pragma unroll
        for (int j = 0; j < QueryItems; ++j) {
            const int v = tid + j * Threads, h = v / 32, d = (v % 32) * 8;
            query[half][j] = h0 + h < heads
                ? *reinterpret_cast<const float4 *>(q + (size_t(token) * heads + h0 + h) * 512 + half * 256 + d)
                : make_float4(0, 0, 0, 0);
        }
    }
    if (tid < 32) { sh.maximum[tid] = -1e30f; sh.sum[tid] = 0; }
    float acc[Values][4] = {};
    const int ar = group * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
    const int ac = (lane >> 4) * 8;

    for (int tile = 0; tile < (count + 63) / 64; ++tile) {
        __syncthreads();
        for (int slot = warp; slot < 64; slot += Warps) {
            const int c = tile * 64 + slot;
            const __nv_bfloat16 *src = nullptr;
            const uint8_t *packed = nullptr;
            int mode = 1;
            bool valid = c < count;
            if (valid) {
                if (c < slots) {
                    const int p = first + c;
                    if (p < 0 || p > pos) valid = false;
                    else if (p >= start) src = chunk + (size_t(batch) * rows + p - start) * 512;
                    else if (ring8) { packed = ring8 + (size_t(batch) * window + p % window) * ringBytes; mode = ringMode; }
                    else src = ring + (size_t(batch) * window + p % window) * 512;
                } else {
                    const int id = selected[c - slots];
                    if (id < 0 || id >= capacity) valid = false;
                    else if (compressed8) { packed = compressed8 + (size_t(batch) * capacity + id) * cmpBytes; mode = cmpMode; }
                    else src = compressed + (size_t(batch) * capacity + id) * 512;
                }
            }
            if (!valid) {
                for (int v = lane; v < 64; v += 32)
                    *reinterpret_cast<float4 *>(&sh.kv[slot][v * 8]) = make_float4(0, 0, 0, 0);
            } else if (packed) {
                if (vecUnpack) V41LoadKvRow16Bf16(packed, 512, lane * 16, mode, &sh.kv[slot][lane * 16]);
                else for (int d = lane; d < 512; d += 32)
                    sh.kv[slot][d] = __float2bfloat16_rn(V41LoadKvRow(packed, 512, d, mode));
            } else {
                for (int v = lane; v < 64; v += 32)
                    *reinterpret_cast<float4 *>(&sh.kv[slot][v * 8]) = reinterpret_cast<const float4 *>(src)[v];
            }
            if (lane == 0) sh.valid[slot] = valid;
        }
        float dot[Dots][2][4] = {};
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            #pragma unroll
            for (int j = 0; j < QueryItems; ++j) {
                const int v = tid + j * Threads;
                *reinterpret_cast<float4 *>(&sh.stage.q[v / 32][(v % 32) * 8]) = query[half][j];
            }
            __syncthreads();
            #pragma unroll 2
            for (int k = 0; k < 256; k += 32) {
                uint32_t a[4], b[2];
                V41LdmX4(a, &sh.stage.q[ar][k + ac]);
                #pragma unroll
                for (int n = 0; n < Dots; ++n) {
                    V41LdmX2(b, &sh.kv[nw * (64 / (Warps / 2)) + n * 8 + (lane & 7)][half * 256 + k + ((lane >> 3) & 1) * 8]);
                    V41MmaBf16(dot[n][0], a, b);
                }
                V41LdmX4(a, &sh.stage.q[ar][k + 16 + ac]);
                #pragma unroll
                for (int n = 0; n < Dots; ++n) {
                    V41LdmX2(b, &sh.kv[nw * (64 / (Warps / 2)) + n * 8 + (lane & 7)][half * 256 + k + 16 + ((lane >> 3) & 1) * 8]);
                    V41MmaBf16(dot[n][1], a, b);
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (int n = 0; n < Dots; ++n) {
            const int r = group * 16 + lane / 4, c = nw * (64 / (Warps / 2)) + n * 8 + (lane % 4) * 2;
            sh.stage.scores[r][c] = dot[n][0][0] + dot[n][1][0];
            sh.stage.scores[r][c+1] = dot[n][0][1] + dot[n][1][1];
            sh.stage.scores[r+8][c] = dot[n][0][2] + dot[n][1][2];
            sh.stage.scores[r+8][c+1] = dot[n][0][3] + dot[n][1][3];
        }
        __syncthreads();
        #pragma unroll
        for (int hg = 0; hg < 512 / Threads; ++hg) {
            const int r = tid / 16 + hg * (Threads / 16), sub = tid % 16;
            float values[4], maximum = -FLT_MAX;
            bool valid[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = sub * 4 + j;
                valid[j] = sh.valid[c];
                values[j] = valid[j] ? sh.stage.scores[r][c] * scale : -FLT_MAX;
                maximum = fmaxf(maximum, values[j]);
            }
            #pragma unroll
            for (int offset = 1; offset < 16; offset *= 2)
                maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
            const float old = sh.maximum[r], next = fmaxf(old, maximum);
            float sum = 0;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float p = valid[j] ? __expf(values[j] - next) : 0.f;
                sum += p;
                sh.p[r][sub * 4 + j] = __float2bfloat16_rn(p);
            }
            #pragma unroll
            for (int offset = 1; offset < 16; offset *= 2)
                sum += __shfl_xor_sync(0xffffffff, sum, offset);
            // All lanes must finish reading the previous shared maximum
            // before its owner overwrites it. Shuffles are not memory fences.
            __syncwarp();
            if (sub == 0) {
                const float alpha = __expf(old - next);
                sh.alpha[r] = alpha; sh.maximum[r] = next;
                sh.sum[r] = sh.sum[r] * alpha + sum;
            }
        }
        __syncthreads();
        const float a0 = sh.alpha[group * 16 + lane / 4], a1 = sh.alpha[group * 16 + lane / 4 + 8];
        #pragma unroll
        for (int n = 0; n < Values; ++n) {
            acc[n][0] *= a0; acc[n][1] *= a0;
            acc[n][2] *= a1; acc[n][3] *= a1;
        }
        #pragma unroll
        for (int k = 0; k < 64; k += 16) {
            uint32_t a[4]; V41LdmX4(a, &sh.p[ar][k + ac]);
            #pragma unroll
            for (int n = 0; n < Values; ++n) {
                uint32_t b[2]; V41LdmX2T(b, &sh.kv[k + (lane & 15)][nw * Slice + n * 8]);
                V41MmaBf16(acc[n], a, b);
            }
        }
    }
    __syncthreads();
    const int r = group * 16 + lane / 4, col = nw * Slice + (lane % 4) * 2;
    const float inv0 = h0+r < heads ? 1.f / (sh.sum[r] + expf(sink[h0+r] - sh.maximum[r])) : 0.f;
    const float inv1 = h0+r+8 < heads ? 1.f / (sh.sum[r+8] + expf(sink[h0+r+8] - sh.maximum[r+8])) : 0.f;
    #pragma unroll
    for (int n = 0; n < Values; ++n) {
        if (h0+r < heads) {
            auto *out = output + (size_t(token)*heads+h0+r)*512 + col+n*8;
            out[0] = __float2bfloat16_rn(acc[n][0]*inv0); out[1] = __float2bfloat16_rn(acc[n][1]*inv0);
        }
        if (h0+r+8 < heads) {
            auto *out = output + (size_t(token)*heads+h0+r+8)*512 + col+n*8;
            out[0] = __float2bfloat16_rn(acc[n][2]*inv1); out[1] = __float2bfloat16_rn(acc[n][3]*inv1);
        }
    }
#endif
}
