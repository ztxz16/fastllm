#pragma once

// Native register-fragment implementation of Clover local-memory attention.
// Precision/dataflow reference (Apache-2.0):
// https://github.com/WilliamZR/sglang/blob/058939af24d5cdc44bb4b8852d0ddfab555ed71e/python/sglang/kernels/ops/mamba/clover_gdn_triton.py
// State is converted in registers only; the committed recurrent cache stays FP32.
__device__ __forceinline__ unsigned CloverPreviewPair(float a, float b) {
    __nv_bfloat162_raw v = __floats2bfloat162_rn(a, b);
    return (unsigned)v.x | ((unsigned)v.y << 16);
}
__device__ __forceinline__ void CloverPreviewMmaSync(float *c, const unsigned *a, const unsigned *b) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}
__device__ __forceinline__ void CloverPreviewLoadQuery(unsigned *a, const __nv_bfloat16 *p) {
#if __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                 : "r"((unsigned)__cvta_generic_to_shared(p))
                 : "memory");
#endif
}
__device__ __forceinline__ void CloverPreviewLoadKey(unsigned *b, const __nv_bfloat16 *p) {
#if __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(b[0]), "=r"(b[1])
                 : "r"((unsigned)__cvta_generic_to_shared(p))
                 : "memory");
#endif
}
template <int Tokens, bool Indirect>
__global__ void CloverPreviewGdnMma(const __nv_bfloat16 *query, const __nv_bfloat16 *key,
                                    const __nv_bfloat16 *value, const float *g, const float *state,
                                    const float *scale, const float *bias, const float *hc, const int *slots,
                                    const unsigned char *valid, __nv_bfloat16 *output, int kh, int vh,
                                    int qstride, int vstride, CloverCudaGdnCaches caches) {
#if __CUDA_ARCH__ >= 800
    CloverDependencyStart();
    int tid = threadIdx.x, lane = tid % 32, warp = tid / 32, request = blockIdx.z, head = blockIdx.y,
        first = blockIdx.x * 16, keyHead = head / (vh / kh), slot = slots ? slots[request] : request;
    if (slot < 0) {
        for (int i = tid; i < 16 * 128; i += 256)
            output[(((size_t)request * Tokens + first + i / 128) * vh + head) * 128 + i % 128] =
                __float2bfloat16(0.f);
        return;
    }
    if constexpr (Indirect)
        state = caches.state[slot];
    int stateSlot = Indirect ? 0 : slot;
    const float *headState = state + ((size_t)stateSlot * vh + head) * 128 * 128;
    // Eight-column padding avoids shared-memory bank conflicts in ldmatrix.
    // Each warp owns sixteen value columns and all sixteen query rows.
    __shared__ __align__(32) __nv_bfloat16 q[16][136], k[Tokens][136];
#pragma unroll
    for (int t = warp; t < 16; t += 8) {
        float x[4], sum = 0.f;
        size_t row = (size_t)request * Tokens + first + t;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            x[j] = (!valid || valid[row])
                       ? __bfloat162float(query[row * qstride + keyHead * 128 + lane + j * 32])
                       : 0.f;
            sum += x[j] * x[j];
        }
        float inv = rsqrtf(WarpSum(sum) + 1.e-6f);
#pragma unroll
        for (int j = 0; j < 4; ++j)
            q[t][lane + j * 32] = __float2bfloat16(x[j] * inv);
    }
#pragma unroll
    for (int t = warp; t < Tokens; t += 8) {
        float x[4], sum = 0;
        size_t row = (size_t)request * Tokens + t;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            x[j] = __bfloat162float(key[row * qstride + keyHead * 128 + lane + j * 32]);
            sum += x[j] * x[j];
        }
        float inv = rsqrtf(WarpSum(sum) + 1.e-6f);
#pragma unroll
        for (int j = 0; j < 4; ++j)
            k[t][lane + j * 32] = __float2bfloat16(x[j] * inv);
    }
    __syncthreads();
    // Pack the read-only FP32 state directly into BF16 MMA operands. Reuse
    // each state tile for every query instead of reloading it per query group.
    unsigned sr[8][2][2];
#pragma unroll
    for (int d = 0; d < 8; ++d)
#pragma unroll
        for (int n = 0; n < 2; ++n) {
            int vi = warp * 16 + n * 8 + lane / 4;
            int offset = vi * 128 + d * 16 + (lane % 4) * 2;
            sr[d][n][0] = CloverPreviewPair(headState[offset], headState[offset + 1]);
            sr[d][n][1] = CloverPreviewPair(headState[offset + 8], headState[offset + 9]);
        }
    float history[2][4] = {}, dots[Tokens / 8][4] = {};
    unsigned ar[8][4];
#pragma unroll
    for (int d = 0; d < 8; ++d)
        CloverPreviewLoadQuery(ar[d], &q[lane % 16][d * 16 + (lane / 16) * 8]);
#pragma unroll
    for (int d = 0; d < 8; ++d) {
#pragma unroll
        for (int n = 0; n < 2; ++n)
            CloverPreviewMmaSync(history[n], ar[d], sr[d][n]);
    }
#pragma unroll
    for (int n = 0; n < Tokens / 8; ++n) {
#pragma unroll
        for (int d = 0; d < 8; ++d) {
            unsigned br[2];
            CloverPreviewLoadKey(br, &k[n * 8 + lane % 8][d * 16 + ((lane / 8) % 2) * 8]);
            CloverPreviewMmaSync(dots[n], ar[d], br);
        }
    }
    // Replicate the small QK tile per value warp so softmax probabilities
    // remain in registers in the layout consumed by the final MMA.
    int count = __popc(
        __ballot_sync(0xffffffff, lane < Tokens && (!valid || valid[(size_t)request * Tokens + lane])));
    float h = bias[head] + hc[head] * __logf(fmaxf(count, 1) / 32.f), hw[2];
    unsigned probabilities[Tokens / 16][4];
#pragma unroll
    for (int row = 0; row < 2; ++row) {
        float maximum = h;
#pragma unroll
        for (int n = 0; n < Tokens / 8; ++n)
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                int t = n * 8 + (lane % 4) * 2 + j;
                float x = (!valid || valid[(size_t)request * Tokens + t]) ? dots[n][row * 2 + j] * scale[head]
                                                                          : -CUDART_INF_F;
                dots[n][row * 2 + j] = x;
                maximum = fmaxf(maximum, x);
            }
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 1, 4));
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, 2, 4));
        float he = __expf(h - maximum), sum = 0.f;
#pragma unroll
        for (int n = 0; n < Tokens / 8; ++n)
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                float x = __expf(dots[n][row * 2 + j] - maximum);
                dots[n][row * 2 + j] = x;
                sum += x;
            }
        sum += __shfl_xor_sync(0xffffffff, sum, 1, 4);
        sum += __shfl_xor_sync(0xffffffff, sum, 2, 4);
        float denom = sum + he;
        hw[row] = __fdividef(he, denom);
#pragma unroll
        for (int n = 0; n < Tokens / 8; ++n)
            probabilities[n / 2][(n % 2) * 2 + row] = CloverPreviewPair(
                __fdividef(dots[n][row * 2], denom), __fdividef(dots[n][row * 2 + 1], denom));
    }
    float local[2][4] = {};
#pragma unroll
    for (int t = 0; t < Tokens; t += 16) {
        const unsigned *a = probabilities[t / 16];
#pragma unroll
        for (int n = 0; n < 2; ++n) {
            unsigned b[2];
            int vi = warp * 16 + n * 8 + lane / 4, kt = t + (lane % 4) * 2;
#pragma unroll
            for (int d = 0; d < 2; ++d) {
                int row = kt + d * 8;
                unsigned lo =
                    __bfloat16_as_ushort(value[((size_t)request * Tokens + row) * vstride + head * 128 + vi]);
                unsigned hi = __bfloat16_as_ushort(
                    value[((size_t)request * Tokens + row + 1) * vstride + head * 128 + vi]);
                b[d] = lo | (hi << 16);
            }
            CloverPreviewMmaSync(local[n], a, b);
        }
    }
#pragma unroll
    for (int n = 0; n < 2; ++n)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int qi = lane / 4 + (j / 2) * 8, vi = warp * 16 + n * 8 + (lane % 4) * 2 + j % 2;
            size_t row = (size_t)request * Tokens + first + qi;
            float result =
                history[n][j] * ((1.f / sqrtf(128.f)) * __expf(g[row * vh + head]) * hw[j / 2]) + local[n][j];
            output[(row * vh + head) * 128 + vi] = __float2bfloat16(!valid || valid[row] ? result : 0.f);
        }
#endif
}
bool CloverPreviewMmaAvailable() {
    int device;
    if (cudaGetDevice(&device) != cudaSuccess)
        return false;
    static thread_local int previous = -1;
    static thread_local bool available = false;
    if (device != previous) {
        cudaFuncAttributes attributes{};
        if (cudaFuncGetAttributes(&attributes, CloverPreviewGdnMma<16, false>) != cudaSuccess)
            return false;
        available = attributes.binaryVersion >= 80 && attributes.ptxVersion >= 80;
        previous = device;
    }
    return available;
}
