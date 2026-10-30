// Native CUDA block delta-rule prefill. Derived from the causal recurrence
// and checked against the Apache-2.0 Diffuspace Clover reference and SGLang's
// FLA block algorithm; no SGLang, Triton or cuBLAS runtime dependency.
// BF16 block products use FP32 accumulation; triangular diagonal solves and
// committed states remain FP32. FP64 gate prefixes avoid cancellation for
// extreme decay values. Snapshot-producing calls use the scalar recurrence.
#ifndef FASTLLM_CLOVER_GDN_PREFILL_CUH
#define FASTLLM_CLOVER_GDN_PREFILL_CUH
#include <limits>
namespace {
namespace clover_prefill {
constexpr int C = 64, D = 128, InverseStride = 68;
struct Workspace {
    float *prepared, *lower, *p, *tail, *beta;
    double *g;
    __nv_bfloat16 *rhs, *u, *inverse, *history, *delta, *w, *scaledQ, *scaledK, *qk;
    int batch, tokens, keyHeads, valueHeads;
    size_t bytes;
    Workspace(void *base, int b, int t, int kh, int vh) : batch(b), tokens(t), keyHeads(kh), valueHeads(vh) {
        size_t offset = 0, groups = (size_t)((t + C - 1) / C) * b * vh;
        auto allocate = [&](size_t count, size_t unit) -> void * {
            offset = (offset + 127) & ~size_t(127);
            auto *ptr = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(base) + offset);
            offset += count * unit;
            return ptr;
        };
        prepared = (float *)allocate((size_t)b * t * (2 * kh * D + vh), sizeof(float));
        lower = (float *)allocate(groups * C * C, sizeof(float));
        p = (float *)allocate(groups * C, sizeof(float));
        tail = (float *)allocate(groups * C, sizeof(float));
        beta = (float *)allocate(groups * C, sizeof(float));
        g = (double *)allocate(groups * C, sizeof(double));
        rhs = (__nv_bfloat16 *)allocate(groups * C * D * 2, sizeof(__nv_bfloat16));
        u = (__nv_bfloat16 *)allocate(groups * C * D, sizeof(__nv_bfloat16));
        inverse = (__nv_bfloat16 *)allocate(groups * C * C, sizeof(__nv_bfloat16));
        history = (__nv_bfloat16 *)allocate(groups * D * D, sizeof(__nv_bfloat16));
        delta = (__nv_bfloat16 *)allocate(groups * C * D, sizeof(__nv_bfloat16));
        w = (__nv_bfloat16 *)allocate(groups * C * D, sizeof(__nv_bfloat16));
        scaledQ = (__nv_bfloat16 *)allocate(groups * C * D, sizeof(__nv_bfloat16));
        scaledK = (__nv_bfloat16 *)allocate(groups * C * D, sizeof(__nv_bfloat16));
        qk = (__nv_bfloat16 *)allocate(groups * C * C, sizeof(__nv_bfloat16));
        bytes = (offset + 127) & ~size_t(127);
    }
};
__global__ void Prepare(const float *prepared, const __nv_bfloat16 *values, const float *g,
                        const __nv_bfloat16 *beta, Workspace w, int b, int tokens, int kh, int vh,
                        int valueStride) {
    int chunk = blockIdx.x, bh = blockIdx.y, head = bh % vh, request = bh / vh;
    int group = chunk * b * vh + bh, keyHead = head / (vh / kh);
    size_t count = (size_t)b * tokens * kh * D;
    __shared__ double prefix[C];
    __shared__ float powers[C], tails[C], rates[C];
    __shared__ float transpose[C][D + 1];
    if (threadIdx.x < C) {
        int t = threadIdx.x, token = chunk * C + t, lane = t % 32;
        double sum = token < tokens ? g[((size_t)request * tokens + token) * vh + head] : 0.;
        for (int offset = 1; offset < 32; offset *= 2) {
            double value = __shfl_up_sync(0xffffffff, sum, offset);
            if (lane >= offset)
                sum += value;
        }
        prefix[t] = sum;
    }
    __syncthreads();
    if (threadIdx.x >= 32 && threadIdx.x < C)
        prefix[threadIdx.x] += prefix[31];
    __syncthreads();
    if (threadIdx.x < C) {
        int t = threadIdx.x, token = chunk * C + t;
        powers[t] = expf(prefix[t]);
        tails[t] = expf(prefix[C - 1] - prefix[t]);
        rates[t] =
            token < tokens ? __bfloat162float(beta[((size_t)request * tokens + token) * vh + head]) : 0.f;
        w.g[(size_t)group * C + t] = prefix[t];
        w.p[(size_t)group * C + t] = powers[t];
        w.tail[(size_t)group * C + t] = tails[t];
        w.beta[(size_t)group * C + t] = rates[t];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < C * D; i += blockDim.x) {
        int t = i / D, d = i % D, token = chunk * C + t;
        float k = 0;
        if (token < tokens) {
            size_t row = (size_t)request * tokens + token;
            k = prepared[count + (row * kh + keyHead) * D + d];
        }
        transpose[t][d] = rates[t] * powers[t] * k;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < C * D; i += blockDim.x) {
        int t = i % C, d = i / C;
        w.rhs[(size_t)group * C * D * 2 + D * C + i] = __float2bfloat16(transpose[t][d]);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < C * D; i += blockDim.x) {
        int t = i / D, d = i % D, token = chunk * C + t;
        float v =
            token < tokens
                ? __bfloat162float(values[((size_t)request * tokens + token) * valueStride + head * D + d])
                : 0.f;
        transpose[t][d] = rates[t] * v;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < C * D; i += blockDim.x) {
        int t = i % C, d = i / C;
        w.rhs[(size_t)group * C * D * 2 + i] = __float2bfloat16(transpose[t][d]);
    }
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
using namespace nvcuda;
using A = wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::col_major>;
using B = wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major>;
using Acc = wmma::fragment<wmma::accumulator, 16, 16, 8, float>;
__device__ void MmaTF32(Acc &acc, A a, B b) {
#if __CUDA_ARCH__ >= 800
#pragma unroll
    for (int j = 0; j < A::num_elements; ++j)
        a.x[j] = wmma::__float_to_tf32(a.x[j]);
#pragma unroll
    for (int j = 0; j < B::num_elements; ++j)
        b.x[j] = wmma::__float_to_tf32(b.x[j]);
    wmma::mma_sync(acc, a, b, acc);
#endif
}
#endif

// Form four diagonal inverses with scalar FP32 substitutions, then merge
// diagonal blocks using -D^-1 B A^-1. Matrices are column major throughout.
__global__ void Inverse(Workspace w) {
#if __CUDA_ARCH__ >= 800
    constexpr int CS = InverseStride;
    int group = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    extern __shared__ __align__(32) float inverseShared[];
    auto *lower = inverseShared, *inv = lower + C * CS, *tmp = inv + C * CS;
    for (int i = threadIdx.x; i < C * C; i += blockDim.x) {
        lower[(i / C) * CS + i % C] = w.lower[(size_t)group * C * C + i];
        inv[(i / C) * CS + i % C] = 0.f;
    }
    __syncthreads();
    float column[16];
#pragma unroll
    for (int row = 0; row < 16; ++row) {
        float x = row == lane ? 1.f : 0.f;
#pragma unroll
        for (int j = 0; j < row; ++j)
            x -= lower[(warp * 16 + j) * CS + warp * 16 + row] * column[j];
        column[row] = x;
        if (lane < 16)
            inv[(warp * 16 + lane) * CS + warp * 16 + row] = x;
    }
    __syncthreads();
    // Two 16+16 merges followed by one 32+32 merge; each warp owns
    // one 16x16 output tile, so no shared writes overlap.
#pragma unroll
    for (int level = 0; level < 2; ++level) {
        int size = 16 << level;
        int row = level == 0 ? (warp / 2) * 32 + 16 : 32 + (warp / 2) * 16;
        int col = level == 0 ? (warp / 2) * 32 : (warp % 2) * 16;
        bool active = level != 0 || (warp % 2) == 0;
        if (active) {
            Acc a;
            wmma::fill_fragment(a, 0.f);
            int start = level == 0 ? (warp / 2) * 32 : 0;
            for (int j = 0; j < size; j += 8) {
                A left;
                B right;
                wmma::load_matrix_sync(left, lower + (start + j) * CS + row, CS);
                wmma::load_matrix_sync(right, inv + col * CS + start + j, CS);
                MmaTF32(a, left, right);
            }
            wmma::store_matrix_sync(tmp + col * CS + row, a, CS, wmma::mem_col_major);
        }
        __syncthreads();
        if (active) {
            Acc a;
            wmma::fill_fragment(a, 0.f);
            int start = level == 0 ? (warp / 2) * 32 + 16 : 32;
            for (int j = 0; j < size; j += 8) {
                A left;
                B right;
                wmma::load_matrix_sync(left, inv + (start + j) * CS + row, CS);
                wmma::load_matrix_sync(right, tmp + col * CS + start + j, CS);
                MmaTF32(a, left, right);
            }
#pragma unroll
            for (int j = 0; j < Acc::num_elements; ++j)
                a.x[j] = -a.x[j];
            wmma::store_matrix_sync(inv + col * CS + row, a, CS, wmma::mem_col_major);
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < C * C; i += blockDim.x)
        w.inverse[(size_t)group * C * C + i] = __float2bfloat16(inv[(i / C) * CS + i % C]);
#endif
}

struct RegisterA {
    unsigned x[4];
};
struct RegisterB {
    unsigned x[2];
};
__device__ __forceinline__ RegisterA ReadA(const __nv_bfloat16 *p, int stride) {
#if __CUDA_ARCH__ >= 800

    int lane = threadIdx.x % 32;
    unsigned address = __cvta_generic_to_shared(p + (lane % 16) * stride + (lane / 16) * 8);
    RegisterA a;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a.x[0]), "=r"(a.x[1]), "=r"(a.x[2]), "=r"(a.x[3])
                 : "r"(address));
    return a;

#else
    return RegisterA{};
#endif
}
__device__ __forceinline__ RegisterB ReadB(const __nv_bfloat16 *p, int stride) {
#if __CUDA_ARCH__ >= 800

    int lane = threadIdx.x % 32;
    unsigned address = __cvta_generic_to_shared(p + (lane % 16) * stride);
    RegisterB b;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                 : "=r"(b.x[0]), "=r"(b.x[1])
                 : "r"(address));
    return b;

#else
    return RegisterB{};
#endif
}
__device__ __forceinline__ void RegisterMma(float *c, const RegisterA &a, const RegisterB &b) {
#if __CUDA_ARCH__ >= 800

    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "r"(b.x[0]), "r"(b.x[1]));

#else
#endif
}
constexpr int WS = C + 8, HS = D + 8, WeightSize = D * WS, StateSize = 32 * HS, DeltaSize = 32 * WS;
// U has the same padded 32-value by 64-token layout as the delta tile.
constexpr int KeySize = C * HS, BufferSize = WeightSize + KeySize + DeltaSize;
constexpr int AdvanceSharedBytes = (2 * BufferSize + StateSize + DeltaSize) * sizeof(__nv_bfloat16);
__global__ void Advance(Workspace w, float *state, int bhCount, int chunks) {
#if __CUDA_ARCH__ >= 800
    extern __shared__ __align__(32) __nv_bfloat16 advanceShared[];
    auto *buffers = advanceShared, *states = buffers + 2 * BufferSize, *deltas = states + StateSize;
    int bh = blockIdx.y, vs = blockIdx.x * 32, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int vr = (warp / 2) * 16, kr = (warp % 2) * 64, tr = (warp % 2) * 32, row = lane / 4,
        col = (lane % 4) * 2;
    float accum[8][4];
#pragma unroll
    for (int n = 0; n < 8; ++n) {
        int key = kr + n * 8 + col;
        size_t base = (size_t)bh * D * D + (vs + vr + row) * D + key;
        accum[n][0] = state[base];
        accum[n][1] = state[base + 1];
        accum[n][2] = state[base + 8 * D];
        accum[n][3] = state[base + 8 * D + 1];
    }
    auto stage = [&](__nv_bfloat16 *dst, const __nv_bfloat16 *src, int rows, int cols, int stride) {
        for (int i = threadIdx.x; i < rows * (cols / 8); i += blockDim.x) {
            unsigned address =
                __cvta_generic_to_shared(dst + (i / (cols / 8)) * stride + (i % (cols / 8)) * 8);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(address),
                         "l"((const uint4 *)src + i));
        }
    };
    auto prefetch = [&](int chunk) {
        int group = chunk * bhCount + bh;
        auto *dst = buffers + (chunk % 2) * BufferSize;
        stage(dst, w.w + (size_t)group * C * D, D, C, WS);
        stage(dst + WeightSize, w.scaledK + (size_t)group * C * D, C, D, HS);
        stage(dst + WeightSize + KeySize, w.u + (size_t)group * C * D + vs * C, 32, C, WS);
        asm volatile("cp.async.commit_group;");
    };
    prefetch(0);
    for (int chunk = 0; chunk < chunks; ++chunk) {
        int group = chunk * bhCount + bh;
        auto *weights = buffers + (chunk % 2) * BufferSize, *keys = weights + WeightSize;
        auto *values = keys + KeySize;
        if (chunk + 1 < chunks)
            prefetch(chunk + 1);
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            int key = kr + n * 8 + col;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                float x = accum[n][h * 2], y = accum[n][h * 2 + 1];
                auto high = __floats2bfloat162_rn(x, y);
                int index = (vr + row + h * 8) * HS + key;
                *(__nv_bfloat162 *)(states + index) = high;
            }
        }
        if (chunk + 1 < chunks)
            asm volatile("cp.async.wait_group 1;");
        else
            asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        for (int i = threadIdx.x; i < 32 * (D / 8); i += blockDim.x) {
            int row = i / (D / 8), col = (i % (D / 8)) * 8;
            *(uint4 *)(w.history + (size_t)group * D * D + (vs + row) * D + col) =
                *(const uint4 *)(states + row * HS + col);
        }
        float product[4][4] = {};
#pragma unroll
        for (int k = 0; k < D; k += 16) {
            auto ah = ReadA(states + vr * HS + k, HS);
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                auto bh = ReadB(weights + k * WS + tr + n * 8, WS);
                RegisterMma(product[n], ah, bh);
            }
        }
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            int t = tr + n * 8 + col;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                int value = vr + row + h * 8;
                float x = __bfloat162float(values[value * WS + t]) - product[n][h * 2];
                float y = __bfloat162float(values[value * WS + t + 1]) - product[n][h * 2 + 1];
                auto high = __floats2bfloat162_rn(x, y);
                *(__nv_bfloat162 *)(deltas + value * WS + t) = high;
            }
        }
        float decay = w.p[(size_t)group * C + C - 1];
#pragma unroll
        for (int n = 0; n < 8; ++n)
            for (int j = 0; j < 4; ++j)
                accum[n][j] *= decay;
        // Publish all delta tiles for the vector stores and state update.
        __syncthreads();
        for (int i = threadIdx.x; i < 32 * (C / 8); i += blockDim.x) {
            int row = i / (C / 8), col = (i % (C / 8)) * 8;
            *(uint4 *)(w.delta + (size_t)group * C * D + (vs + row) * C + col) =
                *(const uint4 *)(deltas + row * WS + col);
        }
#pragma unroll
        for (int t = 0; t < C; t += 16) {
            auto ah = ReadA(deltas + vr * WS + t, WS);
#pragma unroll
            for (int n = 0; n < 8; ++n) {
                auto bh = ReadB(keys + t * HS + kr + n * 8, HS);
                RegisterMma(accum[n], ah, bh);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int n = 0; n < 8; ++n) {
        int key = kr + n * 8 + col;
        size_t base = (size_t)bh * D * D + (vs + vr + row) * D + key;
        *(float2 *)(state + base) = make_float2(accum[n][0], accum[n][1]);
        *(float2 *)(state + base + 8 * D) = make_float2(accum[n][2], accum[n][3]);
    }
#endif
}

constexpr int OutputValue = 64, OutputKeyStride = 136, OutputTimeStride = 72;
constexpr int OutputElements = 2 * C * OutputKeyStride + 2 * C * OutputTimeStride;
constexpr int OutputSharedBytes = OutputElements * sizeof(__nv_bfloat16);
__device__ __forceinline__ RegisterB ReadBColumn(const __nv_bfloat16 *p, int stride) {
#if __CUDA_ARCH__ >= 800

    int lane = threadIdx.x % 32;
    unsigned address = __cvta_generic_to_shared(p + (lane % 8) * stride + ((lane / 8) % 2) * 8);
    RegisterB b;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(b.x[0]), "=r"(b.x[1])
                 : "r"(address));
    return b;

#else
    return RegisterB{};
#endif
}

__device__ __forceinline__ RegisterA ReadAColumn(const __nv_bfloat16 *p, int stride) {
#if __CUDA_ARCH__ >= 800

    int lane = threadIdx.x % 32;
    unsigned address = __cvta_generic_to_shared(p + (lane % 16) * stride + (lane / 16) * 8);
    RegisterA a;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a.x[0]), "=r"(a.x[2]), "=r"(a.x[1]), "=r"(a.x[3])
                 : "r"(address));
    return a;

#else
    return RegisterA{};
#endif
}
__global__ void FusedOutput(Workspace w, __nv_bfloat16 *out, int tokens, int vh, int bhCount) {
#if __CUDA_ARCH__ >= 800
    extern __shared__ __align__(32) __nv_bfloat16 outputShared[];
    auto *q = outputShared, *history = q + C * OutputKeyStride, *qk = history + C * OutputKeyStride,
         *delta = qk + C * OutputTimeStride;
    int group = blockIdx.y, vs = blockIdx.x * OutputValue, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int chunk = group / bhCount, bh = group % bhCount, request = bh / vh, head = bh % vh;

    auto stage = [&](__nv_bfloat16 *dst, const __nv_bfloat16 *src, int rows, int cols, int stride) {
        for (int i = threadIdx.x; i < rows * (cols / 8); i += blockDim.x)
            *(uint4 *)(dst + (i / (cols / 8)) * stride + (i % (cols / 8)) * 8) = ((const uint4 *)src)[i];
    };
    stage(q, w.scaledQ + (size_t)group * C * D, C, D, OutputKeyStride);
    stage(history, w.history + (size_t)group * D * D + vs * D, C, D, OutputKeyStride);
    stage(qk, w.qk + (size_t)group * C * C, C, C, OutputTimeStride);
    stage(delta, w.delta + (size_t)group * C * D + vs * C, C, C, OutputTimeStride);
    __syncthreads();
    // Each warp owns four row tiles and two column tiles. Reuse their
    // fragments while preserving each output's original K accumulation order.
    float accum[8][4] = {};
#pragma unroll
    for (int k = 0; k < D; k += 16) {
        RegisterB right[2];
#pragma unroll
        for (int n = 0; n < 2; ++n)
            right[n] = ReadBColumn(history + (warp * 8 + n * 32) * OutputKeyStride + k, OutputKeyStride);
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            auto a = ReadA(q + r * 16 * OutputKeyStride + k, OutputKeyStride);
#pragma unroll
            for (int n = 0; n < 2; ++n)
                RegisterMma(accum[r * 2 + n], a, right[n]);
        }
    }
#pragma unroll
    for (int k = 0; k < C; k += 16) {
        RegisterB right[2];
#pragma unroll
        for (int n = 0; n < 2; ++n)
            right[n] = ReadBColumn(delta + (warp * 8 + n * 32) * OutputTimeStride + k, OutputTimeStride);
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            auto a = ReadAColumn(qk + k * OutputTimeStride + r * 16, OutputTimeStride);
#pragma unroll
            for (int n = 0; n < 2; ++n)
                RegisterMma(accum[r * 2 + n], a, right[n]);
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int tile = warp + i * 4, t = (tile / 8) * 16 + lane / 4, v = vs + (tile % 8) * 8 + (lane % 4) * 2;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            int token = chunk * C + t + h * 8;
            if (token < tokens) {
                size_t index = (((size_t)request * tokens + token) * vh + head) * D + v;
                *(__nv_bfloat162 *)(out + index) =
                    __floats2bfloat162_rn(accum[i][h * 2], accum[i][h * 2 + 1]);
            }
        }
    }
#endif
}

constexpr int PairStride = 136, PairElements = 2 * C * PairStride;
constexpr int PairSharedBytes = PairElements * sizeof(__nv_bfloat16);
__device__ __forceinline__ void PairSplit(float2 v, __nv_bfloat16 *high) {
    auto h = __floats2bfloat162_rn(v.x, v.y);
    *(__nv_bfloat162 *)high = h;
}
__global__ void FusedPairs(Workspace w) {
#if __CUDA_ARCH__ >= 800
    extern __shared__ __align__(32) __nv_bfloat16 pairShared[];
    auto *q = pairShared, *k = q + C * PairStride;
    __shared__ double gates[C];
    __shared__ float rates[C];
    int group = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (threadIdx.x < C) {
        gates[threadIdx.x] = w.g[(size_t)group * C + threadIdx.x];
        rates[threadIdx.x] = w.beta[(size_t)group * C + threadIdx.x];
    }
    for (int i = threadIdx.x; i < C * (D / 2); i += blockDim.x) {
        int t = i / (D / 2), col = (i % (D / 2)) * 2;
        size_t index = (size_t)group * C * D + t * D + col;
        int bh = group % (w.batch * w.valueHeads), chunk = group / (w.batch * w.valueHeads);
        int token = chunk * C + t, request = bh / w.valueHeads,
            keyHead = (bh % w.valueHeads) / (w.valueHeads / w.keyHeads);
        size_t source = (((size_t)request * w.tokens + token) * w.keyHeads + keyHead) * D + col;
        size_t qcount = (size_t)w.batch * w.tokens * w.keyHeads * D;
        float2 qv = make_float2(0.f, 0.f), kv = qv;
        if (token < w.tokens) {
            qv = *(float2 *)(w.prepared + source);
            kv = *(float2 *)(w.prepared + qcount + source);
        }
        PairSplit(qv, q + t * PairStride + col);
        PairSplit(kv, k + t * PairStride + col);
        float power = w.p[(size_t)group * C + t], tail = w.tail[(size_t)group * C + t];
        *(__nv_bfloat162 *)(w.scaledQ + index) = __floats2bfloat162_rn(qv.x * power, qv.y * power);
        *(__nv_bfloat162 *)(w.scaledK + index) = __floats2bfloat162_rn(kv.x * tail, kv.y * tail);
    }
    __syncthreads();
    float kk[8][4] = {}, qk[8][4] = {};
#pragma unroll
    for (int d = 0; d < D; d += 16) {
        RegisterB right[2];
#pragma unroll
        for (int n = 0; n < 2; ++n)
            right[n] = ReadBColumn(k + (warp * 8 + n * 32) * PairStride + d, PairStride);
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            int row = r * 16;
            if (row + 15 >= warp * 8) {
                auto a = ReadA(k + row * PairStride + d, PairStride);
                auto aq = ReadA(q + row * PairStride + d, PairStride);
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    if (row + 15 >= warp * 8 + n * 32) {
                        RegisterMma(kk[r * 2 + n], a, right[n]);
                        RegisterMma(qk[r * 2 + n], aq, right[n]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int tile = warp + i * 4, row = (tile / 8) * 16 + lane / 4, col = (tile % 8) * 8 + (lane % 4) * 2;
#pragma unroll
        for (int h = 0; h < 2; ++h)
            for (int j = 0; j < 2; ++j) {
                int rr = row + h * 8, cc = col + j;
                size_t index = (size_t)group * C * C + cc * C + rr;
                float scale = rr >= cc ? expf(gates[rr] - gates[cc]) : 0.f;
                w.lower[index] = rr == cc ? 1.f : (rr > cc ? kk[i][h * 2 + j] * rates[rr] * scale : 0.f);
                w.qk[index] = __float2bfloat16(rr >= cc ? qk[i][h * 2 + j] * scale : 0.f);
            }
    }
#endif
}

__global__ void RecomputeWU(Workspace w) {
#if __CUDA_ARCH__ >= 800
    constexpr int Stride = 72;
    __shared__ __align__(32) __nv_bfloat16 inverse[C * Stride], rhs[C * Stride], result[C * Stride];
    int group = blockIdx.y, column = blockIdx.x * 64, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    for (int i = threadIdx.x; i < C * (C / 8); i += blockDim.x) {
        int col = i / (C / 8), row = (i % (C / 8)) * 8;
        *(uint4 *)(inverse + col * Stride + row) = ((uint4 *)(w.inverse + (size_t)group * C * C))[i];
    }
    for (int i = threadIdx.x; i < C * (C / 8); i += blockDim.x) {
        int col = i / (C / 8), row = (i % (C / 8)) * 8;
        *(uint4 *)(rhs + col * Stride + row) = ((uint4 *)(w.rhs + (size_t)group * C * D * 2 + column * C))[i];
    }
    __syncthreads();
    float accum[8][4] = {};
#pragma unroll
    for (int k = 0; k < C; k += 16) {
        RegisterB right[2];
#pragma unroll
        for (int n = 0; n < 2; ++n)
            right[n] = ReadBColumn(rhs + (warp * 8 + n * 32) * Stride + k, Stride);
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            auto a = ReadAColumn(inverse + k * Stride + r * 16, Stride);
#pragma unroll
            for (int n = 0; n < 2; ++n)
                RegisterMma(accum[r * 2 + n], a, right[n]);
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int tile = warp + i * 4, row = (tile / 8) * 16 + lane / 4, col = (tile % 8) * 8 + (lane % 4) * 2;
#pragma unroll
        for (int h = 0; h < 2; ++h)
            for (int j = 0; j < 2; ++j)
                result[(col + j) * Stride + row + h * 8] = __float2bfloat16(accum[i][h * 2 + j]);
    }
    __syncthreads();
    auto *out = column < D ? (w.u + (size_t)group * C * D + column * C)
                           : (w.w + (size_t)group * C * D + (column - D) * C);
    for (int i = threadIdx.x; i < C * (C / 8); i += blockDim.x) {
        int col = i / (C / 8), row = (i % (C / 8)) * 8;
        ((uint4 *)out)[i] = *(uint4 *)(result + col * Stride + row);
    }
#endif
}

// Configure before launching any kernels, so an unsupported device leaves all
// caller-owned buffers untouched. The loaded PTX version matters after JIT.
bool Supported() {
    int device;
    if (cudaGetDevice(&device) != cudaSuccess)
        return false;
    static thread_local int previous = -1;
    static thread_local bool supported = false;
    if (device == previous)
        return supported;
    cudaStreamCaptureStatus capture;
    if (cudaStreamIsCapturing(cudaStreamPerThread, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone)
        return false;
    cudaFuncAttributes attr;
    int sharedLimit = 0;
    if (cudaFuncGetAttributes(&attr, Advance) != cudaSuccess ||
        cudaDeviceGetAttribute(&sharedLimit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) != cudaSuccess)
        return false;
    previous = device;
    supported = false;
    if (attr.binaryVersion < 80 || attr.ptxVersion < 80 ||
        sharedLimit < AdvanceSharedBytes + (int)attr.sharedSizeBytes)
        return false;
    if (cudaFuncSetAttribute(FusedPairs, cudaFuncAttributeMaxDynamicSharedMemorySize, PairSharedBytes) !=
            cudaSuccess ||
        cudaFuncSetAttribute(Inverse, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             3 * C * InverseStride * sizeof(float)) != cudaSuccess ||
        cudaFuncSetAttribute(Advance, cudaFuncAttributeMaxDynamicSharedMemorySize, AdvanceSharedBytes) !=
            cudaSuccess ||
        cudaFuncSetAttribute(FusedOutput, cudaFuncAttributeMaxDynamicSharedMemorySize, OutputSharedBytes) !=
            cudaSuccess)
        return false;
    supported = true;
    return true;
}
size_t Bytes(int batch, int tokens, int kh, int vh) {
    if (batch <= 0 || tokens <= 0 || tokens > 65535 || kh <= 0 || vh <= 0 || vh > 65535 || vh % kh ||
        (size_t)batch * vh * ((tokens + C - 1) / C) > 65535)
        return 0;
    size_t bytes = Workspace(nullptr, batch, tokens, kh, vh).bytes;
    if (bytes > size_t(std::numeric_limits<int>::max()) || !Supported())
        return 0;
    return bytes;
}
bool Run(const void *query, const void *key, const void *value, const float *g, const void *beta,
         float *state, void *output, int batch, int tokens, int kh, int vh, int stride, void *workspace,
         size_t workspaceBytes, void *stream) {
    size_t required = Bytes(batch, tokens, kh, vh);
    if (!required || !query || !key || !value || !g || !beta || !state || !output || !workspace ||
        workspaceBytes < required || (reinterpret_cast<uintptr_t>(workspace) & 127) ||
        (reinterpret_cast<uintptr_t>(query) & 1) || (reinterpret_cast<uintptr_t>(key) & 1) ||
        (reinterpret_cast<uintptr_t>(value) & 1) || (reinterpret_cast<uintptr_t>(beta) & 1) ||
        (reinterpret_cast<uintptr_t>(g) & 3) || (reinterpret_cast<uintptr_t>(state) & 7) ||
        (reinterpret_cast<uintptr_t>(output) & 3) || stride < 0 || (stride && stride < vh * D))
        return false;
    cudaStream_t s = Stream(stream);
    int bh = batch * vh, chunks = (tokens + C - 1) / C, groups = bh * chunks;
    Workspace w(workspace, batch, tokens, kh, vh);
    size_t qk = (size_t)batch * tokens * kh * D;
    CloverPrepareCausalGdn<<<dim3(kh, tokens, batch), 32, 0, s>>>(
        (const __nv_bfloat16 *)query, (const __nv_bfloat16 *)key, g, w.prepared, w.prepared + qk,
        w.prepared + 2 * qk, tokens, kh, vh, stride ? stride : kh * D);
    if (cudaGetLastError() != cudaSuccess)
        return false;
    Prepare<<<dim3(chunks, bh), 256, 0, s>>>(w.prepared, (const __nv_bfloat16 *)value, g,
                                             (const __nv_bfloat16 *)beta, w, batch, tokens, kh, vh,
                                             stride ? stride : vh * D);
    if (cudaGetLastError() != cudaSuccess)
        return false;
    FusedPairs<<<groups, 128, PairSharedBytes, s>>>(w);
    if (cudaGetLastError() != cudaSuccess)
        return false;
    Inverse<<<groups, 128, 3 * C * InverseStride * sizeof(float), s>>>(w);
    if (cudaGetLastError() != cudaSuccess)
        return false;
    RecomputeWU<<<dim3(4, groups), 128, 0, s>>>(w);
    if (cudaGetLastError() != cudaSuccess)
        return false;
    Advance<<<dim3(4, bh), 128, AdvanceSharedBytes, s>>>(w, state, bh, chunks);
    if (cudaGetLastError() != cudaSuccess)
        return false;
    FusedOutput<<<dim3(2, groups), 128, OutputSharedBytes, s>>>(w, (__nv_bfloat16 *)output, tokens, vh, bh);
    return cudaGetLastError() == cudaSuccess;
}
} // namespace clover_prefill
} // namespace
#endif
