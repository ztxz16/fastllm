#include <thread>
#include <tuple>
// src/kernels/cuda/exl3_legacy.cu - v1.1.8's EXL3 decode kernels, kept for NVIDIA GPUs before sm_80 (V100, T4,
// RTX 20): #180's decode kernels run the tiles on the tensor cores from sm_80 on, and their CUDA-core fallback was
// slower here than these (a V100 with EXL3-3.05bpw: decode 4.9 -> 4.4 tok/s).  exl3.cu's mv / moe_gate_up / moe_down
// / win_experts / prepare / several_ok select these by GPU architecture; the prompt path is exl3.cu's on every GPU.  The code below is v1.1.8's, the prompt path's kernels left out
// and decode8's trellis windows cut eight at a time (as exl3.cu's frag8) - the same values.
//
// The codebooks and the trellis window arithmetic follow exllamav3 (MIT, turboderp - exllamav3_ext/quant/
// codebook.cuh, exl3_dq.cuh, hadamard_inner.cuh); the kernels around them are shaped for this engine's decode:
//
//   * a block owns 128 outputs (8 tiles - exactly one output rotation block) over a slice of at most 4096 inputs;
//     its 8 warps take the slice's 16-row tile rows in turn, each staging the row's 8 tiles in shared memory with
//     coalesced loads and decoding 8 weights per lane (one fragment: 2 columns x 4 rows), F32 accumulation;
//   * the input rotation runs per block on the slice it reads (F32 or q8_1 in, FP16 in shared memory), so the
//     engine's activation buffers are used as they are;
//   * a matrix with more than 4096 inputs, or too few output blocks to fill the device, splits its inputs over
//     blocks: the partial sums go to a per-stream scratch and the LAST block of an output block adds them up in
//     split order (deterministic) and runs the output rotation.
#if !defined(STRATA_USE_HIP)   // (AMD runs exl3.cu's kernels; EXL3 does not run there anyway)
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/exl3_io.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "strata/kernels/dp4a.hpp"   // sm_60: __dp4a -> the vmad fallback

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <utility>

namespace strata::kernels::exl3::legacy {
namespace {

constexpr int kThreads = 256, kWarps = 8;
constexpr int kCols = 128;                // outputs per block: 8 tiles, one rotation block
constexpr int kMaxKR = 4096;              // inputs per block (the rotated slice lives in shared memory)
constexpr float kHad = 0.088388347648f;   // 1 / sqrt(128)

struct Q81 {   // the engine's q8_1 block (ggml block_q8_1): d, sum, 32 int8
    __half2 ds;
    int8_t qs[32];
};

std::atomic<int> g_launch_errors{0};
void launch_check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "exl3 %s: %s\n", what, cudaGetErrorString(e));
        g_launch_errors.fetch_add(1, std::memory_order_relaxed);
    }
}

// ---------------------------------------------------------------- the codebooks
__device__ __forceinline__ uint32_t bytesum_plus(uint32_t x, uint32_t acc) {
#if defined(STRATA_USE_HIP)
    return acc + (x & 0xffu) + ((x >> 8) & 0xffu) + ((x >> 16) & 0xffu) + (x >> 24);
#else
    return __dp4a(x, 0x01010101u, acc);
#endif
}

// two states -> two values, FP16 arithmetic as exllamav3's decode_3inst_2 (the tensor-core kernels take the half2)
template <int cb>
__device__ __forceinline__ __half2 decode2h(uint32_t s0, uint32_t s1) {
    if constexpr (cb == 2) {
        const uint32_t a = bytesum_plus(s0 * 0x83DCD12Du, 0x6400u);
        const uint32_t b = bytesum_plus(s1 * 0x83DCD12Du, 0x6400u);
        const __half2 h = __halves2half2(__ushort_as_half((unsigned short) a), __ushort_as_half((unsigned short) b));
        return __hfma2(h, __half2half2(__ushort_as_half(0x1eee)), __half2half2(__ushort_as_half(0xc931)));
    } else {
        uint32_t x0 = cb == 1 ? s0 * 0xCBAC1FEDu : s0 * 89226354u + 64248484u;
        uint32_t x1 = cb == 1 ? s1 * 0xCBAC1FEDu : s1 * 89226354u + 64248484u;
        x0 = (x0 & 0x8fff8fffu) ^ 0x3b603b60u;
        x1 = (x1 & 0x8fff8fffu) ^ 0x3b603b60u;
        const __half2 lo = __halves2half2(__ushort_as_half((unsigned short) (x0 & 0xffffu)),
                                          __ushort_as_half((unsigned short) (x1 & 0xffffu)));
        const __half2 hi = __halves2half2(__ushort_as_half((unsigned short) (x0 >> 16)),
                                          __ushort_as_half((unsigned short) (x1 >> 16)));
        return __hadd2(lo, hi);
    }
}
template <int cb>
__device__ __forceinline__ float2 decode2(uint32_t s0, uint32_t s1) {
    return __half22float2(decode2h<cb>(s0, s1));
}

// The lane's 8 steps (lane * 8 + j): v[j] sits at tile row (lane & 3) * 2 + (j & 1) + 8 (j >> 1 & 1), column
// (lane >> 2) + 8 (j >> 2).  Step p's state is the 16-bit window ending at bit S(p) = floor((p + 1) K2 / 2) of the
// tile's MSB-first bit stream (w: its 4 K2 words), wrapping around the tile.  v1.1.8 cut every step's window on its
// own (two shared-memory words and a 64-bit shift a step); the lane's 8 windows are cut together here, from one or two
// words with funnel shifts, as exl3.cu's frag8 (exllamav3's grouping per K, exl3_dq.cuh) - the same states, so the
// same values bit for bit (v1.1.8's per-step windows were most of the ~10 instructions a weight here; a V100 with
// EXL3-2.05bpw: the routed gate/up 237 -> 179 us, down 85 -> 68 us)
__device__ __forceinline__ uint32_t fsh(uint32_t lo, uint32_t hi, int s) {
    return (uint32_t) ((((uint64_t) hi << 32) | (uint64_t) lo) >> s);
}
__device__ __forceinline__ void states4(uint32_t a, uint32_t b, uint32_t (&s)[8]) {
    const uint32_t x = fsh(b, a, 20);
    s[0] = x >> 8;
    s[1] = x >> 4;
    s[2] = x;
    s[3] = b >> 16;
    s[4] = b >> 12;
    s[5] = b >> 8;
    s[6] = b >> 4;
    s[7] = b;
}
template <int K, int G>
__device__ __forceinline__ void states_group(const uint32_t* w, int p, uint32_t* s) {
    constexpr int NW = 8 * K;
    const int b0 = (p + 257) * K - 16, b2 = b0 + (G - 1) * K + 16;
    const int i0 = b0 >> 5, i2 = (b2 - 1) >> 5, s2 = ((i2 + 1) << 5) - b2;
    const uint32_t a = w[i0 % NW], b = w[i2 % NW];
#pragma unroll
    for (int j = 0; j < G; ++j) s[G - 1 - j] = fsh(b, a, s2 + j * K);
}
template <int K>
__device__ __forceinline__ void states8_int(const uint32_t* w, int lane, uint32_t (&s)[8]) {
    const int p = lane * 8;
    if constexpr (K == 4) {
        states4(w[(lane + 31) & 31], w[lane], s);
    } else if constexpr (K == 1 || K == 2) {
        constexpr int NW = 8 * K, per = 32 / (8 * K);
        const int i1 = lane / per, i0 = (i1 + NW - 1) % NW;
        const uint32_t b = fsh(w[i1], w[i0], (per - 1 - lane % per) * 8 * K);
#pragma unroll
        for (int j = 0; j < 8; ++j) s[j] = b >> ((7 - j) * K);
    } else if constexpr (K == 3) {
        constexpr int NW = 24;
        const int b1 = (p + 257) * 3, b0 = b1 - 16, b2 = b1 + 21;
        const int i0 = b0 >> 5, i2 = (b2 - 1) >> 5, s2 = ((i2 + 1) << 5) - b2;
        const uint32_t a = w[i0 % NW], b = w[i2 % NW];
        const uint32_t hi = fsh(b, a, s2), lo = fsh(b, a, s2 + 12);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            s[7 - j] = hi >> (3 * j);
            s[3 - j] = lo >> (3 * j);
        }
    } else if constexpr (K == 7) {
#pragma unroll
        for (int g = 0; g < 4; ++g) states_group<K, 2>(w, p + 2 * g, s + 2 * g);
    } else {   // 5, 6, 8
        states_group<K, 4>(w, p, s);
        states_group<K, 4>(w, p + 4, s + 4);
    }
}
template <int K2>
__device__ __forceinline__ void states8_half(const uint32_t* w, int lane, uint32_t (&s)[8]) {
    constexpr int KA = K2 / 2, NW = 4 * K2, span = 18 + 3 * KA;
    const int e7 = (lane * 4 + 4) * K2 + 128 * K2;
#pragma unroll
    for (int g = 0; g < 2; ++g) {
        const int e = e7 - (1 - g) * 2 * K2;
        const int hi = (e - 1) >> 5, lo = (e - span) >> 5, sh = ((hi + 1) << 5) - e;
        const uint32_t x = fsh(w[hi % NW], w[lo % NW], sh);
        s[4 * g + 3] = x;
        s[4 * g + 2] = x >> (KA + 1);
        s[4 * g + 1] = x >> (2 * KA + 1);
        s[4 * g + 0] = x >> (3 * KA + 2);
    }
}
template <int K2, int cb>
__device__ __forceinline__ void decode8(const uint32_t* w, int lane, float* v) {
    uint32_t s[8];
    if constexpr ((K2 & 1) != 0) states8_half<K2>(w, lane, s);
    else states8_int<K2 / 2>(w, lane, s);
#pragma unroll
    for (int j = 0; j < 8; j += 2) {
        const float2 d = decode2<cb>(s[j] & 0xffffu, s[j + 1] & 0xffffu);
        v[j] = d.x;
        v[j + 1] = d.y;
    }
}

// ---------------------------------------------------------------- the rotation
// the 128-point transform of the 128 values a warp holds 4 per lane (value lane * 4 + i), over sqrt(128)
__device__ __forceinline__ void had128(float* v, int lane) {
    const float s0 = v[0] + v[1], d0 = v[0] - v[1], s1 = v[2] + v[3], d1 = v[2] - v[3];
    v[0] = s0 + s1;
    v[1] = d0 + d1;
    v[2] = s0 - s1;
    v[3] = d0 - d1;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float o = __shfl_xor_sync(0xffffffffu, v[i], m);
            v[i] = (lane & m) ? o - v[i] : v[i] + o;
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] *= kHad;
}

__device__ __forceinline__ float h2f(uint16_t b) { return __half2float(__ushort_as_half(b)); }

// xs[i] = H(x * suh)[k0 + i] for i < kr (kr % 128 == 0); x from xf (F32) or xq (q8_1); every warp of the block
__device__ void rotate_in(const float* __restrict__ xf, const Q81* __restrict__ xq, const uint16_t* __restrict__ suh,
                          int k0, int kr, __half* xs,
                          const uint16_t* x16 = nullptr, bool x_bf16 = false) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int c = warp; c < kr / 128; c += kWarps) {
        const int base = k0 + c * 128 + lane * 4;
        float v[4];
        if (x16 != nullptr) {
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = io_load16(x16[base + i], x_bf16);
        } else if (xf != nullptr) {
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = xf[base + i];
        } else {
            const Q81* b = xq + base / 32;
            const float d = __low2float(b->ds);
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = d * (float) b->qs[base % 32 + i];
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] *= h2f(suh[base + i]);
        had128(v, lane);
#pragma unroll
        for (int i = 0; i < 4; ++i) xs[c * 128 + lane * 4 + i] = __float2half_rn(v[i]);
    }
}

// ---------------------------------------------------------------- the block GEMV
// a lane's 4 weights of one column times the activation's 4 inputs, added to a: the rounding spelled out, so the
// one-activation and the several-activation GEMVs (block_gemv, block_gemv_n) give each activation the same bits
__device__ __forceinline__ float dot4_acc(float a, const float* v, float x0, float x1, float x2, float x3) {
    return __fadd_rn(a, __fmaf_rn(v[3], x3, __fmaf_rn(v[2], x2, __fmaf_rn(v[1], x1, __fmul_rn(v[0], x0)))));
}

// red[c] (c < 128) = sum over the 16-row slices [ks0, ks1) of x' . W'[:, tc0 * 16 + c], x' = xs from slice ks0 on.
// tr: the matrix's first tile, ld tiles per slice row.  stage: kWarps * 32 K2 words, red: kWarps * 128 floats.
// Ends with a __syncthreads (red[0..127] complete).
template <int K2, int cb>
__device__ void block_gemv(const uint32_t* __restrict__ tr, int ld, int tc0, int ks0, int ks1, const __half* xs,
                           uint32_t* stage, float* red) {
    constexpr int W = 4 * K2, TW = 8 * W;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    uint32_t* st = stage + warp * TW;
    float acc[8][2];
#pragma unroll
    for (int t = 0; t < 8; ++t) acc[t][0] = acc[t][1] = 0.0f;
    const int kk = (lane & 3) * 2;
    for (int ks = ks0 + warp; ks < ks1; ks += kWarps) {
        const uint32_t* src = tr + ((size_t) ks * ld + tc0) * W;
        __syncwarp();
#pragma unroll
        for (int i = lane; i < TW; i += 32) st[i] = src[i];
        __syncwarp();
        const __half* xk = xs + (ks - ks0) * 16;
        const float x0 = __half2float(xk[kk]), x1 = __half2float(xk[kk + 1]);
        const float x2 = __half2float(xk[kk + 8]), x3 = __half2float(xk[kk + 9]);
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            float v[8];
            decode8<K2, cb>(st + t * W, lane, v);
            acc[t][0] = dot4_acc(acc[t][0], v, x0, x1, x2, x3);
            acc[t][1] = dot4_acc(acc[t][1], v + 4, x0, x1, x2, x3);
        }
    }
    // the 4 lanes of one lane / 4 hold the same two columns of every tile
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            acc[t][c] += __shfl_xor_sync(0xffffffffu, acc[t][c], 1);
            acc[t][c] += __shfl_xor_sync(0xffffffffu, acc[t][c], 2);
        }
    if ((lane & 3) == 0)
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            red[warp * kCols + t * 16 + (lane >> 2)] = acc[t][0];
            red[warp * kCols + t * 16 + 8 + (lane >> 2)] = acc[t][1];
        }
    __syncthreads();
    if (threadIdx.x < kCols) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) s += red[w * kCols + threadIdx.x];
        red[threadIdx.x] = s;
    }
    __syncthreads();
}

// block_gemv for nc activations at once (nc <= kNC; activation c's rotated slice at xs + c * kMaxKR): each lane decodes
// its 8 weights of a tile once and adds them into every activation's sums - as exllamav3's small-m GEMV and fused MoE
// kernels share a tile's decode among the rows of an MMA, but with block_gemv's own F32 arithmetic, so outc[c * 128 +
// i] (i < 128) is bit for bit block_gemv's red[i] for activation c alone.  Ends with a __syncthreads.
constexpr int kNC = 4;
template <int K2, int cb>
__device__ void block_gemv_n(const uint32_t* __restrict__ tr, int ld, int tc0, int ks0, int ks1, const __half* xs,
                             int nc, uint32_t* stage, float* red, float* outc) {
    constexpr int W = 4 * K2, TW = 8 * W;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    uint32_t* st = stage + warp * TW;
    float acc[kNC][8][2];
#pragma unroll
    for (int c = 0; c < kNC; ++c)
#pragma unroll
        for (int t = 0; t < 8; ++t) acc[c][t][0] = acc[c][t][1] = 0.0f;
    const int kk = (lane & 3) * 2;
    for (int ks = ks0 + warp; ks < ks1; ks += kWarps) {
        const uint32_t* src = tr + ((size_t) ks * ld + tc0) * W;
        __syncwarp();
#pragma unroll
        for (int i = lane; i < TW; i += 32) st[i] = src[i];
        __syncwarp();
        float x[kNC][4];
#pragma unroll
        for (int c = 0; c < kNC; ++c) {
            const __half* xk = xs + c * kMaxKR + (ks - ks0) * 16;
            if (c < nc) {
                x[c][0] = __half2float(xk[kk]);
                x[c][1] = __half2float(xk[kk + 1]);
                x[c][2] = __half2float(xk[kk + 8]);
                x[c][3] = __half2float(xk[kk + 9]);
            }
        }
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            float v[8];
            decode8<K2, cb>(st + t * W, lane, v);
#pragma unroll
            for (int c = 0; c < kNC; ++c)
                if (c < nc) {
                    acc[c][t][0] = dot4_acc(acc[c][t][0], v, x[c][0], x[c][1], x[c][2], x[c][3]);
                    acc[c][t][1] = dot4_acc(acc[c][t][1], v + 4, x[c][0], x[c][1], x[c][2], x[c][3]);
                }
        }
    }
    // per activation, block_gemv's reduction
#pragma unroll
    for (int c = 0; c < kNC; ++c) {
        if (c >= nc) break;
#pragma unroll
        for (int t = 0; t < 8; ++t)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                acc[c][t][h] += __shfl_xor_sync(0xffffffffu, acc[c][t][h], 1);
                acc[c][t][h] += __shfl_xor_sync(0xffffffffu, acc[c][t][h], 2);
            }
        if ((lane & 3) == 0)
#pragma unroll
            for (int t = 0; t < 8; ++t) {
                red[warp * kCols + t * 16 + (lane >> 2)] = acc[c][t][0];
                red[warp * kCols + t * 16 + 8 + (lane >> 2)] = acc[c][t][1];
            }
        __syncthreads();
        if (threadIdx.x < kCols) {
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < kWarps; ++w) s += red[w * kCols + threadIdx.x];
            outc[c * kCols + threadIdx.x] = s;
        }
        __syncthreads();
    }
}

// one warp: y[c] = alpha * (H(red) * svh)[c] + bias[c] for the block's 128 outputs
__device__ __forceinline__ void epilogue(const float* red, const uint16_t* svh, float alpha, const float* bias,
                                         float* y, uint16_t* y16, bool y_bf16) {
    const int lane = threadIdx.x & 31;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = red[lane * 4 + i];
    had128(v, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c = lane * 4 + i;
        const float value = __fmaf_rn(__fmul_rn(alpha, v[i]), h2f(svh[c]), bias != nullptr ? bias[c] : 0.0f);
        if (y16 != nullptr) y16[c] = io_store16(value, y_bf16);
        else y[c] = value;
    }
}

// ---------------------------------------------------------------- mv
struct KJob {
    const uint32_t* tr;
    const uint16_t* suh;
    const uint16_t* svh;
    const float* xf;
    const Q81* xq;
    float* y;
    const uint16_t* x16;
    uint16_t* y16;
    bool x_bf16, y_bf16;
    const float* bias;
    float* part;       // S x n partial sums (S > 1); mv_n_kernel: nc x S x n
    unsigned* cnt;     // n / 128 counters, zero between launches
    float alpha;
    int k, n, ld, S, kr;
    int nc;            // mv_n_kernel: the activations (1..kNC), xf / xq / y advanced by these per activation
    size_t xf_cs, xq_cs, y_cs;   // (floats, bytes, floats)
};
struct KBatch {
    KJob j[kMaxMvJobs];
    int blk_end[kMaxMvJobs];
    int n;
};

// the dynamic shared memory of the several-activation kernels (mv_n_kernel, win_gu_kernel, win_down_kernel):
// red[kWarps][128] outc[kNC][128] gu[2][kNC][128] (F32), stage[kWarps * 32 K2] (words), xs[kNC][kMaxKR] (FP16) - up to
// 58 KB, past the 48 KB a launch gets without asking (set_smem)
constexpr size_t smem_n_bytes(int K2) {
    return (size_t) (kWarps * kCols + kNC * kCols + 2 * kNC * kCols) * 4 + (size_t) kWarps * 32 * K2 * 4 +
           (size_t) kNC * kMaxKR * 2;
}
struct SmemN {
    float *red, *outc, *gu;
    uint32_t* stage;
    __half* xs;
};
template <int K2>
__device__ __forceinline__ SmemN smem_n(unsigned char* p) {
    SmemN s;
    s.red = (float*) p;
    s.outc = s.red + kWarps * kCols;
    s.gu = s.outc + kNC * kCols;
    s.stage = (uint32_t*) (s.gu + 2 * kNC * kCols);
    s.xs = (__half*) (s.stage + kWarps * 32 * K2);
    return s;
}

template <int K2, int cb>
__global__ void __launch_bounds__(kThreads) mv_kernel(const KBatch b) {
    __shared__ __half xs[kMaxKR];
    __shared__ uint32_t stage[kWarps * 32 * K2];
    __shared__ float red[kWarps * kCols];
    __shared__ int last;
    int ji = 0;
    while ((int) blockIdx.x >= b.blk_end[ji]) ++ji;
    const KJob& J = b.j[ji];
    const int local = (int) blockIdx.x - (ji > 0 ? b.blk_end[ji - 1] : 0);
    const int g = local / J.S, sp = local % J.S;
    const int k0 = sp * J.kr, k1 = min(J.k, k0 + J.kr);
    rotate_in(J.xf, J.xq, J.suh, k0, k1 - k0, xs, J.x16, J.x_bf16);
    __syncthreads();
    block_gemv<K2, cb>(J.tr, J.ld, g * 8, k0 / 16, k1 / 16, xs, stage, red);
    const int warp = threadIdx.x >> 5;
    if (J.S == 1) {
        if (warp == 0)
            epilogue(red, J.svh + g * kCols, J.alpha, J.bias != nullptr ? J.bias + g * kCols : nullptr,
                     J.y != nullptr ? J.y + g * kCols : nullptr,
                     J.y16 != nullptr ? J.y16 + g * kCols : nullptr, J.y_bf16);
        return;
    }
    if (threadIdx.x < kCols) J.part[(size_t) sp * J.n + g * kCols + threadIdx.x] = red[threadIdx.x];
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(&J.cnt[g], 1u) == (unsigned) (J.S - 1);
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (threadIdx.x < kCols) {
        float s = 0.0f;
        for (int i = 0; i < J.S; ++i) s += *(volatile const float*) &J.part[(size_t) i * J.n + g * kCols + threadIdx.x];
        red[threadIdx.x] = s;
    }
    __syncthreads();
    if (warp == 0)
        epilogue(red, J.svh + g * kCols, J.alpha, J.bias != nullptr ? J.bias + g * kCols : nullptr, J.y != nullptr ? J.y + g * kCols : nullptr,
                     J.y16 != nullptr ? J.y16 + g * kCols : nullptr, J.y_bf16);
    if (threadIdx.x == 0) J.cnt[g] = 0u;
}

// mv_kernel for a job's nc activations at once (a verify window's tokens): the same input split, the weights read and
// decoded once for all of them (block_gemv_n), each activation's partials and sums in mv_kernel's order - every
// activation's y is mv_kernel's for it alone, bit for bit
template <int K2, int cb>
__global__ void __launch_bounds__(kThreads) mv_n_kernel(const KBatch b) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    const SmemN sm = smem_n<K2>(smem_raw);
    __shared__ int last;
    int ji = 0;
    while ((int) blockIdx.x >= b.blk_end[ji]) ++ji;
    const KJob& J = b.j[ji];
    const int local = (int) blockIdx.x - (ji > 0 ? b.blk_end[ji - 1] : 0);
    const int g = local / J.S, sp = local % J.S;
    const int k0 = sp * J.kr, k1 = min(J.k, k0 + J.kr);
    for (int c = 0; c < J.nc; ++c)
        rotate_in(J.xf != nullptr ? J.xf + c * J.xf_cs : nullptr,
                  J.xq != nullptr ? (const Q81*) ((const uint8_t*) J.xq + c * J.xq_cs) : nullptr, J.suh, k0, k1 - k0,
                  sm.xs + c * kMaxKR, J.x16 != nullptr ? J.x16 + c * J.xf_cs : nullptr, J.x_bf16);
    __syncthreads();
    block_gemv_n<K2, cb>(J.tr, J.ld, g * 8, k0 / 16, k1 / 16, sm.xs, J.nc, sm.stage, sm.red, sm.outc);
    const int warp = threadIdx.x >> 5;
    const float* bias = J.bias != nullptr ? J.bias + g * kCols : nullptr;
    if (J.S == 1) {
        if (warp < J.nc)
            epilogue(sm.outc + warp * kCols, J.svh + g * kCols, J.alpha, bias, J.y != nullptr ? J.y + warp * J.y_cs + g * kCols : nullptr,
                     J.y16 != nullptr ? J.y16 + warp * J.y_cs + g * kCols : nullptr, J.y_bf16);
        return;
    }
    if (threadIdx.x < kCols)
        for (int c = 0; c < J.nc; ++c)
            J.part[((size_t) c * J.S + sp) * J.n + g * kCols + threadIdx.x] = sm.outc[c * kCols + threadIdx.x];
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(&J.cnt[g], 1u) == (unsigned) (J.S - 1);
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (threadIdx.x < kCols)
        for (int c = 0; c < J.nc; ++c) {
            float s = 0.0f;
            for (int i = 0; i < J.S; ++i)
                s += *(volatile const float*) &J.part[((size_t) c * J.S + i) * J.n + g * kCols + threadIdx.x];
            sm.outc[c * kCols + threadIdx.x] = s;
        }
    __syncthreads();
    if (warp < J.nc) epilogue(sm.outc + warp * kCols, J.svh + g * kCols, J.alpha, bias, J.y != nullptr ? J.y + warp * J.y_cs + g * kCols : nullptr,
                     J.y16 != nullptr ? J.y16 + warp * J.y_cs + g * kCols : nullptr, J.y_bf16);
    if (threadIdx.x == 0) J.cnt[g] = 0u;
}

// ---------------------------------------------------------------- the routed experts
struct KMoe {
    const unsigned long long* plan_ptr;
    const float* plan_w;
    const float* cpu_part;
    const int* cpu_flag;
    const float* x;
    const uint16_t* x16;
    bool x_bf16;
    float* h;            // [k][n_ff]
    const float* sh_out;
    float* out;
    float* part;         // [k][n_embd]
    unsigned* cnt;       // n_embd / 128
    size_t suh[3], svh[3], tr[3];
    int k, n_embd, n_ff;
    float limit;
};

// the pieces the token's experts (moe_*_kernel) and a window's grouped experts (win_*_kernel) share, so each window
// token gets the token path's bits: one warp's output block of 128 (H, then svh) ...
__device__ __forceinline__ void block_out(const float* red, const uint16_t* svh, float* dst) {
    const int lane = threadIdx.x & 31;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = red[lane * 4 + i];
    had128(v, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = __fmul_rn(v[i], h2f(svh[lane * 4 + i]));
}
// ... the clamped swiglu ...
__device__ __forceinline__ float swiglu(float gv, float uv, float limit) {
    if (limit > 0.0f) {
        gv = fminf(gv, limit);
        uv = fminf(fmaxf(uv, -limit), limit);
    }
    return __fmul_rn(__fdiv_rn(gv, __fadd_rn(1.0f, expf(-gv))), uv);
}
// ... and one output's sum: the experts' rows in plan order (p[i]: expert i's, 0 for one not on the device), the CPU
// lane's part when flagged (cpu_part: null when not), the shared expert's row (sh: 0 without one)
__device__ __forceinline__ float combine(const float* plan_w, const float* p, int k, const float* cpu_part, float sh) {
    float m = 0.0f;
    for (int i = 0; i < k; ++i) m = __fmaf_rn(plan_w[i], p[i], m);
    if (cpu_part != nullptr) m = __fadd_rn(m, *cpu_part);
    return __fadd_rn(m, sh);
}

// grid (n_ff / 128, k): expert blockIdx.y's gate and up for 128 h values, then the clamped swiglu
template <int K2, int cb>
__global__ void __launch_bounds__(kThreads) moe_gate_up_kernel(const KMoe a) {
    __shared__ __half xs[2][kMaxKR];
    __shared__ uint32_t stage[kWarps * 32 * K2];
    __shared__ float red[kWarps * kCols];
    __shared__ float gu[2][kCols];
    const int ei = blockIdx.y, g = blockIdx.x;
    const unsigned long long ptr = a.plan_ptr[ei];
    if (ptr == 0ull) return;
    const uint8_t* blob = (const uint8_t*) ptr;
    const int warp = threadIdx.x >> 5;
    for (int r = 0; r < 2; ++r) rotate_in(a.x, nullptr, (const uint16_t*) (blob + a.suh[r]), 0, a.n_embd, xs[r], a.x16, a.x_bf16);
    __syncthreads();
    for (int r = 0; r < 2; ++r) {
        block_gemv<K2, cb>((const uint32_t*) (blob + a.tr[r]), a.n_ff / 16, g * 8, 0, a.n_embd / 16, xs[r], stage, red);
        if (warp == 0) block_out(red, (const uint16_t*) (blob + a.svh[r]) + g * kCols, gu[r]);
        __syncthreads();
    }
    if (threadIdx.x < kCols)
        a.h[(size_t) ei * a.n_ff + g * kCols + threadIdx.x] = swiglu(gu[0][threadIdx.x], gu[1][threadIdx.x], a.limit);
}

// moe_gate_up_kernel with gate and up in blocks of their own (grid (n_ff / 128, k, 2), z: gate / up): a token's ~5
// VRAM experts gave the V100's 80 SMs 80 blocks, each through 2 x 4096 inputs in turn (179 us against moe_down's 68
// for as many bytes).  Each block's product and block_out are moe_gate_up_kernel's; it leaves its 128 values in part
// ([k][2][n_ff], the stream's scratch - live only within the kernel), and the second of an (expert, output block)'s
// pair to finish takes both through the swiglu - the same bits; moe_gate_up_kernel handles limited scratch.
template <int K2, int cb>
__global__ void __launch_bounds__(kThreads) moe_gate_up2_kernel(const KMoe a) {
    __shared__ __half xs[kMaxKR];
    __shared__ uint32_t stage[kWarps * 32 * K2];
    __shared__ float red[kWarps * kCols];
    __shared__ int last;
    const int ei = blockIdx.y, g = blockIdx.x, r = blockIdx.z;
    const unsigned long long ptr = a.plan_ptr[ei];
    if (ptr == 0ull) return;
    const uint8_t* blob = (const uint8_t*) ptr;
    const int warp = threadIdx.x >> 5;
    rotate_in(a.x, nullptr, (const uint16_t*) (blob + a.suh[r]), 0, a.n_embd, xs, a.x16, a.x_bf16);
    __syncthreads();
    block_gemv<K2, cb>((const uint32_t*) (blob + a.tr[r]), a.n_ff / 16, g * 8, 0, a.n_embd / 16, xs, stage, red);
    float* gu = a.part + ((size_t) ei * 2) * a.n_ff + g * kCols;
    if (warp == 0) block_out(red, (const uint16_t*) (blob + a.svh[r]) + g * kCols, gu + (size_t) r * a.n_ff);
    __threadfence();
    __syncthreads();
    unsigned* cnt = a.cnt + (size_t) ei * (a.n_ff / kCols) + g;
    if (threadIdx.x == 0) last = atomicAdd(cnt, 1u) == 1u;
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (threadIdx.x < kCols)
        a.h[(size_t) ei * a.n_ff + g * kCols + threadIdx.x] =
            swiglu(*(volatile const float*) &gu[threadIdx.x], *(volatile const float*) &gu[a.n_ff + threadIdx.x], a.limit);
    if (threadIdx.x == 0) *cnt = 0u;
}

// grid (n_embd / 128, k): expert blockIdx.y's down rows for 128 outputs; the last block of each output block adds
// the experts up in plan order (as the engine's moe_down), then the CPU lane's part and the shared expert
template <int K2, int cb>
__global__ void __launch_bounds__(kThreads) moe_down_kernel(const KMoe a) {
    __shared__ __half xs[kMaxKR];
    __shared__ uint32_t stage[kWarps * 32 * K2];
    __shared__ float red[kWarps * kCols];
    __shared__ int last;
    const int ei = blockIdx.y, g = blockIdx.x;
    const unsigned long long ptr = a.plan_ptr[ei];
    const int warp = threadIdx.x >> 5;
    float* part = a.part + (size_t) ei * a.n_embd + g * kCols;
    if (ptr != 0ull) {
        const uint8_t* blob = (const uint8_t*) ptr;
        rotate_in(a.h + (size_t) ei * a.n_ff, nullptr, (const uint16_t*) (blob + a.suh[2]), 0, a.n_ff, xs);
        __syncthreads();
        block_gemv<K2, cb>((const uint32_t*) (blob + a.tr[2]), a.n_embd / 16, g * 8, 0, a.n_ff / 16, xs, stage, red);
        if (warp == 0) block_out(red, (const uint16_t*) (blob + a.svh[2]) + g * kCols, part);
    } else if (threadIdx.x < kCols) {
        part[threadIdx.x] = 0.0f;
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(&a.cnt[g], 1u) == (unsigned) (a.k - 1);
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (threadIdx.x < kCols) {
        const int r = g * kCols + threadIdx.x;
        float p[8];
        for (int i = 0; i < a.k; ++i) p[i] = *(volatile const float*) &a.part[(size_t) i * a.n_embd + r];
        a.out[r] = combine(a.plan_w, p, a.k, a.cpu_flag != nullptr && *a.cpu_flag ? a.cpu_part + r : nullptr,
                           a.sh_out != nullptr ? a.sh_out[r] : 0.0f);
    }
    if (threadIdx.x == 0) a.cnt[g] = 0u;
}

// ---------------------------------------------------------------- a verify window's routed experts
// gf::moe_win_experts' grouping (group g: one expert, its blob at grp_ptr[g]; its entries e = t * 8 + i at
// grp_ent[grp_start[g] ..)), each group's weights read and decoded once for up to kNC of its tokens at a time (as
// exllamav3's fused MoE runs a run of slots that picked one expert as the rows of one MMA tile), with the token
// path's arithmetic per entry: every token's output is moe_gate_up + moe_down's for it alone, bit for bit
struct KWin {
    const unsigned long long* grp_ptr;
    const int *grp_start, *grp_ent, *n_groups;
    const unsigned long long* plan_ptr;
    const float* plan_w;
    const float* cpu_part;
    const int* cpu_flag;
    const float* x;
    size_t x_stride;
    float* h;            // [T * 8][n_ff]
    float* part;         // [T * 8][n_embd]
    const float* sh_out; // [T][n_embd]
    float* out;
    size_t out_stride;
    size_t suh[3], svh[3], tr[3];
    int k, n_embd, n_ff;
    float limit;
};

// grid (n_ff / 128, groups at most): group blockIdx.y's gate and up for 128 h values of each entry, the clamped swiglu
template <int K2, int cb>
__global__ void __launch_bounds__(kThreads) win_gu_kernel(const KWin a) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    const SmemN sm = smem_n<K2>(smem_raw);
    const int grp = blockIdx.y, g = blockIdx.x;
    if (grp >= *a.n_groups) return;
    const uint8_t* blob = (const uint8_t*) a.grp_ptr[grp];
    const int warp = threadIdx.x >> 5;
    const int m1 = a.grp_start[grp + 1];
    for (int m0 = a.grp_start[grp]; m0 < m1; m0 += kNC) {
        const int nc = min(kNC, m1 - m0);
        for (int r = 0; r < 2; ++r) {
            for (int c = 0; c < nc; ++c)
                rotate_in(a.x + (size_t) (a.grp_ent[m0 + c] >> 3) * a.x_stride, nullptr,
                          (const uint16_t*) (blob + a.suh[r]), 0, a.n_embd, sm.xs + c * kMaxKR);
            __syncthreads();
            block_gemv_n<K2, cb>((const uint32_t*) (blob + a.tr[r]), a.n_ff / 16, g * 8, 0, a.n_embd / 16, sm.xs, nc,
                                sm.stage, sm.red, sm.outc);
            if (warp < nc)
                block_out(sm.outc + warp * kCols, (const uint16_t*) (blob + a.svh[r]) + g * kCols,
                          sm.gu + (r * kNC + warp) * kCols);
            __syncthreads();
        }
        if (threadIdx.x < kCols)
            for (int c = 0; c < nc; ++c)
                a.h[(size_t) a.grp_ent[m0 + c] * a.n_ff + g * kCols + threadIdx.x] =
                    swiglu(sm.gu[c * kCols + threadIdx.x], sm.gu[(kNC + c) * kCols + threadIdx.x], a.limit);
        __syncthreads();
    }
}

// grid (n_embd / 128, groups at most): group blockIdx.y's down rows for 128 outputs of each entry into its part rows
template <int K2, int cb>
__global__ void __launch_bounds__(kThreads) win_down_kernel(const KWin a) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    const SmemN sm = smem_n<K2>(smem_raw);
    const int grp = blockIdx.y, g = blockIdx.x;
    if (grp >= *a.n_groups) return;
    const uint8_t* blob = (const uint8_t*) a.grp_ptr[grp];
    const int warp = threadIdx.x >> 5;
    const int m1 = a.grp_start[grp + 1];
    for (int m0 = a.grp_start[grp]; m0 < m1; m0 += kNC) {
        const int nc = min(kNC, m1 - m0);
        for (int c = 0; c < nc; ++c)
            rotate_in(a.h + (size_t) a.grp_ent[m0 + c] * a.n_ff, nullptr, (const uint16_t*) (blob + a.suh[2]), 0,
                      a.n_ff, sm.xs + c * kMaxKR);
        __syncthreads();
        block_gemv_n<K2, cb>((const uint32_t*) (blob + a.tr[2]), a.n_embd / 16, g * 8, 0, a.n_ff / 16, sm.xs, nc,
                            sm.stage, sm.red, sm.outc);
        if (warp < nc)
            block_out(sm.outc + warp * kCols, (const uint16_t*) (blob + a.svh[2]) + g * kCols,
                      a.part + (size_t) a.grp_ent[m0 + warp] * a.n_embd + g * kCols);
        __syncthreads();
    }
}

// grid (n_embd / 128, T): each token's outputs, moe_down_kernel's combine (an entry off the device adds 0)
__global__ void __launch_bounds__(kCols) win_combine_kernel(const KWin a) {
    const int t = blockIdx.y, r = blockIdx.x * kCols + threadIdx.x;
    float p[8];
    for (int i = 0; i < a.k; ++i) {
        const int e = t * 8 + i;
        p[i] = a.plan_ptr[e] != 0ull ? a.part[(size_t) e * a.n_embd + r] : 0.0f;
    }
    a.out[(size_t) t * a.out_stride + r] =
        combine(a.plan_w + t * 8, p, a.k, a.cpu_flag[t] ? a.cpu_part + (size_t) t * a.n_embd + r : nullptr,
                a.sh_out != nullptr ? a.sh_out[(size_t) t * a.n_embd + r] : 0.0f);
}

// ---------------------------------------------------------------- dispatch
// F(K2, cb) for K 1..8 (K2 = 2 K) and the three codebooks, and for the half-integer K 1.5 / 2.5 / 3.5 (K2 3, 5, 7)
// with mul1 (exl3.cu's dispatch)
template <template <int, int> class F, typename... A>
bool dispatch(int K2, int cb, A&&... args) {
#define EXL3_CASE(KK)                                                    \
    case KK:                                                             \
        if (cb == 2) F<KK, 2>::run(std::forward<A>(args)...);            \
        else if (cb == 1) F<KK, 1>::run(std::forward<A>(args)...);       \
        else F<KK, 0>::run(std::forward<A>(args)...);                    \
        return true;
#define EXL3_HALF(KK)                                                    \
    case KK:                                                             \
        if (cb != 2) return false;                                       \
        F<KK, 2>::run(std::forward<A>(args)...);                         \
        return true;
    if (cb < 0 || cb > 2) return false;
    switch (K2) {
        EXL3_CASE(2) EXL3_CASE(4) EXL3_CASE(6) EXL3_CASE(8) EXL3_CASE(10) EXL3_CASE(12) EXL3_CASE(14) EXL3_CASE(16)
        EXL3_HALF(3) EXL3_HALF(5) EXL3_HALF(7)
        default: return false;
    }
#undef EXL3_HALF
#undef EXL3_CASE
}

template <int K2, int cb> struct RunMv {
    static void run(const KBatch& b, int blocks, cudaStream_t s) { mv_kernel<K2, cb><<<blocks, kThreads, 0, s>>>(b); }
};
template <int K2, int cb> struct RunMvN {
    static void run(const KBatch& b, int blocks, cudaStream_t s) {
        mv_n_kernel<K2, cb><<<blocks, kThreads, smem_n_bytes(K2), s>>>(b);
    }
};
template <int K2, int cb> struct RunWin {
    static void run(const KWin& a, int groups, cudaStream_t s) {
        win_gu_kernel<K2, cb><<<dim3((unsigned) (a.n_ff / kCols), (unsigned) groups), kThreads, smem_n_bytes(K2), s>>>(a);
    }
};
template <int K2, int cb> struct RunWinDown {
    static void run(const KWin& a, int groups, cudaStream_t s) {
        win_down_kernel<K2, cb><<<dim3((unsigned) (a.n_embd / kCols), (unsigned) groups), kThreads, smem_n_bytes(K2),
                                 s>>>(a);
    }
};
template <int K2, int cb> struct AllowSmemN {
    static void run() {
        const int b = (int) smem_n_bytes(K2);
        cudaFuncSetAttribute(mv_n_kernel<K2, cb>, cudaFuncAttributeMaxDynamicSharedMemorySize, b);
        cudaFuncSetAttribute(win_gu_kernel<K2, cb>, cudaFuncAttributeMaxDynamicSharedMemorySize, b);
        cudaFuncSetAttribute(win_down_kernel<K2, cb>, cudaFuncAttributeMaxDynamicSharedMemorySize, b);
    }
};
template <int K2, int cb> struct RunGU {
    static void run(const KMoe& a, cudaStream_t s) {
        moe_gate_up_kernel<K2, cb><<<dim3((unsigned) (a.n_ff / kCols), (unsigned) a.k), kThreads, 0, s>>>(a);
    }
};
template <int K2, int cb> struct RunGU2 {
    static void run(const KMoe& a, cudaStream_t s) {
        moe_gate_up2_kernel<K2, cb><<<dim3((unsigned) (a.n_ff / kCols), (unsigned) a.k, 2u), kThreads, 0, s>>>(a);
    }
};
template <int K2, int cb> struct RunDown {
    static void run(const KMoe& a, cudaStream_t s) {
        moe_down_kernel<K2, cb><<<dim3((unsigned) (a.n_embd / kCols), (unsigned) a.k), kThreads, 0, s>>>(a);
    }
};
// ---------------------------------------------------------------- per-stream scratch
constexpr size_t kPartFloats = (size_t) 2 << 20;   // split partials: 8 MB
constexpr size_t kCounters = 16384;
constexpr size_t kHFloats = (size_t) 8 * 8192;     // the experts' h
struct Scratch {
    float* part = nullptr;
    unsigned* cnt = nullptr;
    float* h = nullptr;
};
std::mutex g_mu;
std::map<std::tuple<int, cudaStream_t, std::thread::id>, Scratch> g_scratch;

Scratch* scratch(cudaStream_t s) {
    int dev = 0;
    cudaGetDevice(&dev);
    // Per-thread default stream handles compare equal across host threads.
    const auto key = std::make_tuple(dev, s,
        (s == cudaStreamPerThread || s == nullptr) ? std::this_thread::get_id() : std::thread::id{});
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_scratch.find(key);
    if (it != g_scratch.end()) return &it->second;
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(s, &cs) == cudaSuccess && cs != cudaStreamCaptureStatusNone) {
        std::fprintf(stderr, "exl3: scratch for a stream first seen inside a graph capture (call exl3::prepare)\n");
        return nullptr;
    }
    Scratch sc;
    if (cudaMalloc(&sc.part, kPartFloats * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&sc.cnt, kCounters * sizeof(unsigned)) != cudaSuccess ||
        cudaMalloc(&sc.h, kHFloats * sizeof(float)) != cudaSuccess ||
        cudaMemsetAsync(sc.cnt, 0, kCounters * sizeof(unsigned), s) != cudaSuccess) {
        std::fprintf(stderr, "exl3: scratch did not allocate\n");
        return nullptr;
    }
    cudaStreamSynchronize(s);
    return &(g_scratch[key] = sc);
}

int sm_count() {
    int dev = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    return std::max(1, sms);
}

// the several-activation kernels may use smem_n_bytes of shared memory: asked once per device (prepare, before a graph
// capture; else at the first such launch).  A device that cannot give a block that much (48 KB GPUs) runs one
// activation at a time instead
constexpr int kMaxDev = 64;
std::atomic<int> g_smem_n[kMaxDev];   // 0 not asked, 1 allowed, 2 refused
bool allow_smem_n() {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= kMaxDev) return false;
    const int st = g_smem_n[dev].load(std::memory_order_acquire);
    if (st != 0) return st == 1;
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_smem_n[dev].load(std::memory_order_relaxed) != 0) return g_smem_n[dev].load() == 1;
    int optin = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    bool ok = (size_t) optin >= smem_n_bytes(16);   // the widest bitrate, K 8
    if (ok) {
        for (int k2 = 2; k2 <= 16; ++k2)
            for (int cb = 0; cb <= 2; ++cb)
                if (k2_supported(k2, cb)) dispatch<AllowSmemN>(k2, cb);
        ok = cudaGetLastError() == cudaSuccess;
    }
    if (!ok)
        std::fprintf(stderr, "exl3: CUDA%d gives a block %d bytes of shared memory, the several-activation kernels "
                             "need %zu - a verify window's tokens run one at a time there\n", dev, optin,
                     smem_n_bytes(16));
    g_smem_n[dev].store(ok ? 1 : 2, std::memory_order_release);
    return ok;
}

// mv's input split for m: S blocks of kr inputs per output block, at most kMaxKR inputs a block and about two blocks
// per SM (the same for every activation count: mv_kernel and mv_n_kernel sum each activation alike)
void mv_split(const Mat& m, int sms, int& S, int& kr) {
    const int groups = m.n / kCols;
    S = (m.k + kMaxKR - 1) / kMaxKR;
    while (groups * S < 2 * sms && m.k / (S * 2) >= 256) S *= 2;
    kr = ((m.k + S - 1) / S + 127) / 128 * 128;
    S = (m.k + kr - 1) / kr;
}

}  // namespace

int launch_errors() { return g_launch_errors.load(std::memory_order_relaxed); }

bool prepare(cudaStream_t s) {
    if (scratch(s) == nullptr) return false;
    (void) allow_smem_n();
    return true;
}

bool several_ok() { return allow_smem_n(); }

bool mv(const MvJob* jobs, int n, cudaStream_t s) {
    if (n <= 0 || n > kMaxMvJobs) return false;
    Scratch* sc = scratch(s);
    if (sc == nullptr) return false;
    const int sms = sm_count();
    // the launch units: a job of one activation is mv_kernel's, a job of several mv_n_kernel's in runs of up to kNC
    // (one mv_kernel unit an activation where the device lacks the shared memory)
    struct Unit {
        int j, c0, nc;
    };
    Unit units[kMaxMvJobs * kMaxMvCols];
    int nu = 0;
    bool several = false;
    for (int j = 0; j < n; ++j) several = several || jobs[j].ncol > 1;
    const int run = several && allow_smem_n() ? kNC : 1;
    for (int j = 0; j < n; ++j) {
        const Mat& m = *jobs[j].m;
        if (!supported(m) || (jobs[j].y == nullptr && jobs[j].y16 == nullptr) ||
            (jobs[j].xf == nullptr && jobs[j].xq == nullptr && jobs[j].x16 == nullptr) ||
            jobs[j].ncol < 1 || jobs[j].ncol > kMaxMvCols) {
            std::fprintf(stderr, "exl3 mv: job %d unsupported (k %d n %d K %g cb %d, %d activations)\n", j, m.k, m.n,
                         m.k2 / 2.0, m.cb, jobs[j].ncol);
            return false;
        }
        for (int c0 = 0; c0 < jobs[j].ncol; c0 += run) units[nu++] = {j, c0, std::min(run, jobs[j].ncol - c0)};
    }
    bool done[sizeof(units) / sizeof(units[0])] = {};
    for (int i = 0; i < nu; ++i) {
        if (done[i]) continue;
        const Mat& m0 = *jobs[units[i].j].m;
        const bool multi = units[i].nc > 1;
        // a launch: the units of one (K, codebook, kernel), as many as KBatch and the split scratch take (the next
        // launch on the stream may use the scratch again: this one's last blocks are done with it by then)
        KBatch b{};
        int blocks = 0, nb = 0;
        size_t part_at = 0, cnt_at = 0;
        for (int u = i; u < nu && nb < kMaxMvJobs; ++u) {
            const MvJob& job = jobs[units[u].j];
            const Mat& m = *job.m;
            if (done[u] || m.k2 != m0.k2 || m.cb != m0.cb || (units[u].nc > 1) != multi) continue;
            const int c0 = units[u].c0, nc = units[u].nc, groups = m.n / kCols;
            int S = 1, kr = 0;
            mv_split(m, sms, S, kr);
            const size_t need = S > 1 ? (size_t) S * m.n * nc : 0;
            if (S > 1 && (part_at + need > kPartFloats || cnt_at + (size_t) groups > kCounters)) {
                if (nb > 0) continue;   // the next launch
                std::fprintf(stderr, "exl3 mv: scratch too small for k %d n %d (%d activations)\n", m.k, m.n, nc);
                return false;
            }
            done[u] = true;
            KJob& J = b.j[nb];
            J.tr = m.trellis;
            J.suh = m.suh;
            J.svh = m.svh;
            J.xf = job.xf != nullptr ? job.xf + c0 * job.xf_cs : nullptr;
            J.xq = job.xq != nullptr ? (const Q81*) ((const uint8_t*) job.xq + c0 * job.xq_cs) : nullptr;
            J.y = job.y != nullptr ? job.y + c0 * job.y_cs : nullptr;
            J.x16 = job.x16 != nullptr ? job.x16 + c0 * job.xf_cs : nullptr;
            J.y16 = job.y16 != nullptr ? job.y16 + c0 * job.y_cs : nullptr;
            J.x_bf16 = job.x_bf16;
            J.y_bf16 = job.y_bf16;
            J.bias = job.bias;
            J.alpha = job.alpha;
            J.k = m.k;
            J.n = m.n;
            J.ld = m.ld > 0 ? m.ld : m.n / 16;
            J.S = S;
            J.kr = kr;
            J.part = sc->part + part_at;
            J.cnt = sc->cnt + cnt_at;
            J.nc = nc;
            J.xf_cs = job.xf_cs;
            J.xq_cs = job.xq_cs;
            J.y_cs = job.y_cs;
            part_at += need;
            if (S > 1) cnt_at += (size_t) groups;
            blocks += groups * S;
            b.blk_end[nb++] = blocks;
        }
        b.n = nb;
        if (!(multi ? dispatch<RunMvN>(m0.k2, m0.cb, b, blocks, s) : dispatch<RunMv>(m0.k2, m0.cb, b, blocks, s)))
            return false;
    }
    launch_check("mv");
    return true;
}

static bool moe_args(const MoeArgs& a, KMoe& m, cudaStream_t s) {
    Scratch* sc = scratch(s);
    if (sc == nullptr) return false;
    if (a.k < 1 || a.k > 8 || a.n_embd % kCols != 0 || a.n_ff % kCols != 0 || a.n_embd > kMaxKR ||
        a.n_ff > kMaxKR || (size_t) a.k * a.n_ff > kHFloats || (size_t) a.k * a.n_embd > kPartFloats ||
        (size_t) (a.n_embd / kCols) > kCounters) {
        std::fprintf(stderr, "exl3 moe: geometry not covered (k %d n_embd %d n_ff %d)\n", a.k, a.n_embd, a.n_ff);
        return false;
    }
    m = KMoe{};
    m.plan_ptr = a.plan_ptr;
    m.plan_w = a.plan_w;
    m.cpu_part = a.cpu_part;
    m.cpu_flag = a.cpu_flag;
    m.x16 = a.x16;
    m.x_bf16 = a.x_bf16;
    m.h = sc->h;
    m.part = sc->part;
    m.cnt = sc->cnt;
    for (int r = 0; r < 3; ++r) {
        m.suh[r] = a.lay.suh[r];
        m.svh[r] = a.lay.svh[r];
        m.tr[r] = a.lay.trellis[r];
    }
    m.k = a.k;
    m.n_embd = a.n_embd;
    m.n_ff = a.n_ff;
    m.limit = a.limit;
    return true;
}

void moe_gate_up(const MoeArgs& a, const float* x, float* h, cudaStream_t s) {
    KMoe m;
    if (!moe_args(a, m, s)) return;
    if (a.lay.k2[0] != a.lay.k2[1]) {
        std::fprintf(stderr, "exl3 moe_gate_up: gate K %g != up K %g\n", a.lay.k2[0] / 2.0, a.lay.k2[1] / 2.0);
        return;
    }
    m.x = x;
    if (h != nullptr) m.h = h;
    // (the split's scratch: [k][2][n_ff] partials, k * n_ff / 128 counters)
    if ((size_t) a.k * 2 * a.n_ff <= kPartFloats && (size_t) a.k * (a.n_ff / kCols) <= kCounters) {
        if (!dispatch<RunGU2>(a.lay.k2[0], a.lay.cb, m, s))
            std::fprintf(stderr, "exl3 moe_gate_up: K %g cb %d\n", a.lay.k2[0] / 2.0, a.lay.cb);
        launch_check("moe_gate_up (split)");
        return;
    }
    if (!dispatch<RunGU>(a.lay.k2[0], a.lay.cb, m, s))
        std::fprintf(stderr, "exl3 moe_gate_up: K %g cb %d\n", a.lay.k2[0] / 2.0, a.lay.cb);
    launch_check("moe_gate_up");
}

void moe_down(const MoeArgs& a, const float* h, const float* sh_out, float* out, cudaStream_t s) {
    KMoe m;
    if (!moe_args(a, m, s)) return;
    if (h != nullptr) m.h = const_cast<float*>(h);
    m.sh_out = sh_out;
    m.out = out;
    if (!dispatch<RunDown>(a.lay.k2[2], a.lay.cb, m, s))
        std::fprintf(stderr, "exl3 moe_down: K %g cb %d\n", a.lay.k2[2] / 2.0, a.lay.cb);
    launch_check("moe_down");
}

bool win_experts(const WinArgs& a, cudaStream_t s) {
    if (a.T < 1 || a.k < 1 || a.k > 8 || a.n_embd % kCols != 0 || a.n_ff % kCols != 0 || a.n_embd > kMaxKR ||
        a.n_ff > kMaxKR || a.lay.k2[0] != a.lay.k2[1] || a.h == nullptr || a.part == nullptr || a.out == nullptr) {
        std::fprintf(stderr, "exl3 win_experts: not covered (T %d k %d n_embd %d n_ff %d, gate K %g up K %g)\n", a.T,
                     a.k, a.n_embd, a.n_ff, a.lay.k2[0] / 2.0, a.lay.k2[1] / 2.0);
        return false;
    }
    if (!allow_smem_n()) return false;
    KWin w{};
    w.grp_ptr = a.grp_ptr;
    w.grp_start = a.grp_start;
    w.grp_ent = a.grp_ent;
    w.n_groups = a.n_groups;
    w.plan_ptr = a.plan_ptr;
    w.plan_w = a.plan_w;
    w.cpu_part = a.cpu_part;
    w.cpu_flag = a.cpu_flag;
    w.x = a.x;
    w.x_stride = a.x_stride;
    w.h = a.h;
    w.part = a.part;
    w.sh_out = a.sh_out;
    w.out = a.out;
    w.out_stride = a.out_stride;
    for (int r = 0; r < 3; ++r) {
        w.suh[r] = a.lay.suh[r];
        w.svh[r] = a.lay.svh[r];
        w.tr[r] = a.lay.trellis[r];
    }
    w.k = a.k;
    w.n_embd = a.n_embd;
    w.n_ff = a.n_ff;
    w.limit = a.limit;
    const int groups = a.T * a.k;   // at most (the blocks past the real count return at once)
    if (!dispatch<RunWin>(a.lay.k2[0], a.lay.cb, w, groups, s) ||
        !dispatch<RunWinDown>(a.lay.k2[2], a.lay.cb, w, groups, s))
        return false;
    win_combine_kernel<<<dim3((unsigned) (a.n_embd / kCols), (unsigned) a.T), kCols, 0, s>>>(w);
    launch_check("win_experts");
    return true;
}

}  // namespace strata::kernels::exl3::legacy
#endif
