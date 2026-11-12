// Derived from Strata fused_gr.cu, pinned commit 9f7d977d220e9fdf2ec53cf2620cdadac1cfb24e.
// See third_party/strata/README.md for provenance and FastLLM adaptations.
/*
MIT License

Copyright (c) 2026 Niko1221 and the Strata contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#pragma once
namespace fastllm_strata_hc {
constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;
constexpr int THREADS = 256, WARPS = 8, UPM_COLS = 16, UPM_BLOCKS = N / UPM_COLS;
constexpr int kFusedGrMaxT = 4;
struct FusedGrArgs {
    const float *R, *w_norm;
    const void *bo_prev, *inj_prev;
    const uint16_t *w_down, *w_up, *w_inject;
    float *R_out, *inject_out;
    void *mixed;
    float eps;
    int blockType, injectionType, mixedType;
    bool apply;
};
struct GrMulti { FusedGrArgs a[kFusedGrMaxT]; int T; };
__device__ __forceinline__ float load_block(const FusedGrArgs &a, size_t i) {
    return Qwen4CudaLoadActivation(a.bo_prev, a.blockType, i);
}
__device__ __forceinline__ float4 load_block4(const FusedGrArgs &a, size_t i) {
    if (a.blockType == int(fastllm::FLOAT32)) return *reinterpret_cast<const float4*>(static_cast<const float*>(a.bo_prev)+i);
    return make_float4(load_block(a,i),load_block(a,i+1),load_block(a,i+2),load_block(a,i+3));
}
__device__ __forceinline__ float load_injection(const FusedGrArgs &a, size_t i) {
    return Qwen4CudaLoadActivation(a.inj_prev, a.injectionType, i);
}
__device__ __forceinline__ float combine(float b, float g, float r) {
    return fmaf(b,g,r);
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
// warp_sum of 8 independent values at once, token t's total ending in every lane whose bits 4..2 spell t ((lane >> 2) & 7).  The same
// additions as warp_sum(a[t]) in the same order (offset 16, 8, 4, 2, 1; a + b = b + a, so a lane adding its partner's value to its own gets the
// bits the plain butterfly gets): each level only exchanges the half of the values the partner keeps, 4 + 2 + 1 + 1 + 1 shuffles instead of 40.
__device__ __forceinline__ float warp_sum8(const float (&a)[8], int lane) {
    const bool h4 = (lane & 16) != 0, h3 = (lane & 8) != 0, h2 = (lane & 4) != 0;
    float b[4], c[2];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float send = h4 ? a[j] : a[j + 4], keep = h4 ? a[j + 4] : a[j];
        b[j] = keep + __shfl_xor_sync(0xffffffffu, send, 16);
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const float send = h3 ? b[j] : b[j + 2], keep = h3 ? b[j + 2] : b[j];
        c[j] = keep + __shfl_xor_sync(0xffffffffu, send, 8);
    }
    const float send = h2 ? c[0] : c[1], keep = h2 ? c[1] : c[0];
    float d = keep + __shfl_xor_sync(0xffffffffu, send, 4);
    d += __shfl_xor_sync(0xffffffffu, d, 2);
    d += __shfl_xor_sync(0xffffffffu, d, 1);
    return d;
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }

// 8 FP16 packed in a uint4, unpacked to two float4 once when reused across tokens.
struct Weightx8 { float4 a, b; };
__device__ __forceinline__ Weightx8 unpack8(const uint4 w) {
    const float2 x = __half22float2(*reinterpret_cast<const half2 *>(&w.x));
    const float2 y = __half22float2(*reinterpret_cast<const half2 *>(&w.y));
    const float2 z = __half22float2(*reinterpret_cast<const half2 *>(&w.z));
    const float2 q = __half22float2(*reinterpret_cast<const half2 *>(&w.w));
    return {make_float4(x.x,x.y,y.x,y.y),make_float4(z.x,z.y,q.x,q.y)};
}
__device__ __forceinline__ float dot8u(const Weightx8& w, const float4 x0, const float4 x1) {
    float acc = 0.0f;
    acc = fmaf(w.a.x, x0.x, acc);
    acc = fmaf(w.a.y, x0.y, acc);
    acc = fmaf(w.a.z, x0.z, acc);
    acc = fmaf(w.a.w, x0.w, acc);
    acc = fmaf(w.b.x, x1.x, acc);
    acc = fmaf(w.b.y, x1.y, acc);
    acc = fmaf(w.b.z, x1.z, acc);
    acc = fmaf(w.b.w, x1.w, acc);
    return acc;
}
__device__ __forceinline__ float dot8u_ptr(const Weightx8& w, const float* x) {
    const float4* x4 = reinterpret_cast<const float4*>(x);
    return dot8u(w, x4[0], x4[1]);
}

// 8 FP16 packed in a uint4 against 8 16-byte-aligned floats (same 8-FMA order).
__device__ __forceinline__ float dot8(const uint4 w, const float* x) {
    return dot8u_ptr(unpack8(w), x);
}

// FastLLM layout: the low/high float4 halves of a warp's eight-value
// vectors occupy separate contiguous planes. This removes shared-bank
// conflicts while preserving every multiply and reduction in its old order.
template<bool PLANAR> __device__ __forceinline__ int shared_index(int i) {
    return PLANAR ? (i & ~255) + ((i & 248) >> 1) + ((i & 4) << 5) + (i & 3) : i;
}
template<bool PLANAR> __device__ __forceinline__ float4 shared_load4(const float *base, int i) {
    return *reinterpret_cast<const float4 *>(base + shared_index<PLANAR>(i));
}
template<bool PLANAR> __device__ __forceinline__ float shared_dot8(const Weightx8 &w, const float *base, int i) {
    return dot8u(w,shared_load4<PLANAR>(base,i),shared_load4<PLANAR>(base,i+4));
}
constexpr int PR = LR + HC;                        // partial rows per (token, stream): 320 down + 4 inject

// The v3 kernels can issue weight loads first into registers, so the DRAM fetch overlaps the
// prologue (gr_down: staging R / bo / w_norm into shared memory; gr_up: the partial sums -> lo) instead of starting behind it.
// Loads, values and arithmetic order remain unchanged.
// ld.global.nc as a volatile asm so the compiler cannot sink the load back to its first use, behind the barrier.
__device__ __forceinline__ uint4 ldg_v4(const uint4* p) {
#if defined(__HIPCC__)
    return *p;   // (PTX does not exist on HIP; the plain load)
#else
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
#endif
}
constexpr int TQ3 = N / 8 / 32;                    // uint4 weight chunks per lane in one stream's slice (10)

// PF (BW-DENSE, bitwise the PF = 0 kernel): bit 0 issues the staging's loads (R, bo, w_norm of every token) before using any
// of them; bit 1 loads the warp's weight row into registers first. R2V == 1, S == 1 only.
// RB (R2V > 1 only, bitwise equivalent): the dot phase loops over the lane's chunks first, then the tokens, then
// the warp's R2V rows, so a token's 8 activations are read from shared memory once for all R2V rows (it was once per row).  Each (row, token) accumulator
// adds the same chunk dots in the same ascending order, then the same warp reduction.
template <int S, int MAX_T = kFusedGrMaxT, bool EXACT_T = false, int R2V = 1, int PF = 0, bool RB = false, bool PLANAR = false>
__global__ void __launch_bounds__(THREADS) gr_down_v3_kernel(GrMulti m, float* __restrict__ part, float* __restrict__ ssg) {
    extern __shared__ __align__(16) float xs[];    // [T][N / S]
    constexpr int R2 = R2V;                        // down rows per warp (R2V > 1: fewer blocks, each stages its stream slice once for R2V x the rows; the same dots)
    __shared__ float red[WARPS][MAX_T];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = EXACT_T ? MAX_T : m.T;
    // S = 2: each stream's 2560 columns in two halves (blockIdx.y = stream * S + half): twice the blocks
    constexpr int SL = N / S, TQS = SL / 8 / 32;
    const int rg = blockIdx.x, c = blockIdx.y / S, h = blockIdx.y - (blockIdx.y / S) * S;
    constexpr int NDB = LR / (WARPS * R2);          // down row blocks per stream; block NDB = the inject rows
    const bool inject_block = rg == NDB;
    // warp w owns rows row0 + w * R2 + r (r < R2); the inject block: warps 0-3, one row each
    const int row0 = inject_block ? warp : (rg * WARPS + warp) * R2;
    const int nrows = inject_block ? ((m.a[0].w_inject != nullptr && warp < HC) ? 1 : 0) : R2;
    const bool active = nrows > 0;
    const uint16_t* wbase = inject_block ? m.a[0].w_inject : m.a[0].w_down;
    uint4 wq[(PF & 2) ? TQS : 1];   // PF bit 1: this warp's row of weights, loaded before the staging below
    if constexpr (PF != 0) static_assert(R2V == 1 && S == 1, "the prefetch variants are for the one-row-a-warp kernels");
    if constexpr ((PF & 2) != 0) {
        if (active) {
            const uint4* w4 = reinterpret_cast<const uint4*>(wbase + (size_t) row0 * D + (size_t) c * N + (size_t) h * SL);
#pragma unroll
            for (int q = 0; q < TQS; ++q) wq[q] = ldg_v4(w4 + lane + 32 * q);
        }
    }
    float ssp[MAX_T];
    if constexpr ((PF & 1) != 0) {
        // the same per-thread sums in the same order (items i = t, t + THREADS, ... ascending, tokens in order), with the
        // loads of a group of tokens in flight together instead of one dependent L2 round trip per item
        constexpr int NI = (SL / 4 + THREADS - 1) / THREADS;
        constexpr int TG = MAX_T < 4 ? MAX_T : 4;
        float4 gv[NI];
#pragma unroll
        for (int ii = 0; ii < NI; ++ii) {
            const int i = t + ii * THREADS;
            gv[ii] = i < SL / 4 ? reinterpret_cast<const float4*>(m.a[0].w_norm + (size_t) c * N + (size_t) h * SL)[i] : make_float4(0.f, 0.f, 0.f, 0.f);
        }
#pragma unroll
        for (int k0 = 0; k0 < MAX_T; k0 += TG) {
            float4 rv[TG][NI], bv[TG][NI];
#pragma unroll
            for (int kk = 0; kk < TG; ++kk) {
                const int k = k0 + kk;
                if (k >= MAX_T || (!EXACT_T && k >= T)) continue;
                const FusedGrArgs& a = m.a[k];
                const float4* R4 = reinterpret_cast<const float4*>(a.R + (size_t) c * N + (size_t) h * SL);

#pragma unroll
                for (int ii = 0; ii < NI; ++ii) {
                    const int i = t + ii * THREADS;
                    rv[kk][ii] = i < SL / 4 ? R4[i] : make_float4(0.f, 0.f, 0.f, 0.f);
                    bv[kk][ii] = (a.apply && i < SL / 4) ? load_block4(a, (size_t) h * SL + i * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
                }
            }
#pragma unroll
            for (int kk = 0; kk < TG; ++kk) {
                const int k = k0 + kk;
                if (k >= MAX_T) continue;
                ssp[k] = 0.0f;
                if (!EXACT_T && k >= T) continue;
                const FusedGrArgs& a = m.a[k];
                const float gw = a.apply ? load_injection(a, c) : 0.0f;
                float4* X4 = reinterpret_cast<float4*>(xs + (size_t) k * SL);
#pragma unroll
                for (int ii = 0; ii < NI; ++ii) {
                    const int i = t + ii * THREADS;
                    if (i < SL / 4) {
                        float4 r = rv[kk][ii];
                        if (a.apply) {
                            const float4 b = bv[kk][ii];
                            r.x = combine(b.x, gw, r.x); r.y = combine(b.y, gw, r.y);
                            r.z = combine(b.z, gw, r.z); r.w = combine(b.w, gw, r.w);
                        }
                        const float4 g = gv[ii];
                        ssp[k] += r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
                        X4[shared_index<PLANAR>(i * 4) / 4] = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
                    }
                }
            }
        }
    } else {
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        ssp[k] = 0.0f;
        if (!EXACT_T && k >= T) continue;
        const FusedGrArgs& a = m.a[k];
        const float gw = a.apply ? load_injection(a, c) : 0.0f;
        const float4* R4 = reinterpret_cast<const float4*>(a.R + (size_t) c * N + (size_t) h * SL);
        const float4* G4 = reinterpret_cast<const float4*>(a.w_norm + (size_t) c * N + (size_t) h * SL);

        float4* X4 = reinterpret_cast<float4*>(xs + (size_t) k * SL);
        for (int i = t; i < SL / 4; i += THREADS) {
            float4 r = R4[i];
            if (a.apply) {
                const float4 b = load_block4(a, (size_t) h * SL + i * 4);
                r.x = combine(b.x, gw, r.x); r.y = combine(b.y, gw, r.y);
                r.z = combine(b.z, gw, r.z); r.w = combine(b.w, gw, r.w);
            }
            const float4 g = G4[i];
            ssp[k] += r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
            X4[shared_index<PLANAR>(i * 4) / 4] = make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
        }
    }
    }
#pragma unroll
    for (int k = 0; k < MAX_T; ++k) {
        if (!EXACT_T && k >= T) break;
        const float v = warp_sum(ssp[k]);
        if (lane == 0) red[warp][k] = v;
    }
    __syncthreads();
    if (rg == 0 && t < T) {
        float sum = 0.0f;
        for (int w = 0; w < WARPS; ++w) sum += red[w][t];
        ssg[(t * HC + c) * S + h] = sum;
    }
    if (!active) return;
    if constexpr (RB && R2V > 1) {
        static_assert(S == 1 && EXACT_T && PF == 0, "the register-blocked down dots are for the exact-T, one-half, plain-staging kernels");
        float accr[R2][MAX_T];
#pragma unroll
        for (int r = 0; r < R2; ++r)
#pragma unroll
            for (int k = 0; k < MAX_T; ++k) accr[r][k] = 0.0f;
        const uint4* w4r[R2];
#pragma unroll
        for (int r = 0; r < R2; ++r)
            w4r[r] = reinterpret_cast<const uint4*>(wbase + (size_t) (row0 + (r < nrows ? r : 0)) * D + (size_t) c * N + (size_t) h * SL);
        uint4 wch[R2][TQS];
#pragma unroll
        for (int q = 0; q < TQS; ++q)
#pragma unroll
            for (int r = 0; r < R2; ++r) wch[r][q] = __ldg(w4r[r] + lane + 32 * q);
#pragma unroll
        for (int q = 0; q < TQS; ++q) {
            const int j = lane + 32 * q;
            Weightx8 wv[R2];
#pragma unroll
            for (int r = 0; r < R2; ++r) wv[r] = unpack8(wch[r][q]);
#pragma unroll
            for (int k = 0; k < MAX_T; ++k) {
                const float4 x0 = shared_load4<PLANAR>(xs + (size_t)k * SL, j * 8);
                const float4 x1 = shared_load4<PLANAR>(xs + (size_t)k * SL, j * 8 + 4);
#pragma unroll
                for (int r = 0; r < R2; ++r) accr[r][k] += dot8u(wv[r], x0, x1);
            }
        }
#pragma unroll
        for (int r = 0; r < R2; ++r) {
            if (r >= nrows) break;
            const int prow = inject_block ? LR + warp : row0 + r;
            if constexpr (MAX_T >= 5 && MAX_T <= 8) {
                float accs[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int k = 0; k < MAX_T; ++k) accs[k] = accr[r][k];
                const float d8 = warp_sum8(accs, lane);
                const int k = (lane >> 2) & 7;
                if ((lane & 3) == 0 && k < MAX_T) part[(((size_t) k * HC + c) * S + h) * PR + prow] = d8;
            } else {
#pragma unroll
                for (int k = 0; k < MAX_T; ++k) {
                    const float v = warp_sum(accr[r][k]);
                    if (lane == 0) part[(((size_t) k * HC + c) * S + h) * PR + prow] = v;
                }
            }
        }
        return;
    }
#pragma unroll
    for (int r = 0; r < R2; ++r) {
        if (r >= nrows) break;
        const uint4* w4 = reinterpret_cast<const uint4*>(wbase + (size_t) (row0 + r) * D + (size_t) c * N + (size_t) h * SL);
        float acc[MAX_T];
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) acc[k] = 0.0f;
#pragma unroll
        for (int q = 0; q < TQS; ++q) {
            const int j = lane + 32 * q;
            Weightx8 wvq;
            if constexpr ((PF & 2) != 0) wvq = unpack8(wq[q]);
            else wvq = unpack8(__ldg(w4 + j));
#pragma unroll
            for (int k = 0; k < MAX_T; ++k)
                if (EXACT_T || k < T) acc[k] += shared_dot8<PLANAR>(wvq, xs + (size_t)k * SL, j * 8);
        }
        const int prow = inject_block ? LR + warp : row0 + r;
        if constexpr (EXACT_T && MAX_T >= 5 && MAX_T <= 8) {   // warp_sum8: token k's total sits in lanes 4k..4k+3
            float accs[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int k = 0; k < MAX_T; ++k) accs[k] = acc[k];
            const float d8 = warp_sum8(accs, lane);
            const int k = (lane >> 2) & 7;
            if ((lane & 3) == 0 && k < MAX_T) part[(((size_t) k * HC + c) * S + h) * PR + prow] = d8;
        } else {
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) {
            if (!EXACT_T && k >= T) break;
            const float v = warp_sum(acc[k]);
            if (lane == 0) part[(((size_t) k * HC + c) * S + h) * PR + prow] = v;
        }
        }
    }
}

// PF (BW-DENSE, bitwise the PF = 0 kernel): bit 0 issues the loads of the partial sums and of every row's R / w_norm / bo / inj before
// their first use (the in-place store of R in the row loop otherwise ends each row's loads at a full latency); bit 1 loads the
// warp's w_up rows first. S == 1, EXACT_T only.
// PF (BW-DENSE, bitwise the PF = 0 kernel): bit 0 issues the loads of the partial sums and of every row's R / w_norm / bo / inj before
// their first use (the in-place store of R in the row loop otherwise ends each row's loads at a full latency); bit 1 loads the
// warp's w_up rows first. S == 1, EXACT_T only.
template <int S, int MAX_T = kFusedGrMaxT, bool EXACT_T = false, int PF = 0, int COLS = 16, bool PLANAR = false>
__global__ void __launch_bounds__(THREADS) gr_up_v3_kernel(GrMulti m, const float* __restrict__ part,
                                                           const float* __restrict__ ssg) {
    __shared__ __align__(16) float lo[MAX_T][PLANAR ? ((LR + 255) / 256) * 256 : LR];
    __shared__ float rsS[MAX_T][HC];
    __shared__ float g[MAX_T][HC][COLS];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int T = EXACT_T ? MAX_T : m.T;
    const int d0 = blockIdx.x * COLS;
    constexpr int RPW = HC * COLS / WARPS;     // 8 rows per warp
    uint4 pwa[(PF & 2) ? RPW : 1], pwb[(PF & 2) ? RPW : 1];   // PF bit 1: the warp's 8 rows of w_up, loaded before the prologue
    if constexpr (PF != 0) static_assert(S == 1 && EXACT_T, "the prefetch variants are for the exact-T, one-half kernels");
    if constexpr ((PF & 2) != 0) {
#pragma unroll
        for (int q = 0; q < RPW; ++q) {
            const int r = warp + q * WARPS;
            const int c = r / COLS, dd = r - c * COLS, i = c * N + d0 + dd;
            const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) i * LR);
            pwa[q] = ldg_v4(w4 + lane);
            pwb[q] = lane < LR / 8 - 32 ? ldg_v4(w4 + 32 + lane) : make_uint4(0, 0, 0, 0);
        }
    }
    constexpr int NIT = (MAX_T * LR + THREADS - 1) / THREADS;
    float ppv[(PF & 1) ? NIT : 1][HC];                        // PF bit 0: the partial sums lo is built from
    float prv[(PF & 1) ? RPW : 1], pwn[(PF & 1) ? RPW : 1], pbo[(PF & 1) ? RPW : 1], pip[(PF & 1) ? RPW : 1];
    if constexpr ((PF & 1) != 0) {
#pragma unroll
        for (int it = 0; it < NIT; ++it) {
            const int i = t + it * THREADS;
#pragma unroll
            for (int c = 0; c < HC; ++c) {
                ppv[it][c] = 0.0f;
                if (i < MAX_T * LR) {
                    const int k = i / LR, r = i - k * LR;
                    ppv[it][c] = part[(((size_t) k * HC + c) * S + 0) * PR + r];
                }
            }
        }
        if (lane < T) {
            const FusedGrArgs& a = m.a[lane];
#pragma unroll
            for (int q = 0; q < RPW; ++q) {
                const int r = warp + q * WARPS;
                const int c = r / COLS, dd = r - c * COLS, i = c * N + d0 + dd;
                prv[q] = a.R[i];
                pwn[q] = a.w_norm[i];
                pbo[q] = 0.0f;
                pip[q] = 0.0f;
                if (a.apply) { pbo[q] = load_block(a, d0 + dd); pip[q] = load_injection(a, c); }
            }
        }
    }
    if (t < T * HC) {
        const int k = t / HC, c = t - k * HC;
        float ss = 0.0f;
#pragma unroll
        for (int h = 0; h < S; ++h) ss += ssg[t * S + h];
        const float r = rsqrtf(ss / (float) N + m.a[k].eps);
        rsS[k][c] = r;

    }
    __syncthreads();
    if constexpr ((PF & 1) != 0) {
#pragma unroll
        for (int it = 0; it < NIT; ++it) {
            const int i = t + it * THREADS;
            if (i < MAX_T * LR) {
                const int k = i / LR, r = i - k * LR;
                float sum = 0.0f;
#pragma unroll
                for (int c = 0; c < HC; ++c) {
                    float p = 0.0f;
                    p += ppv[it][c];
                    sum = fmaf(rsS[k][c], p, sum);
                }
                const float x = sum / (float) HC;
                lo[k][shared_index<PLANAR>(r)] = x / (1.0f + __expf(-x));
            }
        }
    } else
    for (int i = t; i < T * LR; i += THREADS) {
        const int k = i / LR, r = i - k * LR;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            float p = 0.0f;
#pragma unroll
            for (int h = 0; h < S; ++h) p += part[(((size_t) k * HC + c) * S + h) * PR + r];
            sum = fmaf(rsS[k][c], p, sum);
        }
        const float x = sum / (float) HC;
        lo[k][shared_index<PLANAR>(r)] = x / (1.0f + __expf(-x));
    }
    if (blockIdx.x == 0 && t < T * HC) {
        const int k = t / HC, cc = t - k * HC;
        if (m.a[k].w_inject != nullptr) {
            float sum = 0.0f;
#pragma unroll
            for (int c = 0; c < HC; ++c) {
                float p = 0.0f;
#pragma unroll
                for (int h = 0; h < S; ++h) p += part[(((size_t) k * HC + c) * S + h) * PR + LR + cc];
                sum = fmaf(rsS[k][c], p, sum);
            }
            m.a[k].inject_out[cc] = 2.0f * sigmoidf_(sum / (float) HC);
        }
    }
    __syncthreads();
#pragma unroll
    for (int q = 0; q < RPW; ++q) {
        const int r = warp + q * WARPS;
        const int c = r / COLS, dd = r - c * COLS, i = c * N + d0 + dd;
        Weightx8 wa, wb;
        if constexpr ((PF & 2) != 0) {
            wa = unpack8(pwa[q]);
            wb = unpack8(pwb[q]);
        } else {
            const uint4* w4 = reinterpret_cast<const uint4*>(m.a[0].w_up + (size_t) i * LR);
            wa = unpack8(__ldg(w4 + lane));
            wb = unpack8(lane < LR / 8 - 32 ? __ldg(w4 + 32 + lane) : make_uint4(0, 0, 0, 0));
        }
        float rv = 0.0f, wn = 0.0f, bo = 0.0f, ip = 0.0f;
        bool apply = false;
        if (lane < T) {
            const FusedGrArgs& a = m.a[lane];
            apply = a.apply;
            if constexpr ((PF & 1) != 0) { rv = prv[q]; wn = pwn[q]; bo = pbo[q]; ip = pip[q]; }
            else {
                rv = a.R[i];
                wn = a.w_norm[i];
                if (apply) { bo = load_block(a, d0 + dd); ip = load_injection(a, c); }
            }
        }
        float mine = 0.0f;
        if constexpr (EXACT_T && MAX_T >= 5 && MAX_T <= 8) {   // the tokens' warp sums together (warp_sum8, missing tokens 0): the same bits
            float accs[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int k = 0; k < MAX_T; ++k) {
                float acc = shared_dot8<PLANAR>(wa, lo[k], lane * 8);
                if (lane < LR / 8 - 32) acc += shared_dot8<PLANAR>(wb, lo[k], (32 + lane) * 8);
                accs[k] = acc;
            }
            const float d8 = warp_sum8(accs, lane);
            mine = __shfl_sync(0xffffffffu, d8, (lane & 7) << 2);
        } else {
#pragma unroll
        for (int k = 0; k < MAX_T; ++k) {
            if (!EXACT_T && k >= T) break;
            float acc = shared_dot8<PLANAR>(wa, lo[k], lane * 8);
            if (lane < LR / 8 - 32) acc += shared_dot8<PLANAR>(wb, lo[k], (32 + lane) * 8);
            acc = warp_sum(acc);
            if (lane == k) mine = acc;
        }
        }
        if (lane < T) {
            if (apply) {
                rv = combine(bo, ip, rv);
                m.a[lane].R_out[i] = rv;
            }
            const float x = rv * wn * rsS[lane][c];
            g[lane][c][dd] = x * sigmoidf_(mine);
        }
    }
    __syncthreads();
    for (int i = t; i < T * COLS; i += THREADS) {
        const int k = i / COLS, col = i - k * COLS;
        float sum = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) sum += g[k][c][col];
        const float value = sum / (float) HC;
        // Preserve the original FP32 mix followed by the activation cast,
        // without an intermediate tensor or graph node.
        if (m.a[k].mixedType == int(fastllm::FLOAT32))
            static_cast<float *>(m.a[k].mixed)[d0 + col] = value;
        else if (m.a[k].mixedType == int(fastllm::FLOAT16))
            static_cast<half *>(m.a[k].mixed)[d0 + col] = __float2half_rn(value);
        else
            static_cast<__nv_bfloat16 *>(m.a[k].mixed)[d0 + col] = __float2bfloat16_rn(value);
    }
}

template<int T,int COLS,bool PLANAR=false> void launch(const GrMulti &m, float *part, float *ssg) {
    gr_down_v3_kernel<1,T,true,1,1,false,PLANAR><<<dim3(LR/WARPS+1,HC),THREADS,T*N*sizeof(float),cudaStreamPerThread>>>(m,part,ssg);
    gr_up_v3_kernel<1,T,true,1,COLS,PLANAR><<<N/COLS,THREADS,0,cudaStreamPerThread>>>(m,part,ssg);
}
} // namespace fastllm_strata_hc
