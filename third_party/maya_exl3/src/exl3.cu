// EXL3 format/codebook/tile-decoding origin: ExLlamaV3, Turboderp (MIT).
// https://github.com/turboderp-org/exllamav3
// This implementation was imported via Project Maya and modified by Fastllm.
// See ../NOTICE.txt and both accompanying MIT licenses for the full provenance.
#include <thread>
#include <tuple>
// src/kernels/cuda/exl3.cu - the EXL3 device kernels (the format: include/strata/kernels/exl3.hpp).
//
// The codebooks and the trellis window arithmetic follow exllamav3 (MIT, turboderp - exllamav3_ext/quant/
// codebook.cuh, exl3_dq.cuh, hadamard_inner.cuh), and so does decoding a tile straight into tensor-core fragments
// (its small-m GEMV and GEMM); the kernels around them are shaped for this engine's decode:
//
//   * a block owns 128 outputs (8 tiles - exactly one output rotation block) over a slice of at most 4096 inputs;
//     warp w takes tile column w through every 16-row slice, decoding 8 weights per lane (one fragment: 2 columns x
//     4 rows) - on sm_80 and later straight into one mma.m16n8k16 a tile with up to 8 activations as its columns,
//     F32 accumulation; before sm_80 (and on HIP) F32 FMAs on the CUDA cores;
//   * the input rotation runs per block, 128 inputs at a time (F32 or q8_1 in, FP16 in shared memory), so the
//     engine's activation buffers are used as they are; the next 128 inputs' tile words load while these decode;
//   * a matrix with more than 4096 inputs, or too few output blocks to fill the device, splits its inputs over
//     blocks: the partial sums go to a per-stream scratch and the LAST block of an output block adds them up in
//     split order (deterministic) and runs the output rotation.
//
// NVIDIA GPUs before sm_80 (V100, T4, RTX 20) decode with v1.1.8's kernels instead (exl3_legacy.cu, use_legacy
// below): this file's CUDA-core fallback was slower there (a V100 with EXL3-3.05bpw: decode 4.9 -> 4.4 tok/s).  The
// prompt path below is this file's on every GPU.
#include "strata/kernels/exl3.hpp"
#include "strata/kernels/exl3_io.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#if !defined(STRATA_USE_HIP)
#include <mma.h>
#endif
#include "strata/kernels/dp4a.hpp"   // sm_60: __dp4a -> the vmad fallback

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <type_traits>
#include <utility>

#if !defined(STRATA_USE_HIP)
namespace strata::kernels::exl3::legacy {   // exl3_legacy.cu: v1.1.8's decode kernels
int launch_errors();
bool prepare(cudaStream_t s);
bool several_ok();
bool mv(const MvJob* jobs, int n, cudaStream_t s);
void moe_gate_up(const MoeArgs& a, const float* x, float* h, cudaStream_t s);
void moe_down(const MoeArgs& a, const float* h, const float* sh_out, float* out, cudaStream_t s);
bool win_experts(const WinArgs& a, cudaStream_t s);
}  // namespace strata::kernels::exl3::legacy
#define EXL3_LEGACY(call) \
    if (use_legacy()) return legacy::call
// ... the experts' kernels only where v1.1.8's cover the model: their rotated input slice is at most 4096 wide (GLM's);
// a wider one (DeepSeek-V4.1's 5120) takes this file's kernels on every GPU - the token path and a window's alike, so
// their results stay the same bits
#define EXL3_LEGACY_MOE(a, call) \
    if ((a).n_embd <= 4096 && (a).n_ff <= 4096 && use_legacy()) return legacy::call
#else
#define EXL3_LEGACY(call) (void) 0
#define EXL3_LEGACY_MOE(a, call) (void) 0
#endif

namespace strata::kernels::exl3 {
namespace {

constexpr int kThreads = 256, kWarps = 8;
constexpr int kCols = 128;                // outputs per block: 8 tiles, one rotation block
constexpr int kMaxKR = 4096;              // inputs per block at most (mv's split)
// a routed expert's widths at most (moe_* / win_*: a block streams its inputs through block_gemv's two 128-value chunk
// buffers, so the shared memory does not grow with them; DeepSeek-V4.1's n_embd is 5120)
constexpr int kMaxMoeK = 8192;
constexpr float kHad = 0.088388347648f;   // 1 / sqrt(128)

struct Q81 {   // the engine's q8_1 block (ggml block_q8_1): d, sum, 32 int8
    __half2 ds;
    int8_t qs[32];
};

std::atomic<int> g_launch_errors{0};
thread_local const int* g_skip = nullptr;   // set_skip: the calling thread's flag (nullptr: none)
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
    return __builtin_amdgcn_sad_u8(x, 0u, acc);   // (v_sad_u8: the bytes' |x - 0| summed onto acc, one instruction)
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

__device__ __forceinline__ uint32_t h2u(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }
__device__ __forceinline__ __half2 u2h(uint32_t u) { return *reinterpret_cast<__half2*>(&u); }

// (hi : lo) >> s, the low 32 bits
__device__ __forceinline__ uint32_t fsh(uint32_t lo, uint32_t hi, int s) {
    return (uint32_t) ((((uint64_t) hi << 32) | (uint64_t) lo) >> s);
}

// Step p's state is the 16-bit window ending at bit S(p) of the tile's MSB-first bit stream (w: its 4 K2 words, K2 =
// 2 K), wrapping around the tile: S(p) = (p + 1) K for an integer K, floor((p + 1) K2 / 2) for exllamav3 1.6's
// half-integer K = KA + 0.5 (steps of KA and KA + 1 bits alternating, even steps KA).  A lane decodes its 8 consecutive
// steps lane * 8 .. + 7 together, the windows of a group of steps cut from one or two words with funnel shifts -
// exllamav3's grouping per K (exl3_dq.cuh: dq8_aligned_*bits, dq8, dq4, dq2x2, dq8_half), which the steps' fixed
// offsets keep within two words.  f[i]: steps 2i, 2i + 1 as FP16 values.
// Step lane * 8 + j sits at tile row (lane & 3) * 2 + (j & 1) + 8 (j >> 1 & 1), column (lane >> 2) + 8 (j >> 2).
template <int cb>
__device__ __forceinline__ void decode_states(const uint32_t (&s)[8], uint32_t (&f)[4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) f[i] = h2u(decode2h<cb>(s[2 * i] & 0xffffu, s[2 * i + 1] & 0xffffu));
}
// K = 4: the lane's steps end in its own word b (word lane), the first three reach into a (word lane - 1)
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
// step p .. p + G - 1's windows (G = 4 or 2) from the two words that hold them
template <int K, int G>
__device__ __forceinline__ void states_group(const uint32_t* w, int p, uint32_t* s) {
    constexpr int NW = 8 * K;
    const int b0 = (p + 257) * K - 16, b2 = b0 + (G - 1) * K + 16;
    const int i0 = b0 >> 5, i2 = (b2 - 1) >> 5, s2 = ((i2 + 1) << 5) - b2;
    const uint32_t a = w[i0 % NW], b = w[i2 % NW];
#pragma unroll
    for (int j = 0; j < G; ++j) s[G - 1 - j] = fsh(b, a, s2 + j * K);
}
// an integer K
template <int K, int cb>
__device__ __forceinline__ void frag8_int(const uint32_t* w, int lane, uint32_t (&f)[4]) {
    const int p = lane * 8;
    uint32_t s[8];
    if constexpr (K == 4) {
        states4(w[(lane + 31) & 31], w[lane], s);
    } else if constexpr (K == 1 || K == 2) {
        // the 8 windows overlap in one 32-bit window: steps p + j end 8 - 1 - j K bits apart
        constexpr int NW = 8 * K, per = 32 / (8 * K);   // lanes per word
        const int i1 = lane / per, i0 = (i1 + NW - 1) % NW;
        const uint32_t b = fsh(w[i1], w[i0], (per - 1 - lane % per) * 8 * K);
#pragma unroll
        for (int j = 0; j < 8; ++j) s[j] = b >> ((7 - j) * K);
    } else if constexpr (K == 3) {
        // all 8 windows within two words: two funnel shifts, the rest by shifting
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
    decode_states<cb>(s, f);
}
// a half-integer K (K2 = 2 KA + 1, mul1): exllamav3's dq8_half.  The lane's steps start even, so their windows end at
// S(p + 7) = (p / 2 + 4) K2 and then KA + 1, KA, KA + 1, ... bits earlier; four windows span 18 + 3 KA <= 27 bits, one
// funnel shift of two words each (steps 4..7 from window 7's end, 0..3 from window 3's, 2 K2 bits earlier)
template <int K2, int cb>
__device__ __forceinline__ void frag8_half(const uint32_t* w, int lane, uint32_t (&f)[4]) {
    constexpr int KA = K2 / 2, NW = 4 * K2, span = 18 + 3 * KA;
    const int e7 = (lane * 4 + 4) * K2 + 128 * K2;   // one ring on: no negative bit index
    uint32_t s[8];
#pragma unroll
    for (int g = 0; g < 2; ++g) {
        const int e = e7 - (1 - g) * 2 * K2;
        const int hi = (e - 1) >> 5, lo = (e - span) >> 5, sh = ((hi + 1) << 5) - e;
        const uint32_t x = fsh(w[hi % NW], w[lo % NW], sh);   // the window ending at e in the low bits
        s[4 * g + 3] = x;
        s[4 * g + 2] = x >> (KA + 1);
        s[4 * g + 1] = x >> (2 * KA + 1);
        s[4 * g + 0] = x >> (3 * KA + 2);
    }
    decode_states<cb>(s, f);
}
// K2 = 2 K: an even K2 is the integer K2 / 2, an odd one a half-integer K (mul1 only: exllamav3 has no other)
template <int K2, int cb>
__device__ __forceinline__ void frag8(const uint32_t* w, int lane, uint32_t (&f)[4]) {
    if constexpr ((K2 & 1) != 0) {
        static_assert(cb == 2 && K2 >= 3 && K2 <= 7, "a half-integer K: 1.5, 2.5 or 3.5, mul1");
        frag8_half<K2, cb>(w, lane, f);
    } else {
        frag8_int<K2 / 2, cb>(w, lane, f);
    }
}

// the lane's 8 steps as F32: v[j] = step lane * 8 + j
template <int K2, int cb>
__device__ __forceinline__ void decode8(const uint32_t* w, int lane, float* v) {
    uint32_t f[4];
    frag8<K2, cb>(w, lane, f);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 d = __half22float2(u2h(f[i]));
        v[2 * i] = d.x;
        v[2 * i + 1] = d.y;
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

// the activation a warp rotates into its block's FP16 chunks: F32 (f) or the engine's q8_1 blocks (q)
struct Xin {
    const float* f = nullptr;
    const Q81* q = nullptr;
    const uint16_t* h = nullptr;
    bool bf16 = false;
};

// the 128-input chunk of x from k0 on: the lane's 4 values and their suh ...
__device__ __forceinline__ void chunk_load(const Xin& x, const uint16_t* __restrict__ suh, int k0, int lane,
                                           float (&v)[4], float (&s)[4]) {
    const int base = k0 + lane * 4;
    if (x.h != nullptr) {
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = io_load16(x.h[base + i], x.bf16);
    } else if (x.f != nullptr) {
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = x.f[base + i];
    } else {
        const Q81* b = x.q + base / 32;
        const float d = __low2float(b->ds);
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = d * (float) b->qs[base % 32 + i];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) s[i] = h2f(suh[base + i]);
}
struct XinHalf { const uint16_t* h = nullptr; };   // (FP16 rows: the portable prompt kernels' input)
__device__ __forceinline__ void chunk_load(const XinHalf& x, const uint16_t* suh, int k0, int lane,
                                         float (&v)[4], float (&s)[4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int c = k0 + lane * 4 + i;
        v[i] = h2f(x.h[c]);
        s[i] = h2f(suh[c]);
    }
}
// ... and H(x * suh) of the chunk, FP16, into dst[0..127]
__device__ __forceinline__ void chunk_store(float (&v)[4], const float (&s)[4], int lane, __half* dst) {
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] *= s[i];
    had128(v, lane);
    __half2* d = reinterpret_cast<__half2*>(dst + lane * 4);
    d[0] = __floats2half2_rn(v[0], v[1]);
    d[1] = __floats2half2_rn(v[2], v[3]);
}

// ---------------------------------------------------------------- the block GEMV
// A block owns 128 outputs - 8 tiles, warp w the tile column tc0 + w through all of the block's inputs [k0, k1) - for
// nc <= kTok activations at once.  Per 128-input chunk, warp c (< nc) scales activation c's chunk by suh and rotates
// it into FP16 (xb, two buffers), then each warp decodes its tile of each of the chunk's 8 slices and multiplies it
// with every activation: on sm_80 and later as one mma.m16n8k16 a tile (the decoded tile is the A operand - a lane's 8
// steps are its A fragment - and the activations are the B operand's columns, as exllamav3's small-m GEMV and its
// tensor-core GEMM decode tiles straight into fragments), F32 accumulation; elsewhere on the CUDA cores, in F32.  An
// activation's sums do not depend on the other columns, so a window's token gets the token path's bits.  The next
// chunk's tile words are loaded while this one is decoded.  out[c * 128 + i]: activation c's sum for output i.  Ends
// with a __syncthreads (out complete).
constexpr int kTok = 8;   // activations a block takes at once (the MMA's n)

#if !defined(STRATA_USE_HIP) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define EXL3_TC 1
#endif

// RDNA 3 / 4: v_dot2_f32_f16 - two FP16 products and their FP32 sum onto c in one instruction (no conversions)
#if defined(STRATA_USE_HIP) && (defined(__GFX11__) || defined(__GFX12__))
#define EXL3_DOT2 1
typedef _Float16 exl3_h2 __attribute__((ext_vector_type(2)));
__device__ __forceinline__ float dot2_acc(uint32_t a, uint32_t b, float c) {
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(exl3_h2, a), __builtin_bit_cast(exl3_h2, b), c, false);
}
#endif

#if !defined(EXL3_TC)
// a lane's 4 weights of one column times the activation's 4 inputs, added to a (the CUDA-core product)
__device__ __forceinline__ float dot4_acc(float a, const float* v, float x0, float x1, float x2, float x3) {
    return __fadd_rn(a, __fmaf_rn(v[3], x3, __fmaf_rn(v[2], x2, __fmaf_rn(v[1], x1, __fmul_rn(v[0], x0)))));
}
#else
__device__ __forceinline__ void mma_f32(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#endif

#if defined(STRATA_USE_HIP)
// Preserve the scalar fallback rounding tree when one-row specialization changes compiler scheduling.
// Empty VGPR constraints prevent reassociation without adding arithmetic instructions.
__device__ __forceinline__ float dot4_acc_exact(float a, const float* v, float x0, float x1, float x2, float x3) {
    float p = __fmul_rn(v[0], x0);
    asm volatile("" : "+v"(p));
    p = __fmaf_rn(v[1], x1, p);
    asm volatile("" : "+v"(p));
    p = __fmaf_rn(v[2], x2, p);
    asm volatile("" : "+v"(p));
    p = __fmaf_rn(v[3], x3, p);
    asm volatile("" : "+v"(p));
    return __fadd_rn(a, p);
}
#endif
// a tile's words, and a lane's share of them (word lane + 32 r)
template <int K2>
struct Words {
    static constexpr int W = 4 * K2, R = (W + 31) / 32;
};

// the warp's tile of each of a chunk's 8 slices (slice s at tile0 + s sstride)
template <int K2>
__device__ __forceinline__ void load_tiles(const uint32_t* __restrict__ tile0, size_t sstride, int lane,
                                           uint32_t (&w)[8][Words<K2>::R]) {
    constexpr int W = Words<K2>::W, R = Words<K2>::R;
#pragma unroll
    for (int s = 0; s < 8; ++s)
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int i = lane + 32 * r;
            w[s][r] = (W % 32 == 0 || i < W) ? __ldg(tile0 + (size_t) s * sstride + i) : 0u;
        }
}

// one slice's fragment from the lane's words: K = 4 straight from the registers (the lane's steps end in its own word,
// the first three reach into the previous lane's), other K (a half-integer one too) through the warp's stage (4 K2
// words)
template <int K2, int cb>
__device__ __forceinline__ void slice_frags(const uint32_t (&w)[Words<K2>::R], uint32_t* st, int lane, uint32_t (&f)[4]) {
    if constexpr (K2 == 8) {
        uint32_t s[8];
        states4(__shfl_sync(0xffffffffu, w[0], (lane + 31) & 31), w[0], s);
        decode_states<cb>(s, f);
    } else {
        constexpr int W = Words<K2>::W, R = Words<K2>::R;
        __syncwarp();
#pragma unroll
        for (int r = 0; r < R; ++r)
            if (W % 32 == 0 || lane + 32 * r < W) st[lane + 32 * r] = w[r];
        __syncwarp();
        frag8<K2, cb>(st, lane, f);
    }
}

// the blocks an SM keeps of the block_gemv kernels (their registers), as measured on an RTX 3090 and an RTX 5060 Ti:
// the routed experts four at K <= 4 (a token's 8 experts, 256 blocks, are one wave on 82 SMs), two at wider codes;
// mv two, three at K > 4 on sm_80 .. sm_90 (sm_120 spills at the 80 registers that takes).  Before sm_80 (the CUDA
// cores' product, more registers) two; HIP reads the second bound differently and keeps its default.  (K2 = 2 K.)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template <int K2>
constexpr int kMinBlocksX = K2 <= 8 ? 4 : 2;
#else
template <int K2>
constexpr int kMinBlocksX = 2;
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 1000
template <int K2>
constexpr int kMinBlocksMv = K2 <= 8 ? 2 : 3;
#else
template <int K2>
constexpr int kMinBlocksMv = 2;
#endif
#if defined(STRATA_USE_HIP)
#define EXL3_BOUNDS(n) __launch_bounds__(kThreads)
#else
#define EXL3_BOUNDS(n) __launch_bounds__(kThreads, n)
#endif

// the shared memory of a block_gemv block
template <int K2, int Tokens = kTok>
struct BlockSmem {
    __half xb[2 * Tokens * kCols];       // two chunk buffers, kTok activations each
    uint32_t stage[kWarps * 4 * K2];   // a tile per warp (K != 4)
    float out[Tokens * kCols];
};

// (kTag 1: the mixed kernels' own copy - sharing one with them changes how AMD compiles the others' arithmetic)
template <int K2, int cb, class Input = Xin, int kTag = 0, int Tokens = kTok>
__device__ void block_gemv(const uint32_t* __restrict__ tr, int ld, int tc0, int k0, int k1, const Input& x,
                           const uint16_t* __restrict__ suh, int nc, BlockSmem<K2, Tokens>& sm) {
    constexpr int W = Words<K2>::W, R = Words<K2>::R;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const size_t sstride = (size_t) ld * W;
    const uint32_t* tile = tr + ((size_t) (k0 / 16) * ld + tc0 + warp) * W;
    uint32_t* st = sm.stage + warp * W;
    const bool rot = warp < nc;
    const int nq = (k1 - k0) / kCols;
    uint32_t wc[8][R], wn[8][R];
    float xv[4], xs4[4];
    load_tiles<K2>(tile, sstride, lane, wc);
    if (rot) chunk_load(x, suh, k0, lane, xv, xs4);
#if defined(EXL3_TC)
    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#else
    float acc[Tokens][2];
#pragma unroll
    for (int cc = 0; cc < Tokens; ++cc) acc[cc][0] = acc[cc][1] = 0.0f;
#endif
    for (int q = 0; q < nq; ++q) {
        __half* xq = sm.xb + (q & 1) * (Tokens * kCols);
        if (rot) chunk_store(xv, xs4, lane, xq + warp * kCols);
        __syncthreads();
        if (q + 1 < nq) {
            load_tiles<K2>(tile + (size_t) (q + 1) * 8 * sstride, sstride, lane, wn);
            if (rot) chunk_load(x, suh, k0 + (q + 1) * kCols, lane, xv, xs4);
        }
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            uint32_t f[4];
            slice_frags<K2, cb>(wc[s], st, lane, f);
#if defined(EXL3_TC)
            uint32_t b0 = 0u, b1 = 0u;
            if (g < nc) {
                const __half* xr = xq + g * kCols + s * 16 + 2 * t;
                b0 = *reinterpret_cast<const uint32_t*>(xr);
                b1 = *reinterpret_cast<const uint32_t*>(xr + 8);
            }
            // A: rows (outputs) g / g + 8, columns (inputs) 2t.. / 2t + 8..: steps 0,1 / 4,5 / 2,3 / 6,7
            mma_f32(c, f[0], f[2], f[1], f[3], b0, b1);
#elif defined(EXL3_DOT2)
            // the pairs straight from FP16 (f[i]: steps 2i, 2i + 1; the activation's 2t, 2t + 1 / 2t + 8, 2t + 9)
#pragma unroll
            for (int cc = 0; cc < Tokens; ++cc)
                if (cc < nc) {
                    const __half* xr = xq + cc * kCols + s * 16 + 2 * t;
                    const uint32_t xa = *reinterpret_cast<const uint32_t*>(xr);
                    const uint32_t xb = *reinterpret_cast<const uint32_t*>(xr + 8);
                    acc[cc][0] = dot2_acc(f[1], xb, dot2_acc(f[0], xa, acc[cc][0]));
                    acc[cc][1] = dot2_acc(f[3], xb, dot2_acc(f[2], xa, acc[cc][1]));
                }
#else
            float v[8];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 d = __half22float2(u2h(f[i]));
                v[2 * i] = d.x;
                v[2 * i + 1] = d.y;
            }
#pragma unroll
            for (int cc = 0; cc < Tokens; ++cc)
                if (cc < nc) {
                    const __half* xr = xq + cc * kCols + s * 16 + 2 * t;
                    const float x0 = __half2float(xr[0]), x1 = __half2float(xr[1]);
                    const float x2 = __half2float(xr[8]), x3 = __half2float(xr[9]);
                    #if defined(STRATA_USE_HIP)
                    if constexpr (Tokens == 1) {
                        acc[cc][0] = dot4_acc_exact(acc[cc][0], v, x0, x1, x2, x3);
                        acc[cc][1] = dot4_acc_exact(acc[cc][1], v + 4, x0, x1, x2, x3);
                    } else
#endif
                    {
                        acc[cc][0] = dot4_acc(acc[cc][0], v, x0, x1, x2, x3);
                        acc[cc][1] = dot4_acc(acc[cc][1], v + 4, x0, x1, x2, x3);
                    }
                }
#endif
        }
#pragma unroll
        for (int s = 0; s < 8; ++s)
#pragma unroll
            for (int r = 0; r < R; ++r) wc[s][r] = wn[s][r];
    }
    const int o = warp * 16 + g;
#if defined(EXL3_TC)
    // c: the tile's outputs g (c[0], c[1]) and g + 8 (c[2], c[3]) for activations 2t (c[0], c[2]) and 2t + 1
    if (2 * t < nc) {
        sm.out[(2 * t) * kCols + o] = c[0];
        sm.out[(2 * t) * kCols + o + 8] = c[2];
    }
    if (2 * t + 1 < nc) {
        sm.out[(2 * t + 1) * kCols + o] = c[1];
        sm.out[(2 * t + 1) * kCols + o + 8] = c[3];
    }
#else
    // the 4 lanes of one g hold parts of the same two outputs
#pragma unroll
    for (int cc = 0; cc < Tokens; ++cc)
        if (cc < nc) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                acc[cc][h] += __shfl_xor_sync(0xffffffffu, acc[cc][h], 1);
                acc[cc][h] += __shfl_xor_sync(0xffffffffu, acc[cc][h], 2);
            }
            if (t == 0) {
                sm.out[cc * kCols + o] = acc[cc][0];
                sm.out[cc * kCols + o + 8] = acc[cc][1];
            }
        }
#endif
    __syncthreads();
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
    float* part;       // nc x S x n partial sums (S > 1)
    unsigned* cnt;     // n / 128 counters, zero between launches
    float alpha;
    int k, n, ld, S, kr;
    int nc;            // the activations (1..kTok), xf / xq / y advanced by these per activation
    size_t xf_cs, xq_cs, y_cs;   // (floats, bytes, floats)
};
struct KBatch {
    KJob j[kMaxMvJobs];
    int blk_end[kMaxMvJobs];
    int n;
    const int* skip;   // set_skip's flag: non-zero = a pipelined window's token past a wrong draft (nullptr: none)
};

// AMD CDNA3 / CDNA4 (gfx942, gfx950): a split's sums, a gate / up pair's swiglu and the experts' combine run as kernels
// of their own after the GEMVs (mv_reduce_kernel's note); the retained input-split implementation and
// device code of those GPUs (EXL3_SEP_DEV) and the host's choice per GPU (sep_sums) go together.  Every other GPU keeps
// the last block's hand-off (on a Radeon 8060S, gfx1151, the same bits but not faster: the routed experts ~4% slower)
#if defined(STRATA_USE_HIP) && (defined(__gfx942__) || defined(__gfx950__))
#define EXL3_SEP_DEV 1
#endif

// a launch's blocks: its jobs' (output block, input split) pairs, each for all of the job's nc activations at once
// (block_gemv).  S > 1: the partial sums go to the scratch and the last block of an output block adds them up in split
// order (deterministic) and runs the epilogue
template <int K2, int cb, bool Single = false, bool NativeBfloat = false>
__global__ void EXL3_BOUNDS(kMinBlocksMv<K2>) mv_kernel(const KBatch b) {
    __shared__ __align__(16) BlockSmem<K2, Single ? 1 : kTok> sm;
    __shared__ int last;
    int ji = 0;
    while ((int) blockIdx.x >= b.blk_end[ji]) ++ji;
    const KJob& J = b.j[ji];
    const int nc = Single ? 1 : J.nc;
    const int local = (int) blockIdx.x - (ji > 0 ? b.blk_end[ji - 1] : 0);
    const int g = local / J.S, sp = local % J.S;
    const int k0 = sp * J.kr, k1 = min(J.k, k0 + J.kr);
    const int warp = threadIdx.x >> 5;
    // a skipped token (set_skip): no work.  The flag changes only between launches (skip_gate, on the same stream), so
    // every block of the launch returns and no split counter moves
    if (b.skip != nullptr && *b.skip) return;
    Xin x;
    if (warp < nc) {
        if constexpr (NativeBfloat) {x.h = J.x16 + warp * J.xf_cs; x.bf16 = true;}
        else if (J.x16 != nullptr) { x.h = J.x16 + warp * J.xf_cs; x.bf16 = J.x_bf16; }
        else if (J.xf != nullptr) x.f = J.xf + warp * J.xf_cs;
        else x.q = (const Q81*) ((const uint8_t*) J.xq + warp * J.xq_cs);
    }
    block_gemv<K2, cb>(J.tr, J.ld, g * 8, k0, k1, x, J.suh, nc, sm);
    const float* bias = J.bias != nullptr ? J.bias + g * kCols : nullptr;
    if (J.S == 1) {
        if (warp < nc)
            epilogue(sm.out + warp * kCols, J.svh + g * kCols, J.alpha, bias, !NativeBfloat && J.y != nullptr ? J.y + warp * J.y_cs + g * kCols : nullptr,
                     J.y16 != nullptr ? J.y16 + warp * J.y_cs + g * kCols : nullptr, NativeBfloat || J.y_bf16);
        return;
    }
    if (threadIdx.x < kCols)
        for (int c = 0; c < nc; ++c)
            J.part[((size_t) c * J.S + sp) * J.n + g * kCols + threadIdx.x] = sm.out[c * kCols + threadIdx.x];
#if defined(EXL3_SEP_DEV)
    return;   // (mv_reduce_kernel adds them up)
#endif
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(&J.cnt[g], 1u) == (unsigned) (J.S - 1);
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (threadIdx.x < kCols)
        for (int c = 0; c < nc; ++c) {
            float s = 0.0f;
            for (int i = 0; i < J.S; ++i)
                s += *(volatile const float*) &J.part[((size_t) c * J.S + i) * J.n + g * kCols + threadIdx.x];
            sm.out[c * kCols + threadIdx.x] = s;
        }
    __syncthreads();
    if (warp < nc)
        epilogue(sm.out + warp * kCols, J.svh + g * kCols, J.alpha, bias, !NativeBfloat && J.y != nullptr ? J.y + warp * J.y_cs + g * kCols : nullptr,
                     J.y16 != nullptr ? J.y16 + warp * J.y_cs + g * kCols : nullptr, NativeBfloat || J.y_bf16);
    if (threadIdx.x == 0) J.cnt[g] = 0u;
}

#if defined(STRATA_USE_HIP)
// AMD: a split launch's sums, added up by a second kernel.  mv_kernel's last-block hand-off costs a fence per block,
// and at agent scope that is a write-back and an invalidate of the XCD's whole L2 (buffer_wbl2 / buffer_inv: on a
// MI350X most of a 6-bit attention GEMV's time - DeepSeek-V4.1's wq_b, 512 blocks: 77.7 -> 30.1 us, wo_b 125.8 -> 40.9
// us; its decode with DSpark 52.0 -> 74.3 tok/s); the launch boundary orders them once instead.  The same sums in split order and the same epilogue: the same bits.  Block: one output
// block of a split job for one activation (blockIdx.y), a thread a column; the split's loads go out 8 at a time ahead
// of their adds (a block for all the activations, a thread's nc x S loads one after another: 19.3 us a call with
// DSpark, half the GEMV's own)
__global__ void __launch_bounds__(kCols) mv_reduce_kernel(const KBatch b) {
    __shared__ float out[kCols];
    if (b.skip != nullptr && *b.skip) return;
    int ji = 0, g = (int) blockIdx.x;
    for (;; ++ji) {
        const int groups = b.j[ji].S > 1 ? b.j[ji].n / kCols : 0;
        if (g < groups) break;
        g -= groups;
    }
    const KJob& J = b.j[ji];
    const int c = (int) blockIdx.y;
    if (c >= J.nc) return;
    const float* p = J.part + (size_t) c * J.S * J.n + g * kCols + threadIdx.x;
    float s = 0.0f;
    int i = 0;
    for (; i + 8 <= J.S; i += 8) {
        float v[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) v[u] = p[(size_t) (i + u) * J.n];
#pragma unroll
        for (int u = 0; u < 8; ++u) s += v[u];
    }
    for (; i < J.S; ++i) s += p[(size_t) i * J.n];
    out[threadIdx.x] = s;
    __syncthreads();
    const float* bias = J.bias != nullptr ? J.bias + g * kCols : nullptr;
    if (threadIdx.x < 32) epilogue(out, J.svh + g * kCols, J.alpha, bias, J.y != nullptr ? J.y + c * J.y_cs + g * kCols : nullptr,
                     J.y16 != nullptr ? J.y16 + c * J.y_cs + g * kCols : nullptr, J.y_bf16);
}
#endif

// ---------------------------------------------------------------- the routed experts
struct KMoe {
    const unsigned long long* plan_ptr;
    const float* plan_w;
    const float* cpu_part;
    const int* cpu_flag;
    const float* x;
    const uint16_t* x16;
    bool x_bf16;
    float* h;            // [k][n_ff]: the gate's values, then the swiglu
    float* u;            // [k][n_ff]: the up's values
    const float* sh_out;
    float* out;
    float* part;         // [k][n_embd]
    unsigned* cnt;       // n_embd / 128
    unsigned* cnt_gu;    // k x n_ff / 128: the gate / up pairs
    size_t suh[3], svh[3], tr[3];
    int k, n_embd, n_ff;
    float limit;
    bool indirect, round_bf16, route_output; // Fastllm adapter; disabled by default
    const int* skip;     // set_skip's flag (nullptr: none)
#if defined(STRATA_USE_HIP)
    float* xp;           // gS / dS > 1: the input splits' partial sums (xsplit, AMD)
    int gS, gkr, dS, dkr;   // gate / up and down: the input splits and their width (1, the whole input: none)
#endif
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
// Fastllm adaptation (see NOTICE.txt). Default arguments retain upstream arithmetic.
__device__ __forceinline__ const uint8_t* projection_base(unsigned long long ptr, int role, bool indirect) {
    return (const uint8_t*) (indirect ? ((const unsigned long long*) ptr)[role] : ptr);
}
__device__ __forceinline__ float bf16_boundary(float x, bool enabled) {
    if (!enabled) return x;
    unsigned u = __float_as_uint(x);
    // Preserve NaNs instead of rounding a NaN payload into infinity.
    if ((u & 0x7fffffffU) > 0x7f800000U) return __uint_as_float((u & 0xffff0000U) | 0x00400000U);
    u += 0x7fffU + ((u >> 16) & 1U);
    return __uint_as_float(u & 0xffff0000U);
}
__device__ __forceinline__ float swiglu_boundary(float g, float u, float limit, bool enabled) {
    return bf16_boundary(swiglu(bf16_boundary(g, enabled), bf16_boundary(u, enabled), limit), enabled);
}
// ... the second block of a (gate, up) pair to finish (its counter back at zero for the next launch) ...
__device__ __forceinline__ bool pair_last(unsigned* cnt, int& last) {
#if defined(EXL3_SEP_DEV)
    (void) cnt;
    (void) last;
    return false;   // (moe_swiglu_kernel / win_swiglu_kernel: mv_reduce_kernel's note)
#endif
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(cnt, 1u) == 1u;
    __syncthreads();
    if (!last) return false;
    __threadfence();
    if (threadIdx.x == 0) *cnt = 0u;
    return true;
}
// ... and one output's sum: the experts' rows in plan order (p[i]: expert i's, 0 for one not on the device), the CPU
// lane's part when flagged (cpu_part: null when not), the shared expert's row (sh: 0 without one)
__device__ __forceinline__ float combine(const float* plan_w, const float* p, int k, const float* cpu_part, float sh) {
    float m = 0.0f;
    for (int i = 0; i < k; ++i) m = __fmaf_rn(plan_w[i], p[i], m);
    if (cpu_part != nullptr) m = __fadd_rn(m, *cpu_part);
    return __fadd_rn(m, sh);
}
#if defined(STRATA_USE_HIP)
// an input split's sums for one output in split order (p: split 0's, the next split's `stride` floats on), the loads
// 8 at a time ahead of their adds
__device__ __forceinline__ float split_sum(const float* p, int S, size_t stride) {
    float s = 0.0f;
    int i = 0;
    for (; i + 8 <= S; i += 8) {
        float v[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) v[u] = p[(size_t) (i + u) * stride];
#pragma unroll
        for (int u = 0; u < 8; ++u) s += v[u];
    }
    for (; i < S; ++i) s += p[(size_t) i * stride];
    return s;
}
// a block's raw sums (before block_out) for nc activations into the split scratch: activation c's at row(c)
template <class Row>
__device__ __forceinline__ void split_store(const float* out, int nc, Row row) {
    for (int i = threadIdx.x; i < nc * kCols; i += kThreads) row(i / kCols)[i % kCols] = out[i];
}
#endif

// grid (n_ff / 128, k, 2): expert blockIdx.y's gate (z 0, into h) or up (z 1, into u) for 128 h values; the pair's
// second block to finish turns them into the clamped swiglu, in h
template <int K2, int cb>
__global__ void EXL3_BOUNDS(kMinBlocksX<K2>) moe_gate_up_kernel(const KMoe a) {
    __shared__ __align__(16) BlockSmem<K2> sm;
    __shared__ int last;
#if defined(EXL3_SEP_DEV)
    const int ei = blockIdx.y, g = blockIdx.x, r = (int) blockIdx.z / a.gS, sp = (int) blockIdx.z % a.gS;
#else
    const int ei = blockIdx.y, g = blockIdx.x, r = blockIdx.z;
#endif
    const unsigned long long ptr = a.plan_ptr[ei];
    if (ptr == 0ull) return;
    if (a.skip != nullptr && *a.skip) return;   // a skipped token (set_skip, as mv_kernel)
    const uint8_t* blob = projection_base(ptr, r, a.indirect);
    const int warp = threadIdx.x >> 5;
    Xin x;
    x.f = a.x;
    x.h = a.x16;
    x.bf16 = a.x_bf16;
#if defined(EXL3_SEP_DEV)
    block_gemv<K2, cb>((const uint32_t*) (blob + a.tr[r]), a.n_ff / 16, g * 8, sp * a.gkr,
                      min(a.n_embd, (sp + 1) * a.gkr), x, (const uint16_t*) (blob + a.suh[r]), 1, sm);
    if (a.gS > 1) {   // (moe_gu_red_kernel)
        split_store(sm.out, 1, [&](int) { return a.xp + (((size_t) ei * 2 + r) * a.gS + sp) * a.n_ff + g * kCols; });
        return;
    }
#else
    block_gemv<K2, cb>((const uint32_t*) (blob + a.tr[r]), a.n_ff / 16, g * 8, 0, a.n_embd, x,
                      (const uint16_t*) (blob + a.suh[r]), 1, sm);
#endif
    const size_t row = (size_t) ei * a.n_ff + g * kCols;
    if (warp == 0) block_out(sm.out, (const uint16_t*) (blob + a.svh[r]) + g * kCols, (r == 0 ? a.h : a.u) + row);
    if (!pair_last(a.cnt_gu + ei * (a.n_ff / kCols) + g, last)) return;
    if (threadIdx.x < kCols)
        a.h[row + threadIdx.x] = swiglu_boundary(*(volatile const float*) &a.h[row + threadIdx.x],
                                        *(volatile const float*) &a.u[row + threadIdx.x], a.limit, a.round_bf16);
}

// grid (n_embd / 128, k): expert blockIdx.y's down rows for 128 outputs; the last block of each output block adds
// the experts up in plan order (as the engine's moe_down), then the CPU lane's part and the shared expert
template <int K2, int cb>
__global__ void EXL3_BOUNDS(kMinBlocksX<K2>) moe_down_kernel(const KMoe a) {
    __shared__ __align__(16) BlockSmem<K2> sm;
    __shared__ int last;
    const int ei = blockIdx.y, g = blockIdx.x;
    const unsigned long long ptr = a.plan_ptr[ei];
    const int warp = threadIdx.x >> 5;
    float* part = a.part + (size_t) ei * a.n_embd + g * kCols;
    if (a.skip != nullptr && *a.skip) return;   // a skipped token (set_skip, as mv_kernel)
#if defined(EXL3_SEP_DEV)
    const int sp = blockIdx.z;
    if (a.dS > 1 && ptr == 0ull) return;        // (moe_down_red_kernel writes its zeros)
#endif
    if (ptr != 0ull) {
        const uint8_t* blob = projection_base(ptr, 2, a.indirect);
        Xin x;
        x.f = a.h + (size_t) ei * a.n_ff;
#if defined(EXL3_SEP_DEV)
        block_gemv<K2, cb>((const uint32_t*) (blob + a.tr[2]), a.n_embd / 16, g * 8, sp * a.dkr,
                          min(a.n_ff, (sp + 1) * a.dkr), x, (const uint16_t*) (blob + a.suh[2]), 1, sm);
        if (a.dS > 1) {
            split_store(sm.out, 1, [&](int) { return a.xp + ((size_t) ei * a.dS + sp) * a.n_embd + g * kCols; });
            return;
        }
#else
        block_gemv<K2, cb>((const uint32_t*) (blob + a.tr[2]), a.n_embd / 16, g * 8, 0, a.n_ff, x,
                          (const uint16_t*) (blob + a.suh[2]), 1, sm);
#endif
        if (warp == 0) block_out(sm.out, (const uint16_t*) (blob + a.svh[2]) + g * kCols, part);
    } else if (threadIdx.x < kCols) {
        part[threadIdx.x] = 0.0f;
    }
    // Fastllm hybrid scheduling reduces GPU and CPU routes together later.
    // Every CTA owns one (expert, output block), so no cross-expert counter is
    // needed. RunDown disables input splitting for this output mode.
    if (a.route_output) {
        __syncthreads();
        if (threadIdx.x < kCols) {
            const int r = g * kCols + threadIdx.x;
            a.out[(size_t) ei * a.n_embd + r] = a.plan_w[ei] * bf16_boundary(part[threadIdx.x], a.round_bf16);
        }
        return;
    }
#if defined(EXL3_SEP_DEV)
    return;   // (moe_combine_kernel)
#endif
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(&a.cnt[g], 1u) == (unsigned) (a.k - 1);
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (threadIdx.x < kCols) {
        const int r = g * kCols + threadIdx.x;
        float p[8];
        for (int i = 0; i < a.k; ++i) p[i] = bf16_boundary(*(volatile const float*) &a.part[(size_t) i * a.n_embd + r], a.round_bf16);
        a.out[r] = combine(a.plan_w, p, a.k, a.cpu_flag != nullptr && *a.cpu_flag ? a.cpu_part + r : nullptr,
                           a.sh_out != nullptr ? a.sh_out[r] : 0.0f);
    }
    if (threadIdx.x == 0) a.cnt[g] = 0u;
}

// ---------------------------------------------------------------- a verify window's routed experts
// gf::moe_win_experts' grouping (group g: one expert, its blob at grp_ptr[g]; its entries e = t * 8 + i at
// grp_ent[grp_start[g] ..)), each group's weights read and decoded once for up to kTok of its tokens at a time (as
// exllamav3's fused MoE runs a run of slots that picked one expert as the rows of one MMA tile), with the token
// path's arithmetic per entry: every token's output is moe_gate_up + moe_down's for it alone, bit for bit
struct KWin {
    const unsigned long long* grp_ptr;
    const int *grp_start, *grp_ent, *n_groups;
    const unsigned long long* plan_ptr;
    const float* plan_w;
    const float* cpu_part;
    size_t cpu_stride;
    const int* cpu_flag;
    const float* x;
    const uint16_t* x16;
    bool x_bf16, indirect, round_bf16, route_output, preserve_missing;
    size_t x_stride;
    float* h;            // [T * 8][n_ff]: the gate's values, then the swiglu
    float* u;            // [T * 8][n_ff]: the up's values
    float* part;         // [T * 8][n_embd]
    const float* sh_out; // [T][n_embd]
    float* out;
    size_t out_stride;
    unsigned* cnt_gu;    // groups x n_ff / 128: the gate / up pairs
    size_t suh[3], svh[3], tr[3];
    int k, n_embd, n_ff;
    int first_group = 0, first_token = 0;
    float limit;
#if defined(STRATA_USE_HIP)
    float* xp;           // (KMoe's)
    int gS, gkr, dS, dkr;
#endif
};

// grid (n_ff / 128, groups at most, 2): group blockIdx.y's gate (z 0, into h) or up (z 1, into u) for 128 h values of
// each of its entries, kTok entries a pass; the pair's second block to finish turns them into the clamped swiglu
template <int K2, int cb>
__global__ void EXL3_BOUNDS(kMinBlocksX<K2>) win_gu_kernel(const KWin a) {
    __shared__ __align__(16) BlockSmem<K2> sm;
    __shared__ int last;
#if defined(EXL3_SEP_DEV)
    const int grp = blockIdx.y + a.first_group, g = blockIdx.x, r = (int) blockIdx.z / a.gS, sp = (int) blockIdx.z % a.gS;
#else
    const int grp = blockIdx.y + a.first_group, g = blockIdx.x, r = blockIdx.z;
#endif
    if (grp >= *a.n_groups) return;
    const uint8_t* blob = projection_base(a.grp_ptr[grp], r, a.indirect);
    const int warp = threadIdx.x >> 5;
    const int m0 = a.grp_start[grp], m1 = a.grp_start[grp + 1];
    float* dst = r == 0 ? a.h : a.u;
    for (int mm = m0; mm < m1; mm += kTok) {
        const int nc = min(kTok, m1 - mm);
        Xin x;
        if (warp < nc) {
            const size_t offset = (size_t) (a.grp_ent[mm + warp] >> 3) * a.x_stride;
            x.f = a.x != nullptr ? a.x + offset : nullptr;
            x.h = a.x16 != nullptr ? a.x16 + offset : nullptr;
            x.bf16 = a.x_bf16;
        }
#if defined(EXL3_SEP_DEV)
        block_gemv<K2, cb>((const uint32_t*) (blob + a.tr[r]), a.n_ff / 16, g * 8, sp * a.gkr,
                          min(a.n_embd, (sp + 1) * a.gkr), x, (const uint16_t*) (blob + a.suh[r]), nc, sm);
        if (a.gS > 1)   // (win_gu_red_kernel)
            split_store(sm.out, nc, [&](int c) {
                return a.xp + (((size_t) a.grp_ent[mm + c] * 2 + r) * a.gS + sp) * a.n_ff + g * kCols;
            });
        else if (warp < nc)
#else
        block_gemv<K2, cb>((const uint32_t*) (blob + a.tr[r]), a.n_ff / 16, g * 8, 0, a.n_embd, x,
                          (const uint16_t*) (blob + a.suh[r]), nc, sm);
        if (warp < nc)
#endif
            block_out(sm.out + warp * kCols, (const uint16_t*) (blob + a.svh[r]) + g * kCols,
                      dst + (size_t) a.grp_ent[mm + warp] * a.n_ff + g * kCols);
        __syncthreads();
    }
    if (!pair_last(a.cnt_gu + grp * (a.n_ff / kCols) + g, last)) return;
    if (threadIdx.x < kCols)
        for (int m = m0; m < m1; ++m) {
            const size_t row = (size_t) a.grp_ent[m] * a.n_ff + g * kCols + threadIdx.x;
            a.h[row] = swiglu_boundary(*(volatile const float*) &a.h[row], *(volatile const float*) &a.u[row], a.limit, a.round_bf16);
        }
}

// grid (n_embd / 128, groups at most): group blockIdx.y's down rows for 128 outputs of each entry into its part rows
template <int K2, int cb>
__global__ void EXL3_BOUNDS(kMinBlocksX<K2>) win_down_kernel(const KWin a) {
    __shared__ __align__(16) BlockSmem<K2> sm;
    const int grp = blockIdx.y + a.first_group, g = blockIdx.x;
#if defined(EXL3_SEP_DEV)
    const int sp = blockIdx.z;
#endif
    if (grp >= *a.n_groups) return;
    const uint8_t* blob = projection_base(a.grp_ptr[grp], 2, a.indirect);
    const int warp = threadIdx.x >> 5;
    const int m0 = a.grp_start[grp], m1 = a.grp_start[grp + 1];
    for (int mm = m0; mm < m1; mm += kTok) {
        const int nc = min(kTok, m1 - mm);
        Xin x;
        if (warp < nc) x.f = a.h + (size_t) a.grp_ent[mm + warp] * a.n_ff;
#if defined(EXL3_SEP_DEV)
        block_gemv<K2, cb>((const uint32_t*) (blob + a.tr[2]), a.n_embd / 16, g * 8, sp * a.dkr,
                          min(a.n_ff, (sp + 1) * a.dkr), x, (const uint16_t*) (blob + a.suh[2]), nc, sm);
        if (a.dS > 1)   // (win_down_red_kernel)
            split_store(sm.out, nc, [&](int c) {
                return a.xp + ((size_t) a.grp_ent[mm + c] * a.dS + sp) * a.n_embd + g * kCols;
            });
        else if (warp < nc)
#else
        block_gemv<K2, cb>((const uint32_t*) (blob + a.tr[2]), a.n_embd / 16, g * 8, 0, a.n_ff, x,
                          (const uint16_t*) (blob + a.suh[2]), nc, sm);
        if (warp < nc)
#endif
            block_out(sm.out + warp * kCols, (const uint16_t*) (blob + a.svh[2]) + g * kCols,
                      a.part + (size_t) a.grp_ent[mm + warp] * a.n_embd + g * kCols);
        __syncthreads();
    }
}

// grid (n_embd / 128, T): each token's outputs, moe_down_kernel's combine (an entry off the device adds 0)
__global__ void __launch_bounds__(kCols) win_combine_kernel(const KWin a) {
    const int t = blockIdx.y + a.first_token, r = blockIdx.x * kCols + threadIdx.x;
    float p[8];
    for (int i = 0; i < a.k; ++i) {
        const int e = t * 8 + i;
        p[i] = a.plan_ptr[e] != 0ull ? bf16_boundary(a.part[(size_t) e * a.n_embd + r], a.round_bf16) : 0.0f;
        if (a.route_output && (!a.preserve_missing || a.plan_ptr[e] != 0ull))
            a.out[((size_t) t * a.k + i) * a.n_embd + r] = a.plan_w[e] * p[i];
    }
    if (a.route_output) return;
    a.out[(size_t) t * a.out_stride + r] =
        combine(a.plan_w + t * 8, p, a.k,
                a.cpu_flag != nullptr && a.cpu_flag[t] ? a.cpu_part + (size_t) t * a.cpu_stride + r : nullptr,
                a.sh_out != nullptr ? a.sh_out[(size_t) t * a.n_embd + r] : 0.0f);
}

#if defined(STRATA_USE_HIP)
// AMD: the pieces the last block of a pair / an output block runs elsewhere, as kernels of their own after the GEMVs
// (mv_reduce_kernel's note: a fence a block is a write-back of the XCD's L2) - the same arithmetic, the same bits.
// grid (n_ff / 128, k): moe_gate_up_kernel's clamped swiglu
__global__ void __launch_bounds__(kCols) moe_swiglu_kernel(const KMoe a) {
    const int ei = blockIdx.y;
    if (a.plan_ptr[ei] == 0ull) return;
    if (a.skip != nullptr && *a.skip) return;
    const size_t row = (size_t) ei * a.n_ff + blockIdx.x * kCols + threadIdx.x;
    a.h[row] = swiglu_boundary(a.h[row], a.u[row], a.limit, a.round_bf16);
}
// grid (n_embd / 128): moe_down_kernel's combine
__global__ void __launch_bounds__(kCols) moe_combine_kernel(const KMoe a) {
    if (a.skip != nullptr && *a.skip) return;
    const int r = blockIdx.x * kCols + threadIdx.x;
    float p[8];
    for (int i = 0; i < a.k; ++i) p[i] = a.part[(size_t) i * a.n_embd + r];
    a.out[r] = combine(a.plan_w, p, a.k, a.cpu_flag != nullptr && *a.cpu_flag ? a.cpu_part + r : nullptr,
                       a.sh_out != nullptr ? a.sh_out[r] : 0.0f);
}
// grid (n_ff / 128, groups at most): win_gu_kernel's clamped swiglu, each entry of the group
__global__ void __launch_bounds__(kCols) win_swiglu_kernel(const KWin a) {
    const int grp = blockIdx.y + a.first_group;
    if (grp >= *a.n_groups) return;
    for (int m = a.grp_start[grp]; m < a.grp_start[grp + 1]; ++m) {
        const size_t row = (size_t) a.grp_ent[m] * a.n_ff + blockIdx.x * kCols + threadIdx.x;
        a.h[row] = swiglu_boundary(a.h[row], a.u[row], a.limit, a.round_bf16);
    }
}
// whether this GPU's device code runs those pieces as kernels of their own (EXL3_SEP_DEV above): gfx942 / gfx950,
// cached per device
static bool sep_sums() {
    thread_local int known[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    if (known[dev] == 0) {
        cudaDeviceProp p{};
        known[dev] = cudaGetDeviceProperties(&p, dev) == cudaSuccess &&
                     (std::strncmp(p.gcnArchName, "gfx942", 6) == 0 || std::strncmp(p.gcnArchName, "gfx950", 6) == 0)
                         ? 2 : 1;
    }
    return known[dev] == 2;
}
#define EXL3_MOE_SWIGLU(a, s) \
    do { if (sep_sums()) moe_swiglu_kernel<<<dim3((unsigned) ((a).n_ff / kCols), (unsigned) (a).k), kCols, 0, s>>>(a); } while (0)
#define EXL3_MOE_COMBINE(a, s) \
    do { if (sep_sums()) moe_combine_kernel<<<(unsigned) ((a).n_embd / kCols), kCols, 0, s>>>(a); } while (0)
#define EXL3_WIN_SWIGLU(a, groups, s) \
    do { if (sep_sums()) win_swiglu_kernel<<<dim3((unsigned) ((a).n_ff / kCols), (unsigned) (groups)), kCols, 0, s>>>(a); } while (0)
#else
#define EXL3_MOE_SWIGLU(a, s) (void) 0
#define EXL3_MOE_COMBINE(a, s) (void) 0
#define EXL3_WIN_SWIGLU(a, groups, s) (void) 0
#endif

#if defined(STRATA_USE_HIP)
// an input split's (xsplit) sums, added up in split order and then on as the unsplit kernels: block_out into h / u
// (gate / up) or the part rows (down).  grid (n_ff / 128, k): moe_gate_up_kernel's
__global__ void __launch_bounds__(kCols) moe_gu_red_kernel(const KMoe a) {
    __shared__ float red[2][kCols];
    const int ei = blockIdx.y, g = blockIdx.x, warp = threadIdx.x >> 5;
    const unsigned long long ptr = a.plan_ptr[ei];
    if (ptr == 0ull) return;
    if (a.skip != nullptr && *a.skip) return;
    for (int r = 0; r < 2; ++r)
        red[r][threadIdx.x] =
            split_sum(a.xp + ((size_t) ei * 2 + r) * a.gS * a.n_ff + g * kCols + threadIdx.x, a.gS, a.n_ff);
    __syncthreads();
    const uint8_t* blob = (const uint8_t*) ptr;
    if (warp < 2)
        block_out(red[warp], (const uint16_t*) (blob + a.svh[warp]) + g * kCols,
                  (warp == 0 ? a.h : a.u) + (size_t) ei * a.n_ff + g * kCols);
}
// grid (n_embd / 128, k): moe_down_kernel's
__global__ void __launch_bounds__(kCols) moe_down_red_kernel(const KMoe a) {
    __shared__ float red[kCols];
    const int ei = blockIdx.y, g = blockIdx.x;
    if (a.skip != nullptr && *a.skip) return;
    const unsigned long long ptr = a.plan_ptr[ei];
    float* part = a.part + (size_t) ei * a.n_embd + g * kCols;
    if (ptr == 0ull) {
        part[threadIdx.x] = 0.0f;
        return;
    }
    red[threadIdx.x] = split_sum(a.xp + (size_t) ei * a.dS * a.n_embd + g * kCols + threadIdx.x, a.dS, a.n_embd);
    __syncthreads();
    if (threadIdx.x < 32) block_out(red, (const uint16_t*) ((const uint8_t*) ptr + a.svh[2]) + g * kCols, part);
}
// grid (n_ff / 128, groups at most): win_gu_kernel's, each entry of the group
__global__ void __launch_bounds__(kCols) win_gu_red_kernel(const KWin a) {
    __shared__ float red[2][kCols];
    const int grp = blockIdx.y + a.first_group, g = blockIdx.x, warp = threadIdx.x >> 5;
    if (grp >= *a.n_groups) return;
    const uint8_t* blob = projection_base(a.grp_ptr[grp], warp < 2 ? warp : 0, a.indirect);
    for (int m = a.grp_start[grp]; m < a.grp_start[grp + 1]; ++m) {
        const size_t e = (size_t) a.grp_ent[m];
        for (int r = 0; r < 2; ++r)
            red[r][threadIdx.x] = split_sum(a.xp + (e * 2 + r) * a.gS * a.n_ff + g * kCols + threadIdx.x, a.gS, a.n_ff);
        __syncthreads();
        if (warp < 2)
            block_out(red[warp], (const uint16_t*) (blob + a.svh[warp]) + g * kCols,
                      (warp == 0 ? a.h : a.u) + e * a.n_ff + g * kCols);
        __syncthreads();
    }
}
// grid (n_embd / 128, groups at most): win_down_kernel's, each entry of the group into its part row
__global__ void __launch_bounds__(kCols) win_down_red_kernel(const KWin a) {
    __shared__ float red[kCols];
    const int grp = blockIdx.y + a.first_group, g = blockIdx.x;
    if (grp >= *a.n_groups) return;
    const uint8_t* blob = projection_base(a.grp_ptr[grp], 2, a.indirect);
    for (int m = a.grp_start[grp]; m < a.grp_start[grp + 1]; ++m) {
        const size_t e = (size_t) a.grp_ent[m];
        red[threadIdx.x] = split_sum(a.xp + e * a.dS * a.n_embd + g * kCols + threadIdx.x, a.dS, a.n_embd);
        __syncthreads();
        if (threadIdx.x < 32)
            block_out(red, (const uint16_t*) (blob + a.svh[2]) + g * kCols, a.part + e * a.n_embd + g * kCols);
        __syncthreads();
    }
}
#endif

// ---------------------------------------------------------------- a mixed-precision layer's experts (MixedLayouts)
// Each expert's layout by its class (the class whose VRAM range holds its blob), each part's K picked per block - the
// same arithmetic as the one-layout kernels above for an expert of that K.  K 1-4 (K2 2, 4, 6, 8).
struct KMix {   // (compact: 32 classes in ~1.8 KB of kernel arguments - a blob's offsets fit 32 bits)
    int n;
    unsigned long long lo[kMaxClasses], hi[kMaxClasses];
    unsigned char k2[kMaxClasses][3];
    unsigned tr[kMaxClasses][3], suh[kMaxClasses][3], svh[kMaxClasses][3];
};
static KMix kmix(const MixedLayouts& m) {
    KMix k{};
    k.n = m.n;
    for (int c = 0; c < m.n; ++c) {
        k.lo[c] = m.lo[c];
        k.hi[c] = m.hi[c];
        for (int r = 0; r < 3; ++r) {
            k.k2[c][r] = (unsigned char) m.lay[c].k2[r];
            k.tr[c][r] = (unsigned) m.lay[c].trellis[r];
            k.suh[c][r] = (unsigned) m.lay[c].suh[r];
            k.svh[c][r] = (unsigned) m.lay[c].svh[r];
        }
    }
    return k;
}
__device__ __forceinline__ int mix_cls(const KMix& m, unsigned long long p) {
    for (int c = 0; c < m.n; ++c)
        if (p >= m.lo[c] && p < m.hi[c]) return c;
    return 0;
}
union MixSmem {
    BlockSmem<2> k1;
    BlockSmem<4> k2;
    BlockSmem<6> k3;
    BlockSmem<8> k4;
};
__device__ __forceinline__ float* mix_out(int k2, MixSmem& u) {
    return k2 == 2 ? u.k1.out : k2 == 4 ? u.k2.out : k2 == 6 ? u.k3.out : u.k4.out;
}
// (the token and the window kernels inline it each their own way: their bits agree because the HIP build fuses
// multiply-adds only where the source writes them - CMakeLists.txt, -ffp-contract=on for this file)
template <int cb, class Input>
__device__ __forceinline__ void mix_gemv(int k2, const uint32_t* tr, int ld, int tc0, int k0, int k1, const Input& x,
                                         const uint16_t* suh, int nc, MixSmem& u) {
    switch (k2) {
    case 2: block_gemv<2, cb, Input, 1>(tr, ld, tc0, k0, k1, x, suh, nc, u.k1); break;
    case 4: block_gemv<4, cb, Input, 1>(tr, ld, tc0, k0, k1, x, suh, nc, u.k2); break;
    case 6: block_gemv<6, cb, Input, 1>(tr, ld, tc0, k0, k1, x, suh, nc, u.k3); break;
    default: block_gemv<8, cb, Input, 1>(tr, ld, tc0, k0, k1, x, suh, nc, u.k4); break;
    }
}

// moe_gate_up_kernel, each expert at its class's K
template <int cb>
__global__ void EXL3_BOUNDS(2) moe_gate_up_mix_kernel(const KMoe a, const KMix mx) {
    __shared__ __align__(16) MixSmem sm;
    __shared__ int last;
    const int ei = blockIdx.y, g = blockIdx.x, r = blockIdx.z;
    const unsigned long long ptr = a.plan_ptr[ei];
    if (ptr == 0ull) return;
    if (a.skip != nullptr && *a.skip) return;
    const int c = mix_cls(mx, ptr), k2 = mx.k2[c][r];
    const uint8_t* blob = projection_base(ptr, r, a.indirect);
    const int warp = threadIdx.x >> 5;
    Xin x;
    x.f = a.x;
    x.h = a.x16;
    x.bf16 = a.x_bf16;
    mix_gemv<cb>(k2, (const uint32_t*) (blob + mx.tr[c][r]), a.n_ff / 16, g * 8, 0, a.n_embd, x,
                 (const uint16_t*) (blob + mx.suh[c][r]), 1, sm);
    const size_t row = (size_t) ei * a.n_ff + g * kCols;
    if (warp == 0) block_out(mix_out(k2, sm), (const uint16_t*) (blob + mx.svh[c][r]) + g * kCols, (r == 0 ? a.h : a.u) + row);
    if (!pair_last(a.cnt_gu + ei * (a.n_ff / kCols) + g, last)) return;
    if (threadIdx.x < kCols)
        a.h[row + threadIdx.x] = swiglu_boundary(*(volatile const float*) &a.h[row + threadIdx.x],
                                        *(volatile const float*) &a.u[row + threadIdx.x], a.limit, a.round_bf16);
}

// moe_down_kernel, each expert at its class's K
template <int cb>
__global__ void EXL3_BOUNDS(2) moe_down_mix_kernel(const KMoe a, const KMix mx) {
    __shared__ __align__(16) MixSmem sm;
    __shared__ int last;
    const int ei = blockIdx.y, g = blockIdx.x;
    const unsigned long long ptr = a.plan_ptr[ei];
    const int warp = threadIdx.x >> 5;
    float* part = a.part + (size_t) ei * a.n_embd + g * kCols;
    if (a.skip != nullptr && *a.skip) return;
    if (ptr != 0ull) {
        const int c = mix_cls(mx, ptr), k2 = mx.k2[c][2];
        const uint8_t* blob = projection_base(ptr, 2, a.indirect);
        Xin x;
        x.f = a.h + (size_t) ei * a.n_ff;
        mix_gemv<cb>(k2, (const uint32_t*) (blob + mx.tr[c][2]), a.n_embd / 16, g * 8, 0, a.n_ff, x,
                     (const uint16_t*) (blob + mx.suh[c][2]), 1, sm);
        if (warp == 0) block_out(mix_out(k2, sm), (const uint16_t*) (blob + mx.svh[c][2]) + g * kCols, part);
    } else if (threadIdx.x < kCols) {
        part[threadIdx.x] = 0.0f;
    }
    // Fastllm hybrid scheduling reduces GPU and CPU routes together later.
    // Every CTA owns one (expert, output block), so no cross-expert counter is
    // needed. RunDown disables input splitting for this output mode.
    if (a.route_output) {
        __syncthreads();
        if (threadIdx.x < kCols) {
            const int r = g * kCols + threadIdx.x;
            a.out[(size_t) ei * a.n_embd + r] = a.plan_w[ei] * bf16_boundary(part[threadIdx.x], a.round_bf16);
        }
        return;
    }
#if defined(EXL3_SEP_DEV)
    return;   // (moe_combine_kernel)
#endif
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(&a.cnt[g], 1u) == (unsigned) (a.k - 1);
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (threadIdx.x < kCols) {
        const int r = g * kCols + threadIdx.x;
        float p[8];
        for (int i = 0; i < a.k; ++i) p[i] = bf16_boundary(*(volatile const float*) &a.part[(size_t) i * a.n_embd + r], a.round_bf16);
        a.out[r] = combine(a.plan_w, p, a.k, a.cpu_flag != nullptr && *a.cpu_flag ? a.cpu_part + r : nullptr,
                           a.sh_out != nullptr ? a.sh_out[r] : 0.0f);
    }
    if (threadIdx.x == 0) a.cnt[g] = 0u;
}

// win_gu_kernel / win_down_kernel, each group's expert at its class's K
template <int cb>
__global__ void EXL3_BOUNDS(2) win_gu_mix_kernel(const KWin a, const KMix mx) {
    __shared__ __align__(16) MixSmem sm;
    __shared__ int last;
    const int grp = blockIdx.y + a.first_group, g = blockIdx.x, r = blockIdx.z;
    if (grp >= *a.n_groups) return;
    const unsigned long long ptr = a.grp_ptr[grp];
    const int c = mix_cls(mx, ptr), k2 = mx.k2[c][r];
    const uint8_t* blob = projection_base(ptr, r, a.indirect);
    const int warp = threadIdx.x >> 5;
    const int m0 = a.grp_start[grp], m1 = a.grp_start[grp + 1];
    float* dst = r == 0 ? a.h : a.u;
    for (int mm = m0; mm < m1; mm += kTok) {
        const int nc = min(kTok, m1 - mm);
        Xin x;
        if (warp < nc) {
            const size_t offset = (size_t) (a.grp_ent[mm + warp] >> 3) * a.x_stride;
            x.f = a.x != nullptr ? a.x + offset : nullptr;
            x.h = a.x16 != nullptr ? a.x16 + offset : nullptr;
            x.bf16 = a.x_bf16;
        }
        mix_gemv<cb>(k2, (const uint32_t*) (blob + mx.tr[c][r]), a.n_ff / 16, g * 8, 0, a.n_embd, x,
                     (const uint16_t*) (blob + mx.suh[c][r]), nc, sm);
        if (warp < nc)
            block_out(mix_out(k2, sm) + warp * kCols, (const uint16_t*) (blob + mx.svh[c][r]) + g * kCols,
                      dst + (size_t) a.grp_ent[mm + warp] * a.n_ff + g * kCols);
        __syncthreads();
    }
    if (!pair_last(a.cnt_gu + grp * (a.n_ff / kCols) + g, last)) return;
    if (threadIdx.x < kCols)
        for (int m = m0; m < m1; ++m) {
            const size_t row = (size_t) a.grp_ent[m] * a.n_ff + g * kCols + threadIdx.x;
            a.h[row] = swiglu_boundary(*(volatile const float*) &a.h[row], *(volatile const float*) &a.u[row], a.limit, a.round_bf16);
        }
}
template <int cb>
__global__ void EXL3_BOUNDS(2) win_down_mix_kernel(const KWin a, const KMix mx) {
    __shared__ __align__(16) MixSmem sm;
    const int grp = blockIdx.y + a.first_group, g = blockIdx.x;
    if (grp >= *a.n_groups) return;
    const unsigned long long ptr = a.grp_ptr[grp];
    const int c = mix_cls(mx, ptr), k2 = mx.k2[c][2];
    const uint8_t* blob = projection_base(ptr, 2, a.indirect);
    const int warp = threadIdx.x >> 5;
    const int m0 = a.grp_start[grp], m1 = a.grp_start[grp + 1];
    for (int mm = m0; mm < m1; mm += kTok) {
        const int nc = min(kTok, m1 - mm);
        Xin x;
        if (warp < nc) x.f = a.h + (size_t) a.grp_ent[mm + warp] * a.n_ff;
        mix_gemv<cb>(k2, (const uint32_t*) (blob + mx.tr[c][2]), a.n_embd / 16, g * 8, 0, a.n_ff, x,
                     (const uint16_t*) (blob + mx.suh[c][2]), nc, sm);
        if (warp < nc)
            block_out(mix_out(k2, sm) + warp * kCols, (const uint16_t*) (blob + mx.svh[c][2]) + g * kCols,
                      a.part + (size_t) a.grp_ent[mm + warp] * a.n_embd + g * kCols);
        __syncthreads();
    }
}

__global__ void win_wait_kernel(const unsigned* p, unsigned v) {
    if (threadIdx.x != 0) return;
    while (*(volatile const unsigned*) p != v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(200);
#endif
    }
    __threadfence_system();
}

// ---------------------------------------------------------------- reconstruction
// grid (n / 128, k / 128): the 128 x 128 block of W = diag(suh) H W' H diag(svh), written transposed (dst[c][r])
template <int K2, int cb>
__global__ void __launch_bounds__(kThreads) recon_rows_kernel(const uint32_t* __restrict__ tr, int ld,
                                                              const uint16_t* __restrict__ suh,
                                                              const uint16_t* __restrict__ svh, int k, uint16_t* dst,
                                                              bool bf16) {
    constexpr int W = 4 * K2;
    constexpr int P = kCols + 2;   // padded row (halves)
    __shared__ __half tile[kCols * P];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nb = blockIdx.x, kb = blockIdx.y;
    // decode: warp w takes the 16-row slices w (8 tiles each)
    for (int ks = warp; ks < 8; ks += kWarps) {
        const uint32_t* src = tr + ((size_t) (kb * 8 + ks) * ld + nb * 8) * W;
        for (int t = 0; t < 8; ++t) {
            float v[8];
            decode8<K2, cb>(src + t * W, lane, v);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int r = ks * 16 + (lane & 3) * 2 + (j & 1) + 8 * ((j >> 1) & 1);
                const int c = t * 16 + (lane >> 2) + 8 * (j >> 2);
                tile[r * P + c] = __float2half_rn(v[j]);
            }
        }
    }
    __syncthreads();
    // H along the outputs (each row of 128 is one block)
    for (int r = warp; r < kCols; r += kWarps) {
        float v[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = __half2float(tile[r * P + lane * 4 + i]);
        had128(v, lane);
#pragma unroll
        for (int i = 0; i < 4; ++i) tile[r * P + lane * 4 + i] = __float2half_rn(v[i]);
    }
    __syncthreads();
    // H along the inputs, the scales, the transposed store
    for (int c = warp; c < kCols; c += kWarps) {
        float v[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = __half2float(tile[(lane * 4 + i) * P + c]);
        had128(v, lane);
        const float sv = h2f(svh[nb * kCols + c]);
        uint16_t* d = dst + (size_t) (nb * kCols + c) * k + kb * kCols + lane * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float w = v[i] * sv * h2f(suh[kb * kCols + lane * 4 + i]);
            if (bf16) {
                const uint32_t u = __float_as_uint(w);
                d[i] = (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);   // round to nearest even
            } else {
                d[i] = __half_as_ushort(__float2half_rn(w));
            }
        }
    }
}

// grid (n / 128, k / 16): W' row-major, one tile per warp
template <int K2, int cb>
__global__ void __launch_bounds__(kThreads) recon_inner_kernel(const uint32_t* __restrict__ tr, int ld, int n,
                                                               uint16_t* dst) {
    constexpr int W = 4 * K2;
    const int lane = threadIdx.x & 31, t = threadIdx.x >> 5;
    const int tc = blockIdx.x * 8 + t, ks = blockIdx.y;
    float v[8];
    decode8<K2, cb>(tr + ((size_t) ks * ld + tc) * W, lane, v);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int r = ks * 16 + (lane & 3) * 2 + (j & 1) + 8 * ((j >> 1) & 1);
        const int c = tc * 16 + (lane >> 2) + 8 * (j >> 2);
        dst[(size_t) r * n + c] = __half_as_ushort(__float2half_rn(v[j]));
    }
}

// grid (n / 128, k / 128): W' of the 128 x 128 block, transposed (dst[c][r]) - the decoded values alone.  Each warp
// decodes its slice's 8 tiles into the block's tile (half2 pairs of rows, conflict-free), then writes 256-byte rows.
template <int K2, int cb>
__global__ void __launch_bounds__(kThreads) inner_rows_kernel(const uint32_t* __restrict__ tr, int ld, int k,
                                                              uint16_t* __restrict__ dst) {
    constexpr int W = 4 * K2;
    constexpr int P = kCols + 8;   // a column's 128 rows, padded (halves; 16-byte rows)
    __shared__ __align__(16) __half tile[kCols * P];   // tile[c * P + r]
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int nb = blockIdx.x, kb = blockIdx.y;
    for (int ks = warp; ks < 8; ks += kWarps) {
        const uint32_t* src = tr + ((size_t) (kb * 8 + ks) * ld + nb * 8) * W;
        const int r = ks * 16 + 2 * t;
#pragma unroll
        for (int tc = 0; tc < 8; ++tc) {
            uint32_t f[4];
            frag8<K2, cb>(src + tc * W, lane, f);
            const int c = tc * 16 + g;
            *reinterpret_cast<uint32_t*>(&tile[c * P + r]) = f[0];
            *reinterpret_cast<uint32_t*>(&tile[c * P + r + 8]) = f[1];
            *reinterpret_cast<uint32_t*>(&tile[(c + 8) * P + r]) = f[2];
            *reinterpret_cast<uint32_t*>(&tile[(c + 8) * P + r + 8]) = f[3];
        }
    }
    __syncthreads();
    for (int c = warp; c < kCols; c += kWarps)
        *reinterpret_cast<uint2*>(dst + (size_t) (nb * kCols + c) * k + kb * kCols + lane * 4) =
            *reinterpret_cast<const uint2*>(&tile[c * P + lane * 4]);
}

// one warp per 128 values of a row: the rotated FP16 input of an EXL3 matrix (rows_rot_f16) ...
__global__ void __launch_bounds__(kThreads) rows_rot_f16_kernel(const float* __restrict__ x, int ldx,
                                                                const int* __restrict__ idx, int nrows, int ncols,
                                                                const uint16_t* __restrict__ suh, uint16_t* out) {
    const int lane = threadIdx.x & 31, nb = ncols / kCols;
    const int w = blockIdx.x * kWarps + (threadIdx.x >> 5);
    if (w >= nrows * nb) return;
    const int r = w / nb, b = w % nb;
    Xin xin;
    xin.f = x + (size_t) (idx != nullptr ? idx[r] : r) * ldx;
    float v[4], sc[4];
    chunk_load(xin, suh, b * kCols, lane, v, sc);
    chunk_store(v, sc, lane, reinterpret_cast<__half*>(out) + (size_t) r * ncols + b * kCols);
}
// ... and its output rotation in place (block_out's arithmetic)
__global__ void __launch_bounds__(kThreads) rows_rot_out_kernel(float* y, int ldy, int nrows, int ncols,
                                                                const uint16_t* __restrict__ svh) {
    const int lane = threadIdx.x & 31, nb = ncols / kCols;
    const int w = blockIdx.x * kWarps + (threadIdx.x >> 5);
    if (w >= nrows * nb) return;
    const int r = w / nb, b = w % nb;
    float* p = y + (size_t) r * ldy + b * kCols;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = p[lane * 4 + i];
    had128(v, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) p[lane * 4 + i] = __fmul_rn(v[i], h2f(svh[b * kCols + lane * 4 + i]));
}

__global__ void rows_f16_kernel(const float* __restrict__ x, int ldx, const int* __restrict__ idx, int ncols,
                                uint16_t* __restrict__ out) {
    const int r = blockIdx.y;
    const float* src = x + (size_t) (idx != nullptr ? idx[r] : r) * ldx;
    uint16_t* dst = out + (size_t) r * ncols;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < ncols; c += gridDim.x * blockDim.x)
        dst[c] = __half_as_ushort(__float2half_rn(src[c]));
}

// ---------------------------------------------------------------- the prompt path's tensor-core GEMMs (sm_80+)
// A block: BM rows x 128 outputs (one output rotation block).  Per 128 inputs the block's rows are scaled by suh,
// rotated and stored FP16 in shared memory (the A operands); per 16-row slice each warp decodes its 16 x 16 tile
// straight into mma.m16n8k16's B fragments - a lane's 8 steps ARE the fragment order (steps 0,1 / 2,3 rows 2t+{0,1} /
// 2t+8+{0,1} of column g, steps 4..7 the same of column g + 8) - and multiplies every m-tile of the block with them.
// The output rotation and svh run on the F32 tile in shared memory.
#if !defined(STRATA_USE_HIP)
namespace tc {

constexpr int BM = 32, MT = BM / 16, AP = 136;   // rows a block, its m-tiles, the A rows' pitch (halves)

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}
// the lane's 4 B registers of tile t (words staged in st): b[0..1] columns 0..7, b[2..3] columns 8..15
template <int K2, int cb>
__device__ __forceinline__ void tile_frags(const uint32_t* st, int lane, uint32_t* b) {
    uint32_t f[4];
    frag8<K2, cb>(st, lane, f);
#pragma unroll
    for (int i = 0; i < 4; ++i) b[i] = f[i];
}

// A[r][j] = FP16(H(x_r * suh)[j]), j < 128, for the block's rows (zero past nrows).  x_r: F32 row idx[r] of xf (ldx)
// when xf is given, else FP16 row r of xh (ldx).  Every warp, BM / 8 rows each.
__device__ __forceinline__ void load_a(const float* __restrict__ xf, const int* __restrict__ idx,
                                       const __half* __restrict__ xh, int ldx, int r0, int nrows, int k0,
                                       const uint16_t* __restrict__ suh, __half* A) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int rr = warp; rr < BM; rr += kWarps) {
        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const int r = r0 + rr;
        if (rr < nrows) {
            const int c = k0 + lane * 4;
            if (xf != nullptr) {
                const float* src = xf + (size_t) idx[r] * ldx + c;
#pragma unroll
                for (int i = 0; i < 4; ++i) v[i] = src[i];
            } else {
                const __half* src = xh + (size_t) r * ldx + c;
#pragma unroll
                for (int i = 0; i < 4; ++i) v[i] = __half2float(src[i]);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] *= h2f(suh[c + i]);
            had128(v, lane);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) A[rr * AP + lane * 4 + i] = __float2half_rn(v[i]);
    }
}

// the m-tile's A registers for the 16-column slice s of A
__device__ __forceinline__ void a_frags(const __half* A, int mt, int s, int lane, uint32_t* a) {
    const int g = lane >> 2, t = lane & 3;
    const __half* p = A + (mt * 16 + g) * AP + s * 16 + 2 * t;
    a[0] = *reinterpret_cast<const uint32_t*>(p);
    a[1] = *reinterpret_cast<const uint32_t*>(p + 8 * AP);
    a[2] = *reinterpret_cast<const uint32_t*>(p + 8);
    a[3] = *reinterpret_cast<const uint32_t*>(p + 8 * AP + 8);
}

// acc (this warp's 16 columns of the block's BM rows) -> T[BM][128] F32
__device__ __forceinline__ void store_acc(const float (&acc)[MT][2][4], float* T) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int ns = 0; ns < 2; ++ns) {
            const int c = warp * 16 + ns * 8 + 2 * t, r = mt * 16 + g;
            T[r * 128 + c] = acc[mt][ns][0];
            T[r * 128 + c + 1] = acc[mt][ns][1];
            T[(r + 8) * 128 + c] = acc[mt][ns][2];
            T[(r + 8) * 128 + c + 1] = acc[mt][ns][3];
        }
}

struct GArgs {
    const uint8_t* wbase;     // expert e's blob at wbase + e * stride (dense: 0, the offsets are pointers)
    size_t stride;
    size_t tr[3], suh[3], svh[3];
    int ld[3];                // tiles per 16-row slice of each matrix
    const int* bounds;        // device: expert e's rows [bounds[e], bounds[e + 1]) (null: one matrix, rows [0, m))
    int row0, m;              // local row of a sorted row: bounds[e] - row0 + r
    int k, n;                 // of the matrix (gate/up: n_embd, n_ff)
    const float* xf;          // F32 input rows by index (idx[local row]) ...
    const int* idx;
    const __half* xh;         // ... or FP16 rows by local row
    int ldx;
    float* y;                 // F32 output rows (ldy), y = beta y + out ...
    int ldy;
    float beta;
    __half* h;                // ... or (gate/up) the swiglu as FP16 rows (ldh)
    int ldh;
    float limit;
};

// one matrix: y = beta y + H(H(x * suh) W') * svh.  grid (row blocks, n / 128, experts)
template <int K2, int cb>
__global__ void __launch_bounds__(kThreads) gemm_kernel(const GArgs a) {
    constexpr int W = 4 * K2;
    __shared__ __align__(16) unsigned char sm[BM * 128 * 4];   // A (BM x AP halves), then the F32 tile
    __shared__ uint32_t stage[kWarps][64];
    __half* A = reinterpret_cast<__half*>(sm);
    float* T = reinterpret_cast<float*>(sm);
    const int e = blockIdx.z, rb = blockIdx.x, g = blockIdx.y;
    const int e0 = a.bounds != nullptr ? a.bounds[e] - a.row0 : 0;
    const int ne = a.bounds != nullptr ? a.bounds[e + 1] - a.bounds[e] : a.m;
    const int r0 = e0 + rb * BM, nrows = min(BM, ne - rb * BM);
    if (nrows <= 0) return;
    const uint8_t* blob = a.wbase + (size_t) e * a.stride;
    const uint32_t* tr = (const uint32_t*) (blob + a.tr[0]);
    const uint16_t* suh = (const uint16_t*) (blob + a.suh[0]);
    const uint16_t* svh = (const uint16_t*) (blob + a.svh[0]);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float acc[MT][2][4] = {};
    for (int k0 = 0; k0 < a.k; k0 += 128) {
        __syncthreads();
        load_a(a.xf, a.idx, a.xh, a.ldx, r0, nrows, k0, suh, A);
        __syncthreads();
#pragma unroll 1
        for (int s = 0; s < 8; ++s) {
            const int ks = k0 / 16 + s;
            const uint32_t* src = tr + ((size_t) ks * a.ld[0] + g * 8 + warp) * W;
            __syncwarp();
            for (int i = lane; i < W; i += 32) stage[warp][i] = src[i];
            __syncwarp();
            uint32_t b[4];
            tile_frags<K2, cb>(stage[warp], lane, b);
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                uint32_t af[4];
                a_frags(A, mt, s, lane, af);
                mma16816(acc[mt][0], af, b[0], b[1]);
                mma16816(acc[mt][1], af, b[2], b[3]);
            }
        }
    }
    __syncthreads();
    store_acc(acc, T);
    __syncthreads();
    for (int rr = warp; rr < nrows; rr += kWarps) {
        float v[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = T[rr * 128 + lane * 4 + i];
        had128(v, lane);
        float* yr = a.y + (size_t) (r0 + rr) * a.ldy + g * 128 + lane * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float o = v[i] * h2f(svh[g * 128 + lane * 4 + i]);
            yr[i] = a.beta != 0.0f ? a.beta * yr[i] + o : o;
        }
    }
}

// gate and up (matrices 0 and 1) over the same rows, the clamped swiglu, FP16 h.  grid (row blocks, n / 128, experts)
template <int K2, int cb>
__global__ void __launch_bounds__(kThreads) gate_up_kernel(const GArgs a) {
    constexpr int W = 4 * K2;
    __shared__ __align__(16) unsigned char sm[2 * BM * 128 * 4];   // A_g, A_u, then the two F32 tiles
    __shared__ uint32_t stage[kWarps][64];
    __half* A[2] = {reinterpret_cast<__half*>(sm), reinterpret_cast<__half*>(sm) + BM * AP};
    float* T[2] = {reinterpret_cast<float*>(sm), reinterpret_cast<float*>(sm) + BM * 128};
    const int e = blockIdx.z, rb = blockIdx.x, g = blockIdx.y;
    const int e0 = a.bounds[e] - a.row0, ne = a.bounds[e + 1] - a.bounds[e];
    const int r0 = e0 + rb * BM, nrows = min(BM, ne - rb * BM);
    if (nrows <= 0) return;
    const uint8_t* blob = a.wbase + (size_t) e * a.stride;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float acc[2][MT][2][4] = {};
    for (int k0 = 0; k0 < a.k; k0 += 128) {
        __syncthreads();
        for (int m = 0; m < 2; ++m)
            load_a(a.xf, a.idx, a.xh, a.ldx, r0, nrows, k0, (const uint16_t*) (blob + a.suh[m]), A[m]);
        __syncthreads();
#pragma unroll 1
        for (int s = 0; s < 8; ++s) {
            const int ks = k0 / 16 + s;
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                const uint32_t* src = (const uint32_t*) (blob + a.tr[m]) + ((size_t) ks * a.ld[m] + g * 8 + warp) * W;
                __syncwarp();
                for (int i = lane; i < W; i += 32) stage[warp][i] = src[i];
                __syncwarp();
                uint32_t b[4];
                tile_frags<K2, cb>(stage[warp], lane, b);
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    uint32_t af[4];
                    a_frags(A[m], mt, s, lane, af);
                    mma16816(acc[m][mt][0], af, b[0], b[1]);
                    mma16816(acc[m][mt][1], af, b[2], b[3]);
                }
            }
        }
    }
    __syncthreads();
    store_acc(acc[0], T[0]);
    store_acc(acc[1], T[1]);
    __syncthreads();
    const uint16_t* svg = (const uint16_t*) (a.wbase + (size_t) e * a.stride + a.svh[0]) + g * 128;
    const uint16_t* svu = (const uint16_t*) (a.wbase + (size_t) e * a.stride + a.svh[1]) + g * 128;
    for (int rr = warp; rr < nrows; rr += kWarps) {
        float vg[4], vu[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            vg[i] = T[0][rr * 128 + lane * 4 + i];
            vu[i] = T[1][rr * 128 + lane * 4 + i];
        }
        had128(vg, lane);
        had128(vu, lane);
        __half* hr = a.h + (size_t) (r0 + rr) * a.ldh + g * 128 + lane * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float gv = vg[i] * h2f(svg[lane * 4 + i]), uv = vu[i] * h2f(svu[lane * 4 + i]);
            if (a.limit > 0.0f) {
                gv = fminf(gv, a.limit);
                uv = fminf(fmaxf(uv, -a.limit), a.limit);
            }
            hr[i] = __float2half_rn((gv / (1.0f + expf(-gv))) * uv);
        }
    }
}

}  // namespace tc
#endif

// The prompt path's portable kernels (#298): the tested 8-activation GEMV decoder (block_gemv) on the CUDA cores, F32
// sums - a block owns one expert, up to eight of its routed rows and 128 outputs; no host bounds readback, no expanded
// weights.  AMD runs them (no tensor-core port yet), and NVIDIA before sm_80 (V100, T4, RTX 20: the tensor-core
// kernels below need mma.m16n8k16), which had no prompt path for EXL3 experts before.
template <int K2, int cb, bool gate_up>
__global__ void __launch_bounds__(kThreads) hip_set_kernel(SetArgs a, const float* x, int ldx,
                                                        const int* idx, uint16_t* h, int ldh,
                                                        float* y, int ldy) {
    __shared__ __align__(16) BlockSmem<K2> sm;
    __shared__ float gate[kTok * kCols];
    const int e = blockIdx.z, g = blockIdx.y, warp = threadIdx.x >> 5;
    const int start = a.bounds[e], end = a.bounds[e + 1];
    const int row = start - a.row0 + (int) blockIdx.x * kTok;
    const int nc = min(kTok, end - start - (int) blockIdx.x * kTok);
    if (nc <= 0) return;
    const uint8_t* blob = a.wbase + (size_t) e * a.stride;
    if constexpr (gate_up) {
        Xin in;
        if (warp < nc) in.f = x + (size_t) idx[row + warp] * ldx;
        block_gemv<K2, cb>((const uint32_t*) (blob + a.lay.trellis[0]), a.n_ff / 16,
                          g * 8, 0, a.n_embd, in, (const uint16_t*) (blob + a.lay.suh[0]), nc, sm);
        if (warp < nc)
            block_out(sm.out + warp * kCols, (const uint16_t*) (blob + a.lay.svh[0]) + g * kCols,
                      gate + warp * kCols);
        __syncthreads();
        block_gemv<K2, cb>((const uint32_t*) (blob + a.lay.trellis[1]), a.n_ff / 16,
                          g * 8, 0, a.n_embd, in, (const uint16_t*) (blob + a.lay.suh[1]), nc, sm);
        if (warp < nc) {
            const int lane = threadIdx.x & 31;
            float v[4];
            for (int i = 0; i < 4; ++i) v[i] = sm.out[warp * kCols + lane * 4 + i];
            had128(v, lane);
            for (int i = 0; i < 4; ++i) {
                const int c = lane * 4 + i;
                const float up = v[i] * h2f(((const uint16_t*) (blob + a.lay.svh[1]))[g * kCols + c]);
                h[(size_t) (row + warp) * ldh + g * kCols + c] =
                    __half_as_ushort(__float2half_rn(swiglu(gate[warp * kCols + c], up, a.limit)));
            }
        }
    } else {
        XinHalf in;
        if (warp < nc) in.h = h + (size_t) (row + warp) * ldh;
        block_gemv<K2, cb>((const uint32_t*) (blob + a.lay.trellis[2]), a.n_embd / 16,
                          g * 8, 0, a.n_ff, in, (const uint16_t*) (blob + a.lay.suh[2]), nc, sm);
        if (warp < nc)
            block_out(sm.out + warp * kCols, (const uint16_t*) (blob + a.lay.svh[2]) + g * kCols,
                      y + (size_t) (row + warp) * ldy + g * kCols);
    }
}

#if defined(STRATA_USE_HIP)
// Wider scalar tiles reuse each decoded fragment across more routed rows.
// Keep DOT2/FMA order, rotations and output rounding identical to block_gemv.
template <int K2, int NR> struct HipScalarSmem {
    __half xb[2 * NR * kCols];
    uint32_t stage[kWarps * 4 * K2];
    float out[NR * kCols];
};
template <int K2, int cb, int NR, class Input = Xin>
__device__ void hip_scalar_gemv(const uint32_t* __restrict__ tr, int ld, int tc0, int k0, int k1, const Input (&x)[(NR + 7) / 8],
                           const uint16_t* __restrict__ suh, int nc, HipScalarSmem<K2, NR>& sm) {
    constexpr int W = Words<K2>::W, R = Words<K2>::R;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const size_t sstride = (size_t) ld * W;
    const uint32_t* tile = tr + ((size_t) (k0 / 16) * ld + tc0 + warp) * W;
    uint32_t* st = sm.stage + warp * W;

    const int nq = (k1 - k0) / kCols;
    uint32_t wc[8][R], wn[8][R];
    float xv[(NR + 7) / 8][4], xs4[(NR + 7) / 8][4];
    load_tiles<K2>(tile, sstride, lane, wc);
    for (int j = 0; j < (NR + 7) / 8; ++j)
        if (warp + j * 8 < nc) chunk_load(x[j], suh, k0, lane, xv[j], xs4[j]);
    float acc[NR][2];
#pragma unroll
    for (int cc = 0; cc < NR; ++cc) acc[cc][0] = acc[cc][1] = 0.0f;

    for (int q = 0; q < nq; ++q) {
        __half* xq = sm.xb + (q & 1) * (NR * kCols);
        for (int j = 0; j < (NR + 7) / 8; ++j)
            if (warp + j * 8 < nc) chunk_store(xv[j], xs4[j], lane, xq + (warp + j * 8) * kCols);
        __syncthreads();
        if (q + 1 < nq) {
            load_tiles<K2>(tile + (size_t) (q + 1) * 8 * sstride, sstride, lane, wn);
            for (int j = 0; j < (NR + 7) / 8; ++j)
                if (warp + j * 8 < nc) chunk_load(x[j], suh, k0 + (q + 1) * kCols, lane, xv[j], xs4[j]);
        }
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            uint32_t f[4];
            slice_frags<K2, cb>(wc[s], st, lane, f);
#if defined(EXL3_DOT2)
#pragma unroll
            for (int cc = 0; cc < NR; ++cc)
                if (cc < nc) {
                    const __half* xr = xq + cc * kCols + s * 16 + 2 * t;
                    const uint32_t xa = *reinterpret_cast<const uint32_t*>(xr);
                    const uint32_t xb = *reinterpret_cast<const uint32_t*>(xr + 8);
                    acc[cc][0] = dot2_acc(f[1], xb, dot2_acc(f[0], xa, acc[cc][0]));
                    acc[cc][1] = dot2_acc(f[3], xb, dot2_acc(f[2], xa, acc[cc][1]));
                }
#else
            float v[8];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 d = __half22float2(u2h(f[i]));
                v[2 * i] = d.x;
                v[2 * i + 1] = d.y;
            }
#pragma unroll
            for (int cc = 0; cc < NR; ++cc)
                if (cc < nc) {
                    const __half* xr = xq + cc * kCols + s * 16 + 2 * t;
                    const float x0 = __half2float(xr[0]), x1 = __half2float(xr[1]);
                    const float x2 = __half2float(xr[8]), x3 = __half2float(xr[9]);
                    acc[cc][0] = dot4_acc(acc[cc][0], v, x0, x1, x2, x3);
                    acc[cc][1] = dot4_acc(acc[cc][1], v + 4, x0, x1, x2, x3);
                }
#endif

        }
#pragma unroll
        for (int s = 0; s < 8; ++s)
#pragma unroll
            for (int r = 0; r < R; ++r) wc[s][r] = wn[s][r];
    }
    const int o = warp * 16 + g;
    // the 4 lanes of one g hold parts of the same two outputs
#pragma unroll
    for (int cc = 0; cc < NR; ++cc)
        if (cc < nc) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                acc[cc][h] += __shfl_xor_sync(0xffffffffu, acc[cc][h], 1);
                acc[cc][h] += __shfl_xor_sync(0xffffffffu, acc[cc][h], 2);
            }
            if (t == 0) {
                sm.out[cc * kCols + o] = acc[cc][0];
                sm.out[cc * kCols + o + 8] = acc[cc][1];
            }
        }

    __syncthreads();
}

template <int K2, int cb, bool gate_up, int NR>
__global__ void __launch_bounds__(kThreads) hip_scalar_set_kernel(SetArgs a, const float* x, int ldx,
                                                          const int* idx, uint16_t* h, int ldh,
                                                          float* y, int ldy) {
    __shared__ __align__(16) HipScalarSmem<K2, NR> sm;
    __shared__ float gate[NR * kCols];
    const int e = blockIdx.z, g = blockIdx.y, warp = threadIdx.x >> 5;
    const int start = a.bounds[e], end = a.bounds[e + 1];
    const int row = start - a.row0 + (int) blockIdx.x * NR;
    const int nc = min(NR, end - start - (int) blockIdx.x * NR);
    if (nc <= 0) return;
    const uint8_t* blob = a.wbase + (size_t) e * a.stride;
    if constexpr (gate_up) {
        Xin in[(NR + 7) / 8];
        for (int j = 0; j < (NR + 7) / 8; ++j)
            if (warp + j * 8 < nc) in[j].f = x + (size_t) idx[row + warp + j * 8] * ldx;
        hip_scalar_gemv<K2, cb, NR>((const uint32_t*) (blob + a.lay.trellis[0]), a.n_ff / 16,
                              g * 8, 0, a.n_embd, in, (const uint16_t*) (blob + a.lay.suh[0]), nc, sm);
        for (int j = 0; j < (NR + 7) / 8; ++j) {
            const int c = warp + j * 8;
            if (c < nc) block_out(sm.out + c * kCols, (const uint16_t*) (blob + a.lay.svh[0]) + g * kCols,
                                  gate + c * kCols);
        }
        __syncthreads();
        hip_scalar_gemv<K2, cb, NR>((const uint32_t*) (blob + a.lay.trellis[1]), a.n_ff / 16,
                              g * 8, 0, a.n_embd, in, (const uint16_t*) (blob + a.lay.suh[1]), nc, sm);
        for (int j = 0; j < (NR + 7) / 8; ++j) {
            const int r = warp + j * 8;
            if (r < nc) {
                const int lane = threadIdx.x & 31;
                float v[4];
                for (int i = 0; i < 4; ++i) v[i] = sm.out[r * kCols + lane * 4 + i];
                had128(v, lane);
                for (int i = 0; i < 4; ++i) {
                    const int c = lane * 4 + i;
                    const float up = v[i] * h2f(((const uint16_t*) (blob + a.lay.svh[1]))[g * kCols + c]);
                    h[(size_t) (row + r) * ldh + g * kCols + c] =
                        __half_as_ushort(__float2half_rn(swiglu(gate[r * kCols + c], up, a.limit)));
                }
            }
        }
    } else {
        XinHalf in[(NR + 7) / 8];
        for (int j = 0; j < (NR + 7) / 8; ++j)
            if (warp + j * 8 < nc) in[j].h = h + (size_t) (row + warp + j * 8) * ldh;
        hip_scalar_gemv<K2, cb, NR>((const uint32_t*) (blob + a.lay.trellis[2]), a.n_embd / 16,
                              g * 8, 0, a.n_ff, in, (const uint16_t*) (blob + a.lay.suh[2]), nc, sm);
        for (int j = 0; j < (NR + 7) / 8; ++j) {
            const int c = warp + j * 8;
            if (c < nc) block_out(sm.out + c * kCols, (const uint16_t*) (blob + a.lay.svh[2]) + g * kCols,
                                  y + (size_t) (row + c) * ldy + g * kCols);
        }
    }
}
#endif

// set_mix_kernel's share of SetArgs: the fields it reads.  The whole SetArgs (its MixedLayouts ~3.9 KB) beside KMix
// made 5.8 KB of kernel parameters - over the 4 KB Pascal (sm_6x) allows, so a Pascal build stopped there (#479;
// Volta and newer take 32 KB).  The same values, the same arithmetic: the same bits everywhere.
struct KSetMix {
    const unsigned long long* eptr;
    const int* bounds;
    int row0, n_embd, n_ff;
    float limit;
};
static KSetMix ksetmix(const SetArgs& a) { return KSetMix{a.eptr, a.bounds, a.row0, a.n_embd, a.n_ff, a.limit}; }
static_assert(sizeof(KSetMix) + sizeof(KMix) + 6 * sizeof(void*) + 3 * sizeof(int) <= 4096,
              "set_mix_kernel's parameters must fit the 4 KB Pascal (sm_6x) allows");

// hip_set_kernel for a mixed-precision layer: expert e's blob at a.eptr[e], each part at its class's K
template <int cb, bool gate_up>
__global__ void __launch_bounds__(kThreads) set_mix_kernel(const KSetMix a, KMix mx, const float* x, int ldx,
                                                        const int* idx, uint16_t* h, int ldh,
                                                        float* y, int ldy) {
    __shared__ __align__(16) MixSmem sm;
    __shared__ float gate[kTok * kCols];
    const int e = blockIdx.z, g = blockIdx.y, warp = threadIdx.x >> 5;
    const int start = a.bounds[e], end = a.bounds[e + 1];
    const int row = start - a.row0 + (int) blockIdx.x * kTok;
    const int nc = min(kTok, end - start - (int) blockIdx.x * kTok);
    if (nc <= 0) return;
    const unsigned long long ptr = a.eptr[e];
    const uint8_t* blob = (const uint8_t*) ptr;
    const int c = mix_cls(mx, ptr);
    if constexpr (gate_up) {
        Xin in;
        if (warp < nc) in.f = x + (size_t) idx[row + warp] * ldx;
        const int kg = mx.k2[c][0], ku = mx.k2[c][1];
        mix_gemv<cb>(kg, (const uint32_t*) (blob + mx.tr[c][0]), a.n_ff / 16, g * 8, 0, a.n_embd, in,
                     (const uint16_t*) (blob + mx.suh[c][0]), nc, sm);
        if (warp < nc)
            block_out(mix_out(kg, sm) + warp * kCols, (const uint16_t*) (blob + mx.svh[c][0]) + g * kCols,
                      gate + warp * kCols);
        __syncthreads();
        mix_gemv<cb>(ku, (const uint32_t*) (blob + mx.tr[c][1]), a.n_ff / 16, g * 8, 0, a.n_embd, in,
                     (const uint16_t*) (blob + mx.suh[c][1]), nc, sm);
        if (warp < nc) {
            const int lane = threadIdx.x & 31;
            const float* out = mix_out(ku, sm);
            float v[4];
            for (int i = 0; i < 4; ++i) v[i] = out[warp * kCols + lane * 4 + i];
            had128(v, lane);
            for (int i = 0; i < 4; ++i) {
                const int cc = lane * 4 + i;
                const float up = v[i] * h2f(((const uint16_t*) (blob + mx.svh[c][1]))[g * kCols + cc]);
                h[(size_t) (row + warp) * ldh + g * kCols + cc] =
                    __half_as_ushort(__float2half_rn(swiglu(gate[warp * kCols + cc], up, a.limit)));
            }
        }
    } else {
        XinHalf in;
        if (warp < nc) in.h = h + (size_t) (row + warp) * ldh;
        const int kd = mx.k2[c][2];
        mix_gemv<cb>(kd, (const uint32_t*) (blob + mx.tr[c][2]), a.n_embd / 16, g * 8, 0, a.n_ff, in,
                     (const uint16_t*) (blob + mx.suh[c][2]), nc, sm);
        if (warp < nc)
            block_out(mix_out(kd, sm) + warp * kCols, (const uint16_t*) (blob + mx.svh[c][2]) + g * kCols,
                      y + (size_t) (row + warp) * ldy + g * kCols);
    }
}

// ---------------------------------------------------------------- the prompt path on RDNA matrix cores
// gfx11's v_wmma_f32_16x16x16_f16: hip_set_kernel's work with NR (16 or 32) routed rows a block - each weight tile
// decoded once for all of them - and the products on the matrix cores.  A warp's decoded tile goes through LDS into
// A's layout (lane m and m + 16: output m's 16 inputs); each 16 activations of the slice are a B (lane n and n + 16:
// activation n's 16 inputs); D's element i is output 2i + lane / 16 of the tile for activation lane % 16.  The host
// uses this layout on gfx11; the gfx12 implementation follows below. Both provide physical row tiles of 16 or 32
// (by default 16). The decode and verify kernels keep
// block_gemv (their bits are the token path's).
// Volta / Turing (sm_70 .. sm_75: no mma.m16n8k16, so not the tensor-core kernels below) run the same kernel on
// nvcuda::wmma's 16x16x16 FP16 product with F32 sums: A the warp's decoded tile from wt (row-major, output by input),
// B 16 activations' slice from xb (col-major: an activation's inputs in a row), D stored col-major into out - the same
// operands as gfx11's, the fragments loaded by the library instead of by hand (wt's pitch 12 words: wmma's ldm is a
// multiple of 8 halves; the 8 rows a store takes stay on distinct banks).  wmma_now(): sm_7x unless
// Fastllm selects supported matrix kernels without application environment overrides.
constexpr int kXS = kCols + 8;   // an activation's pitch in xb (halves: 2-way LDS reads)
#if defined(STRATA_USE_HIP)
constexpr int kWtP = 9;          // wt's pitch, words a decoded output row (8 + 1: banks)
#else
constexpr int kWtP = 12;
#endif
template <int K2, int NR>
struct WmmaSmem {   // (out, xb and wt at the same offsets for every K2: the mixed kernel keeps one for all classes)
    float out[NR * kCols];
    __half xb[2 * NR * kXS];
    uint32_t wt[kWarps * 16 * kWtP];   // a warp's decoded tile by output, 8 words a row (kWtP apart)
    uint32_t stage[kWarps * 4 * K2];
};
#if defined(STRATA_USE_HIP) && defined(__GFX11__)
#define EXL3_WMMA 1
typedef _Float16 exl3_h16 __attribute__((ext_vector_type(16)));
typedef float exl3_f8 __attribute__((ext_vector_type(8)));
typedef uint32_t exl3_u8 __attribute__((ext_vector_type(8)));
#endif
#if !defined(STRATA_USE_HIP) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 800
#define EXL3_WMMA_V70 1
#endif

// RDNA4's WMMA has eight FP16 inputs per lane and a different accumulator layout.
// Keep compressed weight words in registers; decode only the tile being consumed.
// Padded activation rows retain 8-byte alignment for two exact 64-bit LDS loads.
// Numerical contract: the EXL3 weights, FP16 operands (including chunk_store's activation rounding),
// and FP32 accumulation are the same formats as block_gemv's RDNA4 DOT2 path. WMMA groups the products
// and additions differently from DOT2's partial accumulators and warp reduction, so rounding and
// low output bits can differ. Full-model routing can amplify those differences in later layers.
// Bitwise equality with the DOT2 path is therefore not the accuracy contract; the existing
// CPU-reference tolerances, finite checks and output guards remain the operator contract.
#if defined(STRATA_USE_HIP) && defined(__GFX12__)
#define EXL3_WMMA_RDNA4 1
typedef _Float16 rdna4_h8 __attribute__((ext_vector_type(8)));
typedef float rdna4_f8 __attribute__((ext_vector_type(8)));
constexpr int kRdna4Stride = kCols + 4;
template <int NR> struct Rdna4Smem {
    __half xb[NR * kRdna4Stride];
    float out[NR * kCols];
};
__device__ __forceinline__ rdna4_h8 rdna4_load_operand(const __half* p, bool valid) {
    if (!valid) return rdna4_h8{};
    const uint2 lo = *reinterpret_cast<const uint2*>(p);
    const uint2 hi = *reinterpret_cast<const uint2*>(p + 4);
    return __builtin_bit_cast(rdna4_h8, make_uint4(lo.x, lo.y, hi.x, hi.y));
}
template <int K2>
__device__ __forceinline__ uint32_t rdna4_tile_word(const uint32_t (&w)[Words<K2>::R], int index) {
    index %= Words<K2>::W;
    const uint32_t lo = __shfl_sync(0xffffffffu, w[0], index & 31);
    if constexpr (Words<K2>::R > 1) {
        const uint32_t hi = __shfl_sync(0xffffffffu, w[1], index & 31);
        return index < 32 ? lo : hi;
    }
    return lo;
}
template <int K2, int cb, int Rows, class Input>
__device__ void rdna4_block_wmma(const uint32_t* tr, int ld, int tc0, int k0, int k1, const Input (&x)[Rows / kWarps],
                              const uint16_t* suh, int nc, Rdna4Smem<Rows>& sm) {
    constexpr int W = Words<K2>::W, R = Words<K2>::R;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, col = lane & 15;
    const int kh = (lane >> 4) * 8;
    const size_t stride = (size_t) ld * W;
    rdna4_f8 acc[Rows / 16] = {};
    for (int q = k0; q < k1; q += kCols) {
#pragma unroll
        for (int j = 0; j < Rows / kWarps; ++j) {
            const int row = warp + j * kWarps;
            if (row < nc) {
                float v[4], scales[4];
                chunk_load(x[j], suh, q, lane, v, scales);
                chunk_store(v, scales, lane, sm.xb + row * kRdna4Stride);
            }
        }
        __syncthreads();
        uint32_t words[8][R];
        load_tiles<K2>(tr + ((size_t) (q / 16) * ld + tc0 + warp) * W, stride, lane, words);
#pragma unroll
        for (int step = 0; step < 8; ++step) {
            rdna4_h8 av, bv;
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const int row = kh + pair * 2;
                const int p = (col & 7) * 32 + (col & 8) / 2 + (row & 6) * 4 + (row & 8) / 4;
                // K2 = twice the bitrate, including half-integer codes.
                const int b0 = (p + 257) * K2 / 2 - 16, b2 = (p + 258) * K2 / 2;
                const int i0 = b0 >> 5, i2 = (b2 - 1) >> 5, shift = (i2 + 1) * 32 - b2;
                const uint32_t lo = rdna4_tile_word<K2>(words[step], i0), hi = rdna4_tile_word<K2>(words[step], i2);
                const uint32_t bits = h2u(decode2h<cb>(fsh(hi, lo, shift + (K2 + 1) / 2) & 0xffffu, fsh(hi, lo, shift) & 0xffffu));
                bv[2 * pair] = __builtin_bit_cast(_Float16, (uint16_t) bits);
                bv[2 * pair + 1] = __builtin_bit_cast(_Float16, (uint16_t) (bits >> 16));
            }
#pragma unroll
            for (int m = 0; m < Rows / 16; ++m) {
                av = rdna4_load_operand(sm.xb + (m * 16 + col) * kRdna4Stride + step * 16 + kh, m * 16 + col < nc);
                acc[m] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(av, bv, acc[m]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = 8 * (lane >> 4) + i;
#pragma unroll
        for (int m = 0; m < Rows / 16; ++m)
            if (m * 16 + row < nc) sm.out[(m * 16 + row) * kCols + warp * 16 + col] = acc[m][i];
    }
    __syncthreads();
}

#endif

// block_gemv for nc <= NR activations (warp w loads activations w, w + 8, ..: in[j] the one at w + 8 j):
// sm.out[n * kCols + o]
template <int K2, int cb, int KS, int NR, class Input, class Smem>
__device__ void block_gemm_w(const uint32_t* __restrict__ tr, int ld, int tc0, int k0, int k1, const Input (&in)[NR / 8],
                             const uint16_t* __restrict__ suh, int nc, Smem& sm) {
#if defined(EXL3_WMMA_RDNA4)
    rdna4_block_wmma<K2, cb, NR>(tr, ld, tc0, k0, k1, in, suh, nc, sm);
#elif defined(EXL3_WMMA)
    static_assert(K2 <= KS, "the shared memory's stage holds K2 words a lane");
    constexpr int W = Words<K2>::W, R = Words<K2>::R, NJ = NR / 8, NB = NR / 16;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3, m = lane & 15;
    const size_t sstride = (size_t) ld * W;
    const uint32_t* tile = tr + ((size_t) (k0 / 16) * ld + tc0 + warp) * W;
    uint32_t* st = sm.stage + warp * W;
    uint32_t* wt = sm.wt + warp * 16 * 9;
    const int nq = (k1 - k0) / kCols;
    uint32_t wc[8][R], wn[8][R];
    float xv[NJ][4], xs[NJ][4];
    load_tiles<K2>(tile, sstride, lane, wc);
#pragma unroll
    for (int j = 0; j < NJ; ++j)
        if (warp + 8 * j < nc) chunk_load(in[j], suh, k0, lane, xv[j], xs[j]);
    exl3_f8 c[NB];
#pragma unroll
    for (int b = 0; b < NB; ++b) c[b] = exl3_f8{};
    for (int q = 0; q < nq; ++q) {
        __half* xq = sm.xb + (q & 1) * (NR * kXS);
#pragma unroll
        for (int j = 0; j < NJ; ++j)
            if (warp + 8 * j < nc) chunk_store(xv[j], xs[j], lane, xq + (warp + 8 * j) * kXS);
        __syncthreads();
        if (q + 1 < nq) {
            load_tiles<K2>(tile + (size_t) (q + 1) * 8 * sstride, sstride, lane, wn);
#pragma unroll
            for (int j = 0; j < NJ; ++j)
                if (warp + 8 * j < nc) chunk_load(in[j], suh, k0 + (q + 1) * kCols, lane, xv[j], xs[j]);
        }
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            uint32_t f[4];
            slice_frags<K2, cb>(wc[s], st, lane, f);
            // f: outputs g (f[0], f[1]) and g + 8 (f[2], f[3]), inputs 2t, 2t + 1 (f[0], f[2]) / 2t + 8, 2t + 9
            wt[g * 9 + t] = f[0];
            wt[g * 9 + 4 + t] = f[1];
            wt[(g + 8) * 9 + t] = f[2];
            wt[(g + 8) * 9 + 4 + t] = f[3];
            __syncwarp();
            exl3_u8 aw;
#pragma unroll
            for (int i = 0; i < 8; ++i) aw[i] = wt[m * 9 + i];
            __syncwarp();   // (wt: the next slice's)
            const exl3_h16 a = __builtin_bit_cast(exl3_h16, aw);
#pragma unroll
            for (int b = 0; b < NB; ++b) {   // (activations past nc: whatever xb holds - their outputs are not written)
                const uint32_t* xr = reinterpret_cast<const uint32_t*>(xq + (b * 16 + m) * kXS + s * 16);
                exl3_u8 bw;
#pragma unroll
                for (int i = 0; i < 8; ++i) bw[i] = xr[i];
                c[b] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, __builtin_bit_cast(exl3_h16, bw), c[b]);
            }
        }
#pragma unroll
        for (int s = 0; s < 8; ++s)
#pragma unroll
            for (int r = 0; r < R; ++r) wc[s][r] = wn[s][r];
    }
#pragma unroll
    for (int b = 0; b < NB; ++b)
        if (b * 16 + m < nc)
#pragma unroll
            for (int i = 0; i < 8; ++i) sm.out[(b * 16 + m) * kCols + warp * 16 + 2 * i + (lane >> 4)] = c[b][i];
    __syncthreads();
#elif defined(EXL3_WMMA_V70)
    namespace wm = nvcuda::wmma;
    static_assert(K2 <= KS, "the shared memory's stage holds K2 words a lane");
    constexpr int W = Words<K2>::W, R = Words<K2>::R, NJ = NR / 8, NB = NR / 16;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const size_t sstride = (size_t) ld * W;
    const uint32_t* tile = tr + ((size_t) (k0 / 16) * ld + tc0 + warp) * W;
    uint32_t* st = sm.stage + warp * W;
    uint32_t* wt = sm.wt + warp * 16 * kWtP;
    const int nq = (k1 - k0) / kCols;
    uint32_t wc[8][R], wn[8][R];
    float xv[NJ][4], xs[NJ][4];
    load_tiles<K2>(tile, sstride, lane, wc);
#pragma unroll
    for (int j = 0; j < NJ; ++j)
        if (warp + 8 * j < nc) chunk_load(in[j], suh, k0, lane, xv[j], xs[j]);
    wm::fragment<wm::accumulator, 16, 16, 16, float> c[NB];
#pragma unroll
    for (int b = 0; b < NB; ++b) wm::fill_fragment(c[b], 0.0f);
    for (int q = 0; q < nq; ++q) {
        __half* xq = sm.xb + (q & 1) * (NR * kXS);
#pragma unroll
        for (int j = 0; j < NJ; ++j)
            if (warp + 8 * j < nc) chunk_store(xv[j], xs[j], lane, xq + (warp + 8 * j) * kXS);
        __syncthreads();
        if (q + 1 < nq) {
            load_tiles<K2>(tile + (size_t) (q + 1) * 8 * sstride, sstride, lane, wn);
#pragma unroll
            for (int j = 0; j < NJ; ++j)
                if (warp + 8 * j < nc) chunk_load(in[j], suh, k0 + (q + 1) * kCols, lane, xv[j], xs[j]);
        }
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            uint32_t f[4];
            slice_frags<K2, cb>(wc[s], st, lane, f);
            wt[g * kWtP + t] = f[0];
            wt[g * kWtP + 4 + t] = f[1];
            wt[(g + 8) * kWtP + t] = f[2];
            wt[(g + 8) * kWtP + 4 + t] = f[3];
            __syncwarp();
            wm::fragment<wm::matrix_a, 16, 16, 16, __half, wm::row_major> a;
            wm::load_matrix_sync(a, reinterpret_cast<const __half*>(wt), 2 * kWtP);
#pragma unroll
            for (int b = 0; b < NB; ++b) {   // (activations past nc: whatever xb holds - their outputs are not used)
                wm::fragment<wm::matrix_b, 16, 16, 16, __half, wm::col_major> bx;
                wm::load_matrix_sync(bx, xq + b * 16 * kXS + s * 16, kXS);
                wm::mma_sync(c[b], a, bx, c[b]);
            }
            __syncwarp();   // (wt: the next slice's)
        }
#pragma unroll
        for (int s = 0; s < 8; ++s)
#pragma unroll
            for (int r = 0; r < R; ++r) wc[s][r] = wn[s][r];
    }
#pragma unroll
    for (int b = 0; b < NB; ++b)
        wm::store_matrix_sync(sm.out + b * 16 * kCols + warp * 16, c[b], kCols, wm::mem_col_major);
    __syncthreads();
#endif
}

// a block's rows [row, row + nc) of one expert (gemm(r, in): part r's NR-row product, its out): gate / up into h
// through the clamped swiglu, or down into y - hip_set_kernel's arithmetic around the product
template <int NR, bool gate_up, class Gemm>
__device__ __forceinline__ void set_rows_w(const SetArgs& a, Gemm&& gemm, const uint16_t* svh0, const uint16_t* svh1,
                                           const uint16_t* svh2, const float* x, int ldx, const int* idx, uint16_t* h,
                                           int ldh, float* y, int ldy, int row, int nc, int g, float* gate) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if constexpr (gate_up) {
        Xin in[NR / 8];
#pragma unroll
        for (int j = 0; j < NR / 8; ++j)
            if (warp + 8 * j < nc) in[j].f = x + (size_t) idx[row + warp + 8 * j] * ldx;
        const float* o = gemm(0, in);
        for (int rr = warp; rr < nc; rr += kWarps) block_out(o + rr * kCols, svh0 + g * kCols, gate + rr * kCols);
        __syncthreads();
        o = gemm(1, in);
        for (int rr = warp; rr < nc; rr += kWarps) {
            float v[4];
            for (int i = 0; i < 4; ++i) v[i] = o[rr * kCols + lane * 4 + i];
            had128(v, lane);
            for (int i = 0; i < 4; ++i) {
                const int cc = lane * 4 + i;
                const float up = v[i] * h2f(svh1[g * kCols + cc]);
                h[(size_t) (row + rr) * ldh + g * kCols + cc] =
                    __half_as_ushort(__float2half_rn(swiglu(gate[rr * kCols + cc], up, a.limit)));
            }
        }
    } else {
        XinHalf in[NR / 8];
#pragma unroll
        for (int j = 0; j < NR / 8; ++j)
            if (warp + 8 * j < nc) in[j].h = h + (size_t) (row + warp + 8 * j) * ldh;
        const float* o = gemm(2, in);
        for (int rr = warp; rr < nc; rr += kWarps)
            block_out(o + rr * kCols, svh2 + g * kCols, y + (size_t) (row + rr) * ldy + g * kCols);
    }
}

template <int K2, int cb, bool gate_up, int NR>
__global__ void __launch_bounds__(kThreads) wmma_set_kernel(SetArgs a, const float* x, int ldx, const int* idx,
                                                         uint16_t* h, int ldh, float* y, int ldy) {
#if defined(EXL3_WMMA_RDNA4)
    __shared__ __align__(16) Rdna4Smem<NR> sm;
#else
    __shared__ __align__(32) WmmaSmem<K2, NR> sm;   // (wmma: 256-bit aligned operands)
#endif
    __shared__ float gate[NR * kCols];
    const int e = blockIdx.z, g = blockIdx.y;
    const int start = a.bounds[e], end = a.bounds[e + 1];
    const int row = start - a.row0 + (int) blockIdx.x * NR;
    const int nc = min(NR, end - start - (int) blockIdx.x * NR);
    if (nc <= 0) return;
    const uint8_t* blob = a.wbase + (size_t) e * a.stride;
    auto gemm = [&](int r, const auto& in) -> const float* {
        const int ld = r < 2 ? a.n_ff / 16 : a.n_embd / 16, k1 = r < 2 ? a.n_embd : a.n_ff;
        block_gemm_w<K2, cb, K2, NR>((const uint32_t*) (blob + a.lay.trellis[r]), ld, g * 8, 0, k1, in,
                                     (const uint16_t*) (blob + a.lay.suh[r]), nc, sm);
        return sm.out;
    };
    set_rows_w<NR, gate_up>(a, gemm, (const uint16_t*) (blob + a.lay.svh[0]), (const uint16_t*) (blob + a.lay.svh[1]),
                            (const uint16_t*) (blob + a.lay.svh[2]), x, ldx, idx, h, ldh, y, ldy, row, nc, g, gate);
}

template <int cb, bool gate_up, int NR>
__global__ void __launch_bounds__(kThreads) wmma_set_mix_kernel(SetArgs a, KMix mx, const float* x, int ldx,
                                                             const int* idx, uint16_t* h, int ldh, float* y,
                                                             int ldy) {
#if defined(EXL3_WMMA_RDNA4)
    __shared__ __align__(16) Rdna4Smem<NR> sm;
#else
    __shared__ __align__(16) WmmaSmem<8, NR> sm;
#endif
    __shared__ float gate[NR * kCols];
    const int e = blockIdx.z, g = blockIdx.y;
    const int start = a.bounds[e], end = a.bounds[e + 1];
    const int row = start - a.row0 + (int) blockIdx.x * NR;
    const int nc = min(NR, end - start - (int) blockIdx.x * NR);
    if (nc <= 0) return;
    const unsigned long long ptr = a.eptr[e];
    const uint8_t* blob = (const uint8_t*) ptr;
    const int c = mix_cls(mx, ptr);
    auto gemm = [&](int r, const auto& in) -> const float* {
        const int ld = r < 2 ? a.n_ff / 16 : a.n_embd / 16, k1 = r < 2 ? a.n_embd : a.n_ff;
        const uint32_t* tr = (const uint32_t*) (blob + mx.tr[c][r]);
        const uint16_t* suh = (const uint16_t*) (blob + mx.suh[c][r]);
        switch (mx.k2[c][r]) {
        case 2: block_gemm_w<2, cb, 8, NR>(tr, ld, g * 8, 0, k1, in, suh, nc, sm); break;
        case 4: block_gemm_w<4, cb, 8, NR>(tr, ld, g * 8, 0, k1, in, suh, nc, sm); break;
        case 6: block_gemm_w<6, cb, 8, NR>(tr, ld, g * 8, 0, k1, in, suh, nc, sm); break;
        default: block_gemm_w<8, cb, 8, NR>(tr, ld, g * 8, 0, k1, in, suh, nc, sm); break;
        }
        return sm.out;
    };
    set_rows_w<NR, gate_up>(a, gemm, (const uint16_t*) (blob + mx.svh[c][0]), (const uint16_t*) (blob + mx.svh[c][1]),
                            (const uint16_t*) (blob + mx.svh[c][2]), x, ldx, idx, h, ldh, y, ldy, row, nc, g, gate);
}

template <int K2, int cb>
__global__ void __launch_bounds__(kThreads) hip_rows_kernel(Mat m, const uint16_t* x, int ldx, int rows,
                                                         float* y, int ldy, float beta) {
    __shared__ __align__(16) BlockSmem<K2> sm;
    const int row = blockIdx.x * kTok, g = blockIdx.y, warp = threadIdx.x >> 5;
    const int nc = min(kTok, rows - row);
    XinHalf in;
    if (warp < nc) in.h = x + (size_t) (row + warp) * ldx;
    block_gemv<K2, cb>(m.trellis, m.ld > 0 ? m.ld : m.n / 16, g * 8, 0, m.k, in, m.suh, nc, sm);
    if (warp < nc) {
        const int lane = threadIdx.x & 31;
        float v[4];
        for (int i = 0; i < 4; ++i) v[i] = sm.out[warp * kCols + lane * 4 + i];
        had128(v, lane);
        for (int i = 0; i < 4; ++i) {
            const int c = g * kCols + lane * 4 + i;
            float* dst = y + (size_t) (row + warp) * ldy + c;
            const float value = v[i] * h2f(m.svh[c]);
            *dst = beta == 0.0f ? value : value + beta * *dst;
        }
    }
}
// The imported unset-policy path uses the original HIP kernels. The optional
// application-policy scalar branch is not part of Fastllm's CUDA integration.
template <int K2, int cb> struct RunHipGateUp {
    static void run(const SetArgs& a, const float* x, int ldx, const int* idx, uint16_t* h, int ldh, cudaStream_t s) {
        hip_set_kernel<K2, cb, true><<<dim3((a.max_rows + kTok - 1) / kTok, a.n_ff / kCols, a.n_exp), kThreads, 0, s>>>
            (a, x, ldx, idx, h, ldh, nullptr, 0);
    }
};
template <int K2, int cb> struct RunHipDown {
    static void run(const SetArgs& a, const uint16_t* h, int ldh, float* y, int ldy, cudaStream_t s) {
        hip_set_kernel<K2, cb, false><<<dim3((a.max_rows + kTok - 1) / kTok, a.n_embd / kCols, a.n_exp), kThreads, 0, s>>>
            (a, nullptr, 0, nullptr, const_cast<uint16_t*>(h), ldh, y, ldy);
    }
};
template <int K2, int cb> struct RunHipRows {
    static void run(const Mat& m, const uint16_t* x, int ldx, int rows, float* y, int ldy, float beta, cudaStream_t s) {
        hip_rows_kernel<K2, cb><<<dim3((rows + kTok - 1) / kTok, m.n / kCols), kThreads, 0, s>>>
            (m, x, ldx, rows, y, ldy, beta);
    }
};

// ---------------------------------------------------------------- dispatch
// F(K2, cb) for K 1..8 (K2 = 2 K) and the three codebooks, and for the half-integer K 1.5 / 2.5 / 3.5 (K2 3, 5, 7)
// with mul1 - exllamav3 1.6 writes no other codebook for them; any other pair is refused (false: nothing launched)
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
    static void run(const KBatch& b, int blocks, cudaStream_t s) {

        // Fastllm: specialize single-row batches on CUDA as well as HIP.
        // Keep the input-split planner and reduction order unchanged. Native
        // BF16 jobs also resolve their I/O types at compile time. Mixed or
        // multi-row batches retain the generic kernel.
        bool single = true;
        for (int i = 0; i < b.n && single; ++i) single = b.j[i].nc == 1;
        if constexpr (cb == 2) {
            bool native = single;
            for (int i=0; i<b.n && native; ++i) native=b.j[i].x16 && b.j[i].y16 && b.j[i].x_bf16 && b.j[i].y_bf16;
            if (native) {mv_kernel<K2,cb,true,true><<<blocks,kThreads,0,s>>>(b);return;}
        }
        if (single) { mv_kernel<K2, cb, true><<<blocks, kThreads, 0, s>>>(b); return; }

        mv_kernel<K2, cb><<<blocks, kThreads, 0, s>>>(b);
    }
};
// AMD: the experts' launches with their input split (xsplit) - its blocks' z and the kernel that adds them up
#if defined(STRATA_USE_HIP)
#define EXL3_XS(n) (n)
#define EXL3_XS_RED(on, kernel, grid, a, s) \
    if (on) kernel<<<grid, kCols, 0, s>>>(a)
#else
#define EXL3_XS(n) 1u
#define EXL3_XS_RED(on, kernel, grid, a, s) (void) 0
#endif
template <int K2, int cb> struct RunWin {
    static void run(const KWin& a, int groups, cudaStream_t s) {
        win_gu_kernel<K2, cb><<<dim3((unsigned) (a.n_ff / kCols), (unsigned) groups, 2u * EXL3_XS(a.gS)), kThreads, 0, s>>>(a);
        EXL3_XS_RED(a.gS > 1, win_gu_red_kernel, dim3((unsigned) (a.n_ff / kCols), (unsigned) groups), a, s);
        EXL3_WIN_SWIGLU(a, groups, s);
    }
};
template <int K2, int cb> struct RunWinDown {
    static void run(const KWin& a, int groups, cudaStream_t s) {
        win_down_kernel<K2, cb><<<dim3((unsigned) (a.n_embd / kCols), (unsigned) groups, EXL3_XS((unsigned) a.dS)), kThreads, 0, s>>>(a);
        EXL3_XS_RED(a.dS > 1, win_down_red_kernel, dim3((unsigned) (a.n_embd / kCols), (unsigned) groups), a, s);
    }
};
template <int K2, int cb> struct RunGU {
    static void run(const KMoe& a, cudaStream_t s) {
        moe_gate_up_kernel<K2, cb><<<dim3((unsigned) (a.n_ff / kCols), (unsigned) a.k, 2u * EXL3_XS(a.gS)), kThreads, 0, s>>>(a);
        EXL3_XS_RED(a.gS > 1, moe_gu_red_kernel, dim3((unsigned) (a.n_ff / kCols), (unsigned) a.k), a, s);
        EXL3_MOE_SWIGLU(a, s);
    }
};
template <int K2, int cb> struct RunDown {
    static void run(const KMoe& a, cudaStream_t s) {
        if (a.route_output) {
            KMoe direct = a;
#if defined(STRATA_USE_HIP)
            direct.dS = 1;
            direct.dkr = a.n_ff;
#endif
            moe_down_kernel<K2, cb><<<dim3((unsigned)(a.n_embd/kCols), (unsigned)a.k), kThreads, 0, s>>>(direct);
            return;
        }
        moe_down_kernel<K2, cb><<<dim3((unsigned) (a.n_embd / kCols), (unsigned) a.k, EXL3_XS((unsigned) a.dS)), kThreads, 0, s>>>(a);
        EXL3_XS_RED(a.dS > 1, moe_down_red_kernel, dim3((unsigned) (a.n_embd / kCols), (unsigned) a.k), a, s);
        EXL3_MOE_COMBINE(a, s);
    }
};
template <int K2, int cb> struct RunRows {
    static void run(const Mat& m, uint16_t* dst, bool bf16, cudaStream_t s) {
        recon_rows_kernel<K2, cb><<<dim3((unsigned) (m.n / kCols), (unsigned) (m.k / kCols)), kThreads, 0, s>>>(
            m.trellis, m.ld > 0 ? m.ld : m.n / 16, m.suh, m.svh, m.k, dst, bf16);
    }
};
#if !defined(STRATA_USE_HIP)
template <int K2, int cb> struct RunGemm {
    static void run(const tc::GArgs& a, dim3 grid, cudaStream_t s) { tc::gemm_kernel<K2, cb><<<grid, kThreads, 0, s>>>(a); }
};
template <int K2, int cb> struct RunGateUp {
    static void run(const tc::GArgs& a, dim3 grid, cudaStream_t s) {
        tc::gate_up_kernel<K2, cb><<<grid, kThreads, 0, s>>>(a);
    }
};
#endif
template <int K2, int cb> struct RunInnerRows {
    static void run(const Mat& m, uint16_t* dst, cudaStream_t s) {
        inner_rows_kernel<K2, cb><<<dim3((unsigned) (m.n / kCols), (unsigned) (m.k / kCols)), kThreads, 0, s>>>(
            m.trellis, m.ld > 0 ? m.ld : m.n / 16, m.k, dst);
    }
};
template <int K2, int cb> struct RunInner {
    static void run(const Mat& m, uint16_t* dst, cudaStream_t s) {
        recon_inner_kernel<K2, cb><<<dim3((unsigned) (m.n / kCols), (unsigned) (m.k / 16)), kThreads, 0, s>>>(
            m.trellis, m.ld > 0 ? m.ld : m.n / 16, m.n, dst);
    }
};

// ---------------------------------------------------------------- per-stream scratch
constexpr size_t kPartFloats = (size_t) 2 << 20;   // split partials: 8 MB
constexpr size_t kCounters = 16384;
constexpr size_t kCntGU = 8192;                    // the counters from here on: the experts' gate / up pairs
constexpr size_t kHFloats = (size_t) 8 * 8192;     // the experts' h
constexpr size_t kUFloats = (size_t) 128 * 4096;   // the experts' up values: a window's 16 tokens x 8, n_ff <= 4096
#if defined(STRATA_USE_HIP)
constexpr size_t kXPFloats = (size_t) 6 << 20;     // the experts' input split partials (xsplit, AMD): 24 MB
// Preserve the imported unsplit default. No application environment controls.
int xsplit_want() { return 1; }

#endif
struct Scratch {
#if defined(STRATA_USE_HIP)
    float* xp = nullptr;   // (xsplit_want() > 1 only)
#endif
    float* part = nullptr;
    unsigned* cnt = nullptr;
    float* h = nullptr;
    float* u = nullptr;
};
std::mutex g_mu;
std::map<std::tuple<int, cudaStream_t, std::thread::id, int>, Scratch> g_scratch;

Scratch* scratch(cudaStream_t s, int lane = 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    // Per-thread default stream handles compare equal across host threads.
    const auto key = std::make_tuple(dev, s,
        (s == cudaStreamPerThread || s == nullptr) ? std::this_thread::get_id() : std::thread::id{}, lane);
    // (the last stream this thread used: an engine thread per device launches without the lock)
    thread_local int t_dev = -1, t_lane = -1;
    thread_local cudaStream_t t_s = nullptr;
    thread_local Scratch* t_sc = nullptr;
    if (t_sc != nullptr && t_dev == dev && t_s == s && t_lane == lane) return t_sc;
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_scratch.find(key);
    if (it != g_scratch.end()) {
        t_dev = dev, t_s = s, t_lane = lane, t_sc = &it->second;
        return &it->second;
    }
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(s, &cs) == cudaSuccess && cs != cudaStreamCaptureStatusNone) {
        std::fprintf(stderr, "exl3: scratch for a stream first seen inside a graph capture (call exl3::prepare)\n");
        return nullptr;
    }
    Scratch sc;
    if (cudaMalloc(&sc.part, kPartFloats * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&sc.cnt, kCounters * sizeof(unsigned)) != cudaSuccess ||
        cudaMalloc(&sc.h, kHFloats * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&sc.u, kUFloats * sizeof(float)) != cudaSuccess ||
#if defined(STRATA_USE_HIP)
        (xsplit_want() > 1 && cudaMalloc(&sc.xp, kXPFloats * sizeof(float)) != cudaSuccess) ||
#endif
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

// Whether the current device decodes with v1.1.8's kernels (exl3_legacy.cu): NVIDIA before sm_80, where this file's
// tiles run on the CUDA cores and were slower. The token path and the verify windows take the same family on a device, so a window stays bit for bit its token path.
// Asked once per device.
bool use_legacy() {
#if defined(STRATA_USE_HIP)
    return false;
#else
    static std::atomic<int> cache[64];   // 0 not asked, 1 v1.1.8's kernels, 2 this file's
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) return false;
    int c = cache[dev].load(std::memory_order_acquire);
    if (c == 0) {
        int major = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        c = major < 8 ? 1 : 2;
        if (c == 1) std::fprintf(stderr, "exl3: CUDA%d decodes with v1.1.8's kernels\n", dev);
        cache[dev].store(c, std::memory_order_release);
    }
    return c == 1;
#endif
}

// the blocks of mv_kernel<K2> an SM of the current device keeps at once (its registers and shared memory), asked once
// per device and bitrate
template <int K2, int cb> struct OccMv {
    static void run(int& n) {
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, mv_kernel<K2, cb>, kThreads, 0) != cudaSuccess) {
            cudaGetLastError();
            n = 2;
        }
    }
};
int mv_occ(int k2) {
    static std::atomic<int> cache[64][17];   // [device][k2]; 0: not asked
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64 || !k2_supported(k2, 2)) return 2;
    int n = cache[dev][k2].load(std::memory_order_acquire);
    if (n == 0) {
        n = 2;
        dispatch<OccMv>(k2, 2, n);
        n = std::max(1, n);
        cache[dev][k2].store(n, std::memory_order_release);
    }
    return n;
}

// mv's input split for m: S blocks of kr inputs per output block, at most kMaxKR inputs a block (the same for every
// activation count, so a job of several sums each activation as a job of one).  About two blocks per SM; where an SM
// keeps three or more (`occ`: the RTX 3090's at K > 4) four a SM, down to 128 inputs a block - its decode sets the pace
// there, and the more, shorter blocks balance it over the SMs (RTX 3090, K6 / K5, us: a KDA layer's 8192 x 4096
// output 47.6 -> 45.5, the dense layers' 4096 x 12288 gate 58.9 -> 52.5 and 12288 x 4096 down 59.3 -> 52.2, the
// shared expert's 4096 x 2048 17.4 -> 16.5); with two a SM, a launch that would spill a short second wave (more blocks
// than the SMs keep, less than one and a half times) takes twice the splits (RTX 5060 Ti, K5: that gate 99.4 -> 84.8
// us, that down 100.6 -> 84.9)
// call_groups: the output blocks the call's jobs of this code count together against the target, not the job's own -
// where an SM keeps three or more, all of them (RTX 3090, K6, a KDA layer's q, k and v, three 4096 x 8192 jobs: 1536
// blocks 119.6 us, 384 blocks 105.9); at two a SM only jobs of the same shape (RTX 5060 Ti: those q, k and v 191 ->
// 185 us, a shared expert's gate and up 35.9 -> 35.0; the MLA layer's q_a and kv_a together would lose 8%).  Never the
// activations: a window's token gets the token path's split
void mv_split(const Mat& m, int sms, int occ, int& S, int& kr, int call_groups = 0) {
    const bool many = occ >= 3;
    const int groups = std::max(m.n / kCols, call_groups);
    const int kr_min = many ? 128 : 256, target = (many ? 4 : 2) * sms, P = occ * sms;
    S = (m.k + kMaxKR - 1) / kMaxKR;
    while (groups * S < target && m.k / (S * 2) >= kr_min) S *= 2;
    if (!many && groups * S > P && 2 * groups * S < 3 * P && m.k / (S * 2) >= kr_min) S *= 2;
    kr = ((m.k + S - 1) / S + 127) / 128 * 128;
    S = (m.k + kr - 1) / kr;
}

}  // namespace

Mat view(const Mat& m, int c0, int nc) {
    Mat v = m;
    v.ld = m.ld > 0 ? m.ld : m.n / 16;
    v.trellis = m.trellis + (size_t) (c0 / 16) * 4 * m.k2;
    v.svh = m.svh + c0;
    v.n = nc;
    return v;
}

bool k2_supported(int k2, int cb) {
    return (k2 >= 2 && k2 <= 16 && (k2 & 1) == 0 && cb >= 0 && cb <= 2) || ((k2 == 3 || k2 == 5 || k2 == 7) && cb == 2);
}

bool supported(const Mat& m) {
    return m.trellis != nullptr && m.suh != nullptr && m.svh != nullptr && m.k > 0 && m.n > 0 && m.k % 128 == 0 &&
           m.n % 128 == 0 && k2_supported(m.k2, m.cb);
}

void set_skip(const int* flag) { g_skip = flag; }

int mv_splits(const Mat& m) {
    if (!supported(m)) return 0;
    int S = 1, kr = 0;
    mv_split(m, sm_count(), mv_occ(m.k2), S, kr);
    return S;
}

int mv_cols(const Mat& m) {
    if (!supported(m)) return 0;
#if !defined(STRATA_USE_HIP)
    if (use_legacy()) return 4;   // (v1.1.8's kernels: the runs of four their callers have always used)
#endif
    const int S = mv_splits(m);
    if (S <= 0) return 0;
    if (S == 1) return kTok;
    return (int) std::max<size_t>(1, std::min<size_t>(kTok, kPartFloats / ((size_t) S * m.n)));
}

int launch_errors() {
#if !defined(STRATA_USE_HIP)
    return g_launch_errors.load(std::memory_order_relaxed) + legacy::launch_errors();
#else
    return g_launch_errors.load(std::memory_order_relaxed);
#endif
}

bool prepare(cudaStream_t s) {
    EXL3_LEGACY(prepare(s));
    return scratch(s) != nullptr;
}

bool several_ok() {
    EXL3_LEGACY(several_ok());
    return true;
}

bool mv(const MvJob* jobs, int n, cudaStream_t s) {
    EXL3_LEGACY(mv(jobs, n, s));
    if (n <= 0 || n > kMaxMvJobs) return false;
    Scratch* sc = scratch(s);
    if (sc == nullptr) return false;
    const int sms = sm_count();
    // the launch units: a job's activations in runs of up to kTok (one pass over its weights each)
    struct Unit {
        int j, c0, nc;
    };
    Unit units[kMaxMvJobs * kMaxMvCols];
    int nu = 0;
    const int run = kTok;
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
    // each job's call_groups (mv_split): the output blocks of the call's jobs of its code - of its shape too where an SM
    // keeps fewer than three
    int cg[kMaxMvJobs] = {};
    for (int j = 0; j < n; ++j) {
        const Mat& a = *jobs[j].m;
        const bool many = mv_occ(a.k2) >= 3;
        for (int q = 0; q < n; ++q) {
            const Mat& b = *jobs[q].m;
            if (b.k2 == a.k2 && b.cb == a.cb && (many || (b.k == a.k && b.n == a.n))) cg[j] += b.n / kCols;
        }
    }
    bool done[sizeof(units) / sizeof(units[0])] = {};
    for (int i = 0; i < nu; ++i) {
        if (done[i]) continue;
        const Mat& m0 = *jobs[units[i].j].m;
        // a launch: the units of one (K, codebook), as many as KBatch and the split scratch take (the next launch on
        // the stream may use the scratch again: this one's last blocks are done with it by then)
        KBatch b{};
        int blocks = 0, nb = 0;
        size_t part_at = 0, cnt_at = 0;
        for (int u = i; u < nu && nb < kMaxMvJobs; ++u) {
            const MvJob& job = jobs[units[u].j];
            const Mat& m = *job.m;
            if (done[u] || m.k2 != m0.k2 || m.cb != m0.cb) continue;
            const int c0 = units[u].c0, nc = units[u].nc, groups = m.n / kCols;
            int S = 1, kr = 0;
            mv_split(m, sms, mv_occ(m.k2), S, kr, cg[units[u].j]);
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
        b.skip = g_skip;
        if (!dispatch<RunMv>(m0.k2, m0.cb, b, blocks, s)) return false;
#if defined(STRATA_USE_HIP)
        int rblocks = 0, rnc = 0;
        for (int j = 0; j < nb; ++j)
            if (b.j[j].S > 1) rblocks += b.j[j].n / kCols, rnc = std::max(rnc, b.j[j].nc);
        if (rblocks > 0 && sep_sums()) mv_reduce_kernel<<<dim3(rblocks, rnc), kCols, 0, s>>>(b);
#endif
    }
    launch_check("mv");
    return true;
}

#if defined(STRATA_USE_HIP)
// the routed experts' input split for an input of k (default: the whole input a block): S blocks of
// kr inputs, their partial sums added up in split order by the *_red kernels
static void xsplit(int k, int& S, int& kr) {
    const int want = xsplit_want();
    kr = ((k + want - 1) / want + kCols - 1) / kCols * kCols;
    S = (k + kr - 1) / kr;
}
static bool xsplit_set(int rows, int n_embd, int n_ff, float* xp, float*& dst, int& gS, int& gkr, int& dS, int& dkr) {
    xsplit(n_embd, gS, gkr);
    xsplit(n_ff, dS, dkr);
    dst = xp;
    if ((size_t) rows * 2 * gS * n_ff > kXPFloats || (size_t) rows * dS * n_embd > kXPFloats) {
        std::fprintf(stderr, "exl3: the experts' input split takes more than its scratch (%d rows)\n", rows);
        return false;
    }
    return true;
}
#endif

static bool moe_args(const MoeArgs& a, KMoe& m, cudaStream_t s) {
    if (a.scratch_lane < 0 || a.scratch_lane > 2) return false;
    Scratch* sc = scratch(s, a.scratch_lane);
    if (sc == nullptr) return false;
    if (a.k < 1 || a.k > 8 || a.n_embd % kCols != 0 || a.n_ff % kCols != 0 || a.n_embd > kMaxMoeK ||
        a.n_ff > kMaxMoeK || (size_t) a.k * a.n_ff > kHFloats || (size_t) a.k * a.n_embd > kPartFloats ||
        (size_t) (a.n_embd / kCols) > kCntGU || kCntGU + (size_t) a.k * (a.n_ff / kCols) > kCounters) {
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
    m.u = sc->u;
    m.part = sc->part;
    m.cnt = sc->cnt;
    m.cnt_gu = sc->cnt + kCntGU;
    for (int r = 0; r < 3; ++r) {
        m.suh[r] = a.lay.suh[r];
        m.svh[r] = a.lay.svh[r];
        m.tr[r] = a.lay.trellis[r];
    }
    m.k = a.k;
    m.n_embd = a.n_embd;
    m.n_ff = a.n_ff;
    m.limit = a.limit;
    m.skip = g_skip;
    m.indirect = a.indirect;
    m.round_bf16 = a.round_bf16;
    m.route_output = a.route_output;
#if defined(STRATA_USE_HIP)
    return xsplit_set(a.k, a.n_embd, a.n_ff, sc->xp, m.xp, m.gS, m.gkr, m.dS, m.dkr);
#else
    return true;
#endif
}

// a mixed layer's launch: the classes' codebook (one for all of them; mixed_ok checked it), as a template argument
template <class F>
static bool mix_launch(const MixedLayouts& mx, F&& f) {
    switch (mx.lay[0].cb) {
    case 0: f(std::integral_constant<int, 0>{}); return true;
    case 1: f(std::integral_constant<int, 1>{}); return true;
    case 2: f(std::integral_constant<int, 2>{}); return true;
    default: return false;
    }
}
bool mixed_supported(const ExpertLayout& lay) {
    if (lay.fmt != 0 || lay.cb < 0 || lay.cb > 2) return false;
    for (int r = 0; r < 3; ++r)
        if (lay.k2[r] != 2 && lay.k2[r] != 4 && lay.k2[r] != 6 && lay.k2[r] != 8) return false;
    return true;
}
static bool mixed_ok(const MixedLayouts& mx, const char* what) {
    bool ok = mx.n >= 1 && mx.n <= kMaxClasses;
    for (int c = 0; ok && c < mx.n; ++c) ok = mixed_supported(mx.lay[c]) && mx.lay[c].cb == mx.lay[0].cb;
    if (!ok) {
        std::fprintf(stderr, "exl3 %s: a mixed layer's classes not covered (K 1-4, one codebook)\n", what);
        g_launch_errors.fetch_add(1, std::memory_order_relaxed);
    }
    return ok;
}

void moe_gate_up(const MoeArgs& a, const float* x, float* h, cudaStream_t s) {
    if (a.mix.n > 0) {   // (a mixed layer: the same arithmetic on every GPU, no legacy kernels)
        KMoe m;
        if (!mixed_ok(a.mix, "moe_gate_up") || !moe_args(a, m, s)) return;
        m.x = x;
        if (h != nullptr) m.h = h;
        const KMix k = kmix(a.mix);
        mix_launch(a.mix, [&](auto cb) {
            moe_gate_up_mix_kernel<decltype(cb)::value><<<dim3((unsigned) (a.n_ff / kCols), (unsigned) a.k, 2u), kThreads, 0, s>>>(m, k);
        });
        EXL3_MOE_SWIGLU(m, s);
        launch_check("moe_gate_up (mixed)");
        return;
    }
    if (!a.indirect && !a.round_bf16) { EXL3_LEGACY_MOE(a, moe_gate_up(a, x, h, s)); }
    KMoe m;
    if (!moe_args(a, m, s)) return;
    if (a.lay.k2[0] != a.lay.k2[1]) {
        std::fprintf(stderr, "exl3 moe_gate_up: gate K %g != up K %g\n", a.lay.k2[0] / 2.0, a.lay.k2[1] / 2.0);
        return;
    }
    m.x = x;
    if (h != nullptr) m.h = h;
    if (!dispatch<RunGU>(a.lay.k2[0], a.lay.cb, m, s))
        std::fprintf(stderr, "exl3 moe_gate_up: K %g cb %d\n", a.lay.k2[0] / 2.0, a.lay.cb);
    launch_check("moe_gate_up");
}

void moe_down(const MoeArgs& a, const float* h, const float* sh_out, float* out, cudaStream_t s) {
    if (a.mix.n > 0) {
        KMoe m;
        if (!mixed_ok(a.mix, "moe_down") || !moe_args(a, m, s)) return;
        if (h != nullptr) m.h = const_cast<float*>(h);
        m.sh_out = sh_out;
        m.out = out;
        const KMix k = kmix(a.mix);
        mix_launch(a.mix, [&](auto cb) {
            moe_down_mix_kernel<decltype(cb)::value><<<dim3((unsigned) (a.n_embd / kCols), (unsigned) a.k), kThreads, 0, s>>>(m, k);
        });
        if (!m.route_output) { EXL3_MOE_COMBINE(m, s); }
        launch_check("moe_down (mixed)");
        return;
    }
    if (!a.indirect && !a.round_bf16 && !a.route_output) { EXL3_LEGACY_MOE(a, moe_down(a, h, sh_out, out, s)); }
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
    if (a.mix.n == 0 && !a.indirect && !a.round_bf16 && !a.route_output && !a.x16)
        EXL3_LEGACY_MOE(a, win_experts(a, s));
    if (a.mix.n > 0 && !mixed_ok(a.mix, "win_experts")) return false;
    if (a.T < 1 || a.k < 1 || a.k > 8 || a.n_embd % kCols != 0 || a.n_ff % kCols != 0 || a.n_embd > kMaxMoeK ||
        a.n_ff > kMaxMoeK || (a.mix.n == 0 && a.lay.k2[0] != a.lay.k2[1]) || a.h == nullptr || a.part == nullptr ||
        a.out == nullptr) {
        std::fprintf(stderr, "exl3 win_experts: not covered (T %d k %d n_embd %d n_ff %d, gate K %g up K %g)\n", a.T,
                     a.k, a.n_embd, a.n_ff, a.lay.k2[0] / 2.0, a.lay.k2[1] / 2.0);
        return false;
    }
    Scratch* sc = scratch(s, a.scratch_lane);
    if (sc == nullptr) return false;
    const int groups = a.group_capacity > 0 ? a.group_capacity : a.T * a.k;
    if ((!a.up && (size_t) a.T * 8 * a.n_ff > kUFloats) ||
        (!a.group_counters && kCntGU + (size_t) groups * (a.n_ff / kCols) > kCounters)) {
        std::fprintf(stderr, "exl3 win_experts: %d tokens are more than the scratch takes\n", a.T);
        return false;
    }
    KWin w{};
    w.grp_ptr = a.grp_ptr;
    w.grp_start = a.grp_start;
    w.grp_ent = a.grp_ent;
    w.n_groups = a.n_groups;
    w.plan_ptr = a.plan_ptr;
    w.plan_w = a.plan_w;
    w.cpu_part = a.cpu_part;
    w.cpu_stride = a.cpu_stride != 0 ? a.cpu_stride : (size_t) a.n_embd;
    w.cpu_flag = a.cpu_flag;
    w.x = a.x;
    w.x16 = a.x16; w.x_bf16 = a.x_bf16;
    w.indirect = a.indirect; w.round_bf16 = a.round_bf16;
    w.route_output = a.route_output; w.preserve_missing = a.preserve_missing;
    w.x_stride = a.x_stride;
    w.h = a.h;
    w.u = a.up ? a.up : sc->u;
    w.cnt_gu = a.group_counters ? a.group_counters : sc->cnt + kCntGU;
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
#if defined(STRATA_USE_HIP)
    if (!xsplit_set(a.T * 8, a.n_embd, a.n_ff, sc->xp, w.xp, w.gS, w.gkr, w.dS, w.dkr)) return false;
#endif
    for (int first = 0; first < groups; first += 65535) {
        w.first_group = first;
        const int launched = std::min(65535, groups - first);
        if (a.mix.n > 0) {
            const KMix k = kmix(a.mix);
            mix_launch(a.mix, [&](auto cb) {
                win_gu_mix_kernel<decltype(cb)::value><<<dim3((unsigned) (a.n_ff / kCols), (unsigned) launched, 2u), kThreads, 0, s>>>(w, k);
                EXL3_WIN_SWIGLU(w, launched, s);
                win_down_mix_kernel<decltype(cb)::value><<<dim3((unsigned) (a.n_embd / kCols), (unsigned) launched), kThreads, 0, s>>>(w, k);
            });
        } else if (!dispatch<RunWin>(a.lay.k2[0], a.lay.cb, w, launched, s) ||
                   !dispatch<RunWinDown>(a.lay.k2[2], a.lay.cb, w, launched, s)) {
            return false;
        }
    }
    if (a.wait_ptr != nullptr) win_wait_kernel<<<1, 32, 0, s>>>(a.wait_ptr, a.wait_val);
    for (int first = 0; first < a.T; first += 65535) {
        w.first_token = first;
        win_combine_kernel<<<dim3((unsigned) (a.n_embd / kCols), (unsigned) std::min(65535, a.T-first)), kCols, 0, s>>>(w);
    }
    launch_check("win_experts");
    return true;
}

// mma_supported() for the current device, cached (the prompt path's launches ask it each time)
bool mma_now() {
#if defined(STRATA_USE_HIP)
    return false;
#else
    static int known[64];   // 0: not asked yet, 1: no, 2: yes
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) return mma_supported();
    if (known[dev] == 0) known[dev] = mma_supported() ? 2 : 1;
    return known[dev] == 2;
#endif
}

// Preserve imported default capabilities: gfx11 on HIP, sm_7x on CUDA.
// Fastllm does not import Maya/Strata application environment policy.
static bool wmma_now() {
#if defined(STRATA_USE_HIP)
    static int known[64];   // 0: not asked yet, 1: no, 2: yes
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) return false;
    if (known[dev] == 0) {
        cudaDeviceProp p{};
        known[dev] = cudaGetDeviceProperties(&p, dev) == cudaSuccess &&
                     std::strncmp(p.gcnArchName, "gfx11", 5) == 0 ? 2 : 1;
    }
    return known[dev] == 2;
#else
    // sm_7x: the Volta / Turing wmma kernels (sm_80 and later take the mma.m16n8k16 ones)
    static int known[64];
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) return false;
    if (known[dev] == 0) {
        int major = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        known[dev] = major == 7 ? 2 : 1;
    }
    return known[dev] == 2;
#endif
}
// Physical WMMA block shape, independent of the logical request batch.
template <class F>
static void with_rows(F&& f) {
    f(std::integral_constant<int, 16>{});
}
template <int K2, int cb> struct RunWmmaGateUp {
    static void run(const SetArgs& a, const float* x, int ldx, const int* idx, uint16_t* h, int ldh, cudaStream_t s) {
        with_rows([&](auto nr) {
            constexpr int NR = decltype(nr)::value;
            wmma_set_kernel<K2, cb, true, NR><<<dim3((a.max_rows + NR - 1) / NR, a.n_ff / kCols, a.n_exp), kThreads, 0,
                                                s>>>(a, x, ldx, idx, h, ldh, nullptr, 0);
        });
    }
};
template <int K2, int cb> struct RunWmmaDown {
    static void run(const SetArgs& a, const uint16_t* h, int ldh, float* y, int ldy, cudaStream_t s) {
        with_rows([&](auto nr) {
            constexpr int NR = decltype(nr)::value;
            wmma_set_kernel<K2, cb, false, NR><<<dim3((a.max_rows + NR - 1) / NR, a.n_embd / kCols, a.n_exp), kThreads,
                                                 0, s>>>(a, nullptr, 0, nullptr, const_cast<uint16_t*>(h), ldh, y, ldy);
        });
    }
};

bool prompt_tc() {
#if defined(STRATA_USE_HIP)
    return false;
#else
    return mma_supported() || wmma_now();
#endif
}

bool prompt_wmma() { return wmma_now(); }

bool mma_supported() {
#if defined(STRATA_USE_HIP)
    return false;
#else
    int dev = 0, major = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    return major >= 8;
#endif
}

void set_gate_up(const SetArgs& sa, const float* x, int ldx, const int* row_idx, uint16_t* h16, int ldh,
                 cudaStream_t s) {
    if (sa.mix.n > 0) {   // (a mixed layer: every expert's own address and class, the portable kernels everywhere)
        if (sa.n_exp <= 0 || sa.max_rows <= 0) return;
        if (sa.eptr == nullptr || !mixed_ok(sa.mix, "set_gate_up")) return;
        const KMix k = kmix(sa.mix);
#if defined(STRATA_USE_HIP)
        if (wmma_now()) {
            mix_launch(sa.mix, [&](auto cb) {
                with_rows([&](auto nr) {
                    constexpr int NR = decltype(nr)::value;
                    wmma_set_mix_kernel<decltype(cb)::value, true, NR>
                        <<<dim3((sa.max_rows + NR - 1) / NR, sa.n_ff / kCols, sa.n_exp), kThreads, 0, s>>>(
                            sa, k, x, ldx, row_idx, h16, ldh, nullptr, 0);
                });
            });
            launch_check("set_gate_up (mixed, matrix cores)");
            return;
        }
#endif
        mix_launch(sa.mix, [&](auto cb) {
            set_mix_kernel<decltype(cb)::value, true>
                <<<dim3((sa.max_rows + kTok - 1) / kTok, sa.n_ff / kCols, sa.n_exp), kThreads, 0, s>>>(
                    ksetmix(sa), k, x, ldx, row_idx, h16, ldh, nullptr, 0);
        });
        launch_check("set_gate_up (mixed)");
        return;
    }
#if !defined(STRATA_USE_HIP)
    if (!mma_now()) {   // (before sm_80: the portable kernels - on sm_7x the wmma ones)
        if (sa.n_exp <= 0 || sa.max_rows <= 0) return;
        if (wmma_now() && sa.lay.k2[0] == sa.lay.k2[1] &&
            dispatch<RunWmmaGateUp>(sa.lay.k2[0], sa.lay.cb, sa, x, ldx, row_idx, h16, ldh, s)) {
            launch_check("set_gate_up (Volta tensor cores)");
            return;
        }
        if (sa.lay.k2[0] != sa.lay.k2[1] || !dispatch<RunHipGateUp>(sa.lay.k2[0], sa.lay.cb, sa, x, ldx, row_idx, h16, ldh, s)) {
            std::fprintf(stderr, "exl3 set_gate_up: unsupported layout\n");
            g_launch_errors.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        launch_check("set_gate_up (portable)");
        return;
    }
    if (sa.lay.k2[0] != sa.lay.k2[1]) {
        std::fprintf(stderr, "exl3 set_gate_up: gate K %g != up K %g\n", sa.lay.k2[0] / 2.0, sa.lay.k2[1] / 2.0);
        g_launch_errors.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (sa.n_exp <= 0 || sa.max_rows <= 0) return;
    tc::GArgs a{};
    a.wbase = sa.wbase;
    a.stride = sa.stride;
    for (int r = 0; r < 2; ++r) {
        a.tr[r] = sa.lay.trellis[r];
        a.suh[r] = sa.lay.suh[r];
        a.svh[r] = sa.lay.svh[r];
        a.ld[r] = sa.n_ff / 16;
    }
    a.bounds = sa.bounds;
    a.row0 = sa.row0;
    a.k = sa.n_embd;
    a.n = sa.n_ff;
    a.xf = x;
    a.idx = row_idx;
    a.ldx = ldx;
    a.h = (__half*) h16;
    a.ldh = ldh;
    a.limit = sa.limit;
    const dim3 grid((unsigned) ((sa.max_rows + tc::BM - 1) / tc::BM), (unsigned) (sa.n_ff / kCols), (unsigned) sa.n_exp);
    if (!dispatch<RunGateUp>(sa.lay.k2[0], sa.lay.cb, a, grid, s))
        std::fprintf(stderr, "exl3 set_gate_up: K %g cb %d\n", sa.lay.k2[0] / 2.0, sa.lay.cb);
    launch_check("set_gate_up");
#else
    if (sa.n_exp <= 0 || sa.max_rows <= 0) return;
    if (wmma_now() && sa.lay.k2[0] == sa.lay.k2[1] &&
        dispatch<RunWmmaGateUp>(sa.lay.k2[0], sa.lay.cb, sa, x, ldx, row_idx, h16, ldh, s)) {
        launch_check("HIP set_gate_up (matrix cores)");
        return;
    }
    if (sa.lay.k2[0] != sa.lay.k2[1] || !dispatch<RunHipGateUp>(sa.lay.k2[0], sa.lay.cb, sa, x, ldx, row_idx, h16, ldh, s)) {
        std::fprintf(stderr, "exl3 HIP set_gate_up: unsupported layout\n");
        g_launch_errors.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    launch_check("HIP set_gate_up");
#endif
}

void set_down(const SetArgs& sa, const uint16_t* h16, int ldh, float* y, int ldy, cudaStream_t s) {
    if (sa.mix.n > 0) {
        if (sa.n_exp <= 0 || sa.max_rows <= 0) return;
        if (sa.eptr == nullptr || !mixed_ok(sa.mix, "set_down")) return;
        const KMix k = kmix(sa.mix);
#if defined(STRATA_USE_HIP)
        if (wmma_now()) {
            mix_launch(sa.mix, [&](auto cb) {
                with_rows([&](auto nr) {
                    constexpr int NR = decltype(nr)::value;
                    wmma_set_mix_kernel<decltype(cb)::value, false, NR>
                        <<<dim3((sa.max_rows + NR - 1) / NR, sa.n_embd / kCols, sa.n_exp), kThreads, 0, s>>>(
                            sa, k, nullptr, 0, nullptr, const_cast<uint16_t*>(h16), ldh, y, ldy);
                });
            });
            launch_check("set_down (mixed, matrix cores)");
            return;
        }
#endif
        mix_launch(sa.mix, [&](auto cb) {
            set_mix_kernel<decltype(cb)::value, false>
                <<<dim3((sa.max_rows + kTok - 1) / kTok, sa.n_embd / kCols, sa.n_exp), kThreads, 0, s>>>(
                    ksetmix(sa), k, nullptr, 0, nullptr, const_cast<uint16_t*>(h16), ldh, y, ldy);
        });
        launch_check("set_down (mixed)");
        return;
    }
#if !defined(STRATA_USE_HIP)
    if (sa.n_exp <= 0 || sa.max_rows <= 0) return;
    if (!mma_now()) {   // (before sm_80: the portable kernels - on sm_7x the wmma ones)
        if (wmma_now() && dispatch<RunWmmaDown>(sa.lay.k2[2], sa.lay.cb, sa, h16, ldh, y, ldy, s)) {
            launch_check("set_down (Volta tensor cores)");
            return;
        }
        if (!dispatch<RunHipDown>(sa.lay.k2[2], sa.lay.cb, sa, h16, ldh, y, ldy, s))
            g_launch_errors.fetch_add(1, std::memory_order_relaxed);
        else
            launch_check("set_down (portable)");
        return;
    }
    tc::GArgs a{};
    a.wbase = sa.wbase;
    a.stride = sa.stride;
    a.tr[0] = sa.lay.trellis[2];
    a.suh[0] = sa.lay.suh[2];
    a.svh[0] = sa.lay.svh[2];
    a.ld[0] = sa.n_embd / 16;
    a.bounds = sa.bounds;
    a.row0 = sa.row0;
    a.k = sa.n_ff;
    a.n = sa.n_embd;
    a.xh = (const __half*) h16;
    a.ldx = ldh;
    a.y = y;
    a.ldy = ldy;
    const dim3 grid((unsigned) ((sa.max_rows + tc::BM - 1) / tc::BM), (unsigned) (sa.n_embd / kCols), (unsigned) sa.n_exp);
    if (!dispatch<RunGemm>(sa.lay.k2[2], sa.lay.cb, a, grid, s))
        std::fprintf(stderr, "exl3 set_down: K %g cb %d\n", sa.lay.k2[2] / 2.0, sa.lay.cb);
    launch_check("set_down");
#else
    if (sa.n_exp <= 0 || sa.max_rows <= 0) return;
    if (wmma_now() && dispatch<RunWmmaDown>(sa.lay.k2[2], sa.lay.cb, sa, h16, ldh, y, ldy, s)) {
        launch_check("HIP set_down (matrix cores)");
        return;
    }
    if (!dispatch<RunHipDown>(sa.lay.k2[2], sa.lay.cb, sa, h16, ldh, y, ldy, s)) {
        g_launch_errors.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    launch_check("HIP set_down");
#endif
}

void gemm_rows(const Mat& m, const uint16_t* x16, int ldx, int rows, float* y, int ldy, float beta, cudaStream_t s) {
#if !defined(STRATA_USE_HIP)
    if (rows <= 0) return;
    if (!mma_now()) {   // (before sm_80: the portable kernels)
        if (!supported(m) || !dispatch<RunHipRows>(m.k2, m.cb, m, x16, ldx, rows, y, ldy, beta, s))
            g_launch_errors.fetch_add(1, std::memory_order_relaxed);
        else
            launch_check("gemm_rows (portable)");
        return;
    }
    tc::GArgs a{};
    a.wbase = nullptr;
    a.tr[0] = (size_t) m.trellis;
    a.suh[0] = (size_t) m.suh;
    a.svh[0] = (size_t) m.svh;
    a.ld[0] = m.ld > 0 ? m.ld : m.n / 16;
    a.m = rows;
    a.k = m.k;
    a.n = m.n;
    a.xh = (const __half*) x16;
    a.ldx = ldx;
    a.y = y;
    a.ldy = ldy;
    a.beta = beta;
    const dim3 grid((unsigned) ((rows + tc::BM - 1) / tc::BM), (unsigned) (m.n / kCols), 1);
    if (!supported(m) || !dispatch<RunGemm>(m.k2, m.cb, a, grid, s)) {
        std::fprintf(stderr, "exl3 gemm_rows: unsupported matrix\n");
        g_launch_errors.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    launch_check("gemm_rows");
#else
    if (rows <= 0) return;
    if (!supported(m) || !dispatch<RunHipRows>(m.k2, m.cb, m, x16, ldx, rows, y, ldy, beta, s)) {
        g_launch_errors.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    launch_check("HIP gemm_rows");
#endif
}

void reconstruct_rows(const Mat& m, uint16_t* dst, bool bf16, cudaStream_t s) {
    if (!supported(m) || !dispatch<RunRows>(m.k2, m.cb, m, dst, bf16, s)) {
        std::fprintf(stderr, "exl3 reconstruct_rows: unsupported matrix\n");
        g_launch_errors.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    launch_check("reconstruct_rows");
}

void reconstruct_inner_rows(const Mat& m, uint16_t* dst, cudaStream_t s) {
    if (!supported(m) || !dispatch<RunInnerRows>(m.k2, m.cb, m, dst, s)) {
        std::fprintf(stderr, "exl3 reconstruct_inner_rows: unsupported matrix\n");
        g_launch_errors.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    launch_check("reconstruct_inner_rows");
}

void rows_rot_f16(const float* x, int ldx, const int* idx, int nrows, int ncols, const uint16_t* suh, uint16_t* out,
                  cudaStream_t s) {
    if (nrows <= 0) return;
    const long long warps = (long long) nrows * (ncols / kCols);
    rows_rot_f16_kernel<<<(unsigned) ((warps + kWarps - 1) / kWarps), kThreads, 0, s>>>(x, ldx, idx, nrows, ncols, suh,
                                                                                         out);
    launch_check("rows_rot_f16");
}

void rows_rot_out(float* y, int ldy, int nrows, int ncols, const uint16_t* svh, cudaStream_t s) {
    if (nrows <= 0) return;
    const long long warps = (long long) nrows * (ncols / kCols);
    rows_rot_out_kernel<<<(unsigned) ((warps + kWarps - 1) / kWarps), kThreads, 0, s>>>(y, ldy, nrows, ncols, svh);
    launch_check("rows_rot_out");
}

void rows_f16(const float* x, int ldx, const int* idx, int nrows, int ncols, uint16_t* out, cudaStream_t s) {
    if (nrows <= 0) return;
    rows_f16_kernel<<<dim3((unsigned) std::min(8, (ncols + 255) / 256), (unsigned) nrows), 256, 0, s>>>(x, ldx, idx,
                                                                                                         ncols, out);
    launch_check("rows_f16");
}

void reconstruct_inner(const Mat& m, uint16_t* dst, cudaStream_t s) {
    if (!supported(m) || !dispatch<RunInner>(m.k2, m.cb, m, dst, s)) {
        std::fprintf(stderr, "exl3 reconstruct_inner: unsupported matrix\n");
        g_launch_errors.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    launch_check("reconstruct_inner");
}

}  // namespace strata::kernels::exl3
