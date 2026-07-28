//
// SM70 (Tesla V100) IQ4_XS matrix-multiply-quantized (MMQ) kernel.
//
// This is a focused, self-contained DP4A-tiled matmul for GGML_TYPE_IQ4_XS
// weights. It is NOT the generic llama.cpp MMQ subsystem: only the IQ4_XS
// DP4A closure was extracted and simplified for a single compute capability
// (SM70), where the Marlin SM75 m16n8/ldmatrix path is unavailable but the
// INT8 dot-product (DP4A, `__dp4a`) is available (sm_61+).
//
// Provenance — derived from llama.cpp "turboquant" (MIT):
//   llama.cpp-turboquant/ggml/src/ggml-cuda/mmq.cuh
//       -> load_tiles_iq4_xs<mmq_y, need_check>  (DP4A branch, ~lines 3121-3184)
//       -> vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>  (~lines 1126-1156)
//       -> mul_mat_q_process_tile                 (DP4A branch, ~lines 3446-3525)
//       -> block_q8_1_mmq, MMQ_TILE_NE_K, MMQ_TILE_Y_K, MMQ_ITER_K, MMQ_NWARPS
//   llama.cpp-turboquant/ggml/src/ggml-cuda/vecdotq.cuh
//       -> vec_dot_q8_0_q8_1_impl                 (~lines 243-255)
//       -> get_int_from_table_16                  (CUDA branch, ~lines 57-80)
//       -> kvalues_iq4nl codebook                 (ggml-common.h ~lines 1200-1202)
//   llama.cpp-turboquant/ggml/src/ggml-cuda/quantize.cu
//       -> quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4> (D4 branch, ~lines 280-373)
//
// The six-bit group scale and nibble→codebook mapping are cross-checked
// against FastLLM's own validated dequantize_block_iq4_xs in
// fastllm-ggml-cuda.cu (~lines 1978-2000), the ground truth for this repo's
// IQ4_XS layout.
//
// Safety contract:
//   * Eligibility: SM70 only, IQ4_XS caller, n in [8,64], m%256==0,
//     k>=128, non-null pointers, supported dtype, shared memory fits.
//     Smaller output projections are materially faster on the legacy path.
//   * Trial path: on any failure returns false WITHOUT writing output,
//     leaving the caller's dequant+cuBLAS / MMVQ fallback intact.
//   * Failed CUDA launch clears the sticky error before returning false.
//   * No persistent expanded-weight copy; nibbles are expanded on-the-fly
//     into shared memory each tile via the codebook LUT.
//   * Only DP4A + __byte_perm. No SM75 instructions (ldmatrix, mma.m16n8).
//

#include "fastllm-cuda.cuh"
#include "fastllm.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "gguf.h"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>

// ---------------------------------------------------------------------------
// Constants — mirror llama.cpp MMQ DP4A tile geometry for IQ4_XS.
//
// IQ4_XS reuses the Q8_0 DP4A tile sizes (mmq_get_dp4a_tile_x_sizes maps
// GGML_TYPE_IQ4_XS → MMQ_DP4A_TXS_Q8_0).  The block_q8_1_mmq activation uses
// the D4 scale layout (one float scale per 32 values, 4 per 128-value block).
//
//   QK8_1 = 32   block_q8_1 values per scale group
//   QR8_1 = 1    quantization ratio
//   QI8_1 = 8    QK8_1 / (4*QR8_1) = 32/(4*1) = 8.  This is the number of
//               32-bit ints containing QK8_1=32 quantized values, divided by
//               4 (QR8_1): one int32 holds 4 int8 values, so QK8_1/4 = 8
//               int32 per 32-value scale group.
//   QI8_0 = 8    same formula for q8_0 weights.
//   QK_K  = 256  values per iq4_xs block (8 sub-blocks of 32).
//   QR4_XS = 2   iq4_xs quantization ratio.
// ---------------------------------------------------------------------------
#define S70_TILE_NE_K  32
#define S70_ITER_K     256
#define S70_NWARPS     8
#define S70_TILE_Y_K   36    // NE_K + NE_K/QI8_1 = 32 + 32/8 = 36
#define S70_Y          128   // weight rows per CTA
#define S70_VDR        8     // VDR_Q8_0_Q8_1_MMQ after IQ4_XS LUT expansion
#define S70_QK_K       256
#define S70_QK8_1      32
#define S70_QI8_0      8     // QK8_0/(4*QR8_0) = 32/4
#define S70_QI8_1      8     // QK8_1/(4*QR8_1) = 32/4
// iq4_xs qs nibble array: QK_K/2 bytes = QK_K/8 int32 per block.
#define S70_QI4_XS     (S70_QK_K / 8)  // = 32

// Weight tile (DP4A Q8_0-equivalent) row strides:
//   qs row stride = 2*NE_K + 1 = 65 int32
//   df row stride = 2*NE_K/QI8_0 = 2*32/8 = 8 float, with +i/4 pad column
#define S70_X_QS_STRIDE  (2 * S70_TILE_NE_K + 1)  // 65
#define S70_X_DF_STRIDE  (2 * S70_TILE_NE_K / S70_QI8_0)  // 8

// ---------------------------------------------------------------------------
// Exact IQ4_NL codebook (shared by IQ4_XS and IQ4_NL).
// ggml-common.h: kvalues_iq4nl = {-127,-104,-83,-65,-49,-35,-22,-10,
//                                  1, 13, 25, 38, 53, 69, 89, 113}
// ---------------------------------------------------------------------------
static __constant__ int8_t s70_iq4nl_table[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113
};

// __byte_perm-based 16-entry table lookup (CUDA branch of
// get_int_from_table_16).  Returns two int32: first holds the 4 codebook bytes
// at even nibble indices of q4, second holds odd nibble indices.
static __device__ __forceinline__ int2
s70_get_int_from_table_16(const int q4, const int8_t * table) {
    const uint32_t * t32 = (const uint32_t *) table;
    const uint32_t sel = (0x32103210u | ((uint32_t)q4 & 0x88888888u) >> 1);
    uint32_t tmp[2];
    #pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t sh = 16u * i;
        const uint32_t lo = __byte_perm(t32[0], t32[1], (uint32_t)q4 >> sh);
        const uint32_t hi = __byte_perm(t32[2], t32[3], (uint32_t)q4 >> sh);
        tmp[i] = __byte_perm(lo, hi, sel >> sh);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420),
                     __byte_perm(tmp[0], tmp[1], 0x7531));
}

static __device__ __forceinline__ int
s70_get_int_b4(const void * x, const int i32) {
    return ((const int *) x)[i32]; // 4-byte aligned
}

// INT8 four-way dot product (DP4A). SM61+ supports __dp4a.
    static __device__ __forceinline__ int
    s70_dp4a(const int a, const int b, int c) {
    #if defined(USE_ROCM)
        const int8_t * a8 = (const int8_t *) &a;
        const int8_t * b8 = (const int8_t *) &b;
        return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
    #elif __CUDA_ARCH__ >= 610
        return __dp4a(a, b, c);
    #else
        const int8_t * a8 = (const int8_t *) &a;
        const int8_t * b8 = (const int8_t *) &b;
        return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
    #endif
    }

// ---------------------------------------------------------------------------
// block_q8_1_mmq — D4 layout activation quantization block.
//
// Matches llama.cpp block_q8_1_mmq (MMQ_Q8_1_DS_LAYOUT_D4): 4 float scales
// (one per 32 values) + 128 int8 quantized values.
// sizeof = 4*4 + 128 = 144 bytes = 36 int32.
// ---------------------------------------------------------------------------
struct s70_block_q8_1_mmq {
    float  d4[4];                      // 1 float scale per 32 values
    int8_t qs[4 * S70_QK8_1];          // 128 int8 values
};

// ---------------------------------------------------------------------------
// Activation quantization kernel: src_t → block_q8_1_mmq (D4), TRANSPOSED.
//
// Ported from quantize.cu quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4>.  The
// output is laid out as qy[k_block][token] (token is the fast/inner index)
// so that the matmul kernel can copy mmq_x contiguous token blocks with a
// single contiguous shared-memory load (matching mul_mat_q_process_tile).
//
// One CTA quantizes 4*QK8_1 = 128 values of one token.  32 threads, each
// loads 4 values, computes amax across the 8 threads sharing a 32-value
// scale group (warp shuffle), writes d4[group] and the int8 qs.
//
// Zero-amax handling: d_inv = 0 → d = 0, q = 0 (avoids Inf/NaN from 127/0).
// ---------------------------------------------------------------------------
template <typename src_t>
static __global__ void
s70_quantize_mmq_q8_1(const src_t * __restrict__ x,
                      s70_block_q8_1_mmq * __restrict__ qy,
                      const int64_t ne0,   // contraction dim (= m)
                      const int     ne1,   // actual token count (= n)
                      const int     ne1_pad) { // padded token stride (= n_padded)
    const int i1   = blockIdx.x;   // token
    const int kb   = blockIdx.y;   // k-block index along contraction dim
    const int tid  = threadIdx.x;   // 0..31

    if (i1 >= ne1) return;  // padding tokens are pre-zeroed by the host

    const int64_t i0 = (int64_t)kb * (4 * S70_QK8_1) + (int64_t)tid * 4;
    const int iqs = tid * 4;  // 0..124, index within this 128-value block

    // Transposed layout: qy[kb * ne1_pad + i1].
    s70_block_q8_1_mmq * y = qy + (int64_t)kb * ne1_pad + i1;

    float4 xi;
    {
        const src_t * px = x + (int64_t)i1 * ne0 + i0;
        xi = make_float4((float)px[0], (float)px[1], (float)px[2], (float)px[3]);
    }

    float amax = fmaxf(fmaxf(fabsf(xi.x), fabsf(xi.y)),
                       fmaxf(fabsf(xi.z), fabsf(xi.w)));

    // Reduce amax across 8 threads (one 32-value scale group = 8 threads × 4 vals).
    #pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, offset));

    // d_inv = 127/amax; guard amax==0 → d=0, q=0.
    const float d_inv = amax > 0.0f ? (127.0f / amax) : 0.0f;

    char4 q;
    q.x = (char)roundf(xi.x * d_inv);
    q.y = (char)roundf(xi.y * d_inv);
    q.z = (char)roundf(xi.z * d_inv);
    q.w = (char)roundf(xi.w * d_inv);

    ((char4 *) y->qs)[iqs / 4] = q;

    // Scale groups at iqs == 0, 32, 64, 96.
    if (iqs % 32 == 0) {
        const float d = d_inv > 0.0f ? (1.0f / d_inv) : 0.0f;
        y->d4[iqs / 32] = d;
    }
}

// ---------------------------------------------------------------------------
// Weight tile loader — DP4A branch of load_tiles_iq4_xs.
//
// Expands one iq4_xs block (QK_K=256 values = 8 sub-blocks of 32) per weight
// row into shared memory using the Q8_0 DP4A tile layout:
//   x_qs[i * S70_X_QS_STRIDE + k0 + {0,4}] : two int32 of codebook-expanded int8
//   x_df[i * S70_X_DF_STRIDE + i/4 + g]    : per-32-value scale d*(ls-32)
//
// x_tile points at this CTA's weight tile in shared memory.  `i` ranges over
// the S70_Y=128 weight rows; need_check clamps i to i_max for tail rows.
// ---------------------------------------------------------------------------
template <int mmq_y, bool need_check>
static __device__ __forceinline__ void
s70_load_tiles_iq4_xs(const char * __restrict__ x, int * __restrict__ x_tile,
                      const int kbx0, const int i_max, const int stride) {
    constexpr int nwarps = S70_NWARPS;
    constexpr int warp_size = 32;
    constexpr int NE_K = S70_TILE_NE_K;

    int   * x_qs = (int *)   x_tile;
    float * x_df = (float *) (x_qs + mmq_y * S70_X_QS_STRIDE);

    // threads_per_row = ITER_K / (4 * QR4_XS) = 256 / 8 = 32.
    constexpr int threads_per_row = S70_ITER_K / (4 * 2 /*QR4_XS*/);
    constexpr int nrows = warp_size / threads_per_row;  // 1
    const int kqsx = threadIdx.x % threads_per_row;

    // --- Expand nibbles into codebook values via LUT ---
    #pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nrows * nwarps) {
        int i = i0 + (nrows == 1 ? threadIdx.y
                                 : threadIdx.y * nrows + threadIdx.x / threads_per_row);
        if (need_check) i = min(i, i_max);

        const block_iq4_xs * bxi =
            (const block_iq4_xs *) x + kbx0 + i * stride;

        const int aux_q4 = s70_get_int_b4(bxi->qs, kqsx);
        const int2 v = s70_get_int_from_table_16(aux_q4, s70_iq4nl_table);
        const int k0 = 8 * (kqsx / 4) + kqsx % 4;

        x_qs[i * S70_X_QS_STRIDE + k0 + 0] = v.x;
        x_qs[i * S70_X_QS_STRIDE + k0 + 4] = v.y;
    }

    // --- Load six-bit scales ---
    // rows_per_warp = warp_size / 8 = 4; 8 warps × 4 rows = 32 rows per pass.
    constexpr int rows_per_warp = warp_size / 8;
    #pragma unroll
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * rows_per_warp) {
        int i = i0 + threadIdx.y * rows_per_warp + threadIdx.x / (NE_K / 4);
        if (need_check) i = min(i, i_max);

        const block_iq4_xs * bxi =
            (const block_iq4_xs *) x + kbx0 + i * stride;

        const float d = __half2float(bxi->d);
        const int g = threadIdx.x % 8;  // sub-block index 0..7
        const int ls = ((bxi->scales_l[g / 2] >> (4 * (g & 1))) & 0x0F)
                     | (((bxi->scales_h >> (2 * g)) & 0x03) << 4);

        x_df[i * S70_X_DF_STRIDE + i / 4 + g] = d * (float)(ls - 32);
    }
}

// ---------------------------------------------------------------------------
// DP4A vector dot — vec_dot_q8_0_q8_1_dp4a, specialized for the IQ4_XS tile.
//
// Accumulates FP32 into sum[].  Each thread owns a (j,i) sub-tile of the
// (mmq_x tokens) × (mmq_y weight rows) output.
// ---------------------------------------------------------------------------
template <int mmq_x, int mmq_y>
static __device__ __forceinline__ void
s70_vec_dot_q8_0_q8_1_dp4a(const int * __restrict__ x, const int * __restrict__ y,
                           float * __restrict__ sum, const int k00) {
    constexpr int nwarps = S70_NWARPS;
    constexpr int warp_size = 32;
    constexpr int NE_K = S70_TILE_NE_K;

    const int   * x_qs = (const int *)   x;
    const float * x_df = (const float *) x + mmq_y * S70_X_QS_STRIDE;
    // block_q8_1_mmq (D4): first 4 int32 are d4 scales, then 32 int32 of qs.
    const int   * y_qs = (const int *)   y + 4;
    const float * y_df = (const float *) y;

    #pragma unroll
    for (int k01 = 0; k01 < NE_K; k01 += S70_VDR) {
        const int k0 = k00 + k01;

        #pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            #pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                // vec_dot_q8_0_q8_1_impl<float, VDR>: dp4a over VDR int32 pairs.
                int sumi = 0;
                const int * xv = &x_qs[i * S70_X_QS_STRIDE + k0];
                const int * yv = &y_qs[j * S70_TILE_Y_K + k0 % NE_K];
                #pragma unroll
                for (int v = 0; v < S70_VDR; ++v)
                    sumi = s70_dp4a(xv[v], yv[v], sumi);

                const float xdf =
                    x_df[i * S70_X_DF_STRIDE + i / (S70_QI8_0 / 2) + k0 / S70_QI8_0];
                const float ydf =
                    y_df[j * S70_TILE_Y_K + (k0 / S70_QI8_1) % (NE_K / S70_QI8_1)];

                sum[j0 / nwarps * mmq_y / warp_size + i0 / warp_size]
                    += xdf * ydf * (float)sumi;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Output write-back — mmq_write_back_dp4a.  Converts FP32 accumulators to the
// destination type and stores into dst[token, output_row].
// ---------------------------------------------------------------------------
template <typename dst_t, int mmq_x, int mmq_y, bool need_check>
static __device__ __forceinline__ void
s70_write_back_dp4a(const float * __restrict__ sum, dst_t * __restrict__ dst,
                    const int stride, const int i_max, const int j_max,
                    const int row_offset, const int token_offset) {
    constexpr int nwarps = S70_NWARPS;
    constexpr int warp_size = 32;

    #pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
        if (j > j_max) return;

        #pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += warp_size) {
            const int i = i0 + threadIdx.x;
            if (need_check && i > i_max) continue;

            const float v =
                sum[j0 / nwarps * mmq_y / warp_size + i0 / warp_size];
            dst[(token_offset + j) * stride + row_offset + i] = (dst_t)v;
        }
    }
}

// ---------------------------------------------------------------------------
// The MMQ kernel for IQ4_XS on SM70.
//
// Grid: (ceil(k / mmq_y), 1).  Each CTA computes one mmq_x (tokens) × mmq_y
// (output rows) tile.  (FastLLM convention: weight [k×m], input [n×m],
// output [n×k].)  Since n ≤ 64 ≤ mmq_x_max, there is exactly one token tile.
//
// The K dimension is walked in ITER_K=256 steps; each step loads one full
// iq4_xs block per weight row into shared mem (via LUT expansion), and two
// halves of the quantized activation tile (each covering NE_K=32 of the
// 64 K-values per q8_1_mmq block — one iq4_xs block spans two q8_1_mmq blocks).
//
//   weight       : block_iq4_xs[k][m/QK_K] (row-major, stride = m/QK_K)
//   qy           : s70_block_q8_1_mmq[m/128][n_padded] (transposed: k-block outer)
//   dst          : dst_t[n][k] (stride = k)
// ---------------------------------------------------------------------------
template <typename dst_t, int mmq_x, int mmq_y, bool need_check>
__launch_bounds__(S70_NWARPS * 32, 1)
static __global__ void
s70_mul_mat_q(const char * __restrict__ weight,
              const s70_block_q8_1_mmq * __restrict__ qy,
              dst_t * __restrict__ dst,
              const int k, const int m, const int n,
              const int weight_stride,   // = m / QK_K
              const int n_pad_stride) {  // = n_padded (token stride in qy)
    constexpr int NE_K = S70_TILE_NE_K;
    constexpr int sz = sizeof(s70_block_q8_1_mmq) / sizeof(int);  // 36

    extern __shared__ int s70_smem[];
    int * tile_y = s70_smem;
    int * tile_x = tile_y + mmq_x * S70_TILE_Y_K;

    // CTA tile origin.
    const int it = blockIdx.x;                // output-row tile index

    const int tile_x_max_i = k - it * mmq_y - 1;
    const int tile_y_max_j = min(n - 1, mmq_x - 1);

    const int kb_blocks_x = m / S70_QK_K;      // iq4_xs blocks per weight row

    constexpr int blocks_per_iter = S70_ITER_K / S70_QK_K;  // 1

    float sum[mmq_x * mmq_y / (S70_NWARPS * 32)] = {0.0f};

    for (int kb0 = 0; kb0 < kb_blocks_x; kb0 += blocks_per_iter) {
        // Load weight tile: expand iq4_xs nibbles + scales into shared mem.
        s70_load_tiles_iq4_xs<mmq_y, need_check>(
            weight, tile_x, it * mmq_y * weight_stride + kb0,
            tile_x_max_i, weight_stride);

        // --- Load first half of activation tile (K-offset 0 within block) ---
        // qy layout [kb_y][token]: contiguous token blocks at
        //   qy + kb_y * n_pad_stride (token tile origin is always 0).
        // First q8_1_mmq sub-block for iq4_xs block kb0 is kb_y = 2*kb0.
        {
            const int kb_y = 2 * kb0;
            const int * by0 = (const int *) qy
                + ((int64_t)kb_y * n_pad_stride) * sz;
            constexpr int tile_y_ints = mmq_x * S70_TILE_Y_K;
            #pragma unroll
            for (int l0 = 0; l0 < tile_y_ints;
                 l0 += S70_NWARPS * 32) {
                const int l = l0 + threadIdx.y * 32 + threadIdx.x;
                if (l < tile_y_ints) tile_y[l] = by0[l];
            }
        }
        __syncthreads();
        s70_vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>(tile_x, tile_y, sum, 0);
        __syncthreads();

        // --- Load second half (K-offset NE_K=32 within block) ---
        {
            const int kb_y = 2 * kb0 + 1;
            const int * by0 = (const int *) qy
                + ((int64_t)kb_y * n_pad_stride) * sz;
            constexpr int tile_y_ints = mmq_x * S70_TILE_Y_K;
            #pragma unroll
            for (int l0 = 0; l0 < tile_y_ints;
                 l0 += S70_NWARPS * 32) {
                const int l = l0 + threadIdx.y * 32 + threadIdx.x;
                if (l < tile_y_ints) tile_y[l] = by0[l];
            }
        }
        __syncthreads();
        s70_vec_dot_q8_0_q8_1_dp4a<mmq_x, mmq_y>(tile_x, tile_y, sum, NE_K);
        __syncthreads();
    }
    s70_write_back_dp4a<dst_t, mmq_x, mmq_y, need_check>(
        sum, dst, k, tile_x_max_i, tile_y_max_j, it * mmq_y, 0);
}

// ===========================================================================
// Host-side eligibility + launch wrapper.
// ===========================================================================

namespace {

bool s70_env_flag(const char *name, bool defaultValue) {
    const char *env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') return defaultValue;
    std::string value(env);
    for (char &ch : value) {
        ch = (char)std::tolower((unsigned char)ch);
    }
    if (value == "0" || value == "off" || value == "false" ||
        value == "no" || value == "disable") return false;
    if (value == "1" || value == "on" || value == "true" ||
        value == "yes" || value == "enable") return true;
    return defaultValue;
}

bool s70_enabled() {
    static const bool enabled =
        s70_env_flag("FASTLLM_CUDA_SM70_IQ4XS_MMQ", true);
    return enabled;
}

bool s70_detailed_log() {
    static const bool enabled =
        s70_env_flag("FASTLLM_CUDA_SM70_IQ4XS_MMQ_LOG", false);
    return enabled;
}

bool s70_current_device(int &device, int &sm) {
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    sm = prop.major * 10 + prop.minor;
    return true;
}

// Smallest tile multiple of 8 in {8,16,32,64} that is >= n.
int s70_pick_mmq_x(int n) {
    if (n <= 8)  return 8;
    if (n <= 16) return 16;
    if (n <= 32) return 32;
    return 64;
}

bool s70_mmq_x_ok(int x) {
    return x == 8 || x == 16 || x == 32 || x == 64;
}

// One default route summary per device; shape-level diagnostics are opt-in.
using LogKey = std::tuple<int, int, int, int, int>;
std::mutex &s70_logmtx() { static std::mutex mutex; return mutex; }
std::set<int> &s70_seen_device() { static std::set<int> seen; return seen; }
std::set<LogKey> &s70_seen_sel() { static std::set<LogKey> seen; return seen; }
std::set<LogKey> &s70_seen_rej() { static std::set<LogKey> seen; return seen; }
std::set<LogKey> &s70_seen_fb() { static std::set<LogKey> seen; return seen; }

void s70_log_sel(int device, int sm, int type, int n, int m, int k) {
    std::lock_guard<std::mutex> lock(s70_logmtx());
    if (s70_detailed_log()) {
        if (s70_seen_sel().insert({sm, type, n, m, k}).second) {
            fprintf(stderr,
                    "[FastLLM][sm70-iq4xs-mmq] selected: type=%d n=%d m=%d k=%d sm=%d device=%d\n",
                    type, n, m, k, sm, device);
        }
    } else if (s70_seen_device().insert(device).second) {
        fprintf(stderr,
                "[FastLLM] SM70 IQ4_XS MMQ enabled on CUDA device %d.\n",
                device);
    }
}

void s70_log_rej(int sm, int type, int n, int m, int k, const char *why) {
    if (!s70_detailed_log()) return;
    std::lock_guard<std::mutex> lock(s70_logmtx());
    if (s70_seen_rej().insert({sm, type, n, m, k}).second) {
        fprintf(stderr,
                "[FastLLM][sm70-iq4xs-mmq] rejected (%s): type=%d n=%d m=%d k=%d sm=%d\n",
                why, type, n, m, k, sm);
    }
}

void s70_log_fb(int sm, int type, int n, int m, int k, const char *why) {
    if (!s70_detailed_log()) return;
    std::lock_guard<std::mutex> lock(s70_logmtx());
    if (s70_seen_fb().insert({sm, type, n, m, k}).second) {
        fprintf(stderr,
                "[FastLLM][sm70-iq4xs-mmq] fallback (%s): type=%d n=%d m=%d k=%d sm=%d\n",
                why, type, n, m, k, sm);
    }
}

struct S70Scratch {
    s70_block_q8_1_mmq *qy = nullptr;
    size_t capacity = 0;
};

s70_block_q8_1_mmq *s70_get_scratch(int device, cudaStream_t stream,
                                    size_t required) {
    using ScratchKey = std::pair<int, uintptr_t>;
    static thread_local std::map<ScratchKey, S70Scratch> scratchByStream;
    S70Scratch &scratch = scratchByStream[{device, (uintptr_t)stream}];
    if (scratch.qy != nullptr && scratch.capacity >= required) {
        return scratch.qy;
    }
    if (scratch.qy != nullptr) {
        cudaError_t err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            cudaGetLastError();
            FastllmCudaSetThreadError();
            return nullptr;
        }
        FastllmCudaFree(scratch.qy);
        scratch.qy = nullptr;
        scratch.capacity = 0;
    }
    scratch.qy = (s70_block_q8_1_mmq *)FastllmCudaMalloc(
        required * sizeof(s70_block_q8_1_mmq));
    if (scratch.qy != nullptr) scratch.capacity = required;
    return scratch.qy;
}

} // namespace

extern "C" {

// Route-introspection helper for tests/diagnostics.
bool FastllmCudaSm70Iq4XsMmqSupported() {
    int device = 0, sm = 0;
    return s70_enabled() && s70_current_device(device, sm) && sm == 70;
}

// See header for full contract.
bool FastllmCudaTrySm70Iq4XsMmq(const void *weight, const void *input,
                                void *output, fastllm::DataType dataType,
                                int n, int m, int k, void *stream) {
    int device = 0, sm = 0;
    const int type = (int)dataType;
    if (!s70_enabled())                 { s70_log_rej(sm,type,n,m,k,"env disabled"); return false; }
    if (!s70_current_device(device,sm)) { s70_log_rej(sm,type,n,m,k,"device query"); return false; }
    if (sm != 70)                       { s70_log_rej(sm,type,n,m,k,"sm!=70");       return false; }
    if (!weight || !input || !output)   { s70_log_rej(sm,type,n,m,k,"null ptr");    return false; }
    if (dataType != fastllm::DataType::FLOAT32 &&
        dataType != fastllm::DataType::FLOAT16 &&
        dataType != fastllm::DataType::BFLOAT16) {
        s70_log_rej(sm,type,n,m,k,"dtype");
        return false;
    }
    if (n < 8 || n > 64)                { s70_log_rej(sm,type,n,m,k,"n range");     return false; }
    if (m <= 0 || (m % S70_QK_K) != 0)  { s70_log_rej(sm,type,n,m,k,"m%256");       return false; }
    if (k < S70_Y)                      { s70_log_rej(sm,type,n,m,k,"k<128");       return false; }

    const int mmq_x = s70_pick_mmq_x(n);
    if (!s70_mmq_x_ok(mmq_x))           { s70_log_rej(sm,type,n,m,k,"tile-x");      return false; }

    constexpr int mmq_y = S70_Y;
    const bool need_check = (k % mmq_y) != 0 || (n != mmq_x);

    const size_t tile_y_bytes = (size_t)mmq_x * S70_TILE_Y_K * sizeof(int);
    const size_t x_qs_ints = (size_t)mmq_y * S70_X_QS_STRIDE;
    const size_t x_df_ints = (size_t)mmq_y * S70_X_DF_STRIDE +
                             (size_t)mmq_y / (S70_QI8_0 / 2);
    const size_t smem_bytes =
        tile_y_bytes + (x_qs_ints + x_df_ints) * sizeof(int);

    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        cudaGetLastError();
        s70_log_rej(sm,type,n,m,k,"getprop");
        return false;
    }
    const size_t smem_limit = prop.sharedMemPerBlockOptin
        ? (size_t)prop.sharedMemPerBlockOptin
        : (size_t)prop.sharedMemPerBlock;
    if (smem_bytes > smem_limit) {
        s70_log_rej(sm,type,n,m,k,"shared mem");
        return false;
    }

    cudaStream_t cuStream = stream ? (cudaStream_t)stream : cudaStreamPerThread;

    const int n_padded = mmq_x;
    const int kb_y_total = m / (4 * S70_QK8_1);
    const size_t qy_count = (size_t)kb_y_total * n_padded;
    s70_block_q8_1_mmq *qy =
        s70_get_scratch(device, cuStream, qy_count);
    if (qy == nullptr) {
        s70_log_fb(sm, type, n, m, k, "q8 scratch");
        return false;
    }

    err = cudaMemsetAsync(
        qy, 0, qy_count * sizeof(s70_block_q8_1_mmq), cuStream);
    if (err != cudaSuccess) {
        cudaGetLastError();
        s70_log_fb(sm, type, n, m, k, "q8 memset");
        return false;
    }

    const dim3 quantBlock(32, 1, 1);
    const dim3 quantGrid((unsigned)n, (unsigned)kb_y_total, 1);
    switch (dataType) {
        case fastllm::DataType::FLOAT32:
            s70_quantize_mmq_q8_1<float><<<quantGrid, quantBlock, 0, cuStream>>>(
                (const float *)input, qy, m, n, n_padded);
            break;
        case fastllm::DataType::FLOAT16:
            s70_quantize_mmq_q8_1<half><<<quantGrid, quantBlock, 0, cuStream>>>(
                (const half *)input, qy, m, n, n_padded);
            break;
        case fastllm::DataType::BFLOAT16:
            s70_quantize_mmq_q8_1<__nv_bfloat16><<<quantGrid, quantBlock, 0, cuStream>>>(
                (const __nv_bfloat16 *)input, qy, m, n, n_padded);
            break;
        default:
            s70_log_rej(sm, type, n, m, k, "dtype-q");
            return false;
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaGetLastError();
        s70_log_fb(sm, type, n, m, k, "q8 launch");
        return false;
    }

    const int weight_stride = m / S70_QK_K;
    const dim3 blockDims(32, S70_NWARPS, 1);
    const dim3 gridDims((unsigned)((k + mmq_y - 1) / mmq_y), 1, 1);

    #define S70_LAUNCH(DT, X, NC)                                                \
        s70_mul_mat_q<DT, X, mmq_y, NC><<<gridDims, blockDims, smem_bytes,       \
                cuStream>>>((const char *)weight, qy, (DT *)output,               \
                            k, m, n, weight_stride, n_padded)

    #define S70_DISP_X(DT)                                                       \
        do {                                                                     \
            if (need_check) {                                                    \
                switch (mmq_x) {                                                 \
                    case 8:  S70_LAUNCH(DT, 8,  true); break;                    \
                    case 16: S70_LAUNCH(DT, 16, true); break;                    \
                    case 32: S70_LAUNCH(DT, 32, true); break;                    \
                    case 64: S70_LAUNCH(DT, 64, true); break;                    \
                    default: s70_log_rej(sm,type,n,m,k,"tile"); return false;   \
                }                                                                \
            } else {                                                             \
                switch (mmq_x) {                                                 \
                    case 8:  S70_LAUNCH(DT, 8,  false); break;                   \
                    case 16: S70_LAUNCH(DT, 16, false); break;                   \
                    case 32: S70_LAUNCH(DT, 32, false); break;                   \
                    case 64: S70_LAUNCH(DT, 64, false); break;                   \
                    default: s70_log_rej(sm,type,n,m,k,"tile"); return false;   \
                }                                                                \
            }                                                                    \
        } while (0)

    switch (dataType) {
        case fastllm::DataType::FLOAT32:  S70_DISP_X(float);         break;
        case fastllm::DataType::FLOAT16:  S70_DISP_X(half);          break;
        case fastllm::DataType::BFLOAT16: S70_DISP_X(__nv_bfloat16); break;
        default:
            s70_log_rej(sm, type, n, m, k, "dtype-mm");
            return false;
    }

    #undef S70_LAUNCH
    #undef S70_DISP_X

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaGetLastError();
        s70_log_fb(sm, type, n, m, k, "launch error");
        return false;
    }

    s70_log_sel(device, sm, type, n, m, k);
    return true;
}

} // extern "C"
