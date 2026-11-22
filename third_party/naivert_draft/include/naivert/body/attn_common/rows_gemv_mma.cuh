#pragma once
#include "naivert/common/load_fp32x8.cuh"
// Multi-row RMSNorm and bf16 GEMV for the Naive N attention bodies.
//
// A decode-time projection is a GEMV with up to eight query rows. The first
// version normalised the rows one after another with two block reductions each
// and ran the GEMV as one warp per weight row times one pass per query, so both
// phases grew linearly with the row count (8 rows: RMSNorm 3.4 -> 15 us,
// projection 4-6 -> 17-31 us). Here every warp normalises one row from
// registers, and the projection is an HMMA m16n8k16 with the 16 weight rows of a
// tile in M and the query rows in N: every weight byte is read once for all
// rows. The K dimension is permuted identically in A and B (each lane loads
// eight consecutive K of a row as one 16-byte piece), which leaves the dot
// product exact.
#include <cuda_bf16.h>
#include <cstddef>
#include <cstdint>
#include <cutlass/arch/barrier.h>

namespace naivert::attn_rows {

struct PackedNormGamma {
  static constexpr int Lanes = 32;
  static constexpr int Halves = 2;
  static constexpr int Vector = 4;
  static constexpr int Piece = Lanes * Halves * Vector;
  __host__ __device__ static constexpr int offset(int k) {
    const int within = k % Piece;
    return (k / Piece) * Piece + ((within / Vector) % Halves) * Lanes * Vector +
           (within / (Halves * Vector)) * Vector + within % Vector;
  }
};

struct K32PackedRow {
  static constexpr int Rows = 16, Page = 32;
  const __nv_bfloat16* base;
  __device__ const __nv_bfloat16* operator+(int k) const {
    return base + (k / Page) * (Rows * Page) + k % Page;
  }
};

// Threads==0 preserves the full-CTA contract. Explicit groups use CUTLASS's
// hardware barrier15; the caller must restrict every operation to that group.
template <int Threads>
__device__ __forceinline__ void projection_sync() {
  if constexpr (Threads == 0) {
    __syncthreads();
  } else {
    static_assert(Threads == 256);
    cutlass::arch::NamedBarrier::sync(
        Threads, static_cast<cutlass::arch::ReservedNamedBarriers>(15));
  }
}

__device__ __forceinline__ float warp_sum_f32(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

// x_norm[row] = bf16(residual[row] * rsqrt(mean(residual[row]^2) + eps) * gamma)
// for kRows rows of kDim bf16; warp w normalises rows w, w + warps, ... from
// registers (kDim / 256 16-byte pieces per lane). Ends with __syncthreads.
template <int kRows, int kDim, bool kWideGamma = false>
__device__ static void normalize_rows(const __nv_bfloat16* residual, const float* gamma,
                                      __nv_bfloat16* x_norm, float eps) {
  static_assert(kDim % 256 == 0, "a lane owns kDim / 256 16-byte pieces of a row");
  constexpr int kPieces = kDim / 256;
  const int warp = int(threadIdx.x) >> 5;
  const int lane = int(threadIdx.x) & 31;
  const int warps = int(blockDim.x) >> 5;
  for (int row = warp; row < kRows; row += warps) {
    const uint4* source =
        reinterpret_cast<const uint4*>(residual + std::size_t(row) * kDim);
    uint4 staged[kPieces];
    float sum = 0.0f;
#pragma unroll
    for (int piece = 0; piece < kPieces; ++piece) {
      staged[piece] = __ldg(source + lane + piece * 32);
      const unsigned words[4] = {staged[piece].x, staged[piece].y, staged[piece].z,
                                 staged[piece].w};
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        const float2 value =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(words + w));
        sum = fmaf(value.x, value.x, sum);
        sum = fmaf(value.y, value.y, sum);
      }
    }
    const float inverse_rms = rsqrtf(warp_sum_f32(sum) / float(kDim) + eps);
#pragma unroll
    for (int piece = 0; piece < kPieces; ++piece) {
      const int k = (lane + piece * 32) * 8;
      const unsigned words[4] = {staged[piece].x, staged[piece].y, staged[piece].z,
                                 staged[piece].w};
      uint4 out;
      unsigned* out_words = reinterpret_cast<unsigned*>(&out);
      float4 g_lo, g_hi;
      if constexpr (kWideGamma) {
        // The attention launcher selects this instantiation only for 32-byte-aligned
        // gamma. Keep default caching: the CuTe no-allocation atom regresses.
        naivert::load_fp32x8_aligned32(gamma + k, g_lo, g_hi);
      } else {
        g_lo = *reinterpret_cast<const float4*>(gamma + k);
        g_hi = *reinterpret_cast<const float4*>(gamma + k + 4);
      }
      const float g[8] = {g_lo.x, g_lo.y, g_lo.z, g_lo.w, g_hi.x, g_hi.y, g_hi.z, g_hi.w};
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        const float2 value =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(words + w));
        __nv_bfloat162 normalised = __floats2bfloat162_rn(
            value.x * inverse_rms * g[2 * w], value.y * inverse_rms * g[2 * w + 1]);
        out_words[w] = *reinterpret_cast<unsigned*>(&normalised);
      }
      *reinterpret_cast<uint4*>(x_norm + std::size_t(row) * kDim + k) = out;
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void mma_bf16_16816(float (&c)[4], unsigned a0, unsigned a1,
                                               unsigned a2, unsigned a3, unsigned b0,
                                               unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// emit(row, query, sum_k W[row][k] * x_norm[query][k]) for the 16-row weight
// tiles covering rows [first_row, first_row + 16 * tiles); row_ptr(row) is the
// bf16 row of W (kDim wide, 16-byte aligned). A CTA works on kTilesPerCta tiles
// at a time (tiles blockIdx.x * kTilesPerCta + ..., then strided by the grid);
// within a tile, kSplits warps each stream a kDim /
// kSplits slice of its 16 rows (two 16-byte loads per row per K=32 step, two
// MMAs) with the query rows in N (zero for n >= kRows). Split partials are
// summed in shared memory in a fixed order (deterministic). `partials` is
// kWarps * 128 floats of shared memory. Every thread of the CTA must call this
// (it synchronises the block). Few large tiles: many K splits keep every SM busy;
// many tiles (the 8192-row indexer projection): fewer splits and several tiles
// per CTA, chosen so the tiles spread over all SMs in one wave.
// Optional kAccumulateK flushes short MMA chains into a separate FP32 sum to
// reduce accumulation error before a BF16 or nonlinear rounding boundary.
template <int kRows, int kDim, int kSplits, int kTilesPerCta, int kWarps, int kAccumulateK = 0, int kThreadCount = 0, class RowPtr,
          class Emit>
__device__ static void project_rows_mma(int first_row, int tiles, RowPtr row_ptr,
                                        const __nv_bfloat16* x_norm, float* partials,
                                        Emit emit, int cta = -1) {
  static_assert(kRows >= 1 && kRows <= 8, "query rows live in the MMA's N = 8");
  static_assert((kDim / kSplits) % 32 == 0 && kDim % kSplits == 0,
                "a K split is a whole number of K=32 steps");
  static_assert(kThreadCount == 0 || kThreadCount >= kWarps * 32);
  constexpr int kActiveWarps = kTilesPerCta * kSplits;   // the remaining warps idle
  static_assert(kActiveWarps <= kWarps, "tiles x K splits are warps of one CTA");
  constexpr int kSplitK = kDim / kSplits;
  static_assert(kAccumulateK == 0 || (kAccumulateK % 32 == 0 && kSplitK % kAccumulateK == 0));
  const int warp = int(threadIdx.x) >> 5;
  const int lane = int(threadIdx.x) & 31;
  const int g = lane >> 2;   // fragment row (and query column) group
  const int q = lane & 3;    // fragment K quad
  const int local_tile = warp / kSplits;
  const int split = warp - local_tile * kSplits;
  const uint4* x_row = reinterpret_cast<const uint4*>(x_norm + std::size_t(g) * kDim);
  const int cta_index = cta < 0 ? int(blockIdx.x) : cta;
  for (int tile_base = cta_index * kTilesPerCta; tile_base < tiles;
       tile_base += int(gridDim.x) * kTilesPerCta) {
    const int tile = tile_base + local_tile;
    if (warp < kActiveWarps && tile < tiles) {
      const __nv_bfloat16* w0 = row_ptr(first_row + tile * 16 + g);
      const __nv_bfloat16* w1 = row_ptr(first_row + tile * 16 + g + 8);
      float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      float accumulated[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      const int k_begin = split * kSplitK;
#pragma unroll 8
      for (int k0 = k_begin; k0 < k_begin + kSplitK; k0 += 32) {
        const int k = k0 + q * 8;
        const uint4 a0 = __ldg(reinterpret_cast<const uint4*>(w0 + k));
        const uint4 a1 = __ldg(reinterpret_cast<const uint4*>(w1 + k));
        uint4 b = make_uint4(0u, 0u, 0u, 0u);
        if (g < kRows) {
          b = x_row[k >> 3];
        }
        // K slots (2q, 2q+1) <- k..k+1, (2q+8, 2q+9) <- k+2..k+3; then k+4..k+7.
        mma_bf16_16816(c, a0.x, a1.x, a0.y, a1.y, b.x, b.y);
        mma_bf16_16816(c, a0.z, a1.z, a0.w, a1.w, b.z, b.w);
        if constexpr (kAccumulateK > 0) {
          if ((k0 - k_begin + 32) % kAccumulateK == 0) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { accumulated[i] += c[i]; c[i] = 0.0f; }
          }
        }
      }
      if constexpr (kAccumulateK > 0) {
#pragma unroll
        for (int i = 0; i < 4; ++i) c[i] = accumulated[i];
      }
      float* mine = partials + warp * 128;
      mine[g * 8 + 2 * q] = c[0];
      mine[g * 8 + 2 * q + 1] = c[1];
      mine[(g + 8) * 8 + 2 * q] = c[2];
      mine[(g + 8) * 8 + 2 * q + 1] = c[3];
    }
    projection_sync<kThreadCount>();
    for (int item = int(threadIdx.x); item < kTilesPerCta * 16 * kRows;
         item += (kThreadCount ? kThreadCount : int(blockDim.x))) {
      const int t = item / (16 * kRows);
      const int rest = item - t * 16 * kRows;
      const int row = rest / kRows;
      const int query = rest - row * kRows;
      if (tile_base + t < tiles) {
        float sum = 0.0f;
#pragma unroll
        for (int s = 0; s < kSplits; ++s) {
          sum += partials[(t * kSplits + s) * 128 + row * 8 + query];
        }
        emit(first_row + (tile_base + t) * 16 + row, query, sum);
      }
    }
    projection_sync<kThreadCount>();
  }
}

// One CTA's 16-row weight tile staged in shared memory ahead of the projection (issued before
// the RMSNorm as 16-byte cp.async pieces), laid out [K block 64][16 rows][128 B] with the 16-byte
// chunk index XORed by (row & 7): a warp's fragment loads (8 row groups x 4 K quads) then touch 32
// distinct chunks over four 128-byte lines (conflict-free) and the MMAs' K order is unchanged.
template <int kDim>
struct StagedTile {
  static constexpr int kRowsPerTile = 16;
  static constexpr int kBlockBytes = kRowsPerTile * 128;   // 2 KiB per 64-K block
  static constexpr int kBytes = kRowsPerTile * kDim * 2;    // 128 KiB at kDim 4096
  static constexpr int kChunks = kBytes / 16;
  static_assert(kDim % 64 == 0, "64-K blocks");
  __device__ static __forceinline__ int offset(int row, int k) {   // k a multiple of 8
    const int kb = k >> 6;
    const int c = (k & 63) >> 3;
    return kb * kBlockBytes + row * 128 + ((c ^ (row & 7)) << 4);
  }
  // Every thread issues its share of the pieces and commits; wait() before the consumer's sync.
  template <class RowPtr>
  __device__ static void issue(RowPtr row_ptr, int first_row, unsigned char* stage) {
    constexpr int kChunksPerRow = kDim / 8;
    for (int i = int(threadIdx.x); i < kChunks; i += int(blockDim.x)) {
      const int row = i / kChunksPerRow;
      const int k = (i - row * kChunksPerRow) * 8;
      const __nv_bfloat16* src = row_ptr(first_row + row) + k;
      unsigned long long src_global;
      asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(src_global) : "l"(src));
      const unsigned dst = static_cast<unsigned>(__cvta_generic_to_shared(stage + offset(row, k)));
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src_global) : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  __device__ static __forceinline__ void wait() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }
};

// project_rows_mma for one tile per CTA (tile = blockIdx.x) whose A operand was staged by
// StagedTile::issue: same fragments, MMAs, partial order and emit as the global-memory version.
template <int kRows, int kDim, int kSplits, int kWarps, int kThreadCount = 0, class Emit>
__device__ static void project_rows_mma_staged(int first_row, int tiles, const unsigned char* stage,
                                               const __nv_bfloat16* x_norm, float* partials, Emit emit) {
  using Tile = StagedTile<kDim>;
  static_assert(kRows >= 1 && kRows <= 8, "query rows live in the MMA's N = 8");
  static_assert((kDim / kSplits) % 32 == 0 && kDim % kSplits == 0, "a K split is a whole number of K=32 steps");
  static_assert(kSplits <= kWarps, "K splits are warps of one CTA");
  constexpr int kSplitK = kDim / kSplits;
  const int warp = int(threadIdx.x) >> 5;
  const int lane = int(threadIdx.x) & 31;
  const int g = lane >> 2;
  const int q = lane & 3;
  const int tile = int(blockIdx.x);
  const uint4* x_row = reinterpret_cast<const uint4*>(x_norm + std::size_t(g) * kDim);
  if (tile < tiles) {
    Tile::wait();
  }
  __syncthreads();   // every thread's pieces have landed
  if (tile < tiles && warp < kSplits) {
    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int k_begin = warp * kSplitK;
#pragma unroll 8
    for (int k0 = k_begin; k0 < k_begin + kSplitK; k0 += 32) {
      const int k = k0 + q * 8;
      const uint4 a0 = *reinterpret_cast<const uint4*>(stage + Tile::offset(g, k));
      const uint4 a1 = *reinterpret_cast<const uint4*>(stage + Tile::offset(g + 8, k));
      uint4 b = make_uint4(0u, 0u, 0u, 0u);
      if (g < kRows) {
        b = x_row[k >> 3];
      }
      mma_bf16_16816(c, a0.x, a1.x, a0.y, a1.y, b.x, b.y);
      mma_bf16_16816(c, a0.z, a1.z, a0.w, a1.w, b.z, b.w);
    }
    float* mine = partials + warp * 128;
    mine[g * 8 + 2 * q] = c[0];
    mine[g * 8 + 2 * q + 1] = c[1];
    mine[(g + 8) * 8 + 2 * q] = c[2];
    mine[(g + 8) * 8 + 2 * q + 1] = c[3];
  }
  static_assert(kThreadCount == 0 || kThreadCount >= kWarps * 32, "the known block size covers the MMA warps");
  __syncthreads();
  if (tile < tiles) {
    for (int item = int(threadIdx.x); item < 16 * kRows;
         item += (kThreadCount ? kThreadCount : int(blockDim.x))) {   // Known block size
      const int row = item / kRows;
      const int query = item - row * kRows;
      float sum = 0.0f;
#pragma unroll
      for (int s = 0; s < kSplits; ++s) {
        sum += partials[s * 128 + row * 8 + query];
      }
      emit(first_row + tile * 16 + row, query, sum);
    }
  }
  __syncthreads();
}

}  // namespace naivert::attn_rows
