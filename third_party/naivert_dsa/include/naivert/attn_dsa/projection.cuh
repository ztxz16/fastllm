#pragma once
// Projection fragments adapted from NaiveRT rows_gemv_mma.cuh; see UPSTREAM.json.
#include <cuda_bf16.h>
namespace fastllm_dsa_projection {
__device__ __forceinline__ int input_offset(int row, int k) {
  return (k / 64) * (8 * 64) + row * 64 + ((k % 64) ^ ((row & 7) * 8));
}

template <int Threads> __device__ __forceinline__ void projection_sync() {
  __syncthreads();
}
__device__ __forceinline__ void mma_bf16_16816(float (&c)[4], unsigned a0, unsigned a1, unsigned a2,
                                               unsigned a3, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
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
// ceil(kRows/8) * kWarps * 128 floats of shared memory. Every thread of the CTA must call this
// (it synchronises the block). Few large tiles: many K splits keep every SM busy;
// many tiles (the 8192-row indexer projection): fewer splits and several tiles
// per CTA, chosen so the tiles spread over all SMs in one wave.
// Optional kAccumulateK flushes short MMA chains into a separate FP32 sum to
// reduce accumulation error before a BF16 or nonlinear rounding boundary.
template <int kRows, int kDim, int kSplits, int kTilesPerCta, int kWarps, int kAccumulateK = 0,
          int kThreadCount = 0, class RowPtr, class Emit>
__device__ static void project_rows_mma(int first_row, int tiles, RowPtr row_ptr, const __nv_bfloat16 *x_norm,
                                        float *partials, Emit emit, int cta = -1) {
  static_assert(kRows >= 1 && kRows <= 16, "one or two N8 query groups");
  static_assert((kDim / kSplits) % 32 == 0 && kDim % kSplits == 0,
                "a K split is a whole number of K=32 steps");
  static_assert(kThreadCount == 0 || kThreadCount >= kWarps * 32);
  constexpr int kGroups = (kRows + 7) / 8;
  constexpr int kActiveWarps = kTilesPerCta * kSplits; // the remaining warps idle
  static_assert(kActiveWarps <= kWarps, "tiles x K splits are warps of one CTA");
  constexpr int kSplitK = kDim / kSplits;
  static_assert(kAccumulateK == 0 ||
                (kAccumulateK % 32 == 0 && kSplitK % (kAccumulateK ? kAccumulateK : 1) == 0));
  const int warp = int(threadIdx.x) >> 5;
  const int lane = int(threadIdx.x) & 31;
  const int g = lane >> 2; // fragment row (and query column) group
  const int q = lane & 3;  // fragment K quad
  const int local_tile = warp / kSplits;
  const int split = warp - local_tile * kSplits;
  const int cta_index = cta < 0 ? int(blockIdx.x) : cta;
  for (int tile_base = cta_index * kTilesPerCta; tile_base < tiles;
       tile_base += int(gridDim.x) * kTilesPerCta) {
    const int tile = tile_base + local_tile;
    if (warp < kActiveWarps && tile < tiles) {
      const __nv_bfloat16 *w0 = row_ptr(first_row + tile * 16 + g);
      const __nv_bfloat16 *w1 = row_ptr(first_row + tile * 16 + g + 8);
      float c[kGroups][4] = {};
      float accumulated[kGroups][4] = {};
      const int k_begin = split * kSplitK;
#pragma unroll 8
      for (int k0 = k_begin; k0 < k_begin + kSplitK; k0 += 32) {
        const int k = k0 + q * 8;
        const uint4 a0 = __ldg(reinterpret_cast<const uint4 *>(w0 + k));
        const uint4 a1 = __ldg(reinterpret_cast<const uint4 *>(w1 + k));
        // Each weight fragment serves both N8 groups before advancing K.
#pragma unroll
        for (int group = 0; group < kGroups; ++group) {
          uint4 b = make_uint4(0u, 0u, 0u, 0u);
          if (group * 8 + g < kRows)
            b = *reinterpret_cast<const uint4 *>(x_norm + group * 8 * kDim + input_offset(g, k));
          mma_bf16_16816(c[group], a0.x, a1.x, a0.y, a1.y, b.x, b.y);
          mma_bf16_16816(c[group], a0.z, a1.z, a0.w, a1.w, b.z, b.w);
          if constexpr (kAccumulateK > 0) {
            if ((k0 - k_begin + 32) % kAccumulateK == 0) {
#pragma unroll
              for (int i = 0; i < 4; ++i) {
                accumulated[group][i] += c[group][i];
                c[group][i] = 0.0f;
              }
            }
          }
        }
      }
#pragma unroll
      for (int group = 0; group < kGroups; ++group) {
        if constexpr (kAccumulateK > 0) {
#pragma unroll
          for (int i = 0; i < 4; ++i) c[group][i] = accumulated[group][i];
        }
        float *mine = partials + (group * kWarps + warp) * 128;
        mine[g * 8 + 2 * q] = c[group][0];
        mine[g * 8 + 2 * q + 1] = c[group][1];
        mine[(g + 8) * 8 + 2 * q] = c[group][2];
        mine[(g + 8) * 8 + 2 * q + 1] = c[group][3];
      }
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
          sum += partials[(query / 8 * kWarps + t * kSplits + s) * 128 + row * 8 + query % 8];
        }
        emit(first_row + (tile_base + t) * 16 + row, query, sum);
      }
    }
    projection_sync<kThreadCount>();
  }
}

} // namespace fastllm_dsa_projection
