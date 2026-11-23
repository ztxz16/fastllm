#pragma once
#include <type_traits>
#include "naivert/body/attn_common/rows_gemv_mma.cuh"

namespace naivert::attn_rows {
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
// kInputStride is the physical query-row stride in bf16 elements; its default
// retains contiguous rows. Padding can distribute query loads across banks.
// Direct single-split output skips the partial buffer and its first barrier.
// Opt in only when emit writes cannot alias queries or another warp's inputs.
// The final CTA barrier remains; emit must not depend on thread ownership.
template <int kRows, int kDim, int kSplits, int kTilesPerCta, int kWarps,
          int kInputStride = kDim, bool kDirectSingleSplitOutput = false,
          bool kSharedWeights = false,
          class RowPtr, class Emit, class Query = std::nullptr_t>
__device__ static void project_strided_rows_mma(int first_row, int tiles, RowPtr row_ptr,
                                        const __nv_bfloat16* x_norm, float* partials,
                                        Emit emit, int cta = -1, Query query_ptr = nullptr) {
  static_assert(kInputStride >= kDim && kInputStride % 8 == 0, "aligned query row stride");
  static_assert(kRows >= 1 && kRows <= 32, "one to four N8 query tiles");
  static_assert(!kDirectSingleSplitOutput || kSplits == 1,
                "direct output requires one K split");
  static_assert((kDim / kSplits) % 32 == 0 && kDim % kSplits == 0,
                "a K split is a whole number of K=32 steps");
  constexpr int kGroups = (kRows + 7) / 8;
  constexpr int kActiveWarps = kTilesPerCta * kSplits;   // the remaining warps idle
  static_assert(kActiveWarps <= kWarps, "tiles x K splits are warps of one CTA");
  constexpr int kSplitK = kDim / kSplits;
  const int warp = int(threadIdx.x) >> 5;
  const int lane = int(threadIdx.x) & 31;
  const int g = lane >> 2;   // fragment row (and query column) group
  const int q = lane & 3;    // fragment K quad
  const int local_tile = warp / kSplits;
  const int split = warp - local_tile * kSplits;
  const uint4* x_row = reinterpret_cast<const uint4*>(x_norm + std::size_t(g) * kInputStride);
  const int cta_index = cta < 0 ? int(blockIdx.x) : cta;
  for (int tile_base = cta_index * kTilesPerCta; tile_base < tiles;
       tile_base += int(gridDim.x) * kTilesPerCta) {
    const int tile = tile_base + local_tile;
    if (warp < kActiveWarps && tile < tiles) {
      const auto w0 = row_ptr(first_row + tile * 16 + g);
      const auto w1 = row_ptr(first_row + tile * 16 + g + 8);
      float c[kGroups][4] = {};
      const int k_begin = split * kSplitK;
#pragma unroll 8
      for (int k0 = k_begin; k0 < k_begin + kSplitK; k0 += 32) {
        const int k = k0 + q * 8;
        uint4 a0, a1;
        if constexpr (kSharedWeights) {
          a0 = *reinterpret_cast<const uint4*>(w0 + k);
          a1 = *reinterpret_cast<const uint4*>(w1 + k);
        } else {
          a0 = __ldg(reinterpret_cast<const uint4*>(w0 + k));
          a1 = __ldg(reinterpret_cast<const uint4*>(w1 + k));
        }
#pragma unroll
        for (int group = 0; group < kGroups; ++group) {
          const int query = group * 8 + g;
          uint4 b = make_uint4(0u,0u,0u,0u);
          if (query < kRows) {
            if constexpr (std::is_same_v<Query, std::nullptr_t>)
              b = x_row[group * kInputStride + (k >> 3)];
            else b = query_ptr(query, k);
          }
          mma_bf16_16816(c[group], a0.x, a1.x, a0.y, a1.y, b.x, b.y);
          mma_bf16_16816(c[group], a0.z, a1.z, a0.w, a1.w, b.z, b.w);
        }
      }
#pragma unroll
      for (int group = 0; group < kGroups; ++group) {
        if constexpr (kDirectSingleSplitOutput) {
          // Preserve the original +0 fold, including signed-zero behavior.
          if (group * 8 + 2 * q < kRows) {
            emit(first_row + tile * 16 + g, group * 8 + 2 * q, __fadd_rn(0.0f, c[group][0]));
            emit(first_row + tile * 16 + g + 8, group * 8 + 2 * q, __fadd_rn(0.0f, c[group][2]));
          }
          if (group * 8 + 2 * q + 1 < kRows) {
            emit(first_row + tile * 16 + g, group * 8 + 2 * q + 1, __fadd_rn(0.0f, c[group][1]));
            emit(first_row + tile * 16 + g + 8, group * 8 + 2 * q + 1, __fadd_rn(0.0f, c[group][3]));
          }
        } else {
          float* mine = partials + (group * kWarps + warp) * 128;
          mine[g * 8 + 2 * q] = c[group][0];
          mine[g * 8 + 2 * q + 1] = c[group][1];
          mine[(g + 8) * 8 + 2 * q] = c[group][2];
          mine[(g + 8) * 8 + 2 * q + 1] = c[group][3];
        }
      }
    }
    if constexpr (!kDirectSingleSplitOutput) {
      __syncthreads();
      for (int item = int(threadIdx.x); item < kTilesPerCta * 16 * kRows;
           item += int(blockDim.x)) {
        const int t = item / (16 * kRows);
        const int rest = item - t * 16 * kRows;
        const int row = rest / kRows;
        const int query = rest - row * kRows;
        if (tile_base + t < tiles) {
          float sum = 0.0f;
#pragma unroll
          for (int s = 0; s < kSplits; ++s) {
            sum += partials[((query / 8) * kWarps + t * kSplits + s) * 128 + row * 8 + query % 8];
          }
          emit(first_row + (tile_base + t) * 16 + row, query, sum);
        }
      }
    }
    __syncthreads();
  }
}

}  // namespace naivert::attn_rows
