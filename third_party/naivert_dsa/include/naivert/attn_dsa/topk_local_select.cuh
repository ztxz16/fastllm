// Per-query single-CTA radix select for the small / medium sparse contexts.
// Included inside AttnDsaBody after topk_grid_select.cuh.
//
// Below kTopKLocalMaxVisible visible tokens the grid-wide select (four digit passes over all
// CTAs, then either the compaction path or the full passes: 4-7 grid barriers per layer) costs
// more in barriers and per-pass fixed work than the selection itself. Here CTA q selects query
// q alone: four 8-bit digit passes over the query's ordered keys (re-read from the L2-resident
// logits each pass, float4 per thread) into shared histograms, a warp-level
// bin walk, then one ordered write pass with a block prefix over (above, equal) counts so the
// ties are taken in ascending token order. Same definition and same result as the grid select
// (value descending, index ascending on ties; selections ascending); one grid barrier after it.

static constexpr int kTopKLocalMaxVisible = 12288;
static constexpr int kTopKLocalHistWords = kTopKGridBins;
static constexpr int kTopKLocalHeaderWords = 128;
static constexpr int kTopKLocalSharedWords = kTopKLocalHeaderWords + kTopKLocalHistWords + kTopKGridBins;
static_assert(sizeof(SharedStorage) >= kTopKLocalSharedWords * sizeof(unsigned),
              "local top-k select must fit shared storage");
static_assert(Scratch::kQueryRows <= kWmmaWarps, "one warp scans per query in the grid path");

__device__ static void select_per_query_topk_local(const float *logits, int *selections, int query,
                                                   int visible, unsigned *shared_words,
                                                   const unsigned *first_hist = nullptr) {
  const int tid = int(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  unsigned *header = shared_words;      // [0] prefix [1] wanted [2..3] bin out
  unsigned *totals = shared_words + 16; // [2 buffers][kWmmaWarps][2]
  unsigned *hist = shared_words + kTopKLocalHeaderWords;
  unsigned *bins = hist + kTopKLocalHistWords;
  const float *row = logits + std::size_t(query) * Scratch::kMaxSeqLen;
  const int vec4 = visible >> 2;
  if (visible <= Scratch::kIndexTopK) {
    // A row with at most 2048 visible tokens inside a sparse launch (the launch's later rows
    // are past 2048): the selection is the identity, the attention reads min(visible, K) slots.
    for (int token = tid; token < visible; token += kThreads) {
      selections[query * Scratch::kIndexTopK + token] = token;
    }
    return;
  }
  if (tid == 0) {
    header[0] = 0u;
    header[1] = unsigned(Scratch::kIndexTopK);
  }
  // Initialize the shared histogram before the existing header-publication join.
  for (int i = tid; i < kTopKLocalHistWords; i += kThreads)
    hist[i] = 0u;
  __syncthreads();
#pragma unroll 1
  for (int pass = 0; pass < kTopKGridPasses; ++pass) {
    const int shift = 24 - 8 * pass;
    const unsigned prefix_mask = pass == 0 ? 0u : (0xffffffffu << unsigned(shift + 8));
    const unsigned want = header[0];
    if (pass == 0 && first_hist) {
      if (warp == 0)
        topk_grid_find_bin_warp(first_hist, header[1], lane, header + 2);
    } else {
      for (int i = tid; i < vec4; i += kThreads) {
        const float4 v = reinterpret_cast<const float4 *>(row)[i];
        const float vals[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const unsigned key = ordered_float_key(vals[j]);
          if ((key & prefix_mask) == want) {
            atomicAdd(hist + ((key >> unsigned(shift)) & 0xffu), 1u);
          }
        }
      }
      for (int token = (vec4 << 2) + tid; token < visible; token += kThreads) {
        const unsigned key = ordered_float_key(row[token]);
        if ((key & prefix_mask) == want) {
          atomicAdd(hist + ((key >> unsigned(shift)) & 0xffu), 1u);
        }
      }
      __syncthreads();
      if (tid < kTopKGridBins) {
        bins[tid] = hist[tid];
      }
      __syncthreads();
      if (warp == 0) {
        topk_grid_find_bin_warp(bins, header[1], lane, header + 2);
      }
    }
    __syncthreads();
    if (tid == 0) {
      header[0] |= header[2] << unsigned(shift);
      header[1] = header[3];
    }
    // All merge readers passed a CTA join; hist is dead until the next pass.
    // Clear while publishing the next prefix, then use the existing join.
    if (pass + 1 < kTopKGridPasses) {
      for (int i = tid; i < kTopKLocalHistWords; i += kThreads)
        hist[i] = 0u;
    }
    __syncthreads();
  }
  const unsigned threshold = header[0];
  const unsigned ties_needed = header[1];
  // Ordered write: chunks of kThreads x 4 tokens, thread t holds tokens t*4 .. t*4+3 of the
  // chunk; slot = (keys above the threshold before the token) + min(tied keys before it,
  // ties_needed). Row storage is padded to kMaxSeqLen (a multiple of 4): float4 loads are safe.
  int *out = selections + query * Scratch::kIndexTopK;
  unsigned carry_gt = 0u;
  unsigned carry_eq = 0u;
  const int chunks = (visible + kThreads * 4 - 1) / (kThreads * 4);
  int buffer = 0;
#pragma unroll 1
  for (int chunk = 0; chunk < chunks; ++chunk) {
    const int token0 = chunk * kThreads * 4 + tid * 4;
    float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (token0 < visible) {
      v = *reinterpret_cast<const float4 *>(row + token0);
    }
    const float vals[4] = {v.x, v.y, v.z, v.w};
    bool gt[4];
    bool eq[4];
    unsigned n_gt = 0u;
    unsigned n_eq = 0u;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      gt[j] = false;
      eq[j] = false;
      if (token0 + j < visible) {
        const unsigned key = ordered_float_key(vals[j]);
        gt[j] = key > threshold;
        eq[j] = key == threshold;
      }
      n_gt += gt[j] ? 1u : 0u;
      n_eq += eq[j] ? 1u : 0u;
    }
    unsigned incl_gt = n_gt;
    unsigned incl_eq = n_eq;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const unsigned up_gt = __shfl_up_sync(0xffffffffu, incl_gt, offset);
      const unsigned up_eq = __shfl_up_sync(0xffffffffu, incl_eq, offset);
      if (lane >= offset) {
        incl_gt += up_gt;
        incl_eq += up_eq;
      }
    }
    unsigned *my_totals = totals + buffer * (kWmmaWarps * 2);
    if (lane == 31) {
      my_totals[warp * 2] = incl_gt;
      my_totals[warp * 2 + 1] = incl_eq;
    }
    __syncthreads();
    unsigned gt_run = carry_gt + incl_gt - n_gt;
    unsigned eq_run = carry_eq + incl_eq - n_eq;
    unsigned chunk_gt = 0u;
    unsigned chunk_eq = 0u;
#pragma unroll
    for (int w = 0; w < kWmmaWarps; ++w) {
      const unsigned tg = my_totals[w * 2];
      const unsigned te = my_totals[w * 2 + 1];
      gt_run += w < warp ? tg : 0u;
      eq_run += w < warp ? te : 0u;
      chunk_gt += tg;
      chunk_eq += te;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      if (gt[j]) {
        out[gt_run + (eq_run < ties_needed ? eq_run : ties_needed)] = token0 + j;
        ++gt_run;
      } else if (eq[j]) {
        if (eq_run < ties_needed) {
          out[gt_run + eq_run] = token0 + j;
        }
        ++eq_run;
      }
    }
    carry_gt += chunk_gt;
    carry_eq += chunk_eq;
    buffer ^= 1;
  }
}
