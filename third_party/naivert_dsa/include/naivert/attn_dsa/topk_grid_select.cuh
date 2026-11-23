// Grid-parallel exact top-k selection for the attn_dsa sparse regime.
// Included inside AttnDsaBody after index_topk_stage.cuh.
//
// The shared stage finds the rank-2048 threshold with a block radix sort in one CTA per query
// (over at most kThreads * kTopKSortItemsPerThread = 8448 keys, so it is wrong past that) and
// then sorts the compacted selections to make their order deterministic. Here every CTA takes
// part, and the result is the same set as the sorted definition (value descending, index
// ascending on ties) for any number of visible tokens, in ascending token order.
//
// Two 8-bit digit passes over the ordered 32-bit keys (float4 per thread) histogram into
// per-CTA shared counters and merge with one global add per non-empty bin per CTA (the top
// bytes of the keys fall into a handful of bins, so direct global atomics serialised there).
// They fix the 16-bit prefix of the threshold. A compaction pass then lists, per query, the
// tokens whose keys lie strictly above that prefix ("above", fewer than 2048 of them) and the
// tokens equal to it (the candidates, with their keys). When every query's candidate list fits
// kTopKCandidateCap, one CTA per query finishes alone: the two low digits over the candidates
// in shared memory, then the winners (above, candidates above the exact threshold, the
// lowest-index ties) sorted by token with a block radix sort. Otherwise (a degenerate key
// distribution), and below kTopKCompactMinVisible tokens where the per-query finish's fixed
// cost is not worth it, the full passes finish the selection over all keys.
//
// Workspace (Scratch::kTopKGridOffset, per query): hist[4][kTopKGridBins], count_gt /
// count_eq / prefix_gt / prefix_eq [kTopKCountSlots], counters[2] (above, candidates),
// above[kIndexTopK], candidate_token[cap], candidate_key[cap]. The histograms and counters
// must be zero before the passes: zero_topk_grid() runs in every CTA before the projection
// barrier.

static constexpr int kTopKGridBins = 256;
static constexpr int kTopKGridPasses = 4;
static constexpr int kTopKChunk = kThreads;                // fallback count/write chunk: one token per thread
static constexpr int kTopKVec = 4;                         // floats per thread per scan step (float4 loads)
static constexpr int kTopKScanChunk = kThreads * kTopKVec; // tokens per scan work item
static constexpr int kTopKMaxChunks =
    (Scratch::kMaxSeqLen + Scratch::kQueryRows + kTopKChunk - 1) / kTopKChunk;
static constexpr int kTopKCountSlots = 4096;
static_assert(kTopKMaxChunks <= kTopKCountSlots, "chunk counts per query must fit");
static_assert(Scratch::kMaxSeqLen % kTopKVec == 0, "float4 scans need a padded logits row");
static constexpr int kTopKGridHistWords = kTopKGridPasses * kTopKGridBins;
static constexpr int kTopKCandidateCap = 8192; // candidates (equal 16-bit prefix) per query
// Below this many visible tokens the full passes (two more digit passes, count, prefix,
// ordered write: no sort) beat the compaction + per-query finish (~10 us fixed).
static constexpr int kTopKCompactMinVisible = 16384;
static constexpr int kTopKAboveWords = Scratch::kIndexTopK;
static constexpr int kTopKCounterWords = 2;
static constexpr int kTopKGridWordsPerQuery =
    kTopKGridHistWords + 4 * kTopKCountSlots + kTopKCounterWords + kTopKAboveWords + 2 * kTopKCandidateCap;
static_assert(Scratch::kTopKGridBytes ==
                  std::size_t(Scratch::kQueryRows) * kTopKGridWordsPerQuery * sizeof(unsigned),
              "top-k grid workspace layout");
static constexpr int kTopKPositionBits = PreserveScoreOrder ? 21 : 17;
static_assert(Scratch::kMaxSeqLen + Scratch::kQueryRows <= (1 << kTopKPositionBits),
              "token sorts of the grid finish use kTopKPositionBits");
// Shared words used by select_per_query_topk_grid (carved from the top-k sort storage, which
// the per-query finish then reuses for its block sorts): header, then the per-query 256-bin
// histograms of the current pass.
static constexpr int kTopKGridSharedHeader = 128;
static constexpr int kTopKGridSharedWords = kTopKGridSharedHeader + Scratch::kQueryRows * kTopKGridBins;
static_assert(sizeof(SharedStorage) >= kTopKGridSharedWords * sizeof(unsigned),
              "top-k grid selection must fit shared storage");
static_assert(kThreads * kTopKSortItemsPerThread >= kTopKCandidateCap,
              "the per-query finish sorts the candidate ties with the block radix sort");
static_assert(kThreads == 384, "SelectionSort (384 x 6) sorts the winners of the grid finish");

__device__ __forceinline__ static unsigned *topk_grid_hist(unsigned *base, int query, int pass) {
  return base + std::size_t(query) * kTopKGridWordsPerQuery + pass * kTopKGridBins;
}
__device__ __forceinline__ static unsigned *topk_grid_count_gt(unsigned *base, int query) {
  return base + std::size_t(query) * kTopKGridWordsPerQuery + kTopKGridHistWords;
}
__device__ __forceinline__ static unsigned *topk_grid_count_eq(unsigned *base, int query) {
  return topk_grid_count_gt(base, query) + kTopKCountSlots;
}
__device__ __forceinline__ static unsigned *topk_grid_prefix_gt(unsigned *base, int query) {
  return topk_grid_count_eq(base, query) + kTopKCountSlots;
}
__device__ __forceinline__ static unsigned *topk_grid_prefix_eq(unsigned *base, int query) {
  return topk_grid_prefix_gt(base, query) + kTopKCountSlots;
}
__device__ __forceinline__ static unsigned *topk_grid_counters(unsigned *base, int query) {
  return topk_grid_prefix_eq(base, query) + kTopKCountSlots;
}
__device__ __forceinline__ static unsigned *topk_grid_above(unsigned *base, int query) {
  return topk_grid_counters(base, query) + kTopKCounterWords;
}
__device__ __forceinline__ static unsigned *topk_grid_candidate_token(unsigned *base, int query) {
  return topk_grid_above(base, query) + kTopKAboveWords;
}
__device__ __forceinline__ static unsigned *topk_grid_candidate_key(unsigned *base, int query) {
  return topk_grid_candidate_token(base, query) + kTopKCandidateCap;
}

// Every CTA zeroes a slice of every query's histograms; CTA 0 the list counters (before the
// projection barrier).
__device__ static void zero_topk_grid(unsigned *base) {
  const int stride = int(cta_count()) * kThreads;
  for (int query = 0; query < Scratch::kQueryRows; ++query) {
    unsigned *hist = topk_grid_hist(base, query, 0);
    for (int i = int(cta_index()) * kThreads + int(threadIdx.x); i < kTopKGridHistWords; i += stride) {
      hist[i] = 0u;
    }
    if (int(cta_index()) == 0 && int(threadIdx.x) < kTopKCounterWords) {
      topk_grid_counters(base, query)[threadIdx.x] = 0u;
    }
  }
}

// Warp-level exclusive scan (lane order). Returns the exclusive prefix; *total gets the sum.
__device__ __forceinline__ static unsigned topk_warp_exclusive_scan(unsigned value, int lane,
                                                                    unsigned *total) {
  unsigned inclusive = value;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const unsigned previous = __shfl_up_sync(0xffffffffu, inclusive, offset);
    if (lane >= offset) {
      inclusive += previous;
    }
  }
  *total = __shfl_sync(0xffffffffu, inclusive, 31);
  return inclusive - value;
}

// One warp per query: find the histogram bin holding the `wanted`-th largest key and the
// rank inside it (1-based). Lanes own eight consecutive bins each, highest bins first, so the
// walk is deterministic; every CTA computes the same answer. out[0] = bin, out[1] = rank.
__device__ __forceinline__ static void topk_grid_find_bin_warp(const unsigned *hist, unsigned wanted,
                                                               int lane, unsigned *out) {
  constexpr int kBinsPerLane = kTopKGridBins / 32; // 8
  unsigned mine[kBinsPerLane];
  unsigned sum = 0u;
#pragma unroll
  for (int i = 0; i < kBinsPerLane; ++i) {
    mine[i] = hist[kTopKGridBins - 1 - (lane * kBinsPerLane + i)];
    sum += mine[i];
  }
  unsigned total = 0u;
  unsigned above = topk_warp_exclusive_scan(sum, lane, &total);
#pragma unroll
  for (int i = 0; i < kBinsPerLane; ++i) {
    if (above < wanted && wanted <= above + mine[i]) {
      out[0] = unsigned(kTopKGridBins - 1 - (lane * kBinsPerLane + i));
      out[1] = wanted - above;
    }
    above += mine[i];
  }
  __syncwarp();
}

// One warp per query: exclusive prefix of the chunk counts, in chunk order.
__device__ __forceinline__ static void topk_grid_chunk_prefix_warp(const unsigned *counts, int chunks,
                                                                   int lane, unsigned *prefix_out) {
  unsigned carry = 0u;
  for (int chunk0 = 0; chunk0 < chunks; chunk0 += 32) {
    const int chunk = chunk0 + lane;
    const unsigned value = chunk < chunks ? counts[chunk] : 0u;
    unsigned total = 0u;
    const unsigned before = topk_warp_exclusive_scan(value, lane, &total);
    if (chunk < chunks) {
      prefix_out[chunk] = carry + before;
    }
    carry += total;
  }
  __syncwarp();
}

// Four consecutive logits of one query (the row is padded to kMaxSeqLen, a multiple of 4, so
// the float4 is always readable; elements past `visible` are masked by the callers).
__device__ __forceinline__ static float4 topk_grid_load4(const float *logits, int query, int token0,
                                                         int visible) {
  if (token0 < visible) {
    return *reinterpret_cast<const float4 *>(logits + std::size_t(query) * Scratch::kMaxSeqLen + token0);
  }
  return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

// One 8-bit digit pass over the keys matching the prefix so far: per-CTA shared histograms
// (shared atomics), then one global add per non-empty bin.
__device__ static void topk_grid_digit_pass(const float *logits, unsigned *grid_base, const int *lengths,
                                            int scan_chunks, int scan_items, int pass, const unsigned *prefix,
                                            unsigned *shared_hist) {
  const int tid = int(threadIdx.x);
  const int shift = 24 - 8 * pass;
  const unsigned prefix_mask = pass == 0 ? 0u : (0xffffffffu << unsigned(shift + 8));
  for (int i = tid; i < Scratch::kQueryRows * kTopKGridBins; i += kThreads) {
    shared_hist[i] = 0u;
  }
  __syncthreads();
  for (int item = int(cta_index()); item < scan_items; item += int(cta_count())) {
    const int query = item / scan_chunks;
    const int chunk = item - query * scan_chunks;
    const int token0 = chunk * kTopKScanChunk + tid * kTopKVec;
    const int visible = lengths[query];
    const unsigned want = prefix[query];
    const float4 v = topk_grid_load4(logits, query, token0, visible);
    const float vals[kTopKVec] = {v.x, v.y, v.z, v.w};
    unsigned *hist = shared_hist + query * kTopKGridBins;
#pragma unroll
    for (int j = 0; j < kTopKVec; ++j) {
      unsigned digit = 0xffffffffu;
      if (token0 + j < visible) {
        const unsigned key = ordered_float_key(vals[j]);
        if ((key & prefix_mask) == want) {
          digit = (key >> unsigned(shift)) & 0xffu;
        }
      }
      if (digit != 0xffffffffu) {
        atomicAdd(hist + digit, 1u);
      }
    }
  }
  __syncthreads();
  for (int i = tid; i < Scratch::kQueryRows * kTopKGridBins; i += kThreads) {
    const unsigned count = shared_hist[i];
    if (count != 0u) {
      atomicAdd(topk_grid_hist(grid_base, i / kTopKGridBins, pass) + (i % kTopKGridBins), count);
    }
  }
  __syncthreads();
}

// Append this warp's flagged tokens to a per-query list (one atomic per warp, unordered;
// entries past `cap` are dropped, the counter keeps counting).
__device__ __forceinline__ static void topk_grid_append_warp(bool flag, unsigned token, unsigned key,
                                                             unsigned *counter, unsigned *tokens,
                                                             unsigned *keys, unsigned cap, int lane) {
  const unsigned mask = __ballot_sync(0xffffffffu, flag);
  if (mask == 0u) {
    return;
  }
  unsigned base = 0u;
  if (lane == 0) {
    base = atomicAdd(counter, unsigned(__popc(mask)));
  }
  base = __shfl_sync(0xffffffffu, base, 0);
  if (flag) {
    const unsigned slot = base + unsigned(__popc(mask & ((1u << unsigned(lane)) - 1u)));
    if (slot < cap) {
      tokens[slot] = token;
      if (keys != nullptr) {
        keys[slot] = key;
      }
    }
  }
}

// Fallback: passes 2..3 over all keys, chunk counts, prefixes, ordered write.
__device__ static void topk_grid_full_finish(const float *logits, int *selections, unsigned *grid_base,
                                             const int *lengths, int chunks, int items, int scan_chunks,
                                             int scan_items, unsigned *shared_words, unsigned *shared_hist) {
  auto grid = make_grid(reinterpret_cast<unsigned char *>(grid_base) - Scratch::kTopKGridOffset);
  const int tid = int(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const unsigned lower_lanes = lane == 0 ? 0u : (1u << unsigned(lane)) - 1u;
  unsigned *warp_gt = shared_words + 32; // [16]
  unsigned *warp_eq = shared_words + 48; // [16]
  unsigned *bin_out = shared_words + 64; // [2 * kQueryRows]
  unsigned *prefix = shared_words + 96;  // [kQueryRows] threshold key so far
  unsigned *wanted = shared_words + 112; // [kQueryRows] rank still wanted
#pragma unroll 1
  for (int pass = 2; pass < kTopKGridPasses; ++pass) {
    const int shift = 24 - 8 * pass;
    topk_grid_digit_pass(logits, grid_base, lengths, scan_chunks, scan_items, pass, prefix, shared_hist);
    grid.sync();
    if (warp < Scratch::kQueryRows) {
      topk_grid_find_bin_warp(topk_grid_hist(grid_base, warp, pass), wanted[warp], lane, bin_out + 2 * warp);
    }
    __syncthreads();
    if (tid < Scratch::kQueryRows) {
      prefix[tid] |= bin_out[2 * tid] << unsigned(shift);
      wanted[tid] = bin_out[2 * tid + 1];
    }
    __syncthreads();
  }
  // Count pass: per (query, chunk), keys above and keys equal to the threshold.
  for (int item = int(cta_index()); item < items; item += int(cta_count())) {
    const int query = item / chunks;
    const int chunk = item - query * chunks;
    const int token = chunk * kTopKChunk + tid;
    bool gt = false;
    bool eq = false;
    if (token < lengths[query]) {
      const unsigned key = ordered_float_key(logits[std::size_t(query) * Scratch::kMaxSeqLen + token]);
      gt = key > prefix[query];
      eq = key == prefix[query];
    }
    const int n_gt = __syncthreads_count(gt);
    const int n_eq = __syncthreads_count(eq);
    if (tid == 0) {
      topk_grid_count_gt(grid_base, query)[chunk] = unsigned(n_gt);
      topk_grid_count_eq(grid_base, query)[chunk] = unsigned(n_eq);
    }
  }
  grid.sync();
  // Exclusive prefix of the chunk counts per query, written to global (CTA 0, one warp per
  // query) so it scales past shared memory; a grid barrier publishes it to the write pass.
  if (int(cta_index()) == 0 && warp < Scratch::kQueryRows) {
    topk_grid_chunk_prefix_warp(topk_grid_count_gt(grid_base, warp), chunks, lane,
                                topk_grid_prefix_gt(grid_base, warp));
    topk_grid_chunk_prefix_warp(topk_grid_count_eq(grid_base, warp), chunks, lane,
                                topk_grid_prefix_eq(grid_base, warp));
  }
  grid.sync();
  // Write pass: slot = (keys above the threshold before this token) + (tied keys taken
  // before it); tied keys are taken in ascending token order until `wanted` are placed.
  for (int item = int(cta_index()); item < items; item += int(cta_count())) {
    const int query = item / chunks;
    const int chunk = item - query * chunks;
    const int token = chunk * kTopKChunk + tid;
    const unsigned threshold = prefix[query];
    const unsigned ties_needed = wanted[query];
    bool gt = false;
    bool eq = false;
    if (token < lengths[query]) {
      const unsigned key = ordered_float_key(logits[std::size_t(query) * Scratch::kMaxSeqLen + token]);
      gt = key > threshold;
      eq = key == threshold;
    }
    const unsigned gt_mask = __ballot_sync(0xffffffffu, gt);
    const unsigned eq_mask = __ballot_sync(0xffffffffu, eq);
    if (lane == 0) {
      warp_gt[warp] = unsigned(__popc(gt_mask));
      warp_eq[warp] = unsigned(__popc(eq_mask));
    }
    __syncthreads();
    unsigned gt_before = topk_grid_prefix_gt(grid_base, query)[chunk];
    unsigned eq_before = topk_grid_prefix_eq(grid_base, query)[chunk];
#pragma unroll
    for (int w = 0; w < kWmmaWarps; ++w) {
      gt_before += w < warp ? warp_gt[w] : 0u;
      eq_before += w < warp ? warp_eq[w] : 0u;
    }
    gt_before += unsigned(__popc(gt_mask & lower_lanes));
    eq_before += unsigned(__popc(eq_mask & lower_lanes));
    const bool selected = gt || (eq && eq_before < ties_needed);
    if (selected) {
      const unsigned slot = gt_before + (eq_before < ties_needed ? eq_before : ties_needed);
      selections[query * Scratch::kIndexTopK + int(slot)] = token;
    }
    __syncthreads();
  }
  grid.sync();
}

// One CTA per query: the two low digits over the candidates in shared memory, then
// the winners assembled and sorted by token. prefix16 = the 16-bit prefix found by the grid
// passes (in the top bits), wanted16 = the rank still wanted inside it. The block sorts
// reuse the shared storage, so every value they need is in registers first.
__device__ static void topk_grid_query_finish(int *selections, unsigned *grid_base, int query,
                                              unsigned prefix16, unsigned wanted16, int visible,
                                              unsigned *shared_words, unsigned *shared_hist,
                                              TopKBlockRadixSort::TempStorage *sort_storage) {
  const int tid = int(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  unsigned *bin_out = shared_words + 64; // [2]
  unsigned *state = shared_words + 96;   // [0] prefix [1] wanted [2] above [3] cands
                                         // [4] fill [5] ties present
  unsigned *hist = shared_hist;          // [256]
  const unsigned *counters = topk_grid_counters(grid_base, query);
  const unsigned *cand_token = topk_grid_candidate_token(grid_base, query);
  const unsigned *cand_key = topk_grid_candidate_key(grid_base, query);
  const unsigned *above = topk_grid_above(grid_base, query);
  int *out = selections + query * Scratch::kIndexTopK;
  __syncthreads(); // the caller's shared header has been read by every thread
  if (tid == 0) {
    state[0] = prefix16;
    state[1] = wanted16;
    state[2] = counters[0];
    state[3] = counters[1];
    state[4] = counters[0];
    state[5] = 0u;
  }
  __syncthreads();
  const int n_above = int(state[2]);
  const int n_cand = int(state[3]);
  // Passes 2 and 3 over the candidates (they all carry the 16-bit prefix).
#pragma unroll 1
  for (int pass = 2; pass < kTopKGridPasses; ++pass) {
    const int shift = 24 - 8 * pass;
    const unsigned prefix_mask = 0xffffffffu << unsigned(shift + 8);
    for (int i = tid; i < kTopKGridBins; i += kThreads) {
      hist[i] = 0u;
    }
    __syncthreads();
    const unsigned want = state[0];
    for (int i = tid; i < n_cand; i += kThreads) {
      const unsigned key = cand_key[i];
      if ((key & prefix_mask) == want) {
        atomicAdd(hist + ((key >> unsigned(shift)) & 0xffu), 1u);
      }
    }
    __syncthreads();
    if (warp == 0) {
      topk_grid_find_bin_warp(hist, state[1], lane, bin_out);
    }
    __syncthreads();
    if (tid == 0) {
      state[0] |= bin_out[0] << unsigned(shift);
      state[1] = bin_out[1];
    }
    __syncthreads();
  }
  const unsigned threshold = state[0];
  const unsigned ties_needed = state[1];
  if constexpr (!PreserveScoreOrder) {
    // Two token bitmaps replace radix sorting of winners and low-index ties.
    constexpr int words = (Scratch::kMaxSeqLen + 31) / 32;
    static_assert(sizeof(SharedStorage) >= (words * 2 + 24) * sizeof(unsigned));
    unsigned *greater = shared_words;
    unsigned *equal = greater + words;
    unsigned *warp_totals = equal + words;
    const int used = (visible + 31) / 32;
    __syncthreads();
    for (int i = tid; i < used; i += kThreads)
      greater[i] = equal[i] = 0;
    __syncthreads();
    for (int i = tid; i < n_above; i += kThreads) {
      unsigned token = above[i];
      atomicOr(greater + token / 32, 1u << (token & 31));
    }
    for (int i = tid; i < n_cand; i += kThreads) {
      unsigned key = cand_key[i], token = cand_token[i];
      if (key > threshold)
        atomicOr(greater + token / 32, 1u << (token & 31));
      else if (key == threshold)
        atomicOr(equal + token / 32, 1u << (token & 31));
    }
    __syncthreads();
    const int per_thread = (used + kThreads - 1) / kThreads;
    unsigned n_gt = 0, n_eq = 0;
    for (int j = 0; j < per_thread; ++j) {
      int word = tid * per_thread + j;
      if (word < used) {
        n_gt += __popc(greater[word]);
        n_eq += __popc(equal[word]);
      }
    }
    unsigned total_gt = 0, total_eq = 0;
    unsigned before_gt = topk_warp_exclusive_scan(n_gt, lane, &total_gt);
    unsigned before_eq = topk_warp_exclusive_scan(n_eq, lane, &total_eq);
    if (!lane) {
      warp_totals[warp * 2] = total_gt;
      warp_totals[warp * 2 + 1] = total_eq;
    }
    __syncthreads();
    for (int w = 0; w < warp; ++w) {
      before_gt += warp_totals[2 * w];
      before_eq += warp_totals[2 * w + 1];
    }
    for (int j = 0; j < per_thread; ++j) {
      int word = tid * per_thread + j;
      if (word < used) {
        unsigned gt = greater[word], eq = equal[word];
        unsigned take = 0;
        if (before_eq < ties_needed) {
          int left = min(unsigned(__popc(eq)), ties_needed - before_eq);
          unsigned scan = eq;
          for (int k = 0; k < left; ++k) {
            unsigned bit = scan & -scan;
            take |= bit;
            scan ^= bit;
          }
        }
        unsigned mask = gt | take;
        unsigned slot = before_gt + min(before_eq, ties_needed);
        while (mask) {
          int bit = __ffs(mask) - 1;
          out[slot++] = word * 32 + bit;
          mask &= mask - 1;
        }
        before_gt += __popc(gt);
        before_eq += __popc(eq);
      }
    }
    __syncthreads();
  } else {
    // Winners, unordered: (a) every token above the prefix, (b) the candidates above the exact
    // threshold (counting the ties on the way).
    for (int i = tid; i < n_above; i += kThreads) {
      out[i] = int(above[i]);
    }
    for (int i = tid; i < n_cand; i += kThreads) {
      const unsigned key = cand_key[i];
      if (key > threshold) {
        const unsigned slot = atomicAdd(state + 4, 1u);
        out[slot] = int(cand_token[i]);
      } else if (key == threshold) {
        atomicAdd(state + 5, 1u);
      }
    }
    __syncthreads();
    const unsigned fill = state[4];
    const unsigned ties_present = state[5];
    __syncthreads();
    if (ties_present <= ties_needed) {
      // (c) every tie is a winner (ties_present == ties_needed); order comes from the final sort.
      for (int i = tid; i < n_cand; i += kThreads) {
        if (cand_key[i] == threshold) {
          const unsigned slot = atomicAdd(state + 4, 1u);
          out[slot] = int(cand_token[i]);
        }
      }
      __syncthreads();
    } else {
      // (c) the ties_needed lowest tokens among the ties: block radix sort of the tie tokens
      //     (blocked arrangement: thread t holds sorted ranks t*items .. t*items+items-1).
      unsigned items[kTopKSortItemsPerThread];
#pragma unroll
      for (int j = 0; j < kTopKSortItemsPerThread; ++j) {
        const int i = tid + j * kThreads;
        items[j] = (i < n_cand && cand_key[i] == threshold) ? cand_token[i] : 0xffffffffu;
      }
      __syncthreads();
      TopKBlockRadixSort(*sort_storage).Sort(items, 0, kTopKPositionBits);
#pragma unroll
      for (int j = 0; j < kTopKSortItemsPerThread; ++j) {
        const int rank = tid * kTopKSortItemsPerThread + j;
        if (rank < int(ties_needed)) {
          out[int(fill) + rank] = int(items[j]);
        }
      }
      __syncthreads();
    }
    // Ascending token order (the attention sum order, hence the output bits, is then stable).
    {
      constexpr int kItems = 6;
      static_assert(kThreads * kItems >= Scratch::kIndexTopK, "winner sort capacity");
      unsigned keys[kItems];
#pragma unroll
      for (int j = 0; j < kItems; ++j) {
        const int index = tid * kItems + j;
        keys[j] = index < Scratch::kIndexTopK ? unsigned(out[index]) : 0xffffffffu;
      }
      __syncthreads();
      SelectionSort(*reinterpret_cast<typename SelectionSort::TempStorage *>(sort_storage))
          .Sort(keys, 0, kTopKPositionBits);
#pragma unroll
      for (int j = 0; j < kItems; ++j) {
        const int index = tid * kItems + j;
        if (index < Scratch::kIndexTopK) {
          out[index] = int(keys[j]);
        }
      }
      __syncthreads();
    }
  } // original radix-sort finish
}

// At intermediate lengths sixteen CTAs cover a row and keep their consecutive
// positions in registers through all radix passes. The final scatter needs only
// two counts per CTA, including the lower-position tie break at the cutoff.
static constexpr int kSelectShards = 16;
static constexpr int kShardItems = 4;
static constexpr int kShardedMaxVisible = kSelectShards * kThreads * kShardItems;
__device__ static void select_per_query_topk_sharded(const float *logits, int *selections,
                                                      unsigned *base, const int *lengths,
                                                      unsigned *shared, bool firstReady) {
  auto grid = make_grid(reinterpret_cast<unsigned char *>(base) - Scratch::kTopKGridOffset);
  const int query = cta_index() / kSelectShards, shard = cta_index() % kSelectShards;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int visible = query < Scratch::kQueryRows ? lengths[query] : 0;
  const bool sparse = visible > Scratch::kIndexTopK;
  const int span = (visible + kSelectShards * 32 - 1) / (kSelectShards * 32) * 32;
  const int first = shard * span + warp * kShardItems * 32 + lane;
  const int end = min((shard + 1) * span, visible);
  unsigned keys[kShardItems];
#pragma unroll
  for (int i = 0; i < kShardItems; ++i) {
    const int token = first + i * 32;
    keys[i] = sparse && token < end ? ordered_float_key(logits[size_t(query) * Scratch::kMaxSeqLen + token]) : 0u;
    if (!sparse && token < end)
      selections[query * Scratch::kIndexTopK + token] = token;
  }

  unsigned prefix = 0, mask = 0, wanted = Scratch::kIndexTopK;
  for (int pass = 0; pass < kTopKGridPasses; ++pass) {
    const int shift = 24 - pass * 8;
    if (pass || !firstReady) {
      if (sparse) {
        if (tid < kTopKGridBins) shared[tid] = 0;
        __syncthreads();
#pragma unroll
        for (int i = 0; i < kShardItems; ++i) {
          const unsigned digit = (keys[i] >> shift) & 255u;
          const bool match = first + i * 32 < end && (keys[i] & mask) == prefix;
          const unsigned live = __ballot_sync(0xffffffffu, match);
          if (match) {
            const unsigned peers = __match_any_sync(live, digit);
            if (lane == __ffs(peers) - 1) atomicAdd(shared + digit, unsigned(__popc(peers)));
          }
        }
        __syncthreads();
        if (tid < kTopKGridBins && shared[tid])
          atomicAdd(topk_grid_hist(base, query, pass) + tid, shared[tid]);
      }
      grid.sync();
    }
    if (sparse) {
      unsigned *cut = shared + kTopKGridBins;
      if (warp == 0) topk_grid_find_bin_warp(topk_grid_hist(base, query, pass), wanted, lane, cut);
      __syncthreads();
      prefix |= cut[0] << shift;
      mask |= 255u << shift;
      wanted = cut[1];
    }
  }

  unsigned greater[kShardItems], equal[kShardItems];
  unsigned *warpGreater = shared, *warpEqual = shared + kWmmaWarps;
  if (sparse) {
    unsigned nGreater = 0, nEqual = 0;
#pragma unroll
    for (int i = 0; i < kShardItems; ++i) {
      const bool live = first + i * 32 < end;
      greater[i] = __ballot_sync(0xffffffffu, live && keys[i] > prefix);
      equal[i] = __ballot_sync(0xffffffffu, live && keys[i] == prefix);
      nGreater += __popc(greater[i]);
      nEqual += __popc(equal[i]);
    }
    if (!lane) {
      warpGreater[warp] = nGreater;
      warpEqual[warp] = nEqual;
    }
    __syncthreads();
    nGreater = __reduce_add_sync(0xffffffffu, lane < kWmmaWarps ? warpGreater[lane] : 0u);
    nEqual = __reduce_add_sync(0xffffffffu, lane < kWmmaWarps ? warpEqual[lane] : 0u);
    if (!tid) {
      topk_grid_count_gt(base, query)[shard] = nGreater;
      topk_grid_count_eq(base, query)[shard] = nEqual;
    }
  }
  grid.sync();
  if (sparse) {
    unsigned beforeG = __reduce_add_sync(0xffffffffu, lane < shard ? topk_grid_count_gt(base, query)[lane] : 0u);
    unsigned beforeE = __reduce_add_sync(0xffffffffu, lane < shard ? topk_grid_count_eq(base, query)[lane] : 0u);
    beforeG += __reduce_add_sync(0xffffffffu, lane < warp ? warpGreater[lane] : 0u);
    beforeE += __reduce_add_sync(0xffffffffu, lane < warp ? warpEqual[lane] : 0u);
    const unsigned prior = (1u << lane) - 1u;
#pragma unroll
    for (int i = 0; i < kShardItems; ++i) {
      const unsigned g = beforeG + __popc(greater[i] & prior), e = beforeE + __popc(equal[i] & prior);
      if ((greater[i] & (1u << lane)) || ((equal[i] & (1u << lane)) && e < wanted))
        selections[query * Scratch::kIndexTopK + g + min(e, wanted)] = first + i * 32;
      beforeG += __popc(greater[i]);
      beforeE += __popc(equal[i]);
    }
  }
  grid.sync();
}

__device__ static void select_per_query_topk_grid(const float *logits, int *selections, unsigned *grid_base,
                                                  const int *lengths, unsigned *shared_words,
                                                  TopKBlockRadixSort::TempStorage *sort_storage,
                                                  bool firstReady = false) {
  auto grid = make_grid(reinterpret_cast<unsigned char *>(grid_base) - Scratch::kTopKGridOffset);
  const int tid = int(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  unsigned *bin_out = shared_words + 64;                        // [2 * kQueryRows]
  unsigned *prefix = shared_words + 96;                         // [kQueryRows] threshold key so far
  unsigned *wanted = shared_words + 112;                        // [kQueryRows] rank still wanted
  unsigned *shared_hist = shared_words + kTopKGridSharedHeader; // [kQueryRows][256]
  static_assert(Scratch::kQueryRows <= kWmmaWarps, "one warp per query in the scans");
  const int max_visible = lengths[Scratch::kQueryRows];
  const int chunks = (max_visible + kTopKChunk - 1) / kTopKChunk;
  const int items = Scratch::kQueryRows * chunks;
  const int scan_chunks = (max_visible + kTopKScanChunk - 1) / kTopKScanChunk;
  const int scan_items = Scratch::kQueryRows * scan_chunks;
  if constexpr (!PreserveScoreOrder && Scratch::kQueryRows == 8 && Groups == 1) {
    if (max_visible > kTopKLocalMaxVisible && max_visible <= kShardedMaxVisible &&
        cta_count() >= kSelectShards * Scratch::kQueryRows) {
      select_per_query_topk_sharded(logits, selections, grid_base, lengths, shared_words, firstReady);
      return;
    }
  }
  if (max_visible <= kTopKLocalMaxVisible) {
    // One CTA per query selects in shared memory; one grid barrier instead of 4-7.
    if (int(cta_index()) < Scratch::kQueryRows) {
      select_per_query_topk_local(logits, selections, int(cta_index()), lengths[cta_index()], shared_words,
                                  firstReady ? topk_grid_hist(grid_base, int(cta_index()), 0) : nullptr);
    }
    grid.sync();
    return;
  }
  if (tid < Scratch::kQueryRows) {
    prefix[tid] = 0u;
    wanted[tid] = unsigned(Scratch::kIndexTopK);
  }
  __syncthreads();

  // The full layer already published digit 0 with the Indexer sweep.
  // Other callers retain both scans and their corresponding grid joins.
#pragma unroll 1
  for (int pass = 0; pass < 2; ++pass) {
    const int shift = 24 - 8 * pass;
    if (!firstReady || pass != 0) {
      topk_grid_digit_pass(logits, grid_base, lengths, scan_chunks, scan_items, pass, prefix, shared_hist);
      grid.sync();
    }
    if (warp < Scratch::kQueryRows) {
      topk_grid_find_bin_warp(topk_grid_hist(grid_base, warp, pass), wanted[warp], lane, bin_out + 2 * warp);
    }
    __syncthreads();
    if (tid < Scratch::kQueryRows) {
      prefix[tid] |= bin_out[2 * tid] << unsigned(shift);
      wanted[tid] = bin_out[2 * tid + 1];
    }
    __syncthreads();
  }
  if (max_visible < kTopKCompactMinVisible) {
    topk_grid_full_finish(logits, selections, grid_base, lengths, chunks, items, scan_chunks, scan_items,
                          shared_words, shared_hist);
    return;
  }
  // Compaction: per query, the tokens above the 16-bit prefix and the candidates equal to it.
  for (int item = int(cta_index()); item < scan_items; item += int(cta_count())) {
    const int query = item / scan_chunks;
    const int chunk = item - query * scan_chunks;
    const int token0 = chunk * kTopKScanChunk + tid * kTopKVec;
    const int visible = lengths[query];
    const unsigned prefix16 = prefix[query];
    const float4 v = topk_grid_load4(logits, query, token0, visible);
    const float vals[kTopKVec] = {v.x, v.y, v.z, v.w};
    unsigned *counters = topk_grid_counters(grid_base, query);
    unsigned key[kTopKVec];
    bool above[kTopKVec];
    bool equal[kTopKVec];
    bool any = false;
#pragma unroll
    for (int j = 0; j < kTopKVec; ++j) {
      key[j] = 0u;
      above[j] = false;
      equal[j] = false;
      if (token0 + j < visible) {
        key[j] = ordered_float_key(vals[j]);
        above[j] = (key[j] & 0xffff0000u) > prefix16;
        equal[j] = (key[j] & 0xffff0000u) == prefix16;
      }
      any |= above[j] | equal[j];
    }
    // Flagged tokens are ~0.2 % of the keys: one ballot skips the appends for most warp steps.
    if (__ballot_sync(0xffffffffu, any) != 0u) {
#pragma unroll
      for (int j = 0; j < kTopKVec; ++j) {
        topk_grid_append_warp(above[j], unsigned(token0 + j), key[j], counters,
                              topk_grid_above(grid_base, query), nullptr, unsigned(kTopKAboveWords), lane);
        topk_grid_append_warp(equal[j], unsigned(token0 + j), key[j], counters + 1,
                              topk_grid_candidate_token(grid_base, query),
                              topk_grid_candidate_key(grid_base, query), unsigned(kTopKCandidateCap), lane);
      }
    }
  }
  grid.sync();
  // Every CTA reads the counters: the finish is chosen uniformly across the grid.
  unsigned overflow = 0u;
  for (int query = 0; query < Scratch::kQueryRows; ++query) {
    overflow |= topk_grid_counters(grid_base, query)[1] > unsigned(kTopKCandidateCap) ? 1u : 0u;
  }
  if (overflow == 0u) {
    if (int(cta_index()) < Scratch::kQueryRows) {
      const int query = int(cta_index());
      topk_grid_query_finish(selections, grid_base, query, prefix[query], wanted[query], lengths[query],
                             shared_words, shared_hist, sort_storage);
    }
    grid.sync();
    return;
  }
  topk_grid_full_finish(logits, selections, grid_base, lengths, chunks, items, scan_chunks, scan_items,
                        shared_words, shared_hist);
}
