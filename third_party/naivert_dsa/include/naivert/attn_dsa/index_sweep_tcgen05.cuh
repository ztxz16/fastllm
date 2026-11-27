// Class-scope fragment of the attn_dsa body (after index_topk_stage.cuh): the index-logit sweep
// on tcgen05 (Naive N only).
//
// Keys: args.ki, e4m3 [pos][128 B] in the UMMA K-major SW128 image (chunk ^ (pos & 7)) with
// args.ki_scale[pos]; queries: the same image with column = (row, head) (rotate_index_queries),
// their scales folded into index_weights_scaled. Per 128-position tile: one 16 KB bulk copy into a
// 3-stage ring; warp 0 issues tcgen05.mma.kind::f8f6f4 (M128 x N x K32, four K steps) per
// pass of up to 256 query columns into one of up to two TMEM accumulators and commits to an
// mbarrier; warps 4..11 (two groups of four, one per accumulator, each warp its 32-lane quadrant)
// read the accumulator with tcgen05.ld, apply relu x w', sum the index heads of each row, multiply by
// ki_scale[pos] and store logits[row][pos] for pos < visible(row). Same math as the mma.sync sweep.
__device__ __forceinline__ static unsigned tc_smem_u32(const void *p) {
  return cute::cast_smem_ptr_to_uint(p);
}
__device__ __forceinline__ static void tc_mbar_init(unsigned long long *bar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(tc_smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ static void tc_mbar_expect_tx(unsigned long long *bar, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(tc_smem_u32(bar)), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ static void tc_mbar_arrive(unsigned long long *bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(tc_smem_u32(bar)) : "memory");
}
__device__ __forceinline__ static void tc_mbar_wait(unsigned long long *bar, unsigned phase) {
  const unsigned addr = tc_smem_u32(bar);
  unsigned done = 0;
  do {
    asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
                 "selp.u32 %0, 1, 0, p;\n}"
                 : "=r"(done)
                 : "r"(addr), "r"(phase)
                 : "memory");
  } while (!done);
}
__device__ __forceinline__ static void tc_bulk_g2s(void *smem, const void *gmem, unsigned bytes,
                                                   unsigned long long *bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
          tc_smem_u32(smem)),
      "l"(gmem), "r"(bytes), "r"(tc_smem_u32(bar))
      : "memory");
}
__device__ __forceinline__ static void tc_umma_commit(unsigned long long *bar) { // one thread
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(tc_smem_u32(bar))
      : "memory");
}
__device__ __forceinline__ static void tc_fence_after() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}
__device__ __forceinline__ static void tc_fence_before() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}
__device__ __forceinline__ static void tc_wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}
template <class ArgsLike>
__device__ static void compute_index_logits_tcgen05(const ArgsLike &args, const unsigned char *iq_image,
                                                    const float *index_weights_scaled, float *logits,
                                                    const int *lengths, SweepStorage &sw,
                                                    bool releaseAllocationPermit, bool firstReady) {
  using S = SweepStorage;
  // With eight query rows and sixteen index heads only warps 4..7 read TMEM.
  // The full layer uses the otherwise idle warps 2,3,8..11 to assist the producer.
  constexpr int kProducers = PreserveScoreOrder ? 1 : 7;
  static_assert(PreserveScoreOrder || S::kPasses == 1, "cooperative producer warp layout");
  using Element = std::conditional_t<Fp8Key, cutlass::float_e4m3_t, cutlass::bfloat16_t>;

  const int tid = int(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int max_visible = lengths[Scratch::kQueryRows];
  const int tiles = (max_visible + S::kTileM - 1) / S::kTileM;
  static_assert(S::kStages >= 2 && S::kTileBytes >= Scratch::kQueryRows * 256 * sizeof(unsigned),
                "the unused A-ring slot must fit the first histogram");
  // firstReady implies tiles <= gridDim: no CTA accesses ring stage 1.
  auto *first_hist = reinterpret_cast<unsigned *>(sw.a_ring + S::kTileBytes);
  auto key = args.indexKey;
  key.Bind(args.indexCache);
  if (tid == 0) {
    for (int s = 0; s < S::kStages; ++s) {
      tc_mbar_init(sw.full + s, unsigned(kProducers));
      tc_mbar_init(sw.empty + s, 1u);
    }
    for (int p = 0; p < S::kPasses; ++p) {
      tc_mbar_init(sw.acc_full + p, 1u);
      tc_mbar_init(sw.acc_empty + p, 4u);
    }
    tc_mbar_init(&sw.b_full, 1u);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  for (int i = tid; i < S::kPaddedCols; i += kThreads) {
    sw.weights[i] = i < S::kCols ? (float)args.weights[min(i / 16, args.activeRows - 1) * 16 + i % 16] : 0.0f;
    sw.scales[i] = i < S::kCols ? index_weights_scaled[i] : 0.0f;
  }
  for (int i = S::kCols * Scratch::kIndexDim * (Fp8Key ? 1 : 2) + tid;
       i < S::kPaddedCols * Scratch::kIndexDim * (Fp8Key ? 1 : 2); i += kThreads) {
    sw.b_image[i] = 0;
  }
  // The full layer accumulates the first radix digit while scores are resident.
  // This join also publishes the zeroed counters to all four epilogue warps.
  if (!PreserveScoreOrder && firstReady) {
    for (int i = tid; i < Scratch::kQueryRows * 256; i += kThreads)
      first_hist[i] = 0;
  }
  // Publish ordinary shared stores to the padded B columns before UMMA.
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  if (warp == 0) {
    cute::TMEM::Allocator1Sm allocator;
    allocator.allocate(S::kTmemCols, &sw.tmem_slot);
    // A full-layer CTA may sweep another request before it exits.
    // Relinquishing the permit forbids subsequent allocations by this CTA.
    if (releaseAllocationPermit)
      allocator.release_allocation_lock();
  }
  __syncthreads();
  const unsigned tmem_base = sw.tmem_slot;
  using AtomK = cute::UMMA::Layout_K_SW128_Atom<Element>; // (8, 128) e4m3, 128-byte swizzle
  auto a_layout =
      cute::tile_to_shape(AtomK{}, cute::Shape<cute::Int<S::kTileM>, cute::Int<Fp8Key ? 128 : 64>>{});
  auto b_layout =
      cute::tile_to_shape(AtomK{}, cute::Shape<cute::Int<S::kPassN>, cute::Int<Fp8Key ? 128 : 64>>{});
  const cute::UMMA::InstrDescriptor idesc =
      cute::UMMA::make_instr_desc<Element, Element, float, S::kTileM, S::kPassN, cute::UMMA::Major::K,
                                  cute::UMMA::Major::K>();
  const unsigned long long idescE = static_cast<unsigned long long>(static_cast<unsigned>(idesc)) << 32;
  using Mma = std::conditional_t<Fp8Key, cute::SM100_MMA_F8F6F4_SS,
                                 cute::SM100_MMA_F16BF16_SS<Element, Element, float, S::kTileM, S::kPassN,
                                                            cute::UMMA::Major::K, cute::UMMA::Major::K>>;
  if (PreserveScoreOrder ? warp == 1 : (warp != 0 && (warp < 4 || warp >= 8))) {
    // Producer: the query image once, then the A tile ring (CTA-strided tiles).
    if (warp == 1 && lane == 0) {
      constexpr unsigned kBBytes = unsigned(S::kCols * Scratch::kIndexDim * (Fp8Key ? 1 : 2));
      tc_mbar_expect_tx(&sw.b_full, kBBytes);
      for (unsigned off = 0; off < kBBytes; off += unsigned(S::kTileBytes)) {
        const unsigned bytes =
            kBBytes - off < unsigned(S::kTileBytes) ? kBBytes - off : unsigned(S::kTileBytes);
        tc_bulk_g2s(sw.b_image + off, iq_image + off, bytes, &sw.b_full);
      }
    }
    int i = 0;
    for (int tile = int(cta_index()); tile < tiles; tile += int(cta_count()), ++i) {
      const int stage = i % S::kStages;
      if (lane == 0)
        tc_mbar_wait(sw.empty + stage, unsigned(((i / S::kStages) & 1) ^ 1));
      __syncwarp();
      constexpr int pieces = Fp8Key ? 8 : 16;
      for (int piece = (PreserveScoreOrder ? 0 : (warp < 4 ? warp - 1 : warp - 5)) * 32 + lane;
           piece < S::kTileM * pieces; piece += kProducers * 32) {
        int row = piece / pieces, token = tile * S::kTileM + row;
        int dimension = (piece % pieces) * (Fp8Key ? 16 : 8);
        int d = dimension * (Fp8Key ? 1 : 2);
        const auto *src =
            (const unsigned char *)key.IndexRow(min(token, max_visible - 1), args.indexStride).data + d;
        auto *dst = sw.a_ring + stage * S::kTileBytes + ImageOffset(row, dimension, S::kTileM);
        cp_async_16(dst, src, token < max_visible ? 16 : 0);
      }
      cp_async_wait_all();
      __syncwarp();
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      if (lane == 0)
        tc_mbar_arrive(sw.full + stage);
    }
  } else if (warp == 0) {
    // MMA issuer.
    tc_mbar_wait(&sw.b_full, 0u);
    tc_fence_after();
    int i = 0;
    for (int tile = int(cta_index()); tile < tiles; tile += int(cta_count()), ++i) {
      const int s = i % S::kStages;
      tc_mbar_wait(sw.full + s, unsigned((i / S::kStages) & 1));
      tc_fence_after();
      auto tA = cute::make_tensor(
          cute::make_smem_ptr(reinterpret_cast<Element *>(sw.a_ring + s * S::kTileBytes)), a_layout);
      const cute::UMMA::SmemDescriptor da = cute::UMMA::make_umma_desc<cute::UMMA::Major::K>(tA);
      for (int p = 0; p < S::kPasses; ++p) {
        tc_mbar_wait(sw.acc_empty + p, unsigned((i & 1) ^ 1));
        tc_fence_after();
        auto tB = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element *>(
                                        sw.b_image + p * S::kPassN * Scratch::kIndexDim * (Fp8Key ? 1 : 2))),
                                    b_layout);
        const cute::UMMA::SmemDescriptor db = cute::UMMA::make_umma_desc<cute::UMMA::Major::K>(tB);
        const unsigned tmem_c = tmem_base + unsigned(p * S::kPassN);
#pragma unroll
        for (int k = 0; k < Scratch::kIndexDim / (Fp8Key ? 32 : 16); ++k) {
          cute::UMMA::SmemDescriptor dak = da;
          cute::UMMA::SmemDescriptor dbk = db;
          dak.start_address_ = static_cast<unsigned short>(
              da.start_address_ + ImageOffset(0, k * (Fp8Key ? 32 : 16), S::kTileM) / 16); // +32 B per K step
          dbk.start_address_ = static_cast<unsigned short>(
              db.start_address_ + ImageOffset(0, k * (Fp8Key ? 32 : 16), S::kPassN) / 16);
          Mma::fma(static_cast<unsigned long long>(dak), static_cast<unsigned long long>(dbk), tmem_c,
                   unsigned(k > 0), idescE);
        }
        if (lane == 0) {
          tc_umma_commit(sw.acc_full + p);
        }
      }
      if (lane == 0) {
        tc_umma_commit(sw.empty + s); // the A stage is free once every MMA of this tile completed
      }
    }
  } else if (warp >= 4) {
    // Epilogue: group p = (warp - 4) / 4 owns accumulator p; quadrant q = warp & 3 owns TMEM lanes 32q..
    const int p = (warp - 4) >> 2;
    const int q = warp & 3;
    if (p < S::kPasses) {
      int i = 0;
      for (int tile = int(cta_index()); tile < tiles; tile += int(cta_count()), ++i) {
        tc_mbar_wait(sw.acc_full + p, unsigned(i & 1));
        tc_fence_after();
        const unsigned addr = tmem_base + (unsigned(32 * q) << 16) + unsigned(p * S::kPassN);
        constexpr int kRowsPerPass = S::kPassN / Scratch::kIndexHeads;
        float row_sum[kRowsPerPass];
#pragma unroll
        for (int r = 0; r < kRowsPerPass; ++r) {
          row_sum[r] = 0.0f;
        }
#pragma unroll
        for (int chunk = 0; chunk < S::kPassN / 32; ++chunk) {
          unsigned v[32];
          cute::SM100_TMEM_LOAD_32dp32b32x::copy(
              addr + unsigned(chunk * 32), v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10],
              v[11], v[12], v[13], v[14], v[15], v[16], v[17], v[18], v[19], v[20], v[21], v[22], v[23],
              v[24], v[25], v[26], v[27], v[28], v[29], v[30], v[31]);
          tc_wait_ld();
          const float *w = sw.weights + p * S::kPassN + chunk * 32;
          constexpr int kGroup = Scratch::kIndexHeads < 32 ? Scratch::kIndexHeads : 32;
#pragma unroll
          for (int first = 0; first < 32; first += kGroup) {
            const int r = (chunk * 32 + first) / Scratch::kIndexHeads;
            float acc = 0.0f;
#pragma unroll
            for (int j = first; j < first + kGroup; ++j) {
              acc = fmaf(fmaxf(__uint_as_float(v[j]) * sw.scales[p * S::kPassN + chunk * 32 + j], 0.0f), w[j],
                         acc);
            }
            row_sum[r] += acc;
          }
        }
        tc_fence_before();
        __syncwarp();
        if (lane == 0) {
          tc_mbar_arrive(sw.acc_empty + p);
        }
        const int pos = tile * S::kTileM + 32 * q + lane;
        const float scale = pos < max_visible ? key.IndexScale(pos, args.indexStride) : 0.0f;
#pragma unroll
        for (int r = 0; r < kRowsPerPass; ++r) {
          const int query = p * kRowsPerPass + r;
          if (query < Scratch::kQueryRows && pos < lengths[query]) {
            float value = row_sum[r] * scale;
            logits[std::size_t(query) * Scratch::kMaxSeqLen + pos] = value;
            if (!PreserveScoreOrder && firstReady) {
              const unsigned digit = ordered_float_key(value) >> 24;
              atomicAdd(first_hist + query * 256 + digit, 1u);
            }
          }
        }
      }
    }
  }
  __syncthreads();
  // The sweep join closes all epilogue writes; the caller grid join publishes
  // the merged histogram together with logits before Top-K consumes either.
  if (!PreserveScoreOrder && firstReady) {
    auto *grid_base = (unsigned *)(args.workspace + Scratch::kTopKGridOffset);
    for (int i = tid; i < Scratch::kQueryRows * 256; i += kThreads) {
      unsigned count = first_hist[i];
      if (count)
        atomicAdd(topk_grid_hist(grid_base, i / 256, 0) + i % 256, count);
    }
  }
  if (warp == 0) {
    cute::TMEM::Allocator1Sm allocator;
    allocator.free(tmem_base, S::kTmemCols);
  }
}

__device__ static unsigned char *score_tile_image(SweepStorage &sw, unsigned char *fourth, int tile) {
    return tile == 3 ? fourth : sw.a_ring + ((tile + 2) % 3) * SweepStorage::kTileBytes;
}

template <class A>
__device__ static void score_tiles(const A &args, const int *lengths, SweepStorage &sw, int prepared,
                                   unsigned char *fourth, int firstTile, int tileStride, int numTiles) {
    using S = SweepStorage;
    using Element = std::conditional_t<Fp8Key, cutlass::float_e4m3_t, cutlass::bfloat16_t>;
    constexpr int bytes = sizeof(Element) * S::kTileM * Scratch::kIndexDim;
    constexpr int kStep = Fp8Key ? 32 : 16, tmemSize = 4 * S::kPassN;
    static_assert(S::kPasses == 1 && S::kPassN == 128 && S::kTileM == 128 && S::kStages == 3);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int max_visible = lengths[8];
    auto *iq = args.workspace + Scratch::kIqPackedOffset;
    auto *sc = (const float *)(args.workspace + Scratch::kIndexWeightsScaledOffset);
    auto *logits = (float *)(args.workspace + Scratch::kIndexLogitsOffset);
    __shared__ unsigned long long done[4];
    if (warp == 1 && !lane) {
        tc_mbar_init(&sw.b_full, 1);
        for (int j = 0; j < numTiles; ++j)
            tc_mbar_init(done + j, 1);
        asm volatile("fence.mbarrier_init.release.cluster; fence.proxy.async.global;" ::: "memory");
        tc_mbar_expect_tx(&sw.b_full, bytes);
        tc_bulk_g2s(sw.b_image, iq, bytes, &sw.b_full);
    }
    if (warp == 0) {
        cute::TMEM::Allocator1Sm alloc;
        alloc.allocate(tmemSize, &sw.tmem_slot);
        alloc.release_allocation_lock();
    }
    if (tid < 128) {
        sw.weights[tid] = tid < args.activeRows * 16 ? float(args.weights[tid]) : 0.f;
        sw.scales[tid] = tid < args.activeRows * 16 ? sc[tid] : 0.f;
    }
    for (int j = prepared; j < numTiles; ++j) {
        const int tile = firstTile + j * tileStride;
        auto key = args.indexKey;
        key.Bind(args.indexCache);
        constexpr int chunks = Scratch::kIndexDim * sizeof(Element) / 16;
        for (int i = tid; i < 128 * chunks; i += kThreads) {
            const int row = i / chunks, d = i % chunks * (16 / sizeof(Element)), tok = tile * 128 + row;
            auto src = (const Element *)key.IndexRow(min(tok, max_visible - 1), args.indexStride).data;
            cp_async_16(score_tile_image(sw, fourth, j) + ImageOffset(row, d, 128), src + d,
                        tok < max_visible ? 16 : 0);
        }
    }
    asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    const unsigned tmem_base = sw.tmem_slot;
    using AtomK = cute::UMMA::Layout_K_SW128_Atom<Element>;
    auto a_layout = cute::tile_to_shape(AtomK{}, cute::Shape<cute::Int<S::kTileM>, cute::Int<128 / sizeof(Element)>>{});
    auto b_layout = cute::tile_to_shape(AtomK{}, cute::Shape<cute::Int<S::kPassN>, cute::Int<128 / sizeof(Element)>>{});
    const cute::UMMA::InstrDescriptor idesc = cute::UMMA::make_instr_desc<Element, Element, float, S::kTileM, S::kPassN,
                                                                          cute::UMMA::Major::K, cute::UMMA::Major::K>();
    const unsigned long long idescE = static_cast<unsigned long long>(static_cast<unsigned>(idesc)) << 32;
    using Mma = std::conditional_t<Fp8Key, cute::SM100_MMA_F8F6F4_SS,
        cute::SM100_MMA_F16BF16_SS<Element, Element, float, S::kTileM, S::kPassN, cute::UMMA::Major::K, cute::UMMA::Major::K>>;

    if (warp == 0) {
        tc_mbar_wait(&sw.b_full, 0);
        tc_fence_after();
        for (int j = 0; j < numTiles; ++j) {
            auto tA = cute::make_tensor(
                cute::make_smem_ptr(reinterpret_cast<Element *>(score_tile_image(sw, fourth, j))), a_layout);
            const cute::UMMA::SmemDescriptor da = cute::UMMA::make_umma_desc<cute::UMMA::Major::K>(tA);
            auto tB = cute::make_tensor(cute::make_smem_ptr(reinterpret_cast<Element *>(sw.b_image)), b_layout);
            const auto db = cute::UMMA::make_umma_desc<cute::UMMA::Major::K>(tB);
            const unsigned tmem_c = tmem_base + unsigned(j * 128);
#pragma unroll
            for (int k = 0; k < Scratch::kIndexDim / kStep; ++k) {
                cute::UMMA::SmemDescriptor dak = da;
                cute::UMMA::SmemDescriptor dbk = db;
                dak.start_address_ = static_cast<unsigned short>(
                    da.start_address_ + (ImageOffset(0, k * kStep, S::kTileM)) / 16); // +32 B per K step
                dbk.start_address_ =
                    static_cast<unsigned short>(db.start_address_ + (ImageOffset(0, k * kStep, S::kPassN)) / 16);
                Mma::fma(static_cast<unsigned long long>(dak), static_cast<unsigned long long>(dbk), tmem_c,
                         unsigned(k > 0), idescE);
            }
            if (lane == 0)
                tc_umma_commit(done + j);
        }
    }
    // TMEM lanes are owned by the warp index modulo four. Other warp groups
    // reduce independent 32-column fragments or another key tile.
    for (int job = warp; job < numTiles * 16; job += kThreads / 32) {
        const int j = job / 16, q = job % 4, chunk = (job / 4) % 4;
        const int tile = firstTile + j * tileStride;
        tc_mbar_wait(done + j, 0);
        tc_fence_after();
        const unsigned addr = tmem_base + (unsigned(32 * q) << 16) + unsigned(j * 128);
        unsigned v[32];
        cute::SM100_TMEM_LOAD_32dp32b32x::copy(addr + unsigned(chunk * 32), v[0], v[1], v[2], v[3], v[4], v[5], v[6],
                                               v[7], v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15], v[16], v[17],
                                               v[18], v[19], v[20], v[21], v[22], v[23], v[24], v[25], v[26], v[27],
                                               v[28], v[29], v[30], v[31]);
        tc_wait_ld();
        const float *w = sw.weights + chunk * 32;
        constexpr int kGroup = Scratch::kIndexHeads < 32 ? Scratch::kIndexHeads : 32;
#pragma unroll
        for (int first = 0; first < 32; first += kGroup) {
            const int r = (chunk * 32 + first) / Scratch::kIndexHeads;
            float acc = 0.0f;
#pragma unroll
            for (int j = first; j < first + kGroup; ++j) {
                acc = fmaf(fmaxf(__uint_as_float(v[j]) * sw.scales[chunk * 32 + j], 0.0f), w[j], acc);
            }
            const int pos = tile * 128 + 32 * q + lane;
            if (pos < lengths[r])
                logits[size_t(r) * Scratch::kMaxSeqLen + pos] = acc + 0.f;
        }
        tc_fence_before();
    }
    __syncthreads();
    if (warp == 0) {
        cute::TMEM::Allocator1Sm alloc;
        alloc.free(tmem_base, tmemSize);
    }
}
