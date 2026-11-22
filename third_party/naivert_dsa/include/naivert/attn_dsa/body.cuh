#pragma once
#include "args.cuh"
#include <cooperative_groups.h>
#include <cuda/atomic>
#include <cub/block/block_radix_sort.cuh>
#include <cuda_runtime.h>
#include <mma.h>
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cutlass/float8.h>
namespace fastllm_dsa_fused {
namespace cg = cooperative_groups;
// PreserveScoreOrder keeps the established partial-fusion arithmetic. The full
// layer consumes exact Top-K in token order, allowing its faster attention path.
template <bool Fp8Key, bool PreserveScoreOrder = true,
          class ScratchLayout = AttnDsaScratchLayout, int Groups = 1> struct Body {
  using Scratch = ScratchLayout;
  __device__ static int cta_count() { return int(gridDim.x) / Groups; }
  __device__ static int cta_index() {
    if constexpr (Groups == 1) return int(blockIdx.x);
    else return int(blockIdx.x) % cta_count();
  }
  struct RequestGrid {
    unsigned *control;
    __device__ void sync() const {
      // Every CTA is resident (cooperative launch). The two words belong to
      // this request and are reset before a full-grid join at launch entry.
      __threadfence();
      __syncthreads();
      if (!threadIdx.x) {
        cuda::atomic_ref<unsigned, cuda::thread_scope_device> arrived(control[0]), epoch(control[1]);
        const unsigned generation = epoch.load(cuda::memory_order_relaxed);
        if (arrived.fetch_add(1, cuda::memory_order_acq_rel) == unsigned(cta_count() - 1)) {
          arrived.store(0, cuda::memory_order_relaxed);
          epoch.store(generation + 1, cuda::memory_order_release);
        } else {
          while (epoch.load(cuda::memory_order_acquire) == generation) __nanosleep(64);
        }
      }
      __syncthreads();
    }
  };
  __device__ static auto make_grid(unsigned char *workspace) {
    if constexpr (Groups == 1) return cg::this_grid();
    else return RequestGrid{reinterpret_cast<unsigned *>(workspace + Scratch::kBarrierOffset)};
  }
  static constexpr int kWmmaWarps = 12, kThreads = 384;
  static constexpr int kTopKSortItemsPerThread = 22;
  using TopKBlockRadixSort =
      cub::BlockRadixSort<unsigned, kThreads, kTopKSortItemsPerThread, cub::NullType, 5>;
  struct alignas(1024) SweepStorage {
    static constexpr int kCols = Scratch::kQueryRows * Scratch::kIndexHeads;
    // TMEM reads 32 columns at a time. Decode pads 16 heads to 32 columns;
    // verify packs two real query rows into each 32-column read.
    static constexpr int kPaddedCols = kCols < 32 ? 32 : kCols;
    static constexpr int kPassN = kPaddedCols > 256 ? 256 : kPaddedCols;
    static constexpr int kPasses = kPaddedCols / kPassN;
    static constexpr int kStages = 3;
    static constexpr int kTileM = 128;
    static constexpr int kTileBytes =
        kTileM * Scratch::kIndexDim * (Fp8Key ? 1 : 2); // 16 KiB FP8 or 32 KiB BF16
    static constexpr int kTmemCols = 2 * kPassN >= 32 ? 2 * kPassN : 32;
    alignas(1024) unsigned char b_image[kPaddedCols * Scratch::kIndexDim * (Fp8Key ? 1 : 2)];
    alignas(1024) unsigned char a_ring[kStages * kTileBytes];
    float weights[kPaddedCols], scales[kPaddedCols];
    unsigned long long full[kStages];
    unsigned long long empty[kStages];
    unsigned long long acc_full[kPasses];
    unsigned long long acc_empty[kPasses];
    unsigned long long b_full;
    unsigned tmem_slot;
  };
  using SelectionSort =
      cub::BlockRadixSort<unsigned, kThreads, 6, cub::NullType, (PreserveScoreOrder ? 5 : 6)>;
  using ScoreSort = cub::BlockRadixSort<unsigned, kThreads, 6, int, 6>;
  union SharedStorage {
    ScoreSort::TempStorage score_sort;
    native_attention::Storage native;
    TopKBlockRadixSort::TempStorage topk_sort;
    SweepStorage sweep;
  };
  __device__ __forceinline__ static void cp_async_16(void *destination, const void *source,
                                                     int source_bytes) {
    unsigned shared_address = static_cast<unsigned>(__cvta_generic_to_shared(destination));
    if (source_bytes == 16) {
      asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                   :
                   : "r"(shared_address), "l"(source)
                   : "memory");
    } else {
      asm volatile("cp.async.ca.shared.global [%0], [%1], 16, 0;\n"
                   :
                   : "r"(shared_address), "l"(source)
                   : "memory");
    }
  }

  __device__ static void cp_async_wait_all() {
    asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
  }
  __device__ static unsigned ordered_float_key(float v) {
    unsigned u = __float_as_uint(v);
    return u & 0x80000000u ? ~u : u ^ 0x80000000u;
  }
  __device__ static int ImageOffset(int row, int dimension, int rows) {
    if constexpr (Fp8Key)
      return row * 128 + (dimension ^ ((row & 7) * 16));
    else
      return (dimension / 64 * rows * 64 + row * 64 + ((dimension % 64) ^ ((row & 7) * 8))) * 2;
  }
#include "index_sweep_tcgen05.cuh"
#include "topk_grid_select.cuh"
#include "topk_local_select.cuh"
  template <class ArgsLike> __device__ static void initialize(const ArgsLike &args) {
    auto *temp = (float *)(args.workspace + Scratch::kAttentionTempOffset);
    auto *flags = (unsigned *)(temp + Scratch::kAttentionControlFloatOffset);
    auto *lengths = (int *)(args.workspace + Scratch::kRadixControlOffset);
    auto *topk = (unsigned *)(args.workspace + Scratch::kTopKGridOffset);
    const int t = threadIdx.x;
    if (cta_index() == 0 && t < 8) {
      const auto &r = args.rows[min(t, args.activeRows - 1)];
      int maximum = 0, minimum = 65536;
      for (int i = 0; i < args.activeRows; ++i) {
        const auto &row = args.rows[i];
        const int visible = max(0, min(row.capacity, row.liveKeys ? *row.liveKeys : row.length));
        maximum = max(maximum, visible);
        minimum = min(minimum, visible);
      }
      int length = max(0, min(r.capacity, r.liveKeys ? *r.liveKeys : r.length));
      // A dense/empty row does not need sparse selection, but the shared
      // histogram passes still need a valid rank. Its real length is restored
      // by dense identity restoration and by native attention; no future token is consumed.
      lengths[t] = length <= 2048 && maximum > 2048 ? maximum : (length ? length : maximum);
      if (!t)
        lengths[8] = maximum;
      flags[t] = flags[8 + t] = 0;
      if constexpr (!PreserveScoreOrder) {
        if (!t)
          flags[8] = minimum <= 2048;
      }
    }
    zero_topk_grid(topk);
  }
  template <bool ComputeAttention = true, class ArgsLike>
  __device__ static void run(const ArgsLike &args, SharedStorage &shared, bool releaseAllocationPermit = true,
                             bool inputsReady = false) {
    auto grid = make_grid(args.workspace);
    auto *iq = args.workspace + Scratch::kIqPackedOffset;
    auto *scales = (float *)(args.workspace + Scratch::kIndexWeightsScaledOffset);

    auto *logits = (float *)(args.workspace + Scratch::kIndexLogitsOffset);
    auto *selections = (int *)(args.workspace + Scratch::kIndexSelectsOffset);
    auto *topk = (unsigned *)(args.workspace + Scratch::kTopKGridOffset);
    auto *temp = (float *)(args.workspace + Scratch::kAttentionTempOffset);
    auto *flags = (unsigned *)(temp + Scratch::kAttentionControlFloatOffset);
    auto *lengths = (int *)(args.workspace + Scratch::kRadixControlOffset);
    const int t = threadIdx.x, warp = t / 32, lane = t % 32;
    if (!inputsReady) {
      initialize(args);
      // A half warp quantizes one 128-element head; amax is the same exact max tree.
      int vector = (cta_index() * kWmmaWarps + warp) * 2 + lane / 16;
      if (vector < 8 * 16) {
        int row = vector / 16, head = vector % 16, d = lane % 16 * 8;
        const auto *src = args.indexQuery + (min(row, args.activeRows - 1) * 16 + head) * 128 + d;
        float v[8], maximum = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          v[j] = (float)src[j];
          maximum = fmaxf(maximum, fabsf(v[j]));
        }
#pragma unroll
        for (int step = 8; step; step >>= 1)
          maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, step));
        float scale = fmaxf(maximum, 1e-4f) / 448.f;
        int dst = ImageOffset(vector, d, 8 * 16);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          auto quantized = __nv_fp8_e4m3(fmaxf(-448.f, fminf(448.f, v[j] / scale)));
          if constexpr (Fp8Key)
            iq[dst + j] = quantized.__x;
          else
            ((__nv_bfloat16 *)(iq + dst))[j] = __float2bfloat16((float)quantized);
        }
        if (!(lane & 15))
          scales[vector] = scale;
      }
      grid.sync();
    }
    if (lengths[8] > 0) {
      // With at most one A tile per CTA, ring stage 1 is unused. Borrow it
      // for the first histogram only on the grid Top-K path. Other lengths
      // retain the original sweep/select without allocating additional shared memory.
      const bool firstReady = !PreserveScoreOrder && lengths[8] > kTopKLocalMaxVisible &&
                              lengths[8] <= int(cta_count()) * SweepStorage::kTileM;
      compute_index_logits_tcgen05(args, iq, scales, logits, lengths, shared.sweep, releaseAllocationPermit,
                                   firstReady);
      grid.sync();
      select_per_query_topk_grid(logits, selections, topk, lengths, (unsigned *)&shared, &shared.topk_sort,
                                 firstReady);
    } else {
      for (int i = cta_index() * kThreads + t; i < 8 * 2048; i += cta_count() * kThreads)
        selections[i] = i % 2048;
      grid.sync();
    }
    if constexpr (PreserveScoreOrder) {
      // Preserve score order and the original 2048-slot attention partitions.
      // Selected tokens start in ascending order; stable 32-bit radix sort retains
      // token-ascending ties, including zero-score rows and the dense/sparse boundary.
      if (cta_index() < args.activeRows) {
        const int query = cta_index();
        const auto &descriptor = args.rows[query];
        const int visible =
            max(0, min(descriptor.capacity, descriptor.liveKeys ? *descriptor.liveKeys : descriptor.length));
        const int count = min(visible, 2048);
        unsigned keys[6], minimum = 0xffffffffu, maximum = 0;
        int positions[6];
#pragma unroll
        for (int j = 0; j < 6; ++j) {
          int slot = t * 6 + j;
          int token = slot < count ? (visible <= 2048 ? slot : selections[query * 2048 + slot]) : -1;
          positions[j] = token;
          keys[j] = token >= 0 ? ordered_float_key(logits[size_t(query) * Scratch::kMaxSeqLen + token]) : 0;
          if (token >= 0) {
            minimum = min(minimum, keys[j]);
            maximum = max(maximum, keys[j]);
          }
        }
#pragma unroll
        for (int step = 16; step; step >>= 1) {
          minimum = min(minimum, __shfl_xor_sync(0xffffffff, minimum, step));
          maximum = max(maximum, __shfl_xor_sync(0xffffffff, maximum, step));
        }
        unsigned *range = (unsigned *)&shared;
        if (!lane) {
          range[warp] = minimum;
          range[12 + warp] = maximum;
        }
        __syncthreads();
        if (!t) {
          for (int w = 1; w < 12; ++w) {
            range[0] = min(range[0], range[w]);
            range[12] = max(range[12], range[12 + w]);
          }
        }
        __syncthreads();
        minimum = count ? range[0] : 0;
        int bits = count ? max(1, 32 - __clz(range[12] - minimum)) : 1;
#pragma unroll
        for (int j = 0; j < 6; ++j)
          keys[j] = positions[j] >= 0 ? keys[j] - minimum : 0;
        __syncthreads();
        ScoreSort(shared.score_sort).SortDescending(keys, positions, 0, bits);
#pragma unroll
        for (int j = 0; j < 6; ++j)
          if (t * 6 + j < 2048)
            selections[query * 2048 + t * 6 + j] = positions[j];
      }
      grid.sync();
    }
    if constexpr (!PreserveScoreOrder) {
      // Grid selection pads short rows to a valid sparse rank when their later
      // rows cross Top-K. Restore dense identities without a score sort.
      // Long-context launches avoid this extra cooperative barrier entirely.
      if (flags[8]) {
        if (cta_index() < args.activeRows) {
          const auto &row = args.rows[cta_index()];
          const int visible = max(0, min(row.capacity, row.liveKeys ? *row.liveKeys : row.length));
          if (visible <= 2048)
            for (int i = t; i < 2048; i += kThreads)
              selections[cta_index() * 2048 + i] = i;
        }
        grid.sync();
      }
    }
    // Expose the exact selected tokens to existing diagnostics and callers.
    for (int i = cta_index() * kThreads + t; i < args.activeRows * 2048; i += cta_count() * kThreads)
      if (args.rows[i / 2048].indices) {
        const auto &row = args.rows[i / 2048];
        ((int *)row.indices)[i % 2048] = (row.liveKeys ? *row.liveKeys : row.length) > 0 ? selections[i] : -1;
      }
    // Native attention consumes the same selections in the same kernel.
    if constexpr (ComputeAttention) if (cta_index() < args.activeRows * 16) {
      const int row = cta_index() / 16;
      auto descriptor = args.rows[row];
      if (descriptor.indices)
        descriptor.indices = selections + row * 2048;
      native_attention::Attend<!PreserveScoreOrder>(args.query, descriptor, args.sink, args.output, temp,
                                                    flags, 8, 1, row, 0, cta_index() % 16, shared.native);
    }
  }
};
} // namespace fastllm_dsa_fused
