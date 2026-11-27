// SM90 EP8 verification with 64 or 128 source rows and one persistent CTA per SM.
#pragma once
#include <flashinfer/megamoe/sm90_mxfp4_mega_moe_fused.cuh>

namespace flashinfer::megamoe {

template <uint32_t kNumSMs, uint32_t kHidden, uint32_t kIntermediateHidden,
          uint32_t kNumMaxTokensPerRank, uint32_t kNumMaxPoolTokens, uint32_t kNumPaddedSFPoolTokens,
          class Prologue>
CUTLASS_GLOBAL __launch_bounds__(384, 1) void sm90_mxfp4_ep8_verify(
    void* y, int* cumulative_local_expert_recv_stats, uint32_t num_tokens,
    const __grid_constant__ layout::SymBuffer<8> sym_buffer,
    const __grid_constant__ cute::TmaDescriptor wideA1, const __grid_constant__ cute::TmaDescriptor wideS1,
    const __grid_constant__ cute::TmaDescriptor wideOut, const __grid_constant__ cute::TmaDescriptor wideA2,
    const __grid_constant__ cute::TmaDescriptor wideS2, const uint8_t* __restrict__ l1_weights,
    const uint8_t* __restrict__ l2_weights, const float* __restrict__ l1_global_scales,
    const float* __restrict__ l2_global_scales, const __grid_constant__ typename Prologue::Argument prologue) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000
  constexpr uint32_t kNumRanks = 8, kNumExperts = 256, kNumTopk = 8, kNumExpertsPerWave = 32;
  constexpr uint32_t BLOCK_N = 256, kLocalMaxRows = 0;
  constexpr bool kFastMath = true, kUseMode2RowDecoder = true, kUseInterleavedScheduler = true;
  constexpr bool kPhaseStamps = false, kGroupLocalRows = false, kLanePackedWeights = false;
  constexpr bool kSingleActiveDispatchWarp = false, kRoutesPublished = true;
  constexpr float kActivationClamp = cute::numeric_limits<float>::infinity();
  constexpr unsigned threads = 384, publishWarps = 4, dispatchThreads = 64;
  constexpr unsigned dispatchBarrier = 0, tileChoiceBarrier = 15;
  const unsigned lane = threadIdx.x % 32;
  const auto buffers = layout::MegaMoEBuffers(sym_buffer.get_base_ptr(), kNumRanks, kNumExperts,
      kNumMaxTokensPerRank, kNumTopk, kHidden, kIntermediateHidden, kNumMaxPoolTokens, kNumPaddedSFPoolTokens);
  const auto workspace = buffers.workspace;
  prologue.template prepareSource<threads>(buffers);
  if (threadIdx.x < publishWarps * 32)
    publish_ep_routes<publishWarps, kNumSMs, kNumExperts>(prologue, workspace, sym_buffer,
                                                        blockIdx.x, threadIdx.x / 32, lane);
  ptx::sync_aligned(threads, tileChoiceBarrier);
  if (threadIdx.x < dispatchThreads) {
    comm::nvlink_barrier<kNumRanks, kNumSMs, dispatchThreads, 0, 1>(workspace, sym_buffer,
        blockIdx.x, threadIdx.x, []() { ptx::sync_aligned(dispatchThreads, dispatchBarrier); }, true, true);
  }
  ptx::sync_aligned(threads, tileChoiceBarrier);

  // Each lane reads one local expert's final count, including all eight source ranks.
  // A 64-row tile saves weight reads when the wider plan removes at least a third of the tiles.
  uint64_t count;
  do {
    count = ptx::ld_volatile(workspace.get_expert_recv_count_sum_ptr(lane));
  } while (static_cast<uint32_t>(count >> 32) != kNumSMs * kNumRanks);
  const uint32_t rows = static_cast<uint32_t>(count);
  const uint32_t tiles32 = __reduce_add_sync(0xffffffffu, (rows + 31) / 32);
  const uint32_t tiles64 = __reduce_add_sync(0xffffffffu, (rows + 63) / 64);
  const auto runTile = [&]<uint32_t BLOCK_M, uint32_t kNumStages, bool kSwapABRequested, bool kUseRSOperand>(
      const cute::TmaDescriptor& tensor_map_l1_acts, const cute::TmaDescriptor& tensor_map_l1_acts_sf,
      const cute::TmaDescriptor& tensor_map_l1_output, const cute::TmaDescriptor& tensor_map_l2_acts,
      const cute::TmaDescriptor& tensor_map_l2_acts_sf) {
    static_assert(sm90_mxfp4_num_threads(kSwapABRequested, kUseRSOperand, BLOCK_M) == threads);
#include <flashinfer/megamoe/sm90_mxfp4_mega_moe_fused_body.inl>
  };
  if (tiles32 * 2 < tiles64 * 3) {
    const auto& maps = prologue.narrowMaps;
    runTile.template operator()<32, 6, true, true>(maps.a1, maps.s1, maps.out, maps.a2, maps.s2);
  } else {
    runTile.template operator()<64, 3, false, false>(wideA1, wideS1, wideOut, wideA2, wideS2);
  }
#endif
}

}  // namespace flashinfer::megamoe
