#pragma once
#include "devices/cuda/naive-n05-dsa.h"
#include "devices/cuda/naive-n05-kv.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstddef>
#include <cstdint>
namespace fastllm_dsa_fused {
struct Args {
  const __nv_bfloat16 *query, *indexQuery, *weights;
  const float *sink;
  FastllmNaiveDSARow rows[8];
  FastllmNaiveKV indexKey;
  const void *const *indexCache;
  int indexStride, activeRows;
  __nv_bfloat16 *output;
  unsigned char *workspace;
};
constexpr std::size_t Align(std::size_t n) {
  return (n + 255u) & ~std::size_t(255u);
}
struct AttnDsaScratchLayout {
  static constexpr int kQueryRows = 8, kIndexHeads = 16, kIndexDim = 128;
  static constexpr int kIndexTopK = 2048, kMaxSeqLen = 65536;
  // Separate completion/control counters from the attention partials by one aligned region.
  // Preserve the established workspace offsets for both fusion variants.
  static constexpr int kAttentionControlFloatOffset = 8 * (16 * 8 * 128 + 16 * 8 * 2) + 64;
  static constexpr std::size_t kIqPackedOffset = 0;
  static constexpr std::size_t kIndexWeightsScaledOffset = 8 * 16 * 128 * 2;
  static constexpr std::size_t kIndexLogitsOffset = Align(kIndexWeightsScaledOffset + 8 * 16 * sizeof(float));
  static constexpr std::size_t kIndexSelectsOffset =
      Align(kIndexLogitsOffset + 8 * kMaxSeqLen * sizeof(float));
  static constexpr std::size_t kRadixControlOffset = Align(kIndexSelectsOffset + 8 * 2048 * sizeof(int));
  static constexpr std::size_t kAttentionTempOffset = Align(kRadixControlOffset + 9 * sizeof(int));
  static constexpr std::size_t kTopKGridOffset =
      Align(kAttentionTempOffset + (kAttentionControlFloatOffset + 16) * sizeof(float));
  static constexpr std::size_t kTopKGridBytes =
      std::size_t(kQueryRows) * (4u * 256u + 4u * 4096u + 2u + 2048u + 2u * 8192u) * sizeof(unsigned);
  static constexpr std::size_t kBytes = kTopKGridOffset + kTopKGridBytes;
};
} // namespace fastllm_dsa_fused
