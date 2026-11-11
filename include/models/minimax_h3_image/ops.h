#pragma once
#include "models/hy_nomas/runtime.h"

namespace fastllm::nomas { struct ParallelContext; }

namespace fastllm::minimax_h3_image {
// Native checkpoint packs [gate, value]. Round SiLU before multiplying the value.
nomas::Tensor gatedSilu(const nomas::Tensor &);
// In-place BF16 head RMSNorm and optional RoPE on [token, Q/K/V, head, 128].
// V is untouched; all eager rounding boundaries are retained.
void qkvNormRotaryInPlace(nomas::Tensor &, const nomas::Tensor &qWeight, const nomas::Tensor &kWeight,
                          const nomas::Tensor *cos = nullptr, const nomas::Tensor *sin = nullptr);
// Batched image VAE: 16 heads of width 64, 32 rotary dimensions; FP32 RMS accumulation.
void vaeQkvNormRotaryInPlace(nomas::Tensor &, const nomas::Tensor &cos, const nomas::Tensor &sin);
// Image DiT's 5376-wide BF16 path. The residual and each affine operation
// retain BF16 rounding; RMS accumulation remains FP32.
nomas::Tensor rmsModulate(const nomas::Tensor &, const nomas::Tensor &weight, const nomas::Tensor &table,
                          const nomas::Tensor &indices, int shift, int scale);
std::pair<nomas::Tensor, nomas::Tensor>
residualRmsModulate(const nomas::Tensor &, const nomas::Tensor &other, const nomas::Tensor &weight,
                    const nomas::Tensor &table, const nomas::Tensor &indices, int gate, int shift, int scale);
// Local packed queries attend all gathered K/V rows without materializing Q,
// K or V separately. BF16, 128-wide heads, non-causal H3 softmax semantics.
// Normalize Q in place and pack normalized K plus unchanged V by head group.
nomas::Tensor prepareSequenceQkv(nomas::Tensor &, const nomas::Tensor &qWeight, const nomas::Tensor &kWeight,
                                 const nomas::Tensor &cos, const nomas::Tensor &sin, int groups,
                                 int gatheredRows = 0, int rowOffset = 0);
nomas::Tensor sequenceAttention(const nomas::Tensor &qkv, const nomas::Tensor &kv, int heads,
                                int firstHead = 0, int headCount = 0);
void copyAttentionHeads(const nomas::Tensor &, nomas::Tensor &, int firstHead);

// Independent head groups may execute on two persistent streams. All storage
// is allocated on the caller stream and joined back before it can be reused.
class SequenceAttentionPipeline {
  public:
    SequenceAttentionPipeline();
    ~SequenceAttentionPipeline();
    SequenceAttentionPipeline(const SequenceAttentionPipeline &) = delete;
    SequenceAttentionPipeline &operator=(const SequenceAttentionPipeline &) = delete;
    int groups(int rows, int ranks) const;
    // Free per-request scratch before the VAE chooses its tile batch size.
    void release();
    nomas::Tensor forward(const nomas::Tensor &qkv, const nomas::Tensor &packedKv,
                          int totalRows, int groups, const nomas::ParallelContext &);
  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Patch rows contain [channel, 16, 16]. Neighbours are the original tiles,
// so corner blending has the same vertical-then-horizontal order as decode().
struct ImageTile {
    int height, width, y, x, overlapY, overlapX, copyHeight, copyWidth;
};
// rgba is FP32 storage used as four packed uint8 channels per pixel; invalid is
// one zero-initialized int in FP32 storage. Both are consumed only by the caller.
void stitchImageTile(const nomas::Tensor &, const nomas::Tensor *above, const nomas::Tensor *left,
                     const ImageTile &, nomas::Tensor &rgba, nomas::Tensor &invalid);
} // namespace fastllm::minimax_h3_image
