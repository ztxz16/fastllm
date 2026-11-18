#ifndef FASTLLM_CUDA_TOKEN_CONSTRAINT_H
#define FASTLLM_CUDA_TOKEN_CONSTRAINT_H
#include "token_constraint.h"
#include <memory>

// Batch configuration runs outside capture. Finish the stream before using
// the synchronous prepare API again. Mask and advance are capture-safe.
// Null programs represent unrestricted rows. IDs are float because
// the embedding/sampling graph uses the exact integer range below 2^24.
std::shared_ptr<void> FastllmCudaCreateTokenConstraintBatch(int capacity, int steps);
void FastllmCudaPrepareTokenConstraintBatch(void *batch,
    const std::vector<std::shared_ptr<const fastllm::TokenConstraintProgram>> &programs,
    const std::vector<int> &states, int vocab);
// Decode tickets own the pinned upload buffers until their sampler readback completes.
void FastllmCudaInitTokenConstraintUploads(void *batch);
void FastllmCudaUpdateTokenConstraintRows(void *batch, int slot,
    const std::vector<std::shared_ptr<const fastllm::TokenConstraintProgram>> &programs,
    const std::vector<int> &states, const std::vector<char> &retain, int vocab);
// Release program references displaced by this upload after its stream completes.
void FastllmCudaReleaseTokenConstraintUpload(void *batch, int slot);
void FastllmCudaTokenConstraintMask(void *batch, int step, uint32_t *mask);
// Invalid branches use a safe embedding ID until the final step reports -1.
void FastllmCudaTokenConstraintAdvance(void *batch, int step, float *tokens, bool finalStep = false);
void FastllmCudaTokenConstraintApply(void *batch, float *logits, int steps);
// Continuous decode uses one private cursor per row, including captured
// padded batches. Invalid IDs remain visible to the host, but embeddings get 0.
void FastllmCudaTokenConstraintApplyRows(void *batch, float *logits, int rows);
void FastllmCudaTokenConstraintAdvanceInPlace(void *batch, int *tokens,
                                           float *embeddingTokens, int rows);

#endif
