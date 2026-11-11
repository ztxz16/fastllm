#ifndef FASTLLM_CUDA_TOKEN_CONSTRAINT_H
#define FASTLLM_CUDA_TOKEN_CONSTRAINT_H
#include "token_constraint.h"
#include <memory>

// Prepare runs outside capture; all other operations are allocation-free and
// capture-safe. Keep the batch alive and finish its stream before preparing a
// new batch. Null programs represent unrestricted rows. IDs are float because
// the embedding/sampling graph uses the exact integer range below 2^24.
std::shared_ptr<void> FastllmCudaCreateTokenConstraintBatch(int capacity, int steps);
void FastllmCudaPrepareTokenConstraintBatch(void *batch,
    const std::vector<std::shared_ptr<const fastllm::TokenConstraintProgram>> &programs,
    const std::vector<int> &states, int vocab);
void FastllmCudaTokenConstraintMask(void *batch, int step, uint32_t *mask);
// Invalid branches use a safe embedding ID until the final step reports -1.
void FastllmCudaTokenConstraintAdvance(void *batch, int step, float *tokens, bool finalStep = false);
void FastllmCudaTokenConstraintApply(void *batch, float *logits, int steps);

#endif
