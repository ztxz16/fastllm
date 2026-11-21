#pragma once
#include <cstdint>
#include <cstddef>

// Persist q directly into per-request chain storage. A zero stride selects
// contiguous vocabulary rows. Sampling uses identical probabilities and RNG.
bool FastllmCudaMtpDraftSamplerRunBatchStrided(void *sampler, float *logits,
    float *proposalProbs, float *deviceTokens, uint64_t offset, int probabilityStride);

// Pack owner-local, step-major sampler outputs into request-major graph input.
bool FastllmCudaMtpGatherVerifyIds(float *output, int width, const float *const *rows,
    const int *strides, const int *lengths, int count, int vocab);

// Reuse the host mirror and event only for sampler graphs of the same shape.
bool FastllmCudaMtpReadDraftIdsAsync(const float *ids, std::size_t count, float *&host, void *&ready);

// Publish the anchors, temperatures and a fresh seed on the proposal graph's stream.
// Draft q is the full temperature-softmax; top-k/top-p are applied by the verifier.
bool FastllmCudaMtpPrepareDraftReplay(void *sampler, float *deviceAnchors, int rows,
    const float *anchors, const float *temperatures);

// Fuse BF16 base+bias (with BF16 rounding) and temperature softmax for large
// unfiltered draft batches. Unsupported cases return false before writing.
bool FastllmCudaMtpDraftSamplerRunBF16BiasBatch(void *sampler, const void *base, const void *bias,
    const uint32_t *mask, int step, int steps, float *proposalProbs,
    float *deviceTokens, uint64_t offset, int probabilityStride);

// Host descriptors; pointed-to storage remains on the current CUDA device.
// Batch APIs stage only these descriptors, never the full-vocabulary caches.
struct FastllmMtpProposalView {
    const float *logits;
    const float *logsumexp;
    const int *tokens;
};
struct FastllmMtpDraftOutput {
    float *logits;
    float *logsumexp;
    int *token;
    float *floatToken;
};

// Proposal logits are cached AFTER temperature, before normalization. All
// outputs stay on the current CUDA stream; the caller owns their lifetime.
// Warm the maximum batch/vocabulary before capturing this primitive in a graph;
// captured graphs require its per-worker scratch addresses to remain stable.
bool FastllmCudaMtpSampleDraftLogits(const float *logits, float *proposalLogits,
    float *proposalLogsumexp, int *output, float *floatOutput,
    const float *temperatures, int batch, int vocab);

// Target probabilities have already received the ordinary sampling constraints.
// Only the first rejection (or bonus) is sampled. Outputs and prefix lengths
// are returned to the host together after GPU verification finishes.
bool FastllmCudaMtpRejectionFromProbs(const float *targetProbs,
    const float *proposalLogits, const float *proposalLogsumexp,
    const int *deviceDraftTokens, int *output, int *accepted,
    int batch, int drafts, int vocab);
// Retry only a rejected row with an empty FP32 residual. Reuse its output slot
// and acceptance RNG (the completed small-sampler counter, or the FlashInfer seed).
bool FastllmCudaMtpRecoverRoundedResidual(const float *p, const float *q, const int *ids,
    int *deviceOutput, int &output, int row, int step, int drafts, int vocab,
    unsigned long long seed, const unsigned long long *counter);

// Same rejection law when the proposal is already stored as probabilities.
// Parallelizes only the first residual/bonus row over vocabulary tiles.
bool FastllmCudaMtpRejectionFromProbabilities(const float *targetProbs,
    const float *proposalProbs, const int *deviceDraftTokens,
    int *output, int *accepted, int batch, int drafts, int vocab);

bool FastllmCudaMtpRejectionSamplingLogits(float *targetLogits,
    const float *proposalLogits, const float *proposalLogsumexp,
    const int *deviceDraftTokens, const float *temperatures,
    const int *topKs, const float *topPs, int *output, int *accepted,
    int batch, int drafts, int vocab);

bool FastllmCudaMtpSampleDraftLogitsBatch(const float *logits,
    const FastllmMtpDraftOutput *outputs, const float *temperatures,
    int *hostTokens, int batch, int vocab);
bool FastllmCudaMtpRejectionFromProbsBatch(const float *targetProbs,
    const FastllmMtpProposalView *proposals, int *output, int *accepted,
    int batch, int drafts, int vocab);
bool FastllmCudaMtpRejectionSamplingLogitsBatch(float *targetLogits,
    const FastllmMtpProposalView *proposals, const float *temperatures,
    const int *topKs, const float *topPs, int *output, int *accepted,
    int batch, int drafts, int vocab);

// Request-owned target sampling, with stable normalization/scratch storage.
// Enqueue borrows logits/proposals until Resolve and may replay a CUDA graph.
// Resolve is the only host-result fence. A plan must not have overlapping calls;
// distinct requests own distinct plans, including across scheduler workers.
void *FastllmCudaMtpCreateVerifier();
void FastllmCudaMtpDestroyVerifier(void *plan);
bool FastllmCudaMtpVerifierEnqueue(void *plan, float *targetLogits,
    const FastllmMtpProposalView *proposals, const float *temperatures,
    const int *topKs, const float *topPs, int batch, int drafts, int vocab,
    bool useGraph);
// Same persistent plan with exact, already-normalized proposal probabilities.
// views.logits points to q; views.logsumexp is unused. Device token arrays and
// all q rows remain borrowed until Resolve. No draft-count specialization limit.
bool FastllmCudaMtpVerifierEnqueueProbabilities(void *plan, float *targetLogits,
    const FastllmMtpProposalView *proposals, const float *temperatures,
    const int *topKs, const float *topPs, int batch, int drafts, int vocab,
    bool useGraph);
bool FastllmCudaMtpVerifierResolve(void *plan, int *output, int *accepted);

// Shared implementation between the normalization and rejection CUDA units.
void *FastllmCudaMtpCreateTargetDistribution(int rows, int vocab);
void FastllmCudaMtpDestroyTargetDistribution(void *distribution);
bool FastllmCudaMtpPrepareTargetDistribution(void *distribution,
    const float *temperatures, const int *topKs, const float *topPs, bool *changed);
bool FastllmCudaMtpNormalizeTargetDistribution(void *distribution,
    float *logits, float **probabilities);

// Retarget only the input-reading normalization nodes; all scratch stays owned.
bool FastllmCudaMtpRebindTargetDistribution(void *graph, void *exec, float *logits);

constexpr int kFastllmMtpSmallRejectionMaxBatch = 8;
inline bool FastllmMtpUsesSmallRejection(int batch, int draftTokens, int vocabSize) {
    return batch <= kFastllmMtpSmallRejectionMaxBatch && draftTokens <= 8 && vocabSize >= 16384;
}

// Submit a same-stream consumer of the accepted prefix before the regular CPU
// result copy. The callback borrows the device pointer only for this submission.
using FastllmMtpAcceptedCallback = void (*)(void *, const int *);
bool FastllmCudaMtpRejectionWithAccepted(float *logits, float *proposalProbs,
    const float *temperatures, const int *topKs, const float *topPs, const int *draftTokenIds,
    int *output, int *acceptedDraftTokens, int batch, int draftTokens, int vocabSize,
    FastllmMtpAcceptedCallback callback, void *context);
