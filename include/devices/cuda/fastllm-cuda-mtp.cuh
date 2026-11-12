#pragma once

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
