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
