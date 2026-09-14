#pragma once

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
