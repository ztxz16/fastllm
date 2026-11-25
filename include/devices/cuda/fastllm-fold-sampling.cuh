#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <functional>

// Aliases into the resident proposal workspace, valid across graph replays.
struct FastllmFoldSelection {
    static constexpr int maxCandidates = 128;
    unsigned *radix;
    int radixWords;
    int *count;
    int blocks;
};
bool FastllmCudaMtpFoldSelection(void *workspace, int sms, FastllmFoldSelection &selection);
bool FastllmCudaMtpFoldFallback(const int *complete, const std::function<bool(cudaStream_t)> &body);
bool FastllmCudaMtpRepairBf16Fold(void *workspace, const void *base, const void *adjust, const void *bias,
        float cap, const float *t, int *k, const float *p, uint64_t *seed, float *q, int *token, int vocab,
        cudaStream_t stream);
bool FastllmCudaMtpDrawBf16Fold(void *workspace, const void *base, const void *adjust, const void *bias,
        const FastllmFoldSelection &selection, float cap, const float *temperature, const int *k, const float *p,
        const uint64_t *seed, float *q, int *token, int *complete, bool first);
