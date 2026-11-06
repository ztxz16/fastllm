#ifndef FASTLLM_CLOVER_SAMPLING_H
#define FASTLLM_CLOVER_SAMPLING_H
#include "fastllm.h"
#include <random>

namespace fastllm {
struct CloverOptions {
    bool speculative = false;
    int width = 32;
    float threshold = .95f;
    bool exact = false;
    float acceptRatio = 1.f, acceptDelta = .6f;
    float presence = .9f, frequency = 0.f;
    float lengthPenalty = 10.f, thinkBias = 10.f;
    int lengthStart = 8192, lengthWindow = 8192;
    int thinkStart = 4096, thinkWindow = 4096, thinkToken = 248069;
    bool ignoreEos = false;
    uint64_t seed = 0;
    bool hasSeed = false;
};

// Position-dependent stop/thinking biases shared by the native scheduler and
// equation tests. Counts exclude the prompt and uncommitted candidates.
// Speculative penalties follow the official Clover SGLang worker;
// low-confidence follows HF generate().
std::vector<float> CloverRowBias(int rows, int vocab, const std::vector<int> &counts,
    int responseLength, const std::vector<int> &visible, const CloverOptions &options,
    bool preview, bool initial = false);
int CloverSample(const float *logits, int vocab, const GenerationConfig &config,
    std::mt19937_64 &rng, float *confidence = nullptr);
}
#endif
