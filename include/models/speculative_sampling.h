#pragma once

#include "fastllm.h"
#include "utils/utils.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace fastllm {

// Candidates are the globally sorted raw-score TopK after all masks/penalties.
// Keep the dense probability vector for the existing rejection/residual logic.
template <class Token, class Score>
inline std::vector<float> SpeculativeTopKDistribution(
        int count, int vocab, const GenerationConfig &config, Token token, Score score) {
    std::vector<float> probabilities(vocab, 0.0f);
    if (count == 1) {
        probabilities[token(0)] = 1;
        return probabilities;
    }
    double sum = 0;
    for (int i = 0; i < count; ++i) {
        float p = std::exp((score(i) - score(0)) / config.temperature);
        probabilities[token(i)] = p;
        sum += p;
    }
    double kept = 0;
    int retained = 0;
    for (; retained < count; ++retained) {
        kept += probabilities[token(retained)];
        if (kept / sum > config.top_p) { ++retained; break; }
    }
    for (int i = 0; i < retained; ++i) probabilities[token(i)] /= kept;
    for (int i = retained; i < count; ++i) probabilities[token(i)] = 0;
    return probabilities;
}

inline std::vector<float> SpeculativeTopKDistribution(
        const float *candidates, int count, int vocab, const GenerationConfig &config) {
    return SpeculativeTopKDistribution(count, vocab, config,
        [&](int i) { return (int)candidates[i * 2]; },
        [&](int i) { return candidates[i * 2 + 1]; });
}

// The same temperature, repetition, top-k and top-p order as LLMSampling.
// Keep the complete, normalized proposal distribution: rejection sampling
// needs q for every token, including tokens the draft did not select.
inline std::vector<float> SpeculativeDistribution(
        const float *logits, int vocab, const GenerationConfig &config,
        const LastTokensUnit &tokens) {
    std::vector<float> values(logits, logits + vocab);
    if (std::abs(config.repeat_penalty - 1.0f) > 1e-6f) {
        int previous = -1;
        for (int token : tokens.tokenSet) {
            if (config.last_n <= 0 && token == previous) continue;
            previous = token;
            if (token >= 0 && token < vocab)
                values[token] = values[token] < 0 ? values[token] * config.repeat_penalty :
                                                   values[token] / config.repeat_penalty;
        }
    }
    std::vector<int> order;
    order.reserve(vocab);
    for (int token = 0; token < vocab; ++token) {
        if (config.tool_call_allowed_token_ids.empty() || std::binary_search(
                config.tool_call_allowed_token_ids.begin(), config.tool_call_allowed_token_ids.end(), token))
            order.push_back(token);
    }
    AssertInFastLLM(!order.empty(), "Empty speculative sampling support.");
    int count = std::min<int>(std::max(1, config.top_k), order.size());
    if (config.temperature <= 0) count = 1;
    std::partial_sort(order.begin(), order.begin() + count, order.end(), [&](int a, int b) {
        return values[a] > values[b] || (values[a] == values[b] && a < b);
    });
    return SpeculativeTopKDistribution(count, vocab, config,
        [&](int i) { return order[i]; }, [&](int i) { return values[order[i]]; });
}

inline int SampleSpeculativeDistribution(const std::vector<float> &probabilities, double uniform) {
    double sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
    AssertInFastLLM(sum > 0 && std::isfinite(sum), "Invalid speculative sampling distribution.");
    double cumulative = 0, threshold = uniform * sum;
    int last = -1;
    for (int token = 0; token < (int)probabilities.size(); ++token) {
        if (probabilities[token] > 0) last = token;
        cumulative += probabilities[token];
        if (cumulative > threshold) return token;
    }
    return last;
}

// Standard speculative rejection sampling (Leviathan et al., Algorithm 1).
// A rejection draws from normalized max(p - q, 0), never from p directly.
inline bool AcceptSpeculativeToken(int token, const std::vector<float> &p,
                                   const std::vector<float> &q, double uniform) {
    AssertInFastLLM(p.size() == q.size() && token >= 0 && token < (int)q.size() && q[token] > 0,
                    "Invalid speculative proposal.");
    return uniform * q[token] < p[token];
}

inline int SampleSpeculativeResidual(const std::vector<float> &p,
                                     const std::vector<float> &q, double uniform) {
    AssertInFastLLM(p.size() == q.size(), "Speculative vocabulary mismatch.");
    std::vector<float> residual(p.size());
    for (size_t i = 0; i < p.size(); ++i) residual[i] = std::max(0.0f, p[i] - q[i]);
    return SampleSpeculativeDistribution(residual, uniform);
}

}
