// Native implementation of the published Clover sampling equations (Apache-2.0):
// Diffuspace/Clover-35B-A3B-Preview/sampling_clover_moe.py and
// WilliamZR/sglang feature/clover-support, clover_penalty.py / clover_utils.py.
#include "clover_sampling.h"
#include <limits>
#include <numeric>
#include <stdexcept>

namespace fastllm {
namespace {
int Argmax(const float *scores, int vocab) {
    int best = 0;
    for (int i = 0; i < vocab; ++i) {
        if (std::isnan(scores[i])) throw std::runtime_error("Clover: NaN logits");
        if (scores[i] > scores[best]) best = i;
    }
    if (!std::isfinite(scores[best])) throw std::runtime_error("Clover: no finite sampling candidate");
    return best;
}
float Ramp(int position, int start, int window, float strength) {
    return std::min(1.f, std::max(0.f, float(position - start) / window)) * strength;
}
}
std::vector<float> CloverRowBias(int rows, int vocab, const std::vector<int> &counts,
        int responseLength, const std::vector<int> &visible, const CloverOptions &o,
        bool preview, bool initial) {
    const bool thinkSeen = o.thinkToken >= 0 && o.thinkToken < vocab && counts[o.thinkToken] > 0;
    const bool visibleThink = std::find(visible.begin(), visible.end(), o.thinkToken) != visible.end();
    const int firstPosition = responseLength + (o.speculative && preview ? 0 : 1);
    const int firstThinkRow = o.speculative ? std::max(int(preview), o.thinkStart + 1 - firstPosition) : 0;
    std::vector<float> bias(rows * 2, 0.f);
    for (int row = 0; row < rows; ++row) {
        int position = firstPosition + row;
        if (!initial && o.lengthPenalty)
            bias[row * 2] = Ramp(position, o.lengthStart, o.lengthWindow, o.lengthPenalty);
        if (!initial && o.thinkBias && !thinkSeen && o.thinkToken >= 0 && o.thinkToken < vocab &&
            ((!o.speculative && !visibleThink) ||
             (o.speculative && row == firstThinkRow && (!preview || !visibleThink))))
            bias[row * 2 + 1] = Ramp(position, o.thinkStart, o.thinkWindow, o.thinkBias);
    }
    return bias;
}
int CloverSample(const float *logits, int vocab, const GenerationConfig &config,
        std::mt19937_64 &rng, float *confidence) {
    const int best = Argmax(logits, vocab);
    int result = best;
    if (config.do_sample) {
        std::vector<float> scaled(vocab);
        std::vector<int> order(vocab);
        std::iota(order.begin(), order.end(), 0);
        for (int i = 0; i < vocab; ++i) scaled[i] = logits[i] / config.temperature;
        // Top-k retains ties at its cutoff, matching torch.topk + masked_fill.
        if (config.top_k > 0 && config.top_k < vocab) {
            std::vector<float> sorted = scaled;
            std::nth_element(sorted.begin(), sorted.begin() + config.top_k - 1, sorted.end(), std::greater<float>());
            const float cutoff = sorted[config.top_k - 1];
            for (auto &v : scaled) if (v < cutoff) v = -std::numeric_limits<float>::infinity();
        }
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return scaled[a] > scaled[b]; });
        const float maximum = scaled[order[0]];
        std::vector<double> mass(vocab);
        double total = 0.;
        for (int i = 0; i < vocab; ++i) { mass[i] = std::exp(scaled[order[i]] - maximum); total += mass[i]; }
        double cumulative = 0., kept = 0.;
        for (int i = 0; i < vocab; ++i) {
            // HF removes a token when its preceding probability mass is > p.
            // Keep the boundary token, including exact cutoff ties.
            if (config.top_p < 1.f && cumulative / total > config.top_p) mass[i] = 0.;
            cumulative += std::exp(scaled[order[i]] - maximum);
            kept += mass[i];
        }
        double draw = std::generate_canonical<double, 53>(rng) * kept;
        result = order[0];
        for (int i = 0; i < vocab; ++i) {
            if (mass[i] == 0.) continue;
            result = order[i]; draw -= mass[i];
            if (draw < 0.) break;
        }
    }
    if (confidence) {
        double total = 0.;
        for (int i = 0; i < vocab; ++i) total += std::exp(logits[i] - logits[best]);
        *confidence = float(std::exp(logits[result] - logits[best]) / total);
    }
    return result;
}
}
