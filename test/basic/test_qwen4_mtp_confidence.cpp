#include "models/qwen4_tp_sampling.h"

#include <iostream>
#include <random>
#include <stdexcept>

static double TopProbability(const std::vector<float> &logits) {
    const double maximum = *std::max_element(logits.begin(), logits.end());
    double sum = 0;
    for (float value : logits) sum += std::exp(value - maximum);
    return 1 / sum;
}

int main() {
    std::mt19937 random(726);
    std::uniform_real_distribution<float> distribution(-12, 12);
    int checks = 0;
    for (int count : {1, 2, 3, 8}) {
        for (float offset : {-1000.0f, 0.0f, 1000.0f}) {
            for (int pattern = 0; pattern < 40; ++pattern) {
                std::vector<std::pair<float, float>> shards;
                std::vector<float> full;
                for (int rank = 0; rank < count; ++rank) {
                    // Unequal shards, ties, and peaked/diffuse distributions.
                    std::vector<float> logits(1 + random() % 251);
                    for (float &value : logits)
                        value = offset + (pattern == 0 ? 0 : distribution(random));
                    shards.push_back({*std::max_element(logits.begin(), logits.end()),
                                      (float)TopProbability(logits)});
                    full.insert(full.end(), logits.begin(), logits.end());
                }
                const double expected = TopProbability(full);
                const double actual = fastllm::qwen4_tp::MergeTopProbability(shards);
                if (std::abs(actual - expected) > 2e-7)
                    throw std::runtime_error("Sharded confidence differs from full-vocabulary softmax");
                ++checks;
            }
        }
    }
    const float infinity = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (const auto &invalid : std::vector<std::vector<std::pair<float, float>>>{
             {}, {{-infinity, nan}}, {{infinity, 1}}, {{nan, 1}}, {{2, 0}}, {{2, nan}}, {{2, 1.1f}}}) {
        if (fastllm::qwen4_tp::MergeTopProbability(invalid) != 0)
            throw std::runtime_error("Invalid confidence must not admit a draft");
    }
    if (fastllm::qwen4_tp::MergeTopProbability({{2, .5f}, {-infinity, nan}}) != .5f)
        throw std::runtime_error("Masked vocabulary shard changes confidence");
    std::cout << "PASS: " << checks << " full-vocabulary comparisons, invalid and masked shards\n";
}
