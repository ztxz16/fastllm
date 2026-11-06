#include "models/speculative_sampling.h"
#include <array>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace fastllm;

static void Require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

static std::vector<float> ReferenceDistribution(
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

static void CheckFullVocabularySampling() {
    std::mt19937 random(73);
    std::normal_distribution<float> normal(0.0f, 4.0f);
    size_t checked = 0;
    for (int vocab : {31, 1023, 1024, 1025, 4096, 152576}) {
        for (int pattern = 0; pattern < 7; ++pattern) {
            std::vector<float> logits(vocab);
            for (int token = 0; token < vocab; ++token) {
                float value = normal(random);
                if (pattern == 1) value = token % 2 ? -0.0f : 0.0f;
                if (pattern == 2) value = std::round(value * 4.0f) * .25f;
                if (pattern == 3) value = (float)token / vocab * 100.0f - 50.0f;
                if (pattern == 4) value = token % 3 == 0 ? 1e30f : -1e30f;
                logits[token] = value;
            }
            if (pattern == 6) logits[vocab / 2] = -std::numeric_limits<float>::infinity();
            std::vector<int> order(vocab), expected(vocab);
            std::iota(order.begin(), order.end(), 0);
            expected = order;
            bool sorted = SortSpeculativeVocabulary(logits, order);
            Require(sorted == (vocab >= 1024 && pattern != 6),
                    "Unexpected radix eligibility/fallback");
            if (sorted) {
                std::sort(expected.begin(), expected.end(), [&](int a, int b) {
                    return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
                });
                Require(order == expected, "Radix sort changed score/tie ordering");
            } else {
                Require(order == expected, "Fallback modified the input order");
            }
            for (float topP : {.01f, .5f, .95f, 1.0f}) {
                GenerationConfig config;
                config.do_sample = true;
                config.top_k = vocab;
                config.top_p = topP;
                config.temperature = pattern % 2 ? .7f : 1.0f;
                LastTokensUnit history(64);
                if (pattern == 5) {
                    config.repeat_penalty = 1.2f;
                    config.last_n = topP < .9f ? 0 : 64;
                    history.Push(0); history.Push(0); history.Push(vocab - 1);
                    for (int token = 0; token < vocab; token += 2)
                        config.tool_call_allowed_token_ids.push_back(token);
                }
                auto reference = ReferenceDistribution(logits.data(), vocab, config, history);
                auto actual = SpeculativeDistribution(logits.data(), vocab, config, history);
                Require(std::memcmp(reference.data(), actual.data(), vocab * sizeof(float)) == 0,
                        "Full-vocabulary probabilities changed");
                for (double uniform : {0.0, .125, .5, .731, std::nextafter(1.0, 0.0)})
                    Require(SampleSpeculativeDistribution(actual, uniform) ==
                            SampleSpeculativeDistribution(reference, uniform),
                            "Full-vocabulary sampled token changed");
                checked += vocab;
            }
        }
    }
    std::cout << "Full-vocabulary exact probability checks: " << checked << "\n";
}

int main() {
    try {
        CheckFullVocabularySampling();
        std::mt19937_64 random(42);
        auto uniform = [&]() { return std::generate_canonical<double, 53>(random); };
        // Includes disjoint support, identical distributions, greedy and a
        // proposal that strongly disagrees with the target. Checking marginal
        // output AND acceptance detects both naive matching and wrong residuals.
        std::vector<std::pair<std::vector<float>, std::vector<float>>> cases = {
            {{.6f, .3f, .1f}, {.1f, .3f, .6f}},
            {{.2f, .8f, 0}, {0, 0, 1}},
            {{.2f, .3f, .5f}, {.2f, .3f, .5f}},
            {{1, 0, 0}, {0, 1, 0}},
            {{0, 1, 0}, {0, 1, 0}},
        };
        for (auto &test : cases) {
            const auto &p = test.first, &q = test.second;
            constexpr int rounds = 200000;
            std::array<int, 3> counts{};
            int accepted = 0;
            for (int i = 0; i < rounds; ++i) {
                int token = SampleSpeculativeDistribution(q, uniform());
                if (AcceptSpeculativeToken(token, p, q, uniform())) ++accepted;
                else token = SampleSpeculativeResidual(p, q, uniform());
                ++counts[token];
            }
            double overlap = 0;
            for (int token = 0; token < 3; ++token) {
                overlap += std::min(p[token], q[token]);
                Require(std::abs((double)counts[token] / rounds - p[token]) < .006,
                        "Rejection sampling changed the target distribution");
            }
            Require(std::abs((double)accepted / rounds - overlap) < .006,
                    "Incorrect acceptance rate");
        }
        GenerationConfig config;
        config.top_k = 3;
        config.top_p = .7;
        float logits[] = {std::log(.6f), std::log(.3f), std::log(.1f)};
        auto p = SpeculativeDistribution(logits, 3, config, LastTokensUnit());
        Require(std::abs(p[0] - 2.f / 3) < 1e-6 && std::abs(p[1] - 1.f / 3) < 1e-6 && p[2] == 0,
                "Incorrect top-p normalization");
        config.top_p = 1;
        config.top_k = 2;
        config.temperature = .5;
        p = SpeculativeDistribution(logits, 3, config, LastTokensUnit());
        Require(std::abs(p[0] - .8f) < 1e-6 && std::abs(p[1] - .2f) < 1e-6 && p[2] == 0,
                "Incorrect temperature/top-k distribution");
        config.tool_call_allowed_token_ids = {2};
        p = SpeculativeDistribution(logits, 3, config, LastTokensUnit());
        Require(p[2] == 1, "Token constraints were lost");
        config.tool_call_allowed_token_ids.clear();
        config.top_k = 1;
        config.repeat_penalty = 2;
        LastTokensUnit history(8);
        history.Push(0); history.Push(0);
        float repeated[] = {4, 1.5, -1};
        p = SpeculativeDistribution(repeated, 3, config, history);
        Require(p[1] == 1, "Repeated-token multiplicity was lost");
        config.last_n = 0;
        p = SpeculativeDistribution(repeated, 3, config, history);
        Require(p[0] == 1, "Unique-token repetition policy was lost");
        // Compact candidates are already ranked raw scores, including any
        // mask/penalty. Their IDs can be sparse in the original vocabulary.
        config.repeat_penalty = 1;
        config.top_k = 3;
        config.top_p = .7f;
        config.temperature = 1;
        const float candidates[] = {7, std::log(.6f), 2, std::log(.3f), 9, std::log(.1f)};
        p = SpeculativeTopKDistribution(candidates, 3, 11, config);
        Require(std::abs(p[7] - 2.f / 3) < 1e-6 && std::abs(p[2] - 1.f / 3) < 1e-6 && p[9] == 0,
                "Compact candidates changed top-p normalization");
        for (int i : {0, 1, 3, 4, 5, 6, 8, 10})
            Require(p[i] == 0, "Compact distribution leaked outside its support");
        config.top_p = 1;
        config.top_k = 2;
        config.temperature = .5f;
        p = SpeculativeTopKDistribution(candidates, 2, 11, config);
        Require(std::abs(p[7] - .8f) < 1e-6 && std::abs(p[2] - .2f) < 1e-6,
                "Compact candidates changed temperature/top-k normalization");
        p = SpeculativeTopKDistribution(candidates, 1, 11, config);
        Require(p[7] == 1, "Single compact candidate is not a point mass");
        std::cout << "Speculative sampling: all tests passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
