#include "models/speculative_sampling.h"
#include <array>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace fastllm;

static void Require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

int main() {
    try {
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
        std::cout << "Speculative sampling: all tests passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
