#include "models/qwen3_5.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "utils/utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fastllm;

namespace {
void Require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

// A zero-layer target makes the real speculative ForwardGPU path produce a
// known distribution, without loading a checkpoint or mocking CUDA sampling.
class SamplingModel : public Qwen3_5Model {
public:
    SamplingModel() {
        block_cnt = 0;
        embed_dim = 128;
        dataType = FLOAT16;
        num_experts = 0;
        num_attention_heads = num_key_value_heads = 2;
        head_dim = head_k_dim = head_v_dim = 128;
        num_k_heads = num_v_heads = 2;
        deviceMap = {{"cuda:0", 1}};
        rms_norm_eps = 1e-6f;
        speculativeCollectAllLogits = true;
        speculativeCaptureFirstTokenLinearState = true;

        const std::string embedding = language_prefix + "embed_tokens.weight";
        weight.AddEmptyWeight(embedding, {128, embed_dim}, FLOAT16);
        weight[embedding].Allocate();
        std::fill_n(reinterpret_cast<uint16_t *>(weight[embedding].cpuData),
                    128 * embed_dim, float_to_half(1.0f));
        const std::string norm = language_prefix + "norm.weight";
        weight.AddEmptyWeight(norm, {embed_dim}, FLOAT32);
        weight[norm].Allocate();
        // Qwen3.5 adds one to its stored norm weights on the first forward.
        std::fill_n(reinterpret_cast<float *>(weight[norm].cpuData), embed_dim, 0.0f);
        weight.AddEmptyWeight("lm_head.weight", {128, embed_dim}, FLOAT16);
        weight["lm_head.weight"].Allocate();
        auto *head = reinterpret_cast<uint16_t *>(weight["lm_head.weight"].cpuData);
        const float probabilities[] = {0.6f, 0.3f, 0.1f};
        for (int token = 0; token < 128; ++token) {
            float logit = token < 3 ? std::log(probabilities[token]) : -100.0f;
            std::fill_n(head + token * embed_dim, embed_dim,
                        float_to_half(logit / embed_dim));
        }
    }

    const std::vector<unsigned char> &Accepted() const {
        return speculativeMtpAccepted;
    }
};

struct Scenario {
    int length, draft, topK;
    float topP, temperature;
    std::array<double, 3> expected;
};

void Run(const std::vector<Scenario> &scenarios, const char *label) {
    SamplingModel model;
    const int batch = scenarios.size();
    std::vector<int> lengths;
    std::vector<float> tokens;
    std::vector<Data> positions;
    std::vector<GenerationConfig> configs;
    positions.reserve(batch);
    for (const auto &scenario : scenarios) {
        lengths.push_back(scenario.length);
        for (int j = 0; j < scenario.length; ++j) {
            tokens.push_back(float(scenario.draft < 0 ? j % 4 : scenario.draft));
        }
        std::vector<float> values(scenario.length);
        std::iota(values.begin(), values.end(), 0.0f);
        positions.emplace_back(FLOAT32, std::vector<int>{1, scenario.length}, values);
        GenerationConfig config;
        config.top_k = scenario.topK;
        config.top_p = scenario.topP;
        config.temperature = scenario.temperature;
        config.repeat_penalty = 1.0f;
        configs.push_back(config);
    }
    std::vector<Data *> positionPtrs;
    for (auto &position : positions) positionPtrs.push_back(&position);
    Data input(FLOAT32, {1, int(tokens.size())}, tokens);
    std::vector<Data *> masks(batch, nullptr);
    std::vector<std::pair<Data *, Data *>> caches;
    LastTokensManager lastTokens;
    // Count proposal rows separately: legacy code only overwrote these rows,
    // while bonus rows already retained their random target samples.
    std::vector<std::array<long long, 3>> counts(batch);
    std::vector<long long> totals(batch, 0);
    const int rounds = 64;
    for (int iteration = 0; iteration < rounds; ++iteration) {
        const auto sampled = model.ForwardGPU(batch, input, masks, positionPtrs,
                                              lengths, caches, configs, lastTokens);
        Require(sampled.size() == tokens.size(), "speculative target row count changed");
        const auto &accepted = model.Accepted();
        const bool needsFlags = std::any_of(configs.begin(), configs.end(),
            [](const GenerationConfig &config) { return !config.IsSimpleGreedy(); });
        if (needsFlags) Require(accepted.size() == sampled.size(), "acceptance row count changed");
        int offset = 0;
        for (int b = 0; b < batch; ++b) {
            const auto &scenario = scenarios[b];
            for (int j = 0; j < scenario.length; ++j) {
                const int row = offset + j, token = sampled[row];
                Require(token >= 0 && token < 3, "sample escaped the filtered distribution");
                const bool candidate = j + 1 < scenario.length && !configs[b].IsSimpleGreedy();
                if (needsFlags) {
                    Require(bool(accepted[row]) == (candidate && token == int(tokens[row + 1])),
                            "MTP acceptance does not match the target sample/draft row");
                }
                if (scenario.length == 1 || j + 1 < scenario.length) {
                    counts[b][token]++;
                    totals[b]++;
                }
                if (scenario.expected[token] == 0.0) {
                    throw std::runtime_error("MTP emitted a zero-probability token");
                }
            }
            offset += scenario.length;
        }
    }
    for (int b = 0; b < batch; ++b) {
        for (int token = 0; token < 3; ++token) {
            const double p = scenarios[b].expected[token];
            const double frequency = double(counts[b][token]) / totals[b];
            const double tolerance = 0.005 + 7.0 * std::sqrt(p * (1.0 - p) / totals[b]);
            Require(std::abs(frequency - p) <= tolerance, "MTP changed the target sampling distribution");
        }
        std::cout << label << " request=" << b << " samples=" << totals[b]
                  << " counts=" << counts[b][0] << ',' << counts[b][1] << ',' << counts[b][2] << '\n';
    }
}
} // namespace

int main(int argc, char **argv) {
    if (FastllmCudaGetDeviceCount() < 1) return 77;
    try {
        const bool batched = argc == 2 && std::string(argv[1]) == "--batch";
        const std::vector<Scenario> scenarios = {
            {513, 1, 3, 1.0f, 1.0f, {0.6, 0.3, 0.1}},
            {257, 3, 3, 1.0f, 1.0f, {0.6, 0.3, 0.1}},
            {385, 1, 2, 0.75f, 1.0f, {2.0 / 3.0, 1.0 / 3.0, 0.0}},
            {129, 1, 1, 1.0f, 0.0f, {1.0, 0.0, 0.0}},
            {513, 1, 3, 1.0f, 0.5f, {36.0 / 46.0, 9.0 / 46.0, 1.0 / 46.0}},
            {65, -1, 3, 1.0f, 1.0f, {0.6, 0.3, 0.1}},
            {1, 1, 3, 1.0f, 1.0f, {0.6, 0.3, 0.1}}
        };
        if (batched) {
            Run(scenarios, "batch");
        } else {
            for (size_t i = 0; i + 1 < scenarios.size(); ++i) Run({scenarios[i]}, "single");
        }
        std::cout << "PASS: Qwen3.5 MTP target sampling and acceptance\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
