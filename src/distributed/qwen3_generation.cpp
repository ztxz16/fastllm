#include "distributed/qwen3_generation.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace fastllm {
namespace distributed {
namespace {
/** 检查采样参数；不修改进程组或随机数状态。 */
void ValidateSampling(const Qwen3GenerationOptions &options) {
    if (!std::isfinite(options.temperature) || options.temperature < 0 ||
        !std::isfinite(options.topP) || options.topP <= 0 || options.topP > 1) {
        throw std::runtime_error("invalid Qwen3 sampling parameters");
    }
}
}

uint32_t SampleQwen3(const std::vector<float> &logits,
    const Qwen3GenerationOptions &options, double uniform) {
    ValidateSampling(options);
    if (logits.empty() || logits.size() > std::numeric_limits<uint32_t>::max() ||
        !std::isfinite(uniform) || uniform < 0 || uniform >= 1) {
        throw std::runtime_error("invalid Qwen3 sampling input");
    }
    for (float value : logits) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("nonfinite Qwen3 logits");
        }
    }
    if (options.temperature == 0) {
        return uint32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
    }

    std::vector<uint32_t> order(logits.size());
    std::iota(order.begin(), order.end(), uint32_t{0});
    auto compare = [&](uint32_t a, uint32_t b) {
        return logits[a] == logits[b] ? a < b : logits[a] > logits[b];
    };
    size_t count = options.topK ? std::min<size_t>(options.topK, order.size()) : order.size();
    std::partial_sort(order.begin(), order.begin() + count, order.end(), compare);
    order.resize(count);

    // 先减最大值再除温度，避免极低温度使正数溢出；负无穷的 exp 合法地变成零。
    std::vector<double> weights(count);
    double total = 0;
    for (size_t i = 0; i < count; ++i) {
        weights[i] = std::exp((double(logits[order[i]]) - logits[order[0]]) / options.temperature);
        total += weights[i];
    }
    if (options.topP < 1) {
        double cumulative = 0;
        size_t kept = 0;
        do {
            cumulative += weights[kept++];
        } while (kept < count && cumulative < options.topP * total);
        count = kept;
        total = cumulative;
    }

    double target = uniform * total, cumulative = 0;
    uint32_t lastPositive = order[0];
    for (size_t i = 0; i < count; ++i) {
        if (weights[i] > 0) {
            lastPositive = order[i];
        }
        cumulative += weights[i];
        if (target < cumulative) {
            return order[i];
        }
    }
    return lastPositive; // 浮点累加舍入的尾部保护，不能选到零概率 token。
}

Qwen3GenerationResult GenerateQwen3(ProcessGroup &group, const std::vector<uint32_t> &prompt,
    const Qwen3GenerationOptions &options,
    const std::function<void(uint32_t, const std::vector<float>&)> &observer) {
    ValidateSampling(options);
    if (prompt.empty() || prompt.size() > 128 || options.maxNewTokens == 0 ||
        uint64_t(prompt.size()) + options.maxNewTokens - 1 > 2048) {
        throw std::runtime_error("invalid Qwen3 generation length (prompt 1..128, capacity <=2048)");
    }
    for (const auto &state : group.Qwen3Requests()) {
        if (state.active) {
            throw std::runtime_error("Qwen3 generation requires no active request");
        }
    }

    Qwen3GenerationResult result;
    std::mt19937_64 random(options.seed);
    bool ownsRequest = false;
    try {
        auto logits = group.BeginQwen3Request(prompt, uint32_t(prompt.size() + options.maxNewTokens - 1));
        ownsRequest = true;
        for (auto eos : options.eosTokenIds) {
            if (eos >= logits.size()) {
                throw std::runtime_error("Qwen3 EOS token outside vocabulary");
            }
        }
        for (uint32_t step = 0; step < options.maxNewTokens; ++step) {
            // 显式定义 53-bit 均匀采样，避免 uniform_real_distribution 的标准库差异。
            double uniform = double(random() >> 11) * (1.0 / 9007199254740992.0);
            uint32_t token = SampleQwen3(logits, options, uniform);
            result.tokenIds.push_back(token);
            if (observer) {
                observer(token, logits);
            }
            if (std::find(options.eosTokenIds.begin(), options.eosTokenIds.end(), token) != options.eosTokenIds.end()) {
                result.finishReason = "eos";
                break;
            }
            if (step + 1 == options.maxNewTokens) {
                result.finishReason = "length";
                break;
            }
            logits = group.DecodeQwen3(token);
        }
        group.EndQwen3Request();
        ownsRequest = false;
    } catch (...) {
        auto original = std::current_exception();
        if (ownsRequest && group.State() == ProcessGroupState::Ready) {
            try {
                group.EndQwen3Request();
            } catch (...) {
                // 保留原错误；End 失败的整组状态由 ProcessGroup 记录，调用者负责 Stop。
            }
        }
        std::rethrow_exception(original);
    }
    return result;
}

} // namespace distributed
} // namespace fastllm
