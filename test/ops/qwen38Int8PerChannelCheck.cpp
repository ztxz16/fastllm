//
// End-to-end check and benchmark for a compressed-tensors W8A8/W8A16 INT8
// Qwen3.8-27B checkpoint loaded through FastLLM's HF path.
//
// Usage:
//   qwen38Int8PerChannelCheck <model_dir> [--tp N] [--tokens N]
//                             [--prompt-text TEXT] [--prefill-tokens N]
//
// The driver exercises the real model graph (dense GDN + full attention
// layers), prints the greedy continuation and reports prefill/decode timing.
// It returns 77 (test skip) when the checkpoint or CUDA devices are missing.
//
#include "fastllm.h"
#include "model.h"

// Mirrors the auto-warmup completion in basellm: NCCL launches asynchronously
// once the memory pool is warm, which is what enables communication/compute
// overlap in production.  The driver has no auto-warmup of its own, so the
// switch is flipped after the warm-up repetition.
extern "C" void FastllmCudaSetNcclForceSync(bool value);

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <iostream>
#include <string>
#include <vector>

using namespace fastllm;

namespace {

constexpr int kSkipReturnCode = 77;

int GetIntOption(int argc, char **argv, const std::string &name,
                 int fallback) {
    for (int i = 1; i + 1 < argc; i++) {
        if (std::strcmp(argv[i], name.c_str()) == 0) {
            return std::atoi(argv[i + 1]);
        }
    }
    return fallback;
}

std::string GetStringOption(int argc, char **argv, const std::string &name,
                            const std::string &fallback) {
    for (int i = 1; i + 1 < argc; i++) {
        if (std::strcmp(argv[i], name.c_str()) == 0) {
            return argv[i + 1];
        }
    }
    return fallback;
}

double NowMs() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0]
                  << " <model_dir> [--tp N] [--tokens N] [--prompt-text TEXT]"
                  << std::endl;
        return kSkipReturnCode;
    }
    const std::string path = argv[1];
    if (!FileExists(path + "/config.json")) {
        std::cout << "[skip] model directory not found: " << path << std::endl;
        return kSkipReturnCode;
    }
    const int tp = GetIntOption(argc, argv, "--tp", 2);
    const int maxTokens = GetIntOption(argc, argv, "--tokens", 48);
    const int prefillTokens = GetIntOption(argc, argv, "--prefill-tokens", 0);
    std::string promptText =
        GetStringOption(argc, argv, "--prompt-text",
                        "The capital of France is");
    const bool runSuite = GetIntOption(argc, argv, "--suite", 0) != 0;
    // Raw prompts skip the chat template: some checkpoints ship Jinja macros
    // this template engine cannot parse, and a raw prompt is the fairest
    // cross-quantization comparison.
    const bool rawPrompt = GetIntOption(argc, argv, "--raw-prompt", 0) != 0;
    // Quality suite: factual, arithmetic, multi-step reasoning, code and a
    // Chinese prompt.  Greedy decoding makes the continuations comparable
    // between quantizations.
    const std::vector<std::string> suitePrompts = {
        "The capital of France is",
        "The first president of the United States was",
        "The largest planet in our solar system is",
        "2 + 2 =",
        "If a train travels 60 miles per hour for 2.5 hours, how many miles "
        "does it travel? Answer:",
        "Write a Python function factorial(n) that returns n! :\n```python\n",
        "\u4e2d\u56fd\u7684\u9996\u90fd\u662f",
        "User: What is 17 * 24? Assistant:",
    };

    if (tp > 1) {
        // Mirrors the ftllm launcher: FASTLLM_TP selects the thread tensor
        // parallel device list, the device map pins the root device.
        std::string spec = "cuda:0";
        for (int i = 1; i < tp; i++) {
            spec += "," + std::to_string(i);
        }
        setenv("FASTLLM_TP", spec.c_str(), 1);
        SetDeviceMap(std::map<std::string, int>{{"cuda:0", 1}});
    } else {
        SetDeviceMap(std::map<std::string, int>{{"cuda:0", 1}});
    }
    SetThreads(12);

    std::cout << "Loading " << path << " (tp=" << tp << ") ..." << std::endl;
    const double loadStart = NowMs();
    auto model = CreateLLMModelFromHF(path, DataType::DATA_AUTO_SOURCE, -1,
                                      false, "", "");
    if (model == nullptr) {
        std::cerr << "failed to create model" << std::endl;
        return 1;
    }
    model->SetDataType(DataType::FLOAT16);
    std::cout << "Loaded in " << (NowMs() - loadStart) / 1000.0 << " s"
              << std::endl;

    if (GetIntOption(argc, argv, "--dump-weights", 0) != 0) {
        // Print every loaded weight so loader regressions (wrong dtype, empty
        // merged tensor, missing scale vector) are visible without running the
        // graph.
        int linearCount = 0, quantizedCount = 0, emptyCount = 0;
        for (auto &item : model->weight.weight) {
            const Data &weight = item.second;
            const bool linear =
                weight.dims.size() == 2 && weight.dims[0] > 64;
            if (linear) {
                linearCount++;
            }
            if (weight.dims.size() == 2 &&
                (weight.dataType == DataType::INT8_PERCHANNEL_S8 ||
                 weight.dataType == DataType::INT8_PERCHANNEL_S8_W8A16)) {
                quantizedCount++;
            }
            if (weight.dims.empty()) {
                emptyCount++;
                std::cout << "EMPTY  " << item.first << std::endl;
            } else if (linear && GetIntOption(argc, argv, "--dump-all", 0) != 0) {
                std::cout << "WEIGHT " << item.first << " dtype="
                          << GetDataTypeName(weight.dataType) << " dims=";
                for (int d : weight.dims) {
                    std::cout << d << ",";
                }
                std::cout << " scales=" << weight.scales.size()
                          << " perChannelAxis=" << weight.perChannelAxis
                          << std::endl;
            }
        }
        std::cout << "weights=" << model->weight.weight.size()
                  << " linear=" << linearCount
                  << " int8PerChannel=" << quantizedCount
                  << " empty=" << emptyCount << std::endl;
        return 0;
    }

    if (runSuite) {
        const int suiteTokens = GetIntOption(argc, argv, "--suite-tokens", 32);
        for (const std::string &prompt : suitePrompts) {
            ChatMessages suiteMessages;
            suiteMessages.push_back({"user", prompt});
            std::string suiteText = rawPrompt ? prompt
                                              : model->ApplyChatTemplate(suiteMessages);
            auto suiteIds = model->weight.tokenizer.Encode(suiteText);
            std::vector<int> suiteInput;
            for (int i = 0; i < suiteIds.Count(0); i++) {
                suiteInput.push_back((int)((float *)suiteIds.cpuData)[i]);
            }
            GenerationConfig suiteConfig;
            suiteConfig.do_sample = false;
            suiteConfig.top_k = 1;
            suiteConfig.output_token_limit = suiteTokens;
            const int suiteHandle =
                model->LaunchResponseTokens(suiteInput, suiteConfig);
            std::vector<int> suiteOutput;
            while ((int)suiteOutput.size() < suiteTokens) {
                const int token = model->FetchResponseTokens(suiteHandle);
                if (token < 0) {
                    break;
                }
                suiteOutput.push_back(token);
            }
            model->AbortResponse(suiteHandle);
            std::vector<float> suiteFloatIds;
            for (int token : suiteOutput) {
                suiteFloatIds.push_back((float)token);
            }
            Data suiteIdData(DataType::FLOAT32, {(int)suiteFloatIds.size()},
                             suiteFloatIds);
            std::string suiteAnswer =
                model->weight.tokenizer.Decode(suiteIdData);
            std::string escaped;
            for (char c : suiteAnswer) {
                if (c == '\n') {
                    escaped += "\\n";
                } else if (c == '\r') {
                    continue;
                } else {
                    escaped += c;
                }
            }
            std::cout << "SUITE\t" << prompt << "\t" << escaped << std::endl;
        }
        return 0;
    }

    ChatMessages messages;
    messages.push_back({"user", promptText});
    std::string prompt = rawPrompt ? promptText
                                   : model->ApplyChatTemplate(messages);
    auto inputIds = model->weight.tokenizer.Encode(prompt);
    std::vector<int> tokens;
    for (int i = 0; i < inputIds.Count(0); i++) {
        tokens.push_back((int)((float *)inputIds.cpuData)[i]);
    }

    // Long-context benchmark prompts use deterministic random token ids so a
    // repeated request cannot hit the prefix cache, and every repetition can
    // use a different prompt of the same length.
    const int prefillRepeats = GetIntOption(argc, argv, "--prefill-repeats", 1);
    std::vector<int> baseTokens = tokens;
    auto buildPrompt = [&](int repetition) {
        if (prefillTokens <= (int)baseTokens.size()) {
            return std::vector<int>(tokens.begin(), tokens.end());
        }
        std::vector<int> result = baseTokens;
        std::mt19937 rng(0x5eed + repetition * 7919);
        while ((int)result.size() < prefillTokens) {
            result.push_back((int)(rng() % 200000) + 1000);
        }
        result.resize(prefillTokens);
        return result;
    };

    std::cout << "prompt tokens: " << tokens.size() << std::endl;

    GenerationConfig config;
    config.do_sample = false;
    config.top_k = 1;
    config.output_token_limit = maxTokens;
    config.output_logits = true;

    std::vector<int> generated;
    std::vector<float> logits;
    std::vector<double> prefillMs;
    std::vector<double> decodeMs;
    for (int repetition = 0; repetition <= prefillRepeats; repetition++) {
        // The first iteration only warms the allocators, algorithm selection
        // and kernel images; the reported numbers start at the second one.
        std::vector<int> requestTokens = buildPrompt(repetition);
        const bool warm = repetition > 0;
        const double start = NowMs();
        const int handle = model->LaunchResponseTokens(requestTokens, config);
        generated.clear();
        double firstTokenMs = -1.0;
        while (true) {
            int token;
            if (generated.empty()) {
                token = model->FetchResponseLogits(handle, logits);
            } else {
                token = model->FetchResponseTokens(handle);
            }
            if (token == -1 || token == -2) {
                if (token == -2) {
                    std::cerr << "prompt too long" << std::endl;
                }
                break;
            }
            if (generated.empty()) {
                firstTokenMs = NowMs() - start;
            }
            generated.push_back(token);
            if ((int)generated.size() >= maxTokens) {
                model->AbortResponse(handle);
                break;
            }
        }
        const double totalMs = NowMs() - start;
        if (!warm) {
            // First request doubles as this driver's warmup; the server flips
            // this switch from its auto-warmup for the same reason.
            if (std::getenv("FASTLLM_TEST_KEEP_FORCE_SYNC") == nullptr) {
                FastllmCudaSetNcclForceSync(false);
            }
        }
        if (warm) {
            prefillMs.push_back(firstTokenMs);
            decodeMs.push_back(totalMs - firstTokenMs);
            std::cout << "run " << repetition << ": prefill "
                      << requestTokens.size() << " tokens in "
                      << firstTokenMs / 1000.0 << " s = "
                      << requestTokens.size() / (firstTokenMs / 1000.0)
                      << " tok/s" << std::endl;
        }
    }
    auto median = [](std::vector<double> values) {
        if (values.empty()) {
            return 0.0;
        }
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    };
    const double firstTokenMs = median(prefillMs);
    const double totalMs = firstTokenMs + median(decodeMs);
    tokens = buildPrompt(prefillRepeats > 0 ? prefillRepeats : 1);

    std::string text;
    {
        std::vector<float> ids;
        ids.reserve(generated.size());
        for (int token : generated) {
            ids.push_back((float)token);
        }
        Data idData(DataType::FLOAT32, {(int)ids.size()}, ids);
        text = model->weight.tokenizer.Decode(idData);
    }
    std::cout << "greedy continuation: " << text << std::endl;
    std::cout << "first tokens:";
    for (size_t i = 0; i < std::min<size_t>(12, generated.size()); i++) {
        std::cout << " " << generated[i];
    }
    std::cout << std::endl;
    if (!logits.empty()) {
        // Report the top-8 logits of the first generated step; this is the
        // strongest local comparison signal against another backend.
        std::vector<std::pair<float, int>> ranked;
        ranked.reserve(logits.size());
        for (int i = 0; i < (int)logits.size(); i++) {
            ranked.push_back({logits[i], i});
        }
        std::partial_sort(ranked.begin(), ranked.begin() + 8, ranked.end(),
                          std::greater<std::pair<float, int>>());
        std::cout << "top-8 logits (token:value):";
        for (int i = 0; i < 8; i++) {
            std::cout << " " << ranked[i].second << ":" << ranked[i].first;
        }
        std::cout << std::endl;
    }

    const double prefillSeconds = firstTokenMs / 1000.0;
    const double decodeSeconds = (totalMs - firstTokenMs) / 1000.0;
    std::cout << "prefill: " << tokens.size() << " tokens in "
              << prefillSeconds << " s = "
              << (prefillSeconds > 0 ? tokens.size() / prefillSeconds : 0.0)
              << " tok/s" << std::endl;
    std::cout << "decode: " << generated.size() - 1 << " tokens in "
              << decodeSeconds << " s = "
              << (decodeSeconds > 0
                      ? (generated.size() - 1) / decodeSeconds
                      : 0.0)
              << " tok/s" << std::endl;
    (void)prefillTokens;
    return 0;
}