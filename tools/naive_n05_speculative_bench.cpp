#include "model.h"
#include "models/naive_n05_flash.h"
#include "json11.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>

using namespace fastllm;
using Json = json11::Json;

// Run paired measurements with the same loaded weights and device placement.
// Access remains in this validation executable; serving has no benchmark flags.
class NaiveBenchmarkAccess : public NaiveN05FlashModel {
public:
    static Json CheckHistory(NaiveN05FlashModel &model, const std::vector<int> &prompt) {
        model.*(&NaiveBenchmarkAccess::draftEnabled) = true;
        model.*(&NaiveBenchmarkAccess::draftTokens) = model.*(&NaiveBenchmarkAccess::draftBlock);
        model.*(&NaiveBenchmarkAccess::draftConfidenceThreshold) = .5f;
        GenerationConfig config;
        config.input_token_length = prompt.size();
        config.output_token_limit = 24;
        auto request = [&](const std::vector<int> &ids) {
            int handle = model.LaunchResponseTokens(ids, config);
            int cached = 0, missed = 0, output = 0;
            model.GetResponseStatistics(handle, cached, missed, output);
            std::vector<int> tokens;
            for (;;) {
                int token = model.FetchResponseTokens(handle);
                if (token < 0) break;
                tokens.push_back(token);
            }
            return std::make_pair(tokens, cached);
        };
        model.SetSaveHistoryChat(false);
        auto fresh = request(prompt);
        model.SetSaveHistoryChat(true);
        auto first = request(prompt);
        auto cached = request(prompt);
        model.SetSaveHistoryChat(false);
        return Json::object{{"fresh_ids", Json(fresh.first)}, {"recorded_ids", Json(first.first)},
            {"cached_ids", Json(cached.first)}, {"cached_tokens", cached.second},
            {"equal", fresh.first == first.first && first.first == cached.first}};
    }
    static Json CompareLogits(NaiveN05FlashModel &model, const std::vector<int> &prompt,
                              const std::vector<int> &continuation) {
        std::vector<std::pair<Data, Data>> sequential(model.block_cnt), batched(model.block_cnt);
        auto forward = [&](const std::vector<int> &tokens, int start,
                           std::vector<std::pair<Data, Data>> &kv, TargetCapture *capture) {
            std::vector<float> values(tokens.begin(), tokens.end()), positions;
            for (int i = 0; i < (int)tokens.size(); ++i) positions.push_back(start + i);
            Data ids(FLOAT32, {1, (int)tokens.size()}, values), pos(FLOAT32, {1, (int)tokens.size()}, positions);
            return (model.*(&NaiveBenchmarkAccess::RunTarget))(ids, pos, kv, capture);
        };
        forward(prompt, 0, sequential, nullptr);
        for (int i = 0; i < model.block_cnt; ++i) {
            batched[i].first.CopyFrom(sequential[i].first);
            batched[i].second.CopyFrom(sequential[i].second);
        }
        TargetCapture capture;
        capture.verifying = true;
        Data batch = forward(continuation, prompt.size(), batched, &capture);
        batch.ToDevice(DataDevice::CPU);
        int vocab = batch.dims.back(), topMatches = 0;
        double maxKL = 0, maxAbs = 0;
        Json::array metrics;
        for (int row = 0; row < (int)continuation.size(); ++row) {
            Data single = forward({continuation[row]}, prompt.size() + row, sequential, nullptr);
            single.ToDevice(DataDevice::CPU);
            const float *a = (float *)single.cpuData;
            const float *b = (float *)batch.cpuData + (size_t)row * vocab;
            int topA = std::max_element(a, a + vocab) - a, topB = std::max_element(b, b + vocab) - b;
            topMatches += topA == topB;
            double sumA = 0, sumB = 0, abs = 0;
            for (int token = 0; token < vocab; ++token) {
                sumA += std::exp(a[token] - a[topA]);
                sumB += std::exp(b[token] - b[topB]);
                abs = std::max(abs, (double)std::abs(a[token] - b[token]));
            }
            double kl = 0;
            for (int token = 0; token < vocab; ++token) {
                double logP = a[token] - a[topA] - std::log(sumA);
                double logQ = b[token] - b[topB] - std::log(sumB);
                kl += std::exp(logP) * (logP - logQ);
            }
            maxKL = std::max(maxKL, kl); maxAbs = std::max(maxAbs, abs);
            metrics.push_back(Json::object{{"kl", kl}, {"max_abs", abs}, {"top1_sequential", topA}, {"top1_batched", topB}});
        }
        return Json::object{{"input_tokens", (int)prompt.size()}, {"verify_tokens", (int)continuation.size()},
            {"max_kl", maxKL}, {"max_abs", maxAbs}, {"top1_matches", topMatches}, {"positions", metrics}};
    }
    static Json Run(NaiveN05FlashModel &model, const std::vector<int> &prompt,
                    const GenerationConfig &config, int candidates, float confidence) {
        AssertInFastLLM(candidates >= 0 && candidates <= model.*(&NaiveBenchmarkAccess::draftBlock),
                        "Benchmark draft_tokens must be in [0, block_size].");
        AssertInFastLLM(confidence >= 0 && confidence <= 1, "Invalid benchmark confidence threshold.");
        model.*(&NaiveBenchmarkAccess::draftEnabled) = candidates > 0;
        model.*(&NaiveBenchmarkAccess::draftTokens) = candidates;
        model.*(&NaiveBenchmarkAccess::draftConfidenceThreshold) = confidence;
        std::vector<std::pair<Data, Data>> kv(model.block_cnt);
        LastTokensManager history(1, config.last_n);
        for (int token : prompt) history.units[0].Push(token);
        std::vector<int> generated;
        Json::array times;
        auto started = std::chrono::steady_clock::now();
        auto seconds = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(); };
        for (int step = 0; step < config.output_token_limit; ++step) {
            std::vector<float> values, positions;
            if (step == 0) {
                for (int token : prompt) values.push_back(token);
                for (int i = 0; i < (int)prompt.size(); ++i) positions.push_back(i);
            } else {
                values.push_back(generated.back());
                positions.push_back(prompt.size() + step - 1);
            }
            Data ids(FLOAT32, {1, (int)values.size()}, values);
            Data pos(FLOAT32, {1, (int)values.size()}, positions);
            int token = model.Forward(ids, Data(), pos, kv, config, history);
            times.push_back(seconds());
            generated.push_back(token);
            history.units[0].Push(token);
            if (token == model.eos_token_id || model.eos_token_ids.count(token) || config.stop_token_ids.count(token)) break;
        }
        Json::object result{{"draft_tokens", candidates}, {"confidence_threshold", confidence},
            {"input_tokens", (int)prompt.size()}, {"output_tokens", (int)generated.size()},
            {"output_ids", Json(generated)}, {"token_seconds", times},
            {"ttft_seconds", times.front()},
            {"decode_tokens_per_second", times.size() > 1 ?
                (times.size() - 1) / (times.back().number_value() - times.front().number_value()) : 0}};
        auto &contexts = model.*(&NaiveBenchmarkAccess::draftContexts);
        auto it = contexts.find(&kv);
        if (it != contexts.end()) {
            const auto &c = *it->second;
            result["rounds"] = (double)c.rounds;
            result["proposed"] = (double)c.proposed;
            result["accepted"] = (double)c.accepted;
            result["acceptance"] = c.proposed ? (double)c.accepted / c.proposed : 0;
            result["tokens_per_round"] = c.rounds ? 1.0 + (double)c.accepted / c.rounds : 1;
            contexts.erase(it);
        }
        return result;
    }
};

int main(int argc, char **argv) {
    if (argc != 5) {
        std::cerr << "Usage: naive_n05_speculative_bench TARGET DRAFT CASES_JSON OUTPUT_JSON\n";
        return 2;
    }
    std::ifstream input(argv[3]);
    std::string text((std::istreambuf_iterator<char>(input)), {}), error;
    auto cases = Json::parse(text, error);
    AssertInFastLLM(error.empty() && cases.is_array() && !cases.array_items().empty(), "Invalid benchmark cases JSON.");
    SetDeviceMap({{"cuda:0", 1}});
    SetMoeDeviceMap({{"numa", 1}});
    setenv("FASTLLM_DSPARK_MODEL_PATH", argv[2], 1);
    auto base = CreateLLMModelFromHF(argv[1], FP8_E4M3);
    auto *model = dynamic_cast<NaiveN05FlashModel *>(base.get());
    AssertInFastLLM(model != nullptr, "Expected Naive-N0.5 model.");
    model->WarmUp();
    Json::array results;
    auto record = [&](Json::object measured, const Json &test, const char *label) {
        measured["name"] = test["name"];
        results.push_back(measured);
        std::ofstream output(argv[4]);
        output << Json(results).dump() << '\n';
        AssertInFastLLM(output.good(), "Cannot write benchmark results.");
        std::cout << label << " " << test["name"].string_value() << std::endl;
    };
    for (const auto &test : cases.array_items()) {
        std::vector<int> prompt;
        for (const auto &token : test["input_ids"].array_items()) prompt.push_back(token.int_value());
        AssertInFastLLM(!prompt.empty(), "Benchmark prompt must not be empty.");
        if (test["repeat_to"].int_value() > 0) {
            auto original = prompt;
            prompt.resize(test["repeat_to"].int_value());
            for (int i = 0; i < (int)prompt.size(); ++i) prompt[i] = original[i % original.size()];
        }
        if (test["cache_check"].bool_value()) {
            record(NaiveBenchmarkAccess::CheckHistory(*model, prompt).object_items(), test, "CACHE_RESULT");
            continue;
        }
        if (!test["verify_ids"].is_null()) {
            std::vector<int> tokens;
            for (auto &token : test["verify_ids"].array_items()) tokens.push_back(token.int_value());
            AssertInFastLLM(!tokens.empty(), "Benchmark verification tokens must not be empty.");
            record(NaiveBenchmarkAccess::CompareLogits(*model, prompt, tokens).object_items(), test, "LOGITS_RESULT");
            continue;
        }
        GenerationConfig config;
        config.input_token_length = prompt.size();
        config.output_token_limit = test["output_tokens"].int_value();
        AssertInFastLLM(config.output_token_limit > 1, "Benchmark needs at least two output tokens.");
        config.top_k = test["top_k"].is_null() ? 1 : test["top_k"].int_value();
        config.top_p = test["top_p"].is_null() ? 1 : test["top_p"].number_value();
        config.temperature = test["temperature"].is_null() ? 1 : test["temperature"].number_value();
        config.do_sample = config.top_k > 1;
        auto measured = NaiveBenchmarkAccess::Run(*model, prompt, config,
            test["draft_tokens"].int_value(), test["confidence_threshold"].is_null() ? .5 :
            test["confidence_threshold"].number_value()).object_items();
        measured["top_k"] = config.top_k;
        measured["top_p"] = config.top_p;
        measured["temperature"] = config.temperature;
        record(std::move(measured), test, "BENCH_RESULT");
    }
}
