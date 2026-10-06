#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include "models/naive_n05_flash.h"
#include "models/speculative_sampling.h"
#include "utils/utils.h"
#include <cmath>
#include <cuda_runtime_api.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
#include <atomic>
#include <dlfcn.h>
static std::atomic<int> verifyGraphLaunches{0}, verifyGraphCaptures{0};
static bool failVerifyBegin = false, failVerifyInstantiate = false, expectVerifyGraph = true;
static int failGraphDevice = 3;
extern "C" bool FastllmCudaGraphLaunch(void *exec) {
    ++verifyGraphLaunches;
    static auto fn = (bool (*)(void *))dlsym(RTLD_NEXT, "FastllmCudaGraphLaunch");
    return fn(exec);
}
extern "C" bool FastllmCudaGraphBeginCapture() {
    ++verifyGraphCaptures;
    int device = 0;
    cudaGetDevice(&device);
    if (failVerifyBegin && device == failGraphDevice)
        return false;
    static auto fn = (bool (*)())dlsym(RTLD_NEXT, "FastllmCudaGraphBeginCapture");
    return fn();
}
extern "C" bool FastllmCudaGraphInstantiate(void *graph, void **exec) {
    int device = 0;
    cudaGetDevice(&device);
    if (failVerifyInstantiate && device == failGraphDevice) {
        *exec = nullptr;
        return false;
    }
    static auto fn = (bool (*)(void *, void **))dlsym(RTLD_NEXT, "FastllmCudaGraphInstantiate");
    return fn(graph, exec);
}
#endif
using namespace fastllm;
// Cross-algorithm BF16 logits can differ because GEMM and single-row GEMV
// use different FP32 reduction orders. Keep the existing model-smoke relative
// error/cosine limits, and bound isolated outliers too. Metadata and same-path
// rollback/replay comparisons below remain exact.
struct LogitError {
    double relative = 0, cosine = 1, maximum = 0, scale = 0;
    bool acceptable = true;
};
static LogitError CompareLogits(const float *actual, const float *expected, size_t count) {
    LogitError result;
    double error = 0, aa = 0, bb = 0, ab = 0;
    for (size_t i = 0; i < count; ++i) {
        double a = actual[i], b = expected[i];
        if (!std::isfinite(a) || !std::isfinite(b)) {
            result.acceptable = false;
            return result;
        }
        error += (a - b) * (a - b);
        aa += a * a; bb += b * b; ab += a * b;
        result.maximum = std::max(result.maximum, std::abs(a - b));
        result.scale = std::max(result.scale, std::abs(b));
    }
    result.relative = std::sqrt(error / std::max(bb, 1e-30));
    result.cosine = aa > 0 && bb > 0 ? ab / std::sqrt(aa * bb) : (aa == bb ? 1 : 0);
    result.acceptable = count > 0 && result.relative <= .02 && result.cosine >= .999 &&
                        result.maximum <= 1e-6 + .02 * result.scale;
    return result;
}
static double largestLogitRelative = 0, largestLogitAbsolute = 0;
static void RequireCloseLogits(const float *actual, const std::vector<float> &expected,
                               const std::string &context) {
    auto error = CompareLogits(actual, expected.data(), expected.size());
    largestLogitRelative = std::max(largestLogitRelative, error.relative);
    largestLogitAbsolute = std::max(largestLogitAbsolute, error.maximum);
    if (!error.acceptable) {
        std::cerr << context << " relative_rmse=" << error.relative
                  << " cosine=" << error.cosine << " max_abs=" << error.maximum
                  << " reference_max=" << error.scale << std::endl;
        throw std::runtime_error(context);
    }
}
static void CheckLogitComparison() {
    const float reference[] = {1, -2, 3, -4};
    float actual[] = {1.001f, -2.002f, 3.003f, -4.004f};
    if (!CompareLogits(actual, reference, 4).acceptable)
        throw std::runtime_error("small BF16 error rejected");
    actual[0] = 2;
    if (CompareLogits(actual, reference, 4).acceptable)
        throw std::runtime_error("large logit error accepted");
    actual[0] = std::numeric_limits<float>::quiet_NaN();
    if (CompareLogits(actual, reference, 4).acceptable)
        throw std::runtime_error("non-finite logit accepted");
}
class Fixture : public NaiveN05FlashModel {
  public:
    int ranks;
    explicit Fixture(int ranks, bool packed = false, bool graph = false) : ranks(ranks) {
        setenv("FASTLLM_TP", ranks > 1 ? std::to_string(ranks).c_str() : "false", 1);
        deviceMap = {{"cuda:0", 1}};
        moeDeviceMap = deviceMap;
        weight.dicts = {{"num_hidden_layers", "2"},
                        {"hidden_size", "256"},
                        {"num_attention_heads", "16"},
                        {"num_key_value_heads", "4"},
                        {"head_dim", "32"},
                        {"v_head_dim", "16"},
                        {"swa_num_attention_heads", "16"},
                        {"swa_num_key_value_heads", "8"},
                        {"swa_head_dim", "32"},
                        {"swa_v_head_dim", "16"},
                        {"hybrid_layer_pattern", "[0,1]"},
                        {"moe_layer_freq", "[0,1]"},
                        {"scoring_func", "sigmoid"},
                        {"sliding_window", "8"},
                        {"index_top_k", "16"},
                        {"indexer_activation_dtype", "bf16"},
                        {"n_routed_experts", "8"},
                        {"num_experts_per_tok", "2"},
                        {"max_position_embeddings", "1024"}};
        if (graph) {
            weight.dicts["index_top_k"] = "2048";
            weight.dicts["max_position_embeddings"] = "8192";
        }
        InitParams();
        SetSaveHistoryChat(false);
        unsigned seed = 7;
        auto add = [&](std::string name, std::vector<int> dims, DataType type, float scale,
                       bool norm = false) {
            if (packed && name.find(".mlp.experts.") != std::string::npos) {
                weight.AddEmptyWeight(name, dims, NVFP4_BLOCK_16_E4M3_PACKED);
                auto &w = weight[name];
                w.directMemory = true;
                w.blockK = 1;
                w.blockM = 16;
                w.Allocate(false);
                size_t stride = w.GetBytes() / dims[0];
                for (int row = 0; row < dims[0]; ++row) {
                    uint8_t *dst = w.cpuData + row * stride;
                    float global =
                        name.find("gateup_proj") != std::string::npos && row >= dims[0] / 2 ? .006f
                                                                                            : .004f;
                    std::memcpy(dst, &global, sizeof(float));
                    for (int group = 0; group < dims[1] / 16; ++group) {
                        for (int j = 0; j < 8; ++j) {
                            seed = seed * 1664525 + 1013904223;
                            dst[4 + group * 9 + j] = seed >> 16;
                        }
                        dst[12 + group * 9] = 56; // E4M3 1.0
                    }
                }
                return;
            }
            weight.AddEmptyWeight(name, dims, type);
            auto &w = weight[name];
            w.Allocate();
            for (int i = 0; i < w.Count(0); ++i) {
                seed = seed * 1664525 + 1013904223;
                float v = norm ? 1.f : scale * (int(seed >> 16) - 32768) / 32768.f;
                if (type == FLOAT32)
                    ((float *)w.cpuData)[i] = v;
                else {
                    uint32_t bits;
                    std::memcpy(&bits, &v, 4);
                    ((uint16_t *)w.cpuData)[i] = bits >> 16;
                }
            }
        };
        add("model.embed_tokens.weight", {256, 256}, BFLOAT16, .2f);
        add("model.norm.weight", {256}, FLOAT32, 0, true);
        add("lm_head.weight", {256, 256}, BFLOAT16, .03f);
        for (int i = 0; i < 2; ++i) {
            auto p = "model.layers." + std::to_string(i);
            auto ap = p + ".self_attn.";
            int kv = i ? 8 : 4;
            add(p + ".input_layernorm.weight", {256}, FLOAT32, 0, true);
            add(p + ".post_attention_layernorm.weight", {256}, FLOAT32, 0, true);
            add(ap + "q_proj.weight", {512, 256}, BFLOAT16, .03f);
            add(ap + "k_proj.weight", {kv * 32, 256}, BFLOAT16, .03f);
            add(ap + "v_proj.weight", {kv * 16, 256}, BFLOAT16, .03f);
            add(ap + "o_proj.weight", {256, 256}, BFLOAT16, .03f);
            add(ap + "attention_sink_bias", {16}, FLOAT32, .1f);
            if (i == 0) {
                add(ap + "indexer.wq.weight", {2048, 256}, BFLOAT16, .03f);
                add(ap + "indexer.wk.weight", {128, 256}, BFLOAT16, .03f);
                add(ap + "indexer.weights_proj.weight", {16, 256}, BFLOAT16, .03f);
                add(ap + "indexer.k_norm.weight", {128}, FLOAT32, 0, true);
                add(ap + "indexer.k_norm.bias", {128}, FLOAT32, .01f);
                add(p + ".mlp.gate_proj.weight", {1024, 256}, BFLOAT16, .02f);
                add(p + ".mlp.up_proj.weight", {1024, 256}, BFLOAT16, .02f);
                add(p + ".mlp.down_proj.weight", {256, 1024}, BFLOAT16, .02f);
            } else {
                add(p + ".mlp.gate.weight", {8, 256}, FLOAT32, .2f);
                add(p + ".mlp.gate.e_score_correction_bias", {8}, FLOAT32, .01f);
                for (int e = 0; e < 8; ++e) {
                    auto ep = p + ".mlp.experts." + std::to_string(e) + ".";
                    add(ep + "gateup_proj.weight", {2048, 256}, BFLOAT16, .02f);
                    add(ep + "down_proj.weight", {256, 1024}, BFLOAT16, .02f);
                }
            }
        }
    }
    std::vector<std::vector<float>> Run() {
        if (RetainCudaWorkspace() != (ranks > 1))
            throw std::runtime_error("unexpected serial/TP workspace policy");
        WarmUp();
        if (elementsInKVCachePerToken != std::max(4, ranks) * 48 + ranks * 128)
            throw std::runtime_error("physical KV cache accounting mismatch");
        std::vector<std::pair<Data, Data>> kv;
        for (int i = 0; i < 2; ++i)
            kv.emplace_back(Data(BFLOAT16), Data(BFLOAT16));
        std::vector<std::vector<float>> out;
        int past = 0;
        for (int n : {3, 17, 1, 1}) {
            std::vector<float> ids(n), positions(n);
            for (int i = 0; i < n; ++i) {
                ids[i] = (past + i) * 3 % 256;
                positions[i] = past + i;
            }
            Data input(FLOAT32, {1, n}, ids), pos(FLOAT32, {1, n}, positions);
            GenerationConfig cfg;
            cfg.output_logits = true;
            std::vector<float> logits;
            Forward(input, Data(), pos, kv, cfg, LastTokensManager(), &logits);
            out.push_back(logits);
            past += n;
            if (kv[0].first.dims[1] != past || kv[1].first.dims[1] != std::min(past, 7))
                throw std::runtime_error("cache metadata mismatch");
        }
        return out;
    }

    void VerifySelections(int ranks) {
        draftTargetLayers = {0, 1};
        int checks = 0;
        for (int topk : {1, 4, 50}) {
            GenerationConfig cfg; cfg.top_k = topk; cfg.temperature = .7f; cfg.top_p = .8f;
            std::vector<std::vector<float>> expectedLogits;
            std::vector<std::map<int, std::vector<char>>> expectedHidden;
            for (bool reference : {true, false}) {
                std::vector<std::pair<Data, Data>> kv(2);
                int past = 0, step = 0;
                for (int rows : {3, 8, 8, 8, 1, 3, 8, 8}) {
                    std::vector<float> ids(rows), pos(rows);
                    for (int row = 0; row < rows; ++row) { ids[row] = (past + row) * 3 % 256; pos[row] = past + row; }
                    Data input(FLOAT32, {1, rows}, ids), positions(FLOAT32, {1, rows}, pos);
                    TargetCapture capture; capture.verifying = past > 0;
                    auto selection = SelectLogits(cfg, true);
                    if (!reference) capture.selection = &selection;
#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
                    int launches = verifyGraphLaunches;
#endif
                    Data logits = RunDraftTarget(input, positions, kv, cfg, capture);
                    if (reference) {
                        logits.ToDevice(DataDevice::CPU);
                        expectedLogits.emplace_back((float *)logits.cpuData, (float *)logits.cpuData + logits.Count(0));
                        expectedHidden.emplace_back();
                    } else {
                        if (selection.candidates.dims.empty() || !logits.dims.empty())
                            throw std::runtime_error("verify compact output missing");
#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
                        if (step == 3 && verifyGraphLaunches - launches != ranks)
                            throw std::runtime_error("compact verify did not replay its graph");
#endif
                        int outputRows = past ? rows : 1;
                        if (topk == 1) {
                            Data full(FLOAT32, {1, outputRows, 256}, expectedLogits[step]);
                            ApplyDraftDevice(); Data top; TopK(full, top, 1); top.ToDevice(DataDevice::CPU);
                            for (int row = 0; row < outputRows; ++row)
                                if (((float *)top.cpuData)[row * 2] != ((float *)selection.candidates.cpuData)[row * 2])
                                    throw std::runtime_error("compact greedy verify token mismatch");
                        } else {
                            for (int row = 0; row < outputRows; ++row) {
                                auto p = SpeculativeDistribution(expectedLogits[step].data() + row * 256, 256, cfg, LastTokensUnit());
                                auto q = SpeculativeTopKDistribution((float *)selection.candidates.cpuData + row * topk * 2, topk, 256, cfg);
                                if (p != q) throw std::runtime_error("compact verify probabilities mismatch");
                            }
                        }
                        if (capture.hidden.size() != expectedHidden[step].size())
                            throw std::runtime_error("compact verify features missing");
                    }
                    for (auto &feature : capture.hidden) {
                        feature.second.ToDevice(DataDevice::CPU);
                        auto *data = (char *)feature.second.cpuData;
                        std::vector<char> bytes(data, data + feature.second.GetBytes());
                        if (reference) expectedHidden.back()[feature.first] = bytes;
                        else if (expectedHidden[step].at(feature.first) != bytes)
                            throw std::runtime_error("compact verify hidden state mismatch");
                    }
                    int keep = past && rows > 1 ? rows - 1 : rows;
                    CommitTargetCache(kv, past, keep); past += keep; ++step;
                    if (!reference) ++checks;
                }
            }
        }
        std::cout << "VERIFY SELECTION PASS checks=" << checks << std::endl;
    }

    void VerifyHead() {
        Data referenceHead(weight["lm_head.weight"]);
        std::vector<std::pair<Data, Data>> kv(2);
        Data ids(FLOAT32, {1, 3}, {1, 2, 3}), pos(FLOAT32, {1, 3}, {0, 1, 2});
        GenerationConfig cfg;
        TargetCapture capture;
        RunDraftTarget(ids, pos, kv, cfg, capture);
        for (int rows : {1, 2, 7, 3, 8, 7}) {
            Data hidden(BFLOAT16, {1, rows, 256});
            hidden.Allocate();
            for (int i = 0; i < hidden.Count(0); ++i) {
                float x = std::sin((i + rows) * .1f);
                uint32_t bits;
                std::memcpy(&bits, &x, 4);
                ((uint16_t *)hidden.cpuData)[i] = bits >> 16;
            }
            hidden.ToDevice(DataDevice::CUDA, std::vector<int>{0});
            Data actual = RunDraftHead(hidden), expected;
            ApplyDraftDevice();
            if (rows > 1)
                MatMulTransB(hidden, referenceHead, expected);
            else
                Linear(hidden, referenceHead, Data(), expected);
            actual.ToDevice(DataDevice::CPU);
            expected.ToDevice(DataDevice::CPU);
            if (actual.dataType != BFLOAT16 || expected.dataType != BFLOAT16 ||
                actual.dims != expected.dims)
                throw std::runtime_error("head output shape or dtype differs");
            int differing = 0;
            for (int i = 0; i < actual.Count(0); ++i)
                differing += ((uint16_t *)actual.cpuData)[i] != ((uint16_t *)expected.cpuData)[i];
            std::cout << "HEAD difference=" << differing << " / " << actual.Count(0) << std::endl;
            bool nonzero = false;
            for (int i = 0; i < expected.Count(0); ++i)
                nonzero |= (((uint16_t *)expected.cpuData)[i] & 0x7fff) != 0;
            if (!nonzero)
                throw std::runtime_error("head regression requires nonzero logits");
            if (differing)
                throw std::runtime_error("head differs");
        }
        std::cout << "HEAD PASS checks=6" << std::endl;
    }
#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
    void VerifyGraphBlocks() {
        draftTargetLayers = {0, 1};
        int checks = 0;
        for (int prompt : {3, 120, 254, 510, 2045, 2049, 4090}) {
            std::vector<std::pair<Data, Data>> reference(2), candidate(2);
            GenerationConfig cfg;
            cfg.input_token_length = prompt;
            cfg.output_token_limit = 96;
            cfg.output_logits = true;
            auto forward = [&](auto &kv, int start, int count, bool verifying) {
                std::vector<float> ids(count), pos(count);
                for (int row = 0; row < count; ++row) {
                    ids[row] = (start + row) * 3 % 256;
                    pos[row] = start + row;
                }
                Data input(FLOAT32, {1, count}, ids), positions(FLOAT32, {1, count}, pos);
                TargetCapture capture;
                capture.verifying = verifying;
                Data out = RunDraftTarget(input, positions, kv, cfg, capture);
                out.ToDevice(DataDevice::CPU);
                if (out.dims != std::vector<int>({1, verifying ? count : 1, 256}) ||
                    capture.hidden.size() != 2)
                    throw std::runtime_error("missing graph logits/features");
                for (int layer = 0; layer < 2; ++layer)
                    if (capture.hidden.at(layer).Count(0) != count * 256)
                        throw std::runtime_error("graph feature shape mismatch");
                return std::vector<float>((float *)out.cpuData,
                                          (float *)out.cpuData + out.Count(0));
            };
            forward(reference, 0, prompt, false);
            forward(candidate, 0, prompt, false);
            int past = prompt;
            for (int keep : {1, 4, 8, 3, 7, 8}) {
                auto block = forward(candidate, past, 8, true);
                for (int row = 0; row < keep; ++row) {
                    auto one = forward(reference, past + row, 1, false);
                    RequireCloseLogits(block.data() + row * 256, one,
                                       "graph verification differs from decode at " +
                                       std::to_string(past) + " row " + std::to_string(row));
                    ++checks;
                }
                CommitTargetCache(candidate, past, keep);
                for (int layer = 0; layer < 2; ++layer)
                    if (candidate[layer].first.dims != reference[layer].first.dims)
                        throw std::runtime_error("graph rollback metadata mismatch");
                past += keep;
            }
        }
        if (expectVerifyGraph &&
            (!verifyGraphCaptures ||
             ((!failVerifyBegin && !failVerifyInstantiate) != (verifyGraphLaunches > 0))))
            throw std::runtime_error("verification graph execution/fallback was not exercised");
        std::cout << "LOGIT ERROR relative_max=" << largestLogitRelative
                  << " absolute_max=" << largestLogitAbsolute << std::endl;
        std::cout << "VERIFY GRAPH PASS checks=" << checks << " captures=" << verifyGraphCaptures
                  << " launches=" << verifyGraphLaunches << std::endl;
    }

#endif
    void VerifyRollback() {
        draftTargetLayers = {0, 1};
        int checks = 0;
        for (int past : {3, 7, 15}) for (int rows : {1, 7, 8})
            for (int keep : {0, 1, rows / 2, rows}) {
                std::vector<std::pair<Data, Data>> clean(2), poisoned(2);
                GenerationConfig cfg;
                cfg.input_token_length = past;
                cfg.output_token_limit = 32;
                cfg.output_logits = true;
                auto forward = [&](auto &kv, int start, int count, bool verifying, bool poison) {
                    std::vector<float> ids(count), pos(count);
                    for (int row = 0; row < count; ++row) {
                        ids[row] = ((start + row) * 3 + (poison && row >= keep ? 71 : 0)) % 256;
                        pos[row] = start + row;
                    }
                    Data input(FLOAT32, {1, count}, ids), positions(FLOAT32, {1, count}, pos);
                    TargetCapture capture;
                    capture.verifying = verifying;
                    Data out = RunDraftTarget(input, positions, kv, cfg, capture);
                    out.ToDevice(DataDevice::CPU);
                    if (out.dims != std::vector<int>({1, verifying ? count : 1, 256}))
                        throw std::runtime_error("rollback logit shape mismatch");
                    return std::vector<float>((float *)out.cpuData,
                                              (float *)out.cpuData + out.Count(0));
                };
                forward(clean, 0, past, false, false);
                forward(poisoned, 0, past, false, false);
                auto expected = forward(clean, past, rows, true, false);
                auto actual = forward(poisoned, past, rows, true, true);
                if (!std::equal(expected.begin(), expected.begin() + keep * 256, actual.begin()))
                    throw std::runtime_error("rejected token changed earlier verify rows");
                // Repeating the same width/input after rejecting every row must be exact.
                CommitTargetCache(clean, past, 0);
                if (forward(clean, past, rows, true, false) != expected)
                    throw std::runtime_error("same-path rollback/replay differs");
                CommitTargetCache(clean, past, keep);
                CommitTargetCache(poisoned, past, keep);
                for (int layer = 0; layer < 2; ++layer)
                    if (clean[layer].first.dims != poisoned[layer].first.dims ||
                        clean[layer].second.dims != poisoned[layer].second.dims)
                        throw std::runtime_error("same-path rollback metadata differs");
                if (forward(clean, past + keep, 1, false, false) !=
                    forward(poisoned, past + keep, 1, false, false))
                    throw std::runtime_error("rejected suffix changed same-path subsequent decode");
                ++checks;
            }
        std::cout << "VERIFY ROLLBACK PASS checks=" << checks << std::endl;
    }
    void VerifyBlocks() {
        // Compare against the same rank count: TP has its own reduction tree.
        draftTargetLayers = {0, 1};
        int checks = 0;
        for (int past : {3, 7, 15, 17})
            for (int rows : {2, 3, 4, 5, 6, 7, 8}) {
                for (int keep : {1, rows / 2, rows}) {
                    std::vector<std::pair<Data, Data>> reference(2), candidate(2);
                    GenerationConfig cfg;
                    cfg.input_token_length = past;
                    cfg.output_token_limit = 32;
                    cfg.output_logits = true;
                    TargetCapture referenceCapture;
                    auto forward = [&](auto &kv, int start, int count) {
                        std::vector<float> ids(count), pos(count), result;
                        for (int i = 0; i < count; ++i) {
                            ids[i] = (start + i) * 3 % 256;
                            pos[i] = start + i;
                        }
                        Data input(FLOAT32, {1, count}, ids), positions(FLOAT32, {1, count}, pos);
                        Data logits = RunDraftTarget(input, positions, kv, cfg, referenceCapture);
                        logits.ToDevice(DataDevice::CPU);
                        result.assign((float *)logits.cpuData,
                                      (float *)logits.cpuData + logits.Count(0));
                        return result;
                    };
                    forward(reference, 0, past);
                    forward(candidate, 0, past);
                    std::vector<float> ids(rows), pos(rows);
                    for (int i = 0; i < rows; ++i) {
                        ids[i] = (past + i) * 3 % 256;
                        pos[i] = past + i;
                    }
                    Data input(FLOAT32, {1, rows}, ids), positions(FLOAT32, {1, rows}, pos);
                    TargetCapture capture;
                    capture.verifying = true;
                    Data block = RunDraftTarget(input, positions, candidate, cfg, capture);
                    block.ToDevice(DataDevice::CPU);
                    if (block.dims != std::vector<int>({1, rows, 256}) ||
                        capture.hidden.size() != 2)
                        throw std::runtime_error("missing verification logits/features");
                    for (int row = 0; row < keep; ++row) {
                        auto one = forward(reference, past + row, 1);
                        const float *actual = (const float *)block.cpuData + row * 256;
                        RequireCloseLogits(actual, one, "block verification differs from decode");
                        ++checks;
                    }
                    CommitTargetCache(candidate, past, keep);
                    for (int layer = 0; layer < 2; ++layer) {
                        if (candidate[layer].first.dims != reference[layer].first.dims)
                            throw std::runtime_error("rollback root length mismatch");
                        if (ranks > 1)
                            for (int device = 0; device < ranks; ++device)
                                if (candidate[layer].first.multiDeviceDatas.at(device)->dims !=
                                    reference[layer].first.multiDeviceDatas.at(device)->dims)
                                    throw std::runtime_error("rollback rank length mismatch");
                    }
                    auto expectedNext = forward(reference, past + keep, 1);
                    auto actualNext = forward(candidate, past + keep, 1);
                    if (actualNext.size() != expectedNext.size())
                        throw std::runtime_error("post-commit logit shape mismatch");
                    RequireCloseLogits(actualNext.data(), expectedNext,
                                       "rejected suffix changed subsequent decode");
                    ++checks;
                }
            }
        std::cout << "LOGIT ERROR relative_max=" << largestLogitRelative
                  << " absolute_max=" << largestLogitAbsolute << std::endl;
        std::cout << "VERIFY BLOCK PASS ranks=" << ranks << " checks=" << checks << std::endl;
    }
};
// A 4096-wide BF16 verification block crosses the TP8 48-KiB policy
// boundary at six rows. Compare full-block reduction against ordinary decode.
static void VerifyRowAllReduce(int ranks) {
    SetCudaGraph(true);
    std::vector<int> devices(ranks);
    for (int i = 0; i < ranks; ++i)
        devices[i] = i;
    if (!FastllmInitNccl(devices))
        throw std::runtime_error("collective init");
    PersistentWorkerGroup workers;
    std::vector<std::exception_ptr> errors(ranks);
    auto run = [&](const std::function<void(int)> &fn) {
        std::fill(errors.begin(), errors.end(), nullptr);
        workers.Run(
            devices,
            [&](int rank) {
                FastllmCudaSetDevice(rank);
                fn(rank);
            },
            errors);
        for (auto e : errors)
            if (e)
                std::rethrow_exception(e);
    };
    int checks = 0;
    for (DataType type : {BFLOAT16, FLOAT32})
        for (int rows : {2, 5, 6, 7, 8}) {
            constexpr int width = 4096;
            std::vector<Data> reference(ranks), block(ranks);
            for (int rank = 0; rank < ranks; ++rank) {
                Data input(type, {rows, width});
                input.Allocate();
                unsigned seed = 37 + rank;
                for (int i = 0; i < rows * width; ++i) {
                    seed = seed * 1664525 + 1013904223;
                    float value = (int(seed >> 16) - 32768) / 2048.f;
                    if (type == FLOAT32)
                        ((float *)input.cpuData)[i] = value;
                    else {
                        uint32_t bits;
                        std::memcpy(&bits, &value, 4);
                        ((uint16_t *)input.cpuData)[i] = bits >> 16;
                    }
                }
                reference[rank].CopyFrom(input);
                block[rank].CopyFrom(input);
                reference[rank].ToDevice(DataDevice::CUDA, std::vector<int>{rank});
                block[rank].ToDevice(DataDevice::CUDA, std::vector<int>{rank});
            }
            run([&](int rank) {
                auto &d = reference[rank];
                for (int row = 0; row < rows; ++row) {
                    void *ptr = (uint8_t *)d.cudaData + (size_t)row * width * d.unitSize;
                    FastllmNcclAllReduce(ptr, ptr, width, type, rank);
                }
                ForceDeviceSync();
            });
            run([&](int rank) {
                auto &d = block[rank];
                if (!FastllmCudaCustomAllReduceRows(d.cudaData, d.cudaData, d.Count(0), width, type,
                                                    rank)) {
                    for (int row = 0; row < rows; ++row) {
                        void *ptr = (uint8_t *)d.cudaData + (size_t)row * width * d.unitSize;
                        FastllmNcclAllReduce(ptr, ptr, width, type, rank);
                    }
                }
                ForceDeviceSync();
            });
            for (int rank = 0; rank < ranks; ++rank) {
                reference[rank].ToDevice(DataDevice::CPU);
                block[rank].ToDevice(DataDevice::CPU);
                if (std::memcmp(reference[rank].cpuData, block[rank].cpuData,
                                block[rank].GetBytes()))
                    throw std::runtime_error("block collective differs from single-row reduction");
                ++checks;
            }
        }
    std::cout << "ROW ALLREDUCE PASS checks=" << checks << std::endl;
}
#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
class DraftFixture : public NaiveN05FlashModel {
  public:
    DraftFixture() {
        embed_dim = 256;
        block_cnt = 2;
        draftLayers = 2;
        draftBlock = 7;
        draftHeads = 8;
        draftKvHeads = 2;
        draftHeadDim = 32;
        draftWindow = 1024;
        deviceMap = {{"cuda:0", 1}};
        unsigned seed = 17;
        auto add = [&](const std::string &name, std::vector<int> dims, bool norm = false) {
            weight.AddEmptyWeight(name, dims, norm ? FLOAT32 : BFLOAT16);
            Data &w = weight[name];
            w.Allocate();
            for (int i = 0; i < w.Count(0); ++i) {
                seed = seed * 1664525u + 1013904223u;
                float value = norm ? 1.f : .04f * ((int)(seed >> 16) - 32768) / 32768.f;
                if (norm)
                    ((float *)w.cpuData)[i] = value;
                else
                    ((uint16_t *)w.cpuData)[i] = Float32ToBFloat16RNEBits(value);
            }
        };
        add("model.embed_tokens.weight", {256, 256});
        add("dspark.mask_embedding", {256});
        add("dspark.norm.weight", {256}, true);
        add("dspark.markov_head.markov_w1.weight", {513, 256});
        add("dspark.markov_head.markov_w2.weight", {513, 256});
        draftTargetLayers = {0, 1};
        add("dspark.fc.weight", {256, 512});
        add("dspark.hidden_norm.weight", {256}, true);
        for (int i = 0; i < draftLayers; ++i) {
            std::string p = "dspark.layers." + std::to_string(i) + ".";
            add(p + "input_layernorm.weight", {256}, true);
            add(p + "post_attention_layernorm.weight", {256}, true);
            for (const char *n : {"q", "k", "v"})
                add(p + "self_attn." + n + "_proj.weight", {n[0] == 'q' ? 256 : 64, 256});
            add(p + "self_attn.o_proj.weight", {256, 256});
            add(p + "self_attn.q_norm.weight", {32}, true);
            add(p + "self_attn.k_norm.weight", {32}, true);
            add(p + "mlp.gate_proj.weight", {512, 256});
            add(p + "mlp.up_proj.weight", {512, 256});
            add(p + "mlp.down_proj.weight", {256, 512});
        }
        // Match the model loader's GPU embedding placement in both paths.
        weight["model.embed_tokens.weight"].ToDevice(DataDevice::CUDA, std::vector<int>{0});
        weight["dspark.mask_embedding"].ToDevice(DataDevice::CUDA, std::vector<int>{0});
    }
    void Run() {
        int checks = 0;
        auto read = [](const Data &value) {
            size_t count = 1;
            for (int dim : value.dims)
                count *= dim;
            std::vector<uint16_t> bits(count);
            if (cudaMemcpy(bits.data(), value.cudaData, count * sizeof(uint16_t),
                           cudaMemcpyDeviceToHost) != cudaSuccess)
                throw std::runtime_error("draft read failed");
            return bits;
        };
        auto hidden = [&](int count, int seed) {
            Data x(BFLOAT16, {1, count, embed_dim});
            x.Allocate();
            for (int i = 0; i < x.Count(0); ++i)
                ((uint16_t *)x.cpuData)[i] = Float32ToBFloat16RNEBits(std::sin((i + seed) * .137f));
            x.ToDevice(DataDevice::CUDA, std::vector<int>{0});
            return x;
        };
        for (int prefix : {3, 80, 249, 250, 1023, 1024, 32768, 3}) {
            auto candidate = CreateDraftContext();
            auto reference = std::make_shared<DraftContext>();
            int count = std::min(prefix, draftWindow - 1);
            Data h = hidden(count, prefix);
            AppendDraftContext(h, prefix - count, *candidate);
            AppendDraftContext(h, prefix - count, *reference);
            for (int round = 0; round < 4; ++round) {
                SetCudaGraph(false);
                Data expected = RunDraft(11 + round, *reference);
                auto bits = read(expected);
                SetCudaGraph(true);
                Data actual = RunDraft(11 + round, *candidate);
                if (bits != read(actual))
                    throw std::runtime_error("draft backbone differs from eager");
                if (candidate->committed != reference->committed)
                    throw std::runtime_error("draft proposal changed committed length");
                Data base(BFLOAT16, {1, draftBlock, 513});
                base.Allocate();
                for (int i = 0; i < base.Count(0); ++i)
                    ((uint16_t *)base.cpuData)[i] = Float32ToBFloat16RNEBits(std::sin((i + round + prefix) * .019f));
                base.ToDevice(DataDevice::CUDA, std::vector<int>{0});
                std::vector<int> expectedIds, actualIds;
                int previous = 11 + round;
                for (int step = 0; step < draftBlock; ++step) {
                    Data id(FLOAT32, {1, 1}, {(float)previous}), latent, bias, logits, top;
                    Embedding(id, weight["dspark.markov_head.markov_w1.weight"], latent);
                    ToDataType(latent, BFLOAT16);
                    Linear(latent, weight["dspark.markov_head.markov_w2.weight"], Data(), bias);
                    Split(base, 1, step, step + 1, logits);
                    AddTo(logits, bias); ToDataType(logits, FLOAT32); TopK(logits, top, 1);
                    top.ToDevice(DataDevice::CPU);
                    previous = (int)((float *)top.cpuData)[0]; expectedIds.push_back(previous);
                }
                bool proposalGraph = RunDraftProposalGraph(11 + round, base, *candidate, actualIds);
                if (!failVerifyBegin && !failVerifyInstantiate) {
                    if (!proposalGraph || actualIds != expectedIds)
                        throw std::runtime_error("GPU proposal chain differs from eager");
                } else if (proposalGraph) throw std::runtime_error("failed proposal graph did not fall back");
                Data tail;
                Split(base, 1, 0, draftBlock - 1, tail);
                if (RunDraftProposalGraph(11, tail, *candidate, actualIds))
                    throw std::runtime_error("incomplete proposal block did not fall back");
                for (int layer = 0; layer < draftLayers; ++layer) {
                    if (read(candidate->kv[layer].first) != read(reference->kv[layer].first) ||
                        read(candidate->kv[layer].second) != read(reference->kv[layer].second))
                        throw std::runtime_error("draft proposal changed visible KV");
                }
                int accepted = round == 3 ? 8 : round + 1;
                Data next = hidden(accepted, round + prefix);
                TargetCapture capture;
                capture.hidden.emplace(0, next);
                capture.hidden.emplace(1, hidden(accepted, round + prefix + 17));
                Data joined, projected, normed;
                Cat(capture.hidden.at(0), capture.hidden.at(1), -1, joined);
                if (accepted > 1) MatMulTransB(joined, weight["dspark.fc.weight"], projected);
                else Linear(joined, weight["dspark.fc.weight"], Data(), projected);
                KimiK3RMSNorm(projected, weight["dspark.hidden_norm.weight"], draftEps, normed);
                std::vector<std::pair<Data, Data>> unused;
                CommitDraftContext(capture, accepted, *candidate, unused);
                AppendDraftContext(normed, reference->committed, *reference);
                for (int layer = 0; layer < draftLayers; ++layer) {
                    if (read(candidate->kv[layer].first) != read(reference->kv[layer].first) ||
                        read(candidate->kv[layer].second) != read(reference->kv[layer].second))
                        throw std::runtime_error("draft commit workspace changed KV");
                }
                ++checks;
            }
            // Exercise the same bounded storage transfer as request removal.
            idleDraftContext = std::move(candidate);
        }
        if (!failVerifyBegin && !failVerifyInstantiate && verifyGraphLaunches == 0)
            throw std::runtime_error("draft did not replay a graph");
        if ((failVerifyBegin || failVerifyInstantiate) && verifyGraphLaunches != 0)
            throw std::runtime_error("failed draft graph was launched");
        std::cout << "DRAFT GRAPH PASS checks=" << checks << " captures=" << verifyGraphCaptures
                  << " launches=" << verifyGraphLaunches << std::endl;
    }
};
#endif

int main(int argc, char **argv) {
    try {
        CheckLogitComparison();
        int ranks = argc > 1 ? std::stoi(argv[1]) : 8;
        if (ranks != 1 && ranks != 2 && ranks != 4 && ranks != 8)
            return 2;
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < ranks)
            return 77;
        if (argc > 2 && std::string(argv[2]) == "reduce_rows") {
            VerifyRowAllReduce(ranks);
            return 0;
        }
        SetThreads(4);
        SetDeviceMap({{"cuda:0", 1}});
        FastllmCudaSetDevice(0);
        if (argc > 2 && std::string(argv[2]) == "selection") {
            SetCudaEmbedding(true); SetCudaGraph(true);
            Fixture fixture(ranks, true, true); fixture.VerifySelections(ranks);
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "head") {
            Fixture fixture(ranks);
            fixture.VerifyHead();
            return 0;
        }
#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
        if (argc > 2 && std::string(argv[2]) == "draft_graph") {
            SetCudaEmbedding(true);
            failGraphDevice = 0;
            failVerifyBegin = argc > 3 && std::string(argv[3]) == "failbegin";
            failVerifyInstantiate = argc > 3 && std::string(argv[3]) == "failinstantiate";
            DraftFixture fixture; fixture.Run();
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "verify_graph") {
            expectVerifyGraph = argc <= 3 || std::string(argv[3]) != "eager";
            SetCudaGraph(expectVerifyGraph);
            SetCudaEmbedding(true);
            failVerifyBegin = argc > 3 && std::string(argv[3]) == "failbegin";
            failVerifyInstantiate = argc > 3 && std::string(argv[3]) == "failinstantiate";
            Fixture fixture(ranks, true, true);
            fixture.VerifyGraphBlocks();
            return 0;
        }
#endif
        if (argc > 2 && std::string(argv[2]) == "rollback") {
            SetCudaGraph(ranks > 1);
            Fixture fixture(ranks, true, ranks > 1);
            fixture.VerifyRollback();
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "verify") {
            Fixture fixture(ranks, argc > 3 && std::string(argv[3]) == "packed");
            fixture.VerifyBlocks();
            return 0;
        }
        std::vector<std::vector<float>> reference;
        {
            Fixture serial(1);
            reference = serial.Run();
        }
        Fixture parallel(ranks);
        auto actual = parallel.Run();
        for (int t = 0; t < 4; ++t) {
            double err = 0, bb = 0, aa = 0, ab = 0, mx = 0;
            for (size_t i = 0; i < actual[t].size(); ++i) {
                double a = actual[t][i], b = reference[t][i];
                err += (a - b) * (a - b);
                mx = std::max(mx, std::abs(a - b));
                aa += a * a;
                bb += b * b;
                ab += a * b;
            }
            double relative = std::sqrt(err / bb), cos = ab / std::sqrt(aa * bb);
            std::cout << "step=" << t << " relative_rmse=" << relative << " cosine=" << cos
                      << " max_abs=" << mx << std::endl;
            if (!std::isfinite(relative) || relative > .02 || cos < .999)
                throw std::runtime_error("TP differs from serial logits");
        }
        std::cout << "MODEL SMOKE PASS" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
