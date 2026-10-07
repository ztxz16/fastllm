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
    bool UseModelSpecificScheduler() const override { return true; }
    void RunModelSpecificScheduler() override {}
    void HistoryChecks(bool graph) {
        auto require = [](bool ok, const char *message) {
            if (!ok) throw std::runtime_error(message);
        };
        auto create = [&](const std::vector<int> &tokens) {
            GenerationConfig cfg;
            cfg.top_k = 1;
            cfg.output_token_limit = 64;
            cfg.output_logits = true;
            cfg.input_token_length = tokens.size();
            // Request creation/restoration must not acquire the forward lock.
            std::lock_guard<std::mutex> lock(forwardLocker);
            return responseContextDict.GetHandle(LaunchResponseTokens(tokens, cfg));
        };
        auto feed = [&](ResponseContext *ctx, int start, int count) {
            std::vector<float> ids(count), positions(count), logits;
            for (int i = 0; i < count; ++i) {
                ids[i] = ctx->allTokens[start + i];
                positions[i] = start + i;
            }
            Data input(FLOAT32, {1, count}, ids), pos(FLOAT32, {1, count}, positions);
            Forward(input, Data(), pos, ctx->pastKeyValues, ctx->generationConfig, LastTokensManager(), &logits);
            return logits;
        };
        using Snapshot = std::vector<std::vector<uint16_t>>;
        auto snapshot = [&](ResponseContext *ctx) {
            Snapshot result;
            for (int layer = 0; layer < 2; ++layer)
                for (int part = 0; part < 2; ++part)
                    for (int rank = 0; rank < ranks; ++rank) {
                        Data &root = part ? ctx->pastKeyValues[layer].second : ctx->pastKeyValues[layer].first;
                        Data local(*root.multiDeviceDatas.at(rank));
                        local.ToDevice(DataDevice::CPU);
                        auto *ptr = (uint16_t *)local.cpuData;
                        result.emplace_back(ptr, ptr + (size_t)local.dims[1] * local.dims[2]);
                    }
            return result;
        };
        // Restore without a suffix forward so the archive itself can be
        // compared bitwise with the original per-rank GPU state.
        auto restoreRanks = [&](Data &cache, int layer, bool value) {
            const int previous = FastllmCudaGetDevice();
            struct RestoreDevice {
                int device;
                ~RestoreDevice() { FastllmCudaSetDevice(device); }
            } restore{previous};
            cache.multiDeviceData = true;
            cache.dataDeviceIds.clear();
            for (int rank = 0; rank < ranks; ++rank) {
                auto *local = new Data(BFLOAT16);
                cache.multiDeviceDatas[rank] = local;
                local->dataDevice = DataDevice::CUDA;
                local->dataDeviceIds = {rank};
                local->isKVCache = true;
                cache.dataDeviceIds.push_back(rank);
                RestoreTensorParallelHistoryRank(cache, layer, value, 128, rank);
            }
            cache.FreeSpace();
            cache.dataDevice = DataDevice::CUDA;
        };
        auto checkHost = [&](ResponseContext *ctx, int length, const Snapshot &expected) {
            require(ctx->cacheLen == length && ctx->preTokens == length, "history hit length mismatch");
            int item = 0;
            for (int layer = 0; layer < 2; ++layer)
                for (int part = 0; part < 2; ++part) {
                    const Data &root = part ? ctx->pastKeyValues[layer].second : ctx->pastKeyValues[layer].first;
                    const int heads = layer ? 8 : 4, dim = part ? 16 : 32, index = !part && !layer ? 128 : 0;
                    const int rows = layer ? std::min(length, 7) : length;
                    require(root.dataDevice == DataDevice::CPU && !root.multiDeviceData &&
                                root.dims == std::vector<int>({1, rows, heads * dim + index}),
                            "host history layout mismatch");
                    for (int rank = 0; rank < ranks; ++rank, ++item) {
                        int width = std::max(1, heads / ranks) * dim, begin = rank * heads / ranks * dim;
                        require(expected[item].size() == (size_t)rows * (width + index), "snapshot dimensions mismatch");
                        for (int row = 0; row < rows; ++row) {
                            auto *actual = (const uint16_t *)root.cpuData + (size_t)row * root.dims[2];
                            auto *ref = expected[item].data() + (size_t)row * (width + index);
                            require(!std::memcmp(actual + begin, ref, width * 2), "archived KV differs from live rank KV");
                            if (index)
                                require(!std::memcmp(actual + heads * dim, ref + width, index * 2),
                                        "archived index differs");
                        }
                    }
                }
        };
        std::vector<int> tokens(36);
        for (int i = 0; i < 36; ++i) tokens[i] = i * 3 % 251;
        std::map<int, Snapshot> gold;
        std::map<int, std::vector<float>> logits;
        SetCudaGraph(graph);
        SetSaveHistoryChat(false);
        auto *base = create(tokens);
        for (int end = 12; end <= 36; ++end) {
            logits[end] = feed(base, end == 12 ? 0 : end - 1, end == 12 ? 12 : 1);
            gold[end] = snapshot(base);
        }
        SetSaveHistoryChat(true);
        auto *seed = create(tokens);
        for (int end = 12; end <= 36; ++end) {
            auto out = feed(seed, end == 12 ? 0 : end - 1, end == 12 ? 12 : 1);
            require(out == logits[end], "history recording changed ordinary logits");
            require(snapshot(seed) == gold[end], "history recording changed live KV");
        }
        TryRecordResponseContext(seed);
        // Fork far behind the live SWA tail, repeat, shorten and extend.
        for (int cut : {12, 13, 16, 31, 35, 36}) {
            std::vector<int> branch(tokens.begin(), tokens.begin() + cut);
            branch.push_back(249);
            auto *ctx = create(branch);
            checkHost(ctx, cut, gold[cut]);
            // A restored rank cache must match the exact saved GPU state
            // before computing any suffix, including replicated KV groups.
            for (int layer = 0; layer < 2; ++layer)
                for (int part = 0; part < 2; ++part) {
                    Data &root = part ? ctx->pastKeyValues[layer].second : ctx->pastKeyValues[layer].first;
                    restoreRanks(root, layer, part != 0);
                }
            require(snapshot(ctx) == gold[cut], "restoring host history changed rank KV");
            feed(ctx, cut, 1);
            TryRecordResponseContext(ctx);
            branch.push_back(248);
            auto *child = create(branch);
            checkHost(child, cut + 1, snapshot(ctx));
        }
        // Re-seed the original archive after deliberate LRU pressure above.
        TryRecordResponseContext(seed);
        auto *repeat = create(tokens);
        checkHost(repeat, 35, gold[35]);
        feed(repeat, 35, 1); // Exercise automatic CPU->TP conversion in Forward.
        require(snapshot(repeat) == gold[36], "exact repeated suffix changed KV");
        auto shorter = std::vector<int>(tokens.begin(), tokens.begin() + 17);
        checkHost(create(shorter), 16, gold[16]);
        // Two distinct live request owners share immutable host chunks, then
        // decode together with different histories and request order.
        std::vector<int> a(tokens.begin(), tokens.begin() + 20), b(tokens.begin(), tokens.begin() + 31);
        a.push_back(240);
        b.push_back(241);
        ResponseContext *contexts[2] = {create(a), create(b)};
        feed(contexts[0], 20, 1);
        feed(contexts[1], 31, 1);
        for (int step = 0; step < 5; ++step) {
            std::vector<float> ids;
            std::vector<Data> positions(2);
            std::vector<Data *> pos;
            std::vector<std::pair<Data *, Data *>> kv;
            std::vector<GenerationConfig> configs;
            for (int j = 0; j < 2; ++j) {
                int i = step % 2 ? 1 - j : j;
                auto *ctx = contexts[i];
                int past = ctx->pastKeyValues[0].first.dims[1], token = 200 + i + step;
                ctx->allTokens.push_back(token);
                ids.push_back(token);
                Data position(FLOAT32, {1, 1}, {float(past)});
                positions[j].CopyFrom(position);
                pos.push_back(&positions[j]);
                for (auto &layer : ctx->pastKeyValues) kv.emplace_back(&layer.first, &layer.second);
                configs.push_back(ctx->generationConfig);
            }
            Data input(FLOAT32, {1, 2}, ids);
            ForwardBatch(2, input, {}, pos, {1, 1}, kv, configs, LastTokensManager());
        }
        for (auto *ctx : contexts) {
            TryRecordResponseContext(ctx);
            auto branch = ctx->allTokens;
            branch.push_back(250);
            checkHost(create(branch), ctx->allTokens.size(), snapshot(ctx));
        }
        // End every live owner, then exercise the production idle-allocation
        // transfer on a prefix hit. CPU archive ownership remains independent.
        {
            std::lock_guard<std::mutex> lock(dictLocker);
            std::vector<int> handles;
            for (const auto &entry : responseContextDict.dicts) handles.push_back(entry.first);
            for (int handle : handles) {
                responseContextDict.GetHandle(handle)->isEnding = true;
                RemoveResponseContext(handle);
            }
        }
        SetSaveHistoryChat(false);
        SetSaveHistoryChat(true);
        auto *sole = create(tokens);
        feed(sole, 0, 12);
        for (int i = 12; i < 36; ++i) feed(sole, i, 1);
        auto expectedSole = snapshot(sole);
        std::vector<void *> pointers;
        for (auto &layer : sole->pastKeyValues)
            for (Data *root : {&layer.first, &layer.second})
                for (auto &rank : root->multiDeviceDatas) pointers.push_back(rank.second->cudaData);
        TryRecordResponseContext(sole);
        {
            std::lock_guard<std::mutex> lock(dictLocker);
            sole->isEnding = true;
            RemoveResponseContext(responseContextDict.dicts.begin()->first);
        }
        auto *hot = create(tokens);
        require(hot->cacheLen == 35, "idle allocation lost history hit");
        if (graph) {
            size_t i = 0;
            for (auto &layer : hot->pastKeyValues)
                for (Data *root : {&layer.first, &layer.second}) {
                    require(root->dataDevice == DataDevice::CPU && root->cpuData && root->multiDeviceData,
                            "history hit did not retain private host rows and idle GPU allocation");
                    for (auto &rank : root->multiDeviceDatas)
                        require(rank.second->cudaData == pointers.at(i++), "idle GPU allocation was replaced");
                }
        }
        feed(hot, 35, 1);
        if (graph) {
            size_t i = 0;
            for (auto &layer : hot->pastKeyValues)
                for (Data *root : {&layer.first, &layer.second})
                    for (auto &rank : root->multiDeviceDatas)
                        require(rank.second->cudaData == pointers.at(i++), "history upload reallocated idle GPU storage");
        }
        require(snapshot(hot) == expectedSole, "idle allocation history restore changed KV");
        SetSaveHistoryChat(false);
        require(create(tokens)->cacheLen == 0, "disabled history still hit");
        std::cout << "TP HISTORY PASS ranks=" << ranks << " graph=" << graph << std::endl;
    }
    void FeatureGraphChecks() {
        draftTargetLayers = {0, 1};
        std::vector<std::vector<float>> expectedLogits;
        std::vector<std::map<int, std::vector<char>>> expectedHidden;
        for (bool graph : {false, true}) {
            SetCudaGraph(graph);
            std::vector<std::pair<Data, Data>> kv(2);
            GenerationConfig cfg;
            cfg.input_token_length = 12;
            cfg.output_token_limit = 64;
            int past = 0;
            for (int step = 0; step < 7; ++step) {
                int rows = step ? 1 : 12;
                std::vector<float> ids(rows), positions(rows);
                for (int i = 0; i < rows; ++i) {
                    ids[i] = (past + i) * 3 % 251;
                    positions[i] = past + i;
                }
                Data input(FLOAT32, {1, rows}, ids), pos(FLOAT32, {1, rows}, positions);
                TargetCapture capture;
    #ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
                int before = verifyGraphLaunches;
    #endif
                Data logits = RunDraftTarget(input, pos, kv, cfg, capture);
                const float *values = (const float *)logits.cpuData;
                std::map<int, std::vector<char>> hidden;
                for (auto &item : capture.hidden) {
                    item.second.ToDevice(DataDevice::CPU);
                    auto *data = (char *)item.second.cpuData;
                    hidden[item.first] = std::vector<char>(data, data + item.second.GetBytes());
                }
                if (capture.hidden.size() != 2) throw std::runtime_error("single-token draft features missing");
                if (!graph) {
                    expectedLogits.emplace_back(values, values + logits.Count(0));
                    expectedHidden.push_back(std::move(hidden));
                } else {
                    RequireCloseLogits(values, expectedLogits[step], "single-token feature graph logits");
                    if (hidden != expectedHidden[step])
                        throw std::runtime_error("single-token feature graph changed hidden states");
    #ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
                    if (step >= 3 && verifyGraphLaunches - before != ranks)
                        throw std::runtime_error("single-token feature graph did not replay");
    #endif
                }
                past += rows;
            }
        }
        std::cout << "FEATURE GRAPH PASS ranks=" << ranks << std::endl;
    }
    void BatchChecks(bool graph, bool merged) {
        if (!canDoBatchForward) throw std::runtime_error("ordinary TP did not advertise batching");
        if (merged) for (int layer = 0; layer < 2; ++layer) {
            std::string base = "model.layers." + std::to_string(layer) + ".self_attn.";
            int rows = 0;
            for (auto name : {"q_proj.weight", "k_proj.weight", "v_proj.weight"})
                rows += weight[base + name].dims[0];
            weight.AddEmptyWeight(base + "mergeqkv.weight", {rows, 256}, BFLOAT16);
            auto &dst = weight[base + "mergeqkv.weight"];
            dst.Allocate(false);
            size_t offset = 0;
            for (auto name : {"q_proj.weight", "k_proj.weight", "v_proj.weight"}) {
                auto &src = weight[base + name];
                std::memcpy(dst.cpuData + offset, src.cpuData, src.GetBytes());
                offset += src.GetBytes();
                weight.weight.erase(base + name);
            }
        }
        int checks = 0;
        for (auto prompts : {std::pair<int,int>{3, 11}, {126, 250}, {254, 510}, {2045, 2049}, {4090, 13}}) {
            std::vector<std::vector<int>> schedule(12, {0, 1});
            schedule[4] = {1, 0}; schedule[5] = {1, 0};
            schedule[8] = {0}; // One request leaves the active decode batch.
            int counts[2] = {0, 0};
            for (const auto &order : schedule) for (int sequence : order) ++counts[sequence];
            std::vector<std::pair<Data,Data>> reference[2], candidate[2];
            std::vector<std::vector<float>> expected[2];
            std::vector<int> expectedTokens[2];
            int lastToken = 0;
            const int prompt[2] = {prompts.first, prompts.second};
            GenerationConfig cfg[2];
            auto forward = [&](std::vector<std::pair<Data,Data>> &kv, int sequence, int start, int rows) {
                std::vector<float> ids(rows), positions(rows);
                for (int j=0;j<rows;++j) { ids[j]=(start+j)*3%251+sequence; positions[j]=start+j; }
                Data input(FLOAT32,{1,rows},ids), pos(FLOAT32,{1,rows},positions);
                std::vector<float> logits;
                lastToken = Forward(input,Data(),pos,kv,cfg[sequence],LastTokensManager(),&logits);
                return logits;
            };
            for (int b=0;b<2;++b) {
                cfg[b].input_token_length=prompt[b]; cfg[b].output_token_limit=64;
                cfg[b].output_logits=true; cfg[b].top_k=1;
                reference[b].resize(2);candidate[b].resize(2);
                SetCudaGraph(false);
                forward(reference[b],b,0,prompt[b]);
                forward(candidate[b],b,0,prompt[b]);
                for (int step=0;step<counts[b];++step) {
                    expected[b].push_back(forward(reference[b],b,prompt[b]+step,1));
                    expectedTokens[b].push_back(lastToken);
                }
            }
            int steps[2]={0,0};
            for (int round=0;round<(int)schedule.size();++round) {
                const auto &order=schedule[round];
                SetCudaGraph(graph);
                if (order.size()==1) {
                    int b=order[0];auto actual=forward(candidate[b],b,prompt[b]+steps[b],1);
                    RequireCloseLogits(actual.data(),expected[b][steps[b]++],"batch survivor logits");
                    ++checks;continue;
                }
                std::vector<std::pair<Data*,Data*>> caches;
                std::vector<Data> positions(order.size());
                std::vector<Data*> pos;
                std::vector<float> ids;
                std::vector<GenerationConfig> configs;
                std::vector<std::vector<float>> actual(order.size());
                std::vector<std::vector<float>*> out;
                for (int i=0;i<(int)order.size();++i) {
                    int b=order[i],past=prompt[b]+steps[b];
                    ids.push_back(past*3%251+b);
                    positions[i].dataType=FLOAT32;positions[i].Resize({1,1});positions[i].Allocate();
                    ((float*)positions[i].cpuData)[0]=past;pos.push_back(&positions[i]);
                    for (auto &layer:candidate[b]) caches.emplace_back(&layer.first,&layer.second);
                    configs.push_back(cfg[b]);out.push_back(&actual[i]);
                    // Exercise compact GPU selection as well as full-logit sampling.
                    configs.back().output_logits = round != 3 && round != 6 && round != 10;
                }
#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
                int launches=verifyGraphLaunches, captures=verifyGraphCaptures;
#endif
                Data input(FLOAT32,{1,(int)order.size()},ids);
                auto tokens = ForwardBatch(order.size(),input,{},pos,std::vector<int>(order.size(),1),caches,configs,LastTokensManager(),&out);
#ifdef FASTLLM_TEST_VERIFY_GRAPH_HOOKS
                if (graph && !failVerifyBegin && !failVerifyInstantiate && prompts.first==3 && round==2 &&
                    (verifyGraphLaunches-launches!=ranks || verifyGraphCaptures!=captures))
                    throw std::runtime_error("batch did not replay exactly one graph per rank");
#endif
                for (int i=0;i<(int)order.size();++i) {
                    int b=order[i];
                    const auto &logits = expected[b][steps[b]++];
                    if (configs[i].output_logits) {
                        if (actual[i].size() != logits.size()) throw std::runtime_error("batch logits missing");
                        RequireCloseLogits(actual[i].data(), logits, "packed batch logits");
                    } else if (tokens[i] != expectedTokens[b][steps[b] - 1]) {
                        throw std::runtime_error("compact batch greedy token mismatch");
                    }
                    if (candidate[b][0].first.dims[1]!=prompt[b]+steps[b] ||
                        candidate[b][1].first.dims[1]!=std::min(prompt[b]+steps[b],7))
                        throw std::runtime_error("batch KV lengths crossed requests");
                    ++checks;
                }
            }
            for (int b=0;b<2;++b) for (int layer=0;layer<2;++layer) for (bool value:{false,true}) {
                auto &a=value?candidate[b][layer].second:candidate[b][layer].first;
                auto &e=value?reference[b][layer].second:reference[b][layer].first;
                for (const auto &entry:a.multiDeviceDatas) {
                    FastllmCudaSetDevice(entry.first);
                    const Data &actual=*entry.second,&expected=*e.multiDeviceDatas.at(entry.first);
                    if (actual.dims!=expected.dims) throw std::runtime_error("batch KV shape mismatch");
                    std::vector<uint16_t> av(actual.Count(0)),ev(expected.Count(0));
                    FastllmCudaCopyFromDeviceToHost(av.data(),actual.cudaData,actual.GetBytes());
                    FastllmCudaCopyFromDeviceToHost(ev.data(),expected.cudaData,expected.GetBytes());
                    std::vector<float> af(av.size()),ef(ev.size());
                    for(size_t j=0;j<av.size();++j){uint32_t x=uint32_t(av[j])<<16,y=uint32_t(ev[j])<<16;std::memcpy(&af[j],&x,4);std::memcpy(&ef[j],&y,4);}
                    RequireCloseLogits(af.data(),ef,"independent batch KV contents");++checks;
                }
            }
        }
        std::cout<<"BATCH PASS ranks="<<ranks<<" graph="<<graph<<" merged="<<merged
                 <<" checks="<<checks<<" relative="<<largestLogitRelative<<std::endl;
    }

    void PackedPrefillChecks() {
        GenerationConfig config;
        config.output_logits = true;
        config.output_token_limit = 16;
        config.top_k = 1;
        std::vector<int> lengths = {3, 7, 1};
        std::vector<std::vector<std::pair<Data, Data>>> reference(3), candidate(3);
        std::vector<Data> positions(3);
        std::vector<Data *> positionPtrs;
        std::vector<std::pair<Data *, Data *>> kv;
        std::vector<float> ids;
        std::vector<std::vector<float>> expected(3), actual(3);
        std::vector<std::vector<float> *> outputs;
        SetCudaGraph(false);
        for (int b = 0; b < 3; ++b) {
            reference[b].resize(2); candidate[b].resize(2);
            std::vector<float> tokens(lengths[b]), pos(lengths[b]);
            for (int j = 0; j < lengths[b]; ++j) { tokens[j] = 5 * j + b; pos[j] = j; }
            Data input(FLOAT32, {1, lengths[b]}, tokens), position(FLOAT32, {1, lengths[b]}, pos);
            Forward(input, Data(), position, reference[b], config, LastTokensManager(), &expected[b]);
            positions[b].CopyFrom(position); positionPtrs.push_back(&positions[b]);
            ids.insert(ids.end(), tokens.begin(), tokens.end());
            for (auto &layer : candidate[b]) kv.emplace_back(&layer.first, &layer.second);
            outputs.push_back(&actual[b]);
        }
        Data input(FLOAT32, {1, (int)ids.size()}, ids);
        std::vector<GenerationConfig> configs(3, config);
        ForwardBatch(3, input, {}, positionPtrs, lengths, kv, configs, LastTokensManager(), &outputs);
        for (int b = 0; b < 3; ++b) {
            if (actual[b].size() != expected[b].size()) throw std::runtime_error("prefill logits missing");
            RequireCloseLogits(actual[b].data(), expected[b], "packed prefill logits");
        }
        // Mixed lengths become a three-request decode batch with independent KV.
        for (int b = 0; b < 3; ++b) {
            Data token(FLOAT32, {1, 1}, {float(50 + b)}), pos(FLOAT32, {1, 1}, {float(lengths[b])});
            Forward(token, Data(), pos, reference[b], config, LastTokensManager(), &expected[b]);
            positions[b].CopyFrom(pos);
        }
        Data decode(FLOAT32, {1, 3}, {50, 51, 52});
        for (bool graph : {false, true}) {
            SetCudaGraph(graph);
            ForwardBatch(3, decode, {}, positionPtrs, {1, 1, 1}, kv, configs, LastTokensManager(), &outputs);
            for (int b = 0; b < 3; ++b) {
                RequireCloseLogits(actual[b].data(), expected[b], "prefill to batch decode logits");
                if (graph) continue;
                Data token(FLOAT32, {1, 1}, {float(50 + b)}), pos(FLOAT32, {1, 1}, {float(lengths[b] + 1)});
                Forward(token, Data(), pos, reference[b], config, LastTokensManager(), &expected[b]);
                positions[b].CopyFrom(pos);
            }
        }
        std::cout << "PACKED PREFILL PASS ranks=" << ranks << " relative=" << largestLogitRelative << std::endl;
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
    DraftFixture(bool merged = false, int kvHeads = 2, int owner = 0) {
        embed_dim = 256;
        block_cnt = 2;
        draftLayers = 2;
        draftBlock = 7;
        draftHeads = 8;
        draftKvHeads = kvHeads;
        draftHeadDim = 32;
        draftWindow = 1024;
        deviceMap = {{"cuda:" + std::to_string(owner), 1}};
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
                add(p + "self_attn." + n + "_proj.weight", {n[0] == 'q' ? 256 : draftKvHeads * draftHeadDim, 256});
            add(p + "self_attn.o_proj.weight", {256, 256});
            add(p + "self_attn.q_norm.weight", {32}, true);
            add(p + "self_attn.k_norm.weight", {32}, true);
            add(p + "mlp.gate_proj.weight", {512, 256});
            add(p + "mlp.up_proj.weight", {512, 256});
            add(p + "mlp.down_proj.weight", {256, 512});
        }
        if (merged) for (int i = 0; i < draftLayers; ++i) {
            const std::string p = "dspark.layers." + std::to_string(i) + ".";
            auto merge = [&](std::vector<std::string> names, std::string out) {
                int rows = 0;
                for (auto &name : names) rows += weight[p + name].dims[0];
                Data &w = weight[p + out];
                w.dataType = BFLOAT16; w.isModelWeight = true; w.Resize({rows, embed_dim}); w.Allocate();
                size_t offset = 0;
                for (auto &name : names) {
                    Data &part = weight[p + name];
                    memcpy(w.cpuData + offset, part.cpuData, part.GetBytes());
                    offset += part.GetBytes();
                    weight.weight.erase(p + name);
                }
            };
            merge({"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"}, "self_attn.mergeqkv.weight");
            merge({"mlp.gate_proj.weight", "mlp.up_proj.weight"}, "mlp.gateup_proj.weight");
        }
        // Match the model loader's GPU embedding placement in both paths.
        weight["model.embed_tokens.weight"].ToDevice(DataDevice::CUDA, std::vector<int>{owner});
        weight["dspark.mask_embedding"].ToDevice(DataDevice::CUDA, std::vector<int>{owner});
    }
    static void RunTPChecks(int ranks, bool mlpOnly, bool singleKV = false, int owner = 0) {
        const std::string tp = (mlpOnly ? "mlp:" : "") + std::to_string(ranks);
        DraftFixture reference(true, singleKV ? 1 : ranks, owner);
        DraftFixture candidate(true, singleKV ? 1 : ranks, owner);
        if (!FastllmInitNccl(std::vector<int>{0, 1})) throw std::runtime_error("target group init");
        const auto generation = FastllmGetNcclGeneration();
        auto bits = [](const Data &x) {
            size_t count = 1; for (int d : x.dims) count *= d;
            std::vector<uint16_t> out(count);
            if (cudaMemcpy(out.data(), x.cudaData, count * 2, cudaMemcpyDeviceToHost) != cudaSuccess)
                throw std::runtime_error("TP fixture read");
            return out;
        };
        auto floats = [&](const Data &x) {
            auto raw = bits(x); std::vector<float> out;
            for (auto value : raw) out.push_back(BFloat16BitsToFloat32(value));
            return out;
        };
        auto hidden = [&](int rows, int seed) {
            Data x(BFLOAT16, {1, rows, 256}); x.Allocate();
            for (int i = 0; i < x.Count(0); ++i)
                ((uint16_t *)x.cpuData)[i] = Float32ToBFloat16RNEBits(std::sin((i + seed) * .137f));
            x.ToDevice(DataDevice::CUDA, std::vector<int>{owner}); return x;
        };
        int checks = 0;
        for (int prefix : {3, 80, 249, 250, 1023, 1024, 32768, 3}) {
            auto a = reference.CreateDraftContext(), b = candidate.CreateDraftContext();
            int count = std::min(prefix, candidate.draftWindow - 1);
            Data h = hidden(count, prefix);
            setenv("FASTLLM_DSPARK_TP", "1", 1); reference.AppendDraftContext(h, prefix - count, *a);
            setenv("FASTLLM_DSPARK_TP", tp.c_str(), 1); candidate.AppendDraftContext(h, prefix - count, *b);
            if (FastllmGetNcclGeneration() != generation) throw std::runtime_error("draft replaced target group");
            for (int layer = 0; layer < candidate.draftLayers; ++layer) {
                const std::string p = "dspark.layers." + std::to_string(layer) + ".";
                for (const char *name : {"self_attn.mergeqkv.weight", "self_attn.o_proj.weight",
                                         "mlp.gateup_proj.weight", "mlp.down_proj.weight"}) {
                    Data &w = candidate.weight[p + name];
                    if (w.cpuData || w.cudaData || w.multiDeviceDatas.size() != (size_t)ranks)
                        throw std::runtime_error("Draft TP retained source or missing rank");
                    const bool replicated = mlpOnly && std::string(name).find("self_attn.") == 0;
                    size_t total = 0; std::vector<uint16_t> first;
                    for (auto &entry : w.multiDeviceDatas) {
                        Data &local = *entry.second; total += local.GetBytes();
                        if (replicated) {
                            if (local.dims != w.dims || !w.IsTensorParallelReplicated())
                                throw std::runtime_error("attention replica shape");
                            FastllmCudaSetDevice(entry.first); auto current = bits(local);
                            if (first.empty()) first = current;
                            else if (first != current) throw std::runtime_error("attention replicas differ");
                        }
                    }
                    if (total != w.GetBytes() * (replicated ? ranks : 1))
                        throw std::runtime_error("Draft TP physical weight bytes");
                }
            }
            FastllmCudaSetDevice(owner);
            for (int round = 0; round < 4; ++round) {
                SetCudaGraph(false); Data expected = reference.RunDraft(11 + round, *a);
                auto ef = floats(expected);
                SetCudaGraph(true); Data actual = candidate.RunDraft(11 + round, *b);
                auto af = floats(actual);
                RequireCloseLogits(af.data(), ef, "TP versus single draft");
                SetCudaGraph(false); Data eager = candidate.RunDraft(11 + round, *b);
                if (bits(actual) != bits(eager)) throw std::runtime_error("TP graph/eager bits differ");
                if (a->committed != b->committed) throw std::runtime_error("TP advanced committed prefix");
                for (int layer = 0; layer < candidate.draftLayers; ++layer) {
                    for (int part = 0; part < 2; ++part) {
                        const Data &ka = part ? a->kv[layer].second : a->kv[layer].first;
                        const Data &kb = part ? b->kv[layer].second : b->kv[layer].first;
                        auto av = floats(ka), bv = floats(kb); std::vector<float> wanted;
                        if (ka.dims[1] != kb.dims[1] || ka.dims[2] != (mlpOnly ? 1 : ranks) * kb.dims[2]) throw std::runtime_error("TP cache layout");
                        for (int row = 0; row < ka.dims[1]; ++row)
                            wanted.insert(wanted.end(), av.begin() + row * ka.dims[2], av.begin() + row * ka.dims[2] + kb.dims[2]);
                        RequireCloseLogits(bv.data(), wanted, "TP cache heads");
                    }
                }
                const int accepted = round == 3 ? 8 : round + 1;
                Data next = hidden(accepted, prefix + round + 91);
                setenv("FASTLLM_DSPARK_TP", "1", 1); reference.AppendDraftContext(next, a->committed, *a);
                setenv("FASTLLM_DSPARK_TP", tp.c_str(), 1); candidate.AppendDraftContext(next, b->committed, *b);
                if (cudaMemsetAsync(next.cudaData, 0, next.GetBytes(), cudaStreamPerThread) != cudaSuccess)
                    throw std::runtime_error("TP context input reuse");
                ++checks;
            }
            reference.idleDraftContext = std::move(a); candidate.idleDraftContext = std::move(b);
        }
        if (!failVerifyBegin && !failVerifyInstantiate && verifyGraphLaunches == 0) throw std::runtime_error("no TP graph replay");
        if ((failVerifyBegin || failVerifyInstantiate) && verifyGraphLaunches != 0) throw std::runtime_error("TP graph failure not rolled back");
        std::cout << "DRAFT " << (mlpOnly ? "MLP " : "") << "TP" << ranks << " PASS checks=" << checks << " relative=" << largestLogitRelative
                  << " maximum=" << largestLogitAbsolute << " captures=" << verifyGraphCaptures << " launches=" << verifyGraphLaunches << std::endl;
        unsetenv("FASTLLM_DSPARK_TP");
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
        if (argc > 2 && std::string(argv[2]) == "feature_graph") {
            SetCudaEmbedding(true);
            Fixture fixture(ranks, true, true);
            fixture.FeatureGraphChecks();
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "history") {
            bool graph = argc <= 3 || std::string(argv[3]) != "eager";
            SetCudaEmbedding(true);
            SetCudaGraph(graph);
            Fixture fixture(ranks, true, true);
            fixture.HistoryChecks(graph);
            return 0;
        }
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
        const std::string draftMode = argc > 2 ? argv[2] : "";
        if (draftMode == "draft_tp" || draftMode == "draft_mlp_tp" ||
            draftMode == "draft_mlp_tp_mqa" || draftMode == "draft_mlp_tp_offset") {
            const bool offset = draftMode == "draft_mlp_tp_offset";
            if (offset && devices < 3) return 77;
            SetCudaEmbedding(true); SetCudaGraph(true);
            failGraphDevice = offset ? 0 : ranks - 1;
            failVerifyBegin = argc > 3 && std::string(argv[3]) == "failbegin";
            failVerifyInstantiate = argc > 3 && std::string(argv[3]) == "failinstantiate";
            DraftFixture::RunTPChecks(ranks, draftMode != "draft_tp",
                                     draftMode == "draft_mlp_tp_mqa", offset ? 2 : 0);
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "draft_graph") {
            SetCudaEmbedding(true);
            failGraphDevice = 0;
            failVerifyBegin = argc > 3 && std::string(argv[3]) == "failbegin";
            failVerifyInstantiate = argc > 3 && std::string(argv[3]) == "failinstantiate";
            { DraftFixture fixture; fixture.Run(); }
            { DraftFixture fixture(true); fixture.Run(); }
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "batch_prefill") {
            SetCudaEmbedding(true);
            Fixture fixture(ranks, true, true);
            fixture.PackedPrefillChecks();
            return 0;
        }
        if (argc > 2 && std::string(argv[2]) == "batch") {
            bool graph = argc <= 3 || std::string(argv[3]) != "eager";
            SetCudaEmbedding(true); SetCudaGraph(graph);
            failGraphDevice = ranks - 1;
            failVerifyBegin = argc > 3 && std::string(argv[3]) == "failbegin";
            failVerifyInstantiate = argc > 3 && std::string(argv[3]) == "failinstantiate";
            Fixture fixture(ranks, true, true);
            fixture.BatchChecks(graph, argc <= 4 || std::string(argv[4]) != "separate");
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
