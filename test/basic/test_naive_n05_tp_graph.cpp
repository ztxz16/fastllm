#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1
#endif
// Standalone shared-library fixture (Linux): link against libfastllm_tools.so
// with -ldl -Wl,-export-dynamic and CUDA's per-thread default stream.
// Run eager, graph, failbegin and failinstantiate in separate processes:
//   test_naive_n05_tp_graph MODE OUTPUT.bin
// All four binary outputs must match; each mode checks metadata and launches.
#include "models/naive_n05_flash.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cmath>
#include <cuda_runtime_api.h>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <atomic>
#include <dlfcn.h>
static bool failBegin = false, failInstantiate = false;
static std::atomic<int> captures{0};
static std::atomic<size_t> downloadBytes{0};
extern "C" void FastllmCudaCopyFromDeviceToHost(void *dst, void *src, size_t bytes) {
    downloadBytes += bytes;
    static auto fn = (void (*)(void *, void *, size_t))dlsym(RTLD_NEXT, "FastllmCudaCopyFromDeviceToHost");
    fn(dst, src, bytes);
}
extern "C" bool FastllmCudaGraphInstantiate(void *graph, void **exec) {
    int device = 0;
    cudaGetDevice(&device);
    if (failInstantiate && device == 3) {
        *exec = nullptr;
        return false;
    }
    static auto fn = (bool (*)(void *, void **))dlsym(RTLD_NEXT, "FastllmCudaGraphInstantiate");
    return fn(graph, exec);
}
extern "C" bool FastllmCudaGraphBeginCapture() {
    ++captures;
    int device = 0;
    cudaGetDevice(&device);
    if (failBegin && device == 3)
        return false;
    static auto fn = (bool (*)())dlsym(RTLD_NEXT, "FastllmCudaGraphBeginCapture");
    return fn();
}
static std::atomic<int> launches{0};
extern "C" bool FastllmCudaGraphLaunch(void *exec) {
    ++launches;
    return cudaGraphLaunch((cudaGraphExec_t)exec, cudaStreamPerThread) == cudaSuccess;
}
using namespace fastllm;
class Fixture : public NaiveN05FlashModel {
  public:
    Fixture() {
        setenv("FASTLLM_TP", "cuda:0,1,2,3,4,5,6,7", 1);
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
                        {"moe_layer_freq", "[0,0]"},
                        {"scoring_func", "sigmoid"},
                        {"sliding_window", "8"},
                        {"index_top_k", "2048"},
                        {"indexer_activation_dtype", "bf16"},
                        {"n_routed_experts", "8"},
                        {"num_experts_per_tok", "2"},
                        {"max_position_embeddings", "8192"}};
        InitParams();
        SetSaveHistoryChat(false);
        maxBatch = 1;
        unsigned seed = 7;
        auto add = [&](std::string name, std::vector<int> dims, DataType type, float scale,
                       bool norm = false) {
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
            }
            add(p + ".mlp.gate_proj.weight", {1024, 256}, BFLOAT16, .02f);
            add(p + ".mlp.up_proj.weight", {1024, 256}, BFLOAT16, .02f);
            add(p + ".mlp.down_proj.weight", {256, 1024}, BFLOAT16, .02f);
        }
    }
    void Run(const char *path, bool graphs) {
        std::ofstream out(path, std::ios::binary);
        for (int prompt : {3, 125, 254, 510, 2046, 2049}) {
            std::vector<std::pair<Data, Data>> kv;
            for (int i = 0; i < 2; ++i)
                kv.emplace_back(Data(BFLOAT16), Data(BFLOAT16));
            int past = 0;
            for (int step = 0; step < 7; ++step) {
                int n = step ? 1 : prompt;
                std::vector<float> ids(n), positions(n);
                for (int i = 0; i < n; ++i) {
                    ids[i] = (past + i) * 3 % 256;
                    positions[i] = past + i;
                }
                Data input(FLOAT32, {1, n}, ids), pos(FLOAT32, {1, n}, positions);
                GenerationConfig cfg;
                cfg.output_logits = true;
                std::vector<float> logits;
                int before = launches.load();
                Forward(input, Data(), pos, kv, cfg, LastTokensManager(), &logits);
                out.write((const char *)logits.data(), logits.size() * sizeof(float));
                past += n;
                int delta = launches.load() - before;
                std::cout << "prompt=" << prompt << " step=" << step << " graph_launches=" << delta
                          << std::endl;
                if (graphs && step == 6 && delta != ((failBegin || failInstantiate) ? 0 : 8))
                    throw std::runtime_error("graph replay not reached");
                if (kv[0].first.dims[1] != past || kv[1].first.dims[1] != std::min(past, 7))
                    throw std::runtime_error("cache metadata mismatch");
            }
        }
        RunRequests(out, graphs);
        RunSelections(out);
    }
    void RunSelections(std::ofstream &out) {
        eos_token_id = 17; eos_token_ids = {17, 23};
        const std::vector<int> modes{0,0,0,0,1,1,1,0,0,0,2,2,2,3,3,3,4,4,4,5,5,5,
                                     6,6,6,2,2,2,7,7,7,0,0,0};
        for (int prompt : {3, 254, 2046}) {
            std::vector<int> expected;
            for (bool reference : {true, false}) {
                std::vector<std::pair<Data, Data>> kv;
                for (int i = 0; i < 2; ++i) kv.emplace_back(Data(BFLOAT16), Data(BFLOAT16));
                int past = 0;
                for (int step = 0; step < (int)modes.size(); ++step) {
                    int n = step ? 1 : prompt;
                    std::vector<float> ids(n), positions(n);
                    for (int i = 0; i < n; ++i) { ids[i] = (past + i) * 3 % 256; positions[i] = past + i; }
                    Data input(FLOAT32, {1, n}, ids), pos(FLOAT32, {1, n}, positions);
                    GenerationConfig cfg;
                    cfg.input_token_length = prompt;
                    cfg.output_logits = reference || modes[step] == 1;
                    if (modes[step] == 2) { cfg.top_k = 4; cfg.top_p = .8f; cfg.temperature = .7f; }
                    if (modes[step] == 3) { cfg.output_token_least = 100; cfg.stop_token_ids = {27}; }
                    if (modes[step] == 4) cfg.tool_call_allowed_token_ids = {17, 23};
                    if (modes[step] == 5) cfg.repeat_penalty = 1.2f;
                    if (modes[step] == 6) { cfg.top_k = 65; cfg.top_p = .9f; cfg.temperature = .7f; }
                    if (modes[step] == 7) { cfg.top_k = 4; cfg.top_p = .8f; cfg.temperature = .9f; }
                    LastTokensManager history(1, 64); history.units[0].Push(17); history.units[0].Push(17); history.units[0].Push(23);
                    std::vector<float> logits;
                    int before = launches.load(); downloadBytes = 0;
                    srand(2026 + step);
                    int token = Forward(input, Data(), pos, kv, cfg, history, &logits);
                    if (reference) expected.push_back(token);
                    else {
                        if (token != expected[step]) throw std::runtime_error("greedy/sampling fallback token mismatch");
                        if (cfg.output_logits != !logits.empty()) throw std::runtime_error("output_logits contract changed");
                        if (modes[step] == 0 && launches.load() - before == 8 && downloadBytes.load() != 64)
                            throw std::runtime_error("greedy graph still downloads full vocabulary");
                        if ((modes[step] == 2 || modes[step] == 7) && launches.load() - before == 8 && downloadBytes.load() != 256)
                            throw std::runtime_error("sampling graph still downloads full vocabulary");
                        if (modes[step] != 0 && modes[step] != 2 && modes[step] != 7 && downloadBytes.load() < 256 * sizeof(float))
                            throw std::runtime_error("sampler did not retain full logits");
                        out.write((const char *)&token, sizeof(token));
                    }
                    past += n;
                    if (kv[0].first.dims[1] != past || kv[1].first.dims[1] != std::min(past, 7))
                        throw std::runtime_error("greedy KV metadata mismatch");
                }
            }
        }
        std::cout << "TP GREEDY AND SAMPLING PASS 102 token comparisons" << std::endl;
    }
    void RunRequests(std::ofstream &out, bool graphs) {
        std::vector<void *> previous;
        int request = 0;
        auto run = [&](int prompt, bool cancelled, bool queued, bool reuse, bool execute = true) {
            std::lock_guard<std::mutex> guard(dictLocker);
            // Let the context dictionary own queued handles, including when a
            // failed assertion unwinds this fixture and shuts down the model.
            int pending = queued ? responseContextDict.CreateHandle() : -1;
            int handle = responseContextDict.CreateHandle();
            ResponseContext *context = responseContextDict.GetHandle(handle);
            context->Init(block_cnt, dataType, kvCacheDataType);
            context->generationConfig.input_token_length = prompt;
            context->generationConfig.output_token_limit = 16;
            context->generationConfig.output_logits = true;
            OnResponseContextCreated(context);
            auto &kv = context->pastKeyValues;
            bool restored = kv[0].first.multiDeviceData;
            if (restored != (graphs && reuse))
                throw std::runtime_error("unexpected request KV allocation reuse");
            for (auto &layer : kv)
                for (Data *root : {&layer.first, &layer.second})
                    if (!root->dims.empty() || !root->expansionDims.empty())
                        throw std::runtime_error("fresh request consumes a scheduler slot");
            if (restored) {
                size_t i = 0;
                for (auto &layer : kv)
                    for (Data *root : {&layer.first, &layer.second})
                        for (auto &item : root->multiDeviceDatas) {
                            if (item.second->cudaData != previous.at(i++) || item.second->dims[1] != 0)
                                throw std::runtime_error("KV ownership or logical length was not reset");
                        }
            }
            const bool expectReplay = restored && !failBegin && !failInstantiate;
            int beforeCaptures = captures.load(), past = 0;
            for (int step = 0; execute && step < 7; ++step) {
                int n = step ? 1 : prompt;
                std::vector<float> ids(n), positions(n);
                for (int i = 0; i < n; ++i) {
                    // Change every request's tokens, including an identical
                    // allocation shape, to catch accidental prefix reuse.
                    ids[i] = ((past + i) * 3 + request * 11) % 256;
                    positions[i] = past + i;
                }
                Data input(FLOAT32, {1, n}, ids), pos(FLOAT32, {1, n}, positions);
                std::vector<float> logits;
                int before = launches.load();
                Forward(input, Data(), pos, kv, context->generationConfig, LastTokensManager(), &logits);
                out.write((const char *)logits.data(), logits.size() * sizeof(float));
                past += n;
                if (step == 1 && expectReplay && launches.load() - before != 8)
                    throw std::runtime_error("second request did not replay on first decode");
                if (kv[0].first.dims[1] != past || kv[1].first.dims[1] != std::min(past, 7))
                    throw std::runtime_error("reused cache metadata mismatch");
            }
            if (execute && expectReplay && captures.load() != beforeCaptures)
                throw std::runtime_error("reused request unexpectedly recaptured");
            previous.clear();
            for (auto &layer : kv)
                for (Data *root : {&layer.first, &layer.second})
                    for (auto &item : root->multiDeviceDatas) previous.push_back(item.second->cudaData);
            context->isAbort = cancelled;
            context->isEnding = !cancelled;
            RemoveResponseContext(handle);
            if (queued) RemoveResponseContext(pending);
            std::cout << "request=" << request++ << " prompt=" << prompt << " restored=" << restored
                      << " captures=" << captures.load() - beforeCaptures << std::endl;
        };
        run(80, false, false, false);
        run(80, true, false, true);   // completed GPU work, cancelled response
        run(85, false, false, true);  // changed length in the same reservation
        run(300, false, false, false);
        run(300, false, false, true);
        run(300, false, true, false); // queued handle prevents ownership transfer
        run(300, false, false, false);
        run(300, true, false, true, false); // abort before any forward
        run(300, false, false, false);
    }
};
int main(int argc, char **argv) {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 8)
            return 77;
        setvbuf(stdout, nullptr, _IONBF, 0);
        if (argc != 3)
            return 2;
        bool graph = std::string(argv[1]) != "eager";
        failBegin = std::string(argv[1]) == "failbegin";
        failInstantiate = std::string(argv[1]) == "failinstantiate";
        SetCudaGraph(graph);
        SetCudaEmbedding(true);
        SetThreads(4);
        SetDeviceMap({{"cuda:0", 1}});
        FastllmCudaSetDevice(0);
        Fixture model;
        model.Run(argv[2], graph);
        std::cout << "TP GRAPH MODEL PASS" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
