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
