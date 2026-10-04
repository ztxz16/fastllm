#include "models/naive_n05_flash.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cmath>
#include <cuda_runtime_api.h>
#include <iostream>
#include <stdexcept>
using namespace fastllm;
class Fixture : public NaiveN05FlashModel {
  public:
    int ranks;
    explicit Fixture(int ranks) : ranks(ranks) {
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
};
int main(int argc, char **argv) {
    try {
        int ranks = argc > 1 ? std::stoi(argv[1]) : 8;
        if (ranks != 2 && ranks != 4 && ranks != 8)
            return 2;
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < ranks)
            return 77;
        SetThreads(4);
        SetDeviceMap({{"cuda:0", 1}});
        FastllmCudaSetDevice(0);
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
