#include "model.h"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace fastllm;
namespace fs = std::filesystem;

int main() {
#ifdef _WIN32
    _putenv_s("FASTLLM_TP", "false");
    _putenv_s("FASTLLM_EP", "1");
#else
    setenv("FASTLLM_TP", "false", 1);
    setenv("FASTLLM_EP", "1", 1);
#endif
    SetDeviceMap({{"cpu", 1}});
    SetMoeDeviceMap({{"cpu", 1}});
    SetThreads(16);
    struct Directory {
        fs::path path = fs::temp_directory_path() / ("fastllm_hf_merge_" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        Directory() { fs::create_directories(path); }
        ~Directory() { std::error_code error; fs::remove_all(path, error); }
    } directory;
    constexpr int experts = 512, width = 8, elements = width * width;
    std::ofstream(directory.path / "config.json") << R"({
        "model_type":"mimo_v2", "enable_dsa":true, "attention_projection_layout":"split",
        "eos_token_id":2,
        "num_hidden_layers":1, "hidden_size":8, "num_attention_heads":4,
        "num_key_value_heads":2, "head_dim":8, "v_head_dim":4,
        "hybrid_layer_pattern":[0], "moe_layer_freq":[1], "scoring_func":"sigmoid",
        "n_routed_experts":512, "num_experts_per_tok":8, "routed_scaling_factor":null
    })";
    std::ostringstream header;
    std::vector<uint16_t> payload;
    header << "{";
    for (int expert = 0; expert < experts; ++expert) {
        for (int part = 0; part < 3; ++part) {
            const char *kind = part == 0 ? "gate" : part == 1 ? "up" : "down";
            const uint16_t value = (part == 0 ? 0x3f00 : part == 1 ? 0x4000 : 0x3e00) + expert % 64;
            size_t offset = payload.size() * 2;
            if (expert || part) header << ",";
            header << "\"model.layers.0.mlp.experts." << expert << "." << kind
                   << "_proj.weight\":{\"dtype\":\"BF16\",\"shape\":[8,8],\"data_offsets\":["
                   << offset << "," << offset + elements * 2 << "]}";
            payload.insert(payload.end(), elements, value);
        }
    }
    header << "}";
    std::string metadata = header.str();
    while (metadata.size() % 8) metadata.push_back(' ');
    {
        std::ofstream file(directory.path / "model.safetensors", std::ios::binary);
        uint64_t length = metadata.size();
        file.write(reinterpret_cast<const char *>(&length), sizeof(length));
        file.write(metadata.data(), metadata.size());
        file.write(reinterpret_cast<const char *>(payload.data()), payload.size() * 2);
        if (!file) throw std::runtime_error("failed to write merge fixture");
    }
    // Thousands of independent gate/up merges race at the same time. Check
    // every payload, not just whether model loading returns successfully.
    for (int repeat = 0; repeat < 12; ++repeat) {
        auto model = CreateLLMModelFromHF(directory.path.string(), BFLOAT16, -1, true);
        for (int expert = 0; expert < experts; ++expert) {
            std::string prefix = "model.layers.0.mlp.experts." + std::to_string(expert) + ".";
            const auto &merged = model->weight.weight.at(prefix + "gateup_proj.weight");
            if (merged.dims != std::vector<int>({width * 2, width}) ||
                merged.dataType != BFLOAT16 || !merged.cpuData)
                throw std::runtime_error("missing or invalid merged expert " + prefix);
            auto values = reinterpret_cast<const uint16_t *>(merged.cpuData);
            for (int i = 0; i < elements * 2; ++i)
                if (values[i] != (i < elements ? 0x3f00 : 0x4000) + expert % 64)
                    throw std::runtime_error("corrupt merged payload " + prefix);
            const auto &down = model->weight.weight.at(prefix + "down_proj.weight");
            if (down.dims != std::vector<int>({width, width}) ||
                down.dataType != BFLOAT16 || !down.cpuData)
                throw std::runtime_error("missing or invalid unmerged expert " + prefix);
            auto downValues = reinterpret_cast<const uint16_t *>(down.cpuData);
            for (int i = 0; i < elements; ++i)
                if (downValues[i] != 0x3e00 + expert % 64)
                    throw std::runtime_error("corrupt unmerged payload " + prefix);
            if (model->weight.weight.count(prefix + "gate_proj.weight") ||
                model->weight.weight.count(prefix + "up_proj.weight"))
                throw std::runtime_error("merge inputs were not released");
        }
    }
    std::cout << "PASS parallel HF weight merging" << std::endl;
}
