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

static void CheckPrefillDraftLoad(const fs::path &target) {
    const auto draft = target / "draft";
    fs::create_directory(draft);
    std::ofstream(draft / "config.json") << R"({
        "architectures":["DSparkDraftModel"],"num_hidden_layers":1,"num_target_layers":1,
        "block_size":3,"sliding_window":16,"hidden_size":8,"num_attention_heads":2,
        "num_key_value_heads":1,"head_dim":4,"vocab_size":16,"rms_norm_eps":0.00001,
        "rope_parameters":{"rope_theta":10000},"layer_types":["sliding_attention"],
        "markov_head_type":"vanilla","dflash_config":{"use_mask_embedding":true,"target_layer_ids":[0]}
    })";
    const std::string prefix = "layers.0.self_attn.";
    json11::Json::object header, index;
    std::vector<uint16_t> payload;
    auto tensor = [&](const std::string &name, std::vector<int> shape, uint16_t value) {
        int count = 1;
        for (int dim : shape) count *= dim;
        const int start = int(payload.size()) * 2;
        payload.insert(payload.end(), count, value);
        header[name] = json11::Json::object{{"dtype", "BF16"}, {"shape", shape},
            {"data_offsets", std::vector<int>{start, int(payload.size()) * 2}}};
        index[name] = "mixed.safetensors";
    };
    tensor("fc.weight", {8, 8}, 0);
    tensor("hidden_norm.weight", {8}, 0x3f80);
    tensor(prefix + "k_proj.weight", {4, 8}, 0x3f00);
    tensor(prefix + "v_proj.weight", {4, 8}, 0x4000);
    tensor(prefix + "k_norm.weight", {4}, 0x3f80);
    tensor(prefix + "q_proj.weight", {8, 8}, 0);
    index["layers.0.mlp.experts.0.gate_proj.weight"] = "absent_experts.safetensors";
    std::ofstream(draft / "model.safetensors.index.json")
        << json11::Json(json11::Json::object{{"weight_map", index}}).dump();
    std::string metadata = json11::Json(header).dump();
    while (metadata.size() % 8) metadata.push_back(' ');
    {
        std::ofstream file(draft / "mixed.safetensors", std::ios::binary);
        uint64_t size = metadata.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        file.write(metadata.data(), size);
        file.write(reinterpret_cast<const char*>(payload.data()), payload.size() * 2);
    }
#ifdef _WIN32
    _putenv_s("FASTLLM_DSPARK_MODEL_PATH", draft.string().c_str());
#else
    setenv("FASTLLM_DSPARK_MODEL_PATH", draft.string().c_str(), 1);
#endif
    auto model = CreateLLMModelFromHF(target.string(), BFLOAT16, -1, true, "", "", false,
            false, FLOAT32, -1, "", {}, true);
    const auto &weights = model->weight.weight;
    const std::set<std::string> expected{"dspark.fc.weight", "dspark.hidden_norm.weight",
            "dspark." + prefix + "mergekv.weight", "dspark." + prefix + "k_norm.weight"};
    std::set<std::string> actual;
    for (const auto &entry : weights)
        if (entry.first.rfind("dspark.", 0) == 0) actual.insert(entry.first);
    const auto &kv = weights.at("dspark." + prefix + "mergekv.weight");
    if (actual != expected || kv.dims != std::vector<int>({8, 8}) || !kv.cpuData ||
        !weights.count("model.layers.0.mlp.experts.0.gateup_proj.weight"))
        throw std::runtime_error("incorrect prefill draft weights or missing target experts");
    const auto *values = reinterpret_cast<const uint16_t*>(kv.cpuData);
    for (int i = 0; i < 64; ++i)
        if (values[i] != (i < 32 ? 0x3f00 : 0x4000))
            throw std::runtime_error("prefill draft KV projection row order changed");
}

int main(int argc, char **argv) {
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
    std::ofstream(directory.path / "tokenizer_config.json") << R"({"tokenizer_class":"PreTrainedTokenizerFast"})";
    std::ofstream(directory.path / "tokenizer.json") << R"({"model":{"vocab":{"x":0}}})";
    std::ofstream(directory.path / "config.json") << R"({
        "model_type":"mimo_v2", "enable_dsa":true, "attention_projection_layout":"split",
        "eos_token_id":2,"vocab_size":16,
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
#if defined(USE_NUMAS) && !defined(_WIN32)
    if (argc == 2 && std::string(argv[1]) == "--host-moe") {
        std::ofstream(directory.path / "config.json") << R"({
            "model_type":"glm4_moe","eos_token_id":2,"vocab_size":16,
            "num_hidden_layers":1,"hidden_size":8,"head_dim":4,"num_attention_heads":2,
            "num_key_value_heads":2,"n_routed_experts":512,"num_experts_per_tok":2,
            "n_shared_experts":1,"max_position_embeddings":16,"rope_theta":10000
        })";
        SetMoeDeviceMap({{"numa", 1}});
        setenv("FASTLLM_CUDA_SERVING_WARMUP", "1", 1);
        auto model = CreateLLMModelFromHF(directory.path.string(), BFLOAT16, -1, true);
        const auto &weights = model->weight.weight;
        for (int expert = 0; expert < experts; ++expert) {
            const auto prefix = "model.layers.0.mlp.experts." + std::to_string(expert) + ".";
            const auto &merged = weights.at(prefix + "gateup_proj.weight");
            const auto &down = weights.at(prefix + "down_proj.weight");
            const int nodes = merged.numasData.size();
            if (!nodes || width % nodes || down.numasData.size() != size_t(nodes) ||
                merged.cpuData || down.cpuData || merged.dataType != BFLOAT16 || down.dataType != BFLOAT16 ||
                merged.dims != std::vector<int>{2 * width, width} || down.dims != std::vector<int>{width, width} ||
                weights.count(prefix + "gate_proj.weight") || weights.count(prefix + "up_proj.weight"))
                throw std::runtime_error("incorrect host expert publication: " + prefix);
            for (int node = 0; node < nodes; ++node) {
                auto *gateUp = reinterpret_cast<const uint16_t *>(merged.numasData[node]);
                auto *downRows = reinterpret_cast<const uint16_t *>(down.numasData[node]);
                for (int i = 0; i < 2 * elements / nodes; ++i)
                    if (gateUp[i] != ((i / width) % 2 ? 0x4000 : 0x3f00) + expert % 64)
                        throw std::runtime_error("incorrect host gate/up row: " + prefix);
                for (int i = 0; i < elements / nodes; ++i)
                    if (downRows[i] != 0x3e00 + expert % 64)
                        throw std::runtime_error("incorrect host down row: " + prefix);
            }
        }
        std::cout << "PASS parallel host import of 512 routed experts" << std::endl;
        return 0;
    }
#endif
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
    CheckPrefillDraftLoad(directory.path);
    std::cout << "PASS parallel HF weight merging" << std::endl;
}
