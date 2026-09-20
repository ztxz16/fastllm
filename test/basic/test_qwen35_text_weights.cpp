#include "model.h"
#include "models/qwen3_5.h"
#include "executor.h"
#include "json11.hpp"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace fastllm;
using Json = json11::Json;
namespace fs = std::filesystem;

namespace {
void Check(bool ok, const std::string &message) {
    if (!ok) throw std::runtime_error(message);
}

Json ReadJson(const fs::path &path) {
    std::ifstream input(path);
    Check(input.good(), "Missing JSON: " + path.string());
    std::string error;
    auto result = Json::parse(std::string(std::istreambuf_iterator<char>(input), {}), error);
    Check(error.empty(), "Invalid JSON: " + error);
    return result;
}

void AddConfig(basellm &model, const Json &value, const std::string &prefix = "") {
    for (const auto &item : value.object_items()) {
        const std::string key = prefix + item.first;
        model.weight.AddDict(key, item.second.is_string() ?
            item.second.string_value() : item.second.dump());
        if (item.second.is_object()) AddConfig(model, item.second, key + ".");
    }
}

void TestFactoryAndMapping() {
    auto model = CreateEmptyLLMModel("qwen3_5_text");
    Check(model->model_type == "qwen3_5" && model->model_struct == "qwen3_5",
          "Text checkpoint missed the Qwen3.5 runtime");
    const std::string prefix = "model.language_model.";
    const std::vector<std::string> names = {
        "model.embed_tokens.weight", "model.norm.weight",
        "model.layers.0.linear_attn.out_proj.weight",
        "model.layers.0.linear_attn.out_proj.weight_scale_inv",
        "mtp.layers.0.self_attn.q_proj.weight", "lm_head.weight",
        "model.visual.blocks.0.attn.proj.weight"};
    auto mapped = model->GetTensorMap(names);
    Check(mapped.size() == names.size(), "Mapping invented checkpoint keys");
    Check(mapped.at(names[0]).at(0) ==
          std::make_pair(prefix + "embed_tokens.weight", DATA_AUTO_EMBEDDING),
          "Text embedding lost source dtype handling");
    Check(mapped.at(names[1]).at(0) ==
          std::make_pair(prefix + "norm.weight", DATA_AUTO_NONE),
          "Text norm mapping changed precision policy");
    Check(mapped.at(names[2]).at(0) ==
          std::make_pair(prefix + "layers.0.linear_attn.out_proj.weight", DATA_AUTO_LINEAR),
          "Text linear weight missed FP8 auto dispatch");
    for (size_t i = 4; i < names.size(); ++i)
        Check(mapped.at(names[i]).at(0).first == names[i], "Non-text namespace was changed");

    auto legacy = CreateEmptyLLMModel("qwen3_5");
    const std::string legacyWeight = prefix + "layers.0.self_attn.q_proj.weight";
    Check(legacy->GetTensorMap({legacyWeight}).at(legacyWeight).at(0) ==
          std::make_pair(legacyWeight, DATA_AUTO_LINEAR), "VL checkpoint mapping regressed");
}

struct Fixture {
    fs::path path = fs::temp_directory_path() /
        ("fastllm-qwen35-text-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    Fixture() { Check(fs::create_directory(path), "Fixture directory already exists"); }
    ~Fixture() {
        std::error_code error;
        for (auto file : {"config.json", "model.safetensors"})
            fs::remove(path / file, error);
        fs::remove(path, error);
    }
};

void TestFp8Load(bool textOnly) {
    Fixture fixture;
    Json::object textConfig{
        {"hidden_size", 256}, {"num_hidden_layers", 1}, {"num_attention_heads", 2},
        {"num_key_value_heads", 1}, {"head_dim", 128}, {"max_position_embeddings", 262144},
        {"eos_token_id", 2}, {"vocab_size", 4},
        {"linear_num_key_heads", 1}, {"linear_num_value_heads", 2},
        {"rope_parameters", Json::object{{"rope_type", "default"}, {"rope_theta", 10000000},
                                        {"partial_rotary_factor", 0.25}}}};
    Json::object config = textOnly ? textConfig :
        Json::object{{"text_config", textConfig}, {"eos_token_id", 2}};
    config["model_type"] = textOnly ? "qwen3_5_text" : "qwen3_5";
    config["quantization_config"] = Json::object{
        {"quant_method", "fp8"}, {"fmt", "e4m3"}, {"weight_block_size", Json::array{128, 128}}};
    std::ofstream(fixture.path / "config.json") << Json(config).dump();

    const std::string inputPrefix = textOnly ? "model." : "model.language_model.";
    const std::string outputPrefix = "model.language_model.";
    Json::object header;
    std::vector<uint8_t> bytes;
    auto tensor = [&](const std::string &name, const std::string &dtype,
                      std::vector<int> shape, const void *data, size_t count) {
        size_t start = bytes.size();
        const auto *p = static_cast<const uint8_t *>(data);
        bytes.insert(bytes.end(), p, p + count);
        header[name] = Json::object{{"dtype", dtype}, {"shape", shape},
                                   {"data_offsets", Json::array{(int)start, (int)bytes.size()}}};
    };
    std::vector<uint8_t> fp8(256 * 256);
    for (size_t i = 0; i < fp8.size(); ++i) fp8[i] = 0x20 + i % 64;
    const float scales[] = {0.25f, 0.5f, 0.75f, 1.0f};
    const std::string projection = "layers.0.linear_attn.out_proj.weight";
    tensor(inputPrefix + projection, "F8_E4M3", {256, 256}, fp8.data(), fp8.size());
    tensor(inputPrefix + projection + "_scale_inv", "F32", {2, 2}, scales, sizeof(scales));
    std::vector<uint16_t> embedding(4 * 256, 0x3f80), ba(2 * 256, 0x3f80);
    std::vector<float> norm(256, 0.0f);
    tensor(inputPrefix + "embed_tokens.weight", "BF16", {4, 256},
           embedding.data(), embedding.size() * sizeof(uint16_t));
    tensor(inputPrefix + "norm.weight", "F32", {256}, norm.data(), norm.size() * sizeof(float));
    tensor(inputPrefix + "layers.0.linear_attn.in_proj_a.weight", "BF16", {2, 256},
           ba.data(), ba.size() * sizeof(uint16_t));
    std::string metadata = Json(header).dump();
    metadata.append((8 - metadata.size() % 8) % 8, ' ');
    uint64_t length = metadata.size();
    {
        std::ofstream output(fixture.path / "model.safetensors", std::ios::binary);
        output.write(reinterpret_cast<const char *>(&length), sizeof(length));
        output.write(metadata.data(), metadata.size());
        output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    }

    // Real HF loader, tiny synthetic tensors, CPU only; no forward or warmup.
    auto model = CreateLLMModelFromHF(fixture.path.string(), DATA_AUTO_SOURCE, -1, true);
    auto &weight = model->weight[outputPrefix + projection];
    Check(weight.dataType == FP8_E4M3 && weight.blockK == 128 && weight.blockM == 128,
          "FP8 Block128 format changed");
    Check(weight.cpuData != nullptr && weight.cudaData == nullptr &&
          weight.GetBytes() == fp8.size() &&
          std::memcmp(weight.cpuData, fp8.data(), fp8.size()) == 0,
          "FP8 payload changed or left CPU");
    Check(weight.scales == std::vector<float>(scales, scales + 4), "Source FP8 scales mismatched");
    auto &embed = model->weight[outputPrefix + "embed_tokens.weight"];
    Check(embed.dataType == BFLOAT16 && embed.GetBytes() == embedding.size() * 2 &&
          std::memcmp(embed.cpuData, embedding.data(), embed.GetBytes()) == 0,
          "BF16 embedding expanded or changed");
    auto &gate = model->weight[outputPrefix + "layers.0.linear_attn.in_proj_a.weight"];
    Check(gate.dataType == FLOAT16 && ((uint16_t *)gate.cpuData)[0] == 0x3c00,
          "Unquantized GDN input projection used FP8 scale handling");
    Check(model->max_positions == 262144 && model->contextPlan.declaredLength == 262144 &&
          !model->contextPlan.configured, "Native context was overridden");
    Check(model->block_cnt == 1 && model->embed_dim == 256, "Flat or nested config lost");
}

class Qwen35ConfigProbe : public Qwen3_5Model {
public:
    int KvHeads() const { return num_key_value_heads; }
};

void CheckCheckpointMetadata(const fs::path &path) {
    auto config = ReadJson(path / "config.json");
    auto index = ReadJson(path / "model.safetensors.index.json");
    Check(config["model_type"].string_value() == "qwen3_5_text",
          "Target checkpoint is not the text-only model");
    // Factory dispatch is covered above. Inspect the derived runtime field:
    // basellm has a separate num_key_value_heads member.
    auto model = std::make_unique<Qwen35ConfigProbe>();
    AddConfig(*model, config);
    model->ConfigureContext({});
    model->InitParams();
    Check(model->block_cnt == 64 && model->embed_dim == 5120 &&
          model->num_attention_heads == 24 && model->KvHeads() == 4 &&
          model->max_positions == 262144 && !model->contextPlan.configured,
          "Target checkpoint geometry was not preserved");
    std::vector<std::string> names;
    for (const auto &item : index["weight_map"].object_items()) {
        names.push_back(item.first);
        Check(fs::is_regular_file(path / item.second.string_value()), "Missing checkpoint shard");
    }
    auto mapped = model->GetTensorMap(names);
    Check(mapped.size() == names.size(), "Target index mapping is incomplete");
    size_t aliases = 0;
    for (const auto &name : names) {
        const auto &entry = mapped.at(name).at(0);
        if (name.rfind("model.", 0) == 0) {
            Check(entry.first == "model.language_model." + name.substr(6),
                  "Target tensor namespace mismatch: " + name);
            ++aliases;
        }
        if (name.size() > 7 && name.substr(name.size() - 7) == ".weight" &&
            index["weight_map"].object_items().count(name + "_scale_inv")) {
            Check(entry.second == DATA_AUTO_LINEAR, "Scaled target tensor missed FP8 dispatch");
        }
    }
    std::cout << "PASS checkpoint metadata: " << names.size() << " tensors, "
              << aliases << " aliases, 64 layers, native context 262144\n";
}
}

int main(int argc, char **argv) {
    try {
        SetThreads(2);
        SetDeviceMap({{"cpu", 1}});
        SetMoeDeviceMap({{"cpu", 1}});
        static_cast<Executor *>(GetExecutor())->SetFirstDevice("cpu");
        TestFactoryAndMapping();
        TestFp8Load(true);
        TestFp8Load(false);
        if (argc == 2) CheckCheckpointMetadata(argv[1]);
        std::cout << "ALL_PASS: Qwen3.5 text/VL mapping and CPU FP8 loader\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
