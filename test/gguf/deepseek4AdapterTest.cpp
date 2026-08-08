#include "deepseekv4.h"
#include "gguf.h"
#include "model.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
    using fastllm::GGUFWeightReplaceRule;

    template <typename T>
    void WritePod(std::ofstream &out, const T &value) {
        out.write(reinterpret_cast<const char*>(&value), sizeof(T));
        assert(out.good());
    }

    void WriteString(std::ofstream &out, const std::string &value) {
        WritePod(out, (uint64_t)value.size());
        out.write(value.data(), (std::streamsize)value.size());
        assert(out.good());
    }

    void WriteMetadataKey(std::ofstream &out, const std::string &key,
                          gguf_type type) {
        WriteString(out, key);
        WritePod(out, (uint32_t)type);
    }

    void WriteMetadataFixture(const std::filesystem::path &path) {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        assert(out.is_open());

        WritePod(out, (uint32_t)0x46554747);
        WritePod(out, (uint32_t)3);
        WritePod(out, (uint64_t)1);
        WritePod(out, (uint64_t)15);

        WriteMetadataKey(out, "general.alignment", GGUF_TYPE_UINT32);
        WritePod(out, (uint32_t)64);

        WriteMetadataKey(out, "float_array", GGUF_TYPE_ARRAY);
        WritePod(out, (uint32_t)GGUF_TYPE_FLOAT32);
        WritePod(out, (uint64_t)3);
        WritePod(out, 10.0f);
        WritePod(out, 11.5f);
        WritePod(out, -2.0f);

        WriteMetadataKey(out, "after_array", GGUF_TYPE_INT32);
        WritePod(out, (int32_t)123456);
        WriteMetadataKey(out, "u8", GGUF_TYPE_UINT8);
        WritePod(out, (uint8_t)250);
        WriteMetadataKey(out, "i8", GGUF_TYPE_INT8);
        WritePod(out, (int8_t)-7);
        WriteMetadataKey(out, "u16", GGUF_TYPE_UINT16);
        WritePod(out, (uint16_t)60000);
        WriteMetadataKey(out, "i16", GGUF_TYPE_INT16);
        WritePod(out, (int16_t)-1234);
        WriteMetadataKey(out, "u32", GGUF_TYPE_UINT32);
        WritePod(out, (uint32_t)4000000000U);
        WriteMetadataKey(out, "f32", GGUF_TYPE_FLOAT32);
        WritePod(out, 1.25f);
        WriteMetadataKey(out, "bool", GGUF_TYPE_BOOL);
        WritePod(out, (uint8_t)1);
        WriteMetadataKey(out, "string", GGUF_TYPE_STRING);
        WriteString(out, "deepseek4");
        WriteMetadataKey(out, "u64", GGUF_TYPE_UINT64);
        WritePod(out, (uint64_t)1234567890123ULL);
        WriteMetadataKey(out, "i64", GGUF_TYPE_INT64);
        WritePod(out, (int64_t)-1234567890123LL);
        WriteMetadataKey(out, "f64", GGUF_TYPE_FLOAT64);
        WritePod(out, 3.25);

        WriteMetadataKey(out, "string_array", GGUF_TYPE_ARRAY);
        WritePod(out, (uint32_t)GGUF_TYPE_STRING);
        WritePod(out, (uint64_t)2);
        WriteString(out, "a");
        WriteString(out, "bc");

        WriteString(out, "blk.0.ffn_gate_tid2eid.weight");
        WritePod(out, (uint32_t)1);
        WritePod(out, (int64_t)4);
        WritePod(out, (uint32_t)GGML_TYPE_I32);
        WritePod(out, (uint64_t)0);

        const size_t position = (size_t)out.tellp();
        const size_t padding =
            (64 - position % 64) % 64;
        std::vector<char> zeros(padding, 0);
        out.write(zeros.data(), (std::streamsize)zeros.size());
        WritePod(out, (int32_t)3);
        WritePod(out, (int32_t)1);
        WritePod(out, (int32_t)2);
        WritePod(out, (int32_t)0);
    }

    void WriteUnsupportedTypeFixture(const std::filesystem::path &path) {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        assert(out.is_open());
        WritePod(out, (uint32_t)0x46554747);
        WritePod(out, (uint32_t)3);
        WritePod(out, (uint64_t)1);
        WritePod(out, (uint64_t)0);
        WriteString(out, "blk.0.ffn_down_exps.weight");
        WritePod(out, (uint32_t)3);
        WritePod(out, (int64_t)2048);
        WritePod(out, (int64_t)4096);
        WritePod(out, (int64_t)256);
        WritePod(out, (uint32_t)40);
        WritePod(out, (uint64_t)0);
    }

    void WriteNestedArrayFixture(const std::filesystem::path &path) {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        assert(out.is_open());
        WritePod(out, (uint32_t)0x46554747);
        WritePod(out, (uint32_t)3);
        WritePod(out, (uint64_t)0);
        WritePod(out, (uint64_t)1);
        WriteMetadataKey(out, "nested", GGUF_TYPE_ARRAY);
        WritePod(out, (uint32_t)GGUF_TYPE_ARRAY);
        WritePod(out, (uint64_t)1);
    }

    void WriteInvalidAlignmentFixture(const std::filesystem::path &path) {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        assert(out.is_open());
        WritePod(out, (uint32_t)0x46554747);
        WritePod(out, (uint32_t)3);
        WritePod(out, (uint64_t)0);
        WritePod(out, (uint64_t)1);
        WriteMetadataKey(out, "general.alignment", GGUF_TYPE_UINT32);
        WritePod(out, (uint32_t)3);
    }

    void WriteTooManyDimensionsFixture(const std::filesystem::path &path) {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        assert(out.is_open());
        WritePod(out, (uint32_t)0x46554747);
        WritePod(out, (uint32_t)3);
        WritePod(out, (uint64_t)1);
        WritePod(out, (uint64_t)0);
        WriteString(out, "output.weight");
        WritePod(out, (uint32_t)(GGML_MAX_DIMS + 1));
    }

    void WriteOverflowDimensionsFixture(const std::filesystem::path &path) {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        assert(out.is_open());
        WritePod(out, (uint32_t)0x46554747);
        WritePod(out, (uint32_t)3);
        WritePod(out, (uint64_t)1);
        WritePod(out, (uint64_t)0);
        WriteString(out, "output.weight");
        WritePod(out, (uint32_t)GGML_MAX_DIMS);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            WritePod(out, (int64_t)std::numeric_limits<int>::max());
        }
        WritePod(out, (uint32_t)GGML_TYPE_F32);
        WritePod(out, (uint64_t)0);
    }

    struct ScopedFixture {
        enum class Kind {
            Metadata,
            UnsupportedType,
            NestedArray,
            InvalidAlignment,
            TooManyDimensions,
            OverflowDimensions,
        };

        std::filesystem::path path;

        explicit ScopedFixture(Kind kind = Kind::Metadata) {
            const auto suffix = std::chrono::high_resolution_clock::now()
                                    .time_since_epoch().count();
            path = std::filesystem::temp_directory_path() /
                   ("fastllm-deepseek4-" + std::to_string(suffix) + ".gguf");
            switch (kind) {
                case Kind::Metadata:
                    WriteMetadataFixture(path);
                    break;
                case Kind::UnsupportedType:
                    WriteUnsupportedTypeFixture(path);
                    break;
                case Kind::NestedArray:
                    WriteNestedArrayFixture(path);
                    break;
                case Kind::InvalidAlignment:
                    WriteInvalidAlignmentFixture(path);
                    break;
                case Kind::TooManyDimensions:
                    WriteTooManyDimensionsFixture(path);
                    break;
                case Kind::OverflowDimensions:
                    WriteOverflowDimensionsFixture(path);
                    break;
            }
        }

        ~ScopedFixture() {
            std::error_code error;
            std::filesystem::remove(path, error);
        }
    };

    const GGUFWeightReplaceRule &FindOnlyRule(
            const std::vector<GGUFWeightReplaceRule> &rules,
            const std::string &source) {
        const GGUFWeightReplaceRule *found = nullptr;
        int matches = 0;
        for (const auto &rule : rules) {
            if (std::regex_search(source, rule.pattern)) {
                found = &rule;
                matches++;
            }
        }
        assert(matches == 1);
        assert(found != nullptr);
        return *found;
    }

    std::string ConvertDirect(
            const std::vector<GGUFWeightReplaceRule> &rules,
            const std::string &source) {
        const auto &rule = FindOnlyRule(rules, source);
        assert(rule.type != GGUFWeightReplaceRule::GGUFWeightReplacePacked);
        assert(rule.names.size() == 1);
        return std::regex_replace(source, rule.pattern, rule.names[0]);
    }

    void ExpectDirect(const std::vector<GGUFWeightReplaceRule> &rules,
                      const std::string &source,
                      const std::string &target) {
        assert(ConvertDirect(rules, source) == target);
    }

    std::vector<std::string> BuildCandidateTensorNames() {
        std::vector<std::string> names = {
            "token_embd.weight",
            "output.weight",
            "output_norm.weight",
            "output_hc_base.weight",
            "output_hc_fn.weight",
            "output_hc_scale.weight",
        };
        const std::vector<std::string> everyLayer = {
            "attn_kv.weight",
            "attn_kv_a_norm.weight",
            "attn_norm.weight",
            "attn_output_a.weight",
            "attn_output_b.weight",
            "attn_q_a.weight",
            "attn_q_a_norm.weight",
            "attn_q_b.weight",
            "attn_sinks.weight",
            "ffn_down_exps.weight",
            "ffn_down_shexp.weight",
            "ffn_gate_exps.weight",
            "ffn_gate_inp.weight",
            "ffn_gate_shexp.weight",
            "ffn_norm.weight",
            "ffn_up_exps.weight",
            "ffn_up_shexp.weight",
            "hc_attn_base.weight",
            "hc_attn_fn.weight",
            "hc_attn_scale.weight",
            "hc_ffn_base.weight",
            "hc_ffn_fn.weight",
            "hc_ffn_scale.weight",
        };
        const std::vector<std::string> compressor = {
            "attn_compressor_ape.weight",
            "attn_compressor_gate.weight",
            "attn_compressor_kv.weight",
            "attn_compressor_norm.weight",
        };
        const std::vector<std::string> indexer = {
            "indexer.attn_q_b.weight",
            "indexer.proj.weight",
            "indexer_compressor_ape.weight",
            "indexer_compressor_gate.weight",
            "indexer_compressor_kv.weight",
            "indexer_compressor_norm.weight",
        };

        for (int layer = 0; layer < 43; layer++) {
            const std::string prefix = "blk." + std::to_string(layer) + ".";
            for (const auto &suffix : everyLayer) {
                names.push_back(prefix + suffix);
            }
            if (layer >= 2) {
                for (const auto &suffix : compressor) {
                    names.push_back(prefix + suffix);
                }
            }
            if (layer < 3) {
                names.push_back(prefix + "ffn_gate_tid2eid.weight");
            } else {
                names.push_back(prefix + "exp_probs_b.bias");
            }
            if (layer >= 2 && layer % 2 == 0) {
                for (const auto &suffix : indexer) {
                    names.push_back(prefix + suffix);
                }
            }
        }
        return names;
    }

    void TestRealGGUFMetadataParsing() {
        ScopedFixture fixture;
        json11::Json config;
        fastllm::ReadGGUFMetaData(fixture.path.string(), config);
        const auto &params = config["params"];

        assert(params["general.alignment"].int_value() == 64);
        const auto &floatArray = params["float_array"].array_items();
        assert(floatArray.size() == 3);
        assert(std::fabs(floatArray[0].number_value() - 10.0) < 1e-6);
        assert(std::fabs(floatArray[1].number_value() - 11.5) < 1e-6);
        assert(std::fabs(floatArray[2].number_value() + 2.0) < 1e-6);
        assert(params["after_array"].int_value() == 123456);
        assert(params["u8"].int_value() == 250);
        assert(params["i8"].int_value() == -7);
        assert(params["u16"].int_value() == 60000);
        assert(params["i16"].int_value() == -1234);
        assert(params["u32"].number_value() == 4000000000.0);
        assert(std::fabs(params["f32"].number_value() - 1.25) < 1e-6);
        assert(params["bool"].bool_value());
        assert(params["string"].string_value() == "deepseek4");
        assert(params["u64"].number_value() == 1234567890123.0);
        assert(params["i64"].number_value() == -1234567890123.0);
        assert(params["f64"].number_value() == 3.25);
        const auto &stringArray = params["string_array"].array_items();
        assert(stringArray.size() == 2);
        assert(stringArray[0].string_value() == "a");
        assert(stringArray[1].string_value() == "bc");

        std::vector<fastllm::ReadGGUFTask> tasks;
        fastllm::AppendGGUFTasks("deepseek_v4", fixture.path.string(), tasks);
        assert(tasks.size() == 1);
        assert(tasks[0].name == "layers.0.ffn.gate.tid2eid");
        assert(tasks[0].tensor.type == GGML_TYPE_I32);
        assert(tasks[0].tensor.dims == std::vector<int>({4}));
        assert(tasks[0].offset % 64 == 0);

        fastllm::Data routeTable;
        fastllm::WeightImportGGUFTensor(
            &routeTable, &tasks[0].tensor, tasks[0].fileName, tasks[0].offset);
        assert(routeTable.dataType == fastllm::DataType::INT32);
        assert(routeTable.Count(0) == 4);
        const int32_t *values = reinterpret_cast<const int32_t*>(routeTable.cpuData);
        assert(values[0] == 3);
        assert(values[1] == 1);
        assert(values[2] == 2);
        assert(values[3] == 0);
    }

    void TestUnsupportedTypeFailsBeforeStride() {
        ScopedFixture fixture(ScopedFixture::Kind::UnsupportedType);
        std::vector<fastllm::ReadGGUFTask> tasks;
        std::string message;
        try {
            fastllm::AppendGGUFTasks(
                "deepseek_v4", fixture.path.string(), tasks);
        } catch (const std::runtime_error &error) {
            message = error.what();
        }
        assert(tasks.empty());
        assert(message.find("blk.0.ffn_down_exps.weight") != std::string::npos);
        assert(message.find("40") != std::string::npos);
        assert(message.find("unsupported GGML type") != std::string::npos);
    }

    std::string AppendFailure(const std::filesystem::path &path) {
        std::vector<fastllm::ReadGGUFTask> tasks;
        std::string message;
        try {
            fastllm::AppendGGUFTasks("deepseek_v4", path.string(), tasks);
        } catch (const std::runtime_error &error) {
            message = error.what();
        }
        assert(tasks.empty());
        return message;
    }

    void TestLoaderSafetyGuards() {
        {
            ScopedFixture fixture(ScopedFixture::Kind::NestedArray);
            json11::Json config;
            std::string metadataMessage;
            try {
                fastllm::ReadGGUFMetaData(fixture.path.string(), config);
            } catch (const std::runtime_error &error) {
                metadataMessage = error.what();
            }
            assert(metadataMessage.find("nested element type") != std::string::npos);
            assert(AppendFailure(fixture.path).find("nested element type") !=
                   std::string::npos);
        }
        {
            ScopedFixture fixture(ScopedFixture::Kind::InvalidAlignment);
            assert(AppendFailure(fixture.path).find(
                       "nonzero power of two") != std::string::npos);
        }
        {
            ScopedFixture fixture(ScopedFixture::Kind::TooManyDimensions);
            assert(AppendFailure(fixture.path).find(
                       "invalid dimension count") != std::string::npos);
        }
        {
            ScopedFixture fixture(ScopedFixture::Kind::OverflowDimensions);
            assert(AppendFailure(fixture.path).find(
                       "size multiplication overflow") != std::string::npos);
        }
        {
            ScopedFixture fixture;
            const auto originalSize = std::filesystem::file_size(fixture.path);
            assert(originalSize > sizeof(int32_t));
            std::filesystem::resize_file(
                fixture.path, originalSize - sizeof(int32_t));

            std::vector<fastllm::ReadGGUFTask> tasks;
            fastllm::AppendGGUFTasks(
                "deepseek_v4", fixture.path.string(), tasks);
            assert(tasks.size() == 1);
            fastllm::Data routeTable;
            std::string message;
            try {
                fastllm::WeightImportGGUFTensor(
                    &routeTable, &tasks[0].tensor,
                    tasks[0].fileName, tasks[0].offset);
            } catch (const std::runtime_error &error) {
                message = error.what();
            }
            assert(message.find("Short payload read") != std::string::npos);
            assert(message.find("ffn_gate_tid2eid") != std::string::npos);
        }
    }

    void TestArchitectureAndMetadata() {
        assert(fastllm::ConvertGGUFTypeToFastllmType("deepseek4") == "deepseek_v4");
        assert(fastllm::ConvertGGUFTypeToFastllmType("deepseek_v4") == "deepseek_v4");

        json11::Json::array compressRatios;
        json11::Json::array clamp;
        for (int layer = 0; layer < 43; layer++) {
            compressRatios.push_back(layer < 2 ? 0 : (layer % 2 == 0 ? 4 : 128));
            clamp.push_back(10.0);
        }
        json11::Json params = json11::Json::object {
            {"deepseek4.block_count", 43},
            {"deepseek4.context_length", 1048576},
            {"deepseek4.embedding_length", 4096},
            {"deepseek4.attention.head_count", 64},
            {"deepseek4.attention.head_count_kv", 1},
            {"deepseek4.attention.key_length", 512},
            {"deepseek4.rope.dimension_count", 64},
            {"deepseek4.attention.q_lora_rank", 1024},
            {"deepseek4.attention.output_lora_rank", 1024},
            {"deepseek4.attention.output_group_count", 8},
            {"deepseek4.attention.sliding_window", 128},
            {"deepseek4.attention.layer_norm_rms_epsilon", 1e-6},
            {"deepseek4.expert_count", 256},
            {"deepseek4.expert_used_count", 6},
            {"deepseek4.expert_shared_count", 1},
            {"deepseek4.expert_feed_forward_length", 2048},
            {"deepseek4.expert_weights_scale", 1.5},
            {"deepseek4.expert_weights_norm", true},
            {"deepseek4.expert_gating_func", 4},
            {"deepseek4.swiglu_clamp_exp", clamp},
            {"deepseek4.attention.indexer.head_count", 64},
            {"deepseek4.attention.indexer.key_length", 128},
            {"deepseek4.attention.indexer.top_k", 512},
            {"deepseek4.attention.compress_ratios", compressRatios},
            {"deepseek4.attention.compress_rope_freq_base", 160000.0},
            {"deepseek4.hyper_connection.count", 4},
            {"deepseek4.hyper_connection.sinkhorn_iterations", 20},
            {"deepseek4.hyper_connection.epsilon", 1e-6},
            {"deepseek4.hash_layer_count", 3},
            {"deepseek4.rope.freq_base", 10000.0},
            {"deepseek4.rope.scaling.type", "yarn"},
            {"deepseek4.rope.scaling.factor", 16.0},
            {"deepseek4.rope.scaling.original_context_length", 65536},
            {"deepseek4.rope.scaling.yarn_beta_fast", 32.0},
            {"deepseek4.rope.scaling.yarn_beta_slow", 1.0},
        };

        fastllm::DeepSeekV4Model model;
        fastllm::ApplyDeepSeekV4GGUFMetadata(&model, params, "deepseek4");
        const auto &dicts = model.weight.dicts;
        assert(model.block_cnt == 43);
        assert(dicts.at("model_type") == "deepseek_v4");
        assert(dicts.at("num_hidden_layers") == "43");
        assert(dicts.at("hidden_size") == "4096");
        assert(dicts.at("num_attention_heads") == "64");
        assert(dicts.at("num_key_value_heads") == "1");
        assert(dicts.at("head_dim") == "512");
        assert(dicts.at("qk_rope_head_dim") == "64");
        assert(dicts.at("n_routed_experts") == "256");
        assert(dicts.at("num_experts_per_tok") == "6");
        assert(dicts.at("n_shared_experts") == "1");
        assert(dicts.at("scoring_func") == "sqrtsoftplus");
        assert(dicts.at("topk_method") == "noaux_tc");
        assert(dicts.at("norm_topk_prob") == "true");
        assert(dicts.at("num_hash_layers") == "3");
        assert(dicts.at("num_nextn_predict_layers") == "0");
        assert(dicts.at("rope_scaling.type") == "yarn");
        assert(std::fabs(std::stof(dicts.at("swiglu_limit")) - 10.0f) < 1e-6f);
        assert(std::fabs(std::stof(dicts.at("compress_rope_theta")) - 160000.0f) < 1e-3f);
        std::string error;
        const auto parsedRatios = json11::Json::parse(dicts.at("compress_ratios"), error);
        assert(error.empty());
        assert(parsedRatios.array_items().size() == 43);
    }

    void TestTensorMappings() {
        const auto rules = fastllm::GetGGUFWeightReplaceRules("deepseek_v4");
        const auto aliasRules = fastllm::GetGGUFWeightReplaceRules("deepseek4");
        assert(rules.size() == aliasRules.size());
        assert(rules.size() == 41);

        ExpectDirect(rules, "token_embd.weight", "embed.weight");
        ExpectDirect(rules, "output.weight", "head.weight");
        ExpectDirect(rules, "output_norm.weight", "norm.weight");
        ExpectDirect(rules, "output_hc_base.weight", "hc_head_base");
        ExpectDirect(rules, "output_hc_fn.weight", "hc_head_fn");
        ExpectDirect(rules, "output_hc_scale.weight", "hc_head_scale");
        ExpectDirect(rules, "blk.7.attn_q_a.weight", "layers.7.attn.wq_a.weight");
        ExpectDirect(rules, "blk.7.attn_q_a_norm.weight", "layers.7.attn.q_norm.weight");
        ExpectDirect(rules, "blk.7.attn_q_b.weight", "layers.7.attn.wq_b.weight");
        ExpectDirect(rules, "blk.7.attn_kv.weight", "layers.7.attn.wkv.weight");
        ExpectDirect(rules, "blk.7.attn_kv_a_norm.weight", "layers.7.attn.kv_norm.weight");
        ExpectDirect(rules, "blk.7.attn_output_a.weight", "layers.7.attn.wo_a.weight");
        ExpectDirect(rules, "blk.7.attn_output_b.weight", "layers.7.attn.wo_b.weight");
        ExpectDirect(rules, "blk.7.attn_sinks.weight", "layers.7.attn.attn_sink");
        ExpectDirect(rules, "blk.7.attn_norm.weight", "layers.7.attn_norm.weight");
        ExpectDirect(rules, "blk.7.attn_compressor_kv.weight", "layers.7.attn.compressor.wkv.weight");
        ExpectDirect(rules, "blk.7.attn_compressor_gate.weight", "layers.7.attn.compressor.wgate.weight");
        ExpectDirect(rules, "blk.7.attn_compressor_ape.weight", "layers.7.attn.compressor.ape");
        ExpectDirect(rules, "blk.7.attn_compressor_norm.weight", "layers.7.attn.compressor.norm.weight");
        ExpectDirect(rules, "blk.8.indexer.attn_q_b.weight", "layers.8.attn.indexer.wq_b.weight");
        ExpectDirect(rules, "blk.8.indexer.proj.weight", "layers.8.attn.indexer.weights_proj.weight");
        ExpectDirect(rules, "blk.8.indexer_compressor_kv.weight", "layers.8.attn.indexer.compressor.wkv.weight");
        ExpectDirect(rules, "blk.8.indexer_compressor_gate.weight", "layers.8.attn.indexer.compressor.wgate.weight");
        ExpectDirect(rules, "blk.8.indexer_compressor_ape.weight", "layers.8.attn.indexer.compressor.ape");
        ExpectDirect(rules, "blk.8.indexer_compressor_norm.weight", "layers.8.attn.indexer.compressor.norm.weight");
        ExpectDirect(rules, "blk.7.ffn_gate_inp.weight", "layers.7.ffn.gate.weight");
        ExpectDirect(rules, "blk.7.exp_probs_b.bias", "layers.7.ffn.gate.bias");
        ExpectDirect(rules, "blk.1.ffn_gate_tid2eid.weight", "layers.1.ffn.gate.tid2eid");
        ExpectDirect(rules, "blk.7.ffn_gate_shexp.weight", "layers.7.ffn.shared_experts.w1.weight");
        ExpectDirect(rules, "blk.7.ffn_up_shexp.weight", "layers.7.ffn.shared_experts.w3.weight");
        ExpectDirect(rules, "blk.7.ffn_down_shexp.weight", "layers.7.ffn.shared_experts.w2.weight");
        ExpectDirect(rules, "blk.7.ffn_norm.weight", "layers.7.ffn_norm.weight");
        ExpectDirect(rules, "blk.7.hc_attn_base.weight", "layers.7.hc_attn_base");
        ExpectDirect(rules, "blk.7.hc_attn_fn.weight", "layers.7.hc_attn_fn");
        ExpectDirect(rules, "blk.7.hc_attn_scale.weight", "layers.7.hc_attn_scale");
        ExpectDirect(rules, "blk.7.hc_ffn_base.weight", "layers.7.hc_ffn_base");
        ExpectDirect(rules, "blk.7.hc_ffn_fn.weight", "layers.7.hc_ffn_fn");
        ExpectDirect(rules, "blk.7.hc_ffn_scale.weight", "layers.7.hc_ffn_scale");

        const auto sourceNames = BuildCandidateTensorNames();
        assert(sourceNames.size() == 1328);
        assert(std::set<std::string>(sourceNames.begin(), sourceNames.end()).size() == 1328);

        std::set<std::string> targets;
        int packedSources = 0;
        for (const auto &source : sourceNames) {
            const auto &rule = FindOnlyRule(rules, source);
            if (rule.type == GGUFWeightReplaceRule::GGUFWeightReplacePacked) {
                packedSources++;
                assert(rule.names.size() == 2);
                const std::string prefix = std::regex_replace(source, rule.pattern, rule.names[0]);
                const std::string suffix = std::regex_replace(source, rule.pattern, rule.names[1]);
                for (int expert = 0; expert < 256; expert++) {
                    const std::string target = prefix + std::to_string(expert) + suffix;
                    assert(target.rfind("layers.", 0) == 0);
                    assert(target.find(".ffn.experts.") != std::string::npos);
                    targets.insert(target);
                }
            } else {
                const std::string target = std::regex_replace(source, rule.pattern, rule.names[0]);
                assert(target != "ignore");
                assert(target.rfind("model.", 0) != 0);
                assert(target.rfind("mtp.", 0) != 0);
                targets.insert(target);
            }
        }
        assert(packedSources == 43 * 3);
        assert(targets.size() == (1328 - packedSources) + packedSources * 256);

        const auto &gateRule = FindOnlyRule(rules, "blk.42.ffn_gate_exps.weight");
        const auto &upRule = FindOnlyRule(rules, "blk.42.ffn_up_exps.weight");
        const auto &downRule = FindOnlyRule(rules, "blk.42.ffn_down_exps.weight");
        assert(std::regex_replace("blk.42.ffn_gate_exps.weight", gateRule.pattern, gateRule.names[0]) +
                   "255" + std::regex_replace("blk.42.ffn_gate_exps.weight", gateRule.pattern, gateRule.names[1]) ==
               "layers.42.ffn.experts.255.w1.weight");
        assert(std::regex_replace("blk.42.ffn_up_exps.weight", upRule.pattern, upRule.names[0]) +
                   "255" + std::regex_replace("blk.42.ffn_up_exps.weight", upRule.pattern, upRule.names[1]) ==
               "layers.42.ffn.experts.255.w3.weight");
        assert(std::regex_replace("blk.42.ffn_down_exps.weight", downRule.pattern, downRule.names[0]) +
                   "255" + std::regex_replace("blk.42.ffn_down_exps.weight", downRule.pattern, downRule.names[1]) ==
               "layers.42.ffn.experts.255.w2.weight");
    }

    void TestExistingMetadataSplit(const std::string &fileName) {
        assert(std::filesystem::is_regular_file(fileName));

        json11::Json config;
        fastllm::ReadGGUFMetaData(fileName, config);
        const auto &params = config["params"];
        assert(config["version"].int_value() == 3);
        assert(config["tensorCount"].int_value() == 0);
        assert(config["metaDataCount"].int_value() > 0);
        assert(params["general.architecture"].string_value() == "deepseek4");
        assert(params["split.count"].int_value() == 3);

        std::vector<fastllm::ReadGGUFTask> tasks;
        fastllm::AppendGGUFTasks("deepseek4", fileName, tasks);
        assert(tasks.empty());

        std::cout << "Existing Q2 metadata-only split preflight passed: "
                  << config["metaDataCount"].int_value()
                  << " metadata entries, zero tensor descriptors or payload reads.\n";
    }
}

int main(int argc, char **argv) {
    TestRealGGUFMetadataParsing();
    TestUnsupportedTypeFailsBeforeStride();
    TestLoaderSafetyGuards();
    TestArchitectureAndMetadata();
    TestTensorMappings();
    if (argc == 2) {
        TestExistingMetadataSplit(argv[1]);
    } else {
        assert(argc == 1);
    }
    std::cout << "DeepSeek V4 GGUF adapter tests passed.\n";
    return 0;
}
