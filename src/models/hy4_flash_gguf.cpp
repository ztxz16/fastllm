#include "hy4_flash_ops.h"
#include "gguf.h"
#include <climits>
#include <cstring>
#include <fstream>

namespace fastllm {
namespace {
using Task = ReadGGUFTask;

class ShardReader {
    std::map<std::string, std::ifstream> files;

  public:
    void Read(const Task &t, size_t offset, void *dest, size_t bytes) {
        if (offset > ggml_nbytes(&t.tensor) || bytes > ggml_nbytes(&t.tensor) - offset)
            throw std::runtime_error("Invalid HY4 GGUF slice: " + t.name);
        auto &f = files[t.fileName];
        if (!f.is_open()) {
            f.exceptions(std::ios::failbit | std::ios::badbit);
            f.open(t.fileName, std::ios::binary);
        }
        f.seekg(t.offset + offset);
        f.read(static_cast<char *>(dest), bytes);
    }

    // Dense CUDA ops consume BF16 matrices and F32 parameters. Leave routed
    // expert blocks packed; they are registered separately with the NUMA backend.
    std::vector<uint8_t> Dense(const Task &t, bool forceBf16 = false) {
        const size_t length = size_t(t.tensor.ne[0]) * t.tensor.ne[1] * t.tensor.ne[2] * t.tensor.ne[3];
        const bool f32 = t.tensor.type == GGML_TYPE_F32 && !forceBf16;
        std::vector<uint8_t> data(length * (f32 ? 4 : 2));
        if (f32 || t.tensor.type == GGML_TYPE_BF16) {
            Read(t, 0, data.data(), data.size());
            return data;
        }
        auto decode = ggml_type_to_float(t.tensor.type);
        if (!decode && t.tensor.type != GGML_TYPE_F32)
            throw std::runtime_error("Unsupported HY4 GGUF dense type: " + t.name);
        const size_t cols = t.tensor.ne[0], rowBytes = ggml_row_size(t.tensor.type, cols);
        const size_t rows = length / cols, tile = std::max<size_t>(1, (1 << 18) / cols);
        std::vector<float> values(tile * cols);
        std::vector<uint8_t> packed(tile * rowBytes);
        auto dst = reinterpret_cast<uint16_t *>(data.data());
        for (size_t row = 0; row < rows; row += tile) {
            size_t n = std::min(tile, rows - row);
            Read(t, row * rowBytes, packed.data(), n * rowBytes);
            if (decode)
                decode(packed.data(), values.data(), n * cols);
            else
                std::memcpy(values.data(), packed.data(), n * cols * sizeof(float));
            for (size_t i = 0; i < n * cols; ++i)
                dst[row * cols + i] = Float32ToBFloat16RNEBits(values[i]);
        }
        return data;
    }
};

const std::map<std::string, std::string> layerNames = {
    {"attn_gate.weight", "self_attn.linear_gate.weight"},
    {"attn_kv_a_mqa.weight", "self_attn.kv_a_proj_with_mqa.weight"},
    {"attn_kv_a_norm.weight", "self_attn.kv_a_layernorm.weight"},
    {"attn_norm.weight", "input_layernorm.weight"},
    {"attn_output.weight", "self_attn.o_proj.weight"},
    {"attn_q_a.weight", "self_attn.q_a_proj.weight"},
    {"attn_q_a_norm.weight", "self_attn.q_a_layernorm.weight"},
    {"attn_q_b.weight", "self_attn.q_b_proj.weight"},
    {"attn_sinks.weight", "self_attn.learnable_sink_param"},
    {"ffn_norm.weight", "post_attention_layernorm.weight"},
    {"post_attention_norm.weight", "self_attn.post_proj_norm.weight"},
    {"ffn_down.weight", "mlp.down_proj.weight"},
    {"ffn_gate.weight", "mlp.gate_proj.weight"},
    {"ffn_up.weight", "mlp.up_proj.weight"},
    {"ffn_down_shexp.weight", "mlp.shared_experts.down_proj.weight"},
    {"ffn_gate_shexp.weight", "mlp.shared_experts.gate_proj.weight"},
    {"ffn_up_shexp.weight", "mlp.shared_experts.up_proj.weight"},
    {"ffn_routed_down.weight", "mlp.routed_expert_down_proj.weight"},
    {"ffn_routed_up.weight", "mlp.routed_expert_up_proj.weight"},
    {"ffn_gate_inp.weight", "mlp.gate.weight"},
    {"exp_probs_b.bias", "mlp.gate.e_score_correction_bias"},
    {"hc_attn_base.weight", "hc_attn_layer.hc_pre.hc_base"},
    {"hc_attn_fn.weight", "hc_attn_layer.hc_pre.hc_fn"},
    {"hc_attn_scale.weight", "hc_attn_layer.hc_pre.hc_scale"},
    {"hc_ffn_base.weight", "hc_mlp_layer.hc_pre.hc_base"},
    {"hc_ffn_fn.weight", "hc_mlp_layer.hc_pre.hc_fn"},
    {"hc_ffn_scale.weight", "hc_mlp_layer.hc_pre.hc_scale"},
    {"indexer.attn_k.weight", "self_attn.compressed_indexer.token_key_projection.weight"},
    {"indexer.attn_q_b.weight", "self_attn.compressed_indexer.query_projection.weight"},
    {"indexer.k_norm.weight", "self_attn.compressed_indexer.token_key_norm.weight"},
    {"indexer.proj.weight", "self_attn.compressed_indexer.head_weight_projection.weight"},
    {"indexer_compressor_ape.weight", "self_attn.compressed_indexer.intra_group_ape"},
    {"indexer_compressor_gate.weight", "self_attn.compressed_indexer.key_gate_projection.weight"},
    {"indexer_compressor_norm.weight", "self_attn.compressed_indexer.key_norm.weight"},
    {"indexer_compressor_value.weight", "self_attn.compressed_indexer.key_value_projection.weight"},
};

} // namespace

void ReadEngramRow(const void *table, bool scalarFp8, int64_t row, uint16_t *out) {
    if (!scalarFp8) {
        std::memcpy(out, static_cast<const uint16_t *>(table) + row * 96, 96 * sizeof(uint16_t));
        return;
    }
    static const FP8E4M3ToFP32Manager values;
    const auto *src = static_cast<const uint8_t *>(table) + row * 96;
    for (int i = 0; i < 96; ++i)
        out[i] = (src[i] & 0x7f) == 0x7f ? uint16_t(0x7fc0) : Float32ToBFloat16RNEBits(values.dict[src[i]]);
}

void Hy4FlashModel::ImportGGUF(const json11::Json &params, const std::vector<Task> &tasks) {
    auto param = [&](const std::string &key) -> const json11::Json & { return params["hy_v4_flash." + key]; };
    std::map<std::string, const Task *> byName;
    for (const auto &t : tasks)
        AssertInFastLLM(byName.emplace(t.name, &t).second, "Duplicate HY4 GGUF weight: " + t.name);
    auto get = [&](const std::string &name) -> const Task & {
        auto it = byName.find(name);
        if (it == byName.end())
            throw std::runtime_error("Missing HY4 GGUF weight: " + name);
        return *it->second;
    };
    ShardReader reader;
    auto install = [&](const std::string &name, const std::string &dtype, const std::vector<int> &dims,
                       std::vector<uint8_t> data) {
        AddSource(name, "", dtype, dims, 0, data.size());
        source->host.emplace(name, std::move(data));
    };
    auto dense = [&](const std::string &name, const Task &t) {
        install(name, t.tensor.type == GGML_TYPE_F32 ? "F32" : "BF16", t.tensor.dims, reader.Dense(t));
    };
    const int layers = param("block_count").int_value() - param("nextn_predict_layers").int_value();
    AssertInFastLLM(layers == block_cnt && layers == 42, "Invalid HY4 GGUF target layer count.");
    const auto &pattern = param("attention.sliding_window_pattern").array_items();
    AssertInFastLLM(pattern.size() >= size_t(layers), "Missing HY4 GGUF attention pattern.");
    json11::Json::array types;
    for (int i = 0; i < layers; ++i)
        types.emplace_back(pattern[i].bool_value() ? "sliding_attention" : "sparse_attention");
    const auto &indexers = param("attention.indexer.types").array_items();
    const auto &clamps = param("swiglu_clamp_exp").array_items();
    AssertInFastLLM(indexers.size() >= size_t(layers) && clamps.size() >= size_t(layers) &&
                        param("leading_dense_block_count").int_value() == 1 &&
                        param("attention.sliding_window").int_value() == 512 &&
                        param("ple.ngram_size").int_value() == 3,
                    "Unsupported HY4 GGUF layer configuration.");
    for (int layer = 0; layer < layers; ++layer)
        AssertInFastLLM(indexers[layer].bool_value() != pattern[layer].bool_value() &&
                            clamps[layer].number_value() == 10.,
                        "Unsupported HY4 GGUF attention or MoE layout.");
    const auto &modules = param("ple.layers").array_items();
    const auto &offsets = param("ple.head_offsets").array_items();
    const auto &sizes = param("ple.head_vocab_sizes").array_items();
    const auto &factors = param("ple.layer_multipliers").array_items();
    const auto &map = param("engram.token_map").array_items();
    const auto &table = get("per_layer_token_embd.weight");
    AssertInFastLLM(offsets.size() == modules.size() * 16 && sizes.size() == offsets.size() &&
                        factors.size() == modules.size() * 3 && !map.empty() &&
                        table.tensor.dims.size() == 2 && table.tensor.ne[0] == 96,
                    "Invalid HY4 GGUF Engram geometry.");
    json11::Json::object vocabSizes;
    int64_t totalRows = 0;
    for (size_t i = 0; i < modules.size(); ++i) {
        int layer = modules[i].int_value();
        AssertInFastLLM(layer >= 0 && layer < layers && !source->engramFactors.count(layer),
                        "Invalid or duplicate HY4 GGUF Engram layer.");
        const std::string p = "model.layers." + std::to_string(layer) + ".engram_layer.";
        json11::Json::array orders;
        int64_t start = totalRows;
        for (int order = 0; order < 2; ++order) {
            json11::Json::array heads;
            for (int j = 0; j < 8; ++j) {
                size_t idx = i * 16 + order * 8 + j;
                AssertInFastLLM(offsets[idx].number_value() == totalRows && sizes[idx].int_value() > 0,
                                "Invalid HY4 GGUF Engram bucket.");
                heads.push_back(sizes[idx]);
                totalRows += sizes[idx].int_value();
            }
            orders.emplace_back(std::move(heads));
        }
        vocabSizes[std::to_string(layer + 1)] = std::move(orders);
        for (int j = 0; j < 3; ++j)
            source->engramFactors[layer][j] = static_cast<int64_t>(factors[i * 3 + j].number_value());
        const bool fp8 = table.tensor.type == GGML_TYPE_I8;
        size_t stride = 96 * (fp8 ? 1 : 2);
        AssertInFastLLM(totalRows <= table.tensor.ne[1], "HY4 Engram table is too small.");
        AddSource(p + "multi_head_embedding.embedding.weight", table.fileName, fp8 ? "E4M3" : "BF16",
                  {int(totalRows - start), 96}, table.offset + start * stride, (totalRows - start) * stride);
        std::vector<uint8_t> ids(map.size() * sizeof(int));
        for (size_t k = 0; k < map.size(); ++k) {
            AssertInFastLLM(map[k].is_number() && map[k].number_value() >= 0 &&
                                map[k].number_value() <= INT_MAX,
                            "Invalid HY4 GGUF compressed token ID.");
            reinterpret_cast<int *>(ids.data())[k] = map[k].int_value();
        }
        install(p + "lookup_table", "I32", {int(map.size())}, std::move(ids));

        // The publisher concatenates the four key projections and the value projection.
        auto split = [&](const std::string &src, int count, const std::vector<int> &shape,
                         const std::function<std::string(int)> &name) {
            const auto &t = get("blk." + std::to_string(layer) + "." + src + ".weight");
            if (src == "engram_kv" && t.tensor.type == GGML_TYPE_Q8_0) {
                const size_t stride = ggml_row_size(t.tensor.type, shape[1]) * shape[0];
                AssertInFastLLM(stride * count == ggml_nbytes(&t.tensor),
                                "Invalid packed Engram projection.");
                for (int j = 0; j < count; ++j)
                    AddSource(p + name(j), t.fileName, "Q8_0", shape, t.offset + j * stride, stride);
                return;
            }
            auto data = reader.Dense(t);
            int elements = 1;
            for (int dim : shape)
                elements *= dim;
            const size_t stride = size_t(elements) * (t.tensor.type == GGML_TYPE_F32 ? 4 : 2);
            AssertInFastLLM(stride * count == data.size(), "Invalid HY4 GGUF fused shape: " + t.name);
            for (int j = 0; j < count; ++j)
                install(p + name(j), t.tensor.type == GGML_TYPE_F32 ? "F32" : "BF16", shape,
                        {data.begin() + j * stride, data.begin() + (j + 1) * stride});
        };
        auto kv = [](int j) { return j == 4 ? std::string("value_proj") : "key_projs." + std::to_string(j); };
        split("engram_kv", 5, {3072, 1536}, [&](int j) { return kv(j) + ".weight"; });
        split("engram_kv_b", 5, {3072}, [&](int j) { return kv(j) + ".bias"; });
        split("engram_gate", 4, {3072}, [](int j) { return "norm_key." + std::to_string(j) + ".weight"; });
        split("engram_lookup_norm", 16, {96},
              [](int j) { return "lookup_per_head_norms." + std::to_string(j) + ".weight"; });
        split("engram_hc_scale", 4, {3072},
              [](int j) { return "unified_per_hc_scales." + std::to_string(j); });
        dense(p + "unified_pre_gamma", get("blk." + std::to_string(layer) + ".engram_pre_gamma.weight"));
    }
    AssertInFastLLM(totalRows == table.tensor.ne[1], "Invalid HY4 GGUF Engram table length.");
    SetConfig(json11::Json::object{
        {"text_config", json11::Json::object{{"hidden_size", param("embedding_length")},
                                             {"num_hidden_layers", layers},
                                             {"layer_types", types},
                                             {"hc_mult", param("hyper_connection.count")},
                                             {"num_experts_per_tok", param("expert_used_count")},
                                             {"routed_expert_hidden_size", param("expert_latent_length")},
                                             {"n_routed_experts", param("expert_count")},
                                             {"moe_intermediate_size", param("expert_feed_forward_length")},
                                             {"routed_scaling_factor", param("expert_weights_scale")},
                                             {"max_position_embeddings", param("context_length")},
                                             {"engram_pad_id", param("engram.pad_token_id")},
                                             {"engram_layer_head_vocab_sizes", vocabSizes}}}});
    source->gguf = true;
    for (auto &entry :
         std::map<std::string, std::string>{{"output.weight", "lm_head.weight"},
                                            {"output_norm.weight", "model.norm.weight"},
                                            {"output_hc_fn.weight", "model.hc_head.hc_head_fn"},
                                            {"output_hc_base.weight", "model.hc_head.hc_head_base"},
                                            {"output_hc_scale.weight", "model.hc_head.hc_head_scale"}})
        dense(entry.second, get(entry.first));
    const auto &embedding = get("token_embd.weight");
    if (embedding.tensor.type == GGML_TYPE_Q8_0)
        AddSource("model.embed_tokens.weight", embedding.fileName, "Q8_0", embedding.tensor.dims,
                  embedding.offset, ggml_nbytes(&embedding.tensor));
    else
        dense("model.embed_tokens.weight", embedding);
    for (int layer = 0; layer < layers; ++layer) {
        const auto raw = "blk." + std::to_string(layer) + ".";
        const auto p = "model.layers." + std::to_string(layer) + ".";
        for (const auto &entry : layerNames) {
            const auto &name = entry.first;
            const bool indexer = name.rfind("indexer", 0) == 0;
            const bool denseFFN =
                name == "ffn_down.weight" || name == "ffn_gate.weight" || name == "ffn_up.weight";
            const bool expert = name.find("_shexp.") != name.npos || name.rfind("ffn_routed_", 0) == 0 ||
                                name == "ffn_gate_inp.weight" || name == "exp_probs_b.bias";
            if ((indexer && !indexers[layer].bool_value()) || (denseFFN && layer != 0) ||
                (expert && layer == 0))
                continue;
            const auto &t = get(raw + name);
            const bool projection = entry.first.rfind("attn_", 0) == 0 || entry.first.rfind("ffn_", 0) == 0 ||
                                    entry.first.rfind("indexer.attn_", 0) == 0;
            if (projection && t.tensor.type == GGML_TYPE_Q8_0 && t.tensor.dims.size() == 2)
                AddSource(p + entry.second, t.fileName, "Q8_0", t.tensor.dims, t.offset,
                          ggml_nbytes(&t.tensor));
            else
                dense(p + entry.second, t);
        }
        dense(p + (layer ? "mlp.combine_output_norm.weight" : "mlp.post_fc2_norm.weight"),
              get(raw + "post_ffw_norm.weight"));
        const auto &k = get(raw + "attn_k_b.weight"), &v = get(raw + "attn_v_b.weight");
        int heads = pattern[layer].bool_value() ? 64 : 32;
        AssertInFastLLM(k.tensor.dims == std::vector<int>({heads, 512, 192}) &&
                            v.tensor.dims == std::vector<int>({heads, 256, 512}),
                        "Invalid HY4 MLA projections.");
        auto key = reader.Dense(k, true), value = reader.Dense(v, true);
        std::vector<uint8_t> packed(size_t(heads) * 448 * 512 * 2);
        for (int h = 0; h < heads; ++h) {
            auto *dst = reinterpret_cast<uint16_t *>(packed.data()) + h * 448 * 512;
            auto *src = reinterpret_cast<const uint16_t *>(key.data()) + h * 192 * 512;
            for (int j = 0; j < 512; ++j)
                for (int i = 0; i < 192; ++i)
                    dst[i * 512 + j] = src[j * 192 + i];
            std::memcpy(dst + 192 * 512, value.data() + size_t(h) * 256 * 512 * 2, 256 * 512 * 2);
        }
        install(p + "self_attn.kv_b_proj.weight", "BF16", {heads * 448, 512}, std::move(packed));
        if (!layer)
            continue;
        const auto &gate = get(raw + "ffn_gate_exps.weight"), &up = get(raw + "ffn_up_exps.weight"),
                   &down = get(raw + "ffn_down_exps.weight");
        for (const auto *t : {&gate, &up, &down})
            AssertInFastLLM(t->tensor.dims == std::vector<int>({512, 1536, 1536}) &&
                                (t->tensor.type == GGML_TYPE_Q4_K || t->tensor.type == GGML_TYPE_Q6_K ||
                                 (t != &down && t->tensor.type == GGML_TYPE_Q3_K)),
                            "Unsupported HY4 GGUF expert shape or type: " + t->name);
        AssertInFastLLM(gate.tensor.type == up.tensor.type, "HY4 GGUF gate and up types must match.");
        for (int expert = 0; expert < 512; ++expert) {
            auto make = [&](const std::string &suffix, const Task &t, bool pair) {
                std::string name = p + "mlp.experts." + std::to_string(expert) + "." + suffix;
                weight.AddEmptyGGMLWeight(name, {pair ? 3072 : 1536, 1536}, DATA_GGUF_FORMAT, t.tensor.type);
                auto &w = weight[name];
                w.isGGUFData = true;
                w.Allocate(false);
                size_t stride = ggml_nbytes(&t.tensor) / 512;
                reader.Read(t, expert * stride, w.cpuData, stride);
                if (pair)
                    reader.Read(up, expert * stride, w.cpuData + stride, stride);
                AddSpecialWeight(name, pair ? "linearSwiglu" : "linearColumn", layer);
                weight.linearNames.insert(name);
                moeLinears.insert(name);
            };
            make("gateup_proj.weight", gate, true);
            make("down_proj.weight", down, false);
        }
    }
}
} // namespace fastllm
