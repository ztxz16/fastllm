#include "inkling2.h"
#include "gguf.h"
#include "json11.hpp"
#include <algorithm>
#include <cstring>
#include <set>

namespace fastllm {
void Inkling2Model::ConfigureGGUF(const json11::Json &meta) {
    auto get = [&](const std::string &key) -> const json11::Json & {
        const auto &v = meta["inkling2_loop." + key];
        if (v.is_null()) throw std::runtime_error("Missing Inkling GGUF metadata: " + key);
        return v;
    };
    auto put = [&](const std::string &key, const json11::Json &v) { weight.AddDict("text_config." + key, v.dump()); };
    for (const auto &[key, source] :
         std::map<std::string, std::string>{{"hidden_size", "embedding_length"},
                                            {"num_hidden_layers", "block_count"},
                                            {"head_dim", "attention.key_length"},
                                            {"d_rel", "attention.relative_length"},
                                            {"rel_extent", "attention.relative_extent"},
                                            {"sliding_window_size", "attention.sliding_window"},
                                            {"sconv_kernel_size", "shortconv.l_cache"},
                                            {"n_layers_per_attn_res_block", "attn_res.block_size"},
                                            {"n_routed_experts", "expert_count"},
                                            {"num_experts_per_tok", "expert_used_count"},
                                            {"intermediate_size", "expert_feed_forward_length"},
                                            {"loop_config.core_start", "loop.start"},
                                            {"loop_config.core_end", "loop.end"},
                                            {"loop_config.num_passes", "loop.count"},
                                            {"loop_config.gate_rank", "loop.gate_rank"},
                                            {"loop_config.is_varloop", "loop.varying"},
                                            {"unpadded_vocab_size", "vocab_size_unpadded"},
                                            {"log_scaling_n_floor", "attention.log_scale_floor"},
                                            {"log_scaling_alpha", "attention.log_scale_alpha"},
                                            {"rms_norm_eps", "attention.layer_norm_rms_epsilon"},
                                            {"qk_l2_clip_threshold", "attention.qk_clip"},
                                            {"rel_l2_clip_threshold", "attention.relative_clip"},
                                            {"mlp_gate_softcap", "activation.situ_beta"},
                                            {"mlp_up_softcap", "activation.situ_linear_beta"},
                                            {"route_scale", "expert_weights_scale"},
                                            {"final_logit_softcapping", "final_logit_softcapping"},
                                            {"model_max_length", "context_length"}})
        put(key, get(source));
    const int layers = get("block_count").int_value();
    for (const auto &[key, source] : std::map<std::string, std::string>{
             {"num_attention_heads", "attention.head_count"}, {"num_key_value_heads", "attention.head_count_kv"}}) {
        const auto &v = get(source);
        if (v.is_number()) {
            put(key, v);
            continue;
        }
        const auto &a = v.array_items();
        if (a.empty() || !a.front().is_number() || int(a.size()) != layers ||
            !std::all_of(a.begin(), a.end(), [&](const auto &e) { return e == a.front(); }))
            throw std::runtime_error("Inkling GGUF needs uniform " + source);
        put(key, a.front());
    }
    const auto &pattern = get("attention.sliding_window_pattern").array_items();
    if (int(pattern.size()) != layers) throw std::runtime_error("Invalid Inkling GGUF sliding-window pattern");
    json11::Json::array local;
    for (int i = 0; i < layers; ++i) {
        if (!pattern[i].is_bool()) throw std::runtime_error("Invalid Inkling GGUF sliding-window flag");
        if (pattern[i].bool_value()) local.emplace_back(i);
    }
    put("local_layer_ids", local);
}

void Inkling2Model::PrepareGGUF(std::vector<ReadGGUFTask> &tasks) {
    if (std::count(placement.experts.begin(), placement.experts.end(), inkling2::ExpertDevice::Cuda) != cfg.layers)
        throw std::runtime_error("Inkling GGUF currently requires CUDA expert placement");
    using Rule = GGUFWeightReplaceRule;
    const std::string root = "model.llm.";
    const std::map<std::string, std::string> globals{{"token_embd.weight", "embed.weight"},
                                                     {"token_embd_norm.weight", "embed_norm.weight"},
                                                     {"output.weight", "unembed.weight"},
                                                     {"output_norm.weight", "norm.weight"},
                                                     {"output_res_query.weight", "attn_res.output_query"}};
    std::map<std::string, std::string> blocks{{"attn_gate.weight", "attn.gate_attn.weight"},
                                              {"attn_k.weight", "attn.wk_dv.weight"},
                                              {"attn_q.weight", "attn.wq_du.weight"},
                                              {"attn_v.weight", "attn.wv_dv.weight"},
                                              {"attn_output.weight", "attn.wo_ud.weight"},
                                              {"attn_rel.weight", "attn.wr_du.weight"},
                                              {"attn_q_clip.weight", "attn.q_l2_clip.scale"},
                                              {"attn_rel_clip.weight", "attn.rel_l2_clip.scale"},
                                              {"attn_sink_proj.weight", "attn.attn_sink_proj.weight"},
                                              {"attn_norm.weight", "attn_norm.weight"},
                                              {"ffn_norm.weight", "mlp_norm.weight"},
                                              {"ffn_gate_inp.weight", "mlp.gate.weight"},
                                              {"exp_probs_b.bias", "mlp.gate.bias"},
                                              {"attn_post_conv.weight", "attn_sconv.weight"},
                                              {"ffn_post_conv.weight", "mlp_sconv.weight"},
                                              {"attn_k_conv_proj.weight", "attn.k_sconv.weight_proj"},
                                              {"attn_v_conv_proj.weight", "attn.v_sconv.weight_proj"},
                                              {"attn_k_inp_norm.weight", "k_input_norm.weight"},
                                              {"attn_v_inp_norm.weight", "v_input_norm.weight"},
                                              {"attn_q_inp_norm.weight", "q_input_norm.weight"}};
    for (const auto &[from, to] :
         std::map<std::string, std::string>{{"q", "attn"}, {"k", "k"}, {"v", "v"}, {"ffn", "mlp"}})
        for (const std::string kind : {"down", "gain", "forget"})
            blocks["loop_" + from + "_ctrl_" + kind + ".weight"] = "controllers." + to + "." + kind + "_weight";

    std::vector<ReadGGUFTask> dense;
    std::vector<std::string> names;
    std::set<std::string> seen;
    int messages = 0;
    for (auto &task : tasks) {
        const auto &name = task.name;
        if (!seen.insert(name).second) throw std::runtime_error("Duplicate Inkling GGUF tensor: " + name);
        std::string target;
        if (globals.count(name))
            target = root + globals.at(name);
        else {
            if (name.compare(0, 4, "blk.") != 0) throw std::runtime_error("Unknown Inkling GGUF tensor: " + name);
            const auto dot = name.find('.', 4);
            if (dot == std::string::npos || dot == 4 || name.find_first_not_of("0123456789", 4) != dot)
                throw std::runtime_error("Invalid Inkling GGUF layer name: " + name);
            const int layer = std::stoi(name.substr(4, dot - 4));
            if (layer < 0 || layer >= cfg.layers) throw std::runtime_error("Invalid Inkling GGUF layer: " + name);
            const auto part = name.substr(dot + 1);
            const auto prefix = root + "layers." + std::to_string(layer) + ".";
            const auto &shape = task.tensor.dims;
            if (part == "loop_msg_query.weight") {
                if (layer < cfg.start || layer >= cfg.end || shape != std::vector<int>{4, cfg.hidden})
                    throw std::runtime_error("Invalid Inkling cross-pass query: " + name);
                Data src;
                WeightImportGGUFTensor(&src, &task.tensor, task.fileName, task.offset,
                                       Rule::GGUFWeightReplaceForceFP32);
                // The GGUF stores [main Q, KV K, KV V, main MLP] per recurrent layer.
                for (int row = 0; row < 4; ++row) {
                    const bool kv = row == 1 || row == 2;
                    auto &dst =
                        weight[root + (kv ? "middle_cycle_cross_pass_kv_queries" : "middle_cycle_cross_pass_queries")];
                    if (dst.dims.empty()) {
                        dst.dataType = FLOAT32;
                        dst.Resize({2 * (cfg.end - cfg.start), cfg.hidden});
                        dst.Allocate();
                    }
                    size_t bytes = cfg.hidden * sizeof(float);
                    std::memcpy(dst.cpuData + (2 * (layer - cfg.start) + (row >= 2)) * bytes, src.cpuData + row * bytes,
                                bytes);
                }
                ++messages;
                continue;
            }
            if (part == "attn_rel_bias.weight" || part == "attn_k_conv.weight" || part == "attn_v_conv.weight") {
                const bool relative = part == "attn_rel_bias.weight";
                const bool globalConv = !relative && !cfg.local[layer];
                target =
                    prefix + (relative ? "attn.rel_logits_proj.proj"
                                       : std::string(part == "attn_k_conv.weight" ? "attn.k_sconv." : "attn.v_sconv.") +
                                             (globalConv ? "static_w" : "weight"));
                if (relative || globalConv) {
                    const int rows = relative ? cfg.extent : cfg.kvHeads * cfg.headDim;
                    const int cols = relative ? cfg.relative : cfg.conv;
                    if (shape != std::vector<int>{rows, cols})
                        throw std::runtime_error("Invalid Inkling GGUF matrix: " + name);
                    Data src;
                    WeightImportGGUFTensor(&src, &task.tensor, task.fileName, task.offset,
                                           Rule::GGUFWeightReplaceForceFP32);
                    auto &dst = weight[target];
                    dst.dataType = FLOAT32;
                    dst.Resize({cols, rows});
                    dst.Allocate();
                    for (int i = 0; i < rows; ++i)
                        for (int j = 0; j < cols; ++j)
                            reinterpret_cast<float *>(dst.cpuData)[(globalConv ? cols - 1 - j : j) * rows + i] =
                                reinterpret_cast<const float *>(src.cpuData)[i * cols + j];
                    continue;
                }
            } else if (part == "ffn_res_query.weight")
                target = root + "attn_res.mlp_queries." + std::to_string(layer);
            else if (part == "attn_res_query.weight" && layer > 0)
                target = root + "attn_res.attn_queries." + std::to_string(layer - 1);
            else if (part == "ffn_gate_up_exps.weight" || part == "ffn_down_exps.weight" ||
                     part == "ffn_gate_up_exps.scale" || part == "ffn_down_exps.scale") {
                const auto key = prefix + "mlp.experts." +
                                 (part.find("gate_up") != std::string::npos ? "w13_weight" : "w2_weight") +
                                 (part.substr(part.size() - 5) == "scale" ? "_scale" : "");
                expertTensors.emplace(key, inkling2::TensorRange{task.fileName, "GGUF", shape, task.offset,
                                                                 task.offset + ggml_nbytes(&task.tensor),
                                                                 int(task.tensor.type)});
                continue;
            } else if (blocks.count(part))
                target = prefix + blocks.at(part);
            else
                throw std::runtime_error("Unknown Inkling GGUF tensor: " + name);
        }
        names.push_back(target);
        task.name = std::move(target);
        dense.push_back(std::move(task));
    }
    if (messages != cfg.end - cfg.start) throw std::runtime_error("Incomplete Inkling GGUF cross-pass queries");
    const auto map = GetTensorMap(names);
    for (auto &t : dense) {
        auto type = map.at(t.name).at(0).second;
        t.replaceType = type == BFLOAT16  ? Rule::GGUFWeightReplaceForceBF16
                        : type == FLOAT16 ? Rule::GGUFWeightReplaceForceFP16
                                          : Rule::GGUFWeightReplaceForceFP32;
    }
    tasks = std::move(dense);
}
} // namespace fastllm
