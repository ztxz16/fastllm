#include "inkling2.h"
#include "gguf.h"
#include "json11.hpp"
#include <algorithm>
#include <cstring>
#include <set>

namespace fastllm {
namespace {
json11::Json::array localLayerIds(const json11::Json &pattern, int layers) {
    const auto &flags = pattern.array_items();
    if (int(flags.size()) != layers) throw std::runtime_error("Invalid Inkling GGUF sliding-window pattern");
    json11::Json::array local;
    for (int i = 0; i < layers; ++i) {
        if (!flags[i].is_bool()) throw std::runtime_error("Invalid Inkling GGUF sliding-window flag");
        if (flags[i].bool_value()) local.emplace_back(i);
    }
    return local;
}

GGUFWeightReplaceRule::GGUFWeightReplaceType denseImportType(DataType type) {
    using Rule = GGUFWeightReplaceRule;
    return type == BFLOAT16 ? Rule::GGUFWeightReplaceForceBF16
         : type == FLOAT16 ? Rule::GGUFWeightReplaceForceFP16 : Rule::GGUFWeightReplaceForceFP32;
}
} // namespace

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
    put("local_layer_ids", localLayerIds(get("attention.sliding_window_pattern"), layers));
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
    for (auto &t : dense) t.replaceType = denseImportType(map.at(t.name).at(0).second);
    tasks = std::move(dense);
}

void Inkling2Model::ConfigureDraftGGUF(const std::string &file, std::vector<ReadGGUFTask> &tasks) {
    json11::Json metadata;
    ReadGGUFMetaData(file, metadata);
    const auto &params = metadata["params"];
    if (params["general.architecture"].string_value() != "dflash")
        throw std::runtime_error("Inkling DSpark requires a dflash GGUF");
    auto get = [&](const std::string &key) -> const json11::Json & {
        const auto &v = params["dflash." + key];
        if (v.is_null()) throw std::runtime_error("Missing Inkling draft GGUF metadata: " + key);
        return v;
    };
    auto put = [&](const std::string &key, const json11::Json &v) { weight.AddDict("mtp_config." + key, v.dump()); };
    AppendGGUFTasks("inkling2_loop", file, tasks);
    auto head = std::find_if(tasks.begin(), tasks.end(), [](const auto &t) { return t.name == "markov_w2.weight"; });
    if (head == tasks.end() || head->tensor.dims.size() != 2)
        throw std::runtime_error("Missing Inkling folded draft projection");
    for (const auto &[key, source] : std::map<std::string, std::string>{
             {"hidden_size", "embedding_length"}, {"num_hidden_layers", "block_count"},
             {"num_attention_heads", "attention.head_count"}, {"num_key_value_heads", "attention.head_count_kv"},
             {"head_dim", "attention.key_length"}, {"intermediate_size", "feed_forward_length"},
             {"max_position_embeddings", "context_length"}, {"rms_norm_eps", "attention.layer_norm_rms_epsilon"},
             {"rope_theta", "rope.freq_base"}, {"block_size", "block_size"}, {"target_layer_ids", "target_layers"}})
        put(key, get(source));
    if (get("attention.key_length") != get("attention.value_length"))
        throw std::runtime_error("Invalid Inkling draft attention geometry");
    put("local_layer_ids", localLayerIds(get("attention.sliding_window_pattern"), get("block_count").int_value()));
    // GGUF counts the current position; the draft KV cache stores preceding tokens.
    put("sliding_window_size", get("attention.sliding_window").int_value() - 1);
    put("markov_rank", head->tensor.dims[1]);
    put("q_bias", false);
    put("o_bias", false);
    weight.AddDict("mtp_config.markov_head_type", "rnn_hidden");
}

void Inkling2Model::LoadDraftGGUF(std::vector<ReadGGUFTask> &tasks) {
    if (!draftEnabled()) throw std::runtime_error("Inkling GGUF draft requires DSpark proposals");
    const auto &d = cfg.draft;
    const int h = cfg.hidden, rank = d.rank, taps = d.taps.size();
    std::map<std::string, ReadGGUFTask *> byName;
    for (auto &task : tasks)
        if (!byName.emplace(task.name, &task).second) throw std::runtime_error("Duplicate draft tensor: " + task.name);
    auto read = [&](Data &dst, const std::string &name, const std::vector<int> &shape, DataType type = FLOAT32) {
        auto it = byName.find(name);
        if (it == byName.end() || it->second->tensor.dims != shape)
            throw std::runtime_error("Missing or incompatible Inkling draft tensor: " + name);
        auto &task = *it->second;
        WeightImportGGUFTensor(&dst, &task.tensor, task.fileName, task.offset, denseImportType(type));
        byName.erase(it);
    };
    const std::string root = "model.llm.draft.";
    auto load = [&](const std::string &gguf, const std::string &key, const std::vector<int> &shape) {
        read(weight[root + key], gguf, shape, shape.size() == 2 ? placement.denseType : FLOAT32);
    };
    load("token_embd_norm.weight", "embed_norm.weight", {h});
    load("output_norm.weight", "norm.weight", {h});
    load("markov_ctx_norm.weight", "markov_head.dflare_norm.weight", {h});
    load("joint_proj.weight", "markov_head.rnn_proj.weight", {3 * rank, 2 * rank + 2 * h});
    load("joint_proj.bias", "markov_head.rnn_proj.bias", {3 * rank});
    load("markov_w1.weight", "markov_head.markov_w1.weight", {cfg.vocab, rank});
    load("markov_w2.weight", "markov_head.folded.weight", {cfg.vocab, rank});
    load("markov_w2.bias", "markov_head.folded.bias", {cfg.vocab});

    Data masks;
    read(masks, "mask_embd.weight", {d.block - 1, h});
    const size_t rowBytes = size_t(h) * sizeof(float);
    for (int i = 1; i < d.block - 1; ++i)
        if (std::memcmp(masks.cpuData, masks.cpuData + i * rowBytes, rowBytes) != 0)
            throw std::runtime_error("Inkling requires a shared embedding for draft mask positions");
    auto &mask = weight[root + "mask_embed"];
    mask.dataType = FLOAT32; mask.Resize({h}); mask.Allocate();
    std::memcpy(mask.cpuData, masks.cpuData, rowBytes);

    auto &mix = weight[root + "fuse_prob"];
    mix.dataType = FLOAT32; mix.Resize({d.layers + 1, taps}); mix.Allocate();
    for (int i = 0; i <= d.layers; ++i) {
        const std::string blk = "blk." + std::to_string(i) + ".";
        Data row;
        read(row, i == d.layers ? "markov_ctx_mix.weight" : blk + "ctx_mix.weight", {taps});
        std::memcpy(mix.cpuData + size_t(i) * taps * sizeof(float), row.cpuData, size_t(taps) * sizeof(float));
        if (i == d.layers) break;
        const std::string layer = "layers." + std::to_string(i) + ".";
        auto proj = [&](const char *gguf, const char *key, std::vector<int> dims) {
            load(blk + gguf + ".weight", layer + key + ".weight", dims);
        };
        proj("ctx_norm", "dflare_norm", {h});
        proj("attn_norm", "input_layernorm", {h});
        proj("ffn_norm", "post_attention_layernorm", {h});
        proj("attn_q_norm", "self_attn.q_norm", {d.dim});
        proj("attn_k_norm", "self_attn.k_norm", {d.dim});
        proj("attn_k_ctx_norm", "self_attn.k_norm_ctx", {d.dim});
        proj("attn_q", "self_attn.q_proj", {d.heads * d.dim, h});
        proj("attn_k", "self_attn.k_proj", {d.kvHeads * d.dim, h});
        proj("attn_v", "self_attn.v_proj", {d.kvHeads * d.dim, h});
        proj("attn_k_ctx", "self_attn.k_proj_ctx", {d.kvHeads * d.dim, h});
        proj("attn_v_ctx", "self_attn.v_proj_ctx", {d.kvHeads * d.dim, h});
        proj("attn_output", "self_attn.o_proj", {h, d.heads * d.dim});
        proj("ffn_gate", "mlp.gate_proj", {d.intermediate, h});
        proj("ffn_up", "mlp.up_proj", {d.intermediate, h});
        proj("ffn_down", "mlp.down_proj", {h, d.intermediate});
    }
    if (!byName.empty()) throw std::runtime_error("Unknown Inkling draft tensor: " + byName.begin()->first);
}
} // namespace fastllm
