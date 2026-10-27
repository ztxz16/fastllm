#include "gguf.h"
#include "executor.h"

namespace fastllm {
    static std::vector<GGUFWeightReplaceRule> DeepSeekV41GGUFWeightRules(bool draft = false) {
        using Rule = GGUFWeightReplaceRule;
        std::vector<Rule> rules;
        auto add = [&](const std::string &pattern, const std::string &name,
                       Rule::GGUFWeightReplaceType type = Rule::GGUFWeightReplaceDirect) {
            rules.emplace_back(std::regex("^" + pattern + "$"), name, type);
        };
        const std::string layer = draft ? "mtp.$1." : "layers.$1.";
        if (draft) {
            add(R"(fc\.weight)", "mtp.0.main_proj.weight");
            add(R"(enc\.output_norm\.weight)", "mtp.0.main_norm.weight");
            add(R"(output_norm\.weight)", "draft_last.norm.weight");
            add(R"(markov_w1\.weight)", "draft_last.markov_head.embed.weight", Rule::GGUFWeightReplaceForceBF16);
            add(R"(markov_w2\.weight)", "draft_last.markov_head.head.weight", Rule::GGUFWeightReplaceForceBF16);
            add(R"(conf_proj\.weight)", "draft_last.confidence_head.proj.weight", Rule::GGUFWeightReplaceForceFP32);
        } else {
            add(R"(token_embd\.weight)", "embed.weight", Rule::GGUFWeightReplaceForceFP32);
            add(R"(output\.weight)", "head.weight");
            add(R"(output_norm\.weight)", "norm.weight");
            add(R"(engram_(token_map|primes|offsets|multipliers)\.weight)", "engram_$1.weight");
            add(R"(blk\.(\d+)\.engram_embd\.weight)", layer + "engram.embed.weight");
            add(R"(blk\.(\d+)\.engram_(q|k)\.weight)", layer + "engram.$2_weight", Rule::GGUFWeightReplaceForceFP32);
            add(R"(blk\.(\d+)\.engram_wkv\.weight)", layer + "engram.wkv.weight");
        }
        add(R"(blk\.(\d+)\.(attn_norm|ffn_norm)\.weight)", layer + "$2.weight");
        add(R"(blk\.(\d+)\.(hc_(?:attn|ffn)_(?:base|fn|scale))\.weight)", layer + "$2");
        add(R"(blk\.(\d+)\.attn_sinks\.weight)", layer + "attn.attn_sink");
        add(R"(blk\.(\d+)\.attn_q_(a|b)\.weight)", layer + "attn.wq_$2.weight");
        add(R"(blk\.(\d+)\.attn_output_(a|b)\.weight)", layer + "attn.wo_$2.weight");
        add(R"(blk\.(\d+)\.attn_kv\.weight)", layer + "attn.wkv.weight");
        add(R"(blk\.(\d+)\.attn_q_a_norm\.weight)", layer + "attn.q_norm.weight");
        add(R"(blk\.(\d+)\.attn_kv_a_norm\.weight)", layer + "attn.kv_norm.weight");
        add(R"(blk\.(\d+)\.attn_compressor_(kv|gate)\.weight)", layer + "attn.compressor.w$2.weight", Rule::GGUFWeightReplaceForceFP32);
        add(R"(blk\.(\d+)\.attn_compressor_norm\.weight)", layer + "attn.compressor.norm.weight");
        add(R"(blk\.(\d+)\.indexer\.attn_q_b\.weight)", layer + "attn.indexer.wq_b.weight", Rule::GGUFWeightReplaceForceBF16);
        add(R"(blk\.(\d+)\.indexer\.attn_k\.weight)", layer + "attn.indexer.wk.weight", Rule::GGUFWeightReplaceForceBF16);
        add(R"(blk\.(\d+)\.indexer\.proj\.weight)", layer + "attn.indexer.weights_proj.weight", Rule::GGUFWeightReplaceForceBF16);
        add(R"(blk\.(\d+)\.indexer\.k_norm\.weight)", layer + "attn.indexer.k_norm.weight");
        add(R"(blk\.(\d+)\.ffn_gate_inp\.weight)", layer + "ffn.gate.weight", Rule::GGUFWeightReplaceForceFP32);
        add(R"(blk\.(\d+)\.exp_probs_b\.bias)", layer + "ffn.gate.bias");
        if (!draft)
            add(R"(blk\.(\d+)\.exp_probs_b_vl\.bias)", layer + "ffn.gate.bias_vl", Rule::GGUFWeightReplaceForceFP32);
        for (const auto &projection : std::vector<std::pair<std::string, std::string>>{
                 {"gate", "w1"}, {"up", "w3"}, {"down", "w2"}}) {
            add(R"(blk\.(\d+)\.ffn_)" + projection.first + R"(_shexp\.weight)",
                layer + "ffn.shared_experts." + projection.second + ".weight");
            rules.emplace_back(std::regex(R"(^blk\.(\d+)\.ffn_)" + projection.first + R"(_exps\.weight$)"),
                std::vector<std::string>{layer + "ffn.experts.", "." + projection.second + ".weight"},
                Rule::GGUFWeightReplacePacked);
        }
        return rules;
    }

    static std::vector<GGUFWeightReplaceRule> DeepSeekV41VisionGGUFWeightRules() {
        using Rule = GGUFWeightReplaceRule;
        std::vector<Rule> rules;
        auto add = [&](const std::string &pattern, const std::string &name,
                       Rule::GGUFWeightReplaceType type = Rule::GGUFWeightReplaceForceFP16) {
            rules.emplace_back(std::regex("^" + pattern + "$"), name, type);
        };
        const std::string block = "vision.blocks.$1.";
        add(R"(mm\.([12])\.weight)", "aligner.w$1.weight");
        add(R"(mm\.([12])\.bias)", "aligner.w$1.bias", Rule::GGUFWeightReplaceForceFP32);
        add(R"(v\.blk\.(\d+)\.attn_qkv\.weight)", block + "attn.wqkv.weight");
        add(R"(v\.blk\.(\d+)\.attn_qkv\.bias)", block + "attn.wqkv.bias", Rule::GGUFWeightReplaceForceFP32);
        add(R"(v\.blk\.(\d+)\.attn_out\.weight)", block + "attn.wo.weight");
        add(R"(v\.blk\.(\d+)\.attn_out\.bias)", block + "attn.wo.bias", Rule::GGUFWeightReplaceForceFP32);
        add(R"(v\.blk\.(\d+)\.ffn_(gate|up)\.weight)", block + "mlp.$2.weight");
        add(R"(v\.blk\.(\d+)\.ffn_down\.weight)", block + "mlp.w2.weight");
        add(R"(v\.blk\.(\d+)\.ln([12])\.weight)", block + "norm$2.weight", Rule::GGUFWeightReplaceForceFP32);
        add(R"(v\.post_ln\.weight)", "vision.norm.weight", Rule::GGUFWeightReplaceForceFP32);
        add(R"(v\.patch_embd\.weight)", "vision.patch_embed.proj.weight");
        add(R"(v\.patch_embd\.bias)", "vision.patch_embed.proj.bias", Rule::GGUFWeightReplaceForceFP32);
        add(R"(v\.token_embd\.img_(start|end))", "image_$1", Rule::GGUFWeightReplaceForceFP32);
        add(R"(v\.image_newline)", "image_newline", Rule::GGUFWeightReplaceForceFP32);
        // Text-only backbone exports may omit these small image-routing biases.
        add(R"(blk\.(\d+)\.exp_probs_b_vl\.bias)", "layers.$1.ffn.gate.bias_vl", Rule::GGUFWeightReplaceForceFP32);
        return rules;
    }

    void AppendDeepSeekV41GGUFAux(basellm *model, const json11::Json &target,
                                 const std::string &path, bool vision,
                                 std::vector<ReadGGUFTask> &tasks) {
        json11::Json document;
        ReadGGUFMetaData(path, document);
        const auto &meta = document["params"];
        const auto require = [](bool ok, const std::string &message) {
            AssertInFastLLM(ok, "DeepSeek V4.1 GGUF add-on: " + message);
        };
        const auto add = [&](const std::string &key, const json11::Json &value) {
            require(!value.is_null(), "missing parameter " + key);
            model->weight.AddDict(key, value.is_string() ? value.string_value() : value.dump());
        };
        require(target["general.architecture"].string_value() == "deepseek41",
                "requires a DeepSeek V4.1 GGUF target");
        std::vector<ReadGGUFTask> auxiliary;
        int layers = 0;
        if (vision) {
            require(meta["general.architecture"].string_value() == "clip" &&
                    meta["clip.projector_type"].string_value() == "deepseek41v" &&
                    meta["clip.vision.projection_dim"] == target["deepseek41.embedding_length"],
                    "incompatible vision architecture/projection dimension");
            if (!meta["deepseek41.vision.source"].is_null())
                require(meta["deepseek41.vision.source"] == target["deepseek41.engram.source"],
                        "vision routing source does not match the backbone");
            for (const auto &pair : std::vector<std::pair<std::string, std::string>>{
                     {"hidden_size", "embedding_length"}, {"num_hidden_layers", "block_count"},
                     {"intermediate_size", "feed_forward_length"}, {"num_attention_heads", "attention.head_count"},
                     {"patch_size", "patch_size"}, {"downsample_ratio", "projector.scale_factor"}})
                add("vision_config." + pair.first, meta["clip.vision." + pair.second]);
            const auto &tokens = target["tokenizer.ggml.tokens"].array_items();
            int imageToken = -1;
            for (int i = 0; i < (int)tokens.size(); ++i)
                if (tokens[i].string_value() == "<｜deepseek_image｜>") imageToken = i;
            require(imageToken >= 0, "target tokenizer has no image token");
            add("image_token_id", imageToken);
            layers = meta["clip.vision.block_count"].int_value();
            require(layers > 0, "empty vision encoder");
            AppendGGUFTasks("deepseek_v41_mmproj", path, auxiliary);
            for (int i = 0; i < layers; ++i) {
                const std::string p = "vision.blocks." + std::to_string(i) + ".mlp.";
                model->weightMergeRules.push_back(WeightMergeRule({
                    WeightMergeRuleSingle({p + "gate.weight", p + "up.weight"}, p + "w1.weight", "linearSwiglu")}));
            }
        } else {
            require(meta["general.architecture"].string_value() == "dflash", "expected DSpark dflash architecture");
            for (const std::string &key : {"embedding_length", "attention.head_count", "attention.key_length",
                     "rope.dimension_count", "attention.q_lora_rank", "attention.output_lora_rank",
                     "attention.output_group_count", "hyper_connection.count", "expert_feed_forward_length"})
                require(!meta["dflash." + key].is_null() && meta["dflash." + key] == target["deepseek41." + key],
                        "draft/target mismatch: " + key);
            layers = meta["dflash.block_count"].int_value();
            require(layers > 0 && layers <= target["deepseek41.block_count"].int_value(), "invalid draft layer count");
            for (const auto &pair : std::vector<std::pair<std::string, std::string>>{
                     {"num_nextn_predict_layers", "block_count"}, {"dspark_block_size", "block_size"},
                     {"dspark_target_layer_ids", "target_layers"}, {"dspark_n_routed_experts", "expert_count"},
                     {"dspark_num_experts_per_tok", "expert_used_count"}})
                add(pair.first, meta["dflash." + pair.second]);
            add("dspark_noise_token_id", meta["tokenizer.ggml.mask_token_id"]);
            add("dspark_external_gguf", true);
            require(meta["dflash.block_size"].int_value() > 0 &&
                    !meta["dflash.target_layers"].array_items().empty(), "missing DSpark block/target layers");
            const auto &ratios = meta["dflash.attention.compress_ratios"].array_items();
            require((int)ratios.size() == layers, "missing draft compression ratios");
            for (const auto &ratio : ratios) require(ratio.int_value() == 0, "compressed draft attention is unsupported");
            AppendGGUFTasks("deepseek_v41_dspark", path, auxiliary);
        }
        std::map<std::string, std::vector<int>> shapes;
        for (auto &task : auxiliary) {
            if (task.name.compare(0, 11, "draft_last.") == 0)
                task.name = "mtp." + std::to_string(layers - 1) + "." + task.name.substr(11);
            if (task.name == "vision.patch_embed.proj.weight") {
                auto &t = task.tensor;
                require(t.dims.size() == 4 && t.dims[1] == 3, "invalid patch convolution");
                t.dims = {t.dims[0], t.dims[1] * t.dims[2] * t.dims[3]};
                t.ne[0] = t.dims[1]; t.ne[1] = t.dims[0]; t.ne[2] = t.ne[3] = 1;
                t.nb[1] = t.nb[0] * t.ne[0]; t.nb[2] = t.nb[3] = t.nb[1] * t.ne[1];
            }
            require(shapes.emplace(task.name, task.tensor.dims).second, "duplicate auxiliary tensor " + task.name);
        }
        const auto shape = [&](const std::string &name, const std::vector<int> &expected) {
            auto found = shapes.find(name);
            require(found != shapes.end() && found->second == expected, "missing/invalid tensor " + name);
        };
        const int hidden = target["deepseek41.embedding_length"].int_value();
        if (vision) {
            const int dim = meta["clip.vision.embedding_length"].int_value();
            const int inter = meta["clip.vision.feed_forward_length"].int_value();
            const int patch = meta["clip.vision.patch_size"].int_value();
            const int scale = meta["clip.vision.projector.scale_factor"].int_value();
            require(dim > 0 && inter > 0 && patch > 0 && scale > 0, "invalid vision dimensions");
            shape("vision.patch_embed.proj.weight", {dim, 3 * patch * patch});
            shape("vision.patch_embed.proj.bias", {dim});
            shape("vision.norm.weight", {dim});
            shape("aligner.w1.weight", {hidden, dim * scale * scale});
            shape("aligner.w2.weight", {hidden, hidden});
            for (const auto &name : {"aligner.w1.bias", "aligner.w2.bias", "image_start", "image_end", "image_newline"})
                shape(name, {hidden});
            for (int i = 0; i < layers; ++i) {
                const std::string p = "vision.blocks." + std::to_string(i);
                shape(p + ".attn.wqkv.weight", {3 * dim, dim});
                shape(p + ".attn.wqkv.bias", {3 * dim});
                shape(p + ".attn.wo.weight", {dim, dim});
                shape(p + ".attn.wo.bias", {dim});
                shape(p + ".norm1.weight", {dim}); shape(p + ".norm2.weight", {dim});
                shape(p + ".mlp.gate.weight", {inter, dim}); shape(p + ".mlp.up.weight", {inter, dim});
                shape(p + ".mlp.w2.weight", {dim, inter});
            }
            for (const auto &item : shapes)
                if (item.first.compare(0, 7, "layers.") == 0)
                    require(item.second == std::vector<int>{target["deepseek41.expert_count"].int_value()},
                            "invalid image routing bias " + item.first);
        } else {
            const std::string last = "mtp." + std::to_string(layers - 1);
            const auto markov = shapes[last + ".markov_head.embed.weight"];
            require(markov.size() == 2 && markov[1] > 0 &&
                    markov == shapes[last + ".markov_head.head.weight"], "invalid Markov head");
            require(markov[0] == (int)target["tokenizer.ggml.tokens"].array_items().size(), "draft vocabulary mismatch");
            add("dspark_markov_rank", markov[1]);
            require(shapes["mtp.0.main_proj.weight"] == std::vector<int>{hidden,
                hidden * (int)meta["dflash.target_layers"].array_items().size()}, "invalid draft projection");
            shape("mtp.0.main_norm.weight", {hidden}); shape(last + ".norm.weight", {hidden});
            shape(last + ".confidence_head.proj.weight", {1, hidden + markov[1]});
            const int experts = meta["dflash.expert_count"].int_value();
            const int inter = meta["dflash.expert_feed_forward_length"].int_value();
            const int hc = meta["dflash.hyper_connection.count"].int_value();
            const int hcOut = hc * (hc + 2);
            const int heads = meta["dflash.attention.head_count"].int_value();
            const int head = meta["dflash.attention.key_length"].int_value();
            const int qrank = meta["dflash.attention.q_lora_rank"].int_value();
            const int orank = meta["dflash.attention.output_lora_rank"].int_value();
            const int groups = meta["dflash.attention.output_group_count"].int_value();
            require(experts > 0 && inter > 0 && hc > 0 && groups > 0 && heads % groups == 0 &&
                    meta["dflash.expert_used_count"].int_value() > 0 &&
                    meta["dflash.expert_used_count"].int_value() <= experts, "invalid draft expert/attention configuration");
            for (int i = 0; i < layers; ++i) {
                const std::string p = "mtp." + std::to_string(i);
                shape(p + ".attn.wkv.weight", {head, hidden});
                shape(p + ".attn.wq_a.weight", {qrank, hidden});
                shape(p + ".attn.wq_b.weight", {heads * head, qrank});
                shape(p + ".attn.wo_a.weight", {groups * orank, heads / groups * head});
                shape(p + ".attn.wo_b.weight", {hidden, groups * orank});
                shape(p + ".attn.q_norm.weight", {qrank}); shape(p + ".attn.kv_norm.weight", {head});
                shape(p + ".attn.attn_sink", {heads});
                shape(p + ".ffn.gate.weight", {experts, hidden}); shape(p + ".ffn.gate.bias", {experts});
                for (const std::string &part : {"attn", "ffn"}) {
                    shape(p + "." + part + "_norm.weight", {hidden});
                    shape(p + ".hc_" + part + "_base", {hcOut});
                    shape(p + ".hc_" + part + "_scale", {3});
                    shape(p + ".hc_" + part + "_fn", {hcOut, hc * hidden});
                }
                for (int e = -1; e < experts; ++e) {
                    const std::string ep = p + ".ffn." + (e < 0 ? "shared_experts" : "experts." + std::to_string(e));
                    shape(ep + ".w1.weight", {inter, hidden}); shape(ep + ".w3.weight", {inter, hidden});
                    shape(ep + ".w2.weight", {hidden, inter});
                }
            }
        }
        require(!auxiliary.empty(), "no auxiliary tensors");
        tasks.insert(tasks.end(), auxiliary.begin(), auxiliary.end());
        printf("[Fastllm] DeepSeek V4.1: external %s GGUF %s (%zu mapped tensors)\n",
               vision ? "vision" : "DSpark", path.c_str(), auxiliary.size());
    }

    static std::vector<GGUFWeightReplaceRule> Glm5NextGGUFWeightRules() {
        using Rule = GGUFWeightReplaceRule;
        const std::string base = "model.language_model.";
        const std::string layer = base + "layers.$1.";
        std::vector<Rule> rules;
        auto add = [&](const std::string &pattern, const std::string &name,
                       Rule::GGUFWeightReplaceType type = Rule::GGUFWeightReplaceDirect) {
            rules.emplace_back(std::regex("^" + pattern + "$"), name, type);
        };
        add(R"(token_embd\.weight)", base + "embed_tokens.weight", Rule::GGUFWeightReplaceForceFP32);
        add(R"(output\.weight)", "lm_head.weight");
        add(R"(output_norm\.weight)", base + "norm.weight", Rule::GGUFWeightReplaceForceFP32);
        add(R"(blk\.(\d+)\.attn_norm\.weight)", layer + "input_layernorm.weight", Rule::GGUFWeightReplaceForceFP32);
        add(R"(blk\.(\d+)\.ffn_norm\.weight)", layer + "post_attention_layernorm.weight", Rule::GGUFWeightReplaceForceFP32);
        add(R"(blk\.(\d+)\.(hc_(?:attn|ffn)_(?:fn|base|scale))(?:\.weight)?)", layer + "$2", Rule::GGUFWeightReplaceForceFP32);
        add(R"(blk\.(\d+)\.attn_(q|k|v)\.weight)", layer + "self_attn.$2_proj.weight");
        add(R"(blk\.(\d+)\.attn_output\.weight)", layer + "self_attn.o_proj.weight");
        add(R"(blk\.(\d+)\.attn_(q_a|q_b)\.weight)", layer + "self_attn.$2_proj.weight");
        add(R"(blk\.(\d+)\.attn_kv_a_mqa\.weight)", layer + "self_attn.kv_a_proj_with_mqa.weight");
        add(R"(blk\.(\d+)\.attn_(q_a|kv_a)_norm\.weight)", layer + "self_attn.$2_layernorm.weight", Rule::GGUFWeightReplaceForceFP32);
        // GGUF transposes K per head and separates K/V. Restore the canonical
        // combined KV-B projection once after loading, before any GPU upload.
        add(R"(blk\.(\d+)\.attn_(k|v)_b\.weight)", layer + "self_attn.gguf_$2_b.weight", Rule::GGUFWeightReplaceForceFP32);
        add(R"(blk\.(\d+)\.ssm_(f_a|f_b|g_a|g_b)\.weight)", layer + "self_attn.$2_proj.weight");
        add(R"(blk\.(\d+)\.ssm_beta\.weight)", layer + "self_attn.b_proj.weight");
        add(R"(blk\.(\d+)\.ssm_conv1d_(q|k|v)\.weight)", layer + "self_attn.$2_conv1d.weight", Rule::GGUFWeightReplaceForceFP32);
        add(R"(blk\.(\d+)\.ssm_a)", layer + "self_attn.gguf_decay", Rule::GGUFWeightReplaceForceFP32);
        add(R"(blk\.(\d+)\.ssm_dt\.(?:bias|weight))", layer + "self_attn.dt_bias", Rule::GGUFWeightReplaceForceFP32);
        add(R"(blk\.(\d+)\.ssm_norm\.weight)", layer + "self_attn.o_norm.weight", Rule::GGUFWeightReplaceForceFP32);
        add(R"(blk\.(\d+)\.indexer\.attn_q_b\.weight)", layer + "self_attn.indexer.wq_b.weight");
        add(R"(blk\.(\d+)\.indexer\.attn_k\.weight)", layer + "self_attn.indexer.wk.weight");
        add(R"(blk\.(\d+)\.indexer\.k_norm\.(weight|bias))", layer + "self_attn.indexer.k_norm.$2", Rule::GGUFWeightReplaceForceFP32);
        add(R"(blk\.(\d+)\.indexer\.proj\.weight)", layer + "self_attn.indexer.weights_proj.weight", Rule::GGUFWeightReplaceForceFP32);
        add(R"(blk\.(\d+)\.indexer_compressor_ape(?:\.weight)?)", layer + "self_attn.indexer.index_kpool_compress_ape", Rule::GGUFWeightReplaceForceFP32);
        add(R"(blk\.(\d+)\.indexer_compressor_gate(?:\.weight)?)", layer + "self_attn.indexer.index_kpool_compress_gate");
        add(R"(blk\.(\d+)\.ffn_(gate|up|down)\.weight)", layer + "mlp.$2_proj.weight");
        add(R"(blk\.(\d+)\.ffn_gate_inp\.weight)", layer + "mlp.gate.weight", Rule::GGUFWeightReplaceForceFP32);
        add(R"(blk\.(\d+)\.exp_probs_b\.bias)", layer + "mlp.gate.e_score_correction_bias", Rule::GGUFWeightReplaceForceFP32);
        add(R"(blk\.(\d+)\.ffn_(gate|up|down)_shexp\.weight)", layer + "mlp.shared_experts.$2_proj.weight");
        rules.emplace_back(std::regex(R"(^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$)"),
            std::vector<std::string>{layer + "mlp.experts.", ".$2_proj.weight"}, Rule::GGUFWeightReplacePacked);
        add(R"(blk\.(\d+)\.nextn\.(enorm|hnorm)\.weight)", layer + "$2.weight", Rule::GGUFWeightReplaceForceFP32);
        add(R"(blk\.(\d+)\.nextn\.shared_head_norm\.weight)", layer + "shared_head.norm.weight", Rule::GGUFWeightReplaceForceFP32);
        add(R"(blk\.(\d+)\.nextn\.eh_proj\.weight)", layer + "eh_proj.weight");
        return rules;
    }

    static std::vector<GGUFWeightReplaceRule> Qwen4GGUFWeightRules() {
        using Rule = GGUFWeightReplaceRule;
        const std::string base = "model.language_model.";
        const std::string layer = base + "layers.$1.";
        std::vector<Rule> rules;
        auto add = [&](const std::string &pattern, const std::string &name,
                       Rule::GGUFWeightReplaceType type = Rule::GGUFWeightReplaceForceFP16) {
            rules.emplace_back(std::regex("^" + pattern + "$"), name, type);
        };
        // Keep large attention/output projections packed. Shared gate/up and
        // auxiliary merges retain FP16. GDN reorders activations at projection.
        add(R"(token_embd\.weight)", base + "embed_tokens.weight", Rule::GGUFWeightReplaceForceFP32);
        add(R"(output\.weight)", "lm_head.weight", Rule::GGUFWeightReplaceDirect);
        add(R"(per_layer_token_embd\.weight)", base + "ple_embedding.weight", Rule::GGUFWeightReplaceDirect);
        add(R"(output_hc_norm\.weight)", base + "hyper_connection_mixer.hc_norm.weight", Rule::GGUFWeightReplaceForceFP32);
        for (const auto &kind : {"down", "up"}) {
            add("output_hc_" + std::string(kind) + R"(\.weight)",
                base + "hyper_connection_mixer.input_mix_weight_" + kind + ".weight");
        }
        for (const auto &connection : {std::pair<std::string, std::string>{"attn", "attn_hyper_connection"},
                                      {"ffn", "mlp_hyper_connection"}}) {
            const std::string source = R"(blk\.(\d+)\.hc_)" + connection.first;
            const std::string target = layer + connection.second + ".";
            add(source + R"(_norm\.weight)", target + "hc_norm.weight", Rule::GGUFWeightReplaceForceFP32);
            add(source + R"(_(down|up)\.weight)", target + "input_mix_weight_$2.weight");
            add(source + R"(_inject\.weight)", target + "block_inject_weight.weight");
        }
        add(R"(blk\.(\d+)\.attn_(q|k|v)\.weight)", layer + "self_attn.$2_proj.weight", Rule::GGUFWeightReplaceDirect);
        add(R"(blk\.(\d+)\.attn_output\.weight)", layer + "self_attn.o_proj.weight", Rule::GGUFWeightReplaceDirect);
        add(R"(blk\.(\d+)\.attn_(q|k)_norm\.weight)", layer + "self_attn.$2_norm.weight", Rule::GGUFWeightReplaceForceFP32);
        add(R"(blk\.(\d+)\.indexer\.(q|k)_proj\.weight)", layer + "self_attn.indexer.index_$2_proj.weight");
        add(R"(blk\.(\d+)\.indexer\.(q|k)_norm\.weight)", layer + "self_attn.indexer.$2_layernorm.weight", Rule::GGUFWeightReplaceForceFP32);
        add(R"(blk\.(\d+)\.attn_qkv\.weight)", layer + "linear_attn.in_proj_qkv.weight", Rule::GGUFWeightReplaceDirect);
        add(R"(blk\.(\d+)\.attn_gate\.weight)", layer + "linear_attn.in_proj_z.weight", Rule::GGUFWeightReplaceDirect);
        add(R"(blk\.(\d+)\.ssm_beta\.weight)", layer + "linear_attn.in_proj_b.weight");
        add(R"(blk\.(\d+)\.ssm_alpha\.weight)", layer + "linear_attn.in_proj_a.weight");
        add(R"(blk\.(\d+)\.ssm_out\.weight)", layer + "linear_attn.out_proj.weight", Rule::GGUFWeightReplaceDirect);
        add(R"(blk\.(\d+)\.ssm_conv1d\.weight)", layer + "linear_attn.conv1d.weight", Rule::GGUFWeightReplaceForceFP32);
        add(R"(blk\.(\d+)\.ssm_a)", layer + "linear_attn.A_log", Rule::GGUFWeightReplaceForceFP32);
        add(R"(blk\.(\d+)\.ssm_dt\.bias)", layer + "linear_attn.dt_bias", Rule::GGUFWeightReplaceForceFP32);
        add(R"(blk\.(\d+)\.ssm_norm\.weight)", layer + "linear_attn.norm.weight", Rule::GGUFWeightReplaceForceFP32);
        add(R"(blk\.(\d+)\.ffn_gate_inp\.weight)", layer + "mlp.gate.weight");
        add(R"(blk\.(\d+)\.ffn_gate_inp_shexp\.weight)", layer + "mlp.shared_expert_gate.weight");
        add(R"(blk\.(\d+)\.ffn_(gate|up|down)_shexp\.weight)", layer + "mlp.shared_expert.$2_proj.weight");
        rules.emplace_back(std::regex(R"(^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$)"),
            std::vector<std::string>{layer + "mlp.experts.", ".$2_proj.weight"}, Rule::GGUFWeightReplacePacked);
        add(R"(blk\.(\d+)\.ple_(key|value)\.weight)", layer + "ple.$2_proj.weight");
        add(R"(blk\.(\d+)\.ple_norm_(key|query|conv)\.weight)", layer + "ple.norm_$2.weight", Rule::GGUFWeightReplaceForceFP32);
        add(R"(blk\.(\d+)\.ple_conv1d\.weight)", layer + "ple.conv1d.weight", Rule::GGUFWeightReplaceForceFP32);
        // External Qwen4 MTP GGUFs retain the HF tensor names. Split packed
        // routed experts before NUMA registration, preserving ordinary blocks.
        rules.emplace_back(std::regex(R"(^mtp\.layers\.0\.mlp\.experts\.gate_up_proj$)"),
            std::vector<std::string>{"mtp.layers.0.mlp.experts.", ".gateup_proj.weight"}, Rule::GGUFWeightReplacePacked);
        rules.emplace_back(std::regex(R"(^mtp\.layers\.0\.mlp\.experts\.down_proj$)"),
            std::vector<std::string>{"mtp.layers.0.mlp.experts.", ".down_proj.weight"}, Rule::GGUFWeightReplacePacked);
        add(R"((mtp\..*norm.*\.weight))", "$1", Rule::GGUFWeightReplaceForceFP32);
        add(R"((mtp\.(?!.*norm).*\.weight))", "$1");
        return rules;
    }

    std::vector <GGUFWeightReplaceRule> GetGGUFWeightReplaceRules(const std::string &arch) {
        static std::map <std::string, std::vector <GGUFWeightReplaceRule> > originalArchRulesDict = {
            {"deepseek_v41", DeepSeekV41GGUFWeightRules()},
            {"deepseek_v41_dspark", DeepSeekV41GGUFWeightRules(true)},
            {"deepseek_v41_mmproj", DeepSeekV41VisionGGUFWeightRules()},
            {"glm5_next", Glm5NextGGUFWeightRules()},
            {"qwen4_exp", Qwen4GGUFWeightRules()},
            {
                "default", 
                {
                    GGUFWeightReplaceRule ( 
                        std::regex(R"(blk\.(\d+)\.attn_(q|k|v)\.(weight|bias))"),
                        "model.layers.$1.self_attn.$2_proj.$3"
                    ), // qkv
                    GGUFWeightReplaceRule ( 
                        std::regex(R"(blk\.(\d+)\.attn_(q|k)_norm\.weight)"),
                        "model.layers.$1.self_attn.$2_norm.weight"
                    ), // qk norm
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_output\.(weight|bias))"),
                        "model.layers.$1.self_attn.o_proj.$2"
                    ), // o 
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ffn_(gate|up|down)\.(weight|bias))"),
                        "model.layers.$1.mlp.$2_proj.$3"
                    ), // mlp 
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_norm\.weight)"),
                        "model.layers.$1.input_layernorm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ffn_norm\.weight)"),
                        "model.layers.$1.post_attention_layernorm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(token_embd.weight)"),
                        "model.embed_tokens.weight", 
                        GGUFWeightReplaceRule::GGUFWeightReplaceForceFP32
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(output.weight)"),
                        "lm_head.weight"
                    ), 
                    GGUFWeightReplaceRule (
                        std::regex(R"(output_norm.weight)"),
                        "model.norm.weight"
                    ),

                    GGUFWeightReplaceRule (
                        std::regex(R"(blk.(\d+).ffn_(gate|up|down)_exps.weight)"),
                        std::vector <std::string> ({"model.layers.$1.mlp.experts.", ".$2_proj.weight"}),
                        GGUFWeightReplaceRule::GGUFWeightReplacePacked
                    ), // experts
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk.(\d+).ffn_(gate|up|down)_shexp.weight)"),
                        "model.layers.$1.mlp.shared_experts.$2_proj.weight"
                    ) // shared experts
                }
            },
            {
                "qwen3_moe", 
                {
                    GGUFWeightReplaceRule ( 
                        std::regex(R"(blk\.(\d+)\.attn_(q|k|v)\.(weight|bias))"),
                        "model.layers.$1.self_attn.$2_proj.$3"
                    ), // qkv
                    GGUFWeightReplaceRule ( 
                        std::regex(R"(blk\.(\d+)\.attn_(q|k)_norm\.weight)"),
                        "model.layers.$1.self_attn.$2_norm.weight"
                    ), // qk norm
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_output\.(weight|bias))"),
                        "model.layers.$1.self_attn.o_proj.$2"
                    ), // o 
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ffn_(gate|up|down)\.(weight|bias))"),
                        "model.layers.$1.mlp.$2_proj.$3"
                    ), // mlp 
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_norm\.weight)"),
                        "model.layers.$1.input_layernorm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ffn_norm\.weight)"),
                        "model.layers.$1.post_attention_layernorm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(token_embd.weight)"),
                        "model.embed_tokens.weight", 
                        GGUFWeightReplaceRule::GGUFWeightReplaceForceFP32
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(output.weight)"),
                        "lm_head.weight"
                    ), 
                    GGUFWeightReplaceRule (
                        std::regex(R"(output_norm.weight)"),
                        "model.norm.weight"
                    ),

                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ffn_gate_inp\.weight)"),
                        "model.layers.$1.mlp.gate.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk.(\d+).ffn_(gate|up|down)_exps.weight)"),
                        std::vector <std::string> ({"model.layers.$1.mlp.experts.", ".$2_proj.weight"}),
                        GGUFWeightReplaceRule::GGUFWeightReplacePacked
                    ), // experts
                }
            },
            {
                "qwen3_5",
                {
                    // UD quants intentionally assign different GGML types to
                    // Q, K and V.  Keep those packed types: Qwen3.5's CUDA
                    // path can execute the three projections separately when
                    // the loader cannot form a homogeneous merged tensor.
                    GGUFWeightReplaceRule (
                        std::regex(R"(^blk\.(\d+)\.attn_(q|k|v)\.weight$)"),
                        "model.language_model.layers.$1.self_attn.$2_proj.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(^blk\.(\d+)\.attn_(q|k|v)\.bias$)"),
                        "model.language_model.layers.$1.self_attn.$2_proj.bias"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(^blk\.(\d+)\.attn_(q|k)_norm\.weight$)"),
                        "model.language_model.layers.$1.self_attn.$2_norm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(^blk\.(\d+)\.attn_output\.(weight|bias)$)"),
                        "model.language_model.layers.$1.self_attn.o_proj.$2"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(^blk\.(\d+)\.ffn_(gate|up|down)\.(weight|bias)$)"),
                        "model.language_model.layers.$1.mlp.$2_proj.$3"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(^blk\.(\d+)\.attn_norm\.weight$)"),
                        "model.language_model.layers.$1.input_layernorm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(^blk\.(\d+)\.post_attention_norm\.weight$)"),
                        "model.language_model.layers.$1.post_attention_layernorm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(^token_embd\.weight$)"),
                        "model.language_model.embed_tokens.weight",
                        GGUFWeightReplaceRule::GGUFWeightReplaceForceFP32
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(^output\.weight$)"),
                        "lm_head.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(^output_norm\.weight$)"),
                        "model.language_model.norm.weight"
                    ),

                    // Qwen3.5 GDN.  Keep heterogeneous qkv/z weights quantized;
                    // the CUDA forward path falls back to two packed GGUF
                    // projections when they cannot be merged losslessly.
                    GGUFWeightReplaceRule (
                        std::regex(R"(^blk\.(\d+)\.attn_qkv\.weight$)"),
                        "model.language_model.layers.$1.linear_attn.in_proj_qkv.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(^blk\.(\d+)\.attn_gate\.weight$)"),
                        "model.language_model.layers.$1.linear_attn.in_proj_z.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(^blk\.(\d+)\.ssm_beta\.weight$)"),
                        "model.language_model.layers.$1.linear_attn.in_proj_b.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(^blk\.(\d+)\.ssm_alpha\.weight$)"),
                        "model.language_model.layers.$1.linear_attn.in_proj_a.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(^blk\.(\d+)\.ssm_conv1d\.weight$)"),
                        "model.language_model.layers.$1.linear_attn.conv1d.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(^blk\.(\d+)\.ssm_a$)"),
                        "model.language_model.layers.$1.linear_attn.A_log"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(^blk\.(\d+)\.ssm_dt\.bias$)"),
                        "model.language_model.layers.$1.linear_attn.dt_bias"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(^blk\.(\d+)\.ssm_norm\.weight$)"),
                        "model.language_model.layers.$1.linear_attn.norm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(^blk\.(\d+)\.ssm_out\.weight$)"),
                        "model.language_model.layers.$1.linear_attn.out_proj.weight"
                    ),

                    // Qwen3.5 GGUF stores the MTP input projection and norms
                    // under the optional NextN block.  The surrounding
                    // transformer weights use the ordinary blk.N names and are
                    // remapped relative to the main-layer count by the GGUF
                    // loader.
                    GGUFWeightReplaceRule (
                        std::regex(R"(^blk\.\d+\.nextn\.eh_proj\.weight$)"),
                        "mtp.fc.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(^blk\.\d+\.nextn\.enorm\.weight$)"),
                        "mtp.pre_fc_norm_embedding.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(^blk\.\d+\.nextn\.hnorm\.weight$)"),
                        "mtp.pre_fc_norm_hidden.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(^blk\.\d+\.nextn\.shared_head_norm\.weight$)"),
                        "mtp.norm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(^blk\.\d+\.nextn\..*$)"),
                        "ignore"
                    )
                }
            },
            {
                "qwen3_5_mmproj",
                {
                    GGUFWeightReplaceRule(
                        std::regex(R"(^v\.blk\.(\d+)\.attn_qkv\.(weight|bias)$)"),
                        "model.visual.blocks.$1.attn.qkv.$2"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^v\.blk\.(\d+)\.attn_out\.(weight|bias)$)"),
                        "model.visual.blocks.$1.attn.proj.$2"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^v\.blk\.(\d+)\.ffn_up\.(weight|bias)$)"),
                        "model.visual.blocks.$1.mlp.linear_fc1.$2"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^v\.blk\.(\d+)\.ffn_down\.(weight|bias)$)"),
                        "model.visual.blocks.$1.mlp.linear_fc2.$2"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^v\.blk\.(\d+)\.ln1\.(weight|bias)$)"),
                        "model.visual.blocks.$1.norm1.$2"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^v\.blk\.(\d+)\.ln2\.(weight|bias)$)"),
                        "model.visual.blocks.$1.norm2.$2"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^v\.post_ln\.(weight|bias)$)"),
                        "model.visual.merger.norm.$1"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^mm\.0\.(weight|bias)$)"),
                        "model.visual.merger.linear_fc1.$1"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^mm\.2\.(weight|bias)$)"),
                        "model.visual.merger.linear_fc2.$1"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^v\.position_embd\.weight$)"),
                        "model.visual.pos_embed.weight"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^v\.patch_embd\.bias$)"),
                        "model.visual.patch_embed.proj.bias"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^v\.patch_embd\.weight\.1$)"),
                        "model.visual.patch_embed.proj.weight.gguf_part1"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^v\.patch_embd\.weight$)"),
                        "model.visual.patch_embed.proj.weight.gguf_part0"
                    )
                }
            },
            {
                "deepseek_v2", 
                {
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_q_(a|b)\.(weight|bias))"),
                        "model.layers.$1.self_attn.q_$2_proj.$3"
                    ), // q_a, q_b
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_kv_a_mqa\.(weight|bias))"),
                        "model.layers.$1.self_attn.kv_a_proj_with_mqa.$2"
                    ), // kv_a_mqa
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_kv_a_norm\.weight)"),
                        "model.layers.$1.self_attn.kv_a_layernorm.weight"
                    ), // kv_a_layernorm
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_kv_b\.(weight|bias))"),
                        "model.layers.$1.self_attn.kv_b_proj.$2",
                        GGUFWeightReplaceRule::GGUFWeightReplaceForceFP16
                    ), // kv_b
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_k_b\.(weight|bias))"),
                        "model.layers.$1.self_attn.kv_b_proj.$2__0",
                        GGUFWeightReplaceRule::GGUFWeightReplaceForceFP16
                    ), // k_b, v_b，有时候这两个分开了
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_v_b\.(weight|bias))"),
                        "model.layers.$1.self_attn.kv_b_proj.$2__1",
                        GGUFWeightReplaceRule::GGUFWeightReplaceForceFP16
                    ), // k_b, v_b，有时候这两个分开了
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_output\.(weight|bias))"),
                        "model.layers.$1.self_attn.o_proj.$2"
                    ), // o 
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_norm\.weight)"),
                        "model.layers.$1.input_layernorm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_q_a_norm\.weight)"),
                        "model.layers.$1.self_attn.q_a_layernorm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ffn_norm\.weight)"),
                        "model.layers.$1.post_attention_layernorm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(token_embd.weight)"),
                        "model.embed_tokens.weight", 
                        GGUFWeightReplaceRule::GGUFWeightReplaceForceFP32
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(output.weight)"),
                        "lm_head.weight"
                    ), 
                    GGUFWeightReplaceRule (
                        std::regex(R"(output_norm.weight)"),
                        "model.norm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ffn_(gate|up|down)\.(weight|bias))"),
                        "model.layers.$1.mlp.$2_proj.$3"
                    ), // mlp 
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ffn_gate_inp\.weight)"),
                        "model.layers.$1.mlp.gate.weight"
                    ), // gate weight
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.exp_probs_b\.bias)"),
                        "model.layers.$1.mlp.gate.e_score_correction_bias"
                    ), // gate bias
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk.(\d+).ffn_(gate|up|down)_exps.weight)"),
                        std::vector <std::string> ({"model.layers.$1.mlp.experts.", ".$2_proj.weight"}),
                        GGUFWeightReplaceRule::GGUFWeightReplacePacked
                    ), // experts
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk.(\d+).ffn_(gate|up|down)_shexp.weight)"),
                        "model.layers.$1.mlp.shared_experts.$2_proj.weight"
                    ) // shared experts
                }
            },
            {
                "minimax_m2", 
                {
                    GGUFWeightReplaceRule ( 
                        std::regex(R"(blk\.(\d+)\.attn_(q|k|v)\.(weight|bias))"),
                        "model.layers.$1.self_attn.$2_proj.$3"
                    ), // qkv
                    GGUFWeightReplaceRule ( 
                        std::regex(R"(blk\.(\d+)\.attn_(q|k)_norm\.weight)"),
                        "model.layers.$1.self_attn.$2_norm.weight"
                    ), // qk norm
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_output\.(weight|bias))"),
                        "model.layers.$1.self_attn.o_proj.$2"
                    ), // o 
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_norm\.weight)"),
                        "model.layers.$1.input_layernorm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ffn_norm\.weight)"),
                        "model.layers.$1.post_attention_layernorm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(token_embd.weight)"),
                        "model.embed_tokens.weight", 
                        GGUFWeightReplaceRule::GGUFWeightReplaceForceFP32
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(output.weight)"),
                        "lm_head.weight"
                    ), 
                    GGUFWeightReplaceRule (
                        std::regex(R"(output_norm.weight)"),
                        "model.norm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ffn_gate_inp\.weight)"),
                        "model.layers.$1.block_sparse_moe.gate.weight"
                    ), // gate weight
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.exp_probs_b\.bias)"),
                        "model.layers.$1.block_sparse_moe.e_score_correction_bias"
                    ), // gate bias
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk.(\d+).ffn_gate_exps.weight)"),
                        std::vector <std::string> ({"model.layers.$1.block_sparse_moe.experts.", ".w1.weight"}),
                        GGUFWeightReplaceRule::GGUFWeightReplacePacked
                    ), // experts gate -> w1
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk.(\d+).ffn_up_exps.weight)"),
                        std::vector <std::string> ({"model.layers.$1.block_sparse_moe.experts.", ".w3.weight"}),
                        GGUFWeightReplaceRule::GGUFWeightReplacePacked
                    ), // experts up -> w3
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk.(\d+).ffn_down_exps.weight)"),
                        std::vector <std::string> ({"model.layers.$1.block_sparse_moe.experts.", ".w2.weight"}),
                        GGUFWeightReplaceRule::GGUFWeightReplacePacked
                    ), // experts down -> w2
                }
            },
            {
                "glm4_moe", 
                {
                    GGUFWeightReplaceRule ( 
                        std::regex(R"(blk\.(\d+)\.attn_(q|k|v)\.(weight|bias))"),
                        "model.layers.$1.self_attn.$2_proj.$3"
                    ), // qkv
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_output\.(weight|bias))"),
                        "model.layers.$1.self_attn.o_proj.$2"
                    ), // o 
                    GGUFWeightReplaceRule ( 
                        std::regex(R"(blk\.(\d+)\.attn_(q|k)_norm\.weight)"),
                        "model.layers.$1.self_attn.$2_norm.weight"
                    ), // qk norm
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ffn_(gate|up|down)\.(weight|bias))"),
                        "model.layers.$1.mlp.$2_proj.$3"
                    ), // mlp 
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ffn_gate_inp\.weight)"),
                        "model.layers.$1.mlp.gate.weight"
                    ), // gate weight
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.exp_probs_b\.bias)"),
                        "model.layers.$1.mlp.gate.e_score_correction_bias"
                    ), // gate bias
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk.(\d+).ffn_(gate|up|down)_exps.weight)"),
                        std::vector <std::string> ({"model.layers.$1.mlp.experts.", ".$2_proj.weight"}),
                        GGUFWeightReplaceRule::GGUFWeightReplacePacked
                    ), // experts
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk.(\d+).ffn_(gate|up|down)_shexp.weight)"),
                        "model.layers.$1.mlp.shared_experts.$2_proj.weight"
                    ), // shared experts
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.post_attention_norm\.weight)"),
                        "model.layers.$1.post_attention_layernorm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_norm\.weight)"),
                        "model.layers.$1.input_layernorm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(token_embd.weight)"),
                        "model.embed_tokens.weight", 
                        GGUFWeightReplaceRule::GGUFWeightReplaceForceFP32
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(output.weight)"),
                        "lm_head.weight"
                    ), 
                    GGUFWeightReplaceRule (
                        std::regex(R"(output_norm.weight)"),
                        "model.norm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(.*nextn.*)"),
                        "ignore"
                    ), // ignore
                }
            }
        };

        if (arch == "glm-dsa" || arch == "glm_moe_dsa") {
            auto rules = originalArchRulesDict["deepseek_v2"];
            // llama.cpp currently runs GLM_DSA with the DeepSeek2 graph, so indexer tensors are unused.
            rules.insert(rules.begin(), GGUFWeightReplaceRule(std::regex(R"(blk\.(\d+)\.indexer\..*)"), "ignore"));
            const auto deviceMap = GetDeviceMap();
            bool mainDeviceUsesCuda = false;
            if (deviceMap.empty()) {
                // An empty map means the executor's current default is used. On a
                // CUDA-capable host that default is CUDA, so treating an empty map
                // as CPU-only would leave V-B quantized for the CUDA attention path.
                auto *executor = static_cast<Executor*>(GetExecutor());
                if (executor != nullptr) {
                    const std::string firstDevice = executor->GetFirstDeviceType();
                    mainDeviceUsesCuda = firstDevice == "cuda" || firstDevice == "multicuda";
                }
            } else {
                for (const auto &device : deviceMap) {
                    if (device.first.find("cuda") != std::string::npos) {
                        mainDeviceUsesCuda = true;
                        break;
                    }
                }
            }
            if (!mainDeviceUsesCuda) {
                // V-B is already laid out as [heads, v_head_dim, kv_lora_rank], so a
                // pure CPU model can keep it quantized and view it directly as a 2-D
                // linear weight. CUDA's absorbed-attention path still consumes V-B
                // through MatMulTransB, which currently requires FP16/FP32; in that
                // case retain the original DeepSeek2 ForceFP16 rule below.
                rules.insert(rules.begin(), GGUFWeightReplaceRule(
                    std::regex(R"(blk\.(\d+)\.attn_v_b\.(weight|bias))"),
                    "model.layers.$1.self_attn.kv_b_proj.$2__1"));
            }
            return rules;
        }

        static std::map <std::string, std::vector <GGUFWeightReplaceRule> > archRulesDict = {
            {"qwen2", originalArchRulesDict["default"]},
            {"qwen35", originalArchRulesDict["qwen3_5"]},
            {"kimi_k2", originalArchRulesDict["deepseek_v2"]},
        };

        for (auto &it : originalArchRulesDict) {
            if (archRulesDict.find(it.first) == archRulesDict.end()) {
                archRulesDict[it.first] = it.second;
            }
        }

        if (archRulesDict.find(arch) != archRulesDict.end()) {
            return archRulesDict[arch];
        }

        printf("Warning: gguf arch %s not found, use default arch.\n", arch.c_str());
        return archRulesDict["default"];
    }
}
