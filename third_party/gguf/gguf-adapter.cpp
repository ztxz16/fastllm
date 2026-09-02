#include "gguf.h"
#include "executor.h"

namespace fastllm {
    std::vector <GGUFWeightReplaceRule> GetGGUFWeightReplaceRules(const std::string &arch) {
        static std::map <std::string, std::vector <GGUFWeightReplaceRule> > originalArchRulesDict = {
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
