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
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.nextn\.eh_proj\.weight)"),
                        "mtp.fc.weight"
                    ), // MTP eh_proj (only present on the nextn block)
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.nextn\.enorm\.weight)"),
                        "mtp.pre_fc_norm_embedding.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.nextn\.hnorm\.weight)"),
                        "mtp.pre_fc_norm_hidden.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.nextn\.shared_head_norm\.weight)"),
                        "mtp.norm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ssm_alpha\.weight)"),
                        "model.language_model.layers.$1.linear_attn.in_proj_a.weight",
                        GGUFWeightReplaceRule::GGUFWeightReplaceUntileVHeads
                    ), // linear_attn in_proj_a (untile V-heads; must precede ssm_a)
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ssm_beta\.weight)"),
                        "model.language_model.layers.$1.linear_attn.in_proj_b.weight",
                        GGUFWeightReplaceRule::GGUFWeightReplaceUntileVHeads
                    ), // linear_attn in_proj_b
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ssm_a)"),
                        "model.language_model.layers.$1.linear_attn.A_log",
                        GGUFWeightReplaceRule::GGUFWeightReplaceUntileVHeads,
                        true // untile V-heads then compose log(-x)
                    ), // A_log (untile V-heads, then invert -exp(A_log) baked into GGUF ssm_a)
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ssm_conv1d\.(weight|bias))"),
                        "model.language_model.layers.$1.linear_attn.conv1d.$2",
                        GGUFWeightReplaceRule::GGUFWeightReplaceUntileVHeads
                    ), // conv1d (untile V channels)
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ssm_dt\.bias)"),
                        "model.language_model.layers.$1.linear_attn.dt_bias",
                        GGUFWeightReplaceRule::GGUFWeightReplaceUntileVHeads
                    ), // dt_bias (untile V-head scalars)
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ssm_norm\.weight)"),
                        "model.language_model.layers.$1.linear_attn.norm.weight"
                    ), // linear_attn norm
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ssm_out\.weight)"),
                        "model.language_model.layers.$1.linear_attn.out_proj.weight"
                    ), // out_proj (stays Direct/tiled; runtime activation reorder before matmul)
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_qkv\.weight)"),
                        "model.language_model.layers.$1.linear_attn.in_proj_qkv.weight",
                        GGUFWeightReplaceRule::GGUFWeightReplaceUntileVHeads
                    ), // in_proj_qkv (untile V-segment rows only)
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_gate\.weight)"),
                        "model.language_model.layers.$1.linear_attn.in_proj_z.weight",
                        GGUFWeightReplaceRule::GGUFWeightReplaceUntileVHeads
                    ), // in_proj_z (gate; untile V rows)
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_(q|k|v)\.(weight|bias))"),
                        "model.language_model.layers.$1.self_attn.$2_proj.$3"
                    ), // full-attention qkv
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_(q|k)_norm\.weight)"),
                        "model.language_model.layers.$1.self_attn.$2_norm.weight"
                    ), // qk norm
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_output\.(weight|bias))"),
                        "model.language_model.layers.$1.self_attn.o_proj.$2"
                    ), // o
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.ffn_(gate|up|down)\.(weight|bias))"),
                        "model.language_model.layers.$1.mlp.$2_proj.$3"
                    ), // mlp
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.attn_norm\.weight)"),
                        "model.language_model.layers.$1.input_layernorm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(blk\.(\d+)\.post_attention_norm\.weight)"),
                        "model.language_model.layers.$1.post_attention_layernorm.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(token_embd\.weight)"),
                        "model.language_model.embed_tokens.weight",
                        GGUFWeightReplaceRule::GGUFWeightReplaceForceFP32
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(output\.weight)"),
                        "lm_head.weight"
                    ),
                    GGUFWeightReplaceRule (
                        std::regex(R"(output_norm\.weight)"),
                        "model.language_model.norm.weight"
                    )
                }
            },
            {
                "qwen3_5_mmproj",
                {
                    GGUFWeightReplaceRule(
                        std::regex(R"(^v\.patch_embd\.weight$)"),
                        "model.visual.patch_embed.proj.weight.0"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^v\.patch_embd\.weight\.1$)"),
                        "model.visual.patch_embed.proj.weight.1"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^v\.patch_embd\.bias$)"),
                        "model.visual.patch_embed.proj.bias"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^v\.position_embd\.weight$)"),
                        "model.visual.pos_embed.weight"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^v\.blk\.(\d+)\.ln1\.(weight|bias)$)"),
                        "model.visual.blocks.$1.norm1.$2"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^v\.blk\.(\d+)\.attn_qkv\.(weight|bias)$)"),
                        "model.visual.blocks.$1.attn.qkv.$2"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^v\.blk\.(\d+)\.attn_out\.(weight|bias)$)"),
                        "model.visual.blocks.$1.attn.proj.$2"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^v\.blk\.(\d+)\.ln2\.(weight|bias)$)"),
                        "model.visual.blocks.$1.norm2.$2"
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
                    )
                }
            },
            {
                "deepseek_v4",
                {
                    GGUFWeightReplaceRule(
                        std::regex(R"(^token_embd\.weight$)"),
                        "embed.weight",
                        GGUFWeightReplaceRule::GGUFWeightReplaceForceFP32
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^output\.weight$)"),
                        "head.weight"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^output_norm\.weight$)"),
                        "norm.weight"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^output_hc_base\.weight$)"),
                        "hc_head_base"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^output_hc_fn\.weight$)"),
                        "hc_head_fn"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^output_hc_scale\.weight$)"),
                        "hc_head_scale"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.attn_q_a\.weight$)"),
                        "layers.$1.attn.wq_a.weight"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.attn_q_a_norm\.weight$)"),
                        "layers.$1.attn.q_norm.weight"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.attn_q_b\.weight$)"),
                        "layers.$1.attn.wq_b.weight"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.attn_kv\.weight$)"),
                        "layers.$1.attn.wkv.weight"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.attn_kv_a_norm\.weight$)"),
                        "layers.$1.attn.kv_norm.weight"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.attn_output_a\.weight$)"),
                        "layers.$1.attn.wo_a.weight"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.attn_output_b\.weight$)"),
                        "layers.$1.attn.wo_b.weight"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.attn_sinks\.weight$)"),
                        "layers.$1.attn.attn_sink"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.attn_norm\.weight$)"),
                        "layers.$1.attn_norm.weight"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.attn_compressor_kv\.weight$)"),
                        "layers.$1.attn.compressor.wkv.weight"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.attn_compressor_gate\.weight$)"),
                        "layers.$1.attn.compressor.wgate.weight"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.attn_compressor_ape\.weight$)"),
                        "layers.$1.attn.compressor.ape"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.attn_compressor_norm\.weight$)"),
                        "layers.$1.attn.compressor.norm.weight"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.indexer\.attn_q_b\.weight$)"),
                        "layers.$1.attn.indexer.wq_b.weight"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.indexer\.proj\.weight$)"),
                        "layers.$1.attn.indexer.weights_proj.weight"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.indexer_compressor_kv\.weight$)"),
                        "layers.$1.attn.indexer.compressor.wkv.weight"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.indexer_compressor_gate\.weight$)"),
                        "layers.$1.attn.indexer.compressor.wgate.weight"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.indexer_compressor_ape\.weight$)"),
                        "layers.$1.attn.indexer.compressor.ape"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.indexer_compressor_norm\.weight$)"),
                        "layers.$1.attn.indexer.compressor.norm.weight"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.ffn_gate_inp\.weight$)"),
                        "layers.$1.ffn.gate.weight"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.exp_probs_b\.bias$)"),
                        "layers.$1.ffn.gate.bias"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.ffn_gate_tid2eid\.weight$)"),
                        "layers.$1.ffn.gate.tid2eid"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.ffn_gate_exps\.weight$)"),
                        std::vector<std::string>({"layers.$1.ffn.experts.", ".w1.weight"}),
                        GGUFWeightReplaceRule::GGUFWeightReplacePacked
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.ffn_up_exps\.weight$)"),
                        std::vector<std::string>({"layers.$1.ffn.experts.", ".w3.weight"}),
                        GGUFWeightReplaceRule::GGUFWeightReplacePacked
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.ffn_down_exps\.weight$)"),
                        std::vector<std::string>({"layers.$1.ffn.experts.", ".w2.weight"}),
                        GGUFWeightReplaceRule::GGUFWeightReplacePacked
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.ffn_gate_shexp\.weight$)"),
                        "layers.$1.ffn.shared_experts.w1.weight"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.ffn_up_shexp\.weight$)"),
                        "layers.$1.ffn.shared_experts.w3.weight"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.ffn_down_shexp\.weight$)"),
                        "layers.$1.ffn.shared_experts.w2.weight"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.ffn_norm\.weight$)"),
                        "layers.$1.ffn_norm.weight"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.hc_attn_base\.weight$)"),
                        "layers.$1.hc_attn_base"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.hc_attn_fn\.weight$)"),
                        "layers.$1.hc_attn_fn"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.hc_attn_scale\.weight$)"),
                        "layers.$1.hc_attn_scale"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.hc_ffn_base\.weight$)"),
                        "layers.$1.hc_ffn_base"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.hc_ffn_fn\.weight$)"),
                        "layers.$1.hc_ffn_fn"
                    ),
                    GGUFWeightReplaceRule(
                        std::regex(R"(^blk\.(\d+)\.hc_ffn_scale\.weight$)"),
                        "layers.$1.hc_ffn_scale"
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
            {"kimi_k2", originalArchRulesDict["deepseek_v2"]},
            {"deepseek4", originalArchRulesDict["deepseek_v4"]},
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
