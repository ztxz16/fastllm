#include "graphllm.h"

namespace fastllm {
    class Minicpm3GraphModelConfig : GraphLLMModelConfig {
    public:
        void InitParams(GraphLLMModel *model) {
            model->rotary_dim = atoi(model->weight.dicts["qk_rope_head_dim"].c_str());
        }

        std::map <std::string, std::vector <std::pair <std::string, DataType> > >
                GetTensorMap(GraphLLMModel *model, const std::vector <std::string> &tensorNames) {
            std::map <std::string, std::vector <std::pair <std::string, DataType> > > ret;
            std::string embeddingName = "model.embed_tokens.weight";
            std::string logitsName = "lm_head.weight";
            std::set <std::string> linearNames = {
                ".self_attn.q_a_proj.weight",
                ".self_attn.q_b_proj.weight",
                ".self_attn.kv_a_proj_with_mqa.weight",
                ".self_attn.kv_b_proj.weight",
                ".self_attn.o_proj.weight",
                ".mlp.gate_proj.weight", 
                ".mlp.up_proj.weight", 
                ".mlp.down_proj.weight"
            };
            ret[embeddingName].push_back(std::make_pair(embeddingName, DataType::DATA_AUTO_EMBEDDING));
            for (int i = 0; i < model->block_cnt; i++) {
                std::string pre = "model.layers." + std::to_string(i);
                for (auto &it : linearNames) {
                    ret[pre + it].push_back(std::make_pair(pre + it, DataType::DATA_AUTO_LINEAR));
                }
            }
            for (auto &name : tensorNames) {
                if (ret[name].size() == 0) {
                    ret[name].push_back(std::make_pair(name, DataType::DATA_AUTO_NONE));
                }
            }
            if (ret.find(logitsName) == ret.end()) {
                ret[embeddingName].push_back(std::make_pair(logitsName, DataType::DATA_AUTO_LINEAR));
            } else {
                ret[logitsName][0].second = DataType::DATA_AUTO_LINEAR;
            }
            return ret;
        }

        void BuildGraph(GraphLLMModel *model) {     
            int qk_nope_head_dim = atoi(model->weight.dicts["qk_nope_head_dim"].c_str());
            int qk_rope_head_dim = atoi(model->weight.dicts["qk_rope_head_dim"].c_str());
            int kv_lora_rank = atoi(model->weight.dicts["kv_lora_rank"].c_str());
            int v_head_dim = atoi(model->weight.dicts["v_head_dim"].c_str());
            float scale_depth = atof(model->weight.dicts["scale_depth"].c_str());
            float attention_scale = scale_depth / std::sqrt(model->block_cnt);
            int dim_model_base = atoi(model->weight.dicts["dim_model_base"].c_str());
            float rms_scale = 1.f / (model->embed_dim / dim_model_base);

            auto &graph = *(model->GetGraph());
            std::map <std::string, ComputeGraphNode> wNodes;
            for (auto &it : model->weight.weight) {
                wNodes[it.first] = ComputeGraphNode(it.first);
            }
            ComputeGraphNode inputIds("inputIds"), positionIds("positionIds"), attentionMask("attentionMask"), atype("atype"), sin("sin"), cos("cos"), seqLens("seqLens");
            ComputeGraphNode hiddenStates("hiddenStates"), attenInput("attenInput"), attenOutput("attenOutput"), attenLastOutput("attenLastOutput");
            ComputeGraphNode qa("qa"), qa_norm("qa_norm"), qb("qb"), q_nope("q_nope"), q_rope("q_rope"), qkv("qkv"), kva("kva"), kv_norm("kv_norm"), kvb("kvb"), compressed_kv("compressed_kv"), k_rope("k_rope"), k_nope("k_nope"), q("q"), k("k"), v("v"), w1("w1"), w2("w2"), w3("w3"), lastTokensStates("lastTokensStates"), logits("logits");
            ComputeGraphNode k_rope_expand("k_rope_expand");
            graph.Embedding(inputIds, wNodes["model.embed_tokens.weight"], hiddenStates);
            graph.Mul(hiddenStates, atof(model->weight.dicts["scale_emb"].c_str()), hiddenStates);
            graph.DataTypeAs(hiddenStates, atype);
            for (int i = 0; i < model->block_cnt; i++) {
                std::string pre = "model.layers." + std::to_string(i);
                ComputeGraphNode pastKey("pastKey." + std::to_string(i)), pastValue("pastValue." + std::to_string(i));
                graph.RMSNorm(hiddenStates, wNodes[pre + ".input_layernorm.weight"], model->rms_norm_eps, attenInput);
                graph.Linear(attenInput, wNodes[pre + ".self_attn.q_a_proj.weight"], wNodes[pre + ".self_attn.q_a_proj.bias"], qa);
                graph.RMSNorm(qa, wNodes[pre + ".self_attn.q_a_layernorm.weight"], model->rms_norm_eps, qa_norm);
                graph.Linear(qa_norm, wNodes[pre + ".self_attn.q_b_proj.weight"], wNodes[pre + ".self_attn.q_b_proj.bias"], qb);
                graph.ExpandHead(qb, qk_nope_head_dim + qk_rope_head_dim);
                graph.Split(qb, -1, 0, qk_nope_head_dim, q_nope);
                graph.Split(qb, -1, qk_nope_head_dim, qk_nope_head_dim + qk_rope_head_dim, q_rope);
                graph.Linear(attenInput, wNodes[pre + ".self_attn.kv_a_proj_with_mqa.weight"], wNodes[pre + ".self_attn.kv_a_proj_with_mqa.bias"], kva);
                graph.Split(kva, -1, 0, kv_lora_rank, compressed_kv);
                graph.Split(kva, -1, kv_lora_rank, kv_lora_rank + qk_rope_head_dim, k_rope);
                graph.ExpandHead(k_rope, qk_rope_head_dim);
                graph.RMSNorm(compressed_kv, wNodes[pre + ".self_attn.kv_a_layernorm.weight"], model->rms_norm_eps, kv_norm);
                graph.Linear(kv_norm, wNodes[pre + ".self_attn.kv_b_proj.weight"], wNodes[pre + ".self_attn.kv_b_proj.bias"], kvb);
                graph.ExpandHead(kvb, qk_nope_head_dim + v_head_dim);
                graph.Split(kvb, -1, 0, qk_nope_head_dim, k_nope);
                graph.Split(kvb, -1, qk_nope_head_dim, qk_nope_head_dim + v_head_dim, v);
                graph.LlamaRotatePosition2D(q_rope, positionIds, sin, cos, model->rotary_dim);
                graph.LlamaRotatePosition2D(k_rope, positionIds, sin, cos, model->rotary_dim);
                graph.Cat(q_nope, q_rope, -1, q);
                graph.Repeat(k_rope, 2, model->num_attention_heads, k_rope_expand);
                graph.Cat(k_nope, k_rope_expand, -1, k);
                graph.FusedAttention(q, pastKey, pastValue, k, v, attenInput, attentionMask, attenOutput, seqLens, 1.0 / sqrt(v_head_dim), 0, 128);
                graph.Linear(attenOutput, wNodes[pre + ".self_attn.o_proj.weight"], wNodes[pre + ".self_attn.o_proj.weight"], attenLastOutput);
                graph.AddTo(hiddenStates, attenLastOutput, attention_scale);
                graph.RMSNorm(hiddenStates, wNodes[pre + ".post_attention_layernorm.weight"], model->rms_norm_eps, attenInput);
                graph.Linear(attenInput, wNodes[pre + ".mlp.gate_proj.weight"], wNodes[pre + ".mlp.gate_proj.bias"], w1);
                graph.Linear(attenInput, wNodes[pre + ".mlp.up_proj.weight"], wNodes[pre + ".mlp.up_proj.bias"], w3);
                graph.Silu(w1, w1);
                graph.MulTo(w1, w3);
                graph.Linear(w1, wNodes[pre + ".mlp.down_proj.weight"], wNodes[pre + ".mlp.down_proj.bias"], w2);
                graph.AddTo(hiddenStates, w2, attention_scale);
            }

            graph.SplitLastTokenStates(hiddenStates, seqLens, lastTokensStates);
            graph.RMSNorm(lastTokensStates, wNodes["model.norm.weight"], model->rms_norm_eps, lastTokensStates);
            graph.Mul(lastTokensStates, rms_scale, lastTokensStates);
            graph.Linear(lastTokensStates, wNodes["lm_head.weight"], wNodes["lm_head.bias"], logits);
            OptimizeComputeGraph(graph, model->weight);
            graph.Update();
        }
    };
    REGISTERGRAPHMODELCONFIG(minicpm3, Minicpm3GraphModelConfig)
}