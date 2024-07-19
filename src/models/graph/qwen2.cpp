#include "graphllm.h"

namespace fastllm {
    class Qwen2GraphModelConfig : GraphLLMModelConfig {
    public:
        void InitParams(GraphLLMModel *model) {
        }

        std::map <std::string, std::vector <std::pair <std::string, DataType> > >
                GetTensorMap(GraphLLMModel *model, const std::vector <std::string> &tensorNames) {
            std::map <std::string, std::vector <std::pair <std::string, DataType> > > ret;
            std::string embeddingName = "model.embed_tokens.weight";
            std::string logitsName = "lm_head.weight";
            std::set <std::string> linearNames = {
                ".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight", ".self_attn.o_proj.weight",
                ".mlp.gate_proj.weight",  ".mlp.up_proj.weight", ".mlp.down_proj.weight"
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
            auto &graph = *(model->GetGraph());
            std::map <std::string, ComputeGraphNode> wNodes;
            for (auto &it : model->weight.weight) {
                wNodes[it.first] = ComputeGraphNode(it.first);
            }
            ComputeGraphNode inputIds("inputIds"), positionIds("positionIds"), attentionMask("attentionMask"), atype("atype"), sin("sin"), cos("cos"), seqLens("seqLens");
            ComputeGraphNode hiddenStates("hiddenStates"), attenInput("attenInput"), attenOutput("attenOutput"), attenLastOutput("attenLastOutput");
            ComputeGraphNode q("q"), k("k"), v("v"), w1("w1"), w2("w2"), w3("w3"), lastTokensStates("lastTokensStates"), logits("logits");
            graph.Embedding(inputIds, wNodes["model.embed_tokens.weight"], hiddenStates);
            graph.DataTypeAs(hiddenStates, atype);
            for (int i = 0; i < model->block_cnt; i++) {
                std::string pre = "model.layers." + std::to_string(i);
                ComputeGraphNode pastKey("pastKey." + std::to_string(i)), pastValue("pastValue." + std::to_string(i));
                graph.RMSNorm(hiddenStates, wNodes[pre + ".input_layernorm.weight"], model->rms_norm_eps, attenInput);
                graph.Linear(attenInput, wNodes[pre + ".self_attn.q_proj.weight"], wNodes[pre + ".self_attn.q_proj.bias"], q);
                graph.Linear(attenInput, wNodes[pre + ".self_attn.k_proj.weight"], wNodes[pre + ".self_attn.k_proj.bias"], k);
                graph.Linear(attenInput, wNodes[pre + ".self_attn.v_proj.weight"], wNodes[pre + ".self_attn.v_proj.bias"], v);
                graph.ExpandHead(q, model->head_dim);
                graph.ExpandHead(k, model->head_dim);
                graph.ExpandHead(v, model->head_dim);
                graph.LlamaRotatePosition2D(q, positionIds, sin, cos, model->rotary_dim);
                graph.LlamaRotatePosition2D(k, positionIds, sin, cos, model->rotary_dim);
                graph.FusedAttention(q, pastKey, pastValue, k, v, attenInput, attentionMask, attenOutput, seqLens, 1.0 / sqrt(model->head_dim), 0, 128);
                graph.Linear(attenOutput, wNodes[pre + ".self_attn.o_proj.weight"], wNodes[pre + ".self_attn.o_proj.bias"], attenLastOutput);
                graph.AddTo(hiddenStates, attenLastOutput);
                graph.RMSNorm(hiddenStates, wNodes[pre + ".post_attention_layernorm.weight"], model->rms_norm_eps, attenInput);
                graph.Linear(attenInput, wNodes[pre + ".mlp.gate_proj.weight"], wNodes[pre + ".mlp.gate_proj.bias"], w1);
                graph.Linear(attenInput, wNodes[pre + ".mlp.up_proj.weight"], wNodes[pre + ".mlp.up_proj.bias"], w3);
                graph.Silu(w1, w1);
                graph.MulTo(w1, w3);
                graph.Linear(w1, wNodes[pre + ".mlp.down_proj.weight"], wNodes[pre + ".mlp.down_proj.bias"], w2);
                graph.AddTo(hiddenStates, w2);
            }

            graph.SplitLastTokenStates(hiddenStates, seqLens, lastTokensStates);
            graph.RMSNorm(lastTokensStates, wNodes["model.norm.weight"], model->rms_norm_eps, lastTokensStates);
            graph.Linear(lastTokensStates, wNodes["lm_head.weight"], wNodes["lm_head.bias"], logits);
            
            OptimizeComputeGraph(graph, model->weight);
            graph.Update();
        }
    };
    REGISTERGRAPHMODELCONFIG(qwen2, Qwen2GraphModelConfig)
}