#include "graphllm.h"

namespace fastllm {
    class TeleChatGraphModelConfig : GraphLLMModelConfig {
    public:
        void InitParams(GraphLLMModel *model) {
            model->block_cnt = atoi(model->weight.dicts["n_layer"].c_str());
            model->max_positions = atoi(model->weight.dicts["seq_length"].c_str());
            model->rope_base = 10000 * pow(3, ((float)model->rotary_dim / (model->rotary_dim - 2)));
            model->rope_factor = 1.0;

            model->pre_prompt = "";
            model->user_role = "<_user>";
            model->bot_role = "<_bot>";
            model->history_sep = "";
        }

        std::map <std::string, std::vector <std::pair <std::string, DataType> > >
                GetTensorMap(GraphLLMModel *model, const std::vector <std::string> &tensorNames) {
            std::set <std::string> linearNames = {
                ".self_attention.query.weight", ".self_attention.key_value.weight", ".self_attention.dense.weight", 
                ".mlp.gate_proj.weight",  ".mlp.up_proj.weight", ".mlp.down_proj.weight"
            };
            std::string embeddingName = "transformer.word_embeddings.weight";
            std::string logitsName = "transformer.lm_head.weight";
            std::map <std::string, std::vector <std::pair <std::string, DataType> > > ret;
            ret[embeddingName].push_back(std::make_pair(embeddingName, DataType::DATA_AUTO_EMBEDDING));
            for (int i = 0; i < model->block_cnt; i++) {
                std::string pre = "transformer.h." + std::to_string(i);
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
            ComputeGraphNode q("q"), kv("kv"), k("k"), v("v"), w1("w1"), w2("w2"), w3("w3"), lastTokensStates("lastTokensStates"), logits("logits");
            graph.Embedding(inputIds, wNodes["transformer.word_embeddings.weight"], hiddenStates);
            graph.DataTypeAs(hiddenStates, atype);
            for (int i = 0; i < model->block_cnt; i++) {
                std::string pre = "transformer.h." + std::to_string(i);
                ComputeGraphNode pastKey("pastKey_" + std::to_string(i)), pastValue("pastValue_" + std::to_string(i));
                graph.RMSNorm(hiddenStates, wNodes[pre + ".input_layernorm.weight"], model->rms_norm_eps, attenInput);
                graph.Linear(attenInput, wNodes[pre + ".self_attention.query.weight"], wNodes[pre + ".self_attention.query.bias"], q);
                graph.Linear(attenInput, wNodes[pre + ".self_attention.key_value.weight"], wNodes[pre + ".self_attention.key_value.bias"], kv);
                graph.ExpandHead(kv, model->head_dim * 2);
                graph.Split(kv, -1, 0, model->head_dim, k);
                graph.Split(kv, -1, model->head_dim, model->head_dim * 2, v);
                graph.ExpandHead(q, model->head_dim);                
                graph.LlamaRotatePosition2D(q, positionIds, sin, cos, model->rotary_dim);
                graph.LlamaRotatePosition2D(k, positionIds, sin, cos, model->rotary_dim);
                graph.FusedAttention(q, pastKey, pastValue, k, v, attenInput, attentionMask, attenOutput, seqLens, 1.0 / sqrt(model->head_dim), 0, 128);
                graph.Linear(attenOutput, wNodes[pre + ".self_attention.dense.weight"], wNodes[pre + ".self_attention.dense.bias"], attenLastOutput);
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
            graph.RMSNorm(lastTokensStates, wNodes["transformer.ln_f.weight"], model->rms_norm_eps, lastTokensStates);
            graph.Linear(lastTokensStates, wNodes["transformer.lm_head.weight"], wNodes["transformer.lm_head.bias"], logits);

            OptimizeComputeGraph(graph, model->weight);
            graph.Update();
        }
    };
    REGISTERGRAPHMODELCONFIG(telechat, TeleChatGraphModelConfig)
}
