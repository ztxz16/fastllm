//
// Created by huangyuyang on 6/24/24.
//

#include "utils.h"

#include "graphllm.h"
#include <sstream>

#include <unordered_map>

#include <cstring>

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif

namespace fastllm {
    GraphLLMModel::GraphLLMModel(const std::string &type) {
        this->model_struct = "graph";
        this->model_type = type;
        if (type == "qwen") {
            this->graphLLMModelConfig = (GraphLLMModelConfig*)(new QwenGraphModelConfig());
        } else if (type == "telechat") {
            this->graphLLMModelConfig = (GraphLLMModelConfig*)(new TeleChatGraphModelConfig());
        } else {
            ErrorInFastLLM("Unsupport graph model type " + type);
        }
    }

    std::map <std::string, std::vector <std::pair <std::string, DataType> > >
                GraphLLMModel::GetTensorMap(const std::vector <std::string> &tensorNames) {
        return this->graphLLMModelConfig->GetTensorMap(this, tensorNames);
    }

    void GraphLLMModel::InitParams() {
        basellm::InitParams();
        num_key_value_heads = num_attention_heads;
        if (this->weight.dicts.find("num_key_value_heads") != this->weight.dicts.end()) {
            num_key_value_heads = atoi(this->weight.dicts["num_key_value_heads"].c_str());
        }
        head_dim = embed_dim / num_attention_heads;
        rotary_dim = head_dim;
        if (this->weight.dicts.find("max_position_embeddings") != this->weight.dicts.end()) {
            max_positions = atoi(this->weight.dicts["max_position_embeddings"].c_str());
        }
        if (this->weight.dicts.find("rms_norm_eps") != this->weight.dicts.end()) {
            rms_norm_eps = atof(this->weight.dicts["rms_norm_eps"].c_str());
        }
        if (this->weight.dicts.find("layer_norm_epsilon") != this->weight.dicts.end()) {
            rms_norm_eps = atof(this->weight.dicts["layer_norm_epsilon"].c_str());
        }
        if (this->weight.dicts.find("rope_scaling.type") != this->weight.dicts.end()) {
            std::string type = this->weight.dicts["rope_scaling.type"];
            if (type == "linear")
               rope_type = RoPEType::LINEAR_SCALE;
            else if (type == "dynamic")
               rope_type = RoPEType::DYMAMIC_NTK;
        }
        if (this->weight.dicts.find("rope_theta") != this->weight.dicts.end()) {
            rope_base = atof(this->weight.dicts["rope_theta"].c_str());
        }
        if (this->weight.dicts.find("rope_scaling.factor") != this->weight.dicts.end()) {
            rope_factor = atof(this->weight.dicts["rope_scaling.factor"].c_str());
        }
        this->graphLLMModelConfig->InitParams(this);
        std::pair<std::vector<float>, std::vector<float>> &&pair = this->UpdateRotaryPosEmb(rope_base, rope_factor);
        sinData.ToDevice(DataDevice::CPU);
        cosData.ToDevice(DataDevice::CPU);
        sinData.CopyFrom(Data(DataType::FLOAT32, { (int)this->sin.size(), (int)this->sin[0].size() }, pair.first));
        cosData.CopyFrom(Data(DataType::FLOAT32, { (int)this->cos.size(), (int)this->cos[0].size() }, pair.second));
    }

    std::pair<std::vector<float>, std::vector<float>> GraphLLMModel::UpdateRotaryPosEmb(float base, float factor, int seqLen) {
        int positions = std::max(max_positions, seqLen);
        sin.resize(positions);
        cos.resize(positions);
        std::vector <float> invFreq;
        for (int i = 0; i < rotary_dim; i += 2) {
            invFreq.push_back(1.0 / pow(base, (float)i / rotary_dim));
        }
        float scale = rope_type == RoPEType::LINEAR_SCALE ? factor : 1.0;
        for (int i = 0; i < positions; i++) {
            sin[i].resize(rotary_dim);
            cos[i].resize(rotary_dim);
            for (int j = 0; j < invFreq.size(); j++) {
                sin[i][j] = ::sin((float)i / scale * invFreq[j]);
                cos[i][j] = ::cos((float)i / scale * invFreq[j]);
            }
        }
        std::vector <float> fsin, fcos;
        for (int i = 0; i < sin.size(); i++) {
            fsin.insert(fsin.end(), sin[i].begin(), sin[i].end());
            fcos.insert(fcos.end(), cos[i].begin(), cos[i].end());
        }
        return std::make_pair(fsin, fcos);
    }

    int GraphLLMModel::Forward(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <float> *retLogits) {
        std::vector <std::vector <float>*> batchLogits;
        batchLogits.push_back(retLogits);
        return ForwardBatch(1, inputIds, attentionMask, positionIds, pastKeyValues, generationConfig, lastTokens, &batchLogits)[0];
    }

    std::vector <int> GraphLLMModel::ForwardBatch(int batch, const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <std::vector <float>*> *retLogits) {
        BuildGraph();
        std::map <std::string, Data*> weightDicts;
        for (auto &it : weight.weight) {
            weightDicts[it.first] = &it.second;
        }
        std::map <std::string, Data*> inputs = {
            {"inputIds", (Data*)&inputIds},
            {"positionIds", (Data*)&positionIds},
            {"attentionMask", (Data*)&attentionMask},
            {"sin", &sinData}, {"cos", &cosData}
        };
        for (int i = 0; i < block_cnt; i++) {
            inputs.insert({"pastKey_" + std::to_string(i), (Data*)&pastKeyValues[i].first});
            inputs.insert({"pastValue_" + std::to_string(i), (Data*)&pastKeyValues[i].second});
        }
        Data logits, topk;
        RunComputeGraph(graph, this->deviceMap, inputs, weightDicts, {{"logits", (Data*)&logits}});
        std::vector <int> lastRet;
        {
            ToDataType(logits, DataType::FLOAT32);
            if (generationConfig.output_logits && retLogits != nullptr) {
                int size = logits.dims.back();
                logits.ToDevice(DataDevice::CPU);
                for (int b = 0; b < batch; b++) {
                    (*retLogits)[b]->resize(size);
                    memcpy((float*)(*retLogits)[b]->data(), 
                        ((float*)logits.cpuData) + ((b + 1) * logits.dims[1] - 1) * size, 
                        size * logits.unitSize);
                }
            }

            if (generationConfig.IsSimpleGreedy()) {
                TopK(logits, topk, 1);
                topk.ToDevice(DataDevice::CPU);
                for (int b = 0; b < batch; b++) {
                    int base = b;
                    lastRet.push_back((int) (((float *) topk.cpuData)[base * 2] + 1e-3));
                }
            } else {
                for (int b = 0; b < batch; b++) {
                    int base = b * logits.dims[1] + logits.dims[1] - 1;
                    lastRet.push_back(LLMSampling(logits, base, generationConfig, lastTokens.units[b]));
                }
            }
        }
        return lastRet;
    }

    std::vector <int> GraphLLMModel::ForwardBatch(int batch,
                                               const Data &inputIds,
                                               const std::vector <Data*> &attentionMask,
                                               const std::vector <Data*> &positionIds,
                                               const std::vector <int> &seqLens,
                                               std::vector <std::pair <Data*, Data*> > &pastKeyValues,
                                               const std::vector <GenerationConfig> &generationConfigs,
                                               const LastTokensManager &lastTokens,
                                               std::vector <std::vector <float>*> *retLogits) {
        ErrorInFastLLM("Unsupport forward batch.\n");
        return {1};
    }

    bool GraphLLMModel::NeedAttentionMask(int qlen, int klen) {
        if (((qlen == 1) || (qlen >= 1024))) {
            return false;
        }
        return true;
    }

    std::string GraphLLMModel::MakeInput(const std::string &history, int round, const std::string &input) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role;
    }

    std::string GraphLLMModel::MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role + output + history_sep;
    }

    void GraphLLMModel::WarmUp() {
        printf("Warmup...\n");
        Data inputIds = Data(DataType::FLOAT32, {1, 1}, {1});
        Data attentionMask = Data(DataType::FLOAT32, {1, 1}, {0});
        Data positionIds = Data(DataType::FLOAT32, {1, 1}, {0, 0});

        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(DataType::FLOAT32),
                                                   Data(DataType::FLOAT32)));
        }
        Forward(inputIds, attentionMask, positionIds, pastKeyValues);
        printf("finish.\n");
    }

    ComputeGraph *GraphLLMModel::GetGraph() {
        return &this->graph;
    }

    void GraphLLMModel::BuildGraph() {
        if (inited) {
            return;
        }
        inited = true;
        this->graphLLMModelConfig->BuildGraph(this);
    }

    void GraphLLMModelConfig::InitParams(GraphLLMModel *model) {
    }

    std::map <std::string, std::vector <std::pair <std::string, DataType> > >
        QwenGraphModelConfig::GetTensorMap(GraphLLMModel *model, const std::vector <std::string> &tensorNames) {
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

    void QwenGraphModelConfig::BuildGraph(GraphLLMModel *model) {
        auto &graph = *(model->GetGraph());
        std::map <std::string, ComputeGraphNode> wNodes;
        for (auto &it : model->weight.weight) {
            wNodes[it.first] = ComputeGraphNode(it.first);
        }
        ComputeGraphNode inputIds("inputIds"), positionIds("positionIds"), attentionMask("attentionMask"), sin("sin"), cos("cos");
        ComputeGraphNode hiddenStates("hiddenStates"), attenInput("attenInput"), attenOutput("attenOutput"), attenLastOutput("attenLastOutput");
        ComputeGraphNode q("q"), k("k"), v("v"), w1("w1"), w2("w2"), w3("w3"), lastTokensStates("lastTokensStates"), logits("logits");
        graph.Embedding(inputIds, wNodes["model.embed_tokens.weight"], hiddenStates);
        for (int i = 0; i < model->block_cnt; i++) {
            std::string pre = "model.layers." + std::to_string(i);
            ComputeGraphNode pastKey("pastKey_" + std::to_string(i)), pastValue("pastValue_" + std::to_string(i));
            graph.RMSNorm(hiddenStates, wNodes[pre + ".input_layernorm.weight"], model->rms_norm_eps, attenInput);
            graph.Linear(attenInput, wNodes[pre + ".self_attn.q_proj.weight"], wNodes[pre + ".self_attn.q_proj.bias"], q);
            graph.Linear(attenInput, wNodes[pre + ".self_attn.k_proj.weight"], wNodes[pre + ".self_attn.k_proj.bias"], k);
            graph.Linear(attenInput, wNodes[pre + ".self_attn.v_proj.weight"], wNodes[pre + ".self_attn.v_proj.bias"], v);
            graph.ExpandHead(q, model->head_dim);
            graph.ExpandHead(k, model->head_dim);
            graph.ExpandHead(v, model->head_dim);
            graph.LlamaRotatePosition2D(q, positionIds, sin, cos, model->rotary_dim);
            graph.LlamaRotatePosition2D(k, positionIds, sin, cos, model->rotary_dim);
            graph.FusedAttention(q, pastKey, pastValue, k, v, attenInput, attentionMask, attenOutput, 1.0 / sqrt(model->head_dim), 1, 128);
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

        graph.SplitLastTokenStates(hiddenStates, lastTokensStates);
        graph.RMSNorm(lastTokensStates, wNodes["model.norm.weight"], model->rms_norm_eps, lastTokensStates);
        graph.Linear(lastTokensStates, wNodes["lm_head.weight"], wNodes["lm_head.bias"], logits);
        graph.Update();
    }

    void TeleChatGraphModelConfig::InitParams(GraphLLMModel *model) {
        model->block_cnt = atoi(model->weight.dicts["n_layer"].c_str());
        model->max_positions = atoi(model->weight.dicts["seq_length"].c_str());
        model->rope_base = 10000 * pow(3, ((float)model->rotary_dim / (model->rotary_dim - 2)));
        model->rope_factor = 1.0;
    }

    std::map <std::string, std::vector <std::pair <std::string, DataType> > >
        TeleChatGraphModelConfig::GetTensorMap(GraphLLMModel *model, const std::vector <std::string> &tensorNames) {        
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

    void TeleChatGraphModelConfig::BuildGraph(GraphLLMModel *model) {
        auto &graph = *(model->GetGraph());
        std::map <std::string, ComputeGraphNode> wNodes;
        for (auto &it : model->weight.weight) {
            wNodes[it.first] = ComputeGraphNode(it.first);
        }
        ComputeGraphNode inputIds("inputIds"), positionIds("positionIds"), attentionMask("attentionMask"), sin("sin"), cos("cos");
        ComputeGraphNode hiddenStates("hiddenStates"), attenInput("attenInput"), attenOutput("attenOutput"), attenLastOutput("attenLastOutput");
        ComputeGraphNode q("q"), kv("kv"), k("k"), v("v"), w1("w1"), w2("w2"), w3("w3"), lastTokensStates("lastTokensStates"), logits("logits");
        graph.Embedding(inputIds, wNodes["transformer.word_embeddings.weight"], hiddenStates);
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
            graph.FusedAttention(q, pastKey, pastValue, k, v, attenInput, attentionMask, attenOutput, 1.0 / sqrt(model->head_dim), 1, 128);
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

        graph.SplitLastTokenStates(hiddenStates, lastTokensStates);
        graph.RMSNorm(lastTokensStates, wNodes["transformer.ln_f.weight"], model->rms_norm_eps, lastTokensStates);
        graph.Linear(lastTokensStates, wNodes["transformer.lm_head.weight"], wNodes["transformer.lm_head.bias"], logits);
        graph.Update();
    }
}
