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
        this->graphLLMModelConfig = GraphLLMModelConfigFactory::CreateGraphLLMModelConfig(type);
        if (this->graphLLMModelConfig == nullptr) {
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
        Data seqLensData = Data(DataType::INT32PARAM, {batch});
        seqLensData.Allocate();
        for (int i = 0; i < seqLensData.Count(0); i++) {
            ((int32_t*)seqLensData.cpuData)[i] = inputIds.dims[1];
        }
        std::map <std::string, Data*> weightDicts;
        for (auto &it : weight.weight) {
            weightDicts[it.first] = &it.second;
        }
        std::vector <std::vector <Data*> > pastKeys, pastValues;
        std::vector <Data*> masks;
        pastKeys.resize(block_cnt);
        pastValues.resize(block_cnt);
        masks.push_back((Data*)&attentionMask);

        Data atype = Data(this->dataType);
        std::map <std::string, Data*> inputs = {
            {"inputIds", (Data*)&inputIds},
            {"positionIds", (Data*)&positionIds},
            {"attentionMask", (Data*)&attentionMask},
            {"atype", (Data*)&atype},
            {"sin", &sinData}, {"cos", &cosData},
            {"seqLens", (Data*)&seqLensData}
        };
        for (int i = 0; i < block_cnt; i++) {
            pastKeys[i].push_back((Data*)&pastKeyValues[i].first);
            pastValues[i].push_back((Data*)&pastKeyValues[i].second);
        }
        Data logits, topk;
        RunComputeGraph(graph, this->deviceMap, inputs, weightDicts, {{"logits", (Data*)&logits}}, pastKeys, pastValues, masks);
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
        BuildGraph();
        Data seqLensData = Data(DataType::INT32PARAM, {(int)seqLens.size()});
        seqLensData.Allocate();
        for (int i = 0; i < seqLensData.Count(0); i++) {
            ((int32_t*)seqLensData.cpuData)[i] = seqLens[i];
        }
        int seqLen = inputIds.dims[1];
        Data allPositionIds;
        int pos = 0;
        allPositionIds.dataType = positionIds[0]->dataType;
        allPositionIds.Resize({1, seqLen});
        allPositionIds.Allocate();
        for (int i = 0; i < batch; i++) {
            memcpy(allPositionIds.cpuData + pos, positionIds[i]->cpuData, (size_t)positionIds[i]->GetBytes());
            pos += positionIds[i]->GetBytes();
        }
        std::map <std::string, Data*> weightDicts;
        for (auto &it : weight.weight) {
            weightDicts[it.first] = &it.second;
        }
        Data atype = Data(this->dataType);
        std::map <std::string, Data*> inputs = {
            {"inputIds", (Data*)&inputIds},
            {"positionIds", (Data*)&allPositionIds},
            {"atype", (Data*)&atype},
            {"sin", &sinData}, {"cos", &cosData},
            {"seqLens", &seqLensData}
        };

        std::vector <std::vector <Data*> > pastKeys, pastValues;
        std::vector <Data*> masks;
        pastKeys.resize(block_cnt);
        pastValues.resize(block_cnt);
        for (int i = 0; i < block_cnt; i++) {
            pastKeys[i].resize(batch);
            pastValues[i].resize(batch);
        }
        masks.resize(batch);
        for (int b = 0; b < batch; b++) {
            masks[b] = attentionMask[b];
            for (int i = 0; i < block_cnt; i++) {
                pastKeys[i][b] = pastKeyValues[b * block_cnt + i].first;
                pastValues[i][b] = pastKeyValues[b * block_cnt + i].second;
            }
            for (int i = 0; i < block_cnt; i++) {
                if (GetKVCacheInCPU()) {
                    pastKeyValues[b * block_cnt + i].first->lockInCPU = true;
                    pastKeyValues[b * block_cnt + i].second->lockInCPU = true;
                } else {
                    if (pastKeyValues[b * block_cnt + i].first->dataDevice == DataDevice::CUDA) {
                        break;
                    }
                    pastKeyValues[b * block_cnt + i].first->ToDevice(DataDevice::CUDA);
                    pastKeyValues[b * block_cnt + i].second->ToDevice(DataDevice::CUDA);
                }
            }
        }

        // 拼batch, 把短句补长
        Data curAttentionMask;
        Data realInputIds;
        int maxLen = 0, totalLen = 0;
        if (batch > 1 && seqLen != seqLens.size()) {        
            for (int i = 0; i < batch; i++) {
                maxLen = std::max(maxLen, seqLens[i]);
            }
            int totalLen = maxLen * batch;
            Data tempInputIds;
            Mul(inputIds, 1.0, tempInputIds);
            ToDataType(tempInputIds, DataType::FLOAT32);
            tempInputIds.ToDevice(DataDevice::CPU);
            allPositionIds.ToDevice(DataDevice::CPU);
            float *floatInputIds = (float*)tempInputIds.cpuData;
            float *floatPositionIds = (float*)allPositionIds.cpuData;
            
            std::vector <float> vmask = std::vector <float> (batch * maxLen * maxLen, 0);
            std::vector <float> vpids = std::vector <float> (batch * maxLen, 0);
            std::vector <float> ids = std::vector <float> (batch * maxLen, 0.0);
            for (int i = 0; i < batch; i++) {
                int len = seqLens[i], base = maxLen - len;
                for (int j = 0; j < len; j++) {
                    ids[i * maxLen + base + j] = (*floatInputIds++);
                    vpids[i * maxLen + base + j] = (*floatPositionIds++);
                }
                std::fill(vmask.data() + i * maxLen * maxLen,
                    vmask.data() + i * maxLen * maxLen + (maxLen - len) * maxLen, 1.0);
                for (int j = maxLen - len; j < maxLen; j++) {
                    std::fill(vmask.data() + i * maxLen * maxLen + j * maxLen,
                                vmask.data() + i * maxLen * maxLen + j * maxLen + maxLen - len, 1.0);
                }
                for (int j = 0; j < len; j++) {
                    for (int k = j + 1; k < len; k++) {
                        vmask[i * maxLen * maxLen + (base + j) * maxLen + base + k] = 1;
                    }
                }
            }

            realInputIds.CopyFrom(Data(DataType::FLOAT32, {1, batch * maxLen}, ids));
            curAttentionMask.CopyFrom(Data(DataType::FLOAT32, {batch, maxLen, maxLen}, vmask));
            allPositionIds.CopyFrom(Data(DataType::FLOAT32, {1, batch * maxLen}, vpids));

            ToDataType(curAttentionMask, this->dataType);
            inputs.insert({"attentionMask", &curAttentionMask});
            inputs["inputIds"] = (Data*)&realInputIds;
            inputs["positionIds"] = (Data*)&allPositionIds;
        }

        Data logits, topk;
        RunComputeGraph(graph, this->deviceMap, inputs, weightDicts, {{"logits", (Data*)&logits}}, pastKeys, pastValues, masks);
        ToDataType(logits, DataType::FLOAT32);
        std::vector <Data> curLogits;
        curLogits.resize(batch);

        std::vector <int> lastRet;
        int total = 0;

        bool all1 = true;
        bool allSimple = true, needLogits = false;
        int maxTopK = 1;
        for (int b = 0; b < batch; b++) {
            if (!generationConfigs[b].IsSimpleGreedy()) {
                allSimple = false;
                break;
            }
        }
        for (int b = 0; b < batch; b++) {
            all1 &= (seqLens[b] == 1);
            needLogits |= generationConfigs[b].output_logits;
            maxTopK = std::max(maxTopK, generationConfigs[b].top_k);
        }

        if (all1 && batch > 1 && allSimple) {
            Data topk;
            TopK(logits, topk, 1);
            topk.ToDevice(DataDevice::CPU);
            float *topkData = (float*)topk.cpuData;
            for (int b = 0; b < batch; b++) {
                lastRet.push_back((int) (topkData[0] + 1e-3));
                topkData += topk.Count(2);
            }
        } else if (all1 && batch > 1 && maxTopK <= 50 && !needLogits) {
            int maxTokenSetSize = 0;
            for (int b = 0; b < batch; b++) {
                maxTokenSetSize = std::max(maxTokenSetSize, (int)lastTokens.units[b].tokenSet.size());
            }
            std::vector <float> penaltyData = std::vector <float> (batch * maxTokenSetSize, -100.0f);
            std::vector <float> penaltyScaleData = std::vector <float> (batch, 1.0f);
            for (int b = 0; b < batch; b++) {
                int curId = 0;
                for (int i : lastTokens.units[b].tokenSet) {
                    penaltyData[b * maxTokenSetSize + curId] = i;
                    curId++;
                }
                penaltyScaleData[b] = generationConfigs[b].repeat_penalty;
            }
            Data penalty, penaltyScale;
            penalty.CopyFrom(Data(DataType::FLOAT32, {batch, maxTokenSetSize}, penaltyData));
            penaltyScale.CopyFrom(Data(DataType::FLOAT32, {batch}, penaltyScaleData));
            RepeatPenalty(logits, penalty, penaltyScale);
            Data topk;
            TopK(logits, topk, maxTopK);
            topk.ToDevice(DataDevice::CPU);
            for (int b = 0; b < batch; b++) {
                lastRet.push_back(LLMSamplingOnly(topk, b, generationConfigs[b]));
            }
        } else {
            /*if (all1 && batch > 1) {
                for (int b = 0; b < batch; b++) {
                    pointersK[b] = (&curLogits[b]);
                }
                SplitBatch(logits, 1, batch, pointersK);
            } else */{
                for (int b = 0; b < batch; b++) {
                    Split(logits, 1, b, b + 1, curLogits[b]);
                }
            }

            for (int b = 0; b < batch; b++) {
                Data &curLogit = curLogits[b];
                if (generationConfigs[b].output_logits && retLogits != nullptr && (*retLogits)[b] != nullptr) {
                    curLogit.ToDevice(DataDevice::CPU);
                    (*retLogits)[b]->resize(curLogit.Count(0));
                    memcpy((float*)(*retLogits)[b]->data(), (float*)curLogit.cpuData, curLogit.GetBytes());
                }
                if (generationConfigs[b].IsSimpleGreedy()) {
                    Data topk;
                    TopK(curLogit, topk, 1);
                    topk.ToDevice(DataDevice::CPU);
                    lastRet.push_back((int) (((float *) topk.cpuData)[0] + 1e-3));
                } else {
                    lastRet.push_back(LLMSampling(curLogit, 0, generationConfigs[b], lastTokens.units[b]));
                }
            }
        }
        return lastRet;
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
        elementsInKVCachePerToken = (long long)block_cnt * 
            (pastKeyValues[0].first.dims[0] * pastKeyValues[0].first.dims[2] + 
             pastKeyValues[0].second.dims[0] * pastKeyValues[0].second.dims[2]);
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

    void GraphLLMModelConfig::Init(const std::string &config) {
    }

    void GraphLLMModelConfig::InitParams(GraphLLMModel *model) {
    }

    static std::map<std::string, GraphLLMModelConfigCreator> *graphLLMModelConfigFactoryCreator = nullptr;

    void GraphLLMModelConfigFactory::RegisterGraphLLMModelConfig(const std::string& type, GraphLLMModelConfigCreator creator) {
        if (graphLLMModelConfigFactoryCreator == nullptr)
            graphLLMModelConfigFactoryCreator = new std::map<std::string, GraphLLMModelConfigCreator>();
        (*graphLLMModelConfigFactoryCreator)[type] = creator;
    }

    GraphLLMModelConfig* GraphLLMModelConfigFactory::CreateGraphLLMModelConfig(const std::string& type) {
        if (graphLLMModelConfigFactoryCreator == nullptr)
            return nullptr;
        auto it = graphLLMModelConfigFactoryCreator->find(type);
        if (it != graphLLMModelConfigFactoryCreator->end()) {
            return it->second();
        } else {
            return nullptr;
        }
    }
}
