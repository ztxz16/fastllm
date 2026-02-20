//
// Created by huangyuyang on 2/19/26.
//

#include "utils.h"

#include "qwen2.h"

#include <sstream>

#include <unordered_map>

#include <cstring>

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif

namespace fastllm {
    int Qwen2Model::Forward(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <float> *retLogits) {
        Data attentionMaskCopy(attentionMask), positionIdsCopy(positionIds);
        std::vector <Data*> attentionMasks = {&attentionMaskCopy};
        std::vector <Data*> positionIdsVec = {&positionIdsCopy};
        std::vector <int> seqLens = {(int)inputIds.dims[1]};
        std::vector <GenerationConfig> generationConfigs = {generationConfig};
        std::vector <std::pair <Data*, Data*> > pagedPastKeyValues;
        for (int i = 0; i < pastKeyValues.size(); i++) {
            pagedPastKeyValues.push_back(std::make_pair(&pastKeyValues[i].first, &pastKeyValues[i].second));
        }
        std::vector <std::vector <float>*> batchLogits;
        batchLogits.push_back(retLogits);
        return ForwardV2(1, inputIds, attentionMasks, positionIdsVec, seqLens,
                         pagedPastKeyValues, generationConfigs, lastTokens, &batchLogits)[0];
    }

    Qwen2Model::Qwen2Model() {
        this->model_struct = "qwen2";
        this->model_type = "qwen2";
        this->use_new_engine = true;

        weight.embeddingNames.insert("model.embed_tokens.weight");
        weight.linearNames = {
            "lm_head.weight", "model.layers.*.mlp.down_proj.weight", "model.layers.*.mlp.up_proj.weight",
            "model.layers.*.mlp.gate_proj.weight",  "model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.gateup_proj.weight",
            "model.layers.*.self_attn.o_proj.weight", "model.layers.*.self_attn.q_proj.weight", "model.layers.*.self_attn.k_proj.weight",
            "model.layers.*.self_attn.v_proj.weight", "model.layers.*.self_attn.mergeqkv.weight", "model.layers.*.self_attn.W_pack.weight"
        };
    }

    void Qwen2Model::InitParams() {
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
        for (int i = 0; i < block_cnt; i++) {
            std::string w1WeightName = "model.layers." + std::to_string(i) + ".mlp.gate_proj.weight";
            std::string w3WeightName = "model.layers." + std::to_string(i) + ".mlp.up_proj.weight";
            std::string swigluWeightName = "model.layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
            this->weightMergeRules.push_back(
                WeightMergeRule({WeightMergeRuleSingle({w1WeightName, w3WeightName}, swigluWeightName, std::string("linearSwiglu"))})
            );

            std::string qWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.weight";
            std::string qBiasName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.bias";
            std::string kWeightName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.weight";
            std::string kBiasName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.bias";
            std::string vWeightName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.weight";
            std::string vBiasName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.bias";
            std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
            std::string mergeQkvBiasName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";
            this->weightMergeRules.push_back(
                WeightMergeRule({WeightMergeRuleSingle({qWeightName, kWeightName, vWeightName}, mergeQkvWeightName, std::string("linear")),
                                 WeightMergeRuleSingle({qBiasName, kBiasName, vBiasName}, mergeQkvBiasName, std::string("bias"))})
            );
        }
    }

    std::vector <int> Qwen2Model::ForwardV2(
        int batch,
        const Data &inputIds,
        const std::vector <Data*> &attentionMask,
        const std::vector <Data*> &positionIds,
        const std::vector <int> &seqLens,
        std::vector <std::pair <Data*, Data*> > &pastKeyValues,
        const std::vector <GenerationConfig> &generationConfigs,
        const LastTokensManager &lastTokens,
        std::vector <std::vector <float>*> *retLogits) {
        int seqLen = inputIds.dims[1];

        Data qkv;
        // Data &qkv = this->forwardDataManager.GetData("qkv");
        Data q;
        // Data &q = this->forwardDataManager.GetData("q");
        Data k;
        // Data &k = this->forwardDataManager.GetData("k");
        Data v;
        // Data &v = this->forwardDataManager.GetData("v");
        Data embeddingResult;
        // Data &embeddingResult = this->forwardDataManager.GetData("embeddingResult");
        Data hiddenStates;
        // Data &hiddenStates = this->forwardDataManager.GetData("hiddenStates");
        Data attenInput;
        // Data &attenInput = this->forwardDataManager.GetData("attenInput");
        Data attenLastOutput;
        // Data &attenLastOutput = this->forwardDataManager.GetData("attenLastOutput");
        std::vector <Data*> pointersK;
        pointersK.resize(batch);


        std::vector<Data*> batchPastKeys;
        std::vector<Data*> batchPastValues;
        batchPastKeys.resize(batch);
        batchPastValues.resize(batch);

        Data allPositionIds;
        // Data &allPositionIds = this->forwardDataManager.GetData("allPositionIds");
        Data qSizes;
        // Data &qSizes = this->forwardDataManager.GetData("qSizes");
        Data pageSizes;
        // Data &pageSizes = this->forwardDataManager.GetData("pageSizes");
        Data pageIndexs;
        // Data &pageIndexs = this->forwardDataManager.GetData("pageIndexs");
        Data lastPageLens;
        // Data &lastPageLens = this->forwardDataManager.GetData("lastPageLens");
        Data insertIndexs;
        // Data &insertIndexs = this->forwardDataManager.GetData("insertIndexs");
        Data insertPositions;
        // Data &insertPositions = this->forwardDataManager.GetData("insertPositions");
        Data attenOutput;
        // Data &attenOutput = this->forwardDataManager.GetData("attenOutput");
        bool generatedBatchDecodeParams = false;
        bool generatedAppendPagedCacheBatchParams = false;

        bool all1 = true;
        for (int i = 0; i < batch; i++) {
            all1 &= (seqLens[i] == 1);
        }
        bool isPrefill = !all1;
        if (all1 && positionIds[0]->dataType == DataType::FLOAT32) {
            std::vector <float> vPositionIds;            
            for (int b = 0; b < batch; b++) {
                vPositionIds.push_back(((float*)positionIds[b]->cpuData)[0]);
            }
            allPositionIds.CopyFrom(Data(DataType::FLOAT32, {1, seqLen}, vPositionIds));
        } else {
            std::vector <float> vPositionIds;            
            for (int b = 0; b < batch; b++) {
                for (int i = 0; i < seqLens[b]; i++) {
                    vPositionIds.push_back(((float*)positionIds[b]->cpuData)[i]);
                }
            }
            allPositionIds.CopyFrom(Data(DataType::FLOAT32, {1, (int)vPositionIds.size()}, vPositionIds));
        }

        Embedding(inputIds, this->weight["model.embed_tokens.weight"], embeddingResult);
        ToDataType(embeddingResult, hiddenStates, this->dataType);
        int seqlen = hiddenStates.dims[1];
        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            std::string inputRmsName = "model.layers." + std::to_string(i) + ".input_layernorm.weight";
            std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
            std::string mergeQkvBiasName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";
            std::string qNormName = "model.layers." + std::to_string(i) + ".self_attn.q_norm.weight";
            std::string kNormName = "model.layers." + std::to_string(i) + ".self_attn.k_norm.weight";
            std::string oWeightName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight";
            std::string oBiasName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.bias";
            std::string postRmsName = "model.layers." + std::to_string(i) + ".post_attention_layernorm.weight";
            std::string swigluWeightName = "model.layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
            std::string downWeightName = "model.layers." + std::to_string(i) + ".mlp.down_proj.weight";

            RMSNorm(hiddenStates, this->weight[inputRmsName], rms_norm_eps, attenInput);
            AttentionPagedBlock(
                &attenInput,
                &weight[mergeQkvWeightName], &weight[mergeQkvBiasName],
                GetEmptyData(), GetEmptyData(),
                &weight[oWeightName], &weight[oBiasName],
                &allPositionIds,
                &pastKeyValues, &batchPastKeys, &batchPastValues,
                &qkv, &q, &attenOutput, &attenLastOutput,
                &insertIndexs, &insertPositions,
                &qSizes, &pageSizes, &pageIndexs, &lastPageLens,
                &generatedAppendPagedCacheBatchParams, &generatedBatchDecodeParams,
                batch, block_cnt, i,
                seqLens,
                num_attention_heads, num_key_value_heads, head_dim,
                rotary_dim, rms_norm_eps,
                rope_base, rope_factor, max_positions,
                rope_type,
                GetKVCacheInCPU(),
                isPrefill,
                &hiddenStates,
                false
            );

            RMSNorm(hiddenStates, this->weight[postRmsName], rms_norm_eps, attenInput);
            MLPBlock(&attenInput, &weight[swigluWeightName], &weight[downWeightName], &v, &q, &hiddenStates);
        }

        Data logits;
        std::vector <Data> curLogits;
        curLogits.resize(batch);

        if (!all1) {
            int total = 0;
            std::vector <Data> lastTokens;
            std::vector <Data*> lastTokenPointers;
            lastTokens.resize(seqLens.size());
            for (int b = 0; b < seqLens.size(); b++) {
                Split(hiddenStates, 1, total + seqLens[b] - 1, total + seqLens[b], lastTokens[b]);
                total += seqLens[b];
                lastTokenPointers.push_back(&lastTokens[b]);
            }
            CatBatch(lastTokenPointers, 1, hiddenStates);
        }

        RMSNorm(hiddenStates, weight["model.norm.weight"], rms_norm_eps, hiddenStates);
        Linear(hiddenStates, weight["lm_head.weight"], *GetEmptyData(), logits);
        ToDataType(logits, DataType::FLOAT32);
        std::vector <int> lastRet;
        int total = 0;

        for (int b = 0; b < batch; b++) {
            if (generationConfigs[b].top_k <= 1) {
                // 禁用simple greedy
                ((GenerationConfig*)&generationConfigs[b])->top_k = 5;
                ((GenerationConfig*)&generationConfigs[b])->top_p = 0.95;
                if (fabs(generationConfigs[b].temperature - 1.0f) < 1e-9) {
                    ((GenerationConfig*)&generationConfigs[b])->temperature = 0.6;
                }
            }
        }

        bool allSimple = true, needLogits = false;
        int maxTopK = 1;
        for (int b = 0; b < batch; b++) {
            if (!generationConfigs[b].IsSimpleGreedy()) {
                allSimple = false;
                break;
            }
        }
        for (int b = 0; b < batch; b++) {
            needLogits |= generationConfigs[b].output_logits;
            maxTopK = std::max(maxTopK, generationConfigs[b].top_k);
        }
        
        ResetLogitsOfEOS(batch, &logits, pastKeyValues, generationConfigs);
        // if (all1) {
        if (true) {
            Data topk;
            TopK(logits, topk, 1);
            topk.ToDevice(DataDevice::CPU);
            float *topkData = (float*)topk.cpuData;
            for (int b = 0; b < batch; b++) {
                lastRet.push_back((int) (topkData[0] + 1e-3));
                topkData += topk.Count(2);
            }
        } else if (batch > 1 && maxTopK <= 50 && !needLogits) {
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
            for (int b = 0; b < batch; b++) {
                pointersK[b] = (&curLogits[b]);
            }
            SplitBatch(logits, 1, batch, pointersK);

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

    bool Qwen2Model::NeedAttentionMask(int qlen, int klen) {
        return false;
    }

    std::string Qwen2Model::MakeInput(const std::string &history, int round, const std::string &input) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role;
    }

    std::string Qwen2Model::MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role + output + history_sep;
    }

    void Qwen2Model::WarmUp() {
        printf("Warmup...\n");
        Data inputIds = Data(DataType::FLOAT32, {1, 1}, {1});
        Data attentionMask = Data(this->dataType, {1, 1}, {0});
        Data positionIds = Data(this->dataType, {1, 1}, {0, 0});

        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(this->dataType),
                                                   Data(this->dataType)));
        }
        if (this->weight.weight.find("lm_head.weight") == this->weight.weight.end()) {
            this->weight["lm_head.weight"] = Data();
            this->weight["lm_head.weight"].CopyFrom(this->weight["model.embed_tokens.weight"]);
            ToDataType(this->weight["lm_head.weight"], this->dataType);
        }
        Forward(inputIds, attentionMask, positionIds, pastKeyValues);
        elementsInKVCachePerToken = (long long)block_cnt * 
            (pastKeyValues[0].first.dims[0] * pastKeyValues[0].first.dims[2] + 
             pastKeyValues[0].second.dims[0] * pastKeyValues[0].second.dims[2]);
        printf("finish.\n");
    }
}
