#include "utils.h"
#include "gemma4.h"

#include <sstream>
#include <unordered_map>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "json11.hpp"

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif

namespace fastllm {
    Gemma4Model::Gemma4Model() {
        this->model_struct = "gemma4";
        this->model_type = "gemma4";
        this->use_new_engine = false;

        this->pre_prompt = "";
        this->user_role = "";
        this->bot_role = "";
        this->history_sep = "";

        block_cnt = 60;
        rotary_dim = 256;

        weight.embeddingNames.insert("model.language_model.embed_tokens.weight");
        weight.linearNames = {
            "lm_head.weight",
            "model.language_model.layers.*.mlp.down_proj.weight",
            "model.language_model.layers.*.mlp.up_proj.weight",
            "model.language_model.layers.*.mlp.gate_proj.weight",
            "model.language_model.layers.*.self_attn.o_proj.weight",
            "model.language_model.layers.*.self_attn.q_proj.weight",
            "model.language_model.layers.*.self_attn.k_proj.weight",
            "model.language_model.layers.*.self_attn.v_proj.weight"
        };
    }

    static std::string GetDictValue(const std::map<std::string, std::string> &dicts,
                                    const std::string &key1,
                                    const std::string &key2,
                                    const std::string &defaultVal = "") {
        auto it = dicts.find(key1);
        if (it != dicts.end()) return it->second;
        it = dicts.find(key2);
        if (it != dicts.end()) return it->second;
        return defaultVal;
    }

    void Gemma4Model::InitParams() {
        basellm::InitParams();
        auto &d = this->weight.dicts;

        std::string val;
        val = GetDictValue(d, "text_config.num_key_value_heads", "num_key_value_heads", "16");
        num_key_value_heads = atoi(val.c_str());

        val = GetDictValue(d, "text_config.num_global_key_value_heads", "num_global_key_value_heads", "4");
        global_num_key_value_heads = atoi(val.c_str());

        val = GetDictValue(d, "text_config.head_dim", "head_dim", "256");
        sliding_head_dim = atoi(val.c_str());

        val = GetDictValue(d, "text_config.global_head_dim", "global_head_dim", "512");
        global_head_dim = atoi(val.c_str());

        val = GetDictValue(d, "text_config.hidden_size", "hidden_size", "5376");
        embed_dim = atoi(val.c_str());

        val = GetDictValue(d, "text_config.num_attention_heads", "num_attention_heads", "32");
        num_attention_heads = atoi(val.c_str());

        val = GetDictValue(d, "text_config.num_hidden_layers", "num_hidden_layers", "60");
        block_cnt = atoi(val.c_str());

        val = GetDictValue(d, "text_config.sliding_window", "sliding_window", "1024");
        sliding_window = atoi(val.c_str());

        val = GetDictValue(d, "text_config.rms_norm_eps", "rms_norm_eps", "1e-6");
        rms_norm_eps = atof(val.c_str());

        val = GetDictValue(d, "text_config.attention_k_eq_v", "attention_k_eq_v", "true");
        attention_k_eq_v = (val == "true" || val == "True" || val == "1");

        val = GetDictValue(d, "text_config.final_logit_softcapping", "final_logit_softcapping", "30.0");
        if (val != "None" && val != "null" && !val.empty())
            final_logit_softcapping = atof(val.c_str());

        val = GetDictValue(d, "text_config.max_position_embeddings", "max_position_embeddings", "262144");
        max_positions = atoi(val.c_str());

        val = GetDictValue(d, "text_config.eos_token_id", "eos_token_id", "");
        if (!val.empty() && val != "None") {
            this->eos_token_id = atoi(val.c_str());
        }
        val = GetDictValue(d, "text_config.bos_token_id", "bos_token_id", "");
        if (!val.empty() && val != "None") {
            this->bos_token_id = atoi(val.c_str());
        }

        layer_types.clear();
        val = GetDictValue(d, "text_config.layer_types", "layer_types", "");
        if (!val.empty() && val[0] == '[') {
            std::string error;
            auto arr = json11::Json::parse(val, error);
            if (error.empty() && arr.is_array()) {
                for (auto &item : arr.array_items()) {
                    if (item.string_value() == "full_attention") {
                        layer_types.push_back(1);
                    } else {
                        layer_types.push_back(0);
                    }
                }
            }
        }
        if ((int)layer_types.size() < block_cnt) {
            layer_types.clear();
            for (int i = 0; i < block_cnt; i++) {
                layer_types.push_back(((i + 1) % 6 == 0) ? 1 : 0);
            }
            layer_types[block_cnt - 1] = 1;
        }

        val = GetDictValue(d, "text_config.rope_parameters.sliding_attention.rope_theta",
                           "rope_parameters.sliding_attention.rope_theta", "10000.0");
        sliding_rope_base = atof(val.c_str());

        val = GetDictValue(d, "text_config.rope_parameters.full_attention.rope_theta",
                           "rope_parameters.full_attention.rope_theta", "1000000.0");
        global_rope_base = atof(val.c_str());

        val = GetDictValue(d, "text_config.rope_parameters.full_attention.partial_rotary_factor",
                           "rope_parameters.full_attention.partial_rotary_factor", "0.25");
        global_partial_rotary_factor = atof(val.c_str());

        head_dim = sliding_head_dim;
        rotary_dim = sliding_head_dim;

        {
            std::vector<float> onesGlobal(global_head_dim, 1.0f);
            weight["__gemma4_v_norm_global.weight"] = Data(DataType::FLOAT32, {global_head_dim}, onesGlobal);
            std::vector<float> onesSliding(sliding_head_dim, 1.0f);
            weight["__gemma4_v_norm_sliding.weight"] = Data(DataType::FLOAT32, {sliding_head_dim}, onesSliding);
        }

        int maxPos = std::max(max_positions, 16384);
        {
            auto pair = UpdateRotaryPosEmb(sliding_rope_base, 1.0f, maxPos, sliding_head_dim);
            slidingSinData.ToDevice(DataDevice::CPU);
            slidingCosData.ToDevice(DataDevice::CPU);
            slidingSinData.CopyFrom(Data(DataType::FLOAT32,
                {(int)slidingSin.size(), (int)slidingSin[0].size()}, pair.first));
            slidingCosData.CopyFrom(Data(DataType::FLOAT32,
                {(int)slidingCos.size(), (int)slidingCos[0].size()}, pair.second));
        }

        {
            int ropeAngles = (int)(global_partial_rotary_factor * global_head_dim / 2);
            int totalHalf = global_head_dim / 2;
            int nopeAngles = totalHalf - ropeAngles;

            std::vector<float> invFreq;
            for (int i = 0; i < ropeAngles; i++) {
                invFreq.push_back(1.0f / pow(global_rope_base, (float)(2 * i) / (float)global_head_dim));
            }
            for (int i = 0; i < nopeAngles; i++) {
                invFreq.push_back(0.0f);
            }

            int positions = std::max(max_positions, 16384);
            globalSin.resize(positions);
            globalCos.resize(positions);
            for (int p = 0; p < positions; p++) {
                globalSin[p].resize(totalHalf);
                globalCos[p].resize(totalHalf);
                for (int j = 0; j < totalHalf; j++) {
                    float angle = (float)p * invFreq[j];
                    globalSin[p][j] = ::sin(angle);
                    globalCos[p][j] = ::cos(angle);
                }
            }

            std::vector<float> fsin, fcos;
            for (int i = 0; i < (int)globalSin.size(); i++) {
                fsin.insert(fsin.end(), globalSin[i].begin(), globalSin[i].end());
                fcos.insert(fcos.end(), globalCos[i].begin(), globalCos[i].end());
            }
            globalSinData.ToDevice(DataDevice::CPU);
            globalCosData.ToDevice(DataDevice::CPU);
            globalSinData.CopyFrom(Data(DataType::FLOAT32,
                {(int)globalSin.size(), (int)globalSin[0].size()}, fsin));
            globalCosData.CopyFrom(Data(DataType::FLOAT32,
                {(int)globalCos.size(), (int)globalCos[0].size()}, fcos));
        }
    }

    std::pair<std::vector<float>, std::vector<float>> Gemma4Model::UpdateRotaryPosEmb(float base, float factor, int seqLen, int dim) {
        int positions = std::max(max_positions, seqLen);
        int halfDim = dim / 2;
        slidingSin.resize(positions);
        slidingCos.resize(positions);
        std::vector<float> invFreq;
        for (int i = 0; i < dim; i += 2) {
            invFreq.push_back(1.0f / pow(base, (float)i / dim));
        }
        for (int i = 0; i < positions; i++) {
            slidingSin[i].resize(halfDim);
            slidingCos[i].resize(halfDim);
            for (int j = 0; j < halfDim; j++) {
                float angle = (float)i * invFreq[j];
                slidingSin[i][j] = ::sin(angle);
                slidingCos[i][j] = ::cos(angle);
            }
        }
        std::vector<float> fsin, fcos;
        for (int i = 0; i < (int)slidingSin.size(); i++) {
            fsin.insert(fsin.end(), slidingSin[i].begin(), slidingSin[i].end());
            fcos.insert(fcos.end(), slidingCos[i].begin(), slidingCos[i].end());
        }
        return std::make_pair(fsin, fcos);
    }

    int Gemma4Model::Forward(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <float> *retLogits) {
        std::vector <std::vector <float>*> batchLogits;
        batchLogits.push_back(retLogits);
        return ForwardBatch(1, inputIds, attentionMask, positionIds, pastKeyValues, generationConfig, lastTokens, &batchLogits)[0];
    }

    std::vector <int> Gemma4Model::ForwardBatch(int batch, const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <std::vector <float>*> *retLogits) {
        int maxLen = inputIds.dims[1];
        Data hiddenStates;
        Data attenInput;
        Data q, k, v, qkv;
        Data attenWeights, attenOutput;
        Data attenLastOutput;
        Data w1, w2, w3;

        std::string embName = "model.language_model.embed_tokens.weight";
        if (weight.weight.find(embName) == weight.weight.end()) {
            embName = "model.embed_tokens.weight";
        }

        if (!prepared) {
            Prepare();
            prepared = true;
        }

        Embedding(inputIds, this->weight[embName], hiddenStates);
        Mul(hiddenStates, sqrt((float)embed_dim), hiddenStates);
        ToDataType(hiddenStates, this->dataType);

        int seqlen = hiddenStates.dims[1];

        std::string layerPrefix = "model.language_model.layers.";
        if (weight.weight.find(layerPrefix + "0.input_layernorm.weight") == weight.weight.end()) {
            layerPrefix = "model.layers.";
        }

        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);

            bool isFullAttn = (layer_types[i] == 1);
            int curHeadDim = isFullAttn ? global_head_dim : sliding_head_dim;
            int curKVHeads = (isFullAttn && attention_k_eq_v) ? global_num_key_value_heads : num_key_value_heads;
            int curRotaryDim = isFullAttn ? global_head_dim / 2 : sliding_head_dim;
            Data *curSinData = isFullAttn ? &globalSinData : &slidingSinData;
            Data *curCosData = isFullAttn ? &globalCosData : &slidingCosData;

            std::string pre = layerPrefix + std::to_string(i);
            std::string qWeightName = pre + ".self_attn.q_proj.weight";
            std::string kWeightName = pre + ".self_attn.k_proj.weight";
            std::string vWeightName = pre + ".self_attn.v_proj.weight";
            std::string oWeightName = pre + ".self_attn.o_proj.weight";
            std::string qNormName = pre + ".self_attn.q_norm.weight";
            std::string kNormName = pre + ".self_attn.k_norm.weight";

            RMSNorm(hiddenStates, this->weight[pre + ".input_layernorm.weight"], rms_norm_eps, attenInput);

            int bsz = attenInput.dims[0];
            seqlen = attenInput.dims[1];

            Linear(attenInput, weight[qWeightName], *GetEmptyData(), q);
            Linear(attenInput, weight[kWeightName], *GetEmptyData(), k);

            bool useAltAttn = isFullAttn && attention_k_eq_v;
            if (useAltAttn) {
                v.CopyFrom(k);
            } else {
                Linear(attenInput, weight[vWeightName], *GetEmptyData(), v);
            }

            q.Reshape({bsz, seqlen, -1, curHeadDim});
            k.Reshape({bsz, seqlen, -1, curHeadDim});
            v.Reshape({bsz, seqlen, -1, curHeadDim});

            RMSNorm(q, this->weight[qNormName], rms_norm_eps, q);
            RMSNorm(k, this->weight[kNormName], rms_norm_eps, k);

            {
                v.ToDevice(DataDevice::CPU);
                if (v.dataType != DataType::FLOAT32) {
                    ToDataType(v, DataType::FLOAT32);
                }
                float *vp = (float*)v.cpuData;
                int lastDim = v.dims.back();
                uint64_t total = v.Count(0);
                int outer = (int)(total / lastDim);
                for (int o = 0; o < outer; o++) {
                    float *row = vp + (uint64_t)o * lastDim;
                    float sumSq = 0.0f;
                    for (int j = 0; j < lastDim; j++) {
                        sumSq += row[j] * row[j];
                    }
                    float scale = 1.0f / sqrtf(sumSq / lastDim + rms_norm_eps);
                    for (int j = 0; j < lastDim; j++) {
                        row[j] = row[j] * scale;
                    }
                }
            }
            Data &pastKey = pastKeyValues[i].first, &pastValue = pastKeyValues[i].second;
            if (GetKVCacheInCPU()) {
                pastKey.lockInCPU = true;
                pastValue.lockInCPU = true;
            } else {
                pastKey.ToDevice(k.dataDevice);
                pastValue.ToDevice(k.dataDevice);
            }

            fastllm::LlamaRotatePosition2D(q, positionIds, *curSinData, *curCosData, curRotaryDim);
            fastllm::LlamaRotatePosition2D(k, positionIds, *curSinData, *curCosData, curRotaryDim);

            PermuteSelf(q, {0, 2, 1, 3});
            PermuteSelf(k, {0, 2, 1, 3});
            PermuteSelf(v, {0, 2, 1, 3});

            std::vector<int> qkvSize = {-1, seqlen, curHeadDim};
            q.Reshape(qkvSize);
            k.Reshape(qkvSize);
            v.Reshape(qkvSize);

            int unitLen = 64;
#ifdef USE_CUDA
            unitLen = 128;
#endif
            while ((pastKey.dims.size() == 0 && (pastKey.expansionDims.size() == 0 || k.dims[1] > pastKey.expansionDims[1]))
                || (pastKey.dims.size() > 0 && pastKey.dims[1] + k.dims[1] > pastKey.expansionDims[1])) {
                std::vector <int> newDims;
                if (pastKey.Count(0) == 0 || pastKey.dims.size() == 0) {
                    newDims = std::vector <int> {k.dims[0], ((k.dims[1] - 1) / unitLen + 1) * unitLen, k.dims[2]};
                } else {
                    newDims = pastKey.dims;
                    newDims[1] += ((k.dims[1] - 1) / unitLen + 1) * unitLen;
                }
                pastKey.Expansion(newDims);
            }
            while ((pastValue.dims.size() == 0 && (pastValue.expansionDims.size() == 0 || v.dims[1] > pastValue.expansionDims[1]))
                || (pastValue.dims.size() > 0 && pastValue.dims[1] + v.dims[1] > pastValue.expansionDims[1])) {
                std::vector <int> newDims;
                if (pastValue.Count(0) == 0 || pastValue.dims.size() == 0) {
                    newDims = std::vector <int> {v.dims[0], ((v.dims[1] - 1) / unitLen + 1) * unitLen, v.dims[2]};
                } else {
                    newDims = pastValue.dims;
                    newDims[1] += ((v.dims[1] - 1) / unitLen + 1) * unitLen;
                }
                pastValue.Expansion(newDims);
            }
            CatDirect(pastKey, k, 1);
            CatDirect(pastValue, v, 1);

            Attention(q, pastKey, pastValue, attentionMask, qkv, q.dims[0] / pastKey.dims[0], 1.0f, 1);

            PermuteSelf(qkv, {1, 0, 2});
            qkv.Reshape({seqlen, bsz, -1});
            PermuteSelf(qkv, {1, 0, 2});

            Linear(qkv, weight[oWeightName], *GetEmptyData(), attenInput);
            RMSNorm(attenInput, this->weight[pre + ".post_attention_layernorm.weight"], rms_norm_eps, attenInput);
            AddTo(hiddenStates, attenInput);

            RMSNorm(hiddenStates, this->weight[pre + ".pre_feedforward_layernorm.weight"], rms_norm_eps, attenInput);

            Linear(attenInput, weight[pre + ".mlp.gate_proj.weight"], *GetEmptyData(), w1);
            Gelu(w1, w1);
            Linear(attenInput, weight[pre + ".mlp.up_proj.weight"], *GetEmptyData(), w3);

            if (this->dataType == DataType::FLOAT16) {
                ToDataType(w1, DataType::FLOAT32);
                ToDataType(w3, DataType::FLOAT32);
            }
            MulTo(w1, w3);
            Linear(w1, weight[pre + ".mlp.down_proj.weight"], *GetEmptyData(), w2);
            if (this->dataType == DataType::FLOAT16) {
                ToDataType(w2, DataType::FLOAT16);
            }

            RMSNorm(w2, this->weight[pre + ".post_feedforward_layernorm.weight"], rms_norm_eps, w2);
            AddTo(hiddenStates, w2);

            std::string layerScalarName = layerPrefix + std::to_string(i) + ".layer_scalar";
            if (weight.weight.find(layerScalarName) != weight.weight.end()) {
                Data &ls = weight[layerScalarName];
                ls.ToDevice(DataDevice::CPU);
                if (ls.dataType != DataType::FLOAT32) {
                    ToDataType(ls, DataType::FLOAT32);
                }
                float scalar = ((float*)ls.cpuData)[0];
                Mul(hiddenStates, scalar, hiddenStates);
            }

        }

        Data logits, topk;
        Data tempHiddenStates;
        Data *lastHiddenStates;
        if (maxLen > 1) {
            Split(hiddenStates, 1, maxLen - 1, maxLen, tempHiddenStates);
            lastHiddenStates = &tempHiddenStates;
        } else {
            lastHiddenStates = &hiddenStates;
        }

        std::vector <int> lastRet;
        {
            auto &hiddenStates = *lastHiddenStates;

            std::string normName = "model.language_model.norm.weight";
            if (weight.weight.find(normName) == weight.weight.end()) {
                normName = "model.norm.weight";
            }
            RMSNorm(hiddenStates, this->weight[normName], rms_norm_eps, hiddenStates);

            std::string lmHeadName = "lm_head.weight";
            Linear(hiddenStates, weight[lmHeadName], *GetEmptyData(), logits);
            ToDataType(logits, DataType::FLOAT32);

            if (final_logit_softcapping > 0.0f) {
                Mul(logits, 1.0f / final_logit_softcapping, logits);
                logits.ToDevice(DataDevice::CPU);
                float *logitsData = (float*)logits.cpuData;
                int total = logits.Count(0);
                for (int j = 0; j < total; j++) {
                    logitsData[j] = tanh(logitsData[j]) * final_logit_softcapping;
                }
            }

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

            TopK(logits, topk, 1);
            topk.ToDevice(DataDevice::CPU);
            for (int b = 0; b < batch; b++) {
                int base = b;
                lastRet.push_back((int) (((float *) topk.cpuData)[base * 2] + 1e-3));
            }
        }
        return lastRet;
    }

    std::vector <int> Gemma4Model::ForwardBatch(int batch,
                                               const Data &inputIds,
                                               const std::vector <Data*> &attentionMask,
                                               const std::vector <Data*> &positionIds,
                                               const std::vector <int> &seqLens,
                                               std::vector <std::pair <Data*, Data*> > &pastKeyValues,
                                               const std::vector <GenerationConfig> &generationConfigs,
                                               const LastTokensManager &lastTokens,
                                               std::vector <std::vector <float>*> *retLogits) {
        if (!prepared) {
            Prepare();
            prepared = true;
        }
        int seqLen = inputIds.dims[1];

        Data hiddenStates;
        Data attenInput;
        Data q, k, v, qkv;
        Data attenWeights, curAttenOutput;
        Data attenLastOutput;
        Data w1, w2, w3;
        std::vector <Data> curContextLayer;
        curContextLayer.resize(batch);
        std::vector <Data> curKs, curVs, curQs;
        curKs.resize(batch);
        curVs.resize(batch);
        curQs.resize(batch);
        std::vector <Data*> pointersK, pointersV, pointersQ;
        pointersK.resize(batch);
        pointersV.resize(batch);
        pointersQ.resize(batch);
        std::vector <Data*> keys, values, qs, attns, masks, contexts;
        keys.resize(batch);
        values.resize(batch);
        qs.resize(batch);
        attns.resize(batch);
        masks.resize(batch);
        contexts.resize(batch);
        Data allPositionIds;

        bool all1 = true;
        for (int i = 0; i < batch; i++) {
            all1 &= (seqLens[i] == 1);
        }
        if (all1 && positionIds[0]->dataType == DataType::FLOAT32) {
            std::vector <float> vPositionIds;
            for (int b = 0; b < batch; b++) {
                vPositionIds.push_back(((float*)positionIds[b]->cpuData)[0]);
            }
            allPositionIds.CopyFrom(Data(DataType::FLOAT32, {1, seqLen}, vPositionIds));
        } else {
            allPositionIds.CopyFrom(*(Data*)positionIds[0]);
            allPositionIds.Expansion({1, seqLen});
            for (int i = 1; i < batch; i++) {
                CatDirect(allPositionIds, *(Data*)positionIds[i], 1);
            }
        }

        std::string embName = "model.language_model.embed_tokens.weight";
        if (weight.weight.find(embName) == weight.weight.end()) {
            embName = "model.embed_tokens.weight";
        }
        Embedding(inputIds, this->weight[embName], hiddenStates);
        Mul(hiddenStates, sqrt((float)embed_dim), hiddenStates);
        ToDataType(hiddenStates, this->dataType);

        int seqlen = hiddenStates.dims[1];

        std::string layerPrefix = "model.language_model.layers.";
        if (weight.weight.find(layerPrefix + "0.input_layernorm.weight") == weight.weight.end()) {
            layerPrefix = "model.layers.";
        }

        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);

            bool isFullAttn = (layer_types[i] == 1);
            int curHeadDim = isFullAttn ? global_head_dim : sliding_head_dim;
            int curKVHeads = (isFullAttn && attention_k_eq_v) ? global_num_key_value_heads : num_key_value_heads;
            int curRotaryDim = isFullAttn ? global_head_dim / 2 : sliding_head_dim;
            Data *curSinData = isFullAttn ? &globalSinData : &slidingSinData;
            Data *curCosData = isFullAttn ? &globalCosData : &slidingCosData;

            std::string pre = layerPrefix + std::to_string(i);
            std::string qWeightName = pre + ".self_attn.q_proj.weight";
            std::string kWeightName = pre + ".self_attn.k_proj.weight";
            std::string vWeightName = pre + ".self_attn.v_proj.weight";
            std::string oWeightName = pre + ".self_attn.o_proj.weight";
            std::string qNormName = pre + ".self_attn.q_norm.weight";
            std::string kNormName = pre + ".self_attn.k_norm.weight";

            RMSNorm(hiddenStates, this->weight[pre + ".input_layernorm.weight"], rms_norm_eps, attenInput);

            int bsz = attenInput.dims[0];
            seqlen = attenInput.dims[1];

            Linear(attenInput, weight[qWeightName], *GetEmptyData(), q);
            Linear(attenInput, weight[kWeightName], *GetEmptyData(), k);

            bool useAltAttn = isFullAttn && attention_k_eq_v;
            if (useAltAttn) {
                v.CopyFrom(k);
            } else {
                Linear(attenInput, weight[vWeightName], *GetEmptyData(), v);
            }

            q.Reshape({bsz, seqlen, -1, curHeadDim});
            k.Reshape({bsz, seqlen, -1, curHeadDim});
            v.Reshape({bsz, seqlen, -1, curHeadDim});

            RMSNorm(q, this->weight[qNormName], rms_norm_eps, q);
            RMSNorm(k, this->weight[kNormName], rms_norm_eps, k);

            {
                v.ToDevice(DataDevice::CPU);
                if (v.dataType != DataType::FLOAT32) {
                    ToDataType(v, DataType::FLOAT32);
                }
                float *vp = (float*)v.cpuData;
                int lastDim = v.dims.back();
                uint64_t total = v.Count(0);
                int outer = (int)(total / lastDim);
                for (int o = 0; o < outer; o++) {
                    float *row = vp + (uint64_t)o * lastDim;
                    float sumSq = 0.0f;
                    for (int j = 0; j < lastDim; j++) {
                        sumSq += row[j] * row[j];
                    }
                    float sc = 1.0f / sqrtf(sumSq / lastDim + rms_norm_eps);
                    for (int j = 0; j < lastDim; j++) {
                        row[j] = row[j] * sc;
                    }
                }
            }

            int cacheOuter = k.dims[2], cacheInner = k.dims[3];
            for (int b = 0; b < batch; b++) {
                Data &pastKey = *pastKeyValues[b * block_cnt + i].first;
                Data &pastValue = *pastKeyValues[b * block_cnt + i].second;
                if (GetKVCacheInCPU()) {
                    pastKey.lockInCPU = true;
                    pastValue.lockInCPU = true;
                } else {
                    pastKey.ToDevice(k.dataDevice);
                    pastValue.ToDevice(k.dataDevice);
                }

                int curLen = seqLens[b];
                int unitLen = 64;
#ifdef USE_CUDA
                unitLen = 128;
#endif
                while ((pastKey.dims.size() == 0 &&
                        (pastKey.expansionDims.size() == 0 || curLen > pastKey.expansionDims[1]))
                    || (pastKey.dims.size() > 0 && pastKey.dims[1] + curLen > pastKey.expansionDims[1])) {
                    std::vector<int> newDims;
                    if (pastKey.Count(0) == 0 || pastKey.dims.size() == 0) {
                        newDims = std::vector<int> {cacheOuter, ((curLen - 1) / unitLen + 1) * unitLen, cacheInner};
                    } else {
                        newDims = pastKey.dims;
                        newDims[1] += ((curLen - 1) / unitLen + 1) * unitLen;
                    }
                    pastKey.Expansion(newDims);
                }
                while ((pastValue.dims.size() == 0 &&
                        (pastValue.expansionDims.size() == 0 || curLen > pastValue.expansionDims[1]))
                    || (pastValue.dims.size() > 0 && pastValue.dims[1] + curLen > pastValue.expansionDims[1])) {
                    std::vector<int> newDims;
                    if (pastValue.Count(0) == 0 || pastValue.dims.size() == 0) {
                        newDims = std::vector<int>{cacheOuter, ((curLen - 1) / unitLen + 1) * unitLen, cacheInner};
                    } else {
                        newDims = pastValue.dims;
                        newDims[1] += ((curLen - 1) / unitLen + 1) * unitLen;
                    }
                    pastValue.Expansion(newDims);
                }
            }

            fastllm::LlamaRotatePosition2D(q, allPositionIds, *curSinData, *curCosData, curRotaryDim);
            fastllm::LlamaRotatePosition2D(k, allPositionIds, *curSinData, *curCosData, curRotaryDim);

            int curEmbedDim = num_attention_heads * curHeadDim;
            Data attenOutput = Data(this->dataType);
            int total = 0;

            PermuteSelf(q, {0, 2, 1, 3});
            PermuteSelf(k, {0, 2, 1, 3});
            PermuteSelf(v, {0, 2, 1, 3});

            std::vector<int> qkvSize = {-1, seqlen, curHeadDim};
            q.Reshape(qkvSize);
            k.Reshape(qkvSize);
            v.Reshape(qkvSize);

            for (int b = 0; b < batch; b++) {
                Split(k, 1, total, total + seqLens[b], curKs[b]);
                Split(v, 1, total, total + seqLens[b], curVs[b]);
                Split(q, 1, total, total + seqLens[b], curQs[b]);
                total += seqLens[b];
            }

            for (int b = 0; b < batch; b++) {
                keys[b] = (pastKeyValues[b * block_cnt + i].first);
                values[b] = (pastKeyValues[b * block_cnt + i].second);
                pointersK[b] = (&curKs[b]);
                pointersV[b] = (&curVs[b]);
            }
            CatDirectBatch(keys, pointersK, 1);
            CatDirectBatch(values, pointersV, 1);

            attenOutput.ToDevice(curQs[0].dataDevice);
            attenOutput.Resize({1, total, curEmbedDim});
            attenOutput.Allocate();
            int curLen = 0;
            for (int b = 0; b < batch; b++) {
                auto &curQ = curQs[b];
                Data &pastKey = *pastKeyValues[b * block_cnt + i].first;
                Data &pastValue = *pastKeyValues[b * block_cnt + i].second;
                curAttenOutput.FakeFrom(attenOutput, curLen * curEmbedDim * attenOutput.unitSize);
                curLen += seqLens[b];

                if (attentionMask[b] == nullptr) {
                    Attention(curQ, pastKey, pastValue, *GetEmptyData(), curAttenOutput, curQ.dims[0] / pastKey.dims[0], 1.0f, 1);
                } else {
                    Attention(curQ, pastKey, pastValue, *attentionMask[b], curAttenOutput, curQ.dims[0] / pastKey.dims[0], 1.0f, 1);
                }
                PermuteSelf(curAttenOutput, {1, 0, 2});
            }

            Linear(attenOutput, weight[oWeightName], *GetEmptyData(), attenLastOutput);

            RMSNorm(attenLastOutput, this->weight[pre + ".post_attention_layernorm.weight"], rms_norm_eps, attenLastOutput);
            AddTo(hiddenStates, attenLastOutput);

            RMSNorm(hiddenStates, this->weight[pre + ".pre_feedforward_layernorm.weight"], rms_norm_eps, attenInput);

            Linear(attenInput, weight[pre + ".mlp.gate_proj.weight"], *GetEmptyData(), w1);
            Gelu(w1, w1);
            Linear(attenInput, weight[pre + ".mlp.up_proj.weight"], *GetEmptyData(), w3);
            if (this->dataType == DataType::FLOAT16) {
                ToDataType(w1, DataType::FLOAT32);
                ToDataType(w3, DataType::FLOAT32);
            }
            MulTo(w1, w3);
            Linear(w1, weight[pre + ".mlp.down_proj.weight"], *GetEmptyData(), w2);
            if (this->dataType == DataType::FLOAT16) {
                ToDataType(w2, DataType::FLOAT16);
            }

            RMSNorm(w2, this->weight[pre + ".post_feedforward_layernorm.weight"], rms_norm_eps, w2);
            AddTo(hiddenStates, w2);

            std::string layerScalarName = layerPrefix + std::to_string(i) + ".layer_scalar";
            if (weight.weight.find(layerScalarName) != weight.weight.end()) {
                Data &ls = weight[layerScalarName];
                ls.ToDevice(DataDevice::CPU);
                if (ls.dataType != DataType::FLOAT32) {
                    ToDataType(ls, DataType::FLOAT32);
                }
                float scalar = ((float*)ls.cpuData)[0];
                Mul(hiddenStates, scalar, hiddenStates);
            }
        }

        std::vector <int> lastRet;
        {
            std::string normName = "model.language_model.norm.weight";
            if (weight.weight.find(normName) == weight.weight.end()) {
                normName = "model.norm.weight";
            }

            std::string lmHeadName = "lm_head.weight";

            RMSNorm(hiddenStates, this->weight[normName], rms_norm_eps, hiddenStates);

            Data logits;
            Linear(hiddenStates, weight[lmHeadName], *GetEmptyData(), logits);
            ToDataType(logits, DataType::FLOAT32);

            if (final_logit_softcapping > 0.0f) {
                Mul(logits, 1.0f / final_logit_softcapping, logits);
                logits.ToDevice(DataDevice::CPU);
                float *logitsData = (float*)logits.cpuData;
                int totalElements = logits.Count(0);
                for (int j = 0; j < totalElements; j++) {
                    logitsData[j] = tanh(logitsData[j]) * final_logit_softcapping;
                }
            }

            if (generationConfigs[0].output_logits && retLogits != nullptr) {
                int size = logits.dims.back();
                logits.ToDevice(DataDevice::CPU);
                for (int b = 0; b < batch; b++) {
                    (*retLogits)[b]->resize(size);
                    int offset = 0;
                    for (int s = 0; s < b; s++) offset += seqLens[s];
                    offset += seqLens[b] - 1;
                    memcpy((float*)(*retLogits)[b]->data(),
                        ((float*)logits.cpuData) + offset * size,
                        size * logits.unitSize);
                }
            }

            Data topk;
            int vocabSize = logits.dims.back();
            std::vector<float> lastLogits(batch * vocabSize);
            logits.ToDevice(DataDevice::CPU);
            int pos = 0;
            for (int b = 0; b < batch; b++) {
                int offset = pos + seqLens[b] - 1;
                memcpy(lastLogits.data() + b * vocabSize,
                       ((float*)logits.cpuData) + offset * vocabSize,
                       vocabSize * sizeof(float));
                pos += seqLens[b];
            }
            Data lastLogitsData(DataType::FLOAT32, {batch, 1, vocabSize}, lastLogits);

            TopK(lastLogitsData, topk, 1);
            topk.ToDevice(DataDevice::CPU);
            for (int b = 0; b < batch; b++) {
                lastRet.push_back((int) (((float *) topk.cpuData)[b * 2] + 1e-3));
            }
        }
        return lastRet;
    }

    bool Gemma4Model::NeedAttentionMask(int qlen, int klen) {
        if (((qlen == 1) || (qlen >= 1024))) {
            return false;
        }
        return true;
    }

    void Gemma4Model::FillLLMInputsBatch(std::vector<std::vector<float>> &inputTokens,
                                          const std::vector<std::map<std::string, int>> &params,
                                          fastllm::Data &inputIds, fastllm::Data &attentionMask,
                                          fastllm::Data &positionIds) {
        inputIds.ToDevice(DataDevice::CPU);
        attentionMask.ToDevice(DataDevice::CPU);
        positionIds.ToDevice(DataDevice::CPU);

        int batch = inputTokens.size();
        int index = params[0].find("index")->second;
        if (index == 0) {
            std::vector <int> seqLens;
            seqLens.resize(batch);
            int maxLen = 0;
            for (int i = 0; i < batch; i++) {
                maxLen = std::max(maxLen, (int)inputTokens[i].size());
                seqLens[i] = (int)inputTokens[i].size();
            }

            std::vector <float> ids = std::vector <float> (batch * maxLen, 0);
            std::vector <float> vpids = std::vector <float> (batch * maxLen, 0);
            std::vector <float> vmask = std::vector <float> (batch * maxLen * maxLen, 0);
            for (int i = 0; i < batch; i++) {
                auto &tokens = inputTokens[i];
                int len = tokens.size(), base = maxLen - len;
                for (int j = 0; j < len; j++) {
                    ids[i * maxLen + base + j] = tokens[j];
                }
                for (int j = 0; j < len; j++) {
                    vpids[i * maxLen + base + j] = j;
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
            inputIds.CopyFrom(Data(DataType::FLOAT32, {batch, maxLen}, ids));
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {batch, maxLen, maxLen}, vmask));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {batch, maxLen}, vpids));
        } else {
            std::vector <float> pids = std::vector <float> (batch);
            std::vector <float> fret;
            for (int i = 0; i < batch; i++) {
                fret.push_back(inputTokens[i][0]);
            }
            int maxLen = 0;
            for (int i = 0; i < batch; i++) {
                int promptLen = params[i].find("promptLen")->second;
                maxLen = std::max(promptLen, maxLen);
                pids[i] = promptLen + index - 1;
            }
            maxLen += index;
            std::vector <float> vmasks = std::vector <float> (batch * maxLen, 0.0f);
            for (int i = 0; i < batch; i++) {
                int curLen = params[i].find("promptLen")->second + index;
                for (int j = 0; j < maxLen - curLen; j++) {
                    vmasks[i * maxLen + j] = 1.0f;
                }
            }
            inputIds.CopyFrom(Data(DataType::FLOAT32, {batch, 1}, fret));
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {batch, 1, maxLen}, vmasks));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {batch, 1}, pids));
        }
    }

    std::string Gemma4Model::MakeInput(const std::string &history, int round, const std::string &input) {
        if (round == 0) {
            return "<bos><|turn>user\n" + input + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
        }
        return history + "<|turn>user\n" + input + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
    }

    std::string Gemma4Model::MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) {
        if (round == 0) {
            return "<bos><|turn>user\n" + input + "<turn|>\n<|turn>model\n" + output + "<turn|>\n";
        }
        return history + "<|turn>user\n" + input + "<turn|>\n<|turn>model\n" + output + "<turn|>\n";
    }

    void Gemma4Model::Prepare() {
        std::string lmHeadName = "lm_head.weight";
        std::string embName = "model.language_model.embed_tokens.weight";
        if (weight.weight.find(embName) == weight.weight.end()) {
            embName = "model.embed_tokens.weight";
        }
        bool lmHeadExists = this->weight.weight.find(lmHeadName) != this->weight.weight.end();
        if (!lmHeadExists || this->weight[lmHeadName].dims.size() == 0) {
            this->weight[lmHeadName].CopyFrom(this->weight[embName]);
            ToDataType(this->weight[lmHeadName], this->dataType);
        }
    }

    void Gemma4Model::WarmUp() {
        Prepare();

        Data inputIds = Data(DataType::FLOAT32, {1, 1}, {1});
        Data attentionMask = Data(DataType::FLOAT32, {1, 1}, {0});
        Data positionIds = Data(DataType::FLOAT32, {1, 1}, {0, 0});

        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(this->dataType),
                                                   Data(this->dataType)));
        }
        Forward(inputIds, attentionMask, positionIds, pastKeyValues);
        elementsInKVCachePerToken = 0;
        for (int i = 0; i < block_cnt; i++) {
            elementsInKVCachePerToken +=
                pastKeyValues[i].first.dims[0] * pastKeyValues[i].first.dims[2] +
                pastKeyValues[i].second.dims[0] * pastKeyValues[i].second.dims[2];
        }
    }
}
