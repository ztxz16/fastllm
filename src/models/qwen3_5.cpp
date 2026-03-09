//
// Created by huangyuyang on 2/19/26.
//

#include "utils.h"

#include "qwen3_5.h"

#include <sstream>

#include <unordered_map>

#include <cstring>

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif

namespace fastllm {
    static void Add1(Data &input) {
        if (input.dims.size() == 0) {
            return;
        }
        float *v = (float*)input.cpuData;
        int len = input.Count(0);
        for (int i = 0; i < len; i++) {
            v[i] += 1.0f;
        }
    }

    const std::string Qwen3_5Model::language_prefix = "model.language_model.";
    const std::string Qwen3_5Model::visual_prefix = "model.visual.";

    int Qwen3_5Model::Forward(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
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

    Qwen3_5Model::Qwen3_5Model() {
        this->model_struct = "qwen3_5";
        this->model_type = "qwen3_5";
        this->use_new_engine = true;

        weight.embeddingNames.insert(language_prefix + "embed_tokens.weight");
        weight.linearNames = {
            "lm_head.weight",
            language_prefix + "layers.*.mlp.down_proj.weight", language_prefix + "layers.*.mlp.up_proj.weight",
            language_prefix + "layers.*.mlp.gate_proj.weight", language_prefix + "layers.*.mlp.gate_proj.weight",
            language_prefix + "layers.*.mlp.gateup_proj.weight",
            language_prefix + "layers.*.self_attn.o_proj.weight", language_prefix + "layers.*.self_attn.q_proj.weight",
            language_prefix + "layers.*.self_attn.k_proj.weight",
            language_prefix + "layers.*.self_attn.v_proj.weight", language_prefix + "layers.*.self_attn.mergeqkv.weight",
            language_prefix + "layers.*.self_attn.W_pack.weight"
        };
    }

    void Qwen3_5Model::InitParams() {
        std::map<std::string, std::string> extra;
        for (auto &it : this->weight.dicts) {
            std::string key = it.first;
            if (key.substr(0, 12) == "text_config.") {
                std::string stripped = key.substr(12);
                if (stripped.substr(0, 16) == "rope_parameters.") {
                    stripped = stripped.substr(16);
                }
                extra[stripped] = it.second;
            }
        }
        for (auto &it : extra) {
            if (this->weight.dicts.find(it.first) == this->weight.dicts.end()) {
                this->weight.dicts[it.first] = it.second;
            }
        }
        basellm::InitParams();
        num_key_value_heads = num_attention_heads;
        if (this->weight.dicts.find("num_key_value_heads") != this->weight.dicts.end()) {
            num_key_value_heads = atoi(this->weight.dicts["num_key_value_heads"].c_str());
        }
        if (this->weight.dicts.find("linear_num_key_heads") != this->weight.dicts.end()) {
            num_k_heads = atoi(this->weight.dicts["linear_num_key_heads"].c_str());
        }
        if (this->weight.dicts.find("linear_num_value_heads") != this->weight.dicts.end()) {
            num_v_heads = atoi(this->weight.dicts["linear_num_value_heads"].c_str());
        }
        if (this->weight.dicts.find("linear_key_head_dim") != this->weight.dicts.end()) {
            head_k_dim = atoi(this->weight.dicts["linear_key_head_dim"].c_str());
        }
        if (this->weight.dicts.find("linear_value_head_dim") != this->weight.dicts.end()) {
            head_v_dim = atoi(this->weight.dicts["linear_value_head_dim"].c_str());
        }
        head_dim = embed_dim / num_attention_heads;
        if (this->weight.dicts.find("head_dim") != this->weight.dicts.end()) {
            head_dim = atoi(this->weight.dicts["head_dim"].c_str());
        }
        if (this->weight.dicts.find("partial_rotary_factor") != this->weight.dicts.end()) {
            rotary_dim = (int)(head_dim * atof(this->weight.dicts["partial_rotary_factor"].c_str()) + 1e-5);
        } else {
            rotary_dim = (int)(head_dim * 0.25 + 1e-5); // qwen3.5的默认值
        }
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
            std::string w1WeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.gate_proj.weight";
            std::string w3WeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.up_proj.weight";
            std::string swigluWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
            this->weightMergeRules.push_back(
                WeightMergeRule({WeightMergeRuleSingle({w1WeightName, w3WeightName}, swigluWeightName, std::string("linearSwiglu"))})
            );
/*
            std::string qWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.q_proj.weight";
            std::string qBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.q_proj.bias";
            std::string kWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.k_proj.weight";
            std::string kBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.k_proj.bias";
            std::string vWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.v_proj.weight";
            std::string vBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.v_proj.bias";
            std::string mergeQkvWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
            std::string mergeQkvBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";
            this->weightMergeRules.push_back(
                WeightMergeRule({WeightMergeRuleSingle({qWeightName, kWeightName, vWeightName}, mergeQkvWeightName, std::string("linear")),
                                 WeightMergeRuleSingle({qBiasName, kBiasName, vBiasName}, mergeQkvBiasName, std::string("bias"))})
            );
*/
        }

        float inv_scale = pow((float)head_k_dim, -0.5);
        std::vector <float> v_inv_scale(head_k_dim, inv_scale);
        Data temp(DataType::FLOAT32, std::vector<int>{head_k_dim}, v_inv_scale);
        inv_scale_data.CopyFrom(temp);
    }

    std::vector <int> Qwen3_5Model::ForwardV2(
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



        if (!initialized_add1) {
            for (int i = 0; i < block_cnt; i++) {
                Add1(this->weight[language_prefix + "layers." + std::to_string(i) + ".input_layernorm.weight"]);
                Add1(this->weight[language_prefix + "layers." + std::to_string(i) + ".self_attn.q_norm.weight"]);
                Add1(this->weight[language_prefix + "layers." + std::to_string(i) + ".self_attn.k_norm.weight"]);
                Add1(this->weight[language_prefix + "layers." + std::to_string(i) + ".post_attention_layernorm.weight"]);
            }
            Add1(this->weight[language_prefix + "norm.weight"]);
            initialized_add1 = true;
        }

        if (this->weight.weight.find("lm_head.weight") == this->weight.weight.end()) {
            // 如果tie weight，那么embedding在cuda上处理
            SetCudaEmbedding(true);
        }
        Embedding(inputIds, this->weight[language_prefix + "embed_tokens.weight"], embeddingResult);

        ToDataType(embeddingResult, hiddenStates, this->dataType);
        int seqlen = hiddenStates.dims[1];
        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            std::string inputRmsName = language_prefix + "layers." + std::to_string(i) + ".input_layernorm.weight";
            std::string mergeQkvWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
            std::string mergeQkvBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";
            std::string qNormName = language_prefix + "layers." + std::to_string(i) + ".self_attn.q_norm.weight";
            std::string kNormName = language_prefix + "layers." + std::to_string(i) + ".self_attn.k_norm.weight";
            std::string oWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.o_proj.weight";
            std::string oBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.o_proj.bias";
            std::string postRmsName = language_prefix + "layers." + std::to_string(i) + ".post_attention_layernorm.weight";
            std::string swigluWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
            std::string downWeightName = language_prefix + "layers." + std::to_string(i) + ".mlp.down_proj.weight";

            RMSNorm(hiddenStates, this->weight[inputRmsName], rms_norm_eps, attenInput);
            int bsz = attenInput.dims[0], seqlen = attenInput.dims[1];
            Data &pastKey = *pastKeyValues[i].first, &pastValue = *pastKeyValues[i].second;

            if (weight.weight.find(language_prefix + "layers." + std::to_string(i) + ".self_attn.o_proj.weight") != weight.weight.end()) {
                // Gate Attention Block
                std::string qWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.q_proj.weight";
                std::string qBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.q_proj.bias";
                std::string kWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.k_proj.weight";
                std::string kBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.k_proj.bias";
                std::string vWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.v_proj.weight";
                std::string vBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.v_proj.bias";
                std::string qkvWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.W_pack.weight";
                std::string oWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.o_proj.weight";
                std::string oBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.o_proj.bias";
                std::string mergeQkvWeightName = language_prefix + "layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
                std::string mergeQkvBiasName = language_prefix + "layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";

                Data qgate, q, gate, k, v;
                Linear(attenInput, weight[qWeightName], weight[qBiasName], qgate);
                qgate.Reshape({bsz, seqlen, -1, this->head_dim * 2});
                Split(qgate, -1, 0, this->head_dim, q);
                Split(qgate, -1, this->head_dim, qgate.dims.back(), gate);
                gate.Reshape({bsz, seqlen, -1});

                Linear(attenInput, weight[kWeightName], weight[kBiasName], k);
                Linear(attenInput, weight[vWeightName], weight[vBiasName], v);                

                k.Reshape({bsz, seqlen, -1, this->head_dim});
                v.Reshape({bsz, seqlen, -1, this->head_dim});

                RMSNorm(q, this->weight[language_prefix + "layers." + std::to_string(i) + ".self_attn.q_norm.weight"], rms_norm_eps, q);
                RMSNorm(k, this->weight[language_prefix + "layers." + std::to_string(i) + ".self_attn.k_norm.weight"], rms_norm_eps, k);
                {
                    float ropeScale = (rope_type == RoPEType::LINEAR_SCALE) ? rope_factor : 1.0f;
                    fastllm::RopeEncoding(q, allPositionIds, rotary_dim, rope_base, ropeScale);
                    fastllm::RopeEncoding(k, allPositionIds, rotary_dim, rope_base, ropeScale);
                }

                PermuteSelf(q, {0, 2, 1, 3});
                PermuteSelf(k, {0, 2, 1, 3});
                PermuteSelf(v, {0, 2, 1, 3});
                std::vector <int> qkvSize = {-1, seqlen, head_dim};
                q.Reshape(qkvSize);
                k.Reshape(qkvSize);
                v.Reshape(qkvSize);

                {
                    // Paged Attention
                    PagedCacheManager *pagedCacheKManager = AllocatePagedCacheManager(
                        i * 2, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, k);
                    PagedCacheManager *pagedCacheVManager = AllocatePagedCacheManager(
                        i * 2 + 1, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, v);
                    AppendPagedCache(*pagedCacheKManager, pastKey, k);
                    AppendPagedCache(*pagedCacheVManager, pastValue, v);
                    AttentionPaged(q, pastKey, pastValue, qkv, q.dims[0] / k.dims[0], 1.0 / sqrt(head_dim), 1, i > 0);
                }

                PermuteSelf(qkv, {1, 0, 2});
                qkv.Reshape({seqlen, bsz, -1});
                PermuteSelf(qkv, {1, 0, 2});

                Sigmoid(gate, gate);
                MulTo(qkv, gate);
                
                Data oBias = (weight.weight.find(oBiasName) != weight.weight.end()) ? weight[oBiasName] : Data();
                Linear(qkv, weight[oWeightName], oBias, attenInput);
            } else {
                // Gated Delta Net Block
                Data &pastKey = *pastKeyValues[i].first, &pastValue = *pastKeyValues[i].second;
                pastKey.isLinearAttention = pastValue.isLinearAttention = true;
                std::string qkvWeightName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.in_proj_qkv.weight";
                std::string zWeightName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.in_proj_z.weight";
                std::string bWeightName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.in_proj_b.weight";
                std::string aWeightName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.in_proj_a.weight";
                std::string conv1dWeightName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.conv1d.weight";
                std::string conv1dBiasName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.conv1d.bias";
                std::string aLogName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.A_log";
                std::string dtBiasName = language_prefix + "layers." + std::to_string(i) + ".linear_attn.dt_bias";

                int kd = num_k_heads * head_k_dim, vd = num_v_heads * head_v_dim;

                Data mixed_qkv, z, b, a, g;
                Linear(attenInput, weight[qkvWeightName], Data(), mixed_qkv);
                Linear(attenInput, weight[zWeightName], Data(), z);
                Linear(attenInput, weight[bWeightName], Data(), b);
                Linear(attenInput, weight[aWeightName], Data(), a);

                // mixed_qkv: (bsz, seqlen, key_dim*2+value_dim) -> transpose to (bsz, key_dim*2+value_dim, seqlen)
                PermuteSelf(mixed_qkv, {0, 2, 1});

                z.Reshape({bsz, seqlen, -1, head_v_dim});
                Data conv;
                if (bsz == 1 && seqlen == 1 && pastKey.dims.size() > 0) {
                    Data hidden_states_new;
                    Cat(pastKey, mixed_qkv, -1, hidden_states_new);
                    Split(hidden_states_new, -1, hidden_states_new.dims.back() - 4, hidden_states_new.dims.back(), pastKey);
                    Conv1DPerChannel(
                        hidden_states_new, weight[conv1dWeightName], weight[conv1dBiasName], 
                        hidden_states_new.dims[1], weight[conv1dWeightName].dims[0], 4, 1, 0, 
                        conv
                    );
                    Split(conv, -1, conv.dims.back() - 1, conv.dims.back(), mixed_qkv);
                    Silu(mixed_qkv, mixed_qkv);
                } else {
                    if (mixed_qkv.dims.back() >= 4) {
                        Split(mixed_qkv, -1, mixed_qkv.dims.back() - 4, mixed_qkv.dims.back(), pastKey);
                        pastKey.expansionDims = pastKey.dims;
                    } else {
                        ErrorInFastLLM("mixed_qkv.dims.back() < 4");
                    }

                    Conv1DPerChannel(
                        mixed_qkv, weight[conv1dWeightName], weight[conv1dBiasName], 
                        mixed_qkv.dims[1], weight[conv1dWeightName].dims[0], 4, 1, 3, 
                        conv
                    );
                    Split(conv, -1, 0, seqlen, mixed_qkv);
                    Silu(mixed_qkv, mixed_qkv);
                }

                // mixed_qkv: (bsz, conv_dim, seqlen) -> transpose back to (bsz, seqlen, conv_dim)
                PermuteSelf(mixed_qkv, {0, 2, 1});

                Split(mixed_qkv, -1, 0, kd, q);
                Split(mixed_qkv, -1, kd, kd + kd, k);
                Split(mixed_qkv, -1, kd + kd, kd + kd + vd, v);
                
                q.Reshape({q.dims[0], q.dims[1], -1, head_k_dim});
                k.Reshape({k.dims[0], k.dims[1], -1, head_k_dim});
                v.Reshape({v.dims[0], v.dims[1], -1, head_v_dim});

                Sigmoid(b, b);
                MambaSoftplus(a, weight[aLogName], weight[dtBiasName], g);


                Data &last_recurrent_state = pastValue;

                Data core_attn_out, core_attn_out_temp;
                if (bsz == 1 && seqlen == 1 && pastKey.dims.size() > 0) {
                    // torch_recurrent_gated_delta_rule
                    {
                        RMSNorm(q, inv_scale_data, rms_norm_eps, q);
                        RMSNorm(k, inv_scale_data, rms_norm_eps, k);
                    }

                    PermuteSelf(q, {0, 2, 1, 3});
                    PermuteSelf(k, {0, 2, 1, 3});
                    PermuteSelf(v, {0, 2, 1, 3});
                    PermuteSelf(b, {0, 2, 1});
                    PermuteSelf(g, {0, 2, 1});

                    int key_batch_size = k.dims[0], key_sequence_length = k.dims[1], key_num_heads = k.dims[2], key_k_head_dim = k.dims[3];
                    int v_head_dim_local = v.dims.back();
                    float scale = 1.0f / pow(q.dims.back(), 0.5);
                    Mul(q, scale, q);

                    RecurrentGatedDeltaRule (
                        q, k, v, g, b,
                        last_recurrent_state, 
                        core_attn_out
                    ); 
                    PermuteSelf(core_attn_out, {0, 2, 1, 3});
                } else {
                    // torch_chunk_gated_delta_rule
                    if (num_v_heads / num_k_heads > 1) {
                        Data qrepeat, krepeat;
                        Mul(q, 1.0f, qrepeat);
                        Mul(k, 1.0f, krepeat);

                        qrepeat.Resize({q.dims[0], q.dims[1], q.dims[2], 1, q.dims[3]});
                        krepeat.Resize({k.dims[0], k.dims[1], k.dims[2], 1, k.dims[3]});

                        Repeat(qrepeat, 3, num_v_heads / num_k_heads, q);
                        Repeat(krepeat, 3, num_v_heads / num_k_heads, k);

                        q.Reshape({q.dims[0], q.dims[1], -1, q.dims.back()});
                        k.Reshape({k.dims[0], k.dims[1], -1, k.dims.back()});
                    }

                    {
                        RMSNorm(q, inv_scale_data, rms_norm_eps, q);
                        RMSNorm(k, inv_scale_data, rms_norm_eps, k);
                    }

                    PermuteSelf(q, {0, 2, 1, 3});
                    PermuteSelf(k, {0, 2, 1, 3});
                    PermuteSelf(v, {0, 2, 1, 3});
                    PermuteSelf(b, {0, 2, 1});
                    PermuteSelf(g, {0, 2, 1});
                    
                    int key_batch_size = k.dims[0], key_sequence_length = k.dims[1], key_num_heads = k.dims[2], key_k_head_dim = k.dims[3];

                    int chunk_size = 64;
                    int v_head_dim_local = v.dims.back();
                    int seq = k.dims[2];
                    int pad_size = (chunk_size - seq % chunk_size) % chunk_size;

                    Data qtemp, qq, kk, vv, bb, gg, decayMask;
                    {
                        Pad(q, 2, pad_size, qtemp);
                        Pad(k, 2, pad_size, kk);
                        Pad(v, 2, pad_size, vv);
                        Pad(b, 2, pad_size, bb);
                        Pad(g, 2, pad_size, gg);
                    }

                    int tot_heads = seq + pad_size;
                    float scale = 1.0f / pow(qtemp.dims.back(), 0.5);
                    Mul(qtemp, scale, qq);

                    bb.Resize({bb.dims[0], bb.dims[1], bb.dims[2], 1});
                    Data k_beta, v_beta;
                    Mul(kk, 1.0f, k_beta);
                    Mul(vv, 1.0f, v_beta);
                    MulTo(k_beta, bb);
                    MulTo(v_beta, bb);

                    qq.Reshape({qq.dims[0], qq.dims[1], -1, chunk_size, qq.dims.back()});
                    kk.Reshape({kk.dims[0], kk.dims[1], -1, chunk_size, kk.dims.back()});
                    k_beta.Reshape({k_beta.dims[0], k_beta.dims[1], -1, chunk_size, k_beta.dims.back()});
                    v_beta.Reshape({v_beta.dims[0], v_beta.dims[1], -1, chunk_size, v_beta.dims.back()});
                    gg.Reshape({gg.dims[0], gg.dims[1], -1, chunk_size});

                    CumSumLastDim(gg);
                    MakeDecayMask(gg, decayMask);

                    Data at, attn;
                    MatMulTransB(k_beta, kk, at);
                    Mul(at, -1.0f, attn);
                    MulTo(attn, decayMask);

                    CausalMask(attn, 0, 0.0f);
                    TransferAttn(attn);
                    MatMul(attn, v_beta, vv);
                    Data k_temp, k_cumdecay;                    
                    Exp(gg, g);

                    Mul(k_beta, 1.0f, k_temp);
                    MulTo(k_temp, g);
                    MatMul(attn, k_temp, k_cumdecay);

                    if (last_recurrent_state.dims.size() == 0) {
                        last_recurrent_state.Resize({key_batch_size, key_sequence_length, key_k_head_dim, v_head_dim_local});
                        last_recurrent_state.Allocate(0.0f);
                    }

                    for (int ci = 0; ci < tot_heads / chunk_size; ci++) {
                        Data q_i, k_i, v_i, decay_mask_i, k_cumdecay_i;
                        Split(qq, 2, ci, ci + 1, q_i);
                        Split(kk, 2, ci, ci + 1, k_i);
                        Split(vv, 2, ci, ci + 1, v_i);

                        q_i.Resize({q_i.dims[0], q_i.dims[1], q_i.dims[3], q_i.dims[4]});
                        k_i.Resize({k_i.dims[0], k_i.dims[1], k_i.dims[3], k_i.dims[4]});
                        v_i.Resize({v_i.dims[0], v_i.dims[1], v_i.dims[3], v_i.dims[4]});

                        Split(decayMask, 2, ci, ci + 1, decay_mask_i);
                        decay_mask_i.Resize({decay_mask_i.dims[0], decay_mask_i.dims[1], decay_mask_i.dims[3], decay_mask_i.dims[4]});

                        MatMulTransB(q_i, k_i, attn);
                        MulTo(attn, decay_mask_i);
                        CausalMask(attn, 1, 0.0f);

                        Split(k_cumdecay, 2, ci, ci + 1, k_cumdecay_i);
                        k_cumdecay_i.Resize({k_cumdecay_i.dims[0], k_cumdecay_i.dims[1], k_cumdecay_i.dims[3], k_cumdecay_i.dims[4]});

                        Data v_prime, v_new;
                        MatMul(k_cumdecay_i, last_recurrent_state, v_prime);
                        Mul(v_prime, -1.0f, v_new);
                        AddTo(v_new, v_i);

                        Data attn_inter, g_i, g_i_exp, q_i_temp;
                        Split(gg, 2, ci, ci + 1, g_i);
                        g_i.Resize({g_i.dims[0], g_i.dims[1], g_i.dims[3], 1});
                        Exp(g_i, g_i_exp);
                        Mul(q_i, 1.0f, q_i_temp);
                        MulTo(q_i_temp, g_i_exp);

                        MatMul(q_i_temp, last_recurrent_state, attn_inter);
                        Data atv;
                        MatMul(attn, v_new, atv);
                        AddTo(atv, attn_inter);
                        atv.Resize({atv.dims[0], atv.dims[1], 1, atv.dims[2], atv.dims[3]});
                        if (ci == 0) {
                            Mul(atv, 1.0f, core_attn_out);
                        } else {
                            Mul(core_attn_out, 1.0f, core_attn_out_temp);
                            Cat(core_attn_out_temp, atv, 3, core_attn_out);
                        }

                        g_i.Resize({g_i.dims[0], g_i.dims[1], g_i.dims[2]});
                        Data g_i_last, g_i_last_repeat, g_i_l_temp;
                        Split(g_i, -1, g_i.dims.back() - 1, g_i.dims.back(), g_i_last);
                        Repeat(g_i_last, -1, g_i.dims.back(), g_i_last_repeat);
                        Mul(g_i, -1.0f, g_i_l_temp);
                        AddTo(g_i_l_temp, g_i_last_repeat);
                        Exp(g_i_l_temp, g_i_l_temp);
                        g_i_l_temp.Resize({g_i_l_temp.dims[0], g_i_l_temp.dims[1], g_i_l_temp.dims[2], 1});
                        MulTo(k_i, g_i_l_temp);
                        PermuteSelf(k_i, {0, 1, 3, 2});

                        Data k_i_v_new;
                        MatMul(k_i, v_new, k_i_v_new);

                        Data g_i_exp_last;
                        Split(g_i_exp, 2, g_i_exp.dims[2] - 1, g_i_exp.dims[2], g_i_exp_last);
                        MulTo(last_recurrent_state, g_i_exp_last);
                        AddTo(last_recurrent_state, k_i_v_new);
                    }

                    core_attn_out.Reshape({core_attn_out.dims[0], core_attn_out.dims[1], -1, core_attn_out.dims.back()});
                    Split(core_attn_out, 2, 0, seq, core_attn_out_temp);
                    PermuteSelf(core_attn_out_temp, {0, 2, 1, 3});
                    Mul(core_attn_out_temp, 1.0f, core_attn_out);
                }

                {
                    std::vector <int> zShape = z.dims;
                    core_attn_out.Reshape({-1, core_attn_out.dims.back()});
                    z.Reshape({-1, z.dims.back()});

                    RMSNorm(core_attn_out, this->weight[language_prefix + "layers." + std::to_string(i) + ".linear_attn.norm.weight"], rms_norm_eps, core_attn_out);
                    Silu(z, z);
                    MulTo(core_attn_out, z);

                    core_attn_out.Reshape({zShape[0], zShape[1], -1});
                    Linear(core_attn_out, 
                        this->weight[language_prefix + "layers." + std::to_string(i) + ".linear_attn.out_proj.weight"], 
                        Data(), attenInput);
                }
            }

            AddTo(hiddenStates, attenInput);
            RMSNorm(hiddenStates, this->weight[postRmsName], rms_norm_eps, attenInput);
            MLPBlock(&attenInput, &weight[swigluWeightName], &weight[downWeightName], &v, &q, &hiddenStates);
        }

        std::string lmHeadWeightName = "lm_head.weight";
        if (this->weight.weight.find(lmHeadWeightName) == this->weight.weight.end()) {
            lmHeadWeightName = language_prefix + "embed_tokens.weight";
        }
        std::vector <int> lastRet;
        LLMSamplingBlock(
            this, &hiddenStates,
            &weight[language_prefix + "norm.weight"], &weight[lmHeadWeightName],
            rms_norm_eps, batch, all1, seqLens,
            pastKeyValues, generationConfigs, lastTokens,
            retLogits, lastRet
        );
        return lastRet;
    }

    bool Qwen3_5Model::NeedAttentionMask(int qlen, int klen) {
        return false;
    }

    std::string Qwen3_5Model::MakeInput(const std::string &history, int round, const std::string &input) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role;
    }

    std::string Qwen3_5Model::MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role + output + history_sep;
    }

    void Qwen3_5Model::WarmUp() {
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
            this->weight["lm_head.weight"].CopyFrom(this->weight[language_prefix + "embed_tokens.weight"]);
            ToDataType(this->weight["lm_head.weight"], this->dataType);
        }
        Forward(inputIds, attentionMask, positionIds, pastKeyValues);
        elementsInKVCachePerToken = (long long)block_cnt * 
            (pastKeyValues[0].first.dims[0] * pastKeyValues[0].first.dims[2] + 
             pastKeyValues[0].second.dims[0] * pastKeyValues[0].second.dims[2]);
        printf("finish.\n");
    }
}
