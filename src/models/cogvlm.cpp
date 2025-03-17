//
// Created by huangyuyang on 9/27/24.
//

#include "utils.h"

#include "cogvlm.h"

#include <sstream>

#include <unordered_map>

#include <cstring>

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif

namespace fastllm {
    extern std::vector <float> GetInterLeavePowerOf2(int n);
    extern std::vector <float> GetInterleave(int n);

    CogvlmModel::CogvlmModel() {
        this->model_struct = "cogvlm";
        this->model_type = "cogvlm";

        // 默认使用 llama3 的提示词和instruction
        this->pre_prompt="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.<|eot_id|>";
        this->user_role="<|start_header_id|>user<|end_header_id|>\n";
        this->bot_role="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n";
        this->history_sep="<|eot_id|>\n";

        block_cnt = 32;
        rotary_dim = 128;

        weight.embeddingNames.insert("model.embed_tokens.weight");
        weight.linearNames = {
            "*conv.weight",
            "*query_key_value.weight", "*dense.weight", "*.mlp.fc1.weight", "*.mlp.fc2.weight",
            "*proj.weight", "*.dense_h_to_4h.weight", "*.dense_4h_to_h.weight", "lm_head.weight"
        };
    }

    void CogvlmModel::InitParams() {
        basellm::InitParams();
        this->vision_hidden_size = atoi(this->weight.dicts["vision_config.hidden_size"].c_str());
        this->vision_image_size = atoi(this->weight.dicts["vision_config.image_size"].c_str());
        this->vision_in_channels = atoi(this->weight.dicts["vision_config.in_channels"].c_str());
        this->vision_patch_size = atoi(this->weight.dicts["vision_config.patch_size"].c_str());
    
        if (this->weight.dicts.find("layer_norm_eps") != this->weight.dicts.end()) {
            this->layer_norm_eps = atof(this->weight.dicts["layer_norm_eps"].c_str());
        }

        num_key_value_heads = 8;
        if (this->weight.dicts.find("num_key_value_heads") != this->weight.dicts.end()) {
            num_key_value_heads = atoi(this->weight.dicts["num_key_value_heads"].c_str());
        }
        if (this->weight.dicts.find("num_multi_query_heads") != this->weight.dicts.end()) {
            num_key_value_heads = atoi(this->weight.dicts["num_multi_query_heads"].c_str());
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
        } else {
            rope_base = 500000;

        }
        if (this->weight.dicts.find("rope_scaling.factor") != this->weight.dicts.end()) {
            rope_factor = atof(this->weight.dicts["rope_scaling.factor"].c_str());
        }
        std::pair<std::vector<float>, std::vector<float>> &&pair = this->UpdateRotaryPosEmb(rope_base, rope_factor, std::max(max_positions, 16384));
        sinData.ToDevice(DataDevice::CPU);
        cosData.ToDevice(DataDevice::CPU);
        sinData.CopyFrom(Data(DataType::FLOAT32, { (int)this->sin.size(), (int)this->sin[0].size() }, pair.first));
        cosData.CopyFrom(Data(DataType::FLOAT32, { (int)this->cos.size(), (int)this->cos[0].size() }, pair.second));
    }

    std::pair<std::vector<float>, std::vector<float>> CogvlmModel::UpdateRotaryPosEmb(float base, float factor, int seqLen) {
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

    int CogvlmModel::Forward(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <float> *retLogits) {
        std::vector <std::vector <float>*> batchLogits;
        batchLogits.push_back(retLogits);
        return ForwardBatch(1, inputIds, attentionMask, positionIds, pastKeyValues, generationConfig, lastTokens, &batchLogits)[0];
    }

    std::vector <int> CogvlmModel::ForwardBatch(int batch, const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <std::vector <float>*> *retLogits) {
        return {};
    }

    std::vector <int> CogvlmModel::ForwardBatch(int batch,
                                               const Data &inputIds,
                                               const std::vector <Data*> &attentionMask,
                                               const std::vector <Data*> &positionIds,
                                               const std::vector <int> &seqLens,
                                               std::vector <std::pair <Data*, Data*> > &pastKeyValues,
                                               const std::vector <GenerationConfig> &generationConfigs,
                                               const LastTokensManager &lastTokens,
                                               std::vector <std::vector <float>*> *retLogits) {
        return {};
    }

    std::vector <int> CogvlmModel::ForwardMultimodal(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector<std::pair<Data, Data> > &pastKeyValues,
                const std::map <std::string, std::vector <Data*> > &multimodalInput,
                const GenerationConfig &generationConfigs,
                const LastTokensManager &lastTokens,
                std::vector <std::vector <float>*> *logits) {
        bool hasVision = false;                    
        Data x, y, z1, z2, cls_token;
        Data qkv, q, k, v, output, attentionOutput, mlp;
        int startPos, endPos;
        
        if (pastKeyValues[0].second.dims.size() == 0) {
            hasVision = true;
            // 如果是首次推理，那么需要做图像embedding，否则已经存在pastKeyValues里面了
            // 0. PatchEmbedding
            Data &imageInput = *multimodalInput.find("images")->second[0];
            Conv2D(imageInput, this->weight["model.vision.patch_embedding.proj.weight"], this->weight["model.vision.patch_embedding.proj.bias"], this->vision_in_channels, this->vision_hidden_size, this->vision_patch_size, this->vision_patch_size, this->vision_patch_size, this->vision_patch_size, 0, 0, y);
            y.Reshape({y.dims[0], y.dims[1], -1});
            PermuteSelf(y, {0, 2, 1});
            Mul(this->weight["model.vision.patch_embedding.cls_embedding"], 1.0, cls_token);
            cls_token.Reshape({1, cls_token.dims[0], cls_token.dims[1]});
            Cat(cls_token, y, 1, x);
            Mul(this->weight["model.vision.patch_embedding.position_embedding.weight"], 1.0, cls_token);
            cls_token.Reshape({1, cls_token.dims[0], cls_token.dims[1]});
            AddTo(x, cls_token);

            // 1. Vision transformer
            Data qk;

            int visionLayers = atoi(this->weight.dicts["vision_config.num_hidden_layers"].c_str());
            int visionNumHeads = atoi(this->weight.dicts["vision_config.num_heads"].c_str());

            ToDataType(x, this->dataType);
            for (int i = 0; i < visionLayers; i++) {
                std::string pre = "model.vision.transformer.layers." + std::to_string(i);
                int B = x.dims[0], L = x.dims[1];
                Linear(x, 
                    this->weight[pre + ".attention.query_key_value.weight"], 
                    this->weight[pre + ".attention.query_key_value.bias"],
                    qkv); 
                qkv.Reshape({B, L, 3, visionNumHeads, -1});
                PermuteSelf(qkv, {2, 0, 1, 3, 4});            
                Split(qkv, 0, 0, 1, q);
                Split(qkv, 0, 1, 2, k);
                Split(qkv, 0, 2, 3, v);

                q.Reshape({-1, q.dims[3], q.dims[4]});
                k.Reshape({-1, k.dims[3], k.dims[4]});
                v.Reshape({-1, v.dims[3], v.dims[4]});

                PermuteSelf(q, {1, 0, 2});
                PermuteSelf(k, {1, 0, 2});
                PermuteSelf(v, {1, 0, 2});

                if (true) {
                    Attention(q, k, v, attentionMask, qkv, q.dims[0] / k.dims[0], 1.0 / sqrt(this->vision_hidden_size / visionNumHeads), 2);
                } else {
                    MatMulTransB(q, k, qk, 1.0 / sqrt(this->vision_hidden_size / visionNumHeads), 1);
                    Softmax(qk, qk, -1);
                    MatMul(qk, v, qkv, 1.0, 1);
                }

                PermuteSelf(qkv, {1, 0, 2});
                qkv.Reshape({B, L, -1});
                Linear(qkv, 
                    this->weight[pre + ".attention.dense.weight"], 
                    this->weight[pre + ".attention.dense.bias"], 
                    output);
                Mul(output, 1.0, attentionOutput);
                Data empty;
                LayerNorm(output, this->weight[pre + ".input_layernorm.weight"], this->weight[pre + ".input_layernorm.bias"], -1, attentionOutput);
                AddTo(x, attentionOutput);
                Linear(x, 
                    this->weight[pre + ".mlp.fc1.weight"], 
                    this->weight[pre + ".mlp.fc1.bias"], 
                    y);
                Gelu(y, y);
                Linear(y, 
                    this->weight[pre + ".mlp.fc2.weight"], 
                    this->weight[pre + ".mlp.fc2.bias"], 
                    mlp);
                LayerNorm(mlp, this->weight[pre + ".post_attention_layernorm.weight"], this->weight[pre + ".post_attention_layernorm.bias"], -1, mlp);
                AddTo(x, mlp);
            }

            ToDataType(x, DataType::FLOAT32);
            Split(x, 1, 1, x.dims[1], y);
            int gridSize = int(sqrt(y.dims[1]) + 1e-9);
            y.Reshape({y.dims[0], gridSize, gridSize, y.dims[2]});
            PermuteSelf(y, {0, 3, 1, 2});

            Conv2D(y, this->weight["model.vision.conv.weight"], this->weight["model.vision.conv.bias"], this->vision_hidden_size, this->vision_hidden_size, 2, 2, 2, 2, 0, 0, x);
            x.Reshape({x.dims[0], x.dims[1], -1});
            PermuteSelf(x, {0, 2, 1});

            // GLU
            Linear(x, this->weight["model.vision.linear_proj.linear_proj.weight"], this->weight["model.vision.linear_proj.linear_proj.bias"], y);
            LayerNorm(y, this->weight["model.vision.linear_proj.norm1.weight"], this->weight["model.vision.linear_proj.norm1.bias"], -1, y);
            Gelu(y, y);
            Linear(y, this->weight["model.vision.linear_proj.gate_proj.weight"], this->weight["model.vision.linear_proj.gate_proj.bias"], z1);
            Silu(z1, z1);
            Linear(y, this->weight["model.vision.linear_proj.dense_h_to_4h.weight"], this->weight["model.vision.linear_proj.dense_h_to_4h.bias"], z2);
            MulTo(z1, z2);
            Linear(z1, this->weight["model.vision.linear_proj.dense_4h_to_h.weight"], this->weight["model.vision.linear_proj.dense_4h_to_h.bias"], x);

            Cat(this->weight["model.vision.boi"], x, 1, y);
            Cat(y, this->weight["model.vision.eoi"], 1, x);

    #ifdef USE_CUDA
            FastllmCudaClearBigBuffer();
    #endif

            Data textEmbedding;
            Embedding(inputIds, this->weight["model.embed_tokens.weight"], textEmbedding);        

            startPos = 1;
            endPos = x.dims[1]; // [start, endPos)之间是image

            Split(textEmbedding, 1, 0, 1, z1);
            Split(textEmbedding, 1, 1, textEmbedding.dims[1], z2);
            Cat(z1, x, 1, y);
            Cat(y, z2, 1, x);
        } else {
            Embedding(inputIds, this->weight["model.embed_tokens.weight"], x);
            startPos = 1;
            endPos = 1;
        }

        ToDataType(x, this->dataType);
        Data &hiddenStates = x;
        Data attenInput, w1, w2, textW2, visionW2, w3;
        Data* sinDataPtr = &sinData;
        Data* cosDataPtr = &cosData;
        for (int i = 0; i < block_cnt; i++) {
// if (hasVision) printf("%d\n", i);
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".input_layernorm.weight"],rms_norm_eps, attenInput);

            int bsz = attenInput.dims[0], seqlen = attenInput.dims[1];

            Data textInput, visionInput;
            Data textQKV, visionQKV, q, k, v;

            if (hasVision) {
                Linear(attenInput, this->weight["model.layers." + std::to_string(i) + ".self_attn.vision_expert_query_key_value.weight"], this->weight["model.layers." + std::to_string(i) + ".self_attn.vision_expert_query_key_value.bias"], qkv);
                std::vector <int> dims = attenInput.dims;
                dims[1] = startPos;
                Data tempInput, tempOutput;
                tempInput.Resize(dims);
                tempInput.FakeFrom(attenInput, 0);
                tempOutput.FakeFrom(qkv, 0);
                Linear(tempInput, this->weight["model.layers." + std::to_string(i) + ".self_attn.language_expert_query_key_value.weight"], this->weight["model.layers." + std::to_string(i) + ".self_attn.language_expert_query_key_value.bias"], tempOutput);

                dims[1] = attenInput.dims[1] - endPos;
                tempInput.Resize(dims);
                tempInput.FakeFrom(attenInput, endPos * attenInput.Count(2) * attenInput.unitSize);
                tempOutput.FakeFrom(qkv, endPos * qkv.Count(2) * qkv.unitSize);
                Linear(tempInput, this->weight["model.layers." + std::to_string(i) + ".self_attn.language_expert_query_key_value.weight"], this->weight["model.layers." + std::to_string(i) + ".self_attn.language_expert_query_key_value.bias"], tempOutput);
            } else {
                Linear(attenInput, this->weight["model.layers." + std::to_string(i) + ".self_attn.language_expert_query_key_value.weight"], this->weight["model.layers." + std::to_string(i) + ".self_attn.language_expert_query_key_value.bias"], qkv);
            }

            int per = qkv.dims.back() / (num_attention_heads / num_key_value_heads + 2);
            int qdim = per * (num_attention_heads / num_key_value_heads);

            Split(qkv, -1, 0, qdim, q);
            Split(qkv, -1, qdim, qdim + per, k);
            Split(qkv, -1, qdim + per, qdim + per * 2, v);

            std::vector <int> qkvSize = {bsz, seqlen, -1, head_dim};
            q.Reshape(qkvSize);
            k.Reshape(qkvSize);
            v.Reshape(qkvSize);

            Data &pastKey = pastKeyValues[i].first, &pastValue = pastKeyValues[i].second;
            if (GetKVCacheInCPU()) {
                pastKey.lockInCPU = true;
                pastValue.lockInCPU = true;
            } else {
                pastKey.ToDevice(k.dataDevice);
                pastValue.ToDevice(k.dataDevice);
            }
            int targetSeqLength = (pastKey.dims.size() > 2) ? pastKey.dims[1] + seqlen : seqlen;
            if (i == 0 && targetSeqLength >= max_positions && RoPEType::DYMAMIC_NTK == rope_type) {
                float scale = pow((rope_factor * targetSeqLength / max_positions) - (rope_factor - 1), rotary_dim / (rotary_dim - 2));
                float newbase = rope_base * scale;
                std::pair<std::vector<float>, std::vector<float>> &&pair = this->UpdateRotaryPosEmb(newbase, rope_factor, targetSeqLength);
                sinDataPtr = new Data(DataType::FLOAT32, {(int)this->sin.size(), (int)this->sin[0].size()}, pair.first);
                cosDataPtr = new Data(DataType::FLOAT32, {(int)this->cos.size(), (int)this->cos[0].size()}, pair.second);
            }

            fastllm::LlamaRotatePosition2D(q, positionIds, *sinDataPtr, *cosDataPtr, rotary_dim);
            fastllm::LlamaRotatePosition2D(k, positionIds, *sinDataPtr, *cosDataPtr, rotary_dim);

            PermuteSelf(q, {0, 2, 1, 3});
            PermuteSelf(k, {0, 2, 1, 3});
            PermuteSelf(v, {0, 2, 1, 3});

            qkvSize = {-1, seqlen, head_dim};
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

            // 1.2 Attention
            // 1.2.0 q * k^T
            Attention(q, pastKey, pastValue, attentionMask, qkv, q.dims[0] / pastKey.dims[0], 1.0 / sqrt(head_dim), 1);
            PermuteSelf(qkv, {1, 0, 2});
            qkv.Reshape({seqlen, bsz, -1});
            PermuteSelf(qkv, {1, 0, 2});

            if (hasVision) {
                Linear(qkv, this->weight["model.layers." + std::to_string(i) + ".self_attn.vision_expert_dense.weight"], this->weight["model.layers." + std::to_string(i) + ".self_attn.vision_expert_dense.bias"], textQKV);
                std::vector <int> dims = qkv.dims;
                dims[1] = startPos;
                Data tempInput, tempOutput;
                tempInput.Resize(dims);
                tempInput.FakeFrom(qkv, 0);
                tempOutput.FakeFrom(textQKV, 0);
                Linear(tempInput, this->weight["model.layers." + std::to_string(i) + ".self_attn.language_expert_dense.weight"], this->weight["model.layers." + std::to_string(i) + ".self_attn.language_expert_dense.bias"], tempOutput);

                dims[1] = qkv.dims[1] - endPos;
                tempInput.Resize(dims);
                tempInput.FakeFrom(qkv, endPos * qkv.Count(2) * qkv.unitSize);
                tempOutput.FakeFrom(textQKV, endPos * textQKV.Count(2) * textQKV.unitSize);
                Linear(tempInput, this->weight["model.layers." + std::to_string(i) + ".self_attn.language_expert_dense.weight"], this->weight["model.layers." + std::to_string(i) + ".self_attn.language_expert_dense.bias"], tempOutput);
                AddTo(hiddenStates, textQKV);
            } else {
                Linear(qkv, this->weight["model.layers." + std::to_string(i) + ".self_attn.language_expert_dense.weight"], this->weight["model.layers." + std::to_string(i) + ".self_attn.language_expert_dense.bias"], textQKV);
                AddTo(hiddenStates, textQKV);
            }

            // 2. MLP
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"], rms_norm_eps, attenInput);

            if (hasVision) {
                Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.vision_mlp.gate_proj.weight"], Data(), w1);
                Silu(w1, w1);
                Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.vision_mlp.up_proj.weight"], Data(), w3);
                MulTo(w1, w3);            
                Linear(w1, weight["model.layers." + std::to_string(i) + ".mlp.vision_mlp.down_proj.weight"], Data(), w2);
                
                std::vector <int> dims = attenInput.dims;
                dims[1] = startPos;
                Data tempInput, tempOutput;
                tempInput.Resize(dims);
                tempInput.FakeFrom(attenInput, 0);
                tempOutput.FakeFrom(w2, 0);
                Linear(tempInput, weight["model.layers." + std::to_string(i) + ".mlp.language_mlp.gate_proj.weight"], Data(), w1);
                Silu(w1, w1);
                Linear(tempInput, weight["model.layers." + std::to_string(i) + ".mlp.language_mlp.up_proj.weight"], Data(), w3);
                MulTo(w1, w3);            
                Linear(w1, weight["model.layers." + std::to_string(i) + ".mlp.language_mlp.down_proj.weight"], Data(), tempOutput);

                dims[1] = attenInput.dims[1] - endPos;
                tempInput.Resize(dims);
                tempInput.FakeFrom(attenInput, endPos * attenInput.Count(2) * attenInput.unitSize);
                tempOutput.FakeFrom(w2, endPos * w2.Count(2) * w2.unitSize);
                Linear(tempInput, weight["model.layers." + std::to_string(i) + ".mlp.language_mlp.gate_proj.weight"], Data(), w1);
                Silu(w1, w1);
                Linear(tempInput, weight["model.layers." + std::to_string(i) + ".mlp.language_mlp.up_proj.weight"], Data(), w3);
                MulTo(w1, w3);            
                Linear(w1, weight["model.layers." + std::to_string(i) + ".mlp.language_mlp.down_proj.weight"], Data(), tempOutput);
            } else {
                Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.language_mlp.gate_proj.weight"], Data(), w1);
                Silu(w1, w1);
                Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.language_mlp.up_proj.weight"], Data(), w3);
                MulTo(w1, w3);            
                Linear(w1, weight["model.layers." + std::to_string(i) + ".mlp.language_mlp.down_proj.weight"], Data(), w2);
            }

            AddTo(hiddenStates, w2);
        }

        Data tempHiddenStates;
        Data *lastHiddenStates;
        int bs = hiddenStates.dims[1];
        if (bs > 1) {
            Split(hiddenStates, 1, bs - 1, bs, tempHiddenStates);
            lastHiddenStates = &tempHiddenStates;
        } else {
            lastHiddenStates = &hiddenStates;
        }

        Data norm, logit;
        RMSNorm(*lastHiddenStates, this->weight["model.norm.weight"], rms_norm_eps, norm);
        Linear(norm, this->weight["lm_head.weight"], Data(), logit);
        
        ToDataType(logit, DataType::FLOAT32);
        std::vector <int> lastRet;
        Data topk;
        TopK(logit, topk, 1);
        topk.ToDevice(DataDevice::CPU);
        float *topkData = (float*)topk.cpuData;
        for (int b = 0; b < 1; b++) {
            lastRet.push_back((int) (topkData[0] + 1e-3));
            topkData += topk.Count(2);
        }

        return lastRet;
    }

    bool CogvlmModel::NeedAttentionMask(int qlen, int klen) {
        if (this->weight.dicts["use_alibi"] != "1" && 
            ((qlen == 1) || (qlen >= 1024))) {
            return false;
        }
        return true;
    }

    void CogvlmModel::FillLLMInputs(std::vector <std::vector <float> > &inputTokens,
                                   const std::map <std::string, int> &params,
                                   Data &inputIds, Data &attentionMask, Data &positionIds) {
        inputIds.ToDevice(DataDevice::CPU);
        attentionMask.ToDevice(DataDevice::CPU);
        positionIds.ToDevice(DataDevice::CPU);

        int index = params.find("index")->second;
        int promptLen = params.find("promptLen")->second;
        int visionTokens = (this->vision_image_size / this->vision_patch_size / 2) * (this->vision_image_size / this->vision_patch_size / 2) + 2;

        if (inputTokens[0].size() > 1) {
            int seqLen = visionTokens + promptLen;
            std::vector <float> pidValue;
            pidValue.resize(seqLen);
            pidValue[0] = 0;
            pidValue[1] = 1;
            pidValue[2] = 2;
            for (int i = 3; i < visionTokens - 1; i++) {
                pidValue[i] = 2;
            }
            for (int i = visionTokens; i < pidValue.size(); i++) {
                pidValue[i] = i - (visionTokens - 3);
            }
            positionIds.CopyFrom(fastllm::Data(fastllm::DataType::FLOAT32, {1, (int)pidValue.size()}, pidValue));
            
            inputIds.CopyFrom(Data(DataType::FLOAT32, {1, (int)inputTokens[0].size()}, inputTokens[0]));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {1, seqLen}, pidValue));
            attentionMask = Data();
        } else {
            inputIds.CopyFrom(Data(DataType::FLOAT32, {1, 1}, inputTokens[0]));
            attentionMask = Data();
            positionIds.CopyFrom(Data(DataType::FLOAT32, {1, 1}, {(float) promptLen + index + 2}));
        }
    }

    void CogvlmModel::FillLLMInputsBatch(std::vector<std::vector<float>> &inputTokens,
                                          const std::vector<std::map<std::string, int>> &params,
                                          fastllm::Data &inputIds, fastllm::Data &attentionMask,
                                          fastllm::Data &positionIds) {
        inputIds.ToDevice(DataDevice::CPU);
        attentionMask.ToDevice(DataDevice::CPU);
        positionIds.ToDevice(DataDevice::CPU);
        
        int visionTokens = (this->vision_image_size / this->vision_patch_size / 2) * (this->vision_image_size / this->vision_patch_size / 2) + 2;
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

    std::string CogvlmModel::MakeInput(const std::string &history, int round, const std::string &input) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role;
    }

    std::string CogvlmModel::MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role + output + history_sep;
    }

    void CogvlmModel::WarmUp() {
        printf("Warmup...\n");
        std::vector <std::vector <float> > fInputTokens = {{1, 1}};
        Data inputIds;
        Data attentionMask;
        Data positionIds;
        FillLLMInputs(fInputTokens, {{"promptLen", 2}, {"index", 0}, {"add_special_tokens", false}},
                      inputIds, attentionMask, positionIds);

        fastllm::GenerationConfig generationConfig;
        std::map <std::string, std::vector <fastllm::Data*> > multimodalInput;
        std::vector <float> imageInput = std::vector <float> (1 * this->vision_in_channels * this->vision_image_size * this->vision_image_size, 0);
        fastllm::Data imageInputData;
        imageInputData.CopyFrom(fastllm::Data(fastllm::DataType::FLOAT32, {1, this->vision_in_channels, this->vision_image_size, this->vision_image_size}, imageInput));
        multimodalInput["images"].push_back(&imageInputData);

        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(DataType::FLOAT32),
                                                   Data(DataType::FLOAT32)));
        }
        ForwardMultimodal(inputIds, attentionMask, positionIds, pastKeyValues, multimodalInput, generationConfig);
        elementsInKVCachePerToken = (long long)block_cnt * 
            (pastKeyValues[0].first.dims[0] * pastKeyValues[0].first.dims[2] + 
             pastKeyValues[0].second.dims[0] * pastKeyValues[0].second.dims[2]);
        printf("finish.\n");
    }
}
