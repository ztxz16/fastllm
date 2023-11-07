//
// Created by huangyuyang on 5/11/23.
//

#include "utils.h"

#include "glm.h"

#include <cmath>

#include <chrono>

#include <algorithm>

#include <map>

#include <sstream>

#include <unordered_map>

#include <cstring>

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif

namespace fastllm {

    GLMModel::GLMModel() {
        this->model_type = "glm";

        this->bos_token_id = 50006;//<|startofpiece|>
        this->eos_token_id = 50007;//<|endofpiece|>

        weight.embeddingNames.insert("word_embeddings.weight");
        weight.embeddingNames.insert("transformer.position_embeddings.weight");
        weight.embeddingNames.insert("transformer.block_position_embeddings.weight");
        weight.tokenizer.type=Tokenizer::GLM;
        weight.tokenizer.Insert("[MASK]",mask_token_id);
        weight.tokenizer.Insert("[sMASK]",smask_token_id);
        weight.tokenizer.Insert("[gMASK]",gmask_token_id);
    }

    int GLMModel::Forward(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                              const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                              const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                              std::vector <float> *logits) {
        std::vector <std::vector <float>*> batchLogits;
        batchLogits.push_back(logits);
        return ForwardBatch(1, inputIds, attentionMask, positionIds, pastKeyValues, generationConfig, lastTokens, &batchLogits)[0];
    }

    std::vector <int> GLMModel::ForwardBatch(
            int batch,
            const Data &inputIds,
            const Data &attentionMask,
            const Data &positionIds,
            std::vector <std::pair <Data, Data> > &pastKeyValues,
            const GenerationConfig &generationConfig,
            const LastTokensManager &lastTokens,
            std::vector <std::vector <float>*> *retLogits) {
        int maxLen = inputIds.dims[1];
        Data attentionMask4D;
        Data attnScoreAdds;
        Data inputEmbeddings;
        Data position_ids_1D;
        Data block_position_ids_1D;
        Data positionEmbeddings;
        Data blockPositionEmbeddings;
        Data attenInput;
        Data qkv, q, k, v,q0;
        Data attnScores;
        Data attnProbs;
        Data attnOutput;
        Data contextLayer;
        Data contextLayerPermute;
        Data mlpInput;
        Data mlpOutput;
        Data middle, middle2;
        Data toSave;
        Data mem2,mem3;
        std::vector<int> lastRet;
        // GLMBlock
        std::string weightPre, weightMiddle;
        weightPre = "transformer.layers.";
        weightMiddle = ".attention";

        {
            Data attentionMask4D_1x;
            attentionMask4D_1x.CopyFrom(attentionMask);
            attentionMask4D_1x.Reshape({1,1,attentionMask.dims[0],attentionMask.dims[1]});
            std::vector<Data*> masks(num_attention_heads);
            for(int i=0;i<num_attention_heads;i++){
                masks[i]=&attentionMask4D_1x;
            }
            CatBatch(masks,1,attentionMask4D);
            std::vector<float> one(attentionMask4D.Count(0),-65504.0);
            attnScoreAdds.CopyFrom(Data(DataType::FLOAT32,attentionMask4D.dims,one));
            AddTo(attnScoreAdds,attentionMask4D,65504.0);
        }
        Embedding(inputIds, this->weight["word_embeddings.weight"], inputEmbeddings);
        Data &hiddenStates = inputEmbeddings;
        Split(positionIds,0,0,1,position_ids_1D);
        Split(positionIds,0,1,2,block_position_ids_1D);
        Embedding(position_ids_1D, this->weight["transformer.position_embeddings.weight"], positionEmbeddings);
        AddTo(hiddenStates,positionEmbeddings);
        Embedding(block_position_ids_1D, this->weight["transformer.block_position_embeddings.weight"], blockPositionEmbeddings);
        AddTo(hiddenStates,blockPositionEmbeddings);
        int memory_length=(pastKeyValues[0].first.dims.size()==0?0:pastKeyValues[0].first.dims.at(1));
        int query_length=hiddenStates.dims.at(1);
        int new_memory_length=memory_length+query_length;
        if(new_memory_length<=query_length){
            Split(hiddenStates,1,hiddenStates.dims.at(1)-new_memory_length,hiddenStates.dims.at(1),toSave);
        }else{
            Split(hiddenStates,1,0,hiddenStates.dims.at(1),toSave);//Copy
        }
        for (int i = 0; i < block_cnt; i++) {
            Data &mem=pastKeyValues[i].first;
            bool hasMem=(mem.dims.size()!=0);
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            std::string inputLNWeightName = "transformer.layers." + std::to_string(i) + ".input_layernorm.weight";
            std::string inputLNBiasName = "transformer.layers." + std::to_string(i) + ".input_layernorm.bias";
            LayerNorm(hiddenStates, weight[inputLNWeightName], weight[inputLNBiasName], -1, attenInput);
            std::string qkvWeightName = weightPre + std::to_string(i) + weightMiddle + ".query_key_value.weight";
            std::string qkvBiasName = weightPre + std::to_string(i) + weightMiddle + ".query_key_value.bias";
            if(!hasMem){
                Linear(attenInput, weight[qkvWeightName], weight[qkvBiasName], qkv);
                int per = qkv.dims.back() / 3;
                Split(qkv, -1, 0, per, q);
                Split(qkv, -1, per, per * 2, k);
                Split(qkv, -1, per * 2, per * 3, v);
            }else{
                LayerNorm(mem, weight[inputLNWeightName], weight[inputLNBiasName], -1, mem2);
                Cat(mem2,attenInput,1,mem3);
                Linear(mem3, weight[qkvWeightName], weight[qkvBiasName], qkv);
                int per = qkv.dims.back() / 3;
                Split(qkv, -1, 0, per, q0);
                Split(qkv, -1, per, per * 2, k);
                Split(qkv, -1, per * 2, per * 3, v);
                int tLen=q0.dims.at(1);
                Split(q0,1,tLen-attenInput.dims.at(1),tLen,q);
            }
            q.Reshape({q.dims[0], q.dims[1], num_attention_heads, -1});
            PermuteSelf(q,{0,2,1,3});
            k.Reshape({k.dims[0], k.dims[1], num_attention_heads, -1});
            //PermuteSelf(k,{0,2,1,3});// (1)
            v.Reshape({v.dims[0], v.dims[1], num_attention_heads, -1});
            PermuteSelf(v,{0,2,1,3});
            //PermuteSelf(k,{0,1,2,3});// (2)
            PermuteSelf(k,{0,2,3,1});// Merged (1) + (2)
            MatMul(q,k,attnScores,scale_attn_1);
            MulTo(attnScores,attentionMask4D);
            AddTo(attnScores,attnScoreAdds);
            Softmax(attnScores, attnProbs, -1);
            MatMul(attnProbs,v,contextLayer);
            PermuteSelf(contextLayer,{0,2,1,3});
            contextLayer.Reshape({contextLayer.dims[0],contextLayer.dims[1],embed_dim});
            std::string denseWeightName = weightPre + std::to_string(i) + weightMiddle + ".dense.weight";
            std::string denseBiasName = weightPre + std::to_string(i) + weightMiddle + ".dense.bias";
            Linear(contextLayer, weight[denseWeightName], weight[denseBiasName], attnOutput);
            AddTo(hiddenStates,attnOutput);
            std::string postLNWeightName =
                    "transformer.layers." + std::to_string(i) + ".post_attention_layernorm.weight";
            std::string postLNBiasName =
                    "transformer.layers." + std::to_string(i) + ".post_attention_layernorm.bias";
            LayerNorm(hiddenStates, weight[postLNWeightName], weight[postLNBiasName], -1, mlpInput);
            std::string fcInKeyName = "transformer.layers." + std::to_string(i) + ".mlp.dense_h_to_4h";
            std::string fcOutKeyName = "transformer.layers." + std::to_string(i) + ".mlp.dense_4h_to_h";
            Linear(mlpInput, weight[fcInKeyName + ".weight"], weight[fcInKeyName + ".bias"], middle);
            GeluNew(middle, middle);
            Linear(middle, weight[fcOutKeyName + ".weight"], weight[fcOutKeyName + ".bias"], mlpOutput);
            AddTo(hiddenStates,mlpOutput);
            if(new_memory_length<=query_length){
                Split(toSave,1,0,toSave.dims.at(1),mem);//Copy
                Split(hiddenStates,1,hiddenStates.dims.at(1)-new_memory_length,hiddenStates.dims.at(1),toSave);
            }else{
                Split(mem,1,mem.dims.at(1)-new_memory_length+query_length,mem.dims.at(1),mem2);
                Cat(mem2,toSave,1,mem);
                Split(hiddenStates,1,0,hiddenStates.dims.at(1),toSave);//Copy
            }
        }
        Data logits, topk;
        LayerNorm(hiddenStates, weight["transformer.final_layernorm.weight"],
                    weight["transformer.final_layernorm.bias"], -1, hiddenStates);
        Linear(hiddenStates, weight["word_embeddings.weight"], Data(), logits);
        if (generationConfig.output_logits && retLogits != nullptr) {
            int size = logits.dims.back();
            logits.ToDevice(DataDevice::CPU);
            for (int b = 0; b < batch; b++) {
                int base = (maxLen - 1) * batch + b;
                (*retLogits)[b]->resize(size);
                memcpy((float*)(*retLogits)[b]->data(), ((float*)logits.cpuData) + base * size, size * logits.unitSize);
            }
        }
        if (generationConfig.IsSimpleGreedy()) {
            TopK(logits, topk, 1);
            topk.ToDevice(DataDevice::CPU);
            for (int b = 0; b < batch; b++) {
                int base = (maxLen - 1) * batch + b;
                lastRet.push_back((int) (((float *) topk.cpuData)[base * 2] + 1e-3));
            }
        } else if (!lastTokens.units.empty()) {
            for (int b = 0; b < batch; b++) {
                int base = (maxLen - 1) * batch + b;
                lastRet.push_back(LLMSampling(logits, base, generationConfig, lastTokens.units[b]));
            }
        }
        return lastRet;
    }

    void GLMModel::FillLLMInputs(std::vector <std::vector <float> > &inputTokens,
                                     const std::map <std::string, int> &params,
                                     Data &inputIds, Data &attentionMask, Data &positionIds) {
        inputIds.ToDevice(DataDevice::CPU);
        attentionMask.ToDevice(DataDevice::CPU);
        positionIds.ToDevice(DataDevice::CPU);

        int index = params.find("index")->second;

        if (index == 0) {
            int mask_pos=-1;
            for (auto &ids: inputTokens) {
                bool hasMask=false;
                for(unsigned int i=0;i<ids.size();i++){
                    const float &id=ids.at(i);
                    if(id==mask_token_id||id==smask_token_id||id==gmask_token_id){
                        hasMask=true;
                        if(mask_pos<0){
                            mask_pos=i+1;
                        }
                        break;
                    }
                }
                ids.insert(ids.begin(),cls_token_id);
                if(!hasMask){
                    if(mask_pos<0){
                        mask_pos=ids.size();
                    }
                    ids.push_back(gmask_token_id);
                }
                ids.push_back(eot_token_id);
                ids.push_back(bos_token_id);
            }

            int seqLen = inputTokens[0].size();
            std::vector<float> vpids=std::vector<float>(seqLen*2,0);//position_ids
            for(int i=0;i<seqLen-1;i++){
                vpids[i]=i;
            }
            for(int i=0;i<seqLen-(seqLen-1);i++){
                vpids[seqLen-1+i]=mask_pos;
                vpids[seqLen+seqLen-1+i]=(i+1);
            }
            vpids[seqLen-1]=mask_pos;
            vpids[seqLen+seqLen-1]=1;
            std::vector<float> vmask=std::vector<float>(seqLen*seqLen,1);//attention_mask
            for(int i=0;i<seqLen-1;i++){
                for(int j=std::max(i+1,seqLen-1);j<seqLen;j++){
                    vmask[seqLen*i+j]=0;
                }
            }
            inputIds.CopyFrom(Data(DataType::FLOAT32, {1, seqLen}, inputTokens[0]));
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {seqLen, seqLen}, vmask));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {2, seqLen}, vpids));
        } else {
            const auto &inputToken=inputTokens[0];
            unsigned long tokenLen=inputToken.size();
            int oldLen=attentionMask.dims.at(1);
            int totalLen=oldLen+tokenLen;
            float *positionDat=reinterpret_cast<float*>(positionIds.cpuData);
            int posLen=positionIds.dims.at(1);
            std::vector<float> newAttention(totalLen,1);
            std::vector<float> newPosition(tokenLen*2);
            for(unsigned int i=0;i<tokenLen;i++){
                newPosition[i]=positionDat[posLen-1];
                newPosition[tokenLen+i]=positionDat[posLen+posLen-1]+1;
            }
            inputIds.CopyFrom(Data(DataType::FLOAT32, {1, static_cast<int>(tokenLen)}, inputTokens[0]));
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {1, totalLen}, newAttention));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {2, static_cast<int>(tokenLen)}, newPosition));
        }
    }

    void GLMModel::InitParams()
    {
        basellm::InitParams();
        head_dim = embed_dim / num_attention_heads;
        scale_attn_1 = 1.0f/sqrt(head_dim);
#ifdef USE_SENTENCEPIECE
        if (this->weight.dicts.find("tokenizer_serialized") != this->weight.dicts.end()) {
            const std::string &hexString=this->weight.dicts["tokenizer_serialized"];
            if(hexString.length()%2!=0){
                std::cerr << "Invalid hex string\n";
            }else{
                std::string decoded;
                for(unsigned int i=0;i<hexString.length();i+=2){
                    decoded.push_back(std::stoi(hexString.substr(i,2),nullptr,16));
                }
                weight.tokenizer.spProcessor=std::make_unique<sentencepiece::SentencePieceProcessor>();
                weight.tokenizer.spProcessor->LoadFromSerializedProto(decoded);
            }
        }
#endif
    }

    void GLMModel::WarmUp() {
//      printf("Warmup...\n");
//	    Data inputIds = Data(DataType::FLOAT32, {1, 1}, {(float)bos_token_id});
//	    Data attentionMask = Data(DataType::FLOAT32, {1, 1}, {0});
//	    Data positionIds = Data(DataType::FLOAT32, {2, 1}, {0, 0});

//	    std::vector <std::pair <Data, Data> > pastKeyValues;
//	    for (int i = 0; i < block_cnt; i++) {
//		    pastKeyValues.push_back(std::make_pair(Data(DataType::FLOAT32),
//		                                           Data(DataType::FLOAT32)));
//	    }
//	    Forward(inputIds, attentionMask, positionIds, pastKeyValues);
//	    printf("finish.\n");
    }

    std::string GLMModel::MakeInput(const std::string &history, int round, const std::string &input) {
        (void)history;
        (void)round;
        return input;
    }

    std::string GLMModel::MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) {
        (void)history;
        (void)round;
        (void)input;
        (void)output;
        return std::string("");
    }
}
