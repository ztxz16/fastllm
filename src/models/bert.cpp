//
// Created by huangyuyang on 4/25/24.
//

#include "bert.h"
#include "utils.h"
#include <sstream>
#include <cstring>

namespace fastllm {
    void BertModel::LoadFromFile(const std::string &fileName) {
        this->weight.LoadFromFile(fileName);
        InitParams();
    }

    void BertModel::InitParams() {
        if (this->weight.dicts.find("layer_norm_eps") != this->weight.dicts.end()) {
            this->layer_norm_eps = atof(this->weight.dicts["layer_norm_eps"].c_str());
        }
        if (this->weight.dicts.find("num_hidden_layers") != this->weight.dicts.end()) {
            block_cnt = atoi(this->weight.dicts["num_hidden_layers"].c_str());
        } else if (this->weight.dicts.find("num_layers") != this->weight.dicts.end()) {
            block_cnt = atoi(this->weight.dicts["num_layers"].c_str());
        }
        if (this->weight.dicts.find("hidden_size") != this->weight.dicts.end()) {
            embed_dim = atoi(this->weight.dicts["hidden_size"].c_str());
        }
        if (this->weight.dicts.find("num_attention_heads") != this->weight.dicts.end()) {
            num_attention_heads = atoi(this->weight.dicts["num_attention_heads"].c_str());
        }
        this->head_dim = embed_dim / num_attention_heads;
    }

    std::vector <std::vector <float> > BertModel::Forward(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &tokenTypeIds,
                const Data &positionIds) {
        // embedding
        Data inputEmbeddings, tokenTypeEmbeddings, positionIdEmbeddings;
        Embedding(inputIds, this->weight["embeddings.word_embeddings.weight"], inputEmbeddings);
        Embedding(tokenTypeIds, this->weight["embeddings.token_type_embeddings.weight"], tokenTypeEmbeddings);
        Embedding(positionIds, this->weight["embeddings.position_embeddings.weight"], positionIdEmbeddings);
        AddTo(inputEmbeddings, tokenTypeEmbeddings);
        AddTo(inputEmbeddings, positionIdEmbeddings);

        Data hiddenStates, firstStates;
        LayerNorm(inputEmbeddings, this->weight["embeddings.LayerNorm.weight"], this->weight["embeddings.LayerNorm.bias"], -1, hiddenStates);

        Data q, k, v, qk, qkv, attnOutput, inter, pooler;

        for (int i = 0; i < this->block_cnt; i++) {
            std::string queryWeightName = "encoder.layer." + std::to_string(i) + ".attention.self.query.weight";
            std::string queryBiasName = "encoder.layer." + std::to_string(i) + ".attention.self.query.bias";
            std::string keyWeightName = "encoder.layer." + std::to_string(i) + ".attention.self.key.weight";
            std::string keyBiasName = "encoder.layer." + std::to_string(i) + ".attention.self.key.bias";
            std::string valueWeightName = "encoder.layer." + std::to_string(i) + ".attention.self.value.weight";
            std::string valueBiasName = "encoder.layer." + std::to_string(i) + ".attention.self.value.bias";
            std::string attnOutputWeightName = "encoder.layer." + std::to_string(i) + ".attention.output.dense.weight";
            std::string attnOutputbiasName = "encoder.layer." + std::to_string(i) + ".attention.output.dense.bias";
            std::string attnLNWeightName = "encoder.layer." + std::to_string(i) + ".attention.output.LayerNorm.weight";
            std::string attnLNbiasName = "encoder.layer." + std::to_string(i) + ".attention.output.LayerNorm.bias";
            std::string interDenseWeightName = "encoder.layer." + std::to_string(i) + ".intermediate.dense.weight";
            std::string interDenseBiasName = "encoder.layer." + std::to_string(i) + ".intermediate.dense.bias";
            std::string outputWeightName = "encoder.layer." + std::to_string(i) + ".output.dense.weight";
            std::string outputbiasName = "encoder.layer." + std::to_string(i) + ".output.dense.bias";
            std::string outputLNWeightName = "encoder.layer." + std::to_string(i) + ".output.LayerNorm.weight";
            std::string outputLNbiasName = "encoder.layer." + std::to_string(i) + ".output.LayerNorm.bias";

            Linear(hiddenStates, this->weight[queryWeightName], this->weight[queryBiasName], q);
            Linear(hiddenStates, this->weight[keyWeightName], this->weight[keyBiasName], k);
            Linear(hiddenStates, this->weight[valueWeightName], this->weight[valueBiasName], v);

            std::vector <int> qdims = {q.dims[0], q.dims[1], this->num_attention_heads, this->head_dim};
            q.Reshape(qdims);
            k.Reshape(qdims);
            v.Reshape(qdims);
            PermuteSelf(q, {0, 2, 1, 3});
            PermuteSelf(k, {0, 2, 1, 3});
            PermuteSelf(v, {0, 2, 1, 3});
            MatMulTransB(q, k, qk, 1.0 / sqrt(this->head_dim), 1);
            AttentionExtendedMask(qk, attentionMask);

            Softmax(qk, qk, -1);
            MatMul(qk, v, qkv, 1.0, 1);

            PermuteSelf(qkv, {0, 2, 1, 3});
            qkv.Reshape({qkv.dims[0], qkv.dims[1], -1});

            Linear(qkv, this->weight[attnOutputWeightName], this->weight[attnOutputbiasName], attnOutput);
            AddTo(hiddenStates, attnOutput);
            LayerNorm(hiddenStates, this->weight[attnLNWeightName], this->weight[attnLNbiasName], -1, hiddenStates);
            
            if (CanRunLinearEx(LinearExType::ExGelu)) {
                LinearEx(hiddenStates, this->weight[interDenseWeightName], this->weight[interDenseBiasName], inter, LinearExType::ExGelu);
            } else {
                Linear(hiddenStates, this->weight[interDenseWeightName], this->weight[interDenseBiasName], inter);
                Gelu(inter, inter);
            }

            Linear(inter, this->weight[outputWeightName], this->weight[outputbiasName], attnOutput);
            AddTo(hiddenStates, attnOutput);
            LayerNorm(hiddenStates, this->weight[outputLNWeightName], this->weight[outputLNbiasName], -1, hiddenStates);
        }

        Split(hiddenStates, 1, 0, 1, firstStates);
        firstStates.Reshape({firstStates.dims[0], -1});
        Linear(firstStates, this->weight["pooler.dense.weight"], this->weight["pooler.dense.bias"], pooler);
        TanH(pooler, pooler);

        firstStates.ToDevice(DataDevice::CPU);
        float *fret = (float*)firstStates.cpuData;
        int batch = firstStates.dims[0], outputDim = firstStates.dims[1];
        std::vector <std::vector <float> > ret;
        ret.resize(batch, std::vector <float> (outputDim, 0.0f));
        for (int i = 0; i < batch; i++) {
            memcpy(ret[i].data(), fret + i * outputDim, outputDim * sizeof(float));
        }

        return ret;
    }

    std::vector <float> BertModel::EmbeddingSentence(const std::string &context) {
        std::vector <std::string> contexts;
        contexts.push_back(context);
        return EmbeddingSentenceBatch(contexts)[0];
    }

    std::vector <std::vector <float> > BertModel::EmbeddingSentenceBatch(const std::vector <std::string> &contexts) {
        int batch = contexts.size(), len = 0;
        std::vector <std::vector <int> > tokens;
        tokens.resize(batch);
        for (int i = 0; i < batch; i++) {
            Data ids = this->weight.tokenizer.Encode("[CLS]" + contexts[i] + "[SEP]");
            for (int j = 0; j < ids.Count(0); j++) {
                tokens[i].push_back((int)(((float*)ids.cpuData)[j]));
            }
            len = std::max(len, (int)tokens[i].size());
        }

        std::vector <float> ids = std::vector <float> (batch * len, 0.0f);
        std::vector <float> seqLens = std::vector <float> (batch, 0.0f);
        std::vector <float> token_type_ids = std::vector <float> (batch * len, 0.0f);
        std::vector <float> attention_mask = std::vector <float> (batch * len, -1e10f);
        std::vector <float> position_ids = std::vector <float> (batch * len, 0.0f);
        for (int i = 0; i < batch; i++) {
            seqLens[i] = tokens[i].size();
            for (int j = 0; j < tokens[i].size(); j++) {
                ids[i * len + j] = tokens[i][j];
                attention_mask[i * len + j] = 0;
                position_ids[i * len + j] = j;
            }
        }

        fastllm::Data inputIds = fastllm::Data(fastllm::DataType::FLOAT32, {batch, len}, ids);
        fastllm::Data attentionMask = fastllm::Data(fastllm::DataType::FLOAT32, {batch, len}, attention_mask);
        fastllm::Data tokenTypeIds = fastllm::Data(fastllm::DataType::FLOAT32, {batch, len}, token_type_ids);
        fastllm::Data positionIds = fastllm::Data(fastllm::DataType::FLOAT32, {batch, len}, position_ids);

// printf("bs = %d, len = %d\n", batch, len); ClearProfiler(); Forward(inputIds, attentionMask, tokenTypeIds, positionIds); PrintProfiler();
        return Forward(inputIds, attentionMask, tokenTypeIds, positionIds);
    }

    void BertModel::WarmUp() {
        printf("Warmup...\n");
        EmbeddingSentence({"1"});
	    printf("finish.\n");
    }
}