//
// Created by huangyuyang on 9/20/24.
//

#include "xlmroberta.h"
#include "utils.h"
#include <sstream>
#include <cstring>

namespace fastllm {
    XlmRobertaModel::XlmRobertaModel() {
        this->model_struct = "xlmroberta";
        this->model_type = "xlmroberta";

        weight.embeddingNames.insert("roberta.embeddings.word_embeddings.weight");
        weight.embeddingNames.insert("roberta.embeddings.position_embeddings.weight");
        weight.embeddingNames.insert("roberta.embeddings.token_type_embeddings.weight");
        weight.linearNames = {
            "classifier.out_proj.weight", "classifier.dense.weight",
            "*.attention.self.query.weight", "*.attention.self.key.weight", "*.attention.self.value.weight",
            "*.attention.output.dense.weight", "*.output.dense.weight", "*.intermediate.dense.weight"
        };
    }

    void XlmRobertaModel::InitParams() {
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

    void XlmRobertaModel::FillBertInputsBatch(const std::vector <std::vector <int> > &tokens,
                                Data &inputIds, Data &attentionMask, Data &tokenTypeIds, Data &positionIds) {
        int batch = tokens.size(), len = 0;
        for (int i = 0; i < batch; i++) {
            len = std::max(len, (int)tokens[i].size());
        }

        std::vector <float> ids = std::vector <float> (batch * len, 0.0f);
        std::vector <float> seqLens = std::vector <float> (batch, 0.0f);
        std::vector <float> token_type_ids = std::vector <float> (batch * len, 0.0f);
        std::vector <float> attention_mask = std::vector <float> (batch * len * len, 1);
        std::vector <float> position_ids = std::vector <float> (batch * len, 0.0f);
        for (int i = 0; i < batch; i++) {
            seqLens[i] = tokens[i].size();
            for (int j = 0; j < tokens[i].size(); j++) {
                ids[i * len + j] = tokens[i][j];
                position_ids[i * len + j] = 2 + j;
                std::fill(&attention_mask[i * len * len + j * len], &attention_mask[i * len * len + j * len + tokens[i].size()], 0.0f);
            }            
        }
        inputIds.CopyFrom(fastllm::Data(fastllm::DataType::FLOAT32, {batch, len}, ids));
        attentionMask.CopyFrom(fastllm::Data(fastllm::DataType::FLOAT32, {batch, len, len}, attention_mask));
        tokenTypeIds.CopyFrom(fastllm::Data(fastllm::DataType::FLOAT32, {batch, len}, token_type_ids));
        positionIds.CopyFrom(fastllm::Data(fastllm::DataType::FLOAT32, {batch, len}, position_ids));
    }

    std::vector <std::vector <float> > XlmRobertaModel::ForwardAll(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &tokenTypeIds,
                const Data &positionIds,
                bool normalize) {
        Data inputEmbeddings, tokenTypeEmbeddings, positionIdEmbeddings;
        Embedding(inputIds, this->weight["roberta.embeddings.word_embeddings.weight"], inputEmbeddings);
        Embedding(tokenTypeIds, this->weight["roberta.embeddings.token_type_embeddings.weight"], tokenTypeEmbeddings);
        Embedding(positionIds, this->weight["roberta.embeddings.position_embeddings.weight"], positionIdEmbeddings);
        AddTo(inputEmbeddings, tokenTypeEmbeddings);
        AddTo(inputEmbeddings, positionIdEmbeddings);
        Data hiddenStates, firstStates;
        LayerNorm(inputEmbeddings, this->weight["roberta.embeddings.LayerNorm.weight"], this->weight["roberta.embeddings.LayerNorm.bias"], -1, hiddenStates);
        int bsz = hiddenStates.dims[0], seqlen = hiddenStates.dims[1];
        Data q, k, v, qk, qkv, attnOutput, inter, pooler, logits;
        for (int i = 0; i < this->block_cnt; i++) {
            std::string queryWeightName = "roberta.encoder.layer." + std::to_string(i) + ".attention.self.query.weight";
            std::string queryBiasName = "roberta.encoder.layer." + std::to_string(i) + ".attention.self.query.bias";
            std::string keyWeightName = "roberta.encoder.layer." + std::to_string(i) + ".attention.self.key.weight";
            std::string keyBiasName = "roberta.encoder.layer." + std::to_string(i) + ".attention.self.key.bias";
            std::string valueWeightName = "roberta.encoder.layer." + std::to_string(i) + ".attention.self.value.weight";
            std::string valueBiasName = "roberta.encoder.layer." + std::to_string(i) + ".attention.self.value.bias";
            std::string attnOutputWeightName = "roberta.encoder.layer." + std::to_string(i) + ".attention.output.dense.weight";
            std::string attnOutputbiasName = "roberta.encoder.layer." + std::to_string(i) + ".attention.output.dense.bias";
            std::string attnLNWeightName = "roberta.encoder.layer." + std::to_string(i) + ".attention.output.LayerNorm.weight";
            std::string attnLNbiasName = "roberta.encoder.layer." + std::to_string(i) + ".attention.output.LayerNorm.bias";
            std::string interDenseWeightName = "roberta.encoder.layer." + std::to_string(i) + ".intermediate.dense.weight";
            std::string interDenseBiasName = "roberta.encoder.layer." + std::to_string(i) + ".intermediate.dense.bias";
            std::string outputWeightName = "roberta.encoder.layer." + std::to_string(i) + ".output.dense.weight";
            std::string outputbiasName = "roberta.encoder.layer." + std::to_string(i) + ".output.dense.bias";
            std::string outputLNWeightName = "roberta.encoder.layer." + std::to_string(i) + ".output.LayerNorm.weight";
            std::string outputLNbiasName = "roberta.encoder.layer." + std::to_string(i) + ".output.LayerNorm.bias";

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

            if (true) {
                q.Reshape({-1, q.dims[2], q.dims[3]});
                k.Reshape({-1, k.dims[2], k.dims[3]});
                v.Reshape({-1, v.dims[2], v.dims[3]});
                Attention(q, k, v, attentionMask, qkv, q.dims[0] / k.dims[0], 1.0 / sqrt(this->head_dim), 1);
                PermuteSelf(qkv, {1, 0, 2});
                qkv.Reshape({seqlen, bsz, -1});
                PermuteSelf(qkv, {1, 0, 2});
            } else {
                MatMulTransB(q, k, qk, 1.0 / sqrt(this->head_dim), 1);
                std::vector <int> dims = qk.dims;
                qk.Reshape({dims[0], -1, dims[3]});
                AttentionMask(qk, attentionMask, -1e9);
                qk.Reshape(dims);
                Softmax(qk, qk, -1);
                MatMul(qk, v, qkv, 1.0, 1);
                PermuteSelf(qkv, {0, 2, 1, 3});
                qkv.Reshape({qkv.dims[0], qkv.dims[1], -1});
            }
            
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
        if (this->weight.weight.find("classifier.dense.weight") != this->weight.weight.end()) {
            Linear(firstStates, this->weight["classifier.dense.weight"], this->weight["classifier.dense.bias"], pooler);
            TanH(pooler, pooler);
            Linear(pooler, this->weight["classifier.out_proj.weight"], this->weight["classifier.out_proj.bias"], logits);
        } else {
            Mul(firstStates, 1.0f, logits);
        }

        logits.ToDevice(DataDevice::CPU);
        float *fret = (float*)logits.cpuData;
        int batch = logits.dims[0], outputDim = logits.dims[1];
        std::vector <std::vector <float> > ret;
        ret.resize(batch, std::vector <float> (outputDim, 0.0f));
        for (int i = 0; i < batch; i++) {
            if (normalize) {
                Normalize(fret + i * outputDim, outputDim);
            }
            memcpy(ret[i].data(), fret + i * outputDim, outputDim * sizeof(float));
        }
        return ret;
    }
}