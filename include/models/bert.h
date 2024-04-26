
#ifndef FASTLLM_BERT_H
#define FASTLLM_BERT_H

#include "basellm.h"
#include "fastllm.h"

namespace fastllm {
    class BertModel {
    public:
        BertModel() {};

        ~BertModel() {
            this->weight.ReleaseWeight();
        };

        void InitParams(); // 初始化参数信息 

        // 推理
        std::vector <std::vector <float> > Forward(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &tokenTypeIds,
                const Data &positionIds);
        
        std::vector <float> EmbeddingSentence(const std::string &context);

        std::vector <std::vector <float> > EmbeddingSentenceBatch(const std::vector <std::string> &contexts);

        void LoadFromFile(const std::string &fileName); // 从文件读取 

        void SaveLowBitModel(const std::string &fileName, int bit); // 存储成量化模型 

        void SaveModel(const std::string &fileName); // 直接导出

        void WarmUp(); // 预热

        std::string model_type;

        float layer_norm_eps = 1e-12;

        int embed_dim = 512;
        int num_attention_heads = 64;
        int head_dim = embed_dim / num_attention_heads;
        int max_positions = 32768;
        int block_cnt = 12;

        WeightMap weight; // 权重
        std::map <std::string, int> deviceMap;
    };
}

#endif //FASTLLM_BERT_H