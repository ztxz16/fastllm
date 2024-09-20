
#ifndef FASTLLM_BERT_H
#define FASTLLM_BERT_H

#include "basellm.h"
#include "fastllm.h"

namespace fastllm {
    class BertModel: public basellm {
    public:
        BertModel() {};

        ~BertModel() {
            this->weight.ReleaseWeight();
        };

        void InitParams(); // 初始化参数信息 

        // 推理
        std::vector <std::vector <float> > ForwardAll(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &tokenTypeIds,
                const Data &positionIds,
                bool normalize);

        // 推理
        virtual int Forward(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector <std::pair <Data, Data> > &pastKeyValues,
                const GenerationConfig &generationConfig = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector <float> *logits = nullptr);
        
        std::vector <float> EmbeddingSentence(const std::string &context, bool normalize);

        std::vector <std::vector <float> > EmbeddingSentenceBatch(const std::vector <std::string> &contexts, bool normalize);

        void LoadFromFile(const std::string &fileName); // 从文件读取 

        void WarmUp(); // 预热

        virtual std::string MakeInput(const std::string &history, int round, const std::string &input);

        virtual std::string MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output);

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