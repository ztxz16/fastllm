
#ifndef FASTLLM_XLMROBERTA_H
#define FASTLLM_XLMROBERTA_H

#include "basellm.h"
#include "fastllm.h"

namespace fastllm {
    class XlmRobertaModel : basellm {
    public:
        XlmRobertaModel();

        ~XlmRobertaModel() {
            this->weight.ReleaseWeight();
        };

        void InitParams(); // 初始化参数信息 

        // 推理
        int Forward(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector <std::pair <Data, Data> > &pastKeyValues,
                const GenerationConfig &generationConfig = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector <float> *logits = nullptr) {return 0;}

        std::string MakeInput(const std::string &history, int round, const std::string &input) {return "";}
        std::string MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) {return "";}

        // 计算相似分数
        // tokens: 输入tokens， tokens[i]代表第i个输入的token序列
        // ret: ret[i]代表第i个输入的相似度
        std::vector <float> ComputeScore(std::vector <std::vector <int> > tokens);

        // 推理
        std::vector <float> Forward(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &tokenTypeIds,
                const Data &positionIds);
        
        std::vector <float> EmbeddingSentence(const std::string &context);

        std::vector <std::vector <float> > EmbeddingSentenceBatch(const std::vector <std::string> &contexts);

        void LoadFromFile(const std::string &fileName); // 从文件读取 

        void WarmUp(); // 预热

        std::string model_type;

        float layer_norm_eps = 1e-12;

        int embed_dim = 512;
        int num_attention_heads = 64;
        int head_dim = embed_dim / num_attention_heads;
        int max_positions = 32768;
        int block_cnt = 12;

        std::map <std::string, int> deviceMap;
    };
}

#endif //FASTLLM_XLMROBERTA_H