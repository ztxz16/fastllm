
#ifndef FASTLLM_XLMROBERTA_H
#define FASTLLM_XLMROBERTA_H

#include "basellm.h"
#include "bert.h"
#include "fastllm.h"

namespace fastllm {
    class XlmRobertaModel : BertModel {
    public:
        XlmRobertaModel();

        ~XlmRobertaModel() {
            this->weight.ReleaseWeight();
        };

        void InitParams(); // 初始化参数信息 

        void FillBertInputsBatch(const std::vector <std::vector <int> > &tokens,
                                Data &inputIds, Data &attentionMask, Data &tokenTypeIds, Data &positionIds);

        // 推理
        std::vector <std::vector <float> > ForwardAll(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &tokenTypeIds,
                const Data &positionIds,
                bool normalize);

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