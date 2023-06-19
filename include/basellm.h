#pragma once
#include "fastllm.h"


// typedef void(*RuntimeResult) (int index, const char* content); //实时生成的内容回调 index: 0开始回复，-1本次回复结束
// typedef void(*RuntimeResultBatch) (int index, std::vector <std::string> &contents); //实时生成的内容回调 index: 0开始回复，-1本次回复结束

using RuntimeResult = std::function<void(int index, const char* content)>;
using RuntimeResultBatch = std::function<void(int index, std::vector <std::string> &contents)>;

namespace fastllm {
    class basellm {
    public:
        basellm() {};
        ~basellm() {};

        virtual void LoadFromFile(const std::string &fileName) = 0; // 从文件读取
        // 推理
        virtual int Forward(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                const Data &penaltyFactor,
                std::vector <std::pair <Data, Data> > &pastKeyValues) = 0;

        virtual std::string Response(const std::string& input, RuntimeResult retCb) = 0; // 根据给出的内容回复

        virtual void ResponseBatch(const std::vector <std::string> &inputs,
                                   std::vector <std::string> &outputs,
                                   RuntimeResultBatch retCb = nullptr) {} // 批量根据给出的内容回复

        virtual void SaveLowBitModel(const std::string &fileName, int bit) {}; // 存储成量化模型

        virtual void WarmUp() {}; // 预热

        virtual void RotatePosition2D(Data &data, const Data &positionIds) {}; // 二维位置编码

        virtual void CausalMask(Data &data, int start) {}; // 因果mask

        int output_token_limit = -1;

        int embed_dim = 4096;
        int num_attention_heads = 32;
        int head_dim = embed_dim / num_attention_heads;
        const int max_positions = 2048;
        int rotary_dim = 64;
        const float scale_attn = sqrt(head_dim);

        int block_cnt = 28;

        bool do_sample = false; // 是否进行采样，如不采样则直接取最大值
        int last_n = 64; // 末尾last_n个token计入重复惩罚
        float repeat_penalty = 1.0f; // 重复惩罚系数
        int top_k = 1; // top_k采样
        float top_p = 1.0; // top_p采样
        float temperature = 1.0; // 温度参数，一般在0.1 ~ 1.0之间，设大这个参数可以带来结果的多样性

        std::vector <std::vector <float> > sin, cos;

        WeightMap weight; // 权重

        Data sinData, cosData;
    };
}