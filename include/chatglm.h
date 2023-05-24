//
// Created by huangyuyang on 5/11/23.
//

#ifndef FASTLLM_CHATGLM_H
#define FASTLLM_CHATGLM_H

#include "fastllm.h"
#include "cmath"

#include <iostream>

namespace fastllm {
    struct ChatGLMModel {
        const int embed_dim = 4096;
        const int num_attention_heads = 32;
        const int head_dim = embed_dim / num_attention_heads;
        const int max_positions = 2048;
        const int rotary_dim = 64;
        const float scale_attn = sqrt(head_dim);

        const int block_cnt = 28;

        std::vector <std::vector <float> > sin, cos;

        WeightMap weight; // 权重

        ChatGLMModel (); // 构造函数

        void LoadFromFile(const std::string &fileName); // 从文件读取

        // 推理
        int Forward(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector <std::pair <Data, Data> > &pastKeyValues);

        std::string Response(const std::string& input, std::ostream* ost = nullptr, bool cli = true); // 根据给出的内容回复

        void SaveLowBitModel(const std::string &fileName, int bit); // 存储成量化模型
    private:
        void RotatePosition2D(Data &data, const Data &positionIds); // 二维位置编码

        void CausalMask(Data &data, int start); // 因果mask？
    };
}

#endif //FASTLLM_CHATGLM_H
