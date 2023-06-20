//
// Created by huangyuyang on 6/15/23.
//

#ifndef FASTLLM_BAICHUAN_H
#define FASTLLM_BAICHUAN_H

#include "basellm.h"
#include "cmath"

#include <iostream>

namespace fastllm {
    class BaichuanModel: public basellm {
    public:
        BaichuanModel (); // 构造函数

        // 推理
        virtual int Forward(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                const Data &penaltyFactor,
                std::vector <std::pair <Data, Data> > &pastKeyValues);

        virtual std::string Response(const std::string& input, RuntimeResult retCb); // 根据给出的内容回复

        virtual void WarmUp(); // 预热

        virtual std::string MakeInput(const std::string &history, int round, const std::string &input); // 根据历史信息和当前输入生成prompt

        virtual std::string MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output); // 根据当前回复更新history
    private:
        virtual void RotatePosition2D(Data &data, const Data &positionIds); // 二维位置编码

        virtual void CausalMask(Data &data, int start) {}; // 因果mask？
    };
}

#endif //FASTLLM_BAICHUAN_H
