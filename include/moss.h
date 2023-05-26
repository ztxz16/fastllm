//
// Created by huangyuyang on 5/12/23.
//

#ifndef TEST_MOSS_H
#define TEST_MOSS_H

#include "basellm.h"
#include "cmath"

namespace fastllm {
    class MOSSModel: public basellm {
	public:
        MOSSModel(); // 构造函数

		virtual void LoadFromFile(const std::string &fileName); // 从文件读取

        // 推理
		virtual int Forward(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector <std::pair <Data, Data> > &pastKeyValues);

		virtual std::string Response(const std::string &input, RuntimeResult retCb); // 根据给出的内容回复

		virtual void SaveLowBitModel(const std::string &fileName, int bit); // 存储成量化模型
    private:
		virtual void RotatePosition2D(Data &data, const Data &positionIds); // 二维位置编码

		virtual void CausalMask(Data &data, int start); // 因果mask？
    };
}

#endif //TEST_MOSS_H
