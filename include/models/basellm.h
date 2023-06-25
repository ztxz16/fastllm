#pragma once
#include "fastllm.h"

#include <mutex>

#ifdef PY_API
#include "Python.h"
#include <pybind11/pytypes.h>
using RuntimeResult = std::function<void(int index, pybind11::bytes content)>;
#else
using RuntimeResult = std::function<void(int index, const char* content)>;
#endif
using RuntimeResultBatch = std::function<void(int index, std::vector <std::string> &contents)>;

namespace fastllm {
    struct ResponseContext {
        std::vector <std::pair <Data, Data> > pastKeyValues;
        std::vector <int> currentTokens;
        TokenPenaltyManager tokenPenaltyManager;

        int preTokens = 0;
        std::map <std::string, int> intParams;

        void Init(int blocks);
    };

    struct ResponseContextDict {
        std::mutex locker;
        std::map <int, ResponseContext*> dicts;

        int CreateHandle();

        ResponseContext* GetHandle(int handleId);

        void RemoveHandle(int handleId);
    };

    class basellm {
    public:
        basellm() {};

        ~basellm() {};

        virtual void LoadFromFile(const std::string &fileName); // 从文件读取

        // 推理
        virtual int Forward(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                const Data &penaltyFactor,
                std::vector<std::pair<Data, Data> > &pastKeyValues) = 0;

        virtual std::string Response(const std::string &input, RuntimeResult retCb) = 0; // 根据给出的内容回复

        virtual void ResponseBatch(const std::vector<std::string> &inputs,
                                   std::vector<std::string> &outputs,
                                   RuntimeResultBatch retCb = nullptr) {} // 批量根据给出的内容回复

        virtual int LaunchResponseTokens(const std::vector <int> &inputTokens) {return -1; }; // 启动一个response任务，返回分配的handleId

        virtual std::pair <bool, std::vector <int> > FetchResponseTokens(int handelId) {return std::make_pair(false, std::vector <int> ());}; // 获取指定handle的输出, bool代表这个handle是否已经结束（或者不存在）

        virtual void SaveLowBitModel(const std::string &fileName, int bit); // 存储成量化模型

        virtual void WarmUp() {}; // 预热

        virtual std::string MakeInput(const std::string &history, int round, const std::string &input) = 0; // 根据历史信息和当前输入生成prompt

        virtual std::string MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) = 0; // 根据当前回复更新history

        std::string model_type;

        std::string pre_prompt; // 最初对话的提示语
        std::string user_role, bot_role, history_sep; // 用于生成每一轮的prompt

        int bos_token_id;
        int eos_token_id;
        int embed_dim = 4096;
        int num_attention_heads = 32;
        int head_dim = embed_dim / num_attention_heads;
        const int max_positions = 2048;
        int rotary_dim = 64;
        const float scale_attn = sqrt(head_dim);
        int block_cnt = 28;

        bool do_sample = false; // 是否进行采样，如不采样则直接取最大值
        int output_token_limit = -1;
        int last_n = 64; // 末尾last_n个token计入重复惩罚
        float repeat_penalty = 1.0f; // 重复惩罚系数
        int top_k = 1; // top_k采样
        float top_p = 1.0; // top_p采样
        float temperature = 1.0; // 温度参数，一般在0.1 ~ 1.0之间，设大这个参数可以带来结果的多样性

        std::vector<std::vector<float> > sin, cos;

        WeightMap weight; // 权重

        Data sinData, cosData;

        ResponseContextDict responseContextDict;
    };
}
