
#ifndef FASTLLM_BASELLM_H
#define FASTLLM_BASELLM_H

#include "fastllm.h"
#include "template.h"

#include <thread>
#include <mutex>
#include <condition_variable>

#ifdef PY_API
#include "Python.h"
#include <pybind11/pytypes.h>
using RuntimeResult = std::function<void(int index, pybind11::bytes content)>;
using RuntimeResultBatch = std::function<void(int index, std::vector <pybind11::bytes> &contents)>;
#else
using RuntimeResult = std::function<void(int index, const char* content)>;
using RuntimeResultBatch = std::function<void(int index, std::vector <std::string> &contents)>;
#endif

namespace fastllm {
    using ChatMessages = std::vector <std::pair <std::string, std::string> >;

    struct ResponseContext {
        bool isEnding = false;
        std::vector <std::pair <Data, Data> > pastKeyValues;
        std::vector <int> currentTokens;
        std::queue <int> resultTokenQueue;
        std::queue <std::vector <float>*> resultLogits;
        GenerationConfig generationConfig;
        LastTokensUnit tokens;

        int preTokens = 0;
        int curTokens = 0;
        std::map <std::string, int> intParams;

        int cacheLen = 0;

        void Init(int blocks, DataType dataType);
    };

    struct ResponseContextDict {
        std::mutex locker;
        std::map <int, ResponseContext*> dicts;

        int CreateHandle();

        ResponseContext* GetHandle(int handleId);

        void RemoveHandle(int handleId);
    };

    struct PastKVCacheMemory {
        std::vector <int> inputToken;
        int tokens;
        int recordTimes = 0;
        long long flushTime;
        std::vector<std::pair<Data, Data> > kv;

        PastKVCacheMemory () {}

        PastKVCacheMemory (const std::vector <int> &prompt, int tokens, long long flushTime, std::vector<std::pair<Data, Data> > *kv);
    };

    struct PastKVCacheManager {
        std::mutex locker;
        int maxRecordNum = 5;
        long long flushTime = 0;
        std::map <std::vector <int>, PastKVCacheMemory*> memorys;

        // 设置最多保存的记录条数
        void SetMaxRecordNum(int maxRecordNum);

        // 插入一条记录，若已存在则增加引用计数
        void Record(const std::vector <int> &inputToken, int tokens, std::vector<std::pair<Data, Data> > *kv);

        // 尝试删除一条记录，若引用计数非0不会真的删除
        void Remove(const std::vector <int> &inputToken);

        // 获取最长匹配的Memory，并加锁
        PastKVCacheMemory *Get(const std::vector <int> &inputToken);

        // 解锁
        void Unlock();
    };

    enum RoPEType { // 位置编码外推类型
        BASE = 0,
        LINEAR_SCALE = 1,
        STATIC_NTK = 2,
        DYMAMIC_NTK = 3,
        YARN = 4
    };

    class basellm {
    public:
        basellm() {};

        ~basellm();

        virtual void LoadFromFile(const std::string &fileName); // 从文件读取 

        virtual void InitParams(); // 初始化参数信息 

        // 根据原始的tensorNames获得映射表
        virtual std::map <std::string, std::vector <std::pair <std::string, DataType> > >
                GetTensorMap(const std::vector <std::string> &tensorNames);

        // 推理
        virtual int Forward(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector<std::pair<Data, Data> > &pastKeyValues,
                const GenerationConfig &generationConfig = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector <float> *logits = nullptr) = 0;

        virtual std::vector <int> ForwardBatch(
                int batch,
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector <std::pair <Data, Data> > &pastKeyValues,
                const GenerationConfig &generationConfig = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector <std::vector <float>*> *logits = nullptr);

        virtual std::vector <int> ForwardBatch(
                int batch,
                const Data &inputIds,
                const std::vector <Data*> &attentionMask,
                const std::vector <Data*> &positionIds,
                const std::vector <int> &seqLens,
                std::vector <std::pair <Data*, Data*> > &pastKeyValues,
                const std::vector <GenerationConfig> &generationConfigs,
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector <std::vector <float>*> *logits = nullptr);

        // 是否需要生成AttentionMask
        virtual bool NeedAttentionMask(int qlen, int klen);

        // 根据输入的tokens生成LLM推理的输入 
        virtual void FillLLMInputs(std::vector <std::vector <float> > &inputTokens,
                                   const std::map <std::string, int> &params,
                                   Data &inputIds, Data &attentionMask, Data &positionIds);

        // 根据输入的tokens生成LLM推理的输入 
        virtual void FillLLMInputsBatch(std::vector <std::vector <float> > &inputTokens,
                                        const std::vector <std::map <std::string, int> > &params,
                                        Data &inputIds, Data &attentionMask, Data &positionIds);

        virtual std::string Response(const std::string &input,
                                     RuntimeResult retCb,
                                     const GenerationConfig &generationConfig = GenerationConfig());

        virtual void ResponseBatch(const std::vector<std::string> &inputs,
                                   std::vector<std::string> &outputs,
                                   RuntimeResultBatch retCb = nullptr,
                                   const GenerationConfig &generationConfig = GenerationConfig()); // 批量根据给出的内容回复 

        virtual int LaunchResponseTokens(const std::vector <int> &inputTokens,
                                         const GenerationConfig &generationConfig = GenerationConfig()); // 启动一个response任务，返回分配的handleId
        
        virtual bool CanFetchResponse(int handleId); // 判断当前是否能fetch到，用于异步操作

        virtual int FetchResponseTokens(int handleId); // 获取指定handle的输出, -1代表输出结束了 

        virtual int FetchResponseLogits(int handleId, std::vector <float> &logits); // 获取指定handle的输出Logits

        virtual void SaveLowBitModel(const std::string &fileName, int bit); // 存储成量化模型 

        virtual void SaveModel(const std::string &fileName); // 直接导出

        virtual void WarmUp() {}; // 预热

        virtual void AddPromptCache(const std::vector <int> &inputTokens);

        virtual std::string MakeInput(const std::string &history, int round, const std::string &input) = 0; // 根据历史信息和当前输入生成prompt

        virtual std::string MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) = 0; // 根据当前回复更新history

        virtual void SetAdapter(const std::string &name);

        virtual void DisableAdapter();

        virtual bool SetSaveHistoryChat(bool save);

        virtual void SetDataType(DataType dataType);

        // messages: [ (role, content) ... ]
        virtual std::string ApplyChatTemplate(const ChatMessages &messages);

        virtual std::vector <int> ApplyChatTemplateToTokens(const ChatMessages &messages);

        virtual std::string ApplyChatTemplate(const JinjaVar &var);

        virtual std::vector <int> ApplyChatTemplateToTokens(const JinjaVar &var);

        std::string model_type;
        std::string model_struct;

        std::string pre_prompt; // 最初对话的提示语
        std::string user_role, bot_role, history_sep; // 用于生成每一轮的prompt

        int bos_token_id;
        int eos_token_id;
        std::set <int> eos_token_ids;
        int embed_dim = 4096;
        int num_attention_heads = 32;
        int head_dim = embed_dim / num_attention_heads;
        int max_positions = 32768;
        int rotary_dim = 64;
        const float scale_attn = sqrt(head_dim);
        int block_cnt = 28;

        std::vector<std::vector<float> > sin, cos;

        WeightMap weight; // 权重

        Data sinData, cosData;

        ResponseContextDict responseContextDict;

        std::thread *mainLoop = nullptr;
        std::mutex mainLoopLocker, dictLocker;
        std::condition_variable dictCV;

        std::map <std::string, int> deviceMap;

        std::string adapterName;

        int tokensLimit = -1;

        PastKVCacheManager pastKVCacheManager;
        bool saveHistoryChat = false;

        std::string lastPrompt = "";
        std::vector<std::pair<Data, Data> > *lastKeyValues = nullptr;
        int lastPromptTokens = 0;
        
        long long elementsInKVCachePerToken = -1; // 每个token使用多少个元素的的KVCache
        long long kvCacheLimit = -1;
        int maxBatch = -1;
        bool verbose = false;

        DataType dataType = DataType::FLOAT32;
        bool isFree = false; // 是否释放
    };
}

#endif //FASTLLM_BASELLM_H