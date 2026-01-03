
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

    enum ResponseContextError {
        ResponseContextErrorNone = 0, ResponseContextErrorPromptTooLong
    };

    class basellm;

    struct ResponseContext {
        bool isEnding = false; // 代表这个请求已经处理完成了，不需要再forward了，但生成的token可能还没有被fetch
        bool isAbort = false; // 代表这个请求被中断了，也就是说不会再有人来fetch它了，那么推理完之后就可以删除这个请求了
        
        std::vector <int> allTokens;
        std::vector <std::pair <Data, Data> > pastKeyValues;
        std::vector <int> currentTokens;
        std::map <std::string, std::vector <Data*> > multimodalInput;
        std::queue <int> resultTokenQueue;
        std::queue <std::vector <float>*> resultLogits;
        GenerationConfig generationConfig;
        LastTokensUnit tokens;
        ResponseContextError error = ResponseContextErrorNone;

        int preTokens = 0;
        int curTokens = 0;
        std::map <std::string, int> intParams;

        int cacheLen = 0;

        void Init(int blocks, DataType dataType);
        void TryRecord(basellm *model);
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
        std::pair <PastKVCacheMemory*, int> Get(const std::vector <int> &inputToken);

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

    struct WeightMergeRuleSingle {
        std::vector <std::string> inputs;
        std::string output;
        std::string type;

        WeightMergeRuleSingle (const std::vector <std::string> &inputs, std::string output, std::string type) :
            inputs(inputs), output(output), type(type) {}
    };

    struct WeightMergeRule {
        // 权重合并的规则
        std::vector <WeightMergeRuleSingle> rules; 
        // 当rules涉及到的所有权重都被读取后，依此遍历rules中的每条规则，如果都满足合并条件，那么执行合并

        std::set <std::string> allInputs; // 所有涉及到的合并前的name

        WeightMergeRule (const std::vector <WeightMergeRuleSingle> &rules) : rules (rules) {
            for (auto &rule : rules) {
                for (auto &input : rule.inputs) {
                    allInputs.insert(input);
                }
            }
        }
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
        
        virtual std::vector <int> ForwardMultimodal(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector<std::pair<Data, Data> > &pastKeyValues,
                const std::map <std::string, std::vector <Data*> > &multimodalInput,
                const GenerationConfig &generationConfigs,
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
                                         const GenerationConfig &generationConfig = GenerationConfig(),
                                         const std::map <std::string, std::vector <Data*> > &multimodalInput = {}); // 启动一个response任务，返回分配的handleId
        
        virtual bool CanFetchResponse(int handleId); // 判断当前是否能fetch到，用于异步操作

        virtual int FetchResponseTokens(int handleId); // 获取指定handle的输出, -1代表输出结束了 

        virtual int FetchResponseLogits(int handleId, std::vector <float> &logits); // 获取指定handle的输出Logits

        virtual void AbortResponse(int handleId); // 中断handleId的请求

        virtual void SaveLowBitModel(const std::string &fileName, int bit); // 存储成量化模型 

        virtual void SaveModel(const std::string &fileName); // 直接导出

        virtual void WarmUp() {}; // 预热

        virtual void AddPromptCache(const std::vector <int> &inputTokens);

        virtual std::string MakeInput(const std::string &history, int round, const std::string &input) = 0; // 根据历史信息和当前输入生成prompt

        virtual std::string MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) = 0; // 根据当前回复更新history

        virtual void SetAdapter(const std::string &name);

        virtual void DisableAdapter();

        virtual bool SetSaveHistoryChat(bool save);

        virtual void SetMoeExperts(int experts);

        virtual void SetDataType(DataType dataType);

        virtual void UpdateRotaryPtr(Data **sinDataPtr, Data **cosDataPtr, const std::string &device);

        // messages: [ (role, content) ... ]
        virtual std::string ApplyChatTemplate(const ChatMessages &messages);

        virtual std::vector <int> ApplyChatTemplateToTokens(const ChatMessages &messages);

        virtual std::string ApplyChatTemplate(const JinjaVar &var);

        virtual std::vector <int> ApplyChatTemplateToTokens(const JinjaVar &var);

        // 输出未满足最低长度时阻止产生EOS
        virtual void ResetLogitsOfEOS(int batch, Data *logits, std::vector <std::pair <Data, Data> > &pastKeyValues, 
            const GenerationConfig &generationConfig);
        virtual void ResetLogitsOfEOS(int batch, Data *logits, std::vector <std::pair <Data*, Data*> > &pastKeyValues, 
            const std::vector <GenerationConfig> &generationConfigs); 

        std::string model_type;
        std::string model_struct;
        bool is_multi_modal = false; // 是否是多模态模型

        std::string pre_prompt; // 最初对话的提示语
        std::string user_role, bot_role, history_sep; // 用于生成每一轮的prompt

        int bos_token_id = -1;
        int eos_token_id = -1;
        std::set <int> eos_token_ids;
        int embed_dim = 4096;
        int num_attention_heads = 32;
        int num_key_value_heads = num_attention_heads;
        float rms_norm_eps = 1e-6;
        int head_dim = embed_dim / num_attention_heads;
        int max_positions = 32768;
        int rotary_dim = 64;
        const float scale_attn = sqrt(head_dim);
        int block_cnt = 28;

        bool use_qk_norm = false;
        // 以下是moe相关参数
        float routed_scaling_factor = 1.0f;
        int n_shared_experts = 0;
        int num_experts_per_tok;
        int num_experts;
        bool norm_topk_prob;

        std::vector <WeightMergeRule> weightMergeRules;
        std::map <std::string, std::string> specialWeights; //一些特殊层，可以提前注册（一般用于TFACC）
        std::set <std::string> cantQuantLinears; // 不能量化的Linear层
        std::set <std::string> moeLinears;

        std::vector<std::vector<float> > sin, cos;

        WeightMap weight; // 权重

        Data sinData, cosData;
        std::map <std::string, Data*> deviceSinDatas, deviceCosDatas; // deviceSinDatas[xxx]代表xxx设备上的sinData

        ResponseContextDict responseContextDict;

        std::thread *mainLoop = nullptr;
        std::mutex mainLoopLocker, dictLocker, forwardLocker;
        std::condition_variable dictCV;

        std::map <std::string, int> deviceMap;
        std::map <std::string, int> moeDeviceMap;

        std::string adapterName;

        int tokensLimit = -1;
        int promptLimit = -1;

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

        int kvCacheId = 0; // 最早使用kv_cache的层编号 （因为有一些混合架构的模型，其中一些block是线性attention）
        bool canDoBatchForward = true; // 是否支持batch推理
    };
}

#endif //FASTLLM_BASELLM_H