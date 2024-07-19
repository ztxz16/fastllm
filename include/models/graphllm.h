//
// Created by huangyuyang on 6/24/24.
//

#ifndef FASTLLM_GRAPHLLM_H
#define FASTLLM_GRAPHLLM_H

#include "basellm.h"
#include "graph.h"
#include "cmath"

namespace fastllm {
    class GraphLLMModelConfig;

    class GraphLLMModel: public basellm {
    public:
        GraphLLMModel (const std::string &type); // 构造函数, graphModel必须要知道type

        virtual void InitParams(); // 初始化参数信息

        virtual std::map <std::string, std::vector <std::pair <std::string, DataType> > >
                GetTensorMap(const std::vector <std::string> &tensorNames);
                
        // 推理
        virtual int Forward(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector <std::pair <Data, Data> > &pastKeyValues,
                const GenerationConfig &generationConfig = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector <float> *logits = nullptr);

        std::vector <int> ForwardBatch(
                int batch,
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector <std::pair <Data, Data> > &pastKeyValues,
                const GenerationConfig &generationConfig = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector <std::vector <float>*> *logits = nullptr);

        std::vector <int> ForwardBatch(
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

        virtual void WarmUp(); // 预热

        virtual std::string MakeInput(const std::string &history, int round, const std::string &input); // 根据历史信息和当前输入生成prompt

        virtual std::string MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output); // 根据当前回复更新history

        std::pair<std::vector<float>, std::vector<float>> UpdateRotaryPosEmb(float base, float factor, int seqLen = 0); // 更新位置编码

        void BuildGraph(); // 构造运算图

        ComputeGraph *GetGraph();

        RoPEType rope_type = RoPEType::BASE;

        float rope_base = 10000.f;

        float rope_factor = 1.f;

        int num_key_value_heads = num_attention_heads;

        float rms_norm_eps = 1e-6;

        GraphLLMModelConfig *graphLLMModelConfig = nullptr;
    protected:
        ComputeGraph graph;

        bool inited = false;
    };

    // 模型配置的基类，用于描述一个具体的modeltype相关的模型信息
    class GraphLLMModelConfig {
    public:
        // 初始化
        virtual void Init(const std::string &config);

        // 参数初始化
        virtual void InitParams(GraphLLMModel *model);

        // 获得 (tensorName -> (weightName, DataType)) 映射关系
        virtual std::map <std::string, std::vector <std::pair <std::string, DataType> > >
                GetTensorMap(GraphLLMModel *model, const std::vector <std::string> &tensorNames) = 0;

        // 建立计算图
        virtual void BuildGraph(GraphLLMModel *model) = 0;
    };

    // 工厂类
    using GraphLLMModelConfigCreator = std::function<GraphLLMModelConfig*()>;
    class GraphLLMModelConfigFactory {
    public:
        static void RegisterGraphLLMModelConfig(const std::string& type, GraphLLMModelConfigCreator creator);
        static GraphLLMModelConfig* CreateGraphLLMModelConfig(const std::string& type);
    };
    
    #define REGISTERGRAPHMODELCONFIG(className, classType) \
    class className##GraphModelConfigHelper { \
        public: \
        className##GraphModelConfigHelper() { \
                GraphLLMModelConfigFactory::RegisterGraphLLMModelConfig(#className, []() { \
                        auto* obj = new classType(); \
                        return (GraphLLMModelConfig*)obj; \
                }); \
        } \
    }; \
    className##GraphModelConfigHelper className##graphModelConfighelper;
}

#endif // FASTLLM_GRAPHLLM_H
