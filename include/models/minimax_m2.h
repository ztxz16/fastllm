//
// Created by huangyuyang on 4/29/25.
//

#ifndef FASTLLM_MINIMAX_M2_H
#define FASTLLM_MINIMAX_M2_H

#include "basellm.h"
#include "llama.h"
#include "utils/persistent_worker_group.h"

#include "cmath"

#include <atomic>
#include <iostream>
#include <map>
#include <mutex>
#include <unordered_map>

namespace fastllm {
    class MinimaxM2Model : public basellm {
    public:
        MinimaxM2Model (); // 构造函数

        virtual void InitParams(); // 初始化参数信息

        // 推理
        virtual int Forward(
                const Data &inputIds,
                const Data &attentionMask,
                const Data &positionIds,
                std::vector <std::pair <Data, Data> > &pastKeyValues,
                const GenerationConfig &generationConfig = GenerationConfig(),
                const LastTokensManager &lastTokens = LastTokensManager(),
                std::vector <float> *logits = nullptr);
        
        std::vector <int> ForwardV2(
                int batch,
                const Data &inputIds,
                const std::vector <Data*> &attentionMask,
                const std::vector <Data*> &positionIds,
                const std::vector <int> &seqLens,
                std::vector <std::pair <Data*, Data*> > &pastKeyValues,
                const std::vector <GenerationConfig> &generationConfigs,
                const LastTokensManager &lastTokens,
                std::vector <std::vector <float>*> *retLogits);

        virtual std::vector <int> ForwardGPU(
                int batch,
                const Data &inputIds,
                const std::vector <Data*> &attentionMask,
                const std::vector <Data*> &positionIds,
                const std::vector <int> &seqLens,
                std::vector <std::pair <Data*, Data*> > &pastKeyValues,
                const std::vector <GenerationConfig> &generationConfigs,
                const LastTokensManager &lastTokens,
                std::vector <std::vector <float>*> *retLogits);
                
        // 根据输入的tokens生成LLM推理的输入
        virtual void FillLLMInputsBatch(std::vector <std::vector <float> > &inputTokens,
                                        const std::vector <std::map <std::string, int> > &params,
                                        Data &inputIds, Data &attentionMask, Data &positionIds);

        // 是否需要生成AttentionMask
        virtual bool NeedAttentionMask(int qlen, int klen);
        
        virtual void WarmUp(); // 预热

        virtual bool CanUseGPUForward() const override;

        virtual PagedCacheManager* GetPagedKVCacheManager(int layerIndex, bool isKey) const override;
        virtual std::vector<std::pair<int, PagedCacheManager*> > GetPagedKVCacheManagers(int layerIndex, bool isKey) const override;

        virtual std::string MakeInput(const std::string &history, int round, const std::string &input); // 根据历史信息和当前输入生成prompt

        virtual std::string MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output); // 根据当前回复更新history

        std::pair<std::vector<float>, std::vector<float>> UpdateRotaryPosEmb(float base, float factor, int seqLen = 0); // 更新位置编码

    protected:
        bool IsThreadTensorParallelEnabled() const;

        void ForwardSingleGPU(
                int gpuId,
                std::map <int, int> ratios,
                int batch,
                const Data &inputIds,
                const Data &positionIds,
                const std::vector <int> &seqLens,
                std::vector <std::pair <Data*, Data*> > &pastKeyValues,
                bool all1,
                bool isPrefill,
                bool tensorParallel,
                bool firstTensorParallelRank,
                int pagedCacheLayerOffset,
                Data &logits,
                Data *precomputedHiddenStates = nullptr);

        Data &GetThreadTensorParallelBias(const std::string &name);

        RoPEType rope_type = RoPEType::BASE;

        float rope_base = 10000.f;

        float rope_factor = 1.f;

        int num_key_value_heads = num_attention_heads;

        float rms_norm_eps = 1e-6;

        bool mergeQKV = false;
        bool mergeSwiglu = false;

        std::vector <std::vector <Data*> > weights;
        std::vector <std::vector <Data*> > biass;
        std::unordered_map <int, std::vector <std::vector <Data*> > > threadTpMoeWeights;
        std::unordered_map <int, std::vector <std::vector <Data*> > > threadTpMoeBiass;
        std::unordered_map <int, std::vector <std::vector <Data*> > > singleGpuMoeWeights;
        std::unordered_map <int, std::vector <std::vector <Data*> > > singleGpuMoeBiass;

        float routed_scaling_factor = 1.0f;

        std::unordered_map <std::string, Data> threadTpEmptyBiases;
        int threadTpPagedCacheBase = -1;
        std::mutex threadTpWeightPrepareLock;
        std::atomic<bool> singleGpuWeightsPrepared{false};
        std::atomic<bool> threadTpWeightsPrepared{false};
        std::vector <int> threadTpPreparedDevices;
        std::map <int, int> threadTpPreparedRatios;
        std::vector <std::map <int, std::vector <std::pair <int, int> > > > threadTpKVHeadSchemes;
        std::map <int, std::vector <std::pair <int, int> > > threadTpLmHeadScheme;
        PersistentWorkerGroup threadTpWorkerGroup;
    };
}

#endif //FASTLLM_MINIMAX_M2_H
