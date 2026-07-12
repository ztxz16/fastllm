//
// Created by huangyuyang on 4/29/25.
//

#ifndef FASTLLM_HY_V3_H
#define FASTLLM_HY_V3_H

#include "basellm.h"
#include "llama.h"
#include "utils/persistent_worker_group.h"

#include "cmath"

#include <atomic>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <unordered_map>

namespace fastllm {
    class HyV3Model : public basellm {
    public:
        HyV3Model (); // 构造函数

        virtual void InitParams(); // 初始化参数信息

        // 根据原始的tensorNames获得映射表
        virtual std::map <std::string, std::vector <std::pair <std::string, DataType> > >
                GetTensorMap(const std::vector <std::string> &tensorNames) override;

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

        virtual void OnAutoWarmupFinished() override;

        virtual bool CanUseGPUForward() const override;

        virtual PagedCacheManager* GetPagedKVCacheManager(int layerIndex, bool isKey) const override;
        virtual std::vector<std::pair<int, PagedCacheManager*> > GetPagedKVCacheManagers(int layerIndex, bool isKey) const override;

        virtual void OnWeightsCreated(const std::set<std::string> &allWeightNames) override;

        virtual int GetWeightLoadPriority(
                const std::string &tensorName,
                const std::vector <std::pair <std::string, DataType> > &mappedWeights) const override;

        virtual bool ShouldLoadWeightSeriallyBeforeOthers(
                const std::string &tensorName,
                const std::vector <std::pair <std::string, DataType> > &mappedWeights) const override;

        virtual void OnWeightLoadGroupStarted(const std::set<std::string> &weightNames) override;

        virtual void OnWeightLoaded(const std::string &weightName,
                                    const std::set<std::string> &finishedWeightNames) override;

        virtual bool IsWeightConsumedAfterLoad(const std::string &weightName) const override;

        virtual void OnWeightLoadGroupFinished() override;

        virtual void OnModelWeightsLoaded() override;

        virtual bool ShouldDelaySpecialWeightCudaMove(const std::string &weightName) const override;

        virtual std::string MakeInput(const std::string &history, int round, const std::string &input); // 根据历史信息和当前输入生成prompt

        virtual std::string MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output); // 根据当前回复更新history

    protected:
        bool IsThreadTensorParallelEnabled() const;

        std::vector <int> ForwardV2ThreadTensorParallel(
                int batch,
                const Data &inputIds,
                const std::vector <Data*> &attentionMask,
                const std::vector <Data*> &positionIds,
                const std::vector <int> &seqLens,
                std::vector <std::pair <Data*, Data*> > &pastKeyValues,
                const std::vector <GenerationConfig> &generationConfigs,
                const LastTokensManager &lastTokens,
                std::vector <std::vector <float>*> *logits = nullptr);

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

        bool ForwardSingleGPUDecodeGraph(
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
                Data &logits);

        void PreCaptureCudaGraphAfterWarmup();

        Data &GetThreadTensorParallelBias(const std::string &name);

        RoPEType rope_type = RoPEType::BASE;

        float rope_base = 10000.f;

        float rope_factor = 1.f;

        int num_key_value_heads = num_attention_heads;

        float rms_norm_eps = 1e-6;

        bool mergeQKV = false;
        bool mergeSwiglu = false;

        bool loadFusedMoePlanned = false;
        bool moeWeightsPrepared = false;
        bool moeFusedWeightsPrepared = false;
        std::vector <char> moeFusedLayerPlanned;
        std::set <std::string> loadFusedMoeSourceWeights;
        std::set <std::string> consumedFusedMoeSourceWeights;
        std::vector <std::vector <Data*> > weights;
        std::vector <std::vector <Data*> > biass;
        std::vector <Data*> moeGate3DWeights;
        std::vector <Data*> moeUp3DWeights;
        std::vector <Data*> moeDown3DWeights;
        std::vector <std::vector <char> > moeGate3DExpertReady;
        std::vector <std::vector <char> > moeUp3DExpertReady;
        std::vector <std::vector <char> > moeDown3DExpertReady;
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

        void PrepareMoeWeights(bool enableFusedMoe);
        bool TryConsumeFusedMoeSourceWeight(const std::string &weightName);
        void TryFinalizeFusedMoeLayerParts(int layer);
        bool TryBuildFusedMoeWeightsFromLoaded();
        bool TryBuildFusedMoeLayerFromLoaded(int layer);
        bool IsFusedMoeLayerPlanned(int layer) const;
        bool HasPlannedFusedMoeLayers() const;
        bool ArePlannedFusedMoeLayersReady() const;
        bool HasFusedMoeWeights(int layer) const;
        Data *GetFusedMoeWeightForDevice(Data *weight, int device) const;
        void PrepareFusedMoeLayerForDevices(int layer, const std::vector <int> &devices,
                                             std::map <int, int> ratios);
        void PrepareFusedMoeWeightsForDevices(const std::vector <int> &devices,
                                              std::map <int, int> ratios);

        bool IsHyV3() const;
        bool IsDenseMlpLayer(int layer) const;
        bool HasSharedMlpLayer(int layer) const;
        std::string GetMoeGateWeightName(int layer) const;
        std::string GetMoeGateBiasName(int layer) const;
        std::string GetSharedMlpPrefix(int layer) const;
        void ApplyRouterActivation(Data &routerLogits) const;

        int first_k_dense_replace = 0;
        bool moe_router_use_sigmoid = false;
    };
}

#endif //FASTLLM_HY_V3_H
