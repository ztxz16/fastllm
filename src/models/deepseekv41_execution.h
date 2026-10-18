#ifndef FASTLLM_DEEPSEEKV41_EXECUTION_H
#define FASTLLM_DEEPSEEKV41_EXECUTION_H

#include "deepseekv41.h"
#include <functional>

namespace fastllm {
    struct DeepSeekV41HybridWorkspace;
    struct DeepSeekV41DecodeWorkspace {
        // Reuse request-row views and overwritten indexer outputs across
        // decode calls. Per-forward validity flags still start empty: retained
        // storage must never imply that an earlier request's indices are valid.
        std::vector<Data> segTopK, segCandidate, segQ, segKV, segAttnOut;
        std::vector<Data> segIndexQ, segIndexWeight;
        Data compressJoined, compressScores, compressLatents, compressKeys;
        std::vector<Data> compressLatentViews, compressKeyViews;

        DeepSeekV41DecodeWorkspace();
        ~DeepSeekV41DecodeWorkspace();
        Data hiddenStates, hiddenTemp, preMix, dsparkPreMix;
        int preMixSeqlen = -1;      // preMix 只在长度变化时重建（内容是常量 one-hot）
        Data attnPre, attnPost, attnComb, ffnPre, ffnPost, ffnComb;
        Data x, attnInput, qr, qNorm, q, kv, attnOut, woAOut, attnProj;
        Data attnInputFloat, rawKVAll, rawScoreAll, qIdxAll, idxWeights, idxWeightsAll;
        Data ffnInput, ffnOut, gateInput, gateLogits, expertIndex, expertScore;
        Data sharedGateup, sharedSwiglu, sharedExpertOut;
        Data w1, w2, w3, tempInput, tempOutput, moeInputTemp, moeOutputTemp;
        Data cpuMoeInput, cpuMoeIndex, cpuMoeScore;
        std::unique_ptr<DeepSeekV41HybridWorkspace> hybrid;
        Data quantizedActivation;
        Data attnInputQuantized, qNormQuantized;
        // TP and CUDA graphs retain one buffer per weight.
        std::map<std::string, Data> quantizedActivations;
        std::map<std::string, Data> reductionBuffers;
    };

    struct DeepSeekV41MoeContext {
        DeepSeekV41DecodeWorkspace &workspace;
        std::vector<Data *> &moeWeights;
        const std::vector<int> &tpDevices;
        int layer, seqlen;
        bool tp, single, dsparkVerify, graphActive, dumpDebug, hasSharedExpertOut;
        bool verifyAllTokens;
        std::function<void()> runSharedSegment;
        std::function<void()> releaseSharedScratch;
        std::mutex *pipelineHostExperts = nullptr;
        int pipelineFirstLayer = -1;
    };

    // Tensor-copy helpers shared with attention/KV code. They preserve graph
    // addresses and know when TP roots contain metadata rather than payloads.
    void V41Assign(Data &, const Data &, const std::vector<int> &);
    void V41ReplicaToCpu(Data &, const Data &, const std::vector<int> &);
#ifdef USE_CUDA
    void V41ResetMultiDevice(Data &);
    void V41SyncRootFromReplica(Data &, const std::vector<int> &);
#endif
    bool V41EnvFlag(const char *);
    bool V41DeviceSpecUsesType(const std::string &, const std::string &);
    bool V41DeviceMapUsesCuda(const std::map<std::string, int> &);
    int V41DeviceMapCudaDeviceCount(const std::map<std::string, int> &);
} // namespace fastllm

namespace fastllm {
// Private to hybrid PP: layers have disjoint owners, while token history and
// logical positions are per stage. Payload stores BF16 hidden plus FP32 HC pre.
struct DeepSeekV41Model::HybridPrefillStage {
    int firstLayer = 0, endLayer = 0;
    DeepSeekV41RequestState *request = nullptr;
    Data *payload = nullptr;
    std::mutex *hostExperts = nullptr;
    bool sampleOutput = false;
};
struct DeepSeekV41Model::ForwardState {
    std::unique_ptr<ForwardWorkspace> work;
    ForwardState(DeepSeekV41Model &model, const std::vector<DeepSeekV41Segment> &inputSegments,
                                                       const Data &inputIds,
                                                       const Data *inputEmbeds,
                                                       const std::vector<int> *imageMask,
                                                       const std::vector<GenerationConfig> &generationConfigsIn,
                                                       const LastTokensManager &lastTokens,
                                                       std::vector<std::vector<float>*> *retLogits,
                                                       std::vector<std::pair<Data*, Data*> > &samplingPastKeyValues,
                                                       HybridPrefillStage *prefill = nullptr);
    ~ForwardState();
    int Layers() const;
    void BeginLayer(int layer);
    void EndLayer(int layer);
    void GpuExperts(int layer);
    void HybridExperts(int layer);
    void BeginHybridStep();
    void PrepareInputs();
    bool FullGraph();
    std::vector<int> Finish();
};
}

#endif // FASTLLM_DEEPSEEKV41_EXECUTION_H
