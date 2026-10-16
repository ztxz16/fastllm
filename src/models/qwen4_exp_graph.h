#pragma once
#include "qwen4_exp.h"
namespace fastllm {
    struct Qwen4ExpModel::DecodeCudaGraphState {
        struct Segment {
            bool warmed = false;
            bool captured = false;
            bool disabled = false;
            bool parallelRegionsUnavailable = false;
            int captureFailures = 0;
            void *graph = nullptr;
            void *exec = nullptr;
            std::vector<void*> reservedPointers;
            std::vector<size_t> allocationRequests;
        };

        std::mutex mutex;
        int device = -1;
        int graphStartLayer = -1;
        bool startBeforeAttention = false;
        std::vector<void *> linearCachePointers;
        std::vector<uint64_t> fullCacheSignature;
        bool wholeGraphMode = false;
        bool hybridDenseMode = false;
        struct HybridBoundary {
            Data input;
            Data injection;
            Data output;
            MoEPrepared moe;
        };
        // CPU experts execute between dense graphs. Each boundary owns stable
        // storage, with dtype and shape established by its eager warmup.
        std::map<int, HybridBoundary> hybridBoundaries;
        Data hiddenStates[2];
        Data attentionInput;
        Data attentionInjection;
        Data attentionOutput;
        Data positionIds;
        Data decodeMeta;
        std::map<int, Data> denseIndices;
        Data logits;
        std::map<int, Data> qsaScoreWorkspaces;
        std::map<int, Data> qsaSelectedWorkspaces;
        std::map<int, Data> qsaTailKeySnapshots;
        std::map<int, Data> qsaTailPositionSnapshots;
        int32_t *pinnedDecodeMeta = nullptr;
        std::map<int, Segment> segments;

        // Layer 0 has no growing full-attention cache. Keep its TP graph
        // through backbone recaptures, but release it with the request slot.
        std::unique_ptr<DecodeCudaGraphState> tpPrefix;
        // Each contiguous device/PLE range owns its captured addresses. Keep
        // earlier pipeline stages alive when another stage grows its KV cache.
        std::map<std::pair<int, int>, std::unique_ptr<DecodeCudaGraphState>> stages;
        // Verifier lengths alternate as the confidence threshold truncates
        // drafts. Each length owns its graph buffers and rollback captures.
        std::map<int, std::unique_ptr<DecodeCudaGraphState>> verifierLengths;
        TargetVerificationCapture verification;
        // A short request keeps eager attention after crossing the sparse
        // threshold, preserving its original key order and reduction shape.
        bool verifierEagerAttention = false;

        void DestroyGraphs();

        void Reset(int newDevice);

        ~DecodeCudaGraphState();
    };
#ifdef USE_CUDA
    struct Qwen4ExpModel::DecodeGraphExecution {
        Qwen4ExpModel &model;
        DecodeCudaGraphState *graphState;
        int graphStartLayer, graphEndLayer, graphSequence, device, decodePreviousLength;
        int wholeGraphRemainder, wholeGraphSparseWidth;
        bool startBeforeAttention, mtpTargetGraph, wholeDenseGraph, hybridDenseGraph;
        bool hybridEagerAttention, linearOnlyStage, wholeGraphReady, qsaDeviceCompatibleMask;
        const Data &attentionMask, &positionIds;
        std::vector<std::pair<Data, Data>> &pastKeyValues;
        RequestState &requestState;
        TargetVerificationCapture *verificationCapture;
        std::function<bool(int, const Data &, Data &, Data &, Data &)> runFullAttentionGraph;
        std::function<void(int, const std::function<void()> &)> runCapturedSegment;
        std::unique_ptr<Data> hybridCausalMask;
        void RunGpu();
        void RunGpuSegment(int startLayer, int nextFullLayer, bool wholeGraph,
                           int wholeGraphVariant);
        void RunGpuSegmentBody(int startLayer, int nextFullLayer, bool firstAttentionReady);
        void RunGpuWholeBody();
        void PrepareHybridMask();
        void RunHybrid();
        void RunHybridDenseBody(int layer);
        void RunHybridFinalBody();
    };
#endif
} // namespace fastllm
