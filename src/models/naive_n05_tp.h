#pragma once
#include "models/naive_n05_flash.h"
#include "utils/decode_graph.h"
#include "utils/cuda_decode_sampling.h"
#ifdef USE_CUDA
#include "devices/cuda/naive-n05-cuda.cuh"
#include "devices/cuda/fastllm-cuda.cuh"

namespace fastllm {
struct NaiveN05FlashModel::TargetWorkspace {
    Data hidden, normed, q, k, v, qkv, packed, packedIndexer, attn, projected;
    const Data *verifyProjectionScratch = nullptr;
    Data projectionQuant, routerInput, router, expertIndex, expertScore;
    Data w1, w2, w3, tempInput, tempOutput, moeOutput, moeInputTemp, moeOutputTemp;
    Data indexQ, indexKey, indexKeyFloat, indexWeights, indices, indexBroadcast, positions;
    Data indexContextGather, indexContextFull, indexContextExchange;
    Data last, logits, logitsBf16, inputIds, liveKeys;
    Data sparseQuery, sparseAttention;
    Data denseGate, denseUp, denseDown;
    const Data *samplingLogits = nullptr; // Serving-owned, fixed maximum shape.
    int samplingOffset = 0;
    bool directSamplingLogits = false;
    std::map<int, Data> features;
    FastllmNaiveDecodeScratch decode;
    const void *const *cacheTable = nullptr;
    const unsigned char *cacheMaps = nullptr;
    int tableRows = 0;
    int capacity = 0;
    std::vector<std::unique_ptr<TargetWorkspace>> sequences;
    void Pointers(std::vector<void *> &out) const {
        for (const Data *p : {&hidden, &normed, &q, &k, &v, &qkv, &packed, &packedIndexer,
                &attn, &projected, &projectionQuant, &routerInput, &router, &expertIndex, &expertScore,
                &w1, &w2, &w3, &tempInput, &tempOutput, &moeOutput, &moeInputTemp, &moeOutputTemp,
                &indexQ, &indexKey, &indexKeyFloat, &indexWeights, &indices, &indexBroadcast,
                &positions, &indexContextGather, &indexContextFull, &indexContextExchange,
                &last, &logits, &logitsBf16, &inputIds, &liveKeys, &sparseQuery, &sparseAttention,
                &denseGate, &denseUp, &denseDown, &decode.indexQuery,
                &decode.indexScale, &decode.indexScores, &decode.topk, &decode.attentionScores,
                &decode.attentionPartial, &decode.attentionCompletion, &decode.dsaFrontReady, &decode.dsaFused, &decode.dsaLayer,
                &decode.windowKey, &decode.windowValue})
            out.push_back(p->isFake ? nullptr : p->cudaData);
        if (samplingLogits) out.push_back(samplingLogits->cudaData);
        for (const auto &feature : features) out.push_back(feature.second.cudaData);
        for (const auto &part : sequences) part->Pointers(out);
    }
};

struct NaiveN05FlashModel::TPDecodeState {
    int lookupCapacity = 0; // Requested bucket, before clipping to physical KV capacity.
    enum Mode { Warm, Prepare, Capture, Replay } mode = Warm;
    struct Storage {
        // All pre-captured shapes share these allocations. Inputs and the KV
        // table keep their maximum-batch strides even for smaller graphs.
        Data inputStorage, cacheTable, cacheMaps, indexScores, indexExchange;
        Data decodeTokenRows[2];
        void *gatherReady = nullptr, *pinnedInput = nullptr;
        std::map<int, void *> commitGraphs;
        ~Storage() {
            if (gatherReady) FastllmCudaEventDestroy(gatherReady);
            if (pinnedInput) FastllmCudaHostFree(pinnedInput);
            if (!commitGraphs.empty()) {
                const int previous = FastllmCudaGetDevice();
                FastllmCudaSetDevice(cacheTable.dataDeviceIds.front());
                for (auto &entry : commitGraphs) FastllmCudaGraphExecDestroy(entry.second);
                FastllmCudaSetDevice(previous);
            }
        }
        std::shared_ptr<void> linearGraphWorkspace;
        std::vector<unsigned char> hostInput;
        std::vector<void *> hostCache;
        std::vector<unsigned char> hostMaps;
        bool cacheDirty = false, mapsDirty = false;
        int maxRows = 0;
        int boundRowEnd = -1;
        TargetWorkspace buffers;
        Data logitsPartial, logitsCandidates, packedFeatures;
        std::map<int, Data> features;
        uint64_t bindingGeneration = 0;
        std::vector<void *> Pointers() const {
            std::vector<void *> out;
            for (const Data *p : {&inputStorage, &cacheTable, &cacheMaps, &indexScores,
                    &indexExchange, &logitsPartial, &logitsCandidates}) out.push_back(p->isFake ? nullptr : p->cudaData);
            for (const auto &feature : features) out.push_back(feature.second.cudaData);
            buffers.Pointers(out);
            return out;
        }
    };
    struct Rank {
        explicit Rank(std::shared_ptr<Storage> workspace = std::make_shared<Storage>())
            : storage(std::move(workspace)) {}
        std::shared_ptr<Storage> storage;
        TargetCapture features;
        void *graph = nullptr, *exec = nullptr;
        void *selectionGraph = nullptr, *selectionExec = nullptr;
        void *pipelineExec[2]{};
        std::shared_ptr<void> draftVerification;
        void *draftFused = nullptr;
        const void *draftHidden = nullptr;
        bool ok = true, directSamplingLogits = false;
        std::vector<void *> communicationPointers;
        std::vector<size_t> captureAllocations;
    };
    std::vector<int> devices;
    std::vector<std::unique_ptr<Rank>> ranks;
    std::vector<void *> cachePointers, reservedPointers;
    std::vector<int> cacheCapacities, cacheLayouts;
    std::vector<int> sequenceRegions, sequenceCapacities, sequenceRows;
    uint64_t ncclGeneration = 0;
    int capacity = 0, region = 0, rows = 1;
    bool verifying = false, collectHidden = false, dynamicCache = false;
    LogitsSelection selection;
    bool warmed = false, captured = false, disabled = false, active = false;
    // A failed capture can still execute the successfully warmed verification
    // kernels eagerly, using the same bounded buffers and dynamic KV bindings.
    bool preparedEager = false;
    // Only scheduler-owned requests may reuse these validated host bindings.
    // Physical page/COW preparation still runs on every verification forward.
    struct VerificationBindings {
        bool valid = false;
        uint64_t cacheEpoch = 0;
        std::vector<const Data *> requests;
        std::vector<uint64_t> identities, storageGenerations;
        std::vector<int> requestedCapacities;
        std::vector<const void *> storages;
        std::vector<int *> lengths;
        std::vector<Data *> paged;
    } verificationBindings;
    void ClearGraphs();
    ~TPDecodeState();
};
struct NaiveN05FlashModel::TPServingState {
    // Destroy graphs and sampler before their shared, immovable output storage.
    Data samplingLogits;
    std::shared_ptr<CudaDecodeSampling> sampling;
    int pipelineInputSlot = -1, pipelineOutputSlot = -1;
    std::vector<const float *> feedbackRows[2];
    bool preparing = false, capturing = false, sealed = false, pipelineReady = false;
    int prepareCapacity = 0, verifyWidth = 0;
    DataType moeType = BFLOAT16, kvType = BFLOAT16;
    uint64_t generation = 0, cacheEpoch = UINT64_MAX;
    std::vector<const Data *> requests;
    std::vector<uint64_t> requestIds;
    std::vector<int> expectedLengths, capacityLimits;
    std::vector<int *> cacheLengths, pageLengths;
    std::vector<void *> cachePointers, pageTables;
    std::vector<int> cacheCapacities;
    bool bindingsDirty = false;
    std::vector<size_t> bindingOrder;
    size_t boundRowsBeforeCompaction = 0;
    std::vector<int> batchBuckets;
    std::map<std::pair<int, int>, std::shared_ptr<TPDecodeState>> graphs;
};
}
#endif
