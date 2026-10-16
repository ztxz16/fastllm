#pragma once
#include "models/naive_n05_flash.h"
#include "utils/decode_graph.h"
#ifdef USE_CUDA
#include "devices/cuda/naive-n05-cuda.cuh"

namespace fastllm {
struct NaiveN05FlashModel::TargetWorkspace {
    Data hidden, normed, q, k, v, qkv, packed, packedIndexer, attn, projected;
    Data routerInput, router, expertIndex, expertScore;
    Data w1, w2, w3, tempInput, tempOutput, moeOutput, moeInputTemp, moeOutputTemp;
    Data indexQ, indexKey, indexKeyFloat, indexWeights, indices, positions;
    Data last, logits, logitsBf16, inputIds, liveKeys;
    Data denseGate, denseUp, denseDown;
    FastllmNaiveDecodeScratch decode;
    const void *const *cacheTable = nullptr;
    const unsigned char *cacheMaps = nullptr;
    int tableRows = 0;
    int capacity = 0;
    std::vector<std::unique_ptr<TargetWorkspace>> sequences;
};

struct NaiveN05FlashModel::TPDecodeState {
    enum Mode { Warm, Prepare, Capture, Replay } mode = Warm;
    struct Storage {
        // All pre-captured shapes share these allocations. Inputs and the KV
        // table keep their maximum-batch strides even for smaller graphs.
        Data inputStorage, cacheTable, cacheMaps, indexScores;
        std::shared_ptr<void> linearGraphWorkspace;
        std::vector<unsigned char> hostInput;
        std::vector<void *> hostCache;
        std::vector<unsigned char> hostMaps;
        bool cacheDirty = false, mapsDirty = false;
        int maxRows = 0;
        TargetWorkspace buffers;
        Data logitsPartial, logitsCandidates;
    };
    struct Rank {
        explicit Rank(std::shared_ptr<Storage> workspace = std::make_shared<Storage>())
            : storage(std::move(workspace)) {}
        std::shared_ptr<Storage> storage;
        TargetCapture features;
        void *graph = nullptr, *exec = nullptr;
        bool ok = true;
        std::vector<void *> communicationPointers;
    };
    std::vector<int> devices;
    std::vector<std::unique_ptr<Rank>> ranks;
    std::vector<void *> cachePointers, reservedPointers;
    std::vector<int> cacheCapacities;
    std::vector<int> sequenceRegions, sequenceCapacities;
    uint64_t ncclGeneration = 0;
    int capacity = 0, region = 0, rows = 1;
    bool verifying = false, collectHidden = false;
    LogitsSelection selection;
    bool warmed = false, captured = false, disabled = false, active = false;
    void ClearGraphs();
    ~TPDecodeState();
};
struct NaiveN05FlashModel::TPServingState {
    bool preparing = false, capturing = false, sealed = false;
    int prepareCapacity = 0;
    DataType moeType = BFLOAT16, kvType = BFLOAT16;
    uint64_t generation = 0, cacheEpoch = UINT64_MAX;
    std::vector<const Data *> requests;
    std::vector<int> expectedLengths, capacityLimits;
    std::vector<int *> cacheLengths, pageLengths;
    std::vector<int> batchBuckets;
    std::map<std::pair<int, int>, std::shared_ptr<TPDecodeState>> graphs;
};
}
#endif
