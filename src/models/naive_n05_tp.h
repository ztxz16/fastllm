#pragma once
#include "models/naive_n05_flash.h"
#ifdef USE_CUDA
#include "devices/cuda/naive-n05-cuda.cuh"

namespace fastllm {
struct NaiveN05FlashModel::TargetWorkspace {
    Data hidden, normed, q, k, v, packed, attn, projected;
    Data routerInput, router, expertIndex, expertScore;
    Data w1, w2, w3, tempInput, tempOutput, moeOutput, moeInputTemp, moeOutputTemp;
    Data indexQ, indexKey, indexKeyFloat, indexWeights, indices, positions;
    Data last, logits, logitsBf16, inputIds, liveKeys;
    Data denseGate, denseUp, denseDown;
    FastllmNaiveDecodeScratch decode;
    int capacity = 0;
};

struct NaiveN05FlashModel::TPDecodeState {
    enum Mode { Warm, Prepare, Capture, Replay } mode = Warm;
    struct Rank {
        TargetWorkspace buffers;
        TargetCapture features;
        Data logitsPartial, logitsCandidates;
        void *graph = nullptr, *exec = nullptr;
        bool ok = true;
        std::vector<void *> communicationPointers;
    };
    std::vector<int> devices;
    std::vector<std::unique_ptr<Rank>> ranks;
    std::vector<void *> cachePointers, reservedPointers;
    std::vector<int> cacheCapacities;
    uint64_t ncclGeneration = 0;
    int capacity = 0, region = 0, rows = 1;
    bool verifying = false;
    LogitsSelection selection;
    bool warmed = false, captured = false, disabled = false, active = false;
    void ClearGraphs();
    ~TPDecodeState();
};
}
#endif
