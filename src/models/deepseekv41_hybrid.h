#pragma once
#include "fastllm.h"
#include "utils/hybrid_moe.h"
#include <map>

namespace fastllm {
// Hybrid router staging has the same lifetime as its decode/graph workspace.
// Pure GPU forwards never allocate it or depend on its contents.
struct DeepSeekV41HybridWorkspace {
    struct MoeCacheWorkspace { Data index, score, output; };
    std::map<int, MoeCacheWorkspace> moeCache;
#ifdef USE_CUDA
    // PP may place a resident expert on another GPU. Keep both sides of the
    // transfer alive for graph replay; never migrate a captured stage tensor.
    struct RemoteMoeWorkspace {
        int source = -1, device = -1;
        void *ready = nullptr, *done = nullptr;
        Data input, index, score, output;
        Data w1, w2, w3, tempInput, tempOutput, inputTemp, outputTemp;
        ~RemoteMoeWorkspace();
    };
    std::map<int, RemoteMoeWorkspace> remoteMoe;
#endif
};
}
