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
};
}
