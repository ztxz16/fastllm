#ifndef FASTLLM_CUDA_CACHE_BUDGET_H
#define FASTLLM_CUDA_CACHE_BUDGET_H

#include <algorithm>

namespace fastllm {
    // Shared by paged AutoWarmup and models with continuous KV storage.
    // Measure available memory after warming the actual serving workspaces.
    inline long long CudaCacheRuntimeHeadroom(long long total, long long available) {
        const long long headroom = std::min(
            std::max(512LL * 1024 * 1024, total / 100), 2LL * 1024 * 1024 * 1024);
        return std::max(0LL, std::min(headroom, available / 4));
    }
}

#endif
