#pragma once
#include <memory>

// BF16 sparse MLA with one independently masked query per row. The caller
// fills row-major token addresses and lengths on the same stream as attention.
struct FastllmCudaMLAToken {
    const void *latent, *rope;
};
struct FastllmCudaMLAPlan;
bool FastllmCudaMLAPagedPlanSupported(int device);
std::shared_ptr<FastllmCudaMLAPlan> FastllmCudaPrepareMLAPaged(int rows, int heads, int candidates);
bool FastllmCudaMLAPaged(const FastllmCudaMLAPlan &plan, const void *query, const void *queryPe,
                         const FastllmCudaMLAToken *tokens, const int *lengths, float scale, void *output,
                         float *naturalLogSumExp);
