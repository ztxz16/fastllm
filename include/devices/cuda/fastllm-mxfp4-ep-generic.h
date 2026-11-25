#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

struct FastllmGenericEp;
bool FastllmGenericEpSupported(const std::vector<int> &devices);
std::shared_ptr<FastllmGenericEp> FastllmCreateGenericEp(const std::vector<int> &devices, int hidden,
                                                         int intermediate, int experts, int topk);
// The caller may release the source buffers after this returns.
void FastllmBindGenericEp(FastllmGenericEp &ep, int rank, int layer, const std::vector<uint64_t> &gateDown);
// Shared decode borrows native FP4 values for the lifetime of the state.
bool FastllmGenericEpTunedEp8(const std::vector<int> &devices);
std::shared_ptr<FastllmGenericEp> FastllmCreateSharedEp8Decode(const std::vector<int> &devices);
// Auto mode also shares native FP4 values for dense prefill on paired NUMA ranks.
std::shared_ptr<FastllmGenericEp> FastllmCreateSharedEp8(const std::vector<int> &devices, bool &densePrefill);
void FastllmBindGenericEp(FastllmGenericEp &ep, int rank, int layer, const std::vector<uint64_t> &gateDown,
                          const std::array<void *, 2> &sharedWeights);
void FastllmRunGenericEp(FastllmGenericEp &ep, int rank, int layer, const void *input, const int32_t *ids,
                         const float *scores, void *output, int rows, int commonRows, const int *liveKeys);
// densePrefixRows is agreed by all ranks and contains only unmasked, real rows.
void FastllmRunGenericEp(FastllmGenericEp &ep, int rank, int layer, const void *input, const int32_t *ids,
                         const float *scores, void *output, int rows, int commonRows, const int *liveKeys,
                         int densePrefixRows);

bool FastllmTryGenericEpRoutedDecode(FastllmGenericEp &ep, int rank, int layer, const void *input,
    const float *probabilities, const float *bias, void *output, const int *live, bool normalize, float scale,
    int rows, int capacity, bool logits);
