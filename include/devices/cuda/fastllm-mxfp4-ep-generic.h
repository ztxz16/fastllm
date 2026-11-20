#pragma once

#include <cstdint>
#include <memory>
#include <vector>

struct FastllmGenericEp;
bool FastllmGenericEpSupported(const std::vector<int> &devices);
std::shared_ptr<FastllmGenericEp> FastllmCreateGenericEp(const std::vector<int> &devices, int hidden,
                                                         int intermediate, int experts, int topk);
// The caller may release the source buffers after this returns.
void FastllmBindGenericEp(FastllmGenericEp &ep, int rank, int layer, const std::vector<uint64_t> &gateDown);
void FastllmRunGenericEp(FastllmGenericEp &ep, int rank, int layer, const void *input, const int32_t *ids,
                         const float *scores, void *output, int rows, int commonRows, const int *liveKeys);
