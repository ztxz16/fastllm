#pragma once
#include "devices/multicuda/fastllm-draft-verify.h"
std::shared_ptr<void> FastllmCudaCreateDraftHead120(const std::vector<int> &devices,
                                                    const std::vector<const fastllm::Data *> &weights,
                                                    const fastllm::Data &w1, const fastllm::Data &w2);
struct FastllmDraftContextWeights120 {
    const fastllm::Data *fc = nullptr, *gamma = nullptr;
    const fastllm::Data *qkv[5]{}, *kNorm[5]{};
};
struct FastllmDraftContextInput120 {
    const void *feature = nullptr;
    void *cache[10]{};
    void *nextBindings = nullptr;
    int anchor = -1;
    FastllmDraftSamplingRequest request{};
};
std::shared_ptr<void> FastllmCudaCreateDraftContext120(const std::vector<int> &devices,
                                                       const FastllmDraftContextWeights120 (&weights)[8]);
void FastllmCudaCommitDraftContext120(void *state, int rank, const FastllmDraftContextInput120 &input,
                                      int count, int old, int position, float eps, float theta);
bool FastllmCudaDraftFeaturesReady120();
