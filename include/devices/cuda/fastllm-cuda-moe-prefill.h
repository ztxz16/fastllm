#pragma once

#include <cstdint>
#include <memory>
#include <unordered_set>

namespace fastllm { class Data; class CudaWorkspace; }

// A serialized eager prefill may temporarily borrow empty/evictable cache
// payload slots. Addresses and graph mapping tables remain stable. Call only
// between complete decode steps; release all scratch before the next decode.
std::shared_ptr<fastllm::CudaWorkspace> FastllmCudaBorrowMoeWorkspace(
    fastllm::Data **weights, int weightsBatch, size_t bytes);
std::shared_ptr<fastllm::CudaWorkspace> FastllmCudaGetMoeWorkspace(const fastllm::Data *gate);

// Shared weight transport for host-expert prefill. The caller retains its
// original projection, activation and reduction arithmetic. Resident records
// are borrowed directly or restored on-device to the original NUMA layout;
// misses retain the existing temporary upload and bounded cache admission.
// One instance spans one serialized layer invocation on the current GPU.
class FastllmCudaMoePrefillCache {
public:
    FastllmCudaMoePrefillCache();
    ~FastllmCudaMoePrefillCache();
    FastllmCudaMoePrefillCache(const FastllmCudaMoePrefillCache &) = delete;
    FastllmCudaMoePrefillCache &operator=(const FastllmCudaMoePrefillCache &) = delete;
    bool Begin(fastllm::Data **weights, int experts, const int32_t *indices,
               const float *scores, int rows, int topk,
               const std::unordered_set<int> &selected, bool admit);
    fastllm::Data **Weights();
    void Upload(int slot, void *stream = nullptr);
    void Release(int slot);
    void Store(int expert);
    // Publish only after every selected expert has finished using its weights.
    // An unfinished/exceptional scope never publishes incomplete payloads.
    void Finish();
private:
    struct State;
    std::unique_ptr<State> state;
};
