#pragma once
#include "fastllm.h"
#include <functional>
#include <memory>
#include <vector>

// Private transport for ordinary canonical GGUF disk decode. CPU accessors
// below never call CUDA; the submitting thread owns all GPU operations.
struct FastllmCudaDiskMoeAsyncLayout {
    int gateType, downType, hidden, inter, experts;
    size_t gateBytes, recordBytes;
};
class FastllmCudaDiskMoeAsync {
    struct Impl;
    std::unique_ptr<Impl> impl;
    explicit FastllmCudaDiskMoeAsync(std::unique_ptr<Impl> state);
public:
    struct HostRequest {
        const float *input;
        const int32_t *ids;
        const float *scores;
        const int32_t *gpu;
        float *output;
    };
    static std::unique_ptr<FastllmCudaDiskMoeAsync> Create(
        const std::vector<FastllmCudaDiskMoeAsyncLayout> &layers, int device);
    ~FastllmCudaDiskMoeAsync();
    const uint8_t **Records(int layer);
    unsigned Published(int layer) const;
    bool Failed() const;
    void ResetFailure();
    HostRequest Request(int layer) const;
    void Complete(int layer, unsigned sequence);
    bool CanRun(int layer, const fastllm::Data &input, const fastllm::Data &index,
                const fastllm::Data &score, const fastllm::Data &output) const;
    bool Run(int layer, const fastllm::Data &input, const fastllm::Data &index,
             const fastllm::Data &score, fastllm::Data &output, unsigned sequence,
             const std::function<void()> &sharedExpert);
};
