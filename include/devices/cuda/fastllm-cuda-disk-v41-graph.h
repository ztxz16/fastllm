#pragma once
#include "fastllm.h"
#include <memory>

// Fixed operands for V4.1 disk experts. The CPU producer never submits CUDA
// while a replay is waiting for it; resident record addresses are leased.
class FastllmCudaDiskV41Graph {
    struct Impl;
    std::unique_ptr<Impl> impl;
    explicit FastllmCudaDiskV41Graph(std::unique_ptr<Impl>);

public:
    struct Host {
        float *input, *output;
        int32_t *ids, *resident;
        float *scores;
        const uint8_t **records;
    };
    static std::unique_ptr<FastllmCudaDiskV41Graph>
    Create(const fastllm::Data &input, const fastllm::Data &index, const fastllm::Data &score,
           fastllm::Data &output, fastllm::DataType weightType, int gateType, int downType, int inter,
           int experts, size_t gateBytes, size_t recordBytes, float limit);
    ~FastllmCudaDiskV41Graph();
    void *Graph() const;
    Host Request() const;
    void Reset();
    bool Published() const;
    void Complete();
};
