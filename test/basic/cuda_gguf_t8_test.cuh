#pragma once

#include "../../src/devices/cuda/fastllm-gguf-kernel-common.cuh"
#include "../../src/devices/cuda/fastllm-gguf-small-mmvq.cuh"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

static void Check(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
static void Cuda(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
struct Allocation {
    void *p = nullptr;
    explicit Allocation(size_t bytes) { Cuda(cudaMalloc(&p, bytes)); }
    ~Allocation() { cudaFree(p); }
    Allocation(const Allocation &) = delete;
    Allocation &operator=(const Allocation &) = delete;
    template <class T> T *as() { return static_cast<T *>(p); }
};
