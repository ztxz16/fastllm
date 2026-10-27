#pragma once

#include "fastllm-cuda.cuh"
#include "executor.h"
#define GGML_COMMON_DECL_CUDA
#include "gguf.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace fastllm;
static void Check(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
static void Cuda(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
static void Allocate(Data &data) {
    data.dataDevice = DataDevice::CUDA;
    data.dataDeviceIds = {0};
    data.Allocate();
}
template <class T> static void Upload(Data &data, const std::vector<T> &values) {
    Cuda(cudaMemcpy(data.cudaData, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
}
static std::vector<half> Download(const Data &data, size_t count) {
    std::vector<half> values(count);
    Cuda(cudaMemcpy(values.data(), data.cudaData, count * sizeof(half), cudaMemcpyDeviceToHost));
    return values;
}

#include "cuda_gguf_weights_test.cuh"
