#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
namespace fastllm::hy4 {
inline void CheckAt(cudaError_t err, int line, const char *operation) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("HY4 CUDA line ") + std::to_string(line) + " " + operation +
                                 ": " + cudaGetErrorString(err));
}
#define Check(operation) CheckAt((operation), __LINE__, #operation)
} // namespace fastllm::hy4
