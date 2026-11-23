#pragma once
#include "tensor.h"
#include <map>
#include <string>
#include <vector>

namespace fastllm::duo::live {
// Read the original shards into CUDA storage, converting each tensor independently.
// Only selected components are materialized. No Python model loader is involved.
class Weights {
  public:
    explicit Weights(int device) : device(device) {}
    void load(const std::string &file, const std::vector<std::string> &prefixes,
              native::Type precision = native::kBFloat16,
              const std::function<void(const std::string &, native::Tensor &)> &transform = {});
    const native::Tensor &get(const std::string &name) const;
    native::Tensor linear(const native::Tensor &x, const std::string &name) const;
    int device;
    std::map<std::string, native::Tensor> tensors;
};
} // namespace fastllm::duo::live
