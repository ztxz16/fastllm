#pragma once
#include <ATen/ATen.h>
#include <map>
#include <string>
#include <vector>

namespace fastllm::duo::live {
// Read the original shards into CUDA storage, converting each tensor independently.
// Only selected components are materialized. No Python model loader is involved.
class Weights {
  public:
    explicit Weights(at::Device device) : device(device) {}
    void load(const std::string &file, const std::vector<std::string> &prefixes,
              at::ScalarType precision = at::kBFloat16,
              const std::function<void(const std::string &, at::Tensor &)> &transform = {});
    const at::Tensor &get(const std::string &name) const;
    at::Tensor linear(const at::Tensor &x, const std::string &name) const;
    at::Device device;
    std::map<std::string, at::Tensor> tensors;
};
} // namespace fastllm::duo::live
