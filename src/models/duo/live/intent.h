#pragma once
#include "weights.h"
#include <filesystem>

namespace fastllm::duo::live {
struct Visual {
    int rows, cols;
    // One entry per temporal pair. Each contains the primary embedding and three deepstack tensors.
    std::vector<std::array<native::Tensor, 4>> pairs;
};
class Intent {
  public:
    Intent(const std::string &model, int device, bool cachePrefill = false,
           const std::filesystem::path &cache = {});
    void writeCache(const std::string &path) const;
    void reset();
    using HiddenReady = std::function<void(const native::Tensor &)>;
    native::Tensor forward(const std::vector<int> &ids, const Visual *visual = nullptr, bool logits = true,
                           const HiddenReady &hiddenReady = {});
    int nextToken() const;
    int length() const { return past; }
    int position() const { return nextPosition; }
    void rewind(int length, int position);

  private:
    Weights base, adapter;
    std::string identity;
    native::Tensor logits, frequency, dynamicMap;
    std::array<native::Tensor, 64> keys, values;
    struct Quant {
        native::Tensor packed, scales;
        int rows, cols;
        native::Tensor dense;
    };
    std::map<std::string, Quant> quant;
    size_t prefillBudget = 0, prefillBytes = 0;
    int past = 0, nextPosition = 0, capacity = 0;
    native::Tensor project(const native::Tensor &x, const std::string &name);
    void compress(const std::string &name, native::Tensor &weight);
};
} // namespace fastllm::duo::live
