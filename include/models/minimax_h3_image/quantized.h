#pragma once
#include "models/minimax_h3/runtime.h"
#include "models/minimax_h3_image/source.h"

namespace fastllm::minimax_h3_image {
// One instance belongs to one runtime/stream. QKV is always [all Q, all K, all
// V]. Allocations follow the caller's TensorPool and release with the model.
class QuantizedWeights {
  public:
    virtual ~QuantizedWeights() = default;
    static std::unique_ptr<QuantizedWeights> load(const std::string &path);
    static void validateDevice(WeightFormat, int device, bool multipleDevices = false);
    virtual void validateModel() const = 0;
    virtual minimax_h3::Tensor linear(const minimax_h3::Tensor &, const std::string &,
                                      minimax_h3::Tensor *output = nullptr) = 0;
    virtual void erasePrefix(const std::string &) = 0;
    virtual void clear() = 0;
    WeightFormat format() const { return format_; }

  protected:
    explicit QuantizedWeights(WeightFormat format) : format_(format) {}

  private:
    WeightFormat format_;
};
} // namespace fastllm::minimax_h3_image
