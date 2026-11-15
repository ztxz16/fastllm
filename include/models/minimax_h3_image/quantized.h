#pragma once
#include "models/minimax_h3/runtime.h"
#include "models/minimax_h3_image/source.h"
#include <stdexcept>

namespace fastllm::minimax_h3_image {
// TP slices checkpoint bytes before upload for both image and video models.
// The video adapter validates its mixed projections separately; image readers
// retain their format-specific validation and QKV/FFN layout conversions.
struct ProjectionOptions {
    bool comfyVideo = false;
    int rank = 0, size = 1;
    // Optional global activation range/scale across TP/SP ranks.
    std::function<void(minimax_h3::Tensor &)> activationMaximum;
    // Set by the Qwen adapter after validating the checkpoint architecture.
    bool qwenImage = false;
    // Optional signed INT32 TP reduction, followed by scaling/rounding once.
    std::function<void(const minimax_h3::Tensor &, const minimax_h3::Tensor &,
                       const minimax_h3::Tensor &, minimax_h3::Tensor &)> rowReduction;
};
// One instance belongs to one runtime/stream. QKV is always [all Q, all K, all
// V]. Allocations follow the caller's TensorPool and release with the model.
class QuantizedWeights {
  public:
    virtual ~QuantizedWeights() = default;
    static std::unique_ptr<QuantizedWeights> load(const std::string &path);
    static std::unique_ptr<QuantizedWeights> load(const std::string &path, WeightFormat,
                                                 const ProjectionOptions &);
    static void validateDevice(WeightFormat, int device, bool multipleDevices = false);
    virtual void validateModel() const = 0;
    virtual minimax_h3::Tensor linear(const minimax_h3::Tensor &, const std::string &,
                                      minimax_h3::Tensor *output = nullptr) = 0;
    virtual void erasePrefix(const std::string &) = 0;
    virtual void clear() = 0;
    // Direct checkpoint names, without H3-specific permutations. Adapters
    // validate their own model layout before using this optional interface.
    virtual minimax_h3::Tensor linearPacked(const minimax_h3::Tensor &,
        const std::vector<std::string> &, minimax_h3::Tensor * = nullptr) {
        throw std::logic_error("direct packed projections are unavailable for this format");
    }
    virtual size_t loadedBytes() const {
        throw std::logic_error("resident byte accounting is unavailable for this format");
    }
    // Adapters with a reusable conditioning prefix can retain conservative
    // GGUF dispatch while building it. Other formats have no phase policy.
    virtual void setPrefill(bool) {}
    WeightFormat format() const { return format_; }

  protected:
    explicit QuantizedWeights(WeightFormat format) : format_(format) {}

  private:
    WeightFormat format_;
};
} // namespace fastllm::minimax_h3_image
