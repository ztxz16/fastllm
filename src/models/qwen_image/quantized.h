#pragma once
#include "models/qwen_image/runtime.h"
#include <memory>

namespace fastllm::qwen_image {
// Format-aware DiT adapter for TorchAO FP8, Comfy ConvRot INT8 and GGUF.
// Text/vision and VAE continue to use their original checkpoint components.
class QuantizedProjections {
  public:
    static std::unique_ptr<QuantizedProjections> load(Runtime &, const std::string &checkpoint,
                                                        nomas::ParallelContext = {});
    ~QuantizedProjections();
    bool isGguf() const;
    const Tensor &get(const std::string &);
    bool contains(const std::string &) const;
    void setPrefill(bool);
    Tensor linear(const Tensor &, const std::vector<std::string> &, Tensor *output = nullptr);
    Tensor row(const Tensor &, const std::string &);
    void clear();
    size_t loadedBytes() const;

  private:
    struct Impl;
    explicit QuantizedProjections(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
};
} // namespace fastllm::qwen_image
