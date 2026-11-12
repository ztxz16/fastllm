#pragma once
#include "models/minimax_h3_image/quantized.h"

namespace fastllm::minimax_h3_image {
// Native E4M3 DiT projections. An instance belongs to one CUDA runtime/stream;
// QKV output is always packed as [all Q, all K, all V].
// Its buffers use the caller's active TensorPool and release with the model.
class Fp8Weights final : public QuantizedWeights {
  public:
    explicit Fp8Weights(const std::string &path);
    ~Fp8Weights() override;
    Fp8Weights(const Fp8Weights &) = delete;
    Fp8Weights &operator=(const Fp8Weights &) = delete;
    static void validateDevice(int device, bool multipleDevices = false);
    void validateModel() const override;
    minimax_h3::Tensor linear(const minimax_h3::Tensor &, const std::string &name,
                              minimax_h3::Tensor *output = nullptr) override;
    void erasePrefix(const std::string &) override;
    void clear() override;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace fastllm::minimax_h3_image
