#pragma once
#include "weights.h"
namespace fastllm::duo::live {
class Aligner {
  public:
    Aligner(const std::string &root, int device);
    // The caller supplies real 16 kHz mono samples and the recognized words' BPE ids.
    std::vector<int> align(const std::vector<float> &pcm, const std::vector<std::vector<int>> &words);

  private:
    Weights weights;
    at::Tensor mel(const std::vector<float> &pcm);
    at::Tensor encode(const at::Tensor &features);
};
} // namespace fastllm::duo::live
