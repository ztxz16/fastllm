#pragma once
#include <string>
namespace fastllm::minimax_h3_image {
struct ModelSource {
    std::string directory;
    std::string fp8Weights;
};
// A complete native model directory, or a self-describing quantized DiT file.
// Quantized repositories can link their complete native base as "base_model".
ModelSource resolveModelSource(const std::string &path);
} // namespace fastllm::minimax_h3_image
