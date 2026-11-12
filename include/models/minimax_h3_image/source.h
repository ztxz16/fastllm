#pragma once
#include <string>
namespace fastllm::minimax_h3_image {
enum class WeightFormat { BF16, FP8, INT8, NVFP4 };
const char *weightFormatName(WeightFormat);
// Matrix payloads and quantization metadata determine the format, never filenames.
WeightFormat inspectWeightFormat(const std::string &path);
struct ModelSource {
    std::string directory;
    std::string quantizedWeights;
    WeightFormat format = WeightFormat::BF16;
};
// A complete native model directory, or a supported quantized DiT file.
// Quantized repositories can link their complete native base as "base_model".
ModelSource resolveModelSource(const std::string &path);
} // namespace fastllm::minimax_h3_image
