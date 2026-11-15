#pragma once
#include "models/minimax_h3_image/quantized.h"
namespace fastllm::minimax_h3_image::detail {
std::unique_ptr<QuantizedWeights> loadInt8(const std::string &, const ProjectionOptions &);
std::unique_ptr<QuantizedWeights> loadNvfp4(const std::string &, const ProjectionOptions &);
std::unique_ptr<QuantizedWeights> loadGguf(const std::string &, const ProjectionOptions &);
void validateGgufDevice(int);
bool ggufMmq(const minimax_h3::Tensor &input, const minimax_h3::Tensor &packed,
             minimax_h3::Tensor &output, uint32_t type);
void validateInt8Device(int);
void validateNvfp4Device(int);
} // namespace fastllm::minimax_h3_image::detail
