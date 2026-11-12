#pragma once
#include "models/minimax_h3_image/quantized.h"
namespace fastllm::minimax_h3_image::detail {
std::unique_ptr<QuantizedWeights> loadInt8(const std::string &);
std::unique_ptr<QuantizedWeights> loadNvfp4(const std::string &);
void validateInt8Device(int);
void validateNvfp4Device(int);
} // namespace fastllm::minimax_h3_image::detail
