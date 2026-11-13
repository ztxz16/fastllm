#include "models/minimax_h3_image/quantized.h"
#include "models/minimax_h3_image/fp8.h"
#include "quantized_backends.h"
#include <stdexcept>

namespace fastllm::minimax_h3_image {
std::unique_ptr<QuantizedWeights> QuantizedWeights::load(const std::string &path) {
    switch (inspectWeightFormat(path)) {
    case WeightFormat::FP8:
        return std::make_unique<Fp8Weights>(path);
    case WeightFormat::INT8:
        return detail::loadInt8(path);
    case WeightFormat::NVFP4:
        return detail::loadNvfp4(path);
    case WeightFormat::GGUF:
        return detail::loadGguf(path);
    default:
        throw std::invalid_argument("expected an H3 Image quantized checkpoint");
    }
}
void QuantizedWeights::validateDevice(WeightFormat format, int device, bool multipleDevices) {
    switch (format) {
    case WeightFormat::FP8:
        Fp8Weights::validateDevice(device, multipleDevices);
        return;
    case WeightFormat::INT8:
        detail::validateInt8Device(device);
        return;
    case WeightFormat::NVFP4:
        detail::validateNvfp4Device(device);
        return;
    case WeightFormat::GGUF:
        detail::validateGgufDevice(device);
        return;
    default:
        throw std::invalid_argument("expected an H3 Image quantized format");
    }
}
} // namespace fastllm::minimax_h3_image
