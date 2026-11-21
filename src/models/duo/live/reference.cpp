#include "reference.h"
#include "models/minimax_h3/runtime.h"
#include "tables.h"
#include <cuda_bf16.h>
#include <cmath>
#include <stdexcept>

namespace fastllm::duo::live {
std::string referencePreset(int width, int height) {
    return "presets/default-" + std::to_string(width) + "x" + std::to_string(height) + ".bf16";
}

std::string encodeReference(const std::string &model, const std::string &rgb, int width, int height) {
    if (!((width == 512 && height == 384) || (width == 256 && height == 192)) ||
        rgb.size() != size_t(width) * height * 3)
        throw std::invalid_argument("invalid Duo reference RGB geometry");
    constexpr float mean[] = {.485f, .456f, .406f}, deviation[] = {.229f, .224f, .225f};
    minimax_h3::VideoTensor pixels(1, height, width, 3);
    for (size_t i = 0; i < rgb.size(); ++i)
        pixels.values[i] = (static_cast<unsigned char>(rgb[i]) / 255.f - mean[i % 3]) / deviation[i % 3];

    minimax_h3::Runtime rt(0, nomas::DType::FP32);
    minimax_h3::VideoVAE encoder(rt, rt, model + "/codec/video_vae");
    const auto moments = encoder.encodeMoments(pixels);
    // The official reference takes the posterior mean, normalizes each channel
    // and packs each 2x2 patch in (channel, dy, dx) order.
    std::vector<__nv_bfloat16> packed;
    packed.reserve(size_t(moments.height) * moments.width * 24);
    for (int y = 0; y < moments.height; y += 2)
        for (int x = 0; x < moments.width; x += 2)
            for (int c = 0; c < 24; ++c)
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx) {
                        const size_t cell = size_t(y + dy) * moments.width + x + dx;
                        float value = (moments.values[cell * 48 + c] - constants::video_latents_mean[c]) /
                                      constants::video_latents_std[c];
                        if (!std::isfinite(value))
                            throw std::runtime_error("non-finite Duo reference latent");
                        packed.emplace_back(value);
                    }
    return {reinterpret_cast<const char *>(packed.data()), packed.size() * sizeof(packed[0])};
}
} // namespace fastllm::duo::live
