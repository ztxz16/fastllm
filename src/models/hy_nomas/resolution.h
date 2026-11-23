#pragma once

namespace fastllm { namespace nomas {
constexpr int maxPixelSide = 1280;
constexpr int maxPixelArea = 1280 * 704;
// PixelUMM decodes up to ten latent frames on a 16-pixel grid.
constexpr int maxGridTokens = 10 * (maxPixelArea / (16 * 16));
constexpr int maxCacheTokens = 65536;

inline bool validResolution(int width, int height) {
    return width >= 64 && width <= maxPixelSide && height >= 64 && height <= maxPixelSide &&
           width % 32 == 0 && height % 32 == 0 && width * height <= maxPixelArea;
}
}} // namespace fastllm::nomas
