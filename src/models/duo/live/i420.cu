#include <cuda_runtime.h>
#include <cstdint>
namespace {
__global__ void convert(const uint8_t *rgb, uint8_t *planes, int w, int h) {
    const int x = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    const int y = 2 * (blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= w || y >= h)
        return;
    const int pixels = w * h;
    rgb += size_t(blockIdx.z) * pixels * 3;
    planes += size_t(blockIdx.z) * pixels * 3 / 2;
    int red = 0, green = 0, blue = 0;
    for (int dy = 0; dy < 2; ++dy)
        for (int dx = 0; dx < 2; ++dx) {
            int pos = (y + dy) * w + x + dx;
            const auto *p = rgb + pos * 3;
            red += p[0];
            green += p[1];
            blue += p[2];
            planes[pos] = 16 + ((66 * p[0] + 129 * p[1] + 25 * p[2] + 128) >> 8);
        }
    int uv = (y / 2) * (w / 2) + x / 2;
    planes[pixels + uv] = 128 + ((-38 * red - 74 * green + 112 * blue + 512) >> 10);
    planes[pixels + pixels / 4 + uv] = 128 + ((112 * red - 94 * green - 18 * blue + 512) >> 10);
}
} // namespace
extern "C" void duo_rgb_to_i420(const uint8_t *rgb, uint8_t *out, int w, int h, int frames,
                                cudaStream_t stream) {
    convert<<<dim3((w + 31) / 32, (h + 15) / 16, frames), dim3(16, 8), 0, stream>>>(rgb, out, w, h);
}
