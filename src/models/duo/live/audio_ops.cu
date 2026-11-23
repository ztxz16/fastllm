#include "tensor.h"
#include <cufft.h>
#include <cub/device/device_reduce.cuh>

namespace fastllm::duo::native {
namespace {
constexpr int threads = 256;
int grid(int64_t n) {
    return int(std::min<int64_t>(65535, (n + 255) / 256));
}
void fftCheck(cufftResult status) {
    if (status != CUFFT_SUCCESS)
        throw std::runtime_error("Duo spectrogram FFT failed: " + std::to_string(status));
}
__global__ void window(const float *pcm, float *output, int samples, int frames) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < frames * 400; i += blockDim.x * gridDim.x) {
        int offset = i % 400, src = i / 400 * 160 - 200 + offset;
        if (src < 0)
            src = -src;
        if (src >= samples)
            src = 2 * samples - 2 - src;
        output[i] = pcm[src] * (.5f - .5f * cosf(6.283185307179586f * offset / 400));
    }
}
__global__ void spectrum(const float2 *fft, float *out, int frames) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < frames * 201; i += blockDim.x * gridDim.x) {
        auto z = fft[i];
        out[i / 201 + frames * (i % 201)] = z.x * z.x + z.y * z.y;
    }
}
__global__ void patches(const __nv_bfloat16 *x, __nv_bfloat16 *out, int batch, int channels, int h, int w,
                        int oh, int ow) {
    const int cols = channels * 9;
    int64_t count = int64_t(batch) * oh * ow * cols;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        int col = i % cols;
        int row = i / cols;
        int y = (row / ow % oh) * 2 + col / 3 % 3 - 1, xx = (row % ow) * 2 + col % 3 - 1;
        out[i] = y < 0 || y >= h || xx < 0 || xx >= w
                     ? __float2bfloat16(0)
                     : x[((row / (oh * ow) * channels + col / 9) * h + y) * w + xx];
    }
}
__global__ void requantize(float *values, const float *total, const float *map, int count) {
    __shared__ float maxima[256];
    float mean = total[0] / count;
    for (int group = blockIdx.x; group < count / 256; group += gridDim.x) {
        int i = group * 256 + threadIdx.x;
        float centered = values[i] - mean;
        maxima[threadIdx.x] = fabsf(centered);
        __syncthreads();
        for (int step = 128; step; step /= 2) {
            if (threadIdx.x < step)
                maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + step]);
            __syncthreads();
        }
        float largest = maxima[0], v = centered / fmaxf(largest, 1.e-30f);
        int lo = 0, hi = 255;
        while (lo < hi) {
            int mid = (hi + lo) / 2;
            if (v > (map[mid] + map[mid + 1]) * .5f)
                lo = mid + 1;
            else
                hi = mid;
        }
        values[i] = map[lo] * largest + mean;
        __syncthreads();
    }
}
} // namespace
Tensor melPower(const std::vector<float> &pcm, int device) {
    DeviceGuard guard(device);
    int frames = pcm.size() / 160;
    auto input = from_blob((void *)pcm.data(), {int64_t(pcm.size())}, kFloat).to(device);
    auto tmp = empty({frames, 400}, input.options()), fft = empty({frames, 201, 2}, input.options()),
         power = empty({201, frames}, input.options());
    window<<<grid(tmp.numel()), threads>>>(input.data_ptr<float>(), tmp.data_ptr<float>(), pcm.size(),
                                           frames);
    check(cudaGetLastError());
    struct Plan {
        cufftHandle h{};
        ~Plan() {
            if (h)
                cufftDestroy(h);
        }
    } plan;
    fftCheck(cufftPlan1d(&plan.h, 400, CUFFT_R2C, frames));
    fftCheck(cufftExecR2C(plan.h, tmp.data_ptr<float>(), fft.data_ptr<float2>()));
    spectrum<<<grid(power.numel()), threads>>>(fft.data_ptr<float2>(), power.data_ptr<float>(), frames);
    check(cudaGetLastError());
    return power;
}
Tensor conv2dStride2(const Tensor &input, const Tensor &w, const Tensor &bias) {
    if (input.dim() != 4 || w.dim() != 4 || w.size(2) != 3 || w.size(3) != 3 || input.size(1) != w.size(1) ||
        input.scalar_type() != kBFloat16)
        throw std::invalid_argument("invalid aligner 3x3 convolution");
    DeviceGuard guard(input.device());
    const int b = input.size(0), c = input.size(1), h = input.size(2), width = input.size(3),
              oh = (h + 1) / 2, ow = (width + 1) / 2;
    auto x = input.contiguous(), cols = empty({b, oh, ow, c * 9}, x.options());
    patches<<<grid(cols.numel()), threads>>>(x.data_ptr<__nv_bfloat16>(), cols.data_ptr<__nv_bfloat16>(), b,
                                             c, h, width, oh, ow);
    check(cudaGetLastError());
    return linear(cols, w.reshape({w.size(0), c * 9}), bias).permute({0, 3, 1, 2});
}
void quantizeScales(Tensor &scales, const Tensor &map) {
    if (scales.scalar_type() != kFloat || scales.numel() % 256 || map.numel() != 256 ||
        scales.numel() > INT_MAX)
        throw std::invalid_argument("invalid NF4 scale groups");
    DeviceGuard guard(scales.device());
    auto sum = empty({1}, scales.options());
    size_t bytes = 0;
    int count = scales.numel();
    check(cub::DeviceReduce::Sum(nullptr, bytes, scales.data_ptr<float>(), sum.data_ptr<float>(), count));
    auto tmp = empty({int64_t(bytes)}, scales.options().dtype(kByte));
    check(cub::DeviceReduce::Sum(tmp.data_ptr(), bytes, scales.data_ptr<float>(), sum.data_ptr<float>(),
                                 count));
    requantize<<<std::min(count / 256, 65535), 256>>>(scales.data_ptr<float>(), sum.data_ptr<float>(),
                                                      map.data_ptr<float>(), count);
    check(cudaGetLastError());
}
} // namespace fastllm::duo::native
