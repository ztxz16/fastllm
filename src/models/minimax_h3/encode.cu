#include "models/minimax_h3/runtime.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>

namespace fastllm {
namespace minimax_h3 {
namespace {
float *ptr(const Tensor &x) {
    return reinterpret_cast<float *>(x.data);
}
void check() {
    auto status = cudaGetLastError();
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
__device__ int reflect(int x, int n) {
    return x < 0 ? -x : x >= n ? 2 * n - 2 - x : x;
}
// Checkpoints store O,I,T,H,W kernels; activations are contiguous T,H,W,C.
__global__ void causalColumns(const float *x, float *out, int rows, int channels, int height, int width,
                              int strideT, int strideS, int pad, int outH, int outW, int start) {
    int kc = channels * 27;
    // A block shares an output pixel. Keep I,T,H,W ordering for the unchanged GEMM,
    // and avoid dividing a flattened 64-bit index for every copied element.
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        int pixel = start + row;
        int t0 = pixel / (outH * outW) * strideT - 2;
        int y0 = pixel / outW % outH * strideS - pad;
        int x0 = pixel % outW * strideS - pad;
        for (int k = blockIdx.y * blockDim.x + threadIdx.x; k < kc; k += gridDim.y * blockDim.x) {
            int c = k / 27, dt = k / 9 % 3, dy = k / 3 % 3, dx = k % 3;
            int t = t0 + dt, y = reflect(y0 + dy, height), z = reflect(x0 + dx, width);
            out[size_t(row) * kc + k] =
                t < 0 ? 0.f : x[((size_t(t) * height + y) * width + z) * channels + c];
        }
    }
}
// Preserve the original channel-major reduction order, but make its reads contiguous.
__global__ void frameChannels(const float *x, float *out, int pixels, int channels) {
    __shared__ float tile[32][33];
    int c = blockIdx.x * 32 + threadIdx.x, p = blockIdx.y * 32 + threadIdx.y;
    size_t base = size_t(blockIdx.z) * pixels * channels;
    for (int i = 0; i < 32; i += 8)
        if (c < channels && p + i < pixels)
            tile[threadIdx.y + i][threadIdx.x] = x[base + size_t(p + i) * channels + c];
    __syncthreads();
    p = blockIdx.y * 32 + threadIdx.x;
    c = blockIdx.x * 32 + threadIdx.y;
    for (int i = 0; i < 32; i += 8)
        if (p < pixels && c + i < channels)
            out[base + size_t(c + i) * pixels + p] = tile[threadIdx.x][threadIdx.y + i];
}
__global__ void frameGroupStats(const float *x, float *out, int count) {
    __shared__ float sums[256];
    size_t base = size_t(blockIdx.x) * count;
    float sum = 0;
    for (int i = threadIdx.x; i < count; i += blockDim.x)
        sum += x[base + i];
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int d = 128; d; d /= 2) {
        if (threadIdx.x < d)
            sums[threadIdx.x] += sums[threadIdx.x + d];
        __syncthreads();
    }
    float mean = sums[0] / count;
    __syncthreads();
    sum = 0;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        float v = x[base + i] - mean;
        sum += v * v;
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int d = 128; d; d /= 2) {
        if (threadIdx.x < d)
            sums[threadIdx.x] += sums[threadIdx.x + d];
        __syncthreads();
    }
    if (!threadIdx.x) {
        out[blockIdx.x * 2] = mean;
        out[blockIdx.x * 2 + 1] = rsqrtf(sums[0] / count + 1e-6f);
    }
}
__global__ void frameGroupSilu(const float *x, float *out, const float *stats, const float *weight,
                               const float *bias, size_t n, int pixels, int channels) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += size_t(gridDim.x) * blockDim.x) {
        int c = i % channels;
        size_t group = i / (size_t(pixels) * channels) * 32 + c / (channels / 32);
        float v = (x[i] - stats[group * 2]) * stats[group * 2 + 1] * weight[c] + bias[c];
        out[i] = v / (1.f + expf(-v));
    }
}
} // namespace

Tensor VideoVAE::encodeTile(const Tensor &pixels, int frames, int height, int width, Trace trace) {
    if (!pixels.isFloat32() || pixels.cols != 3 || frames < 1 || height < 32 || width < 32 || height % 16 ||
        width % 16 || int64_t(frames) * height * width != pixels.rows)
        throw std::invalid_argument("invalid H3 VAE encoder pixels");
    int t = frames, h = height, w = width;
    auto conv = [&](const Tensor &input, const std::string &name, int kernel, int st = 1, int ss = 1) {
        const auto &weight = fullWeights_.get(name + ".weight");
        const auto *bias = fullWeights_.optional(name + ".bias");
        int pad = kernel == 3 && ss == 1 ? 1 : 0;
        int ot = (t - 1) / st + 1, oh = ss == 2 ? h / 2 : h, ow = ss == 2 ? w / 2 : w;
        int kc = input.cols * kernel * kernel * kernel, rows = ot * oh * ow;
        if (weight.cols != kc || input.rows != t * h * w)
            throw std::runtime_error("H3 encoder convolution shape mismatch: " + name);
        Tensor output(rows, weight.rows, DType::FP32);
        for (int first = 0; first < rows;) {
            // Bound im2col scratch and retain the original sub-512 tail shape.
            int count = std::min(8192, rows - first);
            if (count > 512)
                count -= count % 512;
            Tensor columns;
            if (kernel == 1)
                columns = input.rowSlice(first, count);
            else {
                columns = Tensor(count, kc, DType::FP32);
                dim3 blocks(std::min(count, 2048), std::min((kc + 255) / 256, 4));
                causalColumns<<<blocks, 256, 0, cudaStreamPerThread>>>(
                    ptr(input), ptr(columns), count, input.cols, h, w, st, ss, pad, oh, ow, first);
                check();
            }
            auto target = output.rowSlice(first, count);
            full_.linearInto(columns, weight, target, bias);
            first += count;
        }
        return output;
    };
    auto norm = [&](const Tensor &input, const std::string &name) {
        const auto &weight = fullWeights_.get(name + ".weight");
        const auto &bias = fullWeights_.get(name + ".bias");
        if (input.cols % 32 || weight.size() != size_t(input.cols) || bias.size() != weight.size())
            throw std::runtime_error("H3 encoder normalization shape mismatch: " + name);
        Tensor planar(input.rows, input.cols, DType::FP32), stats(t * 32, 2, DType::FP32);
        Tensor output(input.rows, input.cols, DType::FP32);
        int pixels = h * w;
        frameChannels<<<dim3((input.cols + 31) / 32, (pixels + 31) / 32, t), dim3(32, 8), 0,
                        cudaStreamPerThread>>>(ptr(input), ptr(planar), pixels, input.cols);
        frameGroupStats<<<t * 32, 256, 0, cudaStreamPerThread>>>(ptr(planar), ptr(stats),
                                                                 pixels * (input.cols / 32));
        int blocks = std::min<size_t>((input.size() + 255) / 256, 4096);
        frameGroupSilu<<<blocks, 256, 0, cudaStreamPerThread>>>(
            ptr(input), ptr(output), ptr(stats), ptr(weight), ptr(bias), input.size(), pixels, input.cols);
        check();
        return output;
    };
    auto emit = [&](const std::string &name, const Tensor &x) {
        if (trace)
            trace(name, x);
    };
    auto x = conv(pixels, "encoder.conv_in", 3);
    emit("encoder.conv_in", x);
    for (int block = 0; block < 6; ++block) {
        auto prefix = "encoder.down_blocks." + std::to_string(block);
        for (int layer = 0; layer < 2; ++layer) {
            auto p = prefix + ".resnets." + std::to_string(layer);
            auto skip = x;
            auto y = conv(norm(x, p + ".norm1"), p + ".conv1", 3);
            y = conv(norm(y, p + ".norm2"), p + ".conv2", 3);
            if (fullWeights_.optional(p + ".conv_shortcut.weight"))
                skip = conv(skip, p + ".conv_shortcut", 1);
            x = full_.add(skip, y);
            emit(p, x);
        }
        if (block < 4) {
            int st = block == 1 || block == 2 ? 2 : 1;
            x = conv(x, prefix + ".downsamplers.0.conv", 3, st, 2);
            t = (t - 1) / st + 1;
            h /= 2;
            w /= 2;
        }
        emit(prefix, x);
    }
    x = conv(norm(x, "encoder.norm_out"), "encoder.conv_out", 3);
    emit("encoder.conv_out", x);
    x = conv(x, "quant_conv", 1);
    emit("quant_conv", x);
    return x;
}
} // namespace minimax_h3
} // namespace fastllm
