#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "utils/utils.h"
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fastllm;

namespace {
void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

float Half(float value) { return half_to_float(float_to_half(value)); }
float Key(int h, int t, int d) {
    return Half(float((t % 31 * 13 + d * 17 + h * 19) % 31 - 15) / 32.0f);
}
float Value(int h, int t, int d, int pattern) {
    return Half((pattern == 0 ? 0.25f : 0.0f) +
                float((t % 61 * 7 + d * 11 + h * 23) % 61 - 30) / 256.0f);
}
float Query(int h, int d, int pattern) {
    if (pattern == 0) return 0.0f;
    if (pattern == 2) return Half(32.0f * Key(h / 6, 3, d));
    return Half(float((h * 23 + d * 37) % 127 - 63) / 64.0f);
}

template <typename F>
void Init(Data &data, int heads, int rows, int dim, int capacity, int device, F value) {
    data.Expansion({heads, capacity, dim});
    data.Resize({heads, rows, dim});
    auto *dst = reinterpret_cast<uint16_t *>(data.cpuData);
    // Poison spare capacity: reading past logical KV or using the wrong head
    // stride must not silently look like additional zero-valued tokens.
    std::fill_n(dst, data.expansionBytes / sizeof(uint16_t),
                float_to_half(std::numeric_limits<float>::quiet_NaN()));
    for (int h = 0; h < heads; ++h) {
        for (int t = 0; t < rows; ++t) {
            for (int d = 0; d < dim; ++d) {
                dst[h * data.strides[0] + t * dim + d] = float_to_half(value(h, t, d));
            }
        }
    }
    data.ToDevice(DataDevice::CUDA, {device}, true);
}

std::vector<uint16_t> Read(const Data &data) {
    std::vector<uint16_t> result(data.Count(0));
    FastllmCudaCopyFromDeviceToHost(result.data(), data.cudaData,
                                   result.size() * sizeof(uint16_t));
    return result;
}

void RunReferenceCase(int dim, int length, int pattern, bool paddedQuery, int device) {
    const int heads = 24, kvHeads = 4;
    Data q(FLOAT16), k(FLOAT16), v(FLOAT16), output(FLOAT16), mask;
    Init(q, heads, 1, dim, paddedQuery ? 3 : 1, device,
         [&](int h, int, int d) { return Query(h, d, pattern); });
    // Deliberately different physical capacities for K and V.
    Init(k, kvHeads, length, dim, length + 17, device, Key);
    Init(v, kvHeads, length, dim, length + (dim == 256 ? 129 : 17), device,
         [&](int h, int t, int d) { return Value(h, t, d, pattern); });
    output.Resize({heads, 1, dim});
    output.dataDevice = DataDevice::CUDA;
    output.dataDeviceIds = {device};
    output.Allocate();
    const float scale = 1.0f / std::sqrt(float(dim));
    Require(FastllmCudaHalfAttention(q, k, v, mask, output, 6, scale, 0),
            "attention failed");
    auto actual = Read(output);

    // Independent double-precision reference for every output element. K and
    // V have coprime periods; count repeated periods exactly to avoid a large
    // CPU matrix product while still checking the full long-KV softmax sum.
    constexpr int period = 31 * 61;
    double maxError = 0.0;
    for (int h = 0; h < heads; ++h) {
        double scores[31], maxScore = -std::numeric_limits<double>::infinity();
        for (int t = 0; t < 31; ++t) {
            double dot = 0.0;
            for (int d = 0; d < dim; ++d) dot += double(Query(h, d, pattern)) * Key(h / 6, t, d);
            scores[t] = dot * scale;
            maxScore = std::max(maxScore, scores[t]);
        }
        double weights[61] = {}, denominator = 0.0;
        for (int t = 0; t < period; ++t) {
            int count = length / period + (t < length % period);
            double weight = count * std::exp(scores[t % 31] - maxScore);
            weights[t % 61] += weight;
            denominator += weight;
        }
        for (int d = 0; d < dim; ++d) {
            double numerator = 0.0;
            for (int t = 0; t < 61; ++t) numerator += weights[t] * Value(h / 6, t, d, pattern);
            double expected = numerator / denominator;
            double value = half_to_float(actual[h * dim + d]);
            double error = std::abs(value - expected);
            maxError = std::max(maxError, error);
            if (!std::isfinite(value) || error > 0.0005 + 0.001 * std::abs(expected)) {
                std::fprintf(stderr, "dim=%d kv=%d pattern=%d h=%d d=%d actual=%.9g expected=%.9g\n",
                             dim, length, pattern, h, d, value, expected);
            }
            Require(std::isfinite(value) && error <= 0.0005 + 0.001 * std::abs(expected),
                    "attention differs from double-precision reference");
        }
    }
    std::printf("dim=%d kv=%d pattern=%d padded_q=%d max_abs=%.9g PASS\n",
                dim, length, pattern, paddedQuery, maxError);
}

void RunVisibilityCase(int rows, int maskType, bool explicitMask, int device) {
    const int length = 4099, dim = 256;
    Data q(FLOAT16), k(FLOAT16), v(FLOAT16), output(FLOAT16), mask(FLOAT16);
    auto zero = [](int, int, int) { return 0.0f; };
    Init(q, 1, rows, dim, rows, device, zero);
    Init(k, 1, length, dim, length, device, zero);
    Init(v, 1, length, dim, length, device,
         [&](int, int t, int) { return t == length - 1 ? 60000.0f : 0.0f; });
    Init(output, 1, rows, dim, rows, device, zero);
    if (explicitMask) {
        Init(mask, 1, rows, length, rows, device,
             [&](int, int, int t) { return t == length - 1 ? 1.0f : 0.0f; });
    }
    Require(FastllmCudaHalfAttention(q, k, v, mask, output, 1, 1.0f / 16, maskType),
            "masked attention failed");
    auto actual = Read(output);
    for (int r = 0; r < rows; ++r) {
        double expected = explicitMask || (maskType == 0 && r < rows - 1)
                              ? 0.0 : 60000.0 / length;
        for (int d = 0; d < dim; ++d) {
            double value = half_to_float(actual[r * dim + d]);
            Require(std::isfinite(value) && std::abs(value - expected) < 0.02,
                    "attention changed causal or explicit-mask visibility");
        }
    }
    std::printf("rows=%d mask_type=%d explicit_mask=%d PASS\n", rows, maskType, explicitMask);
}
}

int main(int argc, char **argv) {
    try {
        bool longContext = false;
        int device = 0;
        for (int i = 1; i < argc; ++i) {
            std::string arg(argv[i]);
            if (arg == "--long") longContext = true;
            else if (arg == "--device=1") device = 1;
            else throw std::runtime_error("unknown argument");
        }
        if (FastllmCudaGetDeviceCount() <= device) return 77;
        FastllmCudaSetDevice(device);
        if (!FastllmCudaFlashInferSupported()) return 77;
        int maxSharedMemory = 0;
        Require(cudaDeviceGetAttribute(&maxSharedMemory,
                    cudaDevAttrMaxSharedMemoryPerBlockOptin, device) == cudaSuccess,
                "could not query shared memory capacity");
        // This reference tests split attention's FP32 accumulation. The old
        // FP16 PV fallback on 64 KiB devices has different rounding error.
        if (maxSharedMemory < 64 * 1024 + 512) return 77;
        SetThreads(2);
        for (int length : (longContext ? std::vector<int>{163840, 163841, 196608, 196609, 196615}
                                       : std::vector<int>{4097, 4098, 8193})) {
            for (int pattern = 0; pattern < 3; ++pattern) {
                RunReferenceCase(256, length, pattern, true, device);
            }
        }
        if (!longContext) {
            RunReferenceCase(256, 4095, 1, false, device);
            RunReferenceCase(256, 4096, 1, false, device);
            RunReferenceCase(128, 8193, 1, false, device);
            RunVisibilityCase(1, 0, true, device);
            RunVisibilityCase(2, 0, false, device);
            RunVisibilityCase(2, 2, false, device);
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
