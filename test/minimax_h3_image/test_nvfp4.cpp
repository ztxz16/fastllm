#include "json11.hpp"
#include "models/minimax_h3_image/quantized.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
using namespace fastllm::minimax_h3;
using namespace fastllm::minimax_h3_image;
namespace fs = std::filesystem;
void require(bool ok) {
    if (!ok)
        throw std::runtime_error("NVFP4 fixture assertion failed");
}
float bf16(float x) {
    uint32_t b;
    std::memcpy(&b, &x, 4);
    b += 32767 + ((b >> 16) & 1);
    b &= 0xffff0000;
    std::memcpy(&x, &b, 4);
    return x;
}
float fp4(int code) {
    const float levels[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    return (code & 8 ? -1 : 1) * levels[code & 7];
}
float fp8(int code) {
    int exponent = code >> 3, mantissa = code & 7;
    return exponent ? std::ldexp(1 + mantissa / 8.f, exponent - 7) : std::ldexp(float(mantissa), -9);
}
int nearest4(float x) {
    int best = 0;
    float distance = std::abs(x);
    for (int i = 1; i < 8; ++i) {
        float d = std::abs(std::abs(x) - fp4(i));
        if (d < distance || (d == distance && !(i & 1))) {
            best = i;
            distance = d;
        }
    }
    return best + (std::signbit(x) ? 8 : 0);
}
int nearest8(float x) {
    int best = 0;
    float distance = std::abs(x);
    for (int i = 1; i <= 126; ++i) {
        float d = std::abs(x - fp8(i));
        if (d < distance || (d == distance && !(i & 1))) {
            best = i;
            distance = d;
        }
    }
    return best;
}
size_t scaleIndex(int row, int group, int k) {
    return (size_t(row / 128) * (k / 64) + group / 4) * 512 + (row % 32) * 16 + (row % 128 / 32) * 4 +
           group % 4;
}
float weight(int row, int col) {
    return fp4((row * 7 + col * 11) % 16) * std::ldexp(1.f, (row + col / 16) % 4 - 3);
}
enum class Invalid { None, Shape, GlobalScale, GroupSize };
void fixture(const fs::path &path, int k, Invalid invalid = Invalid::None) {
    constexpr int n = 128;
    json11::Json::object h;
    std::vector<char> data;
    auto add = [&](std::string suffix, std::string dtype, json11::Json::array shape, const void *p,
                   size_t bytes) {
        size_t offset = data.size();
        const char *a = static_cast<const char *>(p);
        data.insert(data.end(), a, a + bytes);
        h["blocks.0.mlp.fc2" + suffix] =
            json11::Json::object{{"dtype", dtype},
                                 {"shape", shape},
                                 {"data_offsets", json11::Json::array{double(offset), double(data.size())}}};
    };
    std::vector<uint8_t> q(size_t(n) * k / 2), s(size_t(n) * k / 16);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < k; col += 2)
            q[(size_t(row) * k + col) / 2] =
                ((row * 7 + col * 11) % 16) << 4 | ((row * 7 + (col + 1) * 11) % 16);
        for (int group = 0; group < k / 16; ++group)
            s[scaleIndex(row, group, k)] = nearest8(std::ldexp(1.f, (row + group) % 4 - 3));
    }
    float global = invalid == Invalid::GlobalScale ? std::nanf("") : .125f;
    add(".weight", "U8", {n, k / 2}, q.data(), q.size());
    add(".weight_scale", "F8_E4M3", {n, k / 16}, s.data(), s.size());
    add(".weight_scale_2", "F32", {}, &global, 4);
    auto layer =
        json11::Json::object{{"format", "nvfp4"},
                             {"group_size", invalid == Invalid::GroupSize ? 16.5 : 16},
                             {"orig_shape", json11::Json::array{invalid == Invalid::Shape ? n * 2 : n, k}}};
    auto metadata = json11::Json::object{{"layers", json11::Json::object{{"blocks.0.mlp.fc2", layer}}}};
    h["__metadata__"] = json11::Json::object{{"_quantization_metadata", json11::Json(metadata).dump()}};
    auto text = json11::Json(h).dump();
    uint64_t bytes = text.size();
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<char *>(&bytes), 8);
    f << text;
    f.write(data.data(), data.size());
}
int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count)
        return 77;
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major != 12 || prop.minor != 0)
        return 77;
    try {
        QuantizedWeights::validateDevice(WeightFormat::NVFP4, 0);
    } catch (const std::exception &e) {
        std::cout << e.what() << '\n';
        return 77;
    }
    auto root = fs::temp_directory_path() / ("h3-nvfp4-" + std::to_string(getpid()));
    fs::create_directories(root);
    try {
        Runtime rt(0, DType::BF16);
        for (int k : {5376, 7168, 14336}) {
            auto file = root / "weights.safetensors";
            fixture(file, k);
            auto weights = QuantizedWeights::load(file.string());
            require(weights->format() == WeightFormat::NVFP4);
            std::vector<float> w(size_t(128) * k);
            for (int row = 0; row < 128; ++row)
                for (int c = 0; c < k; ++c)
                    w[size_t(row) * k + c] = weight(row, c);
            for (int m : {1, 29, 129}) {
                std::vector<float> x(size_t(m) * k), q(x.size());
                float amax = 0;
                for (size_t i = k; i < x.size(); ++i) {
                    x[i] = bf16((int(i * 17 % 2001) - 1000) / 128.f);
                    amax = std::max(amax, std::abs(x[i]));
                }
                if (m > 1) {
                    x[k] = 9984;
                    amax = 9984;
                }
                float global = std::max(amax / 2688.f, 1e-12f);
                for (size_t i = 0; i < x.size(); i += 16) {
                    float largest = 0;
                    for (int j = 0; j < 16; ++j)
                        largest = std::max(largest, std::abs(x[i + j]));
                    float scale = fp8(nearest8(largest / (6.f * global))),
                          inv = scale > 0 ? 1.f / (scale * global) : 0;
                    for (int j = 0; j < 16; ++j)
                        q[i + j] = fp4(nearest4(x[i + j] * inv)) * scale;
                }
                auto input = rt.upload(x.data(), m, k);
                for (int repeat = 0; repeat < 2; ++repeat) {
                    Tensor target(m, 128);
                    auto actual =
                        rt.download(weights->linear(input, "blocks.0.mlp.fc2", repeat ? &target : nullptr));
                    double err = 0, norm = 0;
                    for (int row = 0; row < m; ++row)
                        for (int n = 0; n < 128; ++n) {
                            float sum = 0;
                            for (int c = 0; c < k; ++c)
                                sum += q[size_t(row) * k + c] * w[size_t(n) * k + c];
                            float expected = bf16(sum * (global * .125f)),
                                  value = actual[size_t(row) * 128 + n];
                            require(std::isfinite(value));
                            err += double(value - expected) * (value - expected);
                            norm += double(expected) * expected;
                        }
                    double l2 = std::sqrt(err / std::max(norm, 1e-30));
                    std::cout << "k=" << k << " m=" << m << " repeat=" << repeat << " relative_l2=" << l2
                              << '\n';
                    require(l2 < .001);
                }
            }
            weights->erasePrefix("blocks.");
            weights->clear();
        }
        for (auto invalid : {Invalid::Shape, Invalid::GlobalScale, Invalid::GroupSize}) {
            fixture(root / "bad.safetensors", 5376, invalid);
            auto w = QuantizedWeights::load((root / "bad.safetensors").string());
            std::vector<float> x(5376);
            bool rejected = false;
            try {
                w->linear(rt.upload(x.data(), 1, 5376), "blocks.0.mlp.fc2");
            } catch (const std::exception &) {
                rejected = true;
            }
            require(rejected);
        }
        fs::remove_all(root);
        std::cout << "native NVFP4 passed\n";
    } catch (const std::exception &e) {
        fs::remove_all(root);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
