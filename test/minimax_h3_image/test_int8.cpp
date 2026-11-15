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
        throw std::runtime_error("INT8 fixture assertion failed");
}
float bf16(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, 4);
    bits += 32767 + ((bits >> 16) & 1);
    bits &= 0xffff0000;
    std::memcpy(&x, &bits, 4);
    return x;
}
// Dense H4^4 is independent of the warp/register implementation.
float h256(int i, int j) {
    int sign = 1;
    for (int digit = 0; digit < 4; ++digit) {
        if ((i & 3) + (j & 3) == 3)
            sign = -sign;
        i >>= 2;
        j >>= 2;
    }
    return sign / 16.f;
}
void fixture(const fs::path &path, int cols, bool unsloth, bool badMarker = false) {
    json11::Json::object header;
    std::vector<char> payload;
    auto add = [&](std::string name, std::string dtype, json11::Json::array shape, const void *data,
                   size_t bytes) {
        size_t start = payload.size();
        const char *p = static_cast<const char *>(data);
        payload.insert(payload.end(), p, p + bytes);
        header[name] = json11::Json::object{
            {"dtype", dtype},
            {"shape", shape},
            {"data_offsets", json11::Json::array{double(start), double(payload.size())}}};
    };
    constexpr int n = 64;
    std::vector<int8_t> w(n * cols), zero(n);
    std::vector<float> scales(n);
    for (int row = 0; row < n; ++row) {
        scales[row] = (row + 1) / 1024.f;
        for (int c = 0; c < cols; ++c)
            w[size_t(row) * cols + c] = int8_t((row * 17 + c * 31) % 255 - 127);
    }
    if (unsloth) {
        header["__metadata__"] = json11::Json::object{{"h3_convrot_groupsize", "256"}};
        add("transformer_blocks.0.attn.to_q._weight_qdata", "I8", {1, 1}, zero.data(), 1);
        for (auto projection : {"net.0.proj", "net.2"}) {
            std::string p = "transformer_blocks.0.ff." + std::string(projection);
            add(p + "._weight_qdata", "I8", {n, cols}, w.data(), w.size());
            add(p + "._weight_scale", "F32", {n, 1}, scales.data(), scales.size() * 4);
            add(p + "._weight_zero_point", "I8", {n, 1}, zero.data(), zero.size());
        }
    } else {
        const std::string marker =
            badMarker ? R"({"format":"int8_tensorwise","convrot":false,"convrot_groupsize":256})"
                      : R"({"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256})";
        for (auto projection : {"fc1", "fc2"}) {
            std::string p = "blocks.0.mlp." + std::string(projection);
            add(p + ".weight", "I8", {n, cols}, w.data(), w.size());
            add(p + ".weight_scale", "F32", {n, 1}, scales.data(), scales.size() * 4);
            add(p + ".comfy_quant", "U8", {int(marker.size())}, marker.data(), marker.size());
        }
    }
    auto text = json11::Json(header).dump();
    uint64_t size = text.size();
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<char *>(&size), 8);
    out << text;
    out.write(payload.data(), payload.size());
}
int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count)
        return 77;
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major < 8)
        return 77;
    const auto root = fs::temp_directory_path() / ("h3-int8-" + std::to_string(getpid()));
    fs::create_directories(root);
    try {
        Runtime rt(0, DType::BF16);
        for (int cols : {256, 5376, 7168, 14336})
            for (bool unsloth : {false, true}) {
                auto path = root / "weights.safetensors";
                fixture(path, cols, unsloth);
                auto weights = QuantizedWeights::load(path.string());
                require(weights->format() == WeightFormat::INT8);
                for (int rows : {1, 17, 129}) {
                    std::vector<float> input(size_t(rows) * cols), rotated(input.size()), scales(rows);
                    std::vector<int8_t> q(input.size());
                    for (int row = 0; row < rows; ++row)
                        for (int c = 0; c < cols; ++c)
                            input[size_t(row) * cols + c] = row ? ((row * 13 + c * 7) % 201 - 100) / 64.f : 0;
                    for (int row = 0; row < rows; ++row) {
                        float largest = 0;
                        for (int group = 0; group < cols; group += 256)
                            for (int j = 0; j < 256; ++j) {
                                float sum = 0;
                                for (int k = 0; k < 256; ++k)
                                    sum += input[size_t(row) * cols + group + k] * h256(k, j);
                                float v = bf16(sum);
                                rotated[size_t(row) * cols + group + j] = v;
                                largest = std::max(largest, std::abs(v));
                            }
                        scales[row] = std::max(largest / 127.f, 1e-12f);
                        for (int c = 0; c < cols; ++c)
                            q[size_t(row) * cols + c] = int8_t(
                                std::clamp(int(std::nearbyint(rotated[size_t(row) * cols + c] / scales[row])),
                                           -128, 127));
                    }
                    auto x = rt.upload(input.data(), rows, cols);
                    for (auto projection : {"fc1", "fc2"})
                        for (int repeat = 0; repeat < 2; ++repeat) {
                            Tensor destination(rows, 64);
                            auto result = weights->linear(x, "blocks.0.mlp." + std::string(projection),
                                                          repeat ? &destination : nullptr);
                            auto actual = rt.download(result);
                            double err = 0, norm = 0;
                            for (int row = 0; row < rows; ++row)
                                for (int out = 0; out < 64; ++out) {
                                    int source =
                                        unsloth && std::string(projection) == "fc1" ? (out + 32) % 64 : out;
                                    int32_t sum = 0;
                                    for (int c = 0; c < cols; ++c)
                                        sum += int(q[size_t(row) * cols + c]) *
                                               ((source * 17 + c * 31) % 255 - 127);
                                    float expected =
                                        bf16((float(sum) * scales[row]) * ((source + 1) / 1024.f));
                                    float value = actual[size_t(row) * 64 + out];
                                    require(std::isfinite(value));
                                    err += double(value - expected) * (value - expected);
                                    norm += double(expected) * expected;
                                }
                            const double l2 = std::sqrt(err / std::max(norm, 1e-30));
                            std::cout << "cols=" << cols << " rows=" << rows << " unsloth=" << unsloth
                                      << " projection=" << projection << " repeat=" << repeat
                                      << " relative_l2=" << l2 << '\n';
                            require(l2 < 1e-6);
                        }
                }
                weights->erasePrefix("blocks.0.");
                weights->clear();
            }
        fixture(root / "bad.safetensors", 256, false, true);
        auto bad = QuantizedWeights::load((root / "bad.safetensors").string());
        std::vector<float> zeros(256);
        bool rejected = false;
        try {
            bad->linear(rt.upload(zeros.data(), 1, 256), "blocks.0.mlp.fc2");
        } catch (const std::exception &) {
            rejected = true;
        }
        require(rejected);
        fs::remove_all(root);
        std::cout << "native INT8 passed\n";
    } catch (const std::exception &e) {
        fs::remove_all(root);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
