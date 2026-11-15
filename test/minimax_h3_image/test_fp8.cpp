#include "json11.hpp"
#include "models/minimax_h3_image/fp8.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
using namespace fastllm::minimax_h3;
using fastllm::minimax_h3_image::Fp8Weights;
namespace fs = std::filesystem;
void require(bool ok) {
    if (!ok)
        throw std::runtime_error("FP8 fixture assertion failed");
}
template <class F> void rejects(F f) {
    try {
        f();
    } catch (const std::exception &) {
        return;
    }
    throw std::runtime_error("invalid FP8 input accepted");
}
void fixture(const fs::path &path, bool unsloth, float multiplier, bool badScale = false) {
    json11::Json::object h;
    std::vector<char> payload;
    auto add = [&](const std::string &key, const std::string &dtype, json11::Json::array shape,
                   const void *data, size_t bytes) {
        size_t offset = payload.size();
        const auto *p = static_cast<const char *>(data);
        payload.insert(payload.end(), p, p + bytes);
        h[key] = json11::Json::object{
            {"dtype", dtype},
            {"shape", shape},
            {"data_offsets", json11::Json::array{double(offset), double(payload.size())}}};
    };
    std::vector<uint8_t> w(32 * 32);
    for (int row = 0; row < 32; ++row)
        w[row * 32 + row] = (row % 2) ? 0xb8 : 0x38; // +/-1
    std::vector<float> scale(unsloth ? 32 : 1, multiplier);
    if (unsloth)
        for (int i = 0; i < 32; ++i)
            scale[i] *= 1 << (i % 3);
    if (badScale)
        scale[0] = std::nanf("");
    if (unsloth)
        add("transformer_blocks.0.attn.to_q._weight_qdata", "F8_E4M3", {32, 32}, w.data(), w.size());
    for (auto name : {"fc1", "fc2"}) {
        std::string p = unsloth ? std::string("transformer_blocks.0.ff.") +
                                      (std::string(name) == "fc1" ? "net.0.proj" : "net.2")
                                : std::string("blocks.0.mlp.") + name;
        add(p + (unsloth ? "._weight_qdata" : ".weight"), "F8_E4M3", {32, 32}, w.data(), w.size());
        add(p + (unsloth ? "._weight_scale" : ".weight_scale"), "F32", {int(scale.size())}, scale.data(),
            scale.size() * 4);
    }
    std::string header = json11::Json(h).dump();
    uint64_t n = header.size();
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<char *>(&n), 8);
    f << header;
    f.write(payload.data(), payload.size());
}
// Full H3 QKV dimensions exercise layout metadata, independently of model
// weights.
void checkQkvLayout(const fs::path &path, Runtime &rt, int layout) {
    const bool comfy = layout == 1, unsloth = layout == 2;
    constexpr int rows = 21504, cols = 5376;
    std::vector<uint8_t> data(size_t(rows) * cols);
    for (int row = 0; row < rows; ++row)
        data[size_t(row) * cols + row % cols] = 0x38;
    json11::Json::object h{
        {"blocks.0.attn.qkv_proj.weight",
         json11::Json::object{{"dtype", "F8_E4M3"},
                              {"shape", json11::Json::array{rows, cols}},
                              {"data_offsets", json11::Json::array{0, double(data.size())}}}},
        {"blocks.0.attn.qkv_proj.weight_scale",
         json11::Json::object{
             {"dtype", "F32"},
             {"shape", json11::Json::array{}},
             {"data_offsets", json11::Json::array{double(data.size()), double(data.size() + 4)}}}}};
    if (comfy)
        h["__metadata__"] =
            json11::Json::object{{"_quantization_metadata",
                                  R"({"layers":{"blocks.0.attn.qkv_proj":{"format":"float8_e4m3fn"}}})"}};
    std::vector<float> scales(unsloth ? rows : 1, .5f);
    if (unsloth) {
        h.clear();
        for (int row = 0; row < rows; ++row) scales[row] = .25f * (row % 3 + 1);
        for (int part = 0; part < 3; ++part) {
            const std::string key = std::string("transformer_blocks.0.attn.to_") + "qkv"[part];
            const size_t first = size_t(part) * (rows / 3) * cols, bytes = size_t(rows / 3) * cols;
            const size_t scaleFirst = data.size() + size_t(part) * (rows / 3) * 4;
            h[key + "._weight_qdata"] = json11::Json::object{
                {"dtype", "F8_E4M3"}, {"shape", json11::Json::array{rows / 3, cols}},
                {"data_offsets", json11::Json::array{double(first), double(first + bytes)}}};
            h[key + "._weight_scale"] = json11::Json::object{
                {"dtype", "F32"}, {"shape", json11::Json::array{rows / 3}},
                {"data_offsets", json11::Json::array{double(scaleFirst), double(scaleFirst + rows / 3 * 4)}}};
        }
    }
    std::string header = json11::Json(h).dump();
    uint64_t n = header.size();
    {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<char *>(&n), 8);
        f << header;
        f.write(reinterpret_cast<char *>(data.data()), data.size());
        f.write(reinterpret_cast<char *>(scales.data()), scales.size() * 4);
    }
    std::vector<float> x(cols);
    for (int i = 0; i < cols; ++i) x[i] = i % 5 ? 224 : 448;
    auto input = rt.upload(x.data(), 1, cols);
    // Virtual ranks on one GPU check every shard against the CPU layout;
    // this catches permutations which a sum-only TP test would miss.
    for (int size : {1, 2, 4, 8}) {
        for (int rank = 0; rank < size; ++rank) {
            Fp8Weights weights(path.string(), {false, rank, size});
            auto out = rt.download(weights.linear(input, "blocks.0.attn.qkv_proj"));
            require(out.size() == size_t(rows / size));
            const int width = 7168 / size;
            for (int row = 0; row < rows / size; ++row) {
                const int logical = row / width * 7168 + rank * width + row % width;
                const int within = logical % 7168;
                const int source = comfy || unsloth ? logical
                    : (within / 128) * 384 + (logical / 7168) * 128 + within % 128;
                require(out[row] == x[source % cols] * scales[unsloth ? logical : 0]);
            }
        }
    }
}
float roundE4m3(float x) {
    if (x == 0)
        return 0;
    float a = std::min(std::abs(x), 448.f);
    float step = std::ldexp(1.f, std::max(-9, std::ilogb(a) - 3));
    return std::copysign(std::nearbyint(a / step) * step, x);
}
// Exercise the actual vectorized input widths against an independent CPU
// E4M3 reference, including partial tiles, zero/tiny rows and nonuniform
// scales.
void checkProjection(const fs::path &path, Runtime &rt, int cols) {
    constexpr int outputs = 32;
    std::vector<uint8_t> w(outputs * cols);
    std::vector<float> scale(outputs), decoded(outputs * cols);
    for (int n = 0; n < outputs; ++n) {
        scale[n] = .0031f * (n % 5 + 1);
        for (int k = 0; k < cols; ++k) {
            unsigned h = unsigned(n * cols + k) * 1664525u + 1013904223u;
            h ^= h >> 13;
            int exponent = (h >> 10) % 3;
            w[n * cols + k] = uint8_t(0x30 + 8 * exponent + ((h & 1) ? 0x80 : 0));
            decoded[n * cols + k] = std::ldexp((h & 1) ? -1.f : 1.f, exponent - 1);
        }
    }
    const std::string key = "transformer_blocks.0.ff.net.2";
    json11::Json::object header{
        {"transformer_blocks.0.attn.to_q._weight_qdata", json11::Json::object{}},
        {key + "._weight_qdata",
         json11::Json::object{{"dtype", "F8_E4M3"},
                              {"shape", json11::Json::array{outputs, cols}},
                              {"data_offsets", json11::Json::array{0, double(w.size())}}}},
        {key + "._weight_scale",
         json11::Json::object{
             {"dtype", "F32"},
             {"shape", json11::Json::array{outputs}},
             {"data_offsets", json11::Json::array{double(w.size()), double(w.size() + scale.size() * 4)}}}}};
    {
        std::string text = json11::Json(header).dump();
        uint64_t size = text.size();
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<char *>(&size), 8);
        file << text;
        file.write(reinterpret_cast<char *>(w.data()), w.size());
        file.write(reinterpret_cast<char *>(scale.data()), scale.size() * 4);
    }
    Fp8Weights weights(path.string());
    for (int rows : {1, 29, 129}) {
        std::vector<float> x(rows * cols);
        for (int r = 0; r < rows; ++r)
            for (int k = 0; k < cols; ++k)
                x[r * cols + k] = r % 7 == 3 ? 0.f
                                             : std::sin(float(r * 37 + k) * .137f) *
                                                   (r % 7 == 2 ? 1e-9f : float(r % 5 + 1));
        auto input = rt.upload(x.data(), rows, cols);
        x = rt.download(input);
        auto actual = rt.download(weights.linear(input, "blocks.0.mlp.fc2"));
        double error = 0, norm = 0;
        std::vector<float> quantized(cols);
        for (int r = 0; r < rows; ++r) {
            float largest = 0;
            for (int k = 0; k < cols; ++k)
                largest = std::max(largest, std::abs(x[r * cols + k]));
            float xs = std::max(largest / 448.f, 1e-12f);
            for (int k = 0; k < cols; ++k)
                quantized[k] = roundE4m3(x[r * cols + k] / xs);
            for (int n = 0; n < outputs; ++n) {
                double sum = 0;
                for (int k = 0; k < cols; ++k)
                    sum += double(quantized[k]) * decoded[n * cols + k];
                double expected = sum * xs * scale[n], value = actual[r * outputs + n];
                require(std::isfinite(value));
                error += (value - expected) * (value - expected);
                norm += expected * expected;
            }
        }
        double l2 = std::sqrt(error / std::max(norm, 1e-30));
        std::cout << "FP8 CPU oracle " << rows << 'x' << cols << " relative_l2=" << l2 << '\n';
        require(l2 < .01);
    }
}
int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count)
        return 77;
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major * 10 + prop.minor < 89)
        return 77;
    auto root = fs::temp_directory_path() / ("h3-fp8-" + std::to_string(getpid()));
    try {
        fs::create_directories(root);
        Runtime rt(0);
        Fp8Weights::validateDevice(0);
        fastllm::nomas::TensorPool pool(0, 128u << 20);
        fastllm::nomas::TensorPool::Scope scope(pool);
        for (bool unsloth : {false, true}) {
            auto a = root / "a.safetensors", b = root / "b.safetensors";
            fixture(a, unsloth, .25f);
            fixture(b, unsloth, .5f);
            Fp8Weights wa(a.string()), wb(b.string());
            rejects([&] { wa.validateModel(); });
            for (int rows : {1, 29, 64}) {
                std::vector<float> x(rows * 32);
                for (int r = 0; r < rows; ++r)
                    for (int c = 0; c < 32; ++c)
                        x[r * 32 + c] = r % 3 == 0 ? 0 : (c % 2 ? -224 : 448);
                auto input = rt.upload(x.data(), rows, 32);
                for (auto name : {"fc1", "fc2"})
                    for (int repeat = 0; repeat < 3; ++repeat) {
                        for (int which = 0; which < 2; ++which) {
                            Tensor destination(rows, 32);
                            auto &w = which ? wb : wa;
                            auto output = w.linear(input, std::string("blocks.0.mlp.") + name, &destination);
                            require(output.data == destination.data);
                            auto values = rt.download(output);
                            for (int r = 0; r < rows; ++r)
                                for (int c = 0; c < 32; ++c) {
                                    int source = unsloth && std::string(name) == "fc1" ? (c + 16) % 32 : c;
                                    float expected = x[r * 32 + source] * (source % 2 ? -1 : 1) *
                                                     (which ? .5f : .25f) *
                                                     (unsloth ? (1 << (source % 3)) : 1);
                                    require(values[r * 32 + c] == expected);
                                }
                        }
                        if (repeat == 0)
                            wa.erasePrefix("blocks.0.");
                        if (repeat == 1)
                            wa.clear();
                    }
            }
            for (int rank = 0; rank < 2; ++rank) {
                Fp8Weights shard(a.string(), {false, rank, 2});
                std::vector<float> x(32);
                for (int c = 0; c < 32; ++c) x[c] = c % 3 ? 224 : 448;
                auto output = rt.download(shard.linear(rt.upload(x.data(), 1, 32), "blocks.0.mlp.fc1"));
                require(output.size() == 16);
                for (int c = 0; c < 16; ++c) {
                    const int logical = c / 8 * 16 + rank * 8 + c % 8;
                    const int source = unsloth ? (logical + 16) % 32 : logical;
                    require(output[c] == x[source] * (source % 2 ? -1 : 1) * .25f *
                                             (unsloth ? (1 << (source % 3)) : 1));
                }
            }
            wa.clear();
            wb.clear();
            pool.trim();
            require(pool.reusableBytes() == 0);
        }
        for (int layout : {0, 1, 2})
            checkQkvLayout(root / "qkv.safetensors", rt, layout);
        for (int cols : {4096, 5376, 7168, 12288, 14336})
            checkProjection(root / "projection.safetensors", rt, cols);
        fixture(root / "bad.safetensors", false, 1, true);
        Fp8Weights bad((root / "bad.safetensors").string());
        std::vector<float> x(32, 1);
        auto input = rt.upload(x.data(), 1, 32);
        rejects([&] { bad.linear(input, "blocks.0.mlp.fc2"); });
        std::ofstream(root / "truncated.safetensors") << "bad";
        rejects([&] { Fp8Weights invalid((root / "truncated.safetensors").string()); });
        fs::remove_all(root);
        std::cout << "FP8 scaling, padding, FC1 order, instance isolation and "
                     "release passed\n";
    } catch (const std::exception &e) {
        fs::remove_all(root);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
