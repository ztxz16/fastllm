#include "models/minimax_h3_image/fp8.h"
#include "json11.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
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
// Full H3 QKV dimensions exercise layout metadata, independently of model weights.
void checkQkvLayout(const fs::path &path, Runtime &rt, bool comfy) {
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
    std::string header = json11::Json(h).dump();
    uint64_t n = header.size();
    float scale = .5f;
    {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<char *>(&n), 8);
        f << header;
        f.write(reinterpret_cast<char *>(data.data()), data.size());
        f.write(reinterpret_cast<char *>(&scale), 4);
    }
    Fp8Weights weights(path.string());
    std::vector<float> x(cols);
    for (int i = 0; i < cols; ++i)
        x[i] = i % 5 ? 224 : 448;
    auto out = rt.download(weights.linear(rt.upload(x.data(), 1, cols), "blocks.0.attn.qkv_proj"));
    for (int row = 0; row < rows; ++row) {
        int within = row % 7168;
        int source = comfy ? row : (within / 128) * 384 + (row / 7168) * 128 + within % 128;
        require(out[row] == x[source % cols] * scale);
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
                            auto output =
                                w.linear(input, std::string("blocks.0.mlp.") + name, &destination);
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
            wa.clear();
            wb.clear();
            pool.trim();
            require(pool.reusableBytes() == 0);
        }
        for (bool comfy : {false, true})
            checkQkvLayout(root / "qkv.safetensors", rt, comfy);
        fixture(root / "bad.safetensors", false, 1, true);
        Fp8Weights bad((root / "bad.safetensors").string());
        std::vector<float> x(32, 1);
        auto input = rt.upload(x.data(), 1, 32);
        rejects([&] { bad.linear(input, "blocks.0.mlp.fc2"); });
        std::ofstream(root / "truncated.safetensors") << "bad";
        rejects([&] { Fp8Weights invalid((root / "truncated.safetensors").string()); });
        fs::remove_all(root);
        std::cout << "FP8 scaling, padding, FC1 order, instance isolation and release passed\n";
    } catch (const std::exception &e) {
        fs::remove_all(root);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
