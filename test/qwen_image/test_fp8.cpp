#include "quantized.h"
#include "json11.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>

using namespace fastllm::qwen_image;
using Json = json11::Json;
namespace fs = std::filesystem;
namespace {
void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
template<class F> void rejects(F fn, const std::string &reason = "") {
    bool rejected = false;
    try { fn(); } catch (const std::exception &e) {
        rejected = reason.empty() || std::string(e.what()).find(reason) != std::string::npos;
    }
    require(rejected, "invalid FP8 input accepted");
}
float bf16(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, 4);
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000;
    std::memcpy(&x, &bits, 4);
    return x;
}
float e4m3(float x) {
    if (x == 0) return 0;
    const float a = std::min(std::abs(x), 448.f);
    const float step = std::ldexp(1.f, std::max(-9, std::ilogb(a) - 3));
    return std::copysign(std::nearbyint(a / step) * step, x);
}
const std::vector<std::string> names{
    "transformer_blocks.0.attn.to_q", "transformer_blocks.0.attn.to_k", "transformer_blocks.0.attn.to_v"};
void fixture(const fs::path &path, const std::string &fault = "") {
    constexpr int cols = 4096;
    std::vector<char> payload;
    Json::object header, metadata;
    auto add = [&](const std::string &name, const std::string &dtype, Json::array shape,
                   const void *data, size_t bytes) {
        auto start = payload.size();
        payload.insert(payload.end(), static_cast<const char *>(data), static_cast<const char *>(data) + bytes);
        header[name] = Json::object{{"dtype", dtype}, {"shape", shape},
                                   {"data_offsets", Json::array{double(start), double(payload.size())}}};
    };
    for (int part = 0; part < 3; ++part) {
        std::vector<uint8_t> weight(size_t(cols) * cols);
        std::vector<float> scale(cols);
        for (int row = 0; row < cols; ++row) {
            weight[size_t(row) * cols + (row + part * 7) % cols] = row % 2 ? 0xb8 : 0x38;
            scale[row] = .25f * (1 << ((row + part) % 3));
        }
        if (fault == "nonfinite" && part == 0) scale[0] = std::nanf("");
        add(names[part] + "._weight_qdata", "F8_E4M3", {cols, cols}, weight.data(), weight.size());
        if (fault != "missing-scale" || part != 0)
            add(names[part] + "._weight_scale", "F32", {cols, 1}, scale.data(), scale.size() * 4);
        Json::object activation{
            {"float8_dtype", Json::object{{"_data", "float8_e4m3fn"}}},
            {"kernel_preference", Json::object{{"_data", "TORCH"}}},
            {"hp_value_lb", fault == "scale-bound" ? 1e-4 : 1e-12}, {"hp_value_ub", Json()},
            {"granularity", Json::object{{"_type", "PerRow"}, {"_data", Json::object{{"dim", -1}}}}}};
        Json::object data{
            {"block_size", Json::array{fault == "blockwise" ? 128 : 1, cols}},
            {"dtype", Json::object{{"_data", "bfloat16"}}},
            {"act_quant_kwargs", Json::object{{"_data", activation}}}};
        Json value = Json::object{{"_type", "Float8Tensor"}, {"_data", data}};
        metadata[names[part] + ".weight"] = value.dump();
    }
    if (fault == "partial-qkv") header.erase(names[1] + "._weight_qdata");
    if (fault == "mixed-int8") {
        auto descriptor = header[names[0] + "._weight_qdata"].object_items();
        descriptor["dtype"] = "I8";
        header["transformer_blocks.1.attn.to_q.weight"] = descriptor;
    }
    if (fault == "plain") header.clear();
    header["__metadata__"] = metadata;
    const auto text = Json(header).dump();
    const uint64_t size = text.size();
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(&size), 8);
    file << text;
    file.write(payload.data(), payload.size());
    require(bool(file), "fixture write failed");
}
} // namespace
int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess ||
        properties.major * 10 + properties.minor < 89) return 77;
    auto root = fs::temp_directory_path() / ("qwen-fp8-" + std::to_string(getpid()));
    try {
        fs::create_directories(root);
        Runtime rt(0);
        const auto path = root / "weights.safetensors";
        fixture(path);
        auto weights = QuantizedProjections::load(rt, path.string());
        require(bool(weights) && weights->contains(names[0]) && !weights->contains("proj_out"), "FP8 detection");
        require(weights->loadedBytes() == 0, "eager weight allocation");
        for (int rows : {1, 29, 137}) {
            std::vector<float> values(size_t(rows) * 4096);
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < 4096; ++c)
                    values[size_t(r) * 4096 + c] = r % 3 == 0 ? 0 : c % 3 == 0 ? 448 : c % 3 == 1 ? 224 : -112;
            auto input = rt.upload(values.data(), rows, 4096);
            Tensor storage(rows + 5, 12288, input.dtype), output = storage.rowSlice(5, rows);
            auto result = weights->linear(input, names, &output);
            require(result.data == output.data, "destination not reused");
            auto actual = rt.download(result);
            for (int r = 0; r < rows; ++r)
                for (int part = 0; part < 3; ++part)
                    for (int c = 0; c < 4096; ++c) {
                        float expected = values[size_t(r) * 4096 + (c + part * 7) % 4096] *
                                         (c % 2 ? -1 : 1) * .25f * (1 << ((c + part) % 3));
                        require(actual[size_t(r) * 12288 + part * 4096 + c] == expected, "packed QKV order/scales");
                    }
            auto single = rt.download(weights->linear(input, {names[1]}));
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < 4096; ++c)
                    require(single[size_t(r) * 4096 + c] == actual[size_t(r) * 12288 + 4096 + c], "packed/single cache collision");
            require(weights->loadedBytes() > (size_t(48) << 20), "loaded FP8 bytes missing");
            weights->clear();
            require(weights->loadedBytes() == 0, "FP8 buffers retained after clear");
        }
        // Non-power-of-two and tiny maxima exercise TorchAO's BF16 scale
        // rounding and pre-division clamp, independently of the CUDA kernel.
        for (float magnitude : {3.25f, 7.15625f, 1e-13f}) {
            constexpr int rows = 3, cols = 4096;
            std::vector<float> values(rows * cols), quantized(values.size());
            for (int r = 0; r < rows; ++r) {
                float largest = 0;
                for (int c = 0; c < cols; ++c) {
                    const auto i = r * cols + c;
                    values[i] = bf16(magnitude * float((c * 37 + r * 11) % 511 - 255) / 255.f);
                    largest = std::max(largest, std::abs(values[i]));
                }
                const float scale = bf16(std::max(largest, bf16(1e-12f)) / 448.f);
                for (int c = 0; c < cols; ++c)
                    quantized[r * cols + c] = e4m3(values[r * cols + c] / scale) * scale;
            }
            auto actual = rt.download(weights->linear(rt.upload(values.data(), rows, cols), names));
            for (int r = 0; r < rows; ++r)
                for (int part = 0; part < 3; ++part)
                    for (int c = 0; c < cols; ++c) {
                        const float expected = bf16(quantized[r * cols + (c + part * 7) % cols] *
                            (c % 2 ? -1 : 1) * .25f * (1 << ((c + part) % 3)));
                        require(actual[r * cols * 3 + part * cols + c] == expected, "TorchAO activation scale rounding");
                    }
        }
        Runtime fp32(0, true);
        rejects([&] { QuantizedProjections::load(fp32, path.string()); });
        weights.reset();
        for (const auto *fault : {"missing-scale", "blockwise", "partial-qkv", "scale-bound"}) {
            fixture(path, fault);
            rejects([&] { QuantizedProjections::load(rt, path.string()); });
        }
        fixture(path, "nonfinite");
        auto bad = QuantizedProjections::load(rt, path.string());
        std::vector<float> ones(4096, 1);
        rejects([&] { bad->linear(rt.upload(ones.data(), 1, 4096), names); });
        bad.reset();
        fixture(path, "mixed-int8");
        rejects([&] { QuantizedProjections::load(rt, path.string()); }, "mixed FP8 and INT8");
        fixture(root / "plain.safetensors", "plain");
        require(!QuantizedProjections::load(rt, (root / "plain.safetensors").string()), "plain weights changed");
        fs::remove_all(root);
        std::cout << "PASS Qwen FP8 detection, packed/scalar layout, tails, destination, release and rejection checks\n";
    } catch (const std::exception &e) {
        fs::remove_all(root);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
