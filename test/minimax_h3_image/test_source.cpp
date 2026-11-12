#include "models/minimax_h3_image/source.h"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
using fastllm::minimax_h3_image::resolveModelSource;
namespace fs = std::filesystem;
void require(bool ok) {
    if (!ok)
        throw std::runtime_error("source assertion failed");
}
template <class F> void rejects(F f, const std::string &text) {
    try {
        f();
    } catch (const std::exception &e) {
        require(std::string(e.what()).find(text) != std::string::npos);
        return;
    }
    throw std::runtime_error("accepted invalid checkpoint: " + text);
}
void base(const fs::path &p) {
    for (auto name : {"transformer/config.json", "vae/config.json", "text_encoder/config.json",
                      "pdd_config.json", "pdd_fused_heads.safetensors"}) {
        fs::create_directories((p / name).parent_path());
        std::ofstream(p / name) << "{}";
    }
    // A structurally valid non-DiT tensor file is ignored by precision discovery.
    const std::string h = R"({"x":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}})";
    std::ofstream f(p / "pdd_fused_heads.safetensors", std::ios::binary);
    uint64_t n = h.size();
    f.write(reinterpret_cast<const char *>(&n), 8);
    f << h << "00";
    fs::create_directories(p / "tokenizer");
}
void tensor(const fs::path &p, const std::string &dtype) {
    const std::string h = "{\"blocks.0.mlp.fc1.weight\":{\"dtype\":\"" + dtype +
                          "\",\"shape\":[1,1],\"data_offsets\":[0,1]},\"blocks.0.mlp.fc1.comfy_quant\":{"
                          "\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[1,2]}}";
    fs::create_directories(p.parent_path());
    std::ofstream f(p, std::ios::binary);
    uint64_t n = h.size();
    f.write(reinterpret_cast<const char *>(&n), 8);
    f << h << "00";
}
int main() {
    auto root = fs::temp_directory_path() / ("h3-image-source-" + std::to_string(getpid()));
    try {
        fs::create_directories(root);
        base(root / "MiniMax-H3-Image");
        auto original = resolveModelSource((root / "MiniMax-H3-Image").string());
        require(original.fp8Weights.empty());
        auto file = root / "quant" / "arbitrary-name.safetensors";
        tensor(file, "F8_E4M3");
        auto selected = resolveModelSource(file.string());
        require(selected.fp8Weights == file.string());
        require(selected.directory == original.directory);
        require(resolveModelSource(file.parent_path().string()).fp8Weights == file.string());
        fs::create_directory_symlink(root / "MiniMax-H3-Image", root / "quant/base_model");
        require(resolveModelSource(file.string()).directory == original.directory);
        tensor(root / "quant/int8.safetensors", "I8");
        rejects([&] { resolveModelSource(file.parent_path().string()); }, "multiple quantized");
        rejects([&] { resolveModelSource((root / "quant/int8.safetensors").string()); }, "not supported FP8");
        // Selecting an exact file stays deterministic despite other checkpoints.
        require(resolveModelSource(file.string()).fp8Weights == file.string());
        base(root / "quant/MiniMax-H3-Image");
        rejects([&] { resolveModelSource(file.string()); }, "multiple image bases");
        fs::remove_all(root / "quant/MiniMax-H3-Image");
        fs::remove(root / "quant/base_model");
        fs::remove_all(root / "MiniMax-H3-Image");
        rejects([&] { resolveModelSource(file.string()); }, "needs the complete");
        std::ofstream(file, std::ios::trunc) << "bad";
        rejects([&] { resolveModelSource(file.string()); }, "invalid image safetensors");
        rejects([&] { resolveModelSource((root / "missing").string()); }, "does not exist");
        fs::remove_all(root);
        std::cout << "automatic image model source passed\n";
    } catch (const std::exception &e) {
        fs::remove_all(root);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
