#include "models/minimax_h3_image/source.h"
#include "json11.hpp"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <vector>

namespace fastllm::minimax_h3_image {
namespace {
namespace fs = std::filesystem;
bool base(const fs::path &p) {
    return fs::is_regular_file(p / "transformer/config.json") && fs::is_regular_file(p / "vae/config.json") &&
           fs::is_regular_file(p / "text_encoder/config.json") &&
           fs::is_regular_file(p / "pdd_config.json") &&
           fs::is_regular_file(p / "pdd_fused_heads.safetensors") && fs::is_directory(p / "tokenizer");
}
// 0: ordinary tensors, 1: supported E4M3 layout, 2: another quantized layout.
int kind(const fs::path &p) {
    std::ifstream f(p, std::ios::binary);
    uint64_t n = 0;
    f.read(reinterpret_cast<char *>(&n), 8);
    if (!f || n == 0 || n > (16u << 20) || n + 8 >= fs::file_size(p))
        throw std::invalid_argument("invalid image safetensors header: " + p.string());
    std::string data(size_t(n), '\0'), error;
    f.read(data.data(), n);
    auto header = json11::Json::parse(data, error);
    if (!f || !error.empty() || !header.is_object())
        throw std::invalid_argument("invalid image safetensors JSON: " + p.string());
    bool fp8 = false, other = false;
    for (const auto &entry : header.object_items()) {
        const auto &name = entry.first;
        if (name.compare(0, 7, "blocks.") && name.compare(0, 19, "transformer_blocks."))
            continue;
        // Comfy embeds U8 JSON markers and auxiliary scales alongside weights.
        // Precision follows matrix payloads, never those metadata tensors.
        const auto ends = [&](const std::string &suffix) {
            return name.size() >= suffix.size() &&
                   name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
        };
        if (!ends(".weight") && !ends("._weight_qdata"))
            continue;
        const auto dtype = entry.second["dtype"].string_value();
        if (dtype == "F8_E4M3")
            fp8 = true;
        else if (dtype == "I8" || dtype == "U8" || dtype.find("F4") == 0 || ends("._weight_qdata"))
            other = true;
    }
    return other ? 2 : fp8 ? 1 : 0;
}
} // namespace
ModelSource resolveModelSource(const std::string &input) {
    if (input.empty())
        throw std::invalid_argument("image model path is empty");
    fs::path path = fs::absolute(fs::path(input)).lexically_normal();
    fs::path quant;
    if (fs::is_directory(path)) {
        std::vector<fs::path> candidates;
        for (const auto &dir : {path, path / "transformer", path / "diffusion_models"}) {
            if (!fs::is_directory(dir))
                continue;
            for (const auto &entry : fs::directory_iterator(dir))
                if (entry.is_regular_file() && entry.path().extension() == ".safetensors" &&
                    kind(entry.path()))
                    candidates.push_back(entry.path());
        }
        if (candidates.size() > 1)
            throw std::invalid_argument("image directory contains multiple quantized checkpoints; select one "
                                        ".safetensors file as the model path");
        if (candidates.empty()) {
            if (!base(path))
                throw std::invalid_argument(
                    "expected a complete MiniMax-H3-Image directory or a quantized .safetensors file");
            return {path.string(), {}};
        }
        quant = candidates.front();
    } else if (fs::is_regular_file(path) && path.extension() == ".safetensors") {
        quant = path;
    } else {
        throw std::invalid_argument("image model path does not exist or is not a safetensors checkpoint: " +
                                    input);
    }
    if (kind(quant) != 1)
        throw std::invalid_argument(
            "this image checkpoint is not supported FP8 E4M3; INT8 and FP4 are not implemented");
    // Bounded local search; do not scan disks or silently download a base model.
    std::set<fs::path> found;
    auto directory = quant.parent_path();
    for (int level = 0; level < 3; ++level) {
        for (const auto &candidate : {directory, directory / "base_model", directory / "MiniMax-H3-Image"})
            if (base(candidate))
                found.insert(fs::canonical(candidate));
        if (directory == directory.parent_path())
            break;
        directory = directory.parent_path();
    }
    if (found.empty())
        throw std::invalid_argument(
            "FP8 DiT needs the complete MiniMax-H3-Image base; place it alongside the checkpoint repository, "
            "or link it as base_model inside that repository");
    if (found.size() != 1)
        throw std::invalid_argument(
            "multiple image bases found; keep one matching base_model beside the selected FP8 checkpoint");
    return {found.begin()->string(), quant.string()};
}
} // namespace fastllm::minimax_h3_image
