#include "models/minimax_h3_image/source.h"
#include "json11.hpp"
#include "gguf_checkpoint.h"
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
WeightFormat kind(const fs::path &p) {
    std::ifstream f(p, std::ios::binary);
    char magic[4]{};
    f.read(magic, 4);
    if (std::string(magic, 4) == "GGUF") {
        detail::GgufCheckpoint checkpoint(p.string());
        return WeightFormat::GGUF;
    }
    f.clear();
    f.seekg(0);
    uint64_t n = 0;
    f.read(reinterpret_cast<char *>(&n), 8);
    if (!f || n == 0 || n > (16u << 20) || n + 8 >= fs::file_size(p))
        throw std::invalid_argument("invalid image safetensors header: " + p.string());
    std::string data(size_t(n), '\0'), error;
    f.read(data.data(), n);
    auto header = json11::Json::parse(data, error);
    if (!f || !error.empty() || !header.is_object())
        throw std::invalid_argument("invalid image safetensors JSON: " + p.string());
    auto metadataText = header["__metadata__"]["_quantization_metadata"].string_value();
    auto metadata = metadataText.empty() ? json11::Json{} : json11::Json::parse(metadataText, error);
    if (!error.empty())
        throw std::invalid_argument("invalid image quantization metadata");
    WeightFormat result = WeightFormat::BF16;
    bool found = false;
    for (const auto &entry : header.object_items()) {
        const auto &name = entry.first;
        if (name.compare(0, 7, "blocks.") && name.compare(0, 19, "transformer_blocks."))
            continue;
        const auto ends = [&](const std::string &suffix) {
            return name.size() >= suffix.size() &&
                   name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
        };
        const std::string suffix = ends("._weight_qdata") ? "._weight_qdata" : ".weight";
        if (!ends(suffix))
            continue;
        const auto projection = name.substr(0, name.size() - suffix.size());
        const auto projectionEnds = [&](const std::string &part) {
            return projection.size() >= part.size() &&
                   projection.compare(projection.size() - part.size(), part.size(), part) == 0;
        };
        bool mainProjection = false;
        for (const auto &part : {".attn.qkv_proj", ".attn.out_proj", ".mlp.fc1", ".mlp.fc2", ".attn.to_q",
                                 ".attn.to_k", ".attn.to_v", ".attn.to_out.0", ".ff.net.0.proj", ".ff.net.2"})
            mainProjection = mainProjection || projectionEnds(part);
        if (!mainProjection)
            continue;
        const auto dtype = entry.second["dtype"].string_value();
        WeightFormat format;
        if (dtype == "F8_E4M3")
            format = WeightFormat::FP8;
        else if (dtype == "I8")
            format = WeightFormat::INT8;
        else if (dtype == "U8" && ends(".weight") &&
                 metadata["layers"][name.substr(0, name.size() - 7)]["format"].string_value() == "nvfp4")
            format = WeightFormat::NVFP4;
        else if (dtype == "BF16" || dtype == "F32")
            continue;
        else
            throw std::invalid_argument("unsupported H3 Image quantized matrix: " + name);
        if (found && format != result)
            throw std::invalid_argument("mixed H3 Image quantization formats are not supported");
        found = true;
        result = format;
    }
    return result;
}
} // namespace
const char *weightFormatName(WeightFormat format) {
    switch (format) {
    case WeightFormat::FP8:
        return "fp8";
    case WeightFormat::INT8:
        return "int8";
    case WeightFormat::NVFP4:
        return "nvfp4";
    case WeightFormat::GGUF:
        return "gguf";
    default:
        return "bf16";
    }
}
WeightFormat inspectWeightFormat(const std::string &path) { return kind(path); }
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
                if (entry.is_regular_file() &&
                    (entry.path().extension() == ".safetensors" || entry.path().extension() == ".gguf") &&
                    kind(entry.path()) != WeightFormat::BF16)
                    candidates.push_back(entry.path());
        }
        if (candidates.size() > 1)
            throw std::invalid_argument("image directory contains multiple quantized checkpoints; select one "
                                        ".safetensors or .gguf file as the model path");
        if (candidates.empty()) {
            if (!base(path))
                throw std::invalid_argument("expected a complete MiniMax-H3-Image directory or a quantized "
                                            ".safetensors or .gguf file");
            return {path.string(), {}};
        }
        quant = candidates.front();
    } else if (fs::is_regular_file(path) &&
               (path.extension() == ".safetensors" || path.extension() == ".gguf")) {
        quant = path;
    } else {
        throw std::invalid_argument(
            "image model path does not exist or is not a safetensors/GGUF checkpoint: " + input);
    }
    auto format = kind(quant);
    if (format == WeightFormat::BF16)
        throw std::invalid_argument("select a complete BF16 image directory or a quantized DiT file");
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
        throw std::invalid_argument("quantized DiT needs the complete MiniMax-H3-Image base; place it "
                                    "alongside the checkpoint repository, "
                                    "or link it as base_model inside that repository");
    if (found.size() != 1)
        throw std::invalid_argument("multiple image bases found; keep one matching base_model beside the "
                                    "selected quantized checkpoint");
    return {found.begin()->string(), quant.string(), format};
}
} // namespace fastllm::minimax_h3_image
