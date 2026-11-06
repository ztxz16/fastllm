#include "models/minimax_h3/model_source.h"
#include "json11.hpp"
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fastllm { namespace minimax_h3 {
namespace {
namespace fs = std::filesystem;
void require(bool ok, const std::string &message) {
    if (!ok) throw std::invalid_argument("Pika adapter: " + message);
}
bool hasBase(const fs::path &path) {
    for (const char *part : {"transformer", "transformer_ref", "text_encoder", "vae", "audio_vae"})
        if (!fs::is_regular_file(path / part / "config.json")) return false;
    return fs::is_regular_file(path / "tokenizer/tokenizer.json");
}
bool hasAdapter(const fs::path &path) {
    for (const char *part : {"fl2va", "ref2va"})
        for (const char *file : {"manifest.json", "adapter.safetensors"})
            if (fs::exists(path / part / file)) return true;
    return false;
}
} // namespace

void validatePikaManifest(const std::string &dir, bool reference) {
    const auto path = dir + "/manifest.json";
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    require(bool(file) && file.tellg() > 0 && file.tellg() <= 65536, "missing or invalid manifest: " + path);
    std::string bytes(size_t(file.tellg()), '\0'), error;
    file.seekg(0);
    file.read(bytes.data(), bytes.size());
    auto j = json11::Json::parse(bytes, error);
    require(bool(file) && error.empty() && j.is_object(), "invalid manifest JSON: " + path);
    require(j["format_version"].number_value() == 1 && j["model"].string_value() == "pika-flex" &&
            j["rank"].number_value() == 128 && j["alpha"].number_value() == 128, "unsupported format/rank/alpha");
    require(j["task"].string_value() == (reference ? "ref2va" : "fl2va") &&
            j["trunk"].string_value() == (reference ? "transformer_ref" : "transformer"), "task/trunk mismatch");
    require(j["numerics_policy"].string_value() == (reference ? "fp32_conditioning_v1" : "bf16_conditioning_v1"),
            "unsupported conditioning precision");
    require(j["base"]["repo_id"].string_value() == "MiniMaxAI/MiniMax-H3" &&
            j["base"]["revision"].string_value() == "42ed227ee7df40d41602854ae760620d6eb651fe", "unsupported base revision");
    const auto &s = j["sampling"];
    require(s["method"].string_value() == "renoise" && s["video_shift"].number_value() == 12 &&
            s["audio_shift"].number_value() == 3 &&
            s["knots"] == json11::Json::array{1000, 750, 500, 250, 100, 40, 0}, "unsupported sampling schedule");
    require(j["target_modules"] == json11::Json::array{"to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"},
            "unsupported target modules");
}

ModelSource resolveModelSource(const std::string &model, const std::string &base,
                               const std::string &legacyPika) {
    if (model.empty()) throw std::invalid_argument("--model is required");
    if (!legacyPika.empty() && !base.empty())
        throw std::invalid_argument("use --model PIKA --base-model BASE, or --model BASE --pika PIKA");
    if (legacyPika.empty() && !hasAdapter(model)) {
        if (!base.empty()) throw std::invalid_argument("--base-model requires a Pika model directory");
        return {model, {}};
    }
    const auto adapter = legacyPika.empty() ? model : legacyPika;
    validatePikaManifest(adapter + "/fl2va", false);
    validatePikaManifest(adapter + "/ref2va", true);
    // Keep the explicit legacy path authoritative, with its existing loader errors.
    if (!legacyPika.empty()) return {model, adapter};
    if (!base.empty()) {
        require(hasBase(base), "invalid --base-model directory: " + base);
        return {base, adapter};
    }
    const auto root = fs::absolute(model).lexically_normal();
    // Strip a trailing slash before locating a sibling. Do not canonicalize:
    // local symlink views may keep the adapter and base beside one another.
    const auto directory = root.filename().empty() ? root.parent_path() : root;
    const auto linked = directory / "base_model";
    if (fs::exists(linked) || fs::is_symlink(linked)) {
        require(hasBase(linked), "invalid base_model directory or symlink: " + linked.string());
        return {linked.string(), adapter};
    }
    for (const auto &candidate : {directory, directory.parent_path() / "MiniMax-H3"})
        if (hasBase(candidate)) return {candidate.string(), adapter};
    throw std::invalid_argument("Pika H3 Flex detected, but its MiniMax-H3 base was not found; "
                                "place MiniMax-H3 beside the model, add a base_model symlink, "
                                "or pass --base-model PATH");
}
}} // namespace fastllm::minimax_h3
