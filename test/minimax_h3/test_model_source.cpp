#include "models/minimax_h3/model_source.h"
#include "json11.hpp"
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace fastllm::minimax_h3;
namespace fs = std::filesystem;
using Json = json11::Json;
namespace {
void check(bool value, const std::string &message) {
    if (!value) throw std::runtime_error(message);
}
void reject(const std::function<void()> &call, const std::string &message) {
    try { call(); }
    catch (const std::invalid_argument &e) {
        check(std::string(e.what()).find(message) != std::string::npos, e.what());
        return;
    }
    throw std::runtime_error("expected rejection: " + message);
}
void write(const fs::path &path, const std::string &value) {
    fs::create_directories(path.parent_path());
    std::ofstream file(path);
    file << value;
    check(bool(file), "cannot write fixture");
}
void base(const fs::path &path) {
    for (const char *part : {"transformer", "transformer_ref", "text_encoder", "vae", "audio_vae"})
        write(path / part / "config.json", "{}");
    write(path / "tokenizer/tokenizer.json", "{}");
}
Json manifest(bool reference) {
    return Json::object{
        {"format_version", 1}, {"model", "pika-flex"}, {"rank", 128}, {"alpha", 128},
        {"task", reference ? "ref2va" : "fl2va"},
        {"trunk", reference ? "transformer_ref" : "transformer"},
        {"numerics_policy", reference ? "fp32_conditioning_v1" : "bf16_conditioning_v1"},
        {"base", Json::object{{"repo_id", "MiniMaxAI/MiniMax-H3"},
                              {"revision", "42ed227ee7df40d41602854ae760620d6eb651fe"}}},
        {"sampling", Json::object{{"method", "renoise"}, {"video_shift", 12}, {"audio_shift", 3},
                                  {"knots", Json::array{1000, 750, 500, 250, 100, 40, 0}}}},
        {"target_modules", Json::array{"to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"}}
    };
}
void adapter(const fs::path &path) {
    write(path / "fl2va/manifest.json", manifest(false).dump());
    write(path / "ref2va/manifest.json", manifest(true).dump());
}
struct Temporary {
    fs::path path;
    Temporary() {
        std::mt19937_64 random(std::random_device{}());
        for (int i = 0; i < 32; ++i) {
            auto candidate = fs::temp_directory_path() / ("fastllm-h3-source-" + std::to_string(random()));
            if (fs::create_directory(candidate)) { path = candidate; return; }
        }
        throw std::runtime_error("cannot create temporary directory");
    }
    ~Temporary() { std::error_code error; fs::remove_all(path, error); }
};
} // namespace

int main() {
    try {
        Temporary tmp;
        auto original = tmp.path / "MiniMax-H3", pika = tmp.path / "arbitrary-name";
        base(original);
        adapter(pika);
        auto result = resolveModelSource(original.string());
        check(result.baseDirectory == original.string() && result.pikaDirectory.empty(), "original H3 detection");
        // Names alone must never enable Pika, nor change the original loader's path.
        auto fake = (tmp.path / "Pika-H3-Flex").string();
        result = resolveModelSource(fake);
        check(result.baseDirectory == fake && result.pikaDirectory.empty(), "directory name heuristic");
        reject([&] { resolveModelSource(""); }, "--model is required");
        reject([&] { resolveModelSource(original.string(), original.string()); }, "requires a Pika");
        for (const auto &path : {pika.string(), pika.string() + "/", fs::relative(pika).string()}) {
            result = resolveModelSource(path);
            check(fs::equivalent(result.baseDirectory, original) && result.pikaDirectory == path, "sibling base detection");
        }
        auto override = tmp.path / "elsewhere";
        base(override);
        result = resolveModelSource(pika.string(), override.string());
        check(result.baseDirectory == override.string(), "explicit base must take precedence");
        reject([&] { resolveModelSource(pika.string(), fake); }, "invalid --base-model");
        result = resolveModelSource(original.string(), {}, pika.string());
        check(result.baseDirectory == original.string() && result.pikaDirectory == pika.string(), "legacy flags");
        reject([&] { resolveModelSource(original.string(), override.string(), pika.string()); }, "use --model PIKA");
        fs::create_directory_symlink(override, pika / "base_model");
        result = resolveModelSource(pika.string());
        check(fs::equivalent(result.baseDirectory, override), "base_model link precedence");
        fs::remove(pika / "base_model");
        fs::create_directory_symlink(tmp.path / "missing", pika / "base_model");
        reject([&] { resolveModelSource(pika.string()); }, "invalid base_model");
        fs::remove(pika / "base_model");
        base(pika);
        check(fs::equivalent(resolveModelSource(pika.string()).baseDirectory, pika), "combined model directory");

        auto isolated = tmp.path / "isolated/adapter";
        adapter(isolated);
        reject([&] { resolveModelSource(isolated.string()); }, "--base-model PATH");
        // Preserve sibling lookup for symlink views instead of resolving the target's parent.
        auto view = tmp.path / "view";
        fs::create_directories(view);
        fs::create_directory_symlink(isolated, view / "renamed");
        fs::create_directory_symlink(original, view / "MiniMax-H3");
        check(fs::equivalent(resolveModelSource((view / "renamed").string()).baseDirectory, original), "symlink view");

        auto bad = manifest(false).object_items();
        bad["model"] = "unknown-adapter";
        write(pika / "fl2va/manifest.json", Json(bad).dump());
        reject([&] { resolveModelSource(pika.string()); }, "unsupported format");
        adapter(pika);
        bad = manifest(true).object_items();
        bad["base"] = Json::object{{"repo_id", "MiniMaxAI/MiniMax-H3"}, {"revision", "wrong"}};
        write(pika / "ref2va/manifest.json", Json(bad).dump());
        reject([&] { resolveModelSource(pika.string()); }, "unsupported base revision");
        write(pika / "ref2va/manifest.json", "{");
        reject([&] { resolveModelSource(pika.string()); }, "invalid manifest JSON");
        fs::remove(pika / "ref2va/manifest.json");
        reject([&] { resolveModelSource(pika.string()); }, "missing or invalid manifest");
        fs::remove(pika / "fl2va/manifest.json");
        write(pika / "fl2va/adapter.safetensors", "broken download");
        reject([&] { resolveModelSource(pika.string()); }, "missing or invalid manifest");
        std::cout << "PASS H3/Pika local model discovery, base selection, legacy flags and malformed releases\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
