#include "models/minimax_h3/runtime.h"
#include "models/minimax_h3/parallel.h"
#include "json11.hpp"
#include <cmath>
#include <fstream>
#include <map>
#include <stdexcept>

namespace fastllm { namespace minimax_h3 {
namespace {
void require(bool ok, const std::string &message) {
    if (!ok) throw std::invalid_argument("Pika adapter: " + message);
}
json11::Json readManifest(const std::string &path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    require(bool(file) && file.tellg() > 0 && file.tellg() <= 65536, "missing or invalid manifest: " + path);
    std::string bytes(size_t(file.tellg()), '\0'), error;
    file.seekg(0);
    file.read(bytes.data(), bytes.size());
    auto j = json11::Json::parse(bytes, error);
    require(bool(file) && error.empty() && j.is_object(), "invalid manifest JSON");
    return j;
}
void validate(const std::string &dir, const TransformerConfig &c, bool reference) {
    auto j = readManifest(dir + "/manifest.json");
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
    std::ifstream file(dir + "/adapter.safetensors", std::ios::binary | std::ios::ate);
    require(bool(file), "missing adapter.safetensors");
    const auto size = file.tellg();
    uint64_t n = 0;
    file.seekg(0);
    file.read(reinterpret_cast<char *>(&n), 8);
    require(bool(file) && n > 0 && n <= 16 * 1024 * 1024 && uint64_t(size) >= n + 8, "invalid safetensors header");
    std::string bytes(n, '\0'), error;
    file.read(bytes.data(), n);
    auto header = json11::Json::parse(bytes, error);
    require(bool(file) && error.empty() && header.is_object(), "invalid safetensors JSON");
    std::map<std::string, std::pair<int, int>> expected;
    auto projection = [&](const std::string &p, int in, int out) {
        expected[p + ".lora_A.default.weight"] = {128, in};
        expected[p + ".lora_B.default.weight"] = {out, 128};
    };
    auto block = [&](const std::string &p) {
        for (const char *q : {"to_q", "to_k", "to_v"}) projection(p + ".attn." + q, c.hidden, c.heads * c.headDim);
        projection(p + ".attn.to_out.0", c.heads * c.headDim, c.hidden);
        projection(p + ".ff.net.0.proj", c.hidden, c.ffn * 2);
        projection(p + ".ff.net.2", c.ffn, c.hidden);
    };
    for (int i = 0; i < c.layers; ++i) block("transformer_blocks." + std::to_string(i));
    for (int i = 0; i < c.refiners; ++i) block("token_refiner.refiner_blocks." + std::to_string(i));
    size_t count = 0;
    for (const auto &kv : header.object_items()) {
        if (kv.first == "__metadata__") continue;
        auto it = expected.find(kv.first);
        require(it != expected.end(), "unexpected weight " + kv.first);
        const auto &v = kv.second;
        require(v["dtype"].string_value() == "F32" &&
                v["shape"] == json11::Json::array{it->second.first, it->second.second}, "invalid shape/dtype " + kv.first);
        const auto &offsets = v["data_offsets"].array_items();
        require(offsets.size() == 2 && offsets[0].is_number() && offsets[1].is_number(), "invalid offsets");
        double first = offsets[0].number_value(), last = offsets[1].number_value();
        require(first >= 0 && std::floor(first) == first && last <= double(uint64_t(size) - n - 8) &&
                last - first == double(int64_t(it->second.first) * it->second.second * 4), "truncated weight " + kv.first);
        ++count;
    }
    require(count == expected.size(), "missing LoRA weights");
}
} // namespace

std::vector<float> pikaSigmaSchedule(float shift) {
    require(std::isfinite(shift) && shift > 0, "invalid sigma shift");
    std::vector<float> result;
    for (int knot : {1000, 750, 500, 250, 100, 40, 0}) {
        // The release builds the ladder in float64, then casts to float32.
        double t = knot / 1000.;
        result.push_back(float(shift * t / (1. + (shift - 1.) * t)));
    }
    return result;
}
void Transformer::loadPikaAdapter(const std::string &dir, bool reference) {
    require(rt_.dtype() == DType::BF16, "requires BF16 inference");
    if (loraDirectory_ == dir) return;
    validate(dir, config_, reference);
    auto weights = std::make_unique<Weights>(rt_, dir + "/adapter.safetensors");
    auto column = [&](const std::string &p, int groups = 1) {
        // A sees the replicated input; B follows the base output partition.
        parallel_.columnWeights(*weights, p + ".lora_B.default", groups);
    };
    auto row = [&](const std::string &p) {
        // A consumes the local input shard. Its rank-128 output is summed
        // before BF16 rounding; B then runs on the replicated low-rank result.
        parallel_.rowWeights(*weights, p + ".lora_A.default");
    };
    auto block = [&](const std::string &p) {
        for (const char *q : {"to_q", "to_k", "to_v"}) column(p + ".attn." + q);
        row(p + ".attn.to_out.0");
        column(p + ".ff.net.0.proj", 2);
        row(p + ".ff.net.2");
    };
    for (int i = 0; i < config_.layers; ++i) block("transformer_blocks." + std::to_string(i));
    for (int i = 0; i < config_.refiners; ++i) block("token_refiner.refiner_blocks." + std::to_string(i));
    unload();
    lora_ = std::move(weights);
    loraDirectory_ = dir;
}
Tensor Transformer::linear(const Tensor &x, const std::string &p) {
    auto out = weights_.linear(x, p);
    if (lora_) {
        // Match PEFT: round A, B and both GEMMs to BF16, then round the sum.
        // alpha/rank is exactly one for the supported release.
        auto a = rt_.linear(x, lora_->get(p + ".lora_A.default.weight"));
        auto b = rt_.linear(a, lora_->get(p + ".lora_B.default.weight"));
        out = rt_.add(out, b);
    }
    return out;
}
Tensor Transformer::rowLinear(const Tensor &x, const std::string &p) {
    if (parallel_.size() == 1) return linear(x, p);
    // Keep both branches' sums in FP32 until all input partitions contribute.
    // Rounding each base partial to BF16 first would introduce an extra
    // precision boundary absent from the released unmerged LoRA calculation.
    auto projection = parallel_;
    projection.rowReduction = nomas::RowReduction::FP32;
    auto out = projection.row(rt_, weights_, x, p);
    if (lora_) {
        auto a = projection.row(rt_, *lora_, x, p + ".lora_A.default");
        auto b = rt_.linear(a, lora_->get(p + ".lora_B.default.weight"));
        out = rt_.add(out, b);
    }
    return out;
}
void Pipeline::loadPika(const std::string &dir) {
    require(rt_.dtype() == DType::BF16, "requires BF16 inference");
    transformer_.loadPikaAdapter(dir + "/fl2va", false);
    if (!referenceTransformer_)
        referenceTransformer_ = std::make_unique<Transformer>(rt_, full_, directory_ + "/transformer_ref");
    referenceTransformer_->loadPikaAdapter(dir + "/ref2va", true);
    if (parallel_) parallel_->loadPika(dir);
    pikaDirectory_ = dir;
}
}} // namespace fastllm::minimax_h3
