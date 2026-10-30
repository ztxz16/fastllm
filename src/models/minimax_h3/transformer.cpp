#include "models/minimax_h3/runtime.h"
#include "json11.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fastllm {
namespace minimax_h3 {
namespace {
Tensor timeProjection(Runtime &rt, Weights &weights, const Tensor &input, const std::string &prefix) {
    // Each T2VA forward has at most two distinct noise levels. Keep that GEMM
    // shape while caching the whole schedule, rather than changing its reduction.
    std::vector<Tensor> rows;
    for (int first = 0; first < input.rows; first += 2)
        rows.push_back(weights.linear(input.rowSlice(first, std::min(2, input.rows - first)), prefix));
    return rows.size() == 1 ? rows[0] : rt.concat(rows);
}
} // namespace
std::string componentWeights(const std::string &dir) {
    for (const char *name : {"diffusion_pytorch_model.safetensors", "model.safetensors"})
        if (std::filesystem::is_regular_file(dir + "/" + name))
            return dir + "/" + name;
    return dir;
}
TransformerConfig::TransformerConfig(const std::string &dir) {
    std::ifstream file(dir + "/config.json");
    if (!file)
        throw std::runtime_error("missing H3 transformer config");
    std::string text((std::istreambuf_iterator<char>(file)), {}), error;
    auto config = json11::Json::parse(text, error);
    if (!error.empty())
        throw std::runtime_error("invalid H3 transformer config: " + error);
    auto integer = [&](const char *key, int &out) {
        if (config[key].is_number())
            out = config[key].int_value();
    };
    auto real = [&](const char *key, float &out) {
        if (config[key].is_number())
            out = config[key].number_value();
    };
    integer("hidden_size", hidden);
    integer("num_attention_heads", heads);
    integer("attention_head_dim", headDim);
    integer("num_layers", layers);
    integer("num_refiner_layers", refiners);
    integer("ffn_dim", ffn);
    integer("audio_in_channels", audioDim);
    integer("text_dim", textDim);
    integer("freq_dim", freqDim);
    integer("time_embed_dim", timeDim);
    integer("rope_freq_dim", ropeFreq);
    real("norm_eps", normEps);
    real("qk_norm_eps", qkEps);
    real("final_norm_eps", finalEps);
    real("rope_theta", ropeTheta);
    videoDim = config["in_channels"].int_value();
    for (auto &v : config["patch_size"].array_items())
        videoDim *= v.int_value();
    if (hidden <= 0 || heads <= 0 || headDim <= 0 || layers <= 0 || ffn <= 0 || videoDim <= 0 ||
        freqDim <= 0 || freqDim % 2 || timeDim <= 0 || ropeFreq <= 0 || ropeFreq * 6 > headDim)
        throw std::runtime_error("invalid H3 transformer dimensions");
}
void PackedLayout::validate(int count) const {
    size_t n = tags.size();
    if (!n || positions.size() != n * 3 || timeIndices.size() != n || video.empty() || audio.empty() ||
        text.empty())
        throw std::invalid_argument("invalid packed H3 layout");
    std::vector<bool> seen(n);
    for (auto group : {&video, &text, &audio}) {
        int tag = group == &video ? 0 : group == &text ? 1 : 2;
        for (int i : *group) {
            if (i < 0 || size_t(i) >= n || seen[i] || tags[i] != tag)
                throw std::invalid_argument("invalid H3 modality indices");
            seen[i] = true;
        }
    }
    for (size_t i = 0; i < n; ++i)
        if (!seen[i] || timeIndices[i] < 0 || timeIndices[i] >= count)
            throw std::invalid_argument("invalid H3 timestep indices");
}
Transformer::Transformer(Runtime &rt, const std::string &dir)
    : rt_(rt), weights_(rt, componentWeights(dir)), config_(dir) {}
Tensor Transformer::timeEmbedding(const std::vector<float> &times) {
    auto x = rt_.timestepEmbedding(times, config_.freqDim);
    x = rt_.activation(timeProjection(rt_, weights_, x, "time_embedder.linear_1"));
    return timeProjection(rt_, weights_, x, "time_embedder.linear_2");
}
Tensor Transformer::modulation(int layer, const Tensor &temb) {
    auto p = "transformer_blocks." + std::to_string(layer) + ".adaln_proj.linear";
    auto x = timeProjection(rt_, weights_, rt_.activation(temb), p);
    x.rows *= 3;
    x.cols /= 3;
    return x;
}
Tensor Transformer::attention(const Tensor &x, const std::string &p, const Tensor *cs, const Tensor *sn,
                              Trace trace) {
    auto q = weights_.linear(x, p + ".to_q"), k = weights_.linear(x, p + ".to_k"),
         v = weights_.linear(x, p + ".to_v");
    q = rt_.rms(q, weights_.get(p + ".norm_q.weight"), config_.qkEps, config_.headDim);
    k = rt_.rms(k, weights_.get(p + ".norm_k.weight"), config_.qkEps, config_.headDim);
    if (cs) {
        q = rt_.rotary(q, *cs, *sn, config_.headDim);
        k = rt_.rotary(k, *cs, *sn, config_.headDim);
    }
    if (trace) {
        trace("q", q);
        trace("k", k);
        trace("v", v);
    }
    auto out = rt_.selfAttention(q, k, v, config_.heads);
    if (trace)
        trace("attention", out);
    return weights_.linear(out, p + ".to_out.0");
}
Tensor Transformer::feedForward(const Tensor &x, const std::string &p) {
    std::vector<Tensor> chunks;
    for (int first = 0; first < x.rows; first += 512) {
        auto y = weights_.linear(x.rowSlice(first, std::min(512, x.rows - first)), p + ".net.0.proj");
        auto hidden = rt_.columns(y, 0, config_.ffn), gate = rt_.columns(y, config_.ffn, config_.ffn);
        chunks.push_back(weights_.linear(rt_.multiply(hidden, rt_.activation(gate)), p + ".net.2"));
    }
    return chunks.size() == 1 ? chunks[0] : rt_.concat(chunks);
}
Tensor Transformer::refine(const Tensor &input) {
    auto x = weights_.linear(input, "context_embedder");
    for (int i = 0; i < config_.refiners; ++i) {
        auto p = "token_refiner.refiner_blocks." + std::to_string(i);
        x = rt_.add(x,
                    attention(rt_.rms(x, weights_.get(p + ".norm1.weight"), config_.normEps), p + ".attn"));
        x = rt_.add(x,
                    feedForward(rt_.rms(x, weights_.get(p + ".norm2.weight"), config_.normEps), p + ".ff"));
    }
    return rt_.rms(x, weights_.get("token_refiner.final_norm.weight"), config_.finalEps);
}
Tensor Transformer::block(int layer, const Tensor &input, const Tensor &mod, const Tensor &idx,
                          const Tensor &cs, const Tensor &sn, Trace trace) {
    auto p = "transformer_blocks." + std::to_string(layer);
    auto x = rt_.rms(input, weights_.get(p + ".norm1.weight"), config_.normEps);
    if (trace)
        trace("norm1", x);
    x = rt_.modulateRows(x, mod, idx, 0, 1);
    if (trace)
        trace("modulated1", x);
    x = attention(x, p + ".attn", &cs, &sn, trace);
    if (trace)
        trace("attn_out", x);
    auto residual = rt_.residualRows(input, x, mod, idx, 2);
    if (trace)
        trace("residual1", residual);
    x = rt_.rms(residual, weights_.get(p + ".norm2.weight"), config_.normEps);
    x = rt_.modulateRows(x, mod, idx, 3, 4);
    if (trace)
        trace("modulated2", x);
    x = feedForward(x, p + ".ff");
    if (trace)
        trace("ff_out", x);
    return rt_.residualRows(residual, x, mod, idx, 5);
}
void Transformer::prepare(const std::vector<float> &times, std::function<void(int, int)> progress) {
    modulation_.clear();
    auto temb = timeEmbedding(times);
    for (int i = 0; i < config_.layers; ++i) {
        if (progress)
            progress(i, config_.layers);
        modulation_.push_back(modulation(i, temb));
        weights_.erasePrefix("transformer_blocks." + std::to_string(i) + ".adaln_proj.");
    }
    outputModulation_ = timeProjection(rt_, weights_, rt_.activation(temb), "norm_out.linear");
    weights_.erasePrefix("norm_out.linear.");
    // Reuse spare VRAM across denoising steps without changing FP32 arithmetic.
    // Enable only after the one-use AdaLN weights have been discarded.
    weights_.enablePackedCache(size_t(16) << 30);
}
std::pair<Tensor, Tensor> Transformer::forward(const Tensor &video, const Tensor &audio, const Tensor &text,
                                               const PackedLayout &layout, Trace trace) {
    if (modulation_.size() != size_t(config_.layers))
        throw std::logic_error("H3 timestep tables not prepared");
    layout.validate(outputModulation_.rows);
    if (video.rows != int(layout.video.size()) || audio.rows != int(layout.audio.size()) ||
        text.rows != int(layout.text.size()))
        throw std::invalid_argument("H3 input row counts disagree with layout");
    auto vi = rt_.indices(layout.video), ai = rt_.indices(layout.audio), ti = rt_.indices(layout.text);
    Tensor x(layout.tags.size(), config_.hidden, true);
    rt_.scatter(x, refine(text), ti);
    rt_.scatter(x, weights_.linear(video, "proj_in"), vi);
    rt_.scatter(x, weights_.linear(audio, "audio_proj_in"), ai);
    auto rope = rt_.ropeTable(layout.positions, config_.ropeFreq, config_.ropeTheta);
    auto adaln = layout.timeIndices;
    for (size_t i = 0; i < adaln.size(); ++i)
        adaln[i] = adaln[i] * 3 + layout.tags[i];
    auto idx = rt_.indices(adaln);
    for (int i = 0; i < config_.layers; ++i) {
        x = block(i, x, modulation_[i], idx, rope.first, rope.second);
        if (trace)
            trace("block." + std::to_string(i), x);
        weights_.erasePrefix("transformer_blocks." + std::to_string(i) + ".");
    }
    x = rt_.rms(x, weights_.get("norm_out.norm.weight"), config_.finalEps);
    x = rt_.modulateRows(x, outputModulation_, rt_.indices(layout.timeIndices), 0, 1);
    return {weights_.linear(rt_.gather(x, vi), "proj_out"),
            weights_.linear(rt_.gather(x, ai), "audio_proj_out")};
}
void Transformer::unload() {
    weights_.clear();
    modulation_.clear();
    outputModulation_ = {};
}
} // namespace minimax_h3
} // namespace fastllm
