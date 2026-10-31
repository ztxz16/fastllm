#include "models/minimax_h3/runtime.h"
#include "json11.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fastllm {
namespace minimax_h3 {
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
            if (i < 0 || size_t(i) >= n || seen[i] || (tags[i] != tag && !(group == &text && tags[i] == 0)))
                throw std::invalid_argument("invalid H3 modality indices");
            seen[i] = true;
        }
    }
    for (size_t i = 0; i < n; ++i)
        if (!seen[i] || timeIndices[i] < 0 || timeIndices[i] >= count)
            throw std::invalid_argument("invalid H3 timestep indices");
}
Transformer::Transformer(Runtime &rt, Runtime &fp32, const std::string &dir)
    : rt_(rt), fp32_(fp32), weights_(rt, componentWeights(dir)), fullWeights_(fp32, componentWeights(dir)),
      config_(dir) {
    if (!fp32.isFloat32())
        throw std::invalid_argument("H3 projection runtime must be FP32");
}
Tensor Transformer::timeEmbedding(const std::vector<float> &times) {
    auto x = fp32_.timestepEmbedding(times, config_.freqDim);
    x = fp32_.activation(fullWeights_.linear(x, "time_embedder.linear_1"));
    return fullWeights_.linear(x, "time_embedder.linear_2");
}
Tensor Transformer::projectModulation(int layer, const Tensor &input) {
    auto p = "transformer_blocks." + std::to_string(layer) + ".adaln_proj.linear";
    auto x = weights_.linear(input, p);
    x.rows *= 3;
    x.cols /= 3;
    return x;
}
Tensor Transformer::attention(const Tensor &x, const std::string &p, const Tensor *cs, const Tensor *sn,
                              Trace trace) {
    auto q = weights_.linear(x, p + ".to_q"), k = weights_.linear(x, p + ".to_k"),
         v = weights_.linear(x, p + ".to_v");
    if (cs) {
        auto qk = rt_.qkNormRotary(q, k, weights_.get(p + ".norm_q.weight"),
                                 weights_.get(p + ".norm_k.weight"), config_.qkEps, *cs, *sn,
                                 config_.headDim);
        q = qk.first;
        k = qk.second;
    } else {
        q = rt_.rms(q, weights_.get(p + ".norm_q.weight"), config_.qkEps, config_.headDim);
        k = rt_.rms(k, weights_.get(p + ".norm_k.weight"), config_.qkEps, config_.headDim);
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
    Tensor out;
    if (!x.isFloat32())
        out = Tensor(x.rows, config_.hidden, x.dtype);
    // Bound BF16 scratch and preserve the original sub-512 tail GEMM: folding
    // it into a larger batch selects a different reduction/rounding algorithm.
    // FP32 diagnostics retain all their original GEMM shapes.
    int chunk = x.isFloat32() ? 512 : 4096;
    for (int first = 0; first < x.rows;) {
        int count = std::min(chunk, x.rows - first);
        if (count > 512)
            count -= count % 512;
        auto y = weights_.linear(x.rowSlice(first, count), p + ".net.0.proj");
        if (out.data) {
            auto target = out.rowSlice(first, count);
            weights_.linear(rt_.siluMultiply(y), p + ".net.2", false, &target);
        } else {
            chunks.push_back(weights_.linear(rt_.siluMultiply(y), p + ".net.2"));
        }
        first += count;
    }
    if (out.data)
        return out;
    return chunks.size() == 1 ? chunks[0] : rt_.concat(chunks);
}
Tensor Transformer::refine(const Tensor &input) {
    auto x = weights_.linear(rt_.cast(input, rt_.dtype()), "context_embedder");
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
    if (trace)
        trace("norm1", rt_.rms(input, weights_.get(p + ".norm1.weight"), config_.normEps));
    auto x = rt_.rmsModulateRows(input, weights_.get(p + ".norm1.weight"), config_.normEps, mod, idx, 0, 1);
    if (trace)
        trace("modulated1", x);
    x = attention(x, p + ".attn", &cs, &sn, trace);
    if (trace)
        trace("attn_out", x);
    auto residual = rt_.residualRows(input, x, mod, idx, 2);
    if (trace)
        trace("residual1", residual);
    x = rt_.rmsModulateRows(residual, weights_.get(p + ".norm2.weight"), config_.normEps, mod, idx, 3, 4);
    if (trace)
        trace("modulated2", x);
    x = feedForward(x, p + ".ff");
    if (trace)
        trace("ff_out", x);
    return rt_.residualRows(residual, x, mod, idx, 5);
}
void Transformer::prepare(const std::vector<std::vector<float>> &plans,
                          std::function<void(int, int)> progress) {
    if (plans.empty())
        throw std::invalid_argument("empty H3 timestep plans");
    if (plans == plans_ && modulation_.size() == size_t(config_.layers) && outputModulation_.data) {
        if (progress)
            progress(config_.layers, config_.layers);
        return;
    }
    plans_.clear();
    modulation_.clear();
    // Keep each forward's GEMM batch size, including the first one-row plan.
    // Combining plans can select a different cuBLAS reduction and change BF16 rounding.
    std::vector<Tensor> inputs;
    for (const auto &times : plans)
        inputs.push_back(rt_.cast(fp32_.activation(timeEmbedding(times)), rt_.dtype()));
    for (int i = 0; i < config_.layers; ++i) {
        if (progress)
            progress(i, config_.layers);
        std::vector<Tensor> rows;
        for (const auto &input : inputs)
            rows.push_back(projectModulation(i, input));
        modulation_.push_back(rows.size() == 1 ? rows[0] : rt_.concat(rows));
        weights_.erasePrefix("transformer_blocks." + std::to_string(i) + ".adaln_proj.");
    }
    std::vector<Tensor> rows;
    for (const auto &input : inputs)
        rows.push_back(weights_.linear(input, "norm_out.linear"));
    outputModulation_ = rows.size() == 1 ? rows[0] : rt_.concat(rows);
    weights_.erasePrefix("norm_out.linear.");
    // FP32 diagnostics can reuse packed weights across steps; BF16 weights stay resident.
    weights_.enablePackedCache(size_t(16) << 30);
    plans_ = plans;
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
    Tensor x(layout.tags.size(), config_.hidden, rt_.isFloat32());
    rt_.scatter(x, refine(text), ti);
    rt_.scatter(x, rt_.cast(fullWeights_.linear(video, "proj_in"), x.dtype), vi);
    rt_.scatter(x, rt_.cast(fullWeights_.linear(audio, "audio_proj_in"), x.dtype), ai);
    auto rope = rt_.ropeTable(layout.positions, config_.ropeFreq, config_.ropeTheta);
    auto adaln = layout.timeIndices;
    for (size_t i = 0; i < adaln.size(); ++i)
        adaln[i] = adaln[i] * 3 + layout.tags[i];
    auto idx = rt_.indices(adaln);
    for (int i = 0; i < config_.layers; ++i) {
        x = block(i, x, modulation_[i], idx, rope.first, rope.second);
        if (trace)
            trace("block." + std::to_string(i), x);
        if (rt_.isFloat32())
            weights_.erasePrefix("transformer_blocks." + std::to_string(i) + ".");
    }
    x = rt_.rmsModulateRows(x, weights_.get("norm_out.norm.weight"), config_.finalEps,
                           outputModulation_, rt_.indices(layout.timeIndices), 0, 1);
    return {fullWeights_.linear(rt_.cast(rt_.gather(x, vi), DType::FP32), "proj_out"),
            fullWeights_.linear(rt_.cast(rt_.gather(x, ai), DType::FP32), "audio_proj_out")};
}
void Transformer::unload() {
    weights_.clear();
    fullWeights_.clear();
    modulation_.clear();
    outputModulation_ = {};
    plans_.clear();
}
} // namespace minimax_h3
} // namespace fastllm
