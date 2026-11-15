#include "models/qwen_image/runtime.h"
#include "quantized.h"
#include <array>
#include <cmath>
#include <filesystem>
#include <numeric>
#include <stdexcept>

namespace fastllm {
namespace qwen_image {
std::string componentWeights(const std::string &dir) {
    for (const char *name : {"diffusion_pytorch_model.safetensors", "model.safetensors"})
        if (std::filesystem::is_regular_file(dir + "/" + name))
            return dir + "/" + name;
    if (std::filesystem::is_directory(dir)) {
        std::string gguf;
        for (const auto &entry : std::filesystem::directory_iterator(dir))
            if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
                if (!gguf.empty()) throw std::invalid_argument("multiple GGUF files in component directory");
                gguf = entry.path().string();
            }
        if (!gguf.empty()) return gguf;
    }
    return dir;
}
TextEncoder::TextEncoder(Runtime &rt, const std::string &dir)
    : rt_(rt), weights_(rt, componentWeights(dir)) {}
std::vector<Tensor> TextEncoder::vision(const Image &image) {
    return nomas::qwenVision(rt_, weights_, image.width, image.height, image.rgba);
}

Tensor TextEncoder::encode(const std::vector<int> &tokens, const std::vector<Image> &images) {
    if (tokens.empty() || tokens.size() > 4096)
        throw std::invalid_argument("prompt requires 1..4096 tokens");
    const std::string root = "model.language_model.";
    auto x = rt_.embedding(weights_.get(root + "embed_tokens.weight"), tokens);
    std::vector<int> positions(tokens.size());
    std::iota(positions.begin(), positions.end(), 0);
    std::vector<std::vector<Tensor>> visual;
    std::vector<int> starts;
    if (!images.empty()) {
        positions.resize(tokens.size() * 3);
        int cursor = 0, position = 0;
        for (auto &image : images) {
            int gh = image.height / 32, gw = image.width / 32;
            while (cursor < int(tokens.size()) && tokens[cursor] != 151655) {
                for (int a = 0; a < 3; ++a)
                    positions[a * tokens.size() + cursor] = position;
                ++position;
                ++cursor;
            }
            starts.push_back(cursor);
            for (int i = 0; i < gh * gw; ++i) {
                if (cursor >= int(tokens.size()) || tokens[cursor] != 151655)
                    throw std::invalid_argument("image slot count mismatch");
                positions[cursor] = position;
                positions[tokens.size() + cursor] = position + i / gw;
                positions[tokens.size() * 2 + cursor] = position + i % gw;
                ++cursor;
            }
            position += std::max(gh, gw);
            visual.push_back(vision(image));
        }
        for (; cursor < int(tokens.size()); ++cursor, ++position)
            for (int a = 0; a < 3; ++a)
                positions[a * tokens.size() + cursor] = position;
    }
    auto applyVisual = [&](const Tensor &input, int deep) {
        if (visual.empty())
            return input;
        std::vector<Tensor> parts;
        int cursor = 0;
        for (size_t i = 0; i < visual.size(); ++i) {
            if (starts[i] > cursor)
                parts.push_back(input.rowSlice(cursor, starts[i] - cursor));
            auto &features = visual[i][deep < 0 ? 0 : deep + 1];
            parts.push_back(deep < 0 ? features
                                     : rt_.add(input.rowSlice(starts[i], features.rows), features));
            cursor = starts[i] + features.rows;
        }
        if (cursor < input.rows)
            parts.push_back(input.rowSlice(cursor, input.rows - cursor));
        return rt_.concat(parts);
    };
    x = applyVisual(x, -1);
    auto ropePositions = rt_.indices(positions);
    for (int layer = 0; layer < 36; ++layer) {
        auto p = root + "layers." + std::to_string(layer) + ".";
        auto nx = rt_.norm(x, true, &weights_.get(p + "input_layernorm.weight"));
        auto q = weights_.linear(nx, p + "self_attn.q_proj"), k = weights_.linear(nx, p + "self_attn.k_proj"),
             v = weights_.linear(nx, p + "self_attn.v_proj");
        q = rt_.norm(q, true, &weights_.get(p + "self_attn.q_norm.weight"), nullptr, 1e-6f, 128);
        k = rt_.norm(k, true, &weights_.get(p + "self_attn.k_norm.weight"), nullptr, 1e-6f, 128);
        q = rt_.textRope(q, ropePositions, 5000000.f);
        k = rt_.textRope(k, ropePositions, 5000000.f);
        x = rt_.add(
            x, weights_.linear(rt_.attention(q, k, v, 32, nullptr, 128, 8, true), p + "self_attn.o_proj"));
        nx = rt_.norm(x, true, &weights_.get(p + "post_attention_layernorm.weight"));
        auto gate = weights_.linear(nx, p + "mlp.gate_proj");
        x = rt_.add(x, weights_.linear(rt_.siluMultiply(gate, weights_.linear(nx, p + "mlp.up_proj")),
                                       p + "mlp.down_proj"));
        if (layer < 3)
            x = applyVisual(x, layer);
    }
    // The diffusion checkpoint consumes the final decoder residual, before norm.
    return x;
}
DiT::DiT(Runtime &rt, const std::string &dir, nomas::ParallelContext parallel)
    : rt_(rt), parallel_(parallel) {
    if (parallel.size() > 8 || 32 % parallel.size())
        throw std::invalid_argument("Qwen Image supports TP sizes 1, 2, 4 and 8");
    const auto checkpoint = componentWeights(dir);
    quantized_ = QuantizedProjections::load(rt, checkpoint, parallel_);
    if (!quantized_ || !quantized_->isGguf())
        weights_ = std::make_unique<Weights>(rt, checkpoint);
    if (weights_) {
        for (int i = 0; i < 32; ++i) {
            const auto p = "transformer_blocks." + std::to_string(i) + ".";
            for (const char *suffix : {"attn.to_q", "attn.to_k", "attn.to_v", "img_mlp.gate_layer", "img_mlp.proj"})
                parallel_.columnWeights(*weights_, p + suffix);
            for (const char *suffix : {"attn.to_out.0", "img_mlp.out"})
                parallel_.rowWeights(*weights_, p + suffix);
        }
    }
}
DiT::~DiT() = default;
const Tensor &DiT::weight(const std::string &name) {
    return weights_ ? weights_->get(name) : quantized_->get(name);
}
Tensor DiT::linear(const Tensor &x, const std::string &prefix, Tensor *output) {
    if (quantized_ && quantized_->contains(prefix))
        return quantized_->linear(x, {prefix}, output);
    return weights_->linear(x, prefix, false, output);
}
Tensor DiT::rowLinear(const Tensor &x, const std::string &prefix) {
    if (parallel_.size() == 1) return linear(x, prefix);
    if (!quantized_ || !quantized_->contains(prefix)) {
        if (!x.isFloat32()) {
            // Keep FP32 GEMM accumulation, then communicate BF16 partials.
            auto context = parallel_;
            context.rowReduction = nomas::RowReduction::Activation;
            Tensor partial(x.rows, weights_->get(prefix + ".weight").rows, x.dtype);
            context.rowPartial(rt_, *weights_, x, prefix, partial);
            return context.group->sumActivation(std::move(partial), context.rank);
        }
        return parallel_.row(rt_, *weights_, x, prefix);
    }
    return quantized_->row(x, prefix);
}
void DiT::unload() {
    if (quantized_)
        quantized_->clear();
    projections_.clear();
    if (weights_) weights_->clear();
}
size_t DiT::loadedBytes() const {
    size_t bytes = (weights_ ? weights_->loadedBytes() : 0) + (quantized_ ? quantized_->loadedBytes() : 0);
    for (const auto &entry : projections_)
        bytes += entry.second.weight.bytes() + entry.second.bias.bytes();
    return bytes;
}
Tensor DiT::packedLinear(const Tensor &x, const std::vector<std::string> &prefixes) {
    if (quantized_ && prefixes.size() == 2) {
        const auto suffix = std::string("gate_layer");
        const auto &first = prefixes.front();
        if (first.size() > suffix.size() && first.substr(first.size() - suffix.size()) == suffix) {
            const auto stem = first.substr(0, first.size() - suffix.size());
            if (prefixes[1] == stem + "proj" && quantized_->contains(stem + "gate_up"))
                return quantized_->linear(x, {stem + "gate_up"});
        }
    }
    if (quantized_ && quantized_->contains(prefixes.front()))
        return quantized_->linear(x, prefixes);
    auto &projection = projections_[prefixes.front()];
    if (!projection.weight.rows) {
        std::vector<Tensor> weights, biases;
        for (const auto &prefix : prefixes) {
            weights.push_back(weight(prefix + ".weight"));
            if (auto bias = weights_->optional(prefix + ".bias"))
                biases.push_back(*bias);
        }
        if (!biases.empty() && biases.size() != weights.size())
            throw std::invalid_argument("packed projection requires consistent bias layout");
        projection.weight = rt_.concat(weights);
        if (!biases.empty())
            projection.bias = rt_.concat(biases);
        for (const auto &prefix : prefixes)
            weights_->erasePrefix(prefix + ".");
    }
    return rt_.linear(x, projection.weight, projection.bias.rows ? &projection.bias : nullptr);
}
Tensor DiT::forward(const Tensor &latents, const Condition &cond, float timestep, int height, int width,
                    KVCache *cache) {
    const int target = height * width;
    if (height < 1 || width < 1 || target != latents.rows || latents.cols != 64 ||
        cond.features.cols != 4096 ||
        (!cond.imageSlots.empty() && cond.imageSlots.size() != size_t(cond.features.rows)))
        throw std::invalid_argument("invalid Qwen-Image DiT input layout");
    bool cached = cache && !cache->keys.empty();
    if (cached && (cache->keys.size() != 32 || cache->values.size() != 32))
        throw std::invalid_argument("invalid prefix cache");
    // Prefix errors persist in every denoising step. Expand GGUF MMQ only
    // after this request's conditioning cache has been built.
    if (quantized_)
        quantized_->setPrefill(!cached);
    auto image = linear(!cached && cond.latents.rows ? rt_.concat({cond.latents, latents}) : latents,
                        "img_in");
    Tensor text;
    if (!cached) {
        text = rt_.specialNorm(cond.features, weight("txt_in.text_norm.weight"), 0);
        text = linear(rt_.activation(linear(text, "txt_in.in_layer"), true),
                      "txt_in.out_layer");
    }
    struct Segment {
        int start, end;
        bool text;
    };
    std::vector<Segment> segments;
    std::vector<std::array<int, 3>> positions;
    std::vector<Tensor> parts;
    int feature = 0, imageOffset = 0, position = 0, imageIndex = 0;
    auto addImage = [&](int h, int w) {
        for (int i = 0; i < h * w; ++i)
            positions.push_back({position, i / w - (h - h / 2), i % w - (w - w / 2)});
        position += std::max(h, w);
    };
    while (feature < cond.features.rows) {
        int begin = positions.size();
        if (cond.imageSlots.empty() || !cond.imageSlots[feature]) {
            int first = feature;
            while (feature < cond.features.rows && (cond.imageSlots.empty() || !cond.imageSlots[feature])) {
                positions.push_back({position, position, position});
                ++position;
                ++feature;
            }
            if (!cached)
                parts.push_back(text.rowSlice(first, feature - first));
            segments.push_back({begin, int(positions.size()), true});
        } else {
            if (imageIndex >= int(cond.shapes.size()))
                throw std::invalid_argument("missing condition image shape");
            auto [h, w] = cond.shapes[imageIndex++];
            int count = h * w, slots = count / 4;
            if (count % 4 || feature + slots > cond.features.rows)
                throw std::invalid_argument("condition image slots mismatch");
            for (int i = 0; i < slots; ++i)
                if (!cond.imageSlots[feature + i])
                    throw std::invalid_argument("interrupted image block");
            if (!cached)
                parts.push_back(image.rowSlice(imageOffset, count));
            feature += slots;
            imageOffset += count;
            addImage(h, w);
            segments.push_back({begin, int(positions.size()), false});
        }
    }
    if (imageIndex != int(cond.shapes.size()) || imageOffset != cond.latents.rows)
        throw std::invalid_argument("condition image latent count mismatch");
    const int prefix = positions.size();
    addImage(height, width);
    if (!cached)
        parts.push_back(image.rowSlice(imageOffset, target));
    auto x = cached ? image : rt_.concat(parts);
    // The checkpoint uses separate target and zero-time condition modulation.
    const float time = timestep;
    std::vector<float> embedding(512);
    for (int r = 0; r < 2; ++r)
        for (int i = 0; i < 128; ++i) {
            float freq = std::exp(-std::log(10000.f) * float(i) / 128.f),
                  angle = (r == 0 ? time * 1000.f : 0.f) * freq;
            embedding[r * 256 + i] = std::cos(angle);
            embedding[r * 256 + 128 + i] = std::sin(angle);
        }
    auto temb = linear(rt_.activation(linear(rt_.upload(embedding.data(), 2, 256),
                                           "time_text_embed.timestep_embedder.linear_1")),
                       "time_text_embed.timestep_embedder.linear_2");
    auto modulation = linear(rt_.activation(temb), "modulation.1");
    nomas::RopeTable rope;
    if (cached) {
        if (cache->rope.cos.rows != prefix + target || cache->rope.sin.rows != prefix + target)
            throw std::invalid_argument("prefix cache RoPE shape mismatch");
        rope = {cache->rope.cos.rowSlice(prefix, target), cache->rope.sin.rowSlice(prefix, target)};
    } else {
        std::vector<float> cs(size_t(x.rows) * 128), sn(cs.size());
        for (int i = 0; i < x.rows; ++i) {
            auto pos = positions[i];
            int offset = 0;
            const int dims[3] = {16, 56, 56};
            for (int a = 0; a < 3; ++a) {
                for (int j = 0; j < dims[a]; j += 2) {
                    float angle = pos[a] / std::pow(10000.f, float(j) / dims[a]);
                    for (int k = 0; k < 2; ++k) {
                        cs[size_t(i) * 128 + offset + j + k] = std::cos(angle);
                        sn[size_t(i) * 128 + offset + j + k] = std::sin(angle);
                    }
                }
                offset += dims[a];
            }
        }
        rope = rt_.prepareRope(cs.data(), sn.data(), x.rows, 128);
        if (cache)
            cache->rope = rope;
    }
    const int heads = 32 / parallel_.size(), channels = heads * 128;
    for (int layer = 0; layer < 32; ++layer) {
        auto p = "transformer_blocks." + std::to_string(layer) + ".";
        auto nx = rt_.normScale(x, modulation, cached ? 0 : prefix, 0);
        Tensor q(x.rows, channels, x.dtype), k, v;
        if (cached && cache->full) {
            k = cache->keys[layer].rowSlice(prefix, target);
            v = cache->values[layer].rowSlice(prefix, target);
        } else {
            k = Tensor(x.rows, channels, x.dtype);
            v = Tensor(x.rows, channels, x.dtype);
        }
        if (!rt_.isFloat32()) {
            auto qkv = packedLinear(nx, {p + "attn.to_q", p + "attn.to_k", p + "attn.to_v"});
            rt_.qkvNormRope(qkv, weight(p + "attn.norm_q.weight"),
                            weight(p + "attn.norm_k.weight"), rope, q, k, v);
        } else {
            auto rawQ = linear(nx, p + "attn.to_q"), rawK = linear(nx, p + "attn.to_k");
            linear(nx, p + "attn.to_v", &v);
            rt_.qkNormRope(rawQ, rawK, weight(p + "attn.norm_q.weight"),
                           weight(p + "attn.norm_k.weight"), rope, q, k);
        }
        Tensor a;
        if (cached) {
            k = cache->full ? cache->keys[layer] : rt_.concat({cache->keys[layer], k});
            v = cache->full ? cache->values[layer] : rt_.concat({cache->values[layer], v});
            a = rt_.attention(q, k, v, heads);
        } else {
            if (cache) {
                cache->keys.push_back(cache->full ? k : rt_.concat({k.rowSlice(0, prefix)}));
                cache->values.push_back(cache->full ? v : rt_.concat({v.rowSlice(0, prefix)}));
            }
            std::vector<Tensor> outputs;
            for (auto &segment : segments) {
                int count = segment.end - segment.start;
                std::vector<uint8_t> mask;
                if (segment.text && segment.start > 0) {
                    mask.resize(size_t(count) * segment.end);
                    for (int row = 0; row < count; ++row)
                        for (int col = 0; col < segment.end; ++col)
                            mask[size_t(row) * segment.end + col] = col <= row + segment.start;
                }
                outputs.push_back(rt_.attention(
                    q.rowSlice(segment.start, count), k.rowSlice(0, segment.end), v.rowSlice(0, segment.end),
                    heads, mask.empty() ? nullptr : mask.data(), 128, heads, segment.text && segment.start == 0));
            }
            outputs.push_back(rt_.attention(q.rowSlice(prefix, target), k, v, heads));
            a = rt_.concat(outputs);
        }
        x = rt_.gatedAdd(x, rowLinear(a, p + "attn.to_out.0"), modulation, cached ? 0 : prefix, 1);
        nx = rt_.normScale(x, modulation, cached ? 0 : prefix, 2);
        Tensor hidden;
        if (!rt_.isFloat32()) {
            hidden = rt_.siluMultiply(packedLinear(nx, {p + "img_mlp.gate_layer", p + "img_mlp.proj"}), true);
        } else {
            auto gate = linear(nx, p + "img_mlp.gate_layer");
            hidden = rt_.siluMultiply(gate, linear(nx, p + "img_mlp.proj"));
        }
        auto mlp = rowLinear(hidden, p + "img_mlp.out");
        x = rt_.gatedAdd(x, mlp, modulation, cached ? 0 : prefix, 3);
    }
    auto scale = linear(rt_.activation(temb), "norm_out.linear");
    auto out = linear(rt_.normScale(x, scale, cached ? 0 : prefix, 0), "proj_out");
    return cached ? out : out.rowSlice(prefix, target);
}
VAE::VAE(Runtime &rt, const std::string &dir) : rt_(rt), weights_(rt, componentWeights(dir)) {}
void VAE::unload() {
    packedConvs_.clear();
    weights_.clear();
}
size_t VAE::loadedBytes() const {
    size_t bytes = weights_.loadedBytes();
    for (const auto &entry : packedConvs_)
        bytes += entry.second.bytes();
    return bytes;
}
Tensor VAE::conv(const Tensor &x, const std::string &p, int h, int w, int stride, int padEnd) {
    auto found = packedConvs_.find(p);
    bool packed = found != packedConvs_.end();
    auto weight = packed ? found->second : weights_.get(p + ".weight");
    int kernel = int(std::sqrt(weight.cols / x.cols));
    if (!x.isFloat32() && kernel > 1) {
        if (!packed) {
            weight = rt_.packConvWeight(weight, x.cols);
            packedConvs_.emplace(p, weight);
            // Keep only the reordered representation; low-memory unloading owns it.
            weights_.erasePrefix(p + ".weight");
        }
        return rt_.packedConv2d(x, weight, weights_.optional(p + ".bias"), h, w, kernel, stride,
                                stride == 1 ? kernel / 2 : 0, padEnd);
    }
    return rt_.conv2d(x, weight, weights_.optional(p + ".bias"), h, w, kernel, stride,
                      stride == 1 ? kernel / 2 : 0, padEnd);
}
Tensor VAE::residual(const Tensor &x, const std::string &p, int h, int w) {
    auto shortcut = weights_.optional(p + ".conv_shortcut.weight") ? conv(x, p + ".conv_shortcut", h, w) : x;
    auto y = conv(rt_.vaeNormSilu(x, weights_.get(p + ".norm1.gamma")), p + ".conv1", h, w);
    y = conv(rt_.vaeNormSilu(y, weights_.get(p + ".norm2.gamma")), p + ".conv2", h, w);
    return rt_.add(y, shortcut);
}
Tensor VAE::middle(const Tensor &input, const std::string &p, int h, int w) {
    auto x = residual(input, p + ".resnets.0", h, w);
    auto nx = rt_.specialNorm(x, weights_.get(p + ".attentions.0.norm.gamma"), 1);
    auto qkv = conv(nx, p + ".attentions.0.to_qkv", h, w);
    auto q = rt_.columns(qkv, 0, x.cols), k = rt_.columns(qkv, x.cols, x.cols),
         v = rt_.columns(qkv, x.cols * 2, x.cols);
    x = rt_.add(x, conv(rt_.attention(q, k, v, 1), p + ".attentions.0.proj", h, w));
    return residual(x, p + ".resnets.1", h, w);
}
Tensor VAE::decode(const Tensor &latent, int h, int w) {
    auto x = conv(conv(latent, "post_quant_conv", h, w), "decoder.conv_in", h, w);
    x = middle(x, "decoder.mid_block", h, w);
    for (int i = 0; i < 5; ++i) {
        auto p = "decoder.up_blocks." + std::to_string(i);
        auto skip = x;
        for (int j = 0; j < 3; ++j)
            x = residual(x, p + ".resnets." + std::to_string(j), h, w);
        if (i < 4) {
            x = conv(rt_.resize2x(x, h, w), p + ".upsampler.resample.1", h * 2, w * 2);
            x = rt_.add(x, rt_.duplicateUpsample(skip, h, w, x.cols, i < 3 ? 2 : 1));
            h *= 2;
            w *= 2;
        }
    }
    return conv(rt_.vaeNormSilu(x, weights_.get("decoder.norm_out.gamma")),
                "decoder.conv_out", h, w);
}
Tensor VAE::encode(const Tensor &pixels, int h, int w) {
    auto x = conv(pixels, "encoder.conv_in", h, w);
    for (int i = 0; i < 5; ++i) {
        auto p = "encoder.down_blocks." + std::to_string(i);
        auto skip = x;
        for (int j = 0; j < 2; ++j)
            x = residual(x, p + ".resnets." + std::to_string(j), h, w);
        int fs = i < 4 ? 2 : 1, ft = (i > 0 && i < 4) ? 2 : 1;
        if (i < 4)
            x = conv(x, p + ".downsampler.resample.1", h, w, 2, 1);
        x = rt_.add(x, rt_.averageDownsample(skip, h, w, x.cols, fs, ft));
        h /= fs;
        w /= fs;
    }
    x = middle(x, "encoder.mid_block", h, w);
    x = conv(rt_.vaeNormSilu(x, weights_.get("encoder.norm_out.gamma")),
             "encoder.conv_out", h, w);
    return rt_.columns(conv(x, "quant_conv", h, w), 0, 64);
}
} // namespace qwen_image
} // namespace fastllm
