#include "models/qwen_image/runtime.h"
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
    return dir;
}
TextEncoder::TextEncoder(Runtime &rt, const std::string &dir)
    : rt_(rt), weights_(rt, componentWeights(dir)) {}
std::vector<Tensor> TextEncoder::vision(const Image &image) {
    const int gh = image.height / 16, gw = image.width / 16, n = gh * gw;
    if (image.width % 32 || image.height % 32 || image.rgba.size() != size_t(image.width) * image.height * 4)
        throw std::invalid_argument("vision input must be RGBA with dimensions divisible by 32");
    const std::string root = "model.visual.";
    std::vector<float> patches(size_t(n) * 1536), positionsEmbedding(size_t(n) * 1152);
    std::vector<int> positions(size_t(n) * 2);
    auto learned = rt_.download(weights_.get(root + "pos_embed.weight"));
    for (int by = 0, i = 0; by < gh / 2; ++by)
        for (int bx = 0; bx < gw / 2; ++bx)
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx, ++i) {
                    int py = by * 2 + dy, px = bx * 2 + dx;
                    positions[i] = py;
                    positions[n + i] = px;
                    for (int c = 0; c < 3; ++c)
                        for (int t = 0; t < 2; ++t)
                            for (int y = 0; y < 16; ++y)
                                for (int x = 0; x < 16; ++x) {
                                    size_t source = (size_t(py * 16 + y) * image.width + px * 16 + x) * 4;
                                    int alpha = image.rgba[source + 3],
                                        value = (image.rgba[source + c] * alpha + 255 * (255 - alpha) + 127) /
                                                255;
                                    patches[size_t(i) * 1536 + ((c * 2 + t) * 16 + y) * 16 + x] =
                                        (value * (1.f / 255.f) - .5f) * 2.f;
                                }
                    float fy = gh == 1 ? 0 : float(py) * 47.f / (gh - 1),
                          fx = gw == 1 ? 0 : float(px) * 47.f / (gw - 1);
                    int y0 = int(fy), x0 = int(fx), y1 = std::min(y0 + 1, 47), x1 = std::min(x0 + 1, 47);
                    float wy = fy - y0, wx = fx - x0;
                    for (int c = 0; c < 1152; ++c)
                        positionsEmbedding[size_t(i) * 1152 + c] =
                            learned[(y0 * 48 + x0) * 1152 + c] * (1 - wy) * (1 - wx) +
                            learned[(y0 * 48 + x1) * 1152 + c] * (1 - wy) * wx +
                            learned[(y1 * 48 + x0) * 1152 + c] * wy * (1 - wx) +
                            learned[(y1 * 48 + x1) * 1152 + c] * wy * wx;
                }
    auto x = weights_.linear(rt_.upload(patches.data(), n, 1536), root + "patch_embed.proj");
    x = rt_.add(x, rt_.upload(positionsEmbedding.data(), n, 1152));
    auto merge = [&](Tensor input, const std::string &p, bool post) {
        if (post) {
            input.rows /= 4;
            input.cols *= 4;
        }
        input = rt_.norm(input, false, &weights_.get(p + ".norm.weight"), &weights_.get(p + ".norm.bias"));
        if (!post) {
            input.rows /= 4;
            input.cols *= 4;
        }
        return weights_.linear(rt_.geluExact(weights_.linear(input, p + ".linear_fc1")), p + ".linear_fc2");
    };
    std::vector<Tensor> result(4);
    for (int layer = 0; layer < 27; ++layer) {
        auto p = root + "blocks." + std::to_string(layer) + ".";
        auto nx = rt_.norm(x, false, &weights_.get(p + "norm1.weight"), &weights_.get(p + "norm1.bias"));
        auto qkv = weights_.linear(nx, p + "attn.qkv");
        auto q = rt_.visionRope(rt_.columns(qkv, 0, 1152), positions),
             k = rt_.visionRope(rt_.columns(qkv, 1152, 1152), positions), v = rt_.columns(qkv, 2304, 1152);
        x = rt_.add(x, weights_.linear(rt_.attention(q, k, v, 16), p + "attn.proj"));
        nx = rt_.norm(x, false, &weights_.get(p + "norm2.weight"), &weights_.get(p + "norm2.bias"));
        x = rt_.add(x, weights_.linear(rt_.activation(weights_.linear(nx, p + "mlp.linear_fc1"), true),
                                       p + "mlp.linear_fc2"));
        if (layer == 8 || layer == 16 || layer == 24) {
            int i = layer / 8 - 1;
            result[i + 1] = merge(x, root + "deepstack_merger_list." + std::to_string(i), true);
        }
    }
    result[0] = merge(x, root + "merger", false);
    return result;
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
    for (int layer = 0; layer < 36; ++layer) {
        auto p = root + "layers." + std::to_string(layer) + ".";
        auto nx = rt_.norm(x, true, &weights_.get(p + "input_layernorm.weight"));
        auto q = weights_.linear(nx, p + "self_attn.q_proj"), k = weights_.linear(nx, p + "self_attn.k_proj"),
             v = weights_.linear(nx, p + "self_attn.v_proj");
        q = rt_.norm(q, true, &weights_.get(p + "self_attn.q_norm.weight"), nullptr, 1e-6f, 128);
        k = rt_.norm(k, true, &weights_.get(p + "self_attn.k_norm.weight"), nullptr, 1e-6f, 128);
        q = rt_.textRope(q, positions, 5000000.f);
        k = rt_.textRope(k, positions, 5000000.f);
        x = rt_.add(
            x, weights_.linear(rt_.attention(q, k, v, 32, nullptr, 128, 8, true), p + "self_attn.o_proj"));
        nx = rt_.norm(x, true, &weights_.get(p + "post_attention_layernorm.weight"));
        auto gate = rt_.activation(weights_.linear(nx, p + "mlp.gate_proj"));
        x = rt_.add(x, weights_.linear(rt_.multiply(gate, weights_.linear(nx, p + "mlp.up_proj")),
                                       p + "mlp.down_proj"));
        if (layer < 3)
            x = applyVisual(x, layer);
    }
    // The diffusion checkpoint consumes the final decoder residual, before norm.
    return x;
}
DiT::DiT(Runtime &rt, const std::string &dir) : rt_(rt), weights_(rt, componentWeights(dir)) {}
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
    auto image = weights_.linear(!cached && cond.latents.rows ? rt_.concat({cond.latents, latents}) : latents,
                                 "img_in");
    Tensor text;
    if (!cached) {
        text = rt_.specialNorm(cond.features, weights_.get("txt_in.text_norm.weight"), 0);
        text = weights_.linear(rt_.activation(weights_.linear(text, "txt_in.in_layer"), true),
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
    auto temb = weights_.linear(rt_.activation(weights_.linear(rt_.upload(embedding.data(), 2, 256),
                                                               "time_text_embed.timestep_embedder.linear_1")),
                                "time_text_embed.timestep_embedder.linear_2");
    auto modulation = weights_.linear(rt_.activation(temb), "modulation.1");
    std::vector<float> cs(size_t(x.rows) * 128), sn(cs.size());
    int start = cached ? prefix : 0;
    for (int i = 0; i < x.rows; ++i) {
        auto pos = positions[i + start];
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
    for (int layer = 0; layer < 32; ++layer) {
        auto p = "transformer_blocks." + std::to_string(layer) + ".";
        auto nx = rt_.scale(rt_.norm(x, false), modulation, cached ? 0 : prefix, 0);
        auto q = weights_.linear(nx, p + "attn.to_q"), k = weights_.linear(nx, p + "attn.to_k"),
             v = weights_.linear(nx, p + "attn.to_v");
        q = rt_.norm(q, true, &weights_.get(p + "attn.norm_q.weight"), nullptr, 1e-6f, 128);
        k = rt_.norm(k, true, &weights_.get(p + "attn.norm_k.weight"), nullptr, 1e-6f, 128);
        q = rt_.rope(q, cs.data(), sn.data(), 128);
        k = rt_.rope(k, cs.data(), sn.data(), 128);
        Tensor a;
        if (cached) {
            k = rt_.concat({cache->keys[layer], k});
            v = rt_.concat({cache->values[layer], v});
            a = rt_.attention(q, k, v, 32);
        } else {
            if (cache) {
                cache->keys.push_back(rt_.concat({k.rowSlice(0, prefix)}));
                cache->values.push_back(rt_.concat({v.rowSlice(0, prefix)}));
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
                    32, mask.empty() ? nullptr : mask.data(), 128, 32, segment.text && segment.start == 0));
            }
            outputs.push_back(rt_.attention(q.rowSlice(prefix, target), k, v, 32));
            a = rt_.concat(outputs);
        }
        x = rt_.gatedAdd(x, weights_.linear(a, p + "attn.to_out.0"), modulation, cached ? 0 : prefix, 1);
        nx = rt_.scale(rt_.norm(x, false), modulation, cached ? 0 : prefix, 2);
        auto gate = rt_.activation(weights_.linear(nx, p + "img_mlp.gate_layer"));
        auto mlp =
            weights_.linear(rt_.multiply(gate, weights_.linear(nx, p + "img_mlp.proj")), p + "img_mlp.out");
        x = rt_.gatedAdd(x, mlp, modulation, cached ? 0 : prefix, 3);
    }
    auto scale = weights_.linear(rt_.activation(temb), "norm_out.linear");
    auto out = weights_.linear(rt_.scale(rt_.norm(x, false), scale, cached ? 0 : prefix, 0), "proj_out");
    return cached ? out : out.rowSlice(prefix, target);
}
VAE::VAE(Runtime &rt, const std::string &dir) : rt_(rt), weights_(rt, componentWeights(dir)) {}
Tensor VAE::conv(const Tensor &x, const std::string &p, int h, int w, int stride, int padEnd) {
    auto &weight = weights_.get(p + ".weight");
    int kernel = int(std::sqrt(weight.cols / x.cols));
    return rt_.conv2d(x, weight, weights_.optional(p + ".bias"), h, w, kernel, stride,
                      stride == 1 ? kernel / 2 : 0, padEnd);
}
Tensor VAE::residual(const Tensor &x, const std::string &p, int h, int w) {
    auto shortcut = weights_.optional(p + ".conv_shortcut.weight") ? conv(x, p + ".conv_shortcut", h, w) : x;
    auto y =
        conv(rt_.activation(rt_.specialNorm(x, weights_.get(p + ".norm1.gamma"), 1)), p + ".conv1", h, w);
    y = conv(rt_.activation(rt_.specialNorm(y, weights_.get(p + ".norm2.gamma"), 1)), p + ".conv2", h, w);
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
    return conv(rt_.activation(rt_.specialNorm(x, weights_.get("decoder.norm_out.gamma"), 1)),
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
    x = conv(rt_.activation(rt_.specialNorm(x, weights_.get("encoder.norm_out.gamma"), 1)),
             "encoder.conv_out", h, w);
    return rt_.columns(conv(x, "quant_conv", h, w), 0, 64);
}
} // namespace qwen_image
} // namespace fastllm
