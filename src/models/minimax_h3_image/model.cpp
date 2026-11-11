#include "models/minimax_h3_image/model.h"
#include "json11.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace fastllm::minimax_h3_image {
namespace {
json11::Json config(const std::string &path) {
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("cannot open " + path);
    std::string error, data((std::istreambuf_iterator<char>(f)), {});
    auto j = json11::Json::parse(data, error);
    if (!error.empty() || !j.is_object())
        throw std::runtime_error("invalid config: " + path);
    return j;
}
void expect(const json11::Json &j, const std::string &key, int value) {
    if (!j[key].is_number() || j[key].int_value() != value)
        throw std::runtime_error("unsupported MiniMax-H3-Image config: " + key);
}
// Raw checkpoints interleave Q/K/V within each head, including the VAE.
std::array<Tensor, 3> splitQKV(Runtime &rt, Tensor x, int heads, int dim) {
    const int rows = x.rows;
    if (x.cols != heads * dim * 3)
        throw std::runtime_error("invalid interleaved QKV");
    x.rows *= heads;
    x.cols = dim * 3;
    std::array<Tensor, 3> out;
    for (int i = 0; i < 3; ++i) {
        out[i] = rt.columns(x, i * dim, dim);
        out[i].rows = rows;
        out[i].cols = heads * dim;
    }
    return out;
}
Tensor gated(Runtime &rt, const Tensor &x) {
    // The native layout is [gate; value], unlike Diffusers' [value; gate].
    return rt.siluMultiply(rt.columns(x, 0, x.cols / 2), rt.columns(x, x.cols / 2, x.cols / 2));
}
struct Tiles {
    std::vector<int> starts, lengths, overlaps;
};
Tiles splitTiles(int size) {
    if (size <= 256)
        return {{0}, {size}, {}};
    int n = (size + 255) / 256;
    while (256 * n - 64 * (n - 1) < size)
        ++n;
    Tiles t;
    t.lengths.assign(n, 256);
    t.overlaps.assign(n - 1, 64);
    const int extra = (256 * n - 64 * (n - 1) - size) / 16;
    for (int i = 0; i < extra; ++i)
        t.overlaps[i % (n - 1)] += 16;
    t.starts.push_back(0);
    for (int i = 1; i < n; ++i)
        t.starts.push_back(t.starts.back() + 256 - t.overlaps[i - 1]);
    return t;
}
float bf16(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, 4);
    bits += 0x7fff + ((bits >> 16) & 1);
    bits &= 0xffff0000;
    std::memcpy(&x, &bits, 4);
    return x;
}
} // namespace

Transformer::Transformer(Runtime &rt, Runtime &full, const std::string &dir)
    : rt_(rt), full_(full), weights_(rt, minimax_h3::componentWeights(dir + "/transformer")),
      fullWeights_(full, minimax_h3::componentWeights(dir + "/transformer")),
      heads_(full, dir + "/pdd_fused_heads.safetensors") {
    auto j = config(dir + "/transformer/config.json");
    for (auto item : std::vector<std::pair<std::string, int>>{{"hidden_size", 5376},
                                                              {"num_layers", 50},
                                                              {"token_refiner_num_layers", 2},
                                                              {"num_attention_heads", 56},
                                                              {"attention_head_dim", 128},
                                                              {"ffn_hidden_size", 14336},
                                                              {"latents_dim", 24},
                                                              {"text_dim", 5120},
                                                              {"timestep_input_dim", 256},
                                                              {"time_embed_hidden_size", 5376},
                                                              {"time_embed_dim", 2688},
                                                              {"rope_inv_freq_len", 16},
                                                              {"adaln_out_features", 96768},
                                                              {"final_adaln_out_features", 10752}})
        expect(j, item.first, item.second);
    for (const char *key : {"norm_eps", "qk_norm_eps", "final_norm_eps"})
        if (j[key].number_value() != 1e-5)
            throw std::runtime_error(std::string("unsupported image norm: ") + key);
    const auto &patch = j["patch_size"].array_items();
    if (patch.size() != 3 || patch[0].int_value() != 1 || patch[1].int_value() != 2 ||
        patch[2].int_value() != 2 || j["audio_enabled"].bool_value())
        throw std::runtime_error("unsupported image patch/audio config");
    auto pdd = config(dir + "/pdd_config.json");
    expect(pdd, "num_inference_steps", 6);
    const auto &s = pdd["sigmas"].array_items();
    if (s.size() != 7 || s.front().number_value() != 1 || s.back().number_value() != 0)
        throw std::runtime_error("invalid PDD sigma grid");
    for (int i = 0; i < 6; ++i) {
        if (!s[i].is_number() || s[i].number_value() <= s[i + 1].number_value())
            throw std::runtime_error("PDD sigmas must strictly decrease");
        sigmas_.push_back(float(s[i].number_value()));
        times_.push_back(1.f - sigmas_.back());
    }
    sigmas_.push_back(0.f);
}
void Transformer::prepare(Progress progress) {
    if (modulation_.size() == 50)
        return;
    std::vector<Tensor> inputs;
    auto timesteps = times_;
    timesteps.push_back(.999f); // Reference image rows remain at this conditioning level.
    for (float t : timesteps) {
        auto x = full_.timestepEmbedding({t}, 256);
        x = full_.activation(fullWeights_.linear(x, "time_embedder.proj_in"));
        x = fullWeights_.linear(x, "time_embedder.proj_out");
        inputs.push_back(rt_.cast(full_.activation(x), rt_.dtype()));
    }
    // Precomputing one-row GEMMs preserves BF16 rounding and releases ~24 GiB
    // of single-use AdaLN weights before the main transformer becomes resident.
    for (int i = 0; i < 50; ++i) {
        const auto p = "blocks." + std::to_string(i) + ".adaln_proj.linear";
        std::vector<Tensor> rows;
        for (const auto &input : inputs) {
            auto x = weights_.linear(input, p, true);
            x.rows *= 3;
            x.cols /= 3;
            rows.push_back(x);
        }
        modulation_.push_back(rt_.concat(rows));
        weights_.erasePrefix(p + ".");
        if (progress)
            progress("prepare", i + 1, 50);
    }
    std::vector<Tensor> rows;
    for (const auto &input : inputs)
        rows.push_back(weights_.linear(input, "final_layer.adaln_proj.linear", true));
    finalModulation_ = rt_.concat(rows);
    weights_.erasePrefix("final_layer.adaln_proj.");
}
Tensor Transformer::attention(const Tensor &x, const std::string &p, const Tensor *cs, const Tensor *sn) {
    auto qkv = splitQKV(rt_, weights_.linear(x, p + ".qkv_proj"), 56, 128);
    if (cs) {
        auto qk = rt_.qkNormRotary(qkv[0], qkv[1], weights_.get(p + ".q_norm.weight"),
                                   weights_.get(p + ".k_norm.weight"), 1e-5f, *cs, *sn, 128);
        qkv[0] = qk.first;
        qkv[1] = qk.second;
    } else {
        qkv[0] = rt_.rms(qkv[0], weights_.get(p + ".q_norm.weight"), 1e-5f, 128);
        qkv[1] = rt_.rms(qkv[1], weights_.get(p + ".k_norm.weight"), 1e-5f, 128);
    }
    return weights_.linear(rt_.selfAttention(qkv[0], qkv[1], qkv[2], 56), p + ".out_proj");
}
Tensor Transformer::feedForward(const Tensor &x, const std::string &p) {
    Tensor out(x.rows, 5376, x.dtype);
    for (int first = 0; first < x.rows;) {
        int n = std::min(4096, x.rows - first);
        if (n > 512)
            n -= n % 512;
        auto y = gated(rt_, weights_.linear(x.rowSlice(first, n), p + ".fc1"));
        auto target = out.rowSlice(first, n);
        weights_.linear(y, p + ".fc2", false, &target);
        first += n;
    }
    return out;
}
Tensor Transformer::refine(const Tensor &text) {
    auto x = weights_.linear(rt_.cast(text, rt_.dtype()), "condition_proj");
    for (int i = 0; i < 2; ++i) {
        auto p = "token_refiner.blocks." + std::to_string(i);
        x = rt_.add(x, attention(rt_.rms(x, weights_.get(p + ".norm1.weight"), 1e-5f), p + ".attn"));
        x = rt_.add(x, feedForward(rt_.rms(x, weights_.get(p + ".norm2.weight"), 1e-5f), p + ".mlp"));
    }
    return rt_.rms(x, weights_.get("token_refiner.final_norm.weight"), 1e-5f);
}
Tensor Transformer::forward(const Tensor &latent, const Tensor &text, int h, int w, int step, Trace trace,
                            const Conditioning *condition) {
    int referenceRows = 0;
    if (condition)
        for (auto grid : condition->grids)
            referenceRows += grid.first / 2 * (grid.second / 2);
    if (modulation_.size() != 50 || step < 0 || step >= 6 || h % 32 || w % 32 ||
        latent.rows != referenceRows + (h / 32) * (w / 32) || latent.cols != 96 || text.rows < 1 ||
        text.cols != 5120 || (condition && condition->textTags.size() != size_t(text.rows)))
        throw std::invalid_argument("invalid image transformer input");
    const int nt = text.rows, rows = nt + latent.rows, targetRows = h / 32 * (w / 32);
    auto x =
        rt_.concat({refine(text), rt_.cast(fullWeights_.linear(latent, "video_patch_proj"), rt_.dtype())});
    std::vector<float> positions(size_t(rows) * 3);
    std::vector<int> adaln(rows, step * 3);
    for (int i = 0; i < nt; ++i) {
        positions[i * 3] = float(i);
        int tag = condition ? condition->textTags[i] : 1;
        if (tag != 0 && tag != 1)
            throw std::invalid_argument("invalid image/text modality tag");
        adaln[i] += tag;
    }
    int cursor = nt, rotaryTime = nt;
    auto frame = [&](int lh, int lw, bool reference) {
        if (lh < 2 || lw < 2 || lh % 2 || lw % 2)
            throw std::invalid_argument("invalid reference grid");
        const int ph = lh / 2, pw = lw / 2;
        const double area = std::sqrt(double(lh) * lw), rh = lh / area, rw = lw / area;
        for (int y = 0; y < ph; ++y)
            for (int z = 0; z < pw; ++z) {
                const int i = cursor++;
                positions[i * 3] = float(rotaryTime);
                positions[i * 3 + 1] = float(((1 - rh) / 2 + y * (rh / ph)) * 32);
                positions[i * 3 + 2] = float(((1 - rw) / 2 + z * (rw / pw)) * 32);
                if (reference)
                    adaln[i] = 6 * 3;
            }
        ++rotaryTime;
    };
    if (condition)
        for (auto grid : condition->grids)
            frame(grid.first, grid.second, true);
    frame(h / 16, w / 16, false);
    auto rope = rt_.ropeTable(positions, 16, 10000.f);
    auto idx = rt_.indices(adaln);
    if (trace)
        trace("input", x);
    for (int i = 0; i < 50; ++i) {
        const auto p = "blocks." + std::to_string(i);
        auto y = rt_.rmsModulateRows(x, weights_.get(p + ".norm1.weight"), 1e-5f, modulation_[i], idx, 0, 1);
        y = attention(y, p + ".attn", &rope.first, &rope.second);
        x = rt_.residualRows(x, y, modulation_[i], idx, 2);
        y = rt_.rmsModulateRows(x, weights_.get(p + ".norm2.weight"), 1e-5f, modulation_[i], idx, 3, 4);
        x = rt_.residualRows(x, feedForward(y, p + ".mlp"), modulation_[i], idx, 5);
        if (trace)
            trace("block." + std::to_string(i), x);
    }
    x = rt_.rmsModulateRows(x.rowSlice(nt + referenceRows, targetRows),
                            weights_.get("final_layer.norm.weight"), 1e-5f, finalModulation_,
                            rt_.indices(std::vector<int>(targetRows, step)), 0, 1);
    auto weight = heads_.get("video_out.weight");
    weight.rows = 6 * 96;
    weight.cols = 5376;
    auto bias = heads_.get("video_out.bias");
    bias.rows = 6;
    bias.cols = 96;
    auto b = bias.rowSlice(step, 1);
    return full_.linear(rt_.cast(x, DType::FP32), weight.rowSlice(step * 96, 96), &b);
}
void Transformer::unload() {
    weights_.clear();
    fullWeights_.clear();
    heads_.clear();
    modulation_.clear();
    finalModulation_ = {};
}

VAE::VAE(Runtime &rt, Runtime &full, const std::string &dir)
    : model_(dir), rt_(rt), full_(full), weights_(rt, minimax_h3::componentWeights(dir + "/vae")),
      fullWeights_(full, minimax_h3::componentWeights(dir + "/vae")) {
    auto j = config(dir + "/vae/config.json");
    expect(j, "latent_channels", 24);
    auto v = j["vit_decoder_kwargs"];
    expect(v, "num_layers", 24);
    expect(v, "heads", 16);
    expect(v, "dim_head", 64);
    expect(j, "vae_tile_size", 256);
    expect(j, "vae_tile_overlap_min", 64);
    expect(j, "vae_decoder_tiling", 1);
    if (v["rope_theta"].number_value() != 100 || v["rope_dim_ratio"].number_value() != .5 ||
        v["ffn_activation_fn"].string_value() != "silu" || !v["ffn_use_gated"].bool_value())
        throw std::runtime_error("unsupported image VAE rotary/FFN config");
    if (!j["vit_decoder_2d"].bool_value())
        throw std::runtime_error("expected image 2D VAE");
    for (auto &x : j["latents_mean"].array_items())
        mean_.push_back(x.number_value());
    for (auto &x : j["latents_std"].array_items())
        std_.push_back(x.number_value());
    if (mean_.size() != 24 || std_.size() != 24)
        throw std::runtime_error("invalid VAE statistics");
    for (float x : mean_)
        if (!std::isfinite(x))
            throw std::runtime_error("invalid VAE mean");
    for (float x : std_)
        if (!std::isfinite(x) || x <= 0)
            throw std::runtime_error("invalid VAE scale");
    if (rt.dtype() != DType::BF16 && rt.dtype() != DType::FP32)
        throw std::invalid_argument("image VAE supports bf16 or fp32");
}
Tensor VAE::decodeTile(const Tensor &latent, int h, int w, Trace trace) {
    const int n = h * w;
    if (h < 1 || w < 1 || latent.rows != n || latent.cols != 24)
        throw std::invalid_argument("invalid VAE tile");
    auto x = weights_.linear(weights_.linear(rt_.cast(latent, rt_.dtype()), "post_quant_conv"),
                             "decoder.x_embedder");
    auto registers = weights_.get("decoder.register_tokens");
    registers.rows = 4;
    registers.cols = 1024;
    std::vector<float> zero(1024);
    x = rt_.cast(rt_.concat({x, registers, rt_.upload(zero.data(), 1, 1024)}), DType::FP32);
    std::vector<float> cs(size_t(n + 5) * 32, 1.f), sn(cs.size());
    for (int y = 0; y < h; ++y)
        for (int z = 0; z < w; ++z)
            for (int a = 0; a < 2; ++a) {
                float position = 2.f * (((a ? z : y) + .5f) / (a ? w : h)) - 1.f;
                for (int f = 0; f < 8; ++f) {
                    float angle = (2.f * float(M_PI) * position) * (1.f / std::pow(100.f, f / 8.f));
                    for (int half = 0; half < 2; ++half) {
                        auto i = size_t(y * w + z) * 32 + half * 16 + a * 8 + f;
                        cs[i] = std::cos(angle);
                        sn[i] = std::sin(angle);
                    }
                }
            }
    auto cos = full_.upload(cs.data(), n + 5, 32), sin = full_.upload(sn.data(), n + 5, 32);
    if (trace)
        trace("input", x);
    for (int i = 0; i < 24; ++i) {
        const auto p = "decoder.transformer_blocks." + std::to_string(i);
        auto y = rt_.videoRms(x, fullWeights_.get(p + ".norm1.weight"));
        auto qkv = splitQKV(rt_, weights_.linear(y, p + ".attn.to_qkv"), 16, 64);
        auto qk = rt_.videoQkNormRotary(qkv[0], qkv[1], cos, sin);
        y = weights_.linear(rt_.selfAttention(qk.first, qk.second, qkv[2], 16), p + ".attn.to_out");
        x = rt_.scaledResidual(x, y, fullWeights_.get(p + ".scale1"));
        y = rt_.videoRms(x, fullWeights_.get(p + ".norm2.weight"));
        y = weights_.linear(gated(rt_, weights_.linear(y, p + ".ff.w1")), p + ".ff.w2");
        x = rt_.scaledResidual(x, y, fullWeights_.get(p + ".scale2"));
        if (trace)
            trace("block." + std::to_string(i), x);
    }
    x = full_.norm(x.rowSlice(0, n), false, &fullWeights_.get("decoder.norm_out.weight"),
                   &fullWeights_.get("decoder.norm_out.bias"), 1e-5f);
    return weights_.linear(rt_.cast(x, rt_.dtype()), "decoder.proj_out");
}
Image VAE::decode(const std::vector<float> &packed, int height, int width, Progress progress) {
    if (height < 32 || width < 32 || height % 32 || width % 32 ||
        packed.size() != size_t(height / 16) * (width / 16) * 24)
        throw std::invalid_argument("invalid packed image latents");
    const int lh = height / 16, lw = width / 16;
    std::vector<float> latent(packed.size());
    for (int y = 0; y < lh; ++y)
        for (int z = 0; z < lw; ++z)
            for (int c = 0; c < 24; ++c) {
                auto src = (size_t(y / 2) * (lw / 2) + z / 2) * 96 + c * 4 + (y % 2) * 2 + z % 2;
                latent[(size_t(y) * lw + z) * 24 + c] = packed[src] * std_[c] + mean_[c];
            }
    auto ys = splitTiles(height), xs = splitTiles(width);
    std::vector<std::vector<float>> tiles;
    for (size_t iy = 0; iy < ys.starts.size(); ++iy)
        for (size_t ix = 0; ix < xs.starts.size(); ++ix) {
            int th = ys.lengths[iy] / 16, tw = xs.lengths[ix] / 16;
            std::vector<float> tile(size_t(th) * tw * 24);
            for (int y = 0; y < th; ++y) {
                auto src = latent.data() + (size_t(ys.starts[iy] / 16 + y) * lw + xs.starts[ix] / 16) * 24;
                std::copy_n(src, tw * 24, tile.data() + size_t(y) * tw * 24);
            }
            auto pixels = rt_.download(decodeTile(full_.upload(tile.data(), th * tw, 24), th, tw));
            tile.resize(size_t(th * 16) * (tw * 16) * 3);
            for (int y = 0; y < th * 16; ++y)
                for (int z = 0; z < tw * 16; ++z)
                    for (int c = 0; c < 3; ++c)
                        tile[(size_t(y) * tw * 16 + z) * 3 + c] =
                            pixels[(size_t(y / 16) * tw + z / 16) * 768 + c * 256 + y % 16 * 16 + z % 16];
            tiles.push_back(std::move(tile));
            if (progress)
                progress("decode", tiles.size(), ys.starts.size() * xs.starts.size());
        }
    Image out;
    out.width = width;
    out.height = height;
    out.rgba.resize(size_t(width) * height * 4);
    // Match released tiling: blend with original neighbours, vertical then horizontal.
    auto round = [&](float x) { return rt_.dtype() == DType::BF16 ? bf16(x) : x; };
    auto blend = [&](float a, float b, int i, int extent) {
        float alpha = round(round(float(i)) / extent);
        return round(round(a * round(1.f - alpha)) + round(b * alpha));
    };
    const size_t nx = xs.starts.size();
    for (size_t iy = 0; iy < ys.starts.size(); ++iy)
        for (size_t ix = 0; ix < nx; ++ix) {
            const int th = ys.lengths[iy], tw = xs.lengths[ix];
            auto tile = tiles[iy * nx + ix];
            if (iy)
                for (int y = 0; y < ys.overlaps[iy - 1]; ++y)
                    for (int z = 0; z < tw; ++z)
                        for (int c = 0; c < 3; ++c) {
                            auto dst = (size_t(y) * tw + z) * 3 + c;
                            auto src =
                                (size_t(ys.lengths[iy - 1] - ys.overlaps[iy - 1] + y) * tw + z) * 3 + c;
                            tile[dst] =
                                blend(tiles[(iy - 1) * nx + ix][src], tile[dst], y, ys.overlaps[iy - 1]);
                        }
            if (ix)
                for (int y = 0; y < th; ++y)
                    for (int z = 0; z < xs.overlaps[ix - 1]; ++z)
                        for (int c = 0; c < 3; ++c) {
                            auto dst = (size_t(y) * tw + z) * 3 + c;
                            auto src = (size_t(y) * xs.lengths[ix - 1] + xs.lengths[ix - 1] -
                                        xs.overlaps[ix - 1] + z) *
                                           3 +
                                       c;
                            tile[dst] =
                                blend(tiles[iy * nx + ix - 1][src], tile[dst], z, xs.overlaps[ix - 1]);
                        }
            const int ch = th - (iy + 1 < ys.starts.size() ? ys.overlaps[iy] : 0);
            const int cw = tw - (ix + 1 < nx ? xs.overlaps[ix] : 0);
            for (int y = 0; y < ch; ++y)
                for (int z = 0; z < cw; ++z) {
                    auto dst = (size_t(ys.starts[iy] + y) * width + xs.starts[ix] + z) * 4;
                    for (int c = 0; c < 3; ++c) {
                        float v = tile[(size_t(y) * tw + z) * 3 + c];
                        if (!std::isfinite(v))
                            throw std::runtime_error("nonfinite decoded image");
                        const float rgbMean[] = {.485f, .456f, .406f}, rgbStd[] = {.229f, .224f, .225f};
                        out.rgba[dst + c] = uint8_t(std::clamp(v * rgbStd[c] + rgbMean[c], 0.f, 1.f) * 255.f);
                    }
                    out.rgba[dst + 3] = 255;
                }
        }
    return out;
}
minimax_h3::VideoTensor VAE::encode(const Image &image, std::mt19937 &generator, Progress progress,
                                    Trace trace) {
    const int h = image.height / 16 * 16, w = image.width / 16 * 16;
    minimax_h3::VideoTensor pixels(1, h, w, 3);
    const float pixelMean[] = {.485f, .456f, .406f}, pixelStd[] = {.229f, .224f, .225f};
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c)
                pixels.values[(size_t(y) * w + x) * 3 + c] =
                    (image.rgba[(size_t(y) * image.width + x) * 4 + c] / 255.f - pixelMean[c]) / pixelStd[c];
    // The image checkpoint uses the same causal CNN encoder, run on one frame.
    minimax_h3::VideoVAE encoder(full_, full_, model_ + "/vae");
    auto moments = encoder.encodeMoments(pixels, [&](int i, int n) {
        if (progress)
            progress("reference", i, n);
    });
    encoder.unload();
    const size_t cells = size_t(moments.height) * moments.width;
    auto noise = minimax_h3::normalNoise(generator, cells * 24);
    std::vector<float> sampled(cells * 24);
    for (size_t i = 0; i < cells; ++i)
        for (int c = 0; c < 24; ++c)
            sampled[i * 24 + c] = moments.values[i * 48 + c] +
                                  std::exp(.5f * std::clamp(moments.values[i * 48 + c + 24], -30.f, 20.f)) *
                                      noise[size_t(c) * cells + i];
    sampled = full_.download(full_.cast(full_.upload(sampled.data(), cells, 24), DType::FP16));
    minimax_h3::VideoTensor latent(1, moments.height / 2 * 2, moments.width / 2 * 2, 24);
    for (int c = 0; c < 24; ++c)
        for (int y = 0; y < latent.height; ++y)
            for (int x = 0; x < latent.width; ++x)
                latent.values[(size_t(c) * latent.height + y) * latent.width + x] =
                    (sampled[(size_t(y) * moments.width + x) * 24 + c] - mean_[c]) / std_[c];
    if (trace)
        trace("reference_latent", full_.upload(latent.values.data(), 24, latent.height * latent.width));
    return latent; // CHW, unlike the encoder's THWC moments.
}

std::pair<int, int> Options::canvas() const {
    double scale = std::max(1., std::sqrt(double(minGenerationArea) / (double(width) * height)));
    return {std::max(32, int(std::nearbyint(width * scale / 32)) * 32),
            std::max(32, int(std::nearbyint(height * scale / 32)) * 32)};
}
void Options::validate() const {
    if (prompt.empty() || prompt.size() > 32768)
        throw std::invalid_argument("prompt must contain 1..32768 UTF-8 bytes");
    if (width < 32 || height < 32 || width > 8192 || height > 8192 ||
        int64_t(width) * height > 20 * 1024 * 1024 || minGenerationArea < 1024 ||
        minGenerationArea > 4096 * 4096)
        throw std::invalid_argument(
            "size must be 32..8192, at most 20 megapixels; min_generation_area must be 1024..16777216");
    auto size = canvas();
    if (size.first > 8192 || size.second > 8192 || int64_t(size.first) * size.second > 20 * 1024 * 1024)
        throw std::invalid_argument("generation canvas exceeds 8192 pixels per side or 20 megapixels");
    if (images.size() > 16)
        throw std::invalid_argument("at most 16 reference images are supported");
    size_t pixels = 0;
    for (const auto &image : images) {
        if (image.width < 32 || image.height < 32 || image.width > 8192 || image.height > 8192 ||
            int64_t(image.width) * image.height > 20 * 1024 * 1024 ||
            image.rgba.size() != size_t(image.width) * image.height * 4 || image.width > image.height * 4 ||
            image.height > image.width * 4)
            throw std::invalid_argument(
                "reference images require 32..8192 pixels per side, up to 20 MP, aspect ratio 1:4..4:1");
        pixels += size_t(image.width) * image.height;
    }
    if (pixels > 64 * 1024 * 1024)
        throw std::invalid_argument("combined references exceed 64 megapixels");
}
Pipeline::Pipeline(const std::string &model, int device, DType vaeType)
    : model_(model), rt_(device), full_(device, true), vaeRt_(device, vaeType) {}
Image Pipeline::generate(const Options &o, Progress progress, Trace trace,
                         const std::vector<float> *textInput, const std::vector<float> *noise, bool decode) {
    o.validate();
    const auto canvas = o.canvas();
    const int height = canvas.second, width = canvas.first;
    Transformer transformer(rt_, full_, model_);
    Conditioning condition;
    std::vector<Image> references;
    std::vector<minimax_h3::VisionInput> visual;
    std::vector<int> ids;
    nomas::QwenTokenizer tokenizer(model_ + "/tokenizer");
    auto label = [&](const std::string &value) {
        auto tokens = tokenizer.encode(value);
        if (tokens.size() > 4096)
            tokens.resize(4096);
        ids.insert(ids.end(), tokens.begin(), tokens.end());
        condition.textTags.insert(condition.textTags.end(), tokens.size(), 1);
    };
    auto bounded = [](Image image, int minimum, int maximum) {
        for (size_t i = 3; i < image.rgba.size(); i += 4)
            image.rgba[i] = 255;
        int edge = std::min(image.width, image.height);
        if (edge >= minimum && edge <= maximum)
            return image;
        double scale = double(std::clamp(edge, minimum, maximum)) / edge;
        return qwen_image::resizeImage(image,
                                       std::max(32, int(std::nearbyint(image.width * scale / 32)) * 32),
                                       std::max(32, int(std::nearbyint(image.height * scale / 32)) * 32),
                                       qwen_image::ImageResizeFilter::Lanczos, 8192);
    };
    for (size_t i = 0; i < o.images.size(); ++i) {
        references.push_back(bounded(o.images[i], 1024, 2048));
        visual.push_back({minimax_h3::prepareVisionImage(bounded(o.images[i], 1024, 1024)), {}, 151655});
        label("Picture " + std::to_string(i + 1) + ": ");
        int slots = visual.back().first.height / 32 * (visual.back().first.width / 32);
        ids.push_back(151652);
        ids.insert(ids.end(), slots, 151655);
        ids.push_back(151653);
        condition.textTags.insert(condition.textTags.end(), slots + 2, 0);
    }
    label(o.prompt);
    if (ids.empty() || ids.size() > 65536)
        throw std::invalid_argument("reference presentation exceeds 65536 tokens");
    Tensor text;
    if (textInput) {
        if (textInput->empty() || textInput->size() % 5120 || textInput->size() / 5120 > 65536 ||
            (!o.images.empty() && textInput->size() / 5120 != ids.size()))
            throw std::invalid_argument("invalid text fixture");
        text = rt_.upload(textInput->data(), textInput->size() / 5120, 5120);
        if (o.images.empty())
            condition.textTags.assign(text.rows, 1);
    } else {
        if (progress)
            progress("text", 0, 50);
        minimax_h3::TextEncoder encoder(rt_, model_ + "/text_encoder");
        encoder.setLowMemory(true);
        text = encoder.encodeVisual(ids, visual, [&](const std::string &name, const Tensor &) {
            if (progress && name.compare(0, 6, "layer.") == 0)
                progress("text", std::stoi(name.substr(6)) + 1, 50);
        });
        encoder.unload();
        if (progress)
            progress("text", 50, 50);
    }
    visual.clear();
    if (trace)
        trace("text", text);
    std::mt19937 generator(uint32_t(o.seed & 0xffffffff));
    std::vector<minimax_h3::VideoTensor> encoded;
    if (!references.empty()) {
        VAE vae(vaeRt_, full_, model_);
        for (size_t i = 0; i < references.size(); ++i) {
            encoded.push_back(vae.encode(references[i], generator, progress,
                                         trace ? Trace([&, i](const std::string &name, const Tensor &value) {
                                             trace("reference." + std::to_string(i) + "." + name, value);
                                         })
                                               : Trace{}));
            condition.grids.push_back({encoded.back().height, encoded.back().width});
        }
    }
    references.clear();
    const int lh = height / 16, lw = width / 16;
    std::vector<float> initial(size_t(lh) * lw * 24);
    if (noise) {
        if (noise->size() != initial.size())
            throw std::invalid_argument("noise fixture size mismatch");
        initial = *noise;
    } else {
        auto chw = minimax_h3::normalNoise(generator, initial.size());
        for (int y = 0; y < lh; ++y)
            for (int z = 0; z < lw; ++z)
                for (int c = 0; c < 24; ++c)
                    initial[(size_t(y / 2) * (lw / 2) + z / 2) * 96 + c * 4 + y % 2 * 2 + z % 2] =
                        chw[(size_t(c) * lh + y) * lw + z];
    }
    auto latent = full_.upload(initial.data(), (lh / 2) * (lw / 2), 96);
    if (trace)
        trace("noise", latent);
    std::vector<Tensor> referenceTensors;
    for (const auto &ref : encoded) {
        auto random = minimax_h3::normalNoise(generator, ref.values.size());
        std::vector<float> packed(ref.values.size());
        for (int y = 0; y < ref.height; ++y)
            for (int z = 0; z < ref.width; ++z)
                for (int c = 0; c < 24; ++c) {
                    size_t src = (size_t(c) * ref.height + y) * ref.width + z;
                    size_t dst = (size_t(y / 2) * (ref.width / 2) + z / 2) * 96 + c * 4 + y % 2 * 2 + z % 2;
                    packed[dst] = .999f * ref.values[src] + (1.f - .999f) * random[src];
                }
        referenceTensors.push_back(full_.upload(packed.data(), ref.height / 2 * (ref.width / 2), 96));
    }
    Tensor referenceRows;
    if (!referenceTensors.empty())
        referenceRows = full_.concat(referenceTensors);
    referenceTensors.clear();
    encoded.clear();
    if (trace && referenceRows.rows)
        trace("reference_rows", referenceRows);
    transformer.prepare(progress);
    for (int step = 0; step < 6; ++step) {
        Trace nested;
        if (trace)
            nested = [&](const std::string &name, const Tensor &x) {
                trace("step." + std::to_string(step) + "." + name, x);
            };
        auto input = referenceRows.rows ? full_.concat({referenceRows, latent}) : latent;
        auto velocity = transformer.forward(input, text, height, width, step, nested, &condition);
        // Image PDD uses velocity Euler, not H3's denoised interpolation form.
        latent = full_.add(
            latent, full_.affine(velocity, transformer.sigmas()[step] - transformer.sigmas()[step + 1]));
        if (trace)
            trace("latent." + std::to_string(step), latent);
        if (progress)
            progress("denoise", step + 1, 6);
    }
    transformer.unload();
    text = {};
    referenceRows = {};
    if (trace)
        trace("latent", latent);
    if (!decode)
        return {};
    auto host = full_.download(latent);
    latent = {};
    VAE vae(vaeRt_, full_, model_);
    auto image = vae.decode(host, height, width, progress);
    return qwen_image::resizeImage(image, o.width, o.height, qwen_image::ImageResizeFilter::Lanczos, 8192);
}
} // namespace fastllm::minimax_h3_image
