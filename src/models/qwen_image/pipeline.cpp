#include "models/qwen_image/runtime.h"
#include "json11.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>

namespace fastllm {
namespace qwen_image {
namespace {
json11::Json config(const std::string &path) {
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("cannot read " + path);
    std::string s((std::istreambuf_iterator<char>(f)), {}), err;
    auto j = json11::Json::parse(s, err);
    if (!err.empty())
        throw std::runtime_error("invalid config: " + path);
    return j;
}

} // namespace
Pipeline::Pipeline(const std::string &dir, int device)
    : rt_(device), tokenizer_(dir + "/processor"), text_(rt_, dir + "/text_encoder"),
      dit_(rt_, dir + "/transformer"), vae_(rt_, dir + "/vae") {
    auto d = config(dir + "/transformer/config.json");
    if (d["_class_name"].string_value() != "QwenImage21Transformer2DModel" ||
        d["num_layers"].int_value() != 32 || d["num_attention_heads"].int_value() != 32 ||
        !d["causal_condition"].bool_value())
        throw std::invalid_argument("this backend requires the released Qwen-Image-2.1 architecture");
    auto v = config(dir + "/vae/config.json");
    for (auto &x : v["latents_mean"].array_items())
        means_.push_back(x.number_value());
    for (auto &x : v["latents_std"].array_items())
        stds_.push_back(x.number_value());
    if (means_.size() != 64 || stds_.size() != 64)
        throw std::invalid_argument("invalid VAE normalization");
    auto s = config(dir + "/scheduler/scheduler_config.json");
    if (!s["use_dynamic_shifting"].bool_value() || s["time_shift_type"].string_value() != "exponential" ||
        s["stochastic_sampling"].bool_value())
        throw std::invalid_argument("unsupported Qwen image scheduler configuration");
    baseShift_ = s["base_shift"].number_value();
    maxShift_ = s["max_shift"].number_value();
    baseLength_ = s["base_image_seq_len"].number_value();
    maxLength_ = s["max_image_seq_len"].number_value();
    terminal_ = s["shift_terminal"].number_value();
}
void GenerateOptions::validate() const {
    if (width < 64 || height < 64 || width > 2048 || height > 2048 || width % 32 || height % 32 ||
        steps < 2 || steps > 100)
        throw std::invalid_argument("dimensions must be 64..2048 divisible by 32; use 2..100 steps");
    if (prompt.empty() || prompt.size() > 16384)
        throw std::invalid_argument("prompt must contain 1..16384 UTF-8 bytes");
    if (images.size() > 10 || referenceResolution < 256 || referenceResolution > 1024 ||
        referenceResolution % 32)
        throw std::invalid_argument(
            "use at most 10 reference images and reference_resolution 256..1024 divisible by 32");
}
Image Pipeline::generate(const GenerateOptions &o, Progress progress, Trace trace) {
    o.validate();
    rt_.activate();
    struct UnloadWeights {
        TextEncoder &text;
        DiT &dit;
        VAE &vae;
        ~UnloadWeights() {
            text.unload();
            dit.unload();
            vae.unload();
        }
    } cleanup{text_, dit_, vae_};
    std::vector<Image> images;
    std::string imageTemplate;
    for (size_t i = 0; i < o.images.size(); ++i) {
        auto &source = o.images[i];
        if (source.width < 1 || source.height < 1)
            throw std::invalid_argument("invalid reference image dimensions");
        double ratio = double(source.width) / source.height;
        int iw = int(std::nearbyint(o.referenceResolution * std::sqrt(ratio) / 32)) * 32;
        int ih = int(std::nearbyint(o.referenceResolution / std::sqrt(ratio) / 32)) * 32;
        if (int64_t(iw) * ih < 65536)
            throw std::invalid_argument(
                "increase reference_resolution so the resized image contains at least 65536 pixels");
        images.push_back(resizeImage(source, iw, ih));
        if (i)
            imageTemplate += " ";
        imageTemplate += "<image" + std::to_string(i + 1) + "><|vision_start|>";
        for (int j = 0; j < (iw / 32) * (ih / 32); ++j)
            imageTemplate += "<|image_pad|>";
        imageTemplate += "<|vision_end|>";
    }
    auto report = [&](const std::string &stage, int step, int total) {
        if (progress)
            progress(stage, step, total);
    };
    report("encoding", 0, 1);
    const std::string system = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
    auto tokens = tokenizer_.encode(system + "<|im_start|>user\n" + imageTemplate + o.prompt +
                                    "<|im_end|>\n<|im_start|>assistant\n");
    int crop = tokenizer_.encode(system).size();
    Condition cond;
    cond.features = text_.encode(tokens, images).rowSlice(crop, tokens.size() - crop);
    if (!images.empty())
        for (size_t i = crop; i < tokens.size(); ++i)
            cond.imageSlots.push_back(tokens[i] == 151655);
    if (trace)
        trace("prompt", rt_.download(cond.features));
    text_.unload();
    std::vector<Tensor> imageLatents;
    for (auto &image : images) {
        report("encoding_image", int(imageLatents.size()), int(images.size()));
        std::vector<float> pixels(image.rgba.size());
        for (size_t i = 0; i < pixels.size(); ++i)
            pixels[i] = image.rgba[i] * (1.f / 127.5f) - 1.f;
        auto encoded = rt_.download(
            vae_.encode(rt_.upload(pixels.data(), image.width * image.height, 4), image.height, image.width));
        for (size_t i = 0; i < encoded.size(); ++i)
            encoded[i] = (encoded[i] - means_[i % 64]) / stds_[i % 64];
        imageLatents.push_back(rt_.upload(encoded.data(), image.width * image.height / 256, 64));
        if (trace)
            trace("condition." + std::to_string(imageLatents.size() - 1), encoded);
        cond.shapes.emplace_back(image.height / 16, image.width / 16);
    }
    if (!imageLatents.empty())
        cond.latents = rt_.concat(imageLatents);
    imageLatents.clear();
    vae_.unload();
    images.clear();
    int h = o.height / 16, w = o.width / 16;
    std::vector<float> latent(size_t(h) * w * 64);
    std::mt19937_64 rng(o.seed);
    std::normal_distribution<float> normal;
    for (auto &x : latent)
        x = normal(rng);
    if (trace)
        trace("initial_latent", latent);
    double mu = baseShift_ + (h * w - baseLength_) * (maxShift_ - baseShift_) / (maxLength_ - baseLength_),
           em = std::exp(mu);
    std::vector<float> sigmas(o.steps + 1);
    for (int i = 0; i < o.steps; ++i) {
        double s = 1. - double(i) / o.steps;
        sigmas[i] = em / (em + (1. / s - 1.));
    }
    double factor = (1. - sigmas[o.steps - 1]) / (1. - terminal_);
    for (int i = 0; i < o.steps; ++i)
        sigmas[i] = float(1. - (1. - sigmas[i]) / factor);
    sigmas.back() = 0;
    KVCache cache;
    for (int i = 0; i < o.steps; ++i) {
        report("denoising", i, o.steps);
        float t = sigmas[i] * 1000.f / 1000.f;
        auto output = dit_.forward(rt_.upload(latent.data(), h * w, 64), cond, t, h, w, &cache);
        auto noise = rt_.download(output);
        float dt = sigmas[i + 1] - sigmas[i];
        for (size_t j = 0; j < latent.size(); ++j)
            latent[j] = latent[j] + dt * noise[j];
        if (trace)
            trace("latent.step." + std::to_string(i), latent);
    }
    cache = {};
    dit_.unload();
    cond = {};
    report("decoding", 0, 1);
    for (size_t i = 0; i < latent.size(); ++i)
        latent[i] = latent[i] * stds_[i % 64] + means_[i % 64];
    auto decoded = vae_.decode(rt_.upload(latent.data(), h * w, 64), h, w);
    auto pixels = rt_.download(decoded);
    Image image{o.width, o.height, {}};
    image.rgba.resize(pixels.size());
    for (size_t i = 0; i < pixels.size(); ++i)
        image.rgba[i] = uint8_t(std::nearbyint(std::clamp((pixels[i] + 1.f) * .5f, 0.f, 1.f) * 255.f));
    vae_.unload();
    report("completed", o.steps, o.steps);
    return image;
}

} // namespace qwen_image
} // namespace fastllm
