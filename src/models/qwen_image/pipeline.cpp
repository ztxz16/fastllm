#include "models/qwen_image/runtime.h"
#include "json11.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <random>
#include <stdexcept>

namespace fastllm {
namespace qwen_image {
namespace {
// Host scheduler state mirrors BF16 activations; keep FP32 arithmetic between
// the same materialization boundaries as the reference pipeline.
float roundBF16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}
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
size_t checkpointBytes(const std::string &directory) {
    auto path = componentWeights(directory);
    if (std::filesystem::is_regular_file(path))
        return std::filesystem::file_size(path);
    size_t bytes = 0;
    for (const auto &entry : std::filesystem::directory_iterator(path))
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors")
            bytes += entry.file_size();
    return bytes;
}

} // namespace
Pipeline::Pipeline(const std::string &dir, int device, bool fp32)
    : rt_(device, fp32), pool_(device, size_t(1) << 30), tokenizer_(dir + "/processor"),
      text_(rt_, dir + "/text_encoder"),
      dit_(rt_, dir + "/transformer"), vae_(rt_, dir + "/vae") {
    // Checkpoint bytes conservatively include unused tensors and shard headers.
    weightBytes_ = (checkpointBytes(dir + "/text_encoder") + checkpointBytes(dir + "/transformer") +
                    checkpointBytes(dir + "/vae")) * (fp32 ? 2 : 1);
    auto d = config(dir + "/transformer/config.json");
    if (d["_class_name"].string_value() != "QwenImage21Transformer2DModel" ||
        d["num_layers"].int_value() != 32 || d["num_attention_heads"].int_value() != 32 ||
        !d["causal_condition"].bool_value())
        throw std::invalid_argument("this backend requires the released Qwen-Image-2.1 architecture");
    auto v = config(dir + "/vae/config.json");
    for (auto &x : v["latents_mean"].array_items())
        means_.push_back(fp32 ? float(x.number_value()) : roundBF16(x.number_value()));
    for (auto &x : v["latents_std"].array_items())
        stds_.push_back(fp32 ? float(x.number_value()) : roundBF16(x.number_value()));
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
    nomas::TensorPool::Scope scope(pool_);
    auto round = [&](float value) { return rt_.isFloat32() ? value : roundBF16(value); };
    struct UnloadWeights {
        TextEncoder &text;
        DiT &dit;
        VAE &vae;
        Runtime &rt;
        nomas::TensorPool &pool;
        bool keep = false;
        ~UnloadWeights() {
            if (!keep) {
                text.unload();
                dit.unload();
                vae.unload();
            }
            // All request-local tensors have now been released. Give unused
            // scratch pages back even after cancellation or an exception.
            try {
                rt.synchronize();
                pool.trim();
            } catch (...) {}
        }
    } cleanup{text_, dit_, vae_, rt_, pool_};
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
    size_t prefix = tokens.size() - crop, targetPixels = size_t(o.width) * o.height;
    size_t largestPixels = targetPixels;
    for (const auto &image : images) {
        size_t area = size_t(image.width) * image.height;
        prefix += area / 256 - area / 1024;
        largestPixels = std::max(largestPixels, area);
    }
    size_t scalar = rt_.isFloat32() ? 4 : 2, sequence = prefix + targetPixels / 256;
    // Full KV for 32 layers, DiT intermediates, VAE activations and 2 GiB for
    // allocator/cuBLAS scratch. Re-evaluate for every shape and current GPU use.
    size_t workspace = (size_t(2) << 30) + scalar * (sequence * 4096 * (64 + 16) + largestPixels * 1024);
    size_t loaded = text_.loadedBytes() + dit_.loadedBytes() + vae_.loadedBytes();
    bool resident = rt_.availableMemory() + loaded >= weightBytes_ + workspace;
    if (!resident) {
        text_.unload();
        dit_.unload();
        vae_.unload();
        rt_.synchronize();
        pool_.trim();
    }
    Condition cond;
    cond.features = text_.encode(tokens, images).rowSlice(crop, tokens.size() - crop);
    if (!images.empty())
        for (size_t i = crop; i < tokens.size(); ++i)
            cond.imageSlots.push_back(tokens[i] == 151655);
    if (trace)
        trace("prompt", rt_.download(cond.features));
    if (!resident)
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
            encoded[i] = round(round(encoded[i] - means_[i % 64]) / stds_[i % 64]);
        imageLatents.push_back(rt_.upload(encoded.data(), image.width * image.height / 256, 64));
        if (trace)
            trace("condition." + std::to_string(imageLatents.size() - 1), encoded);
        cond.shapes.emplace_back(image.height / 16, image.width / 16);
    }
    if (!imageLatents.empty())
        cond.latents = rt_.concat(imageLatents);
    imageLatents.clear();
    if (!resident)
        vae_.unload();
    images.clear();
    int h = o.height / 16, w = o.width / 16;
    std::vector<float> latent(size_t(h) * w * 64);
    std::mt19937_64 rng(o.seed);
    std::normal_distribution<float> normal;
    for (auto &x : latent)
        x = round(normal(rng));
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
    cache.full = resident;
    auto deviceLatent = rt_.upload(latent.data(), h * w, 64);
    for (int i = 0; i < o.steps; ++i) {
        report("denoising", i, o.steps);
        float t = round(round(sigmas[i] * 1000.f) / 1000.f);
        auto output = dit_.forward(deviceLatent, cond, t, h, w, &cache);
        float dt = sigmas[i + 1] - sigmas[i];
        rt_.eulerStep(deviceLatent, output, dt);
        if (trace)
            trace("latent.step." + std::to_string(i), rt_.download(deviceLatent));
    }
    cache = {};
    if (!resident)
        dit_.unload();
    cond = {};
    report("decoding", 0, 1);
    latent = rt_.download(deviceLatent);
    deviceLatent = {};
    for (size_t i = 0; i < latent.size(); ++i)
        latent[i] = round(round(latent[i] * stds_[i % 64]) + means_[i % 64]);
    auto decoded = vae_.decode(rt_.upload(latent.data(), h * w, 64), h, w);
    auto pixels = rt_.download(decoded);
    Image image{o.width, o.height, {}};
    image.rgba.resize(pixels.size());
    for (size_t i = 0; i < pixels.size(); ++i)
        image.rgba[i] = uint8_t(std::nearbyint(std::clamp((pixels[i] + 1.f) * .5f, 0.f, 1.f) * 255.f));
    if (!resident)
        vae_.unload();
    report("completed", o.steps, o.steps);
    cleanup.keep = resident;
    return image;
}

} // namespace qwen_image
} // namespace fastllm
