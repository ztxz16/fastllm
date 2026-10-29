// Test-only ctypes adapter; not part of the inference library.
#include "models/qwen_image/runtime.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace {
using namespace fastllm::qwen_image;
thread_local std::string lastError;
struct Handle {
    Runtime rt;
    QwenTokenizer tokenizer;
    TextEncoder text;
    DiT dit;
    VAE vae;
    Handle(const char *dir, int dev)
        : rt(dev), tokenizer(std::string(dir) + "/processor"), text(rt, std::string(dir) + "/text_encoder"),
          dit(rt, std::string(dir) + "/transformer"), vae(rt, std::string(dir) + "/vae") {}
};
template <class F> int guarded(F f) {
    try {
        lastError.clear();
        f();
        return 0;
    } catch (const std::exception &e) {
        lastError = e.what();
        return -1;
    }
}
void copy(Handle &h, const Tensor &x, float *out) {
    if (!out)
        throw std::invalid_argument("null output");
    auto v = h.rt.download(x);
    std::memcpy(out, v.data(), v.size() * 4);
}
} // namespace
extern "C" {
void *fastllm_qwen_image_create(const char *dir, int device) {
    try {
        lastError.clear();
        if (!dir)
            throw std::invalid_argument("null model path");
        return new Handle(dir, device);
    } catch (const std::exception &e) {
        lastError = e.what();
        return nullptr;
    }
}
void fastllm_qwen_image_destroy(void *p) { delete static_cast<Handle *>(p); }
const char *fastllm_qwen_image_error() { return lastError.c_str(); }
int fastllm_qwen_image_tokenize(void *p, const char *text, int *tokens, int capacity) {
    int count = 0;
    int status = guarded([&] {
        if (!p || !text)
            throw std::invalid_argument("null tokenizer argument");
        auto v = static_cast<Handle *>(p)->tokenizer.encode(text);
        count = v.size();
        if (capacity < count || !tokens)
            throw std::invalid_argument("token buffer too small");
        std::copy(v.begin(), v.end(), tokens);
    });
    return status < 0 ? status : count;
}
int fastllm_qwen_image_text(void *p, const int *tokens, int count, float *out) {
    return guarded([&] {
        if (!p || !tokens || count < 1 || count > 4096)
            throw std::invalid_argument("invalid text arguments");
        auto &h = *static_cast<Handle *>(p);
        h.rt.activate();
        copy(h, h.text.encode(std::vector<int>(tokens, tokens + count)), out);
        h.text.unload();
    });
}
int fastllm_qwen_image_dit(void *p, const float *latent, const float *features, int tokens, int height,
                           int width, float t, float *out) {
    return guarded([&] {
        if (!p || !latent || !features || tokens < 1 || tokens > 4096 || height < 1 || width < 1 ||
            height > 128 || width > 128)
            throw std::invalid_argument("invalid DiT arguments");
        auto &h = *static_cast<Handle *>(p);
        h.rt.activate();
        Condition c;
        c.features = h.rt.upload(features, tokens, 4096);
        copy(h, h.dit.forward(h.rt.upload(latent, height * width, 64), c, t, height, width), out);
        h.dit.unload();
    });
}
int fastllm_qwen_image_vae(void *p, const float *latent, int height, int width, float *out) {
    return guarded([&] {
        if (!p || !latent || height < 1 || width < 1 || height > 128 || width > 128)
            throw std::invalid_argument("invalid VAE arguments");
        auto &h = *static_cast<Handle *>(p);
        h.rt.activate();
        copy(h, h.vae.decode(h.rt.upload(latent, height * width, 64), height, width), out);
        h.vae.unload();
    });
}
}
