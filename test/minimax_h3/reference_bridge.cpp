#include "models/minimax_h3/runtime.h"
#include <fstream>
#include <stdexcept>

using namespace fastllm::minimax_h3;
namespace {
thread_local std::string error;
struct Model {
    Runtime runtime, full;
    Transformer transformer;
    Model(const char *path, bool fp32) : runtime(0, fp32), full(0, true), transformer(runtime, full, path) {}
};
void save(Runtime &rt, const Tensor &x, const std::string &path) {
    auto data = rt.download(x);
    std::ofstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("cannot write reference tensor: " + path);
    file.write(reinterpret_cast<const char *>(data.data()), data.size() * 4);
}
} // namespace
extern "C" {
const char *h3_error() { return error.c_str(); }
int h3_tokenize(const char *directory, const char *text, int *output, int capacity) {
    try {
        fastllm::nomas::QwenTokenizer tokenizer(directory);
        auto tokens = tokenizer.encode(text);
        if (tokens.size() > size_t(capacity))
            throw std::invalid_argument("token output buffer is too small");
        std::copy(tokens.begin(), tokens.end(), output);
        return tokens.size();
    } catch (const std::exception &e) {
        error = e.what();
        return -1;
    }
}
int h3_layout(int nt, int frames, int height, int width, int audio, float *positions, int *tags) {
    try {
        auto layout = textToVideoLayout(nt, frames, height, width, audio);
        std::copy(layout.positions.begin(), layout.positions.end(), positions);
        std::copy(layout.tags.begin(), layout.tags.end(), tags);
        return layout.tags.size();
    } catch (const std::exception &e) {
        error = e.what();
        return -1;
    }
}
int h3_schedule(int steps, float shift, float *output) {
    try {
        auto schedule = sigmaSchedule(steps, shift);
        std::copy(schedule.begin(), schedule.end(), output);
        return schedule.size();
    } catch (const std::exception &e) {
        error = e.what();
        return -1;
    }
}
void *h3_create(const char *path, int fp32) {
    try {
        return new Model(path, fp32);
    } catch (const std::exception &e) {
        error = e.what();
        return nullptr;
    }
}
void h3_destroy(void *p) { delete static_cast<Model *>(p); }
int h3_block(void *p, int layer, int rows, int timeCount, const float *input, const float *times,
             const int *indices, const float *positions, const char *directory, float *output) {
    try {
        auto &m = *static_cast<Model *>(p);
        auto &rt = m.runtime;
        auto &dit = m.transformer;
        const auto &cfg = dit.config();
        auto x = rt.upload(input, rows, cfg.hidden);
        auto temb = dit.timeEmbedding(std::vector<float>(times, times + timeCount));
        auto mod = dit.projectModulation(layer, rt.cast(m.full.activation(temb), rt.isFloat32()));
        auto rope =
            rt.ropeTable(std::vector<float>(positions, positions + rows * 3), cfg.ropeFreq, cfg.ropeTheta);
        auto trace = [&](const std::string &name, const Tensor &value) {
            if (directory && *directory)
                save(rt, value, std::string(directory) + "/" + name + ".f32");
        };
        trace("temb", temb);
        trace("mod", mod);
        trace("cos", rope.first);
        trace("sin", rope.second);
        auto result = dit.block(layer, x, mod, rt.indices(std::vector<int>(indices, indices + rows)),
                                rope.first, rope.second, trace);
        auto host = rt.download(result);
        dit.unload();
        std::copy(host.begin(), host.end(), output);
        return 0;
    } catch (const std::exception &e) {
        error = e.what();
        return -1;
    }
}
int h3_audio_decode(const char *model, int rows, const float *input, const char *directory, float *output) {
    try {
        Runtime rt(0, true);
        AudioVAE vae(rt, model);
        auto trace = [&](const std::string &name, const Tensor &x) {
            if (directory && *directory)
                save(rt, x, std::string(directory) + "/" + name + ".f32");
        };
        auto result = vae.decode(rt.upload(input, rows, 32), trace);
        if (result.size() != size_t(rows) * 800)
            throw std::runtime_error("incorrect decoded audio length");
        auto host = rt.download(result);
        std::copy(host.begin(), host.end(), output);
        return 0;
    } catch (const std::exception &e) {
        error = e.what();
        return -1;
    }
}
int h3_video_decode(const char *model, int frames, int height, int width, const float *input,
                    const char *directory, float *output, int tiled) {
    try {
        Runtime rt(0, true);
        VideoVAE vae(rt, model);
        auto trace = [&](const std::string &name, const Tensor &x) {
            if (directory && *directory)
                save(rt, x, std::string(directory) + "/" + name + ".f32");
        };
        if (tiled) {
            VideoTensor z(frames, height, width, 24);
            std::copy_n(input, z.values.size(), z.values.data());
            auto video = vae.decode(z);
            std::copy(video.values.begin(), video.values.end(), output);
        } else {
            auto result =
                vae.decodeTile(rt.upload(input, frames * height * width, 24), frames, height, width, trace);
            auto host = rt.download(result);
            std::copy(host.begin(), host.end(), output);
        }
        return 0;
    } catch (const std::exception &e) {
        error = e.what();
        return -1;
    }
}
int h3_text_encode(const char *model, const int *tokens, int count, const char *directory, float *output,
                   int fp32) {
    try {
        Runtime rt(0, fp32);
        TextEncoder encoder(rt, model);
        auto trace = [&](const std::string &name, const Tensor &x) {
            if (directory && *directory)
                save(rt, x, std::string(directory) + "/" + name + ".f32");
        };
        auto result = encoder.encode(std::vector<int>(tokens, tokens + count), trace);
        auto host = rt.download(result);
        std::copy(host.begin(), host.end(), output);
        return 0;
    } catch (const std::exception &e) {
        error = e.what();
        return -1;
    }
}
int h3_transformer(const char *model, int nt, int na, int nv, int timeCount, const float *text,
                   const float *audio, const float *video, const float *times, const int *indices,
                   const float *positions, const char *directory, float *outVideo, float *outAudio,
                   int fp32) {
    try {
        Runtime rt(0, fp32), full(0, true);
        Transformer dit(rt, full, model);
        PackedLayout layout;
        int n = nt + na + nv;
        layout.timeIndices.assign(indices, indices + n);
        layout.positions.assign(positions, positions + n * 3);
        for (int i = 0; i < n; ++i) {
            if (i < nt) {
                layout.text.push_back(i);
                layout.tags.push_back(1);
            } else if (i < nt + na) {
                layout.audio.push_back(i);
                layout.tags.push_back(2);
            } else {
                layout.video.push_back(i);
                layout.tags.push_back(0);
            }
        }
        dit.prepare({std::vector<float>(times, times + timeCount)});
        auto trace = [&](const std::string &name, const Tensor &x) {
            if (directory && *directory)
                save(rt, x, std::string(directory) + "/" + name + ".f32");
        };
        auto result = dit.forward(full.upload(video, nv, 96), full.upload(audio, na, 32),
                                  rt.upload(text, nt, 5120), layout, trace);
        auto v = rt.download(result.first), a = rt.download(result.second);
        std::copy(v.begin(), v.end(), outVideo);
        std::copy(a.begin(), a.end(), outAudio);
        return 0;
    } catch (const std::exception &e) {
        error = e.what();
        return -1;
    }
}
}
