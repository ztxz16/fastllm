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
const char *h3_error() {
    return error.c_str();
}
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
int h3_condition_layout(int nt, const int *textTags, int frames, int height, int width, int audio, int count,
                        const int *anchors, float *positions, int *tags, int *video, int *sound) {
    try {
        auto layout = keyframeLayout(std::vector<int>(textTags, textTags + nt), frames, height, width, audio,
                                     std::vector<int>(anchors, anchors + count));
        std::copy(layout.positions.begin(), layout.positions.end(), positions);
        std::copy(layout.tags.begin(), layout.tags.end(), tags);
        std::copy(layout.video.begin(), layout.video.end(), video);
        std::copy(layout.audio.begin(), layout.audio.end(), sound);
        return layout.tags.size();
    } catch (const std::exception &e) {
        error = e.what();
        return -1;
    }
}
int h3_reference_layout(int nt, const int *textTags, int frames, int height, int width, int audio, int count,
                        const int *geometry, float *positions, int *tags, int *video, int *sound) {
    try {
        std::vector<ReferenceGeometry> refs;
        for (int i = 0; i < count; ++i)
            refs.push_back({Reference::Kind(geometry[i * 5]), geometry[i * 5 + 1], geometry[i * 5 + 2],
                            geometry[i * 5 + 3], geometry[i * 5 + 4]});
        auto layout =
            referenceLayout(std::vector<int>(textTags, textTags + nt), frames, height, width, audio, refs);
        std::copy(layout.positions.begin(), layout.positions.end(), positions);
        std::copy(layout.tags.begin(), layout.tags.end(), tags);
        std::copy(layout.video.begin(), layout.video.end(), video);
        std::copy(layout.audio.begin(), layout.audio.end(), sound);
        return layout.tags.size();
    } catch (const std::exception &e) {
        error = e.what();
        return -1;
    }
}
int h3_resize_image(int height, int width, const uint8_t *rgba, int targetHeight, int targetWidth,
                    int bicubic, uint8_t *out) {
    try {
        Image input{width, height, std::vector<uint8_t>(rgba, rgba + size_t(height) * width * 4)};
        auto result = fastllm::qwen_image::resizeImage(input, targetWidth, targetHeight,
                                                       fastllm::qwen_image::ImageResizeFilter(bicubic), 8192);
        std::copy(result.rgba.begin(), result.rgba.end(), out);
        return 0;
    } catch (const std::exception &e) {
        error = e.what();
        return -1;
    }
}
int h3_resample_audio(const float *pcm, int samples, int channels, int rate, float *output) {
    try {
        GenerateOptions options;
        options.prompt = "test";
        options.referenceImageShortEdge = 64;
        Reference image;
        image.frames.push_back({64, 64, std::vector<uint8_t>(64 * 64 * 4, 255)});
        Reference audio;
        audio.kind = Reference::Kind::Audio;
        audio.channels = channels;
        audio.sampleRate = rate;
        audio.audio.assign(pcm, pcm + size_t(samples) * channels);
        options.references = {std::move(image), std::move(audio)};
        auto prepared = prepareReferences(options);
        std::copy(prepared[1].audio.begin(), prepared[1].audio.end(), output);
        return prepared[1].audio.size() / 2;
    } catch (const std::exception &e) {
        error = e.what();
        return -1;
    }
}
int h3_sample_posterior(int frames, int height, int width, const float *moments, float *out) {
    try {
        Runtime rt(0, true);
        VideoTensor input(frames, height, width, 48);
        std::copy_n(moments, input.values.size(), input.values.data());
        auto result = sampleVideoPosterior(rt, input);
        std::copy(result.values.begin(), result.values.end(), out);
        return 0;
    } catch (const std::exception &e) {
        error = e.what();
        return -1;
    }
}
int h3_normal_noise(uint64_t seed, int count, float *output) {
    try {
        std::mt19937 generator(seed);
        auto values = normalNoise(generator, count);
        std::copy(values.begin(), values.end(), output);
        return 0;
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
void h3_destroy(void *p) {
    delete static_cast<Model *>(p);
}
int h3_block(void *p, int layer, int rows, int timeCount, const float *input, const float *times,
             const int *indices, const float *positions, const char *directory, float *output) {
    try {
        auto &m = *static_cast<Model *>(p);
        auto &rt = m.runtime;
        auto &dit = m.transformer;
        const auto &cfg = dit.config();
        auto x = rt.upload(input, rows, cfg.hidden);
        auto temb = dit.timeEmbedding(std::vector<float>(times, times + timeCount));
        auto mod = dit.projectModulation(layer, rt.cast(m.full.activation(temb), rt.dtype()));
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
int h3_audio_encode(const char *model, int rows, const float *input, const char *directory, float *output) {
    try {
        Runtime rt(0, true);
        AudioVAE vae(rt, model);
        auto result = vae.encode(rt.upload(input, rows, 1), [&](const std::string &name, const Tensor &x) {
            save(rt, x, std::string(directory) + "/" + name + ".f32");
        });
        auto host = rt.download(result);
        std::copy(host.begin(), host.end(), output);
        return 0;
    } catch (const std::exception &e) {
        error = e.what();
        return -1;
    }
}
int h3_video_decode(const char *model, int frames, int height, int width, const float *input,
                    const char *directory, float *output, int tiled, int precision) {
    try {
        Runtime rt(0, true);
        if (precision < 0 || precision > 2)
            throw std::invalid_argument("invalid video precision");
        Runtime compute(0, precision == 0 ? DType::FP32 : precision == 1 ? DType::FP16 : DType::BF16);
        VideoVAE vae(compute, rt, model);
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
int h3_video_encode(const char *model, int frames, int height, int width, const float *input,
                    const char *directory, float *output, int tiled) {
    try {
        Runtime rt(0, true);
        VideoVAE vae(rt, rt, model);
        auto trace = [&](const std::string &name, const Tensor &x) {
            if (directory && *directory)
                save(rt, x, std::string(directory) + "/" + name + ".f32");
        };
        std::vector<float> host;
        if (tiled) {
            VideoTensor pixels(frames, height, width, 3);
            std::copy_n(input, pixels.values.size(), pixels.values.data());
            host = vae.encodeMoments(pixels).values;
        } else {
            host = rt.download(
                vae.encodeTile(rt.upload(input, frames * height * width, 3), frames, height, width, trace));
        }
        std::copy(host.begin(), host.end(), output);
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
int h3_vision_operator(const char *model, const char *operation, int rows, int cols, const float *input,
                       int height, int width, float *output) {
    try {
        Runtime rt(0, false);
        Weights weights(rt, componentWeights(model));
        auto x = rt.upload(input, rows, cols);
        std::string op(operation), root = "model.visual.blocks.0.";
        Tensor y;
        if (op == "gelu")
            y = rt.activation(x, true);
        else if (op == "norm1" || op == "norm2")
            y = rt.norm(x, false, &weights.get(root + op + ".weight"), &weights.get(root + op + ".bias"));
        else if (op == "fc1" || op == "fc2")
            y = weights.linear(x, root + "mlp.linear_" + op);
        else if (op == "qkv" || op == "projected")
            y = weights.linear(x, root + (op == "qkv" ? "attn.qkv" : "attn.proj"));
        else if (op == "attention") {
            int gh = height / 16, gw = width / 16;
            std::vector<int> positions(rows * 2);
            for (int by = 0, i = 0; by < gh / 2; ++by)
                for (int bx = 0; bx < gw / 2; ++bx)
                    for (int dy = 0; dy < 2; ++dy)
                        for (int dx = 0; dx < 2; ++dx, ++i) {
                            positions[i] = by * 2 + dy;
                            positions[rows + i] = bx * 2 + dx;
                        }
            auto q = rt.visionRope(rt.columns(x, 0, 1152), positions),
                 k = rt.visionRope(rt.columns(x, 1152, 1152), positions);
            y = rt.attention(q, k, rt.columns(x, 2304, 1152), 16);
        } else
            throw std::invalid_argument("unknown isolated vision operation");
        auto values = rt.download(y);
        std::copy(values.begin(), values.end(), output);
        return 0;
    } catch (const std::exception &e) {
        error = e.what();
        return -1;
    }
}
int h3_vision_encode(const char *model, int height, int width, const uint8_t *rgba, const char *directory,
                     int fp32, int isolated, const uint8_t *secondFrame) {
    try {
        Runtime rt(0, fp32);
        Weights weights(rt, componentWeights(model));
        std::vector<uint8_t> second;
        if (secondFrame)
            second.assign(secondFrame, secondFrame + size_t(height) * width * 4);
        fastllm::nomas::qwenVision(
            rt, weights, width, height, std::vector<uint8_t>(rgba, rgba + size_t(height) * width * 4),
            [&](const std::string &name, const Tensor &x) {
                if (isolated && name.compare(0, 13, "vision.input.") == 0) {
                    std::ifstream file(std::string(directory) + "/" + name + ".reference.f32",
                                       std::ios::binary);
                    std::vector<float> data(x.size());
                    file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(float));
                    if (!file)
                        throw std::runtime_error("missing isolated vision input");
                    auto target = x;
                    std::vector<int> ids(x.rows);
                    for (int i = 0; i < x.rows; ++i)
                        ids[i] = i;
                    rt.scatter(target, rt.upload(data.data(), x.rows, x.cols), rt.indices(ids));
                }
                save(rt, x, std::string(directory) + "/" + name + ".f32");
            },
            second.empty() ? nullptr : &second);
        return 0;
    } catch (const std::exception &e) {
        error = e.what();
        return -1;
    }
}
int h3_image_text_encode(const char *model, const int *tokens, int count, int height, int width,
                         const uint8_t *rgba, const char *directory, float *output, int fp32, int copies,
                         const uint8_t *secondFrame) {
    try {
        Runtime rt(0, fp32);
        TextEncoder encoder(rt, model);
        Image image{width, height, std::vector<uint8_t>(rgba, rgba + size_t(height) * width * 4)};
        VisionInput visual{image, {}, secondFrame ? 151656 : 151655};
        if (secondFrame)
            visual.second = Image{
                width, height, std::vector<uint8_t>(secondFrame, secondFrame + size_t(height) * width * 4)};
        auto result = encoder.encodeVisual(std::vector<int>(tokens, tokens + count),
                                           std::vector<VisionInput>(copies, visual),
                                           [&](const std::string &name, const Tensor &x) {
                                               save(rt, x, std::string(directory) + "/" + name + ".f32");
                                           });
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
