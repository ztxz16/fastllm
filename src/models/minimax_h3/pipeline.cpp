#include "models/minimax_h3/runtime.h"
#include "json11.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace fastllm {
namespace minimax_h3 {
void GenerateOptions::validate() const {
    if (prompt.empty() || prompt.size() > 32768)
        throw std::invalid_argument("prompt must contain 1..32768 UTF-8 bytes");
    if (width < 64 || height < 64 || width > 4096 || height > 4096 || width % 32 || height % 32 ||
        int64_t(width) * height > 1344 * 768)
        throw std::invalid_argument(
            "size must be multiples of 32, each 64..4096, and at most 1032192 pixels");
    if (frames < 120 || frames > 345 || frames % 17 != 5)
        throw std::invalid_argument("frames must be 124,141,...,345 (24 fps)");
    if (steps < 2 || steps > 100)
        throw std::invalid_argument("num_inference_steps must be 2..100");
}
std::vector<float> sigmaSchedule(int steps, float shift) {
    if (steps < 2 || steps > 1000 || !std::isfinite(shift) || shift <= 0)
        throw std::invalid_argument("invalid H3 sigma schedule");
    std::vector<float> out;
    float step = -1.f / (steps - 1);
    for (int i = 0; i < steps; ++i) {
        // PyTorch linspace uses a float step and a fused start + step * index.
        float base = i < steps / 2 ? std::fma(step, float(i), 1.f) : -step * (steps - i - 1);
        float sigma = (shift * base) / (1.f + (shift - 1.f) * base);
        if (out.empty() || sigma != out.back())
            out.push_back(sigma);
    }
    return out;
}
PackedLayout textToVideoLayout(int nt, int frames, int h, int w, int audio) {
    if (nt < 1 || nt > 4096 || frames < 1 || frames > 1024 || h < 2 || w < 2 || h > 256 || w > 256 || h % 2 ||
        w % 2 || audio < 1 || audio > 16384)
        throw std::invalid_argument("invalid H3 latent geometry");
    PackedLayout out;
    int perFrame = (h / 2) * (w / 2), count = nt + audio * 2 + frames * perFrame;
    out.positions.resize(size_t(count) * 3);
    out.tags.resize(count);
    out.timeIndices.resize(count);
    double sqrtArea = std::sqrt(double(h) * w), rh = h / sqrtArea, rw = w / sqrtArea;
    auto hp = [&](int y) { return ((1 - rh) / 2 + double(y) * (rh / (h / 2))) * 32; };
    auto wp = [&](int x) { return ((1 - rw) / 2 + double(x) * (rw / (w / 2))) * 32; };
    for (int i = 0; i < nt; ++i) {
        out.text.push_back(i);
        out.tags[i] = 1;
        out.positions[i * 3] = i;
    }
    for (int i = 0; i < audio * 2; ++i) {
        int row = nt + i;
        out.audio.push_back(row);
        out.tags[row] = 2;
        out.positions[row * 3] = nt + i % audio;
        out.positions[row * 3 + 2] = wp(i < audio ? 0 : w / 2 - 1);
    }
    double time = nt;
    for (int t = 0; t < frames; ++t) {
        for (int y = 0; y < h / 2; ++y)
            for (int x = 0; x < w / 2; ++x) {
                int row = nt + audio * 2 + t * perFrame + y * (w / 2) + x;
                out.video.push_back(row);
                out.tags[row] = 0;
                out.positions[row * 3] = time;
                out.positions[row * 3 + 1] = hp(y);
                out.positions[row * 3 + 2] = wp(x);
            }
        time += (5.0 / 3.0) * (t % 5 == 0 ? 1 : 4);
    }
    return out;
}
Pipeline::Pipeline(const std::string &dir, int device, bool fp32)
    : Pipeline(dir, device, fp32, parseVideoPrecision({}, fp32)) {}
Pipeline::Pipeline(const std::string &dir, int device, bool fp32, DType videoPrecision)
    : rt_(device, fp32), full_(device, true), videoRuntime_(device, videoPrecision),
      tokenizer_(dir + "/tokenizer"), text_(rt_, dir + "/text_encoder"),
      transformer_(rt_, full_, dir + "/transformer"), video_(videoRuntime_, full_, dir + "/vae"),
      audio_(full_, dir + "/audio_vae") {
    auto stats = [&](const std::string &path, int channels, std::vector<float> &mean,
                     std::vector<float> &std) {
        std::ifstream file(path + "/config.json");
        std::string s((std::istreambuf_iterator<char>(file)), {}), error;
        auto j = json11::Json::parse(s, error);
        for (auto &x : j["latents_mean"].array_items())
            mean.push_back(x.number_value());
        for (auto &x : j["latents_std"].array_items())
            std.push_back(x.number_value());
        if (!error.empty() || mean.size() != size_t(channels) || std.size() != mean.size())
            throw std::runtime_error("invalid H3 latent normalization: " + path);
    };
    stats(dir + "/vae", 24, videoMean_, videoStd_);
    stats(dir + "/audio_vae", 32, audioMean_, audioStd_);
    // The BF16 DiT and mixed-precision VAEs fit together on 72/80 GiB devices.
    // Leave at least 60 GiB initially free before enabling cross-request residency;
    // smaller/shared devices and FP32 diagnostics retain the streaming lifecycle.
    size_t available = 0, total = 0;
    auto status = cudaMemGetInfo(&available, &total);
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
    retainWeights_ = !fp32 && videoPrecision != DType::FP32 && total >= (size_t(68) << 30) &&
                     available >= (size_t(60) << 30);
}
Generation Pipeline::generate(const GenerateOptions &o, Progress progress, Trace trace,
                              const std::vector<float> *initialVideo,
                              const std::vector<float> *initialAudio) {
    return generateImpl(o, std::move(progress), std::move(trace), initialVideo, initialAudio, {}, {});
}
Generation Pipeline::generateMP4(const GenerateOptions &o, const std::string &path,
                                 const std::string &ffmpeg, Progress progress, Trace trace,
                                 const std::vector<float> *initialVideo,
                                 const std::vector<float> *initialAudio) {
    if (path.empty())
        throw std::invalid_argument("MP4 output path is empty");
    return generateImpl(o, std::move(progress), std::move(trace), initialVideo, initialAudio, path, ffmpeg);
}
Generation Pipeline::generateImpl(const GenerateOptions &o, Progress progress, Trace trace,
                                   const std::vector<float> *initialVideo,
                                   const std::vector<float> *initialAudio,
                                   const std::string &path, const std::string &ffmpeg) {
    o.validate();
    rt_.activate();
    auto tick = [&](const std::string &stage, int i, int n) {
        if (progress)
            progress(stage, i, n);
    };
    auto unload = [&] {
        text_.unload();
        transformer_.unload();
        video_.unload();
        audio_.unload();
    };
    try {
        auto tokens = tokenizer_.encode(o.prompt);
        tick("text", 0, 50);
        auto text = text_.encode(tokens, [&](const std::string &name, const Tensor &) {
            tick("text", std::stoi(name.substr(6)) + 1, 50);
        });
        text_.unload();
        if (trace)
            trace("text", text);
        int lt = (o.frames - 5) / 17 * 5 + 2, lh = o.height / 16, lw = o.width / 16,
            at = int(std::nearbyint(o.frames * (40.0 / 24.0)));
        auto layout = textToVideoLayout(tokens.size(), lt, lh, lw, at);
        std::mt19937_64 rng(o.seed);
        std::normal_distribution<float> normal;
        std::vector<float> video(size_t(lt) * lh * lw * 24), audio(size_t(at) * 2 * 32);
        if (initialVideo) {
            if (initialVideo->size() != video.size())
                throw std::invalid_argument("initial video noise size mismatch");
            video = *initialVideo;
        } else
            for (auto &v : video)
                v = normal(rng);
        if (initialAudio) {
            if (initialAudio->size() != audio.size())
                throw std::invalid_argument("initial audio noise size mismatch");
            audio = *initialAudio;
        } else
            for (auto &v : audio)
                v = normal(rng);
        // Initial video fixture uses packed [T,H/2,W/2,C,2,2] layout, matching transformer rows.
        auto v = full_.upload(video.data(), layout.video.size(), 96),
             a = full_.upload(audio.data(), at * 2, 32);
        if (trace) {
            trace("initial_video", v);
            trace("initial_audio", a);
        }
        auto vs = sigmaSchedule(o.steps, 12), as = sigmaSchedule(o.steps, 3);
        if (vs.size() != as.size())
            throw std::runtime_error("H3 audio/video schedule lengths differ");
        std::vector<std::vector<float>> plans;
        std::vector<std::pair<int, int>> timeIndices;
        int offset = 0;
        for (size_t i = 0; i + 1 < vs.size(); ++i) {
            std::vector<float> times{1.f - vs[i], 1.f - as[i]};
            std::sort(times.begin(), times.end());
            times.erase(std::unique(times.begin(), times.end()), times.end());
            int vi = offset + std::lower_bound(times.begin(), times.end(), 1.f - vs[i]) - times.begin();
            int ai = offset + std::lower_bound(times.begin(), times.end(), 1.f - as[i]) - times.begin();
            timeIndices.emplace_back(vi, ai);
            offset += times.size();
            plans.push_back(std::move(times));
        }
        tick("prepare", 0, 50);
        transformer_.prepare(plans, [&](int current, int total) { tick("prepare", current, total); });
        tick("prepare", 50, 50);
        for (size_t i = 0; i + 1 < vs.size(); ++i) {
            auto [vi, ai] = timeIndices[i];
            for (size_t row = 0; row < layout.tags.size(); ++row)
                layout.timeIndices[row] = layout.tags[row] == 2 ? ai : vi;
            tick("denoise", i, vs.size() - 1);
            auto velocity =
                transformer_.forward(v, a, text, layout, [&](const std::string &, const Tensor &) {
                    tick("denoise", i, vs.size() - 1);
                });
            v = full_.step(v, velocity.first, 1.f - vs[i], vs[i], vs[i + 1]);
            a = full_.step(a, velocity.second, 1.f - as[i], as[i], as[i + 1]);
            if (trace) {
                trace("video_step." + std::to_string(i), v);
                trace("audio_step." + std::to_string(i), a);
            }
        }
        video = full_.download(v);
        audio = full_.download(a);
        v = {};
        a = {};
        text = {};
        if (!retainWeights_)
            transformer_.unload();
        VideoTensor latents(lt, lh, lw, 24);
        for (int t = 0; t < lt; ++t)
            for (int y = 0; y < lh; ++y)
                for (int x = 0; x < lw; ++x)
                    for (int c = 0; c < 24; ++c) {
                        size_t row = (size_t(t) * (lh / 2) + y / 2) * (lw / 2) + x / 2;
                        size_t src = row * 96 + (c * 2 + y % 2) * 2 + x % 2;
                        latents.values[((size_t(t) * lh + y) * lw + x) * 24 + c] =
                            video[src] * videoStd_[c] + videoMean_[c];
                    }
        Generation result;
        result.width = o.width;
        result.height = o.height;
        result.frames = o.frames;
        result.audio.resize(size_t(at) * 800 * 2);
        for (int channel = 0; channel < 2; ++channel) {
            tick("audio_decode", channel, 2);
            auto first = audio.data() + size_t(channel) * at * 32;
            for (int t = 0; t < at; ++t)
                for (int c = 0; c < 32; ++c)
                    first[t * 32 + c] = first[t * 32 + c] * audioStd_[c] + audioMean_[c];
            auto samples = full_.download(
                audio_.decode(full_.upload(first, at, 32), [&](const std::string &, const Tensor &) {
                    tick("audio_decode", channel, 2);
                }));
            for (size_t i = 0; i < samples.size(); ++i)
                result.audio[i * 2 + channel] = samples[i];
        }
        if (!retainWeights_)
            audio_.unload();
        tick("video_decode", 0, 1);
        std::unique_ptr<MP4Writer> writer;
        int videoCurrent = 0, videoTotal = 1;
        if (!path.empty())
            writer = std::make_unique<MP4Writer>(result, path, ffmpeg);
        std::function<void(const std::vector<uint8_t> &)> onFrames;
        if (writer)
            onFrames = [&](const std::vector<uint8_t> &rgb) {
                writer->write(rgb.data(), rgb.size(), [&] { tick("video_decode", videoCurrent, videoTotal); });
            };
        result.rgb = video_.decodeRGB(
            latents, [&](int current, int total) {
                videoCurrent = current;
                videoTotal = total;
                tick("video_decode", current, total);
            },
            std::move(onFrames));
        if (!retainWeights_)
            video_.unload();
        tick("video_decode", 1, 1);
        if (result.rgb.size() != size_t(o.frames) * o.width * o.height * 3)
            throw std::runtime_error("decoded H3 frame count mismatch");
        latents = {};
        if (writer) {
            tick("encode", 0, 1);
            writer->finish([&] { tick("encode", 0, 1); });
        } else
            tick("completed", 1, 1);
        return result;
    } catch (...) {
        unload();
        // Finish queued frees before reporting cancellation; otherwise the pool
        // can retain the entire packed cache until another request synchronizes.
        rt_.synchronize();
        throw;
    }
}
} // namespace minimax_h3
} // namespace fastllm
