#include "models/minimax_h3/runtime.h"
#include "models/minimax_h3/parallel.h"
#include "json11.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <cstdio>
#include <stdexcept>

namespace fastllm {
namespace minimax_h3 {
namespace {
double validateGpuMemoryLimit(double gib) {
    if (!std::isfinite(gib) || gib < 0 || gib > 1048576)
        throw std::invalid_argument("gpu-memory-limit must be a finite GiB value between 0 and 1048576");
    if (gib != 0 && gib <= 1)
        throw std::invalid_argument("gpu-memory-limit must exceed the 1 GiB CUDA/library reserve");
    return gib;
}
double validateHostMemoryLimit(double gib) {
    if (!std::isfinite(gib) || gib < 0 || gib > 1048576 || (gib > 0 && gib < 2))
        throw std::invalid_argument("host-memory-limit must be 0 or a finite GiB number of at least 2");
    return gib;
}
double parseMemoryLimit(const std::string &value, const std::string &name) {
    size_t used = 0;
    double gib = 0;
    try {
        gib = std::stod(value, &used);
    } catch (const std::exception &) {
        throw std::invalid_argument(name + " must be a finite GiB number");
    }
    if (used != value.size()) throw std::invalid_argument(name + " must be a GiB number");
    return gib;
}
} // namespace
double parseGpuMemoryLimit(const std::string &value) {
    return validateGpuMemoryLimit(parseMemoryLimit(value, "gpu-memory-limit"));
}
double parseHostMemoryLimit(const std::string &value) {
    return validateHostMemoryLimit(parseMemoryLimit(value, "host-memory-limit"));
}
void GenerateOptions::validate() const {
    if (prompt.empty() || prompt.size() > 32768)
        throw std::invalid_argument("prompt must contain 1..32768 UTF-8 bytes");
    if (width < 64 || height < 64 || width > 4096 || height > 4096 || width % 32 || height % 32 ||
        int64_t(width) * height > 1344 * 768)
        throw std::invalid_argument(
            "size must be multiples of 32, each 64..4096, and at most 1032192 pixels");
    if (frames < 120 || frames > 345 || frames % 17 != 5)
        throw std::invalid_argument("frames must be 124,141,...,345 (24 fps)");
    if (!references.empty() && (firstFrame || lastFrame))
        throw std::invalid_argument("keyframes and Ref2VA references are mutually exclusive");
    int imageCount = 0, videoCount = 0, audioCount = 0;
    for (const auto &ref : references) {
        imageCount += ref.kind == Reference::Kind::Image;
        videoCount += ref.kind == Reference::Kind::Video;
        audioCount += ref.kind == Reference::Kind::Audio;
    }
    if (imageCount > 9 || videoCount > 3 || audioCount > 3 || references.size() > 12 ||
        (audioCount && !imageCount && !videoCount))
        throw std::invalid_argument("invalid reference count: at most 9 images, 3 videos, 3 audio clips, 12 "
                                    "total; audio requires image/video");
    if (referenceImageShortEdge < 64 || referenceImageShortEdge > 2048 || referenceImageShortEdge % 32)
        throw std::invalid_argument("reference_image_short_edge must be a multiple of 32 in 64..2048");
    if (referenceVideoShortEdge < 64 || referenceVideoShortEdge > 768 || referenceVideoShortEdge % 32)
        throw std::invalid_argument("reference_video_short_edge must be a multiple of 32 in 64..768");
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
    if (nt < 1 || nt > 65536 || frames < 1 || frames > 1024 || h < 2 || w < 2 || h > 256 || w > 256 ||
        h % 2 || w % 2 || audio < 1 || audio > 16384)
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
      audio_(full_, dir + "/audio_vae"), directory_(dir) {
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
Pipeline::Pipeline(const std::string &dir, const std::vector<int> &devices, bool fp32, DType videoPrecision)
    : Pipeline(dir, devices, fp32, videoPrecision, nomas::RowReduction::Activation) {}
Pipeline::Pipeline(const std::string &dir, const std::vector<int> &devices, bool fp32, DType videoPrecision,
                   nomas::RowReduction ditReduction)
    : Pipeline(dir, devices.empty() ? throw std::invalid_argument("empty CUDA device list") : devices[0],
               fp32, videoPrecision) {
    if (devices.size() > 1) {
        parallel_ = std::make_unique<ParallelModel>(dir, devices, fp32, videoPrecision, video_, ditReduction);
        video_.setTileRunner([this](size_t count, const std::function<Tensor(VideoVAE &,size_t)> &task) {
            return parallel_->videoTiles(count, task);
        }, devices.size());
    }
    rt_.activate();
}
Pipeline::~Pipeline() = default;
void Pipeline::setGpuMemoryLimit(double gib) {
    validateGpuMemoryLimit(gib);
    if (gib == 0) return;
    rt_.activate();
    int device = 0;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    size_t bytes = size_t(gib * double(size_t(1) << 30)) - (size_t(1) << 30);
    // Runtime workspaces and CUDA/library state precede this scope. Every model
    // tensor, conversion scratch buffer and attention workspace is pool-bound.
    memoryPool_ = std::make_shared<nomas::TensorPool>(device, 0, bytes, true);
    // A generous budget keeps the original resident/prefetched execution path.
    // Conservative headroom covers the supported H3/Pika working shapes; a
    // request that exceeds it still fails against the same physical pool cap.
    lowMemory_ = rt_.isFloat32() || videoRuntime_.isFloat32() || gib < (parallel_ ? 48 : 60);
    text_.unload();
    text_.setMemoryPool(memoryPool_);
    text_.setLowMemory(lowMemory_);
    transformer_.setLowMemory(lowMemory_);
    if (referenceTransformer_) referenceTransformer_->setLowMemory(lowMemory_);
    video_.setLowMemory(lowMemory_);
    audio_.unload();
    if (parallel_) parallel_->setMemoryLimit(bytes, memoryPool_, lowMemory_);
    if (lowMemory_) retainWeights_ = false;
    rt_.activate();
}
void Pipeline::setHostMemoryLimit(double gib) {
    validateHostMemoryLimit(gib);
    if (gib == 0) return;
    hostBudget_ = std::make_shared<nomas::HostMemoryBudget>(size_t((gib - 2) * double(size_t(1) << 30)));
    text_.setHostMemoryBudget(hostBudget_);
    transformer_.setHostMemoryBudget(hostBudget_);
    if (referenceTransformer_) referenceTransformer_->setHostMemoryBudget(hostBudget_);
    video_.setHostMemoryBudget(hostBudget_);
    audio_.setHostMemoryBudget(hostBudget_);
    if (parallel_) parallel_->setHostMemoryBudget(hostBudget_);
}
Generation Pipeline::generate(const GenerateOptions &o, Progress progress, Trace trace,
                              const std::vector<float> *initialVideo,
                              const std::vector<float> *initialAudio) {
    return generateImpl(o, std::move(progress), std::move(trace), initialVideo, initialAudio, {}, {}, true);
}
Generation Pipeline::generateMP4(const GenerateOptions &o, const std::string &path, const std::string &ffmpeg,
                                 Progress progress, Trace trace, const std::vector<float> *initialVideo,
                                 const std::vector<float> *initialAudio, bool retainRGB) {
    if (path.empty())
        throw std::invalid_argument("MP4 output path is empty");
    return generateImpl(o, std::move(progress), std::move(trace), initialVideo, initialAudio, path, ffmpeg, retainRGB);
}
Generation Pipeline::generateImpl(const GenerateOptions &o, Progress progress, Trace trace,
                                  const std::vector<float> *initialVideo,
                                  const std::vector<float> *initialAudio, const std::string &path,
                                  const std::string &ffmpeg, bool retainRGB) {
    o.validate();
    const bool pika = !pikaDirectory_.empty();
    if (pika && o.steps != 6)
        throw std::invalid_argument("Pika H3 Flex requires exactly 6 inference steps");
    rt_.activate();
    auto memoryScope = memoryPool_ ? std::make_unique<nomas::TensorPool::Scope>(*memoryPool_) : nullptr;
    auto tick = [&](const std::string &stage, int i, int n) {
        if (progress)
            progress(stage, i, n);
    };
    auto unload = [&] {
        if (parallel_)
            parallel_->unload();
        text_.unload();
        transformer_.unload();
        if (referenceTransformer_)
            referenceTransformer_->unload();
        video_.unload();
        audio_.unload();
    };
    try {
        auto images = prepareKeyframes(o);
        auto references = prepareReferences(o);
        const bool referenceMode = !references.empty();
        bool conditioned = !images.empty() || !references.empty();
        Transformer *denoiser = &transformer_;
        if (!parallel_ && referenceMode) {
            transformer_.unload();
            if (!referenceTransformer_)
                referenceTransformer_ =
                    std::make_unique<Transformer>(rt_, full_, directory_ + "/transformer_ref");
            referenceTransformer_->setLowMemory(lowMemory_);
            referenceTransformer_->setHostMemoryBudget(hostBudget_);
            denoiser = referenceTransformer_.get();
        } else if (referenceTransformer_)
            referenceTransformer_->unload();
        std::vector<int> tokens, textTags, anchors;
        std::vector<VisionInput> visual;
        auto label = [&](const std::string &value) {
            auto ids = tokenizer_.encode(value);
            tokens.insert(tokens.end(), ids.begin(), ids.end());
            textTags.insert(textTags.end(), ids.size(), 1);
        };
        auto vision = [&](const Image &image, const Image *second = nullptr, int sampledFrames = 0) {
            VisionInput input{prepareVisionImage(image, sampledFrames), {}, second ? 151656 : 151655};
            if (second)
                input.second = prepareVisionImage(*second, sampledFrames);
            int slots = input.first.width / 32 * (input.first.height / 32);
            tokens.push_back(151652);
            tokens.insert(tokens.end(), slots, input.token);
            tokens.push_back(151653);
            textTags.insert(textTags.end(), slots + 2, 0);
            visual.push_back(std::move(input));
        };
        if (o.firstFrame)
            anchors.push_back(0);
        if (o.lastFrame)
            anchors.push_back(-1);
        for (size_t i = 0; i < images.size(); ++i) {
            label("<Picture " + std::to_string(i + 1) + ">: ");
            vision(images[i]);
        }
        int imageCount = 0, videoCount = 0, audioCount = 0;
        for (const auto &ref : references) {
            if (!ref.audio.empty())
                label("<Audio " + std::to_string(++audioCount) + ">: ");
            if (ref.kind == Reference::Kind::Image) {
                label("<Picture " + std::to_string(++imageCount) + ">: ");
                vision(ref.frames[0]);
            } else if (ref.kind == Reference::Kind::Video) {
                label("<Video " + std::to_string(++videoCount) + ">: ");
                int samples = (ref.frames.size() + 11) / 12;
                for (int i = 0; i < samples; i += 2) {
                    int second = std::min(i + 1, samples - 1);
                    char timestamp[64];
                    std::snprintf(timestamp, sizeof(timestamp), "<%.1f seconds>", (i + second) / 4.);
                    label(timestamp);
                    vision(ref.frames[i * 12], &ref.frames[second * 12], samples);
                }
            }
        }
        auto prompt = tokenizer_.encode(o.prompt);
        tokens.insert(tokens.end(), prompt.begin(), prompt.end());
        textTags.insert(textTags.end(), prompt.size(), 1);
        tick("text", 0, 50);
        auto textProgress = [&](const std::string &name, const Tensor &) {
            if (name.compare(0, 6, "layer.") == 0)
                tick("text", std::stoi(name.substr(6)) + 1, 50);
            else
                tick("vision", 0, visual.size());
        };
        auto text = parallel_ ? parallel_->encode(tokens, visual, textProgress)
                              : text_.encodeVisual(tokens, visual, textProgress);
        text_.unload();
        if (trace)
            trace("text", text);
        int lt = (o.frames - 5) / 17 * 5 + 2, lh = o.height / 16, lw = o.width / 16,
            at = int(std::nearbyint(o.frames * (40.0 / 24.0)));
        visual.clear();
        auto pack = [](const std::vector<float> &channels, int frames, int lh, int lw) {
            std::vector<float> rows(channels.size());
            for (int t = 0; t < frames; ++t)
                for (int y = 0; y < lh; ++y)
                    for (int x = 0; x < lw; ++x)
                        for (int c = 0; c < 24; ++c) {
                            size_t dst = ((size_t(t) * (lh / 2) + y / 2) * (lw / 2) + x / 2) * 96 + c * 4 +
                                         y % 2 * 2 + x % 2;
                            rows[dst] = channels[((size_t(c) * frames + t) * lh + y) * lw + x];
                        }
            return rows;
        };
        std::mt19937 conditionGenerator(pika ? o.seed + 7919 : o.seed);
        std::vector<float> condition, audioCondition;
        std::vector<ReferenceGeometry> geometries;
        auto encodeFrames = [&](const std::vector<Image> &frames) {
            int height = frames[0].height, width = frames[0].width;
            VideoTensor pixels(frames.size(), height, width, 3);
            const float mean[] = {.485f, .456f, .406f}, std[] = {.229f, .224f, .225f};
            size_t framePixels = size_t(height) * width;
            for (size_t t = 0; t < frames.size(); ++t)
                for (size_t i = 0; i < framePixels * 3; ++i)
                    pixels.values[t * framePixels * 3 + i] =
                        (float(frames[t].rgba[i / 3 * 4 + i % 3]) / 255.f - mean[i % 3]) / std[i % 3];
            auto moments =
                video_.encodeMoments(pixels, [&](int i, int n) { tick("condition_video_encode", i, n); });
            auto sampled = sampleVideoPosterior(full_, moments);
            auto noise = normalNoise(conditionGenerator, sampled.values.size());
            std::vector<float> normalized(noise.size());
            size_t rows = size_t(sampled.frames) * sampled.height * sampled.width;
            for (size_t row = 0; row < rows; ++row)
                for (int c = 0; c < 24; ++c) {
                    size_t pos = size_t(c) * rows + row;
                    float latent = (sampled.values[row * 24 + c] - videoMean_[c]) / videoStd_[c];
                    normalized[pos] = latent;
                }
            if (pika && !referenceMode)
                normalized = rt_.download(rt_.upload(normalized.data(), rows, 24));
            for (size_t i = 0; i < normalized.size(); ++i)
                normalized[i] = .999f * normalized[i] + (pika ? .001f : 1.f - .999f) * noise[i];
            auto packed = pack(normalized, sampled.frames, sampled.height, sampled.width);
            condition.insert(condition.end(), packed.begin(), packed.end());
            return ReferenceGeometry{frames.size() == 1 ? Reference::Kind::Image : Reference::Kind::Video,
                                     sampled.frames, sampled.height, sampled.width, 0};
        };
        for (const auto &image : images)
            encodeFrames({image});
        for (auto &ref : references) {
            ReferenceGeometry geometry{ref.kind};
            if (ref.kind == Reference::Kind::Video) {
                int count = std::max(1, (int(ref.frames.size()) - 5) / 17) * 17 + 5;
                ref.frames.resize(std::min(size_t(count), ref.frames.size()));
            }
            if (!ref.frames.empty())
                geometry = encodeFrames(ref.frames);
            if (!ref.audio.empty()) {
                size_t samples = ref.audio.size() / 2;
                for (int channel = 0; channel < 2; ++channel) {
                    tick("condition_audio_encode", channel, 2);
                    std::vector<float> mono(samples);
                    for (size_t i = 0; i < samples; ++i)
                        mono[i] = ref.audio[i * 2 + channel];
                    auto encoded = audio_.encode(full_.upload(mono.data(), samples, 1));
                    auto values = full_.download(encoded);
                    geometry.audioFrames = encoded.rows;
                    for (size_t i = 0; i < values.size(); ++i)
                        values[i] = (values[i] - audioMean_[i % 32]) / audioStd_[i % 32];
                    audioCondition.insert(audioCondition.end(), values.begin(), values.end());
                }
            }
            geometries.push_back(geometry);
        }
        auto layout = references.empty() ? keyframeLayout(textTags, lt, lh, lw, at, anchors)
                                         : referenceLayout(textTags, lt, lh, lw, at, geometries);
        int conditionRows = condition.size() / 96, audioConditionRows = audioCondition.size() / 32;
        // Reuse both VAE encoders and decoders when the residency budget permits.
        // Smaller devices keep the streaming lifecycle before loading the denoiser.
        if (conditioned && !retainWeights_) {
            if (parallel_) parallel_->unloadVideo();
            else video_.unload();
            audio_.unload();
        }
        images.clear();
        references.clear();
        std::mt19937_64 rng(o.seed);
        std::normal_distribution<float> normal;
        std::vector<float> video(size_t(lt) * lh * lw * 24), audio(size_t(at) * 2 * 32);
        if (initialVideo) {
            if (initialVideo->size() != video.size())
                throw std::invalid_argument("initial video noise size mismatch");
            video = *initialVideo;
        } else if (conditioned && !pika)
            video = pack(normalNoise(conditionGenerator, video.size()), lt, lh, lw);
        else
            for (auto &v : video)
                v = normal(rng);
        if (initialAudio) {
            if (initialAudio->size() != audio.size())
                throw std::invalid_argument("initial audio noise size mismatch");
            audio = *initialAudio;
        } else if (conditioned && !pika)
            audio = normalNoise(conditionGenerator, audio.size());
        else
            for (auto &v : audio)
                v = normal(rng);
        // Initial video fixture uses packed [T,H/2,W/2,C,2,2] layout, matching transformer rows.
        auto v = full_.upload(video.data(), layout.video.size() - conditionRows, 96),
             a = full_.upload(audio.data(), at * 2, 32);
        if (trace) {
            trace("initial_video", v);
            trace("initial_audio", a);
        }
        if (conditionRows) {
            auto prefix = full_.upload(condition.data(), conditionRows, 96);
            if (trace)
                trace("condition_video", prefix);
            v = full_.concat({prefix, v});
        }
        if (audioConditionRows) {
            auto prefix = full_.upload(audioCondition.data(), audioConditionRows, 32);
            if (trace)
                trace("condition_audio", prefix);
            a = full_.concat({prefix, a});
        }
        auto vs = pika ? pikaSigmaSchedule(12) : sigmaSchedule(o.steps, 12);
        auto as = pika ? pikaSigmaSchedule(3) : sigmaSchedule(o.steps, 3);
        std::mt19937_64 renoiseGenerator(o.seed + 1);
        std::normal_distribution<float> renoiseNormal;
        if (vs.size() != as.size())
            throw std::runtime_error("H3 audio/video schedule lengths differ");
        std::vector<std::vector<float>> plans;
        std::vector<std::array<int, 4>> timeIndices;
        int offset = 0;
        for (size_t i = 0; i + 1 < vs.size(); ++i) {
            std::vector<float> times{1.f - vs[i], 1.f - as[i]};
            float conditionTime = std::max(1.f - vs[i], .999f);
            if (conditionRows)
                times.push_back(conditionTime);
            if (audioConditionRows)
                times.push_back(1.f);
            std::sort(times.begin(), times.end());
            times.erase(std::unique(times.begin(), times.end()), times.end());
            int vi = offset + std::lower_bound(times.begin(), times.end(), 1.f - vs[i]) - times.begin();
            int ai = offset + std::lower_bound(times.begin(), times.end(), 1.f - as[i]) - times.begin();
            int ci = conditionRows ? offset + std::lower_bound(times.begin(), times.end(), conditionTime) -
                                         times.begin()
                                   : vi;
            int aci = audioConditionRows
                          ? offset + std::lower_bound(times.begin(), times.end(), 1.f) - times.begin()
                          : ai;
            timeIndices.push_back({vi, ai, ci, aci});
            offset += times.size();
            plans.push_back(std::move(times));
        }
        tick("prepare", 0, 50);
        auto prepared = [&](int current, int total) { tick("prepare", current, total); };
        if (parallel_)
            parallel_->prepare(plans, referenceMode, prepared);
        else
            denoiser->prepare(plans, prepared);
        tick("prepare", 50, 50);
        for (size_t i = 0; i + 1 < vs.size(); ++i) {
            auto [vi, ai, ci, aci] = timeIndices[i];
            for (size_t row = 0; row < layout.tags.size(); ++row)
                layout.timeIndices[row] = layout.tags[row] == 2 ? ai : vi;
            for (int row = 0; row < conditionRows; ++row)
                layout.timeIndices[layout.video[row]] = ci;
            for (int row = 0; row < audioConditionRows; ++row)
                layout.timeIndices[layout.audio[row]] = aci;
            tick("denoise", i, vs.size() - 1);
            auto denoiseProgress = [&](const std::string &, const Tensor &) {
                tick("denoise", i, vs.size() - 1);
            };
            auto velocity = parallel_ ? parallel_->forward(v, a, text, layout, denoiseProgress)
                                      : denoiser->forward(v, a, text, layout, denoiseProgress);
            auto advance = [&](const Tensor &sample, const Tensor &velocity, float sigma, float next,
                               const std::string &modality) {
                if (!pika) return full_.step(sample, velocity, 1.f - sigma, sigma, next);
                Tensor noise;
                if (next != 0) {
                    std::vector<float> values(sample.size());
                    for (auto &value : values) value = renoiseNormal(renoiseGenerator);
                    noise = full_.upload(values.data(), sample.rows, sample.cols);
                    if (trace) trace("renoise_" + modality + "." + std::to_string(i), noise);
                }
                return full_.renoise(sample, velocity, noise, sigma, next);
            };
            auto generated = advance(v.rowSlice(conditionRows, v.rows - conditionRows),
                                        velocity.first.rowSlice(conditionRows, v.rows - conditionRows),
                                        vs[i], vs[i + 1], "video");
            v = conditionRows ? full_.concat({v.rowSlice(0, conditionRows), generated}) : generated;
            auto generatedAudio =
                advance(a.rowSlice(audioConditionRows, a.rows - audioConditionRows),
                           velocity.second.rowSlice(audioConditionRows, a.rows - audioConditionRows),
                           as[i], as[i + 1], "audio");
            a = audioConditionRows ? full_.concat({a.rowSlice(0, audioConditionRows), generatedAudio})
                                   : generatedAudio;
            if (trace) {
                trace("video_step." + std::to_string(i), v);
                trace("audio_step." + std::to_string(i), a);
            }
        }
        video = full_.download(v.rowSlice(conditionRows, v.rows - conditionRows));
        audio = full_.download(a.rowSlice(audioConditionRows, a.rows - audioConditionRows));
        v = {};
        a = {};
        text = {};
        if (!retainWeights_) {
            if (parallel_) parallel_->unload();
            else denoiser->unload();
        }
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
        std::function<void(std::vector<uint8_t>)> onFrames;
        if (writer)
            onFrames = [&](std::vector<uint8_t> rgb) {
                writer->write(std::move(rgb),
                              [&] { tick("video_decode", videoCurrent, videoTotal); });
            };
        result.rgb = video_.decodeRGB(
            latents,
            [&](int current, int total) {
                videoCurrent = current;
                videoTotal = total;
                tick("video_decode", current, total);
            },
            std::move(onFrames), retainRGB);
        if (!retainWeights_) {
            if (parallel_) parallel_->unloadVideo();
            else video_.unload();
        }
        tick("video_decode", 1, 1);
        if (retainRGB && result.rgb.size() != size_t(o.frames) * o.width * o.height * 3)
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
