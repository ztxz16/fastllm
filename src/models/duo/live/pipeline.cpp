#include "pipeline.h"
#include "reference.h"
#include "tables.h"
#include "avc.h"
#include <c10/cuda/CUDAStream.h>
extern "C" void duo_rgb_to_i420(const uint8_t *, uint8_t *, int, int, int, cudaStream_t);
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <condition_variable>
#include <deque>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <fstream>
namespace fs = std::filesystem;
namespace fastllm::duo::live {
namespace {
template <class T> class Channel {
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<T> data;
    bool closed = false;

  public:
    void close() {
        {
            std::lock_guard lock(mutex);
            closed = true;
        }
        changed.notify_all();
    }
    void put(T value) {
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] { return closed || data.size() < 2; });
        if (closed)
            throw std::runtime_error("pipeline stopped");
        data.push_back(std::move(value));
        changed.notify_all();
    }
    bool take(T &value) {
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] { return closed || !data.empty(); });
        if (data.empty())
            return false;
        value = std::move(data.front());
        data.pop_front();
        changed.notify_all();
        return true;
    }
};
std::vector<float> floats(at::Tensor tensor) {
    tensor = tensor.to(at::kCPU, at::kFloat).contiguous();
    auto ptr = tensor.data_ptr<float>();
    return {ptr, ptr + tensor.numel()};
}
struct Body {
    int index;
    std::vector<float> hidden;
    std::shared_future<void> intentCommitted;
};
} // namespace
GpuLayout GpuLayout::fromProfile(const std::string &profile) {
    if (profile == "single")
        return {0, 0, 0, {}};
    if (profile == "two")
        return {0, 1, 1, {}};
    if (profile == "four")
        return {1, 3, 2, {3}};
    if (profile == "eight")
        return {1, 7, 2, {3, 4, 5, 6, 7}, 3};
    throw std::invalid_argument("expected Duo GPU profile single, two, four or eight");
}

Pipeline::Pipeline(std::string r, fs::path a, int width, int height, GpuLayout layout, bool streamAvc,
                   fs::path cache)
    : root(std::move(r)), assets(std::move(a)), intentCache(std::move(cache)), w(width), h(height),
      devices(std::move(layout)), streamAvc(streamAvc), tokens(root + "/intent_adapter"),
      inputs(root, devices.speech, devices.intent) {}
void Pipeline::check() const {
    if (cancelled.load())
        throw std::runtime_error("session cancelled");
}
std::vector<float> Pipeline::prefix() {
    std::string desc = readFile(assets / "presets/default.txt");
    while (!desc.empty() && desc.back() == '\n')
        desc.pop_back();
    // Conversation tokens only append after these immutable system KV rows.
    if (prefixLength && desc == prefixDescription && intent->length() >= prefixLength) {
        intent->rewind(prefixLength, prefixPosition);
        return prefixFeatures;
    }
    prefixLength = 0;
    intent->reset();
    std::string intro =
        "<|im_start|>system\n<chunk_708> You are the assistant in a real-time full-reciprocal video call.\n"
        "The assistant's visual appearance and scene are described as follows:\n\"";
    std::string instruction =
        "\"\nBegin every speaking chunk with <speak>. Output <wait> if the assistant is "
        "waiting or continuing an unfinished utterance.\n";
    std::string cue = "In this chunk, the assistant should say:";
    auto ids = tokens.bpe.encode(intro + desc + instruction + cue + "<|im_end|>\n");
    std::vector<int> keep;
    size_t start = 0, cueStart = intro.size() + desc.size() + instruction.size();
    for (int row = 0; row < int(ids.size()); ++row) {
        auto stop = start + tokens.pieces.at(ids[row]).size();
        if ((start < intro.size() + desc.size() && stop > intro.size()) ||
            (start < cueStart + cue.size() && stop > cueStart))
            keep.push_back(row);
        start = stop;
    }
    auto features = intent->forward(ids, nullptr, false);
    prefixFeatures =
        floats(features.index_select(0, at::tensor(keep, at::kInt).to(features.device(), at::kLong)));
    prefixDescription = std::move(desc);
    prefixPosition = intent->position();
    prefixLength = intent->length();
    return prefixFeatures;
}
Pipeline::Plan Pipeline::plan(const Visual &visual, const std::vector<int> &body, bool hold, int index,
                              const Intent::HiddenReady &waitReady) {
    std::ostringstream prompt;
    if (!index)
        prompt << "<|im_start|>assistant\n<wait><|im_end|>\n";
    else
        prompt << "<|im_end|>\n";
    prompt << "<|im_start|>user\n";
    for (int j = 0; j < int(visual.pairs.size()); ++j) {
        double seconds = std::nearbyint((17 * index + std::min(2 * j + 1, 16)) * 1000. / 24) / 1000.;
        prompt << "<" << std::fixed << std::setprecision(1) << seconds << " seconds><|vision_start|>";
        for (int k = 0; k < visual.rows * visual.cols; ++k)
            prompt << "<|video_pad|>";
        prompt << "<|vision_end|>";
    }
    auto ids = tokens.bpe.encode(prompt.str());
    if (!body.empty())
        ids.insert(ids.end(), body.begin(), body.end());
    else
        ids.push_back(151677);
    auto suffix = tokens.bpe.encode("<|im_end|>\n<|im_start|>assistant\n");
    ids.insert(ids.end(), suffix.begin(), suffix.end());
    if (hold && (streamAvc || waitReady)) {
        // The endpoint already committed this action. Prefill its causal row together
        // with the prompt and extract the same layer used by the speech planner.
        ids.push_back(151676);
        auto all = intent->forward(ids, &visual, false, waitReady);
        if (waitReady)
            return {{}, {151676}};
        return {floats(all.narrow(0, all.size(0) - 1, 1)), {151676}};
    }
    intent->forward(ids, &visual, !hold);
    std::vector<at::Tensor> hidden;
    std::vector<int> committed;
    int token = hold ? 151676 : intent->nextToken();
    if (token != 151680)
        token = 151676;
    for (int j = 0; j < 64; ++j) {
        check();
        committed.push_back(token);
        hidden.push_back(intent->forward({token}, nullptr, committed.front() == 151680));
        if (committed.front() == 151676)
            break;
        token = intent->nextToken();
        if (token == 151645)
            break;
    }
    return {floats(at::cat(hidden)), committed};
}
void Pipeline::turn(const fs::path &file, const fs::path &dir, const Emit &emit) {
    c10::InferenceMode inference;
    check();
    emit({{"type", "stage"}, {"name", "识别语音"}});
    auto input =
        inputs.prepare(file, dir, tokens, [&](auto &s) { emit({{"type", "transcript"}, {"text", s}}); });
    int waits = 0;
    run(dir, emit, false, [&](int local, int index, Plan &out, const Intent::HiddenReady &) {
        const auto count = int(input.bodies.size());
        if (local >= count + 12 || waits == 4)
            return false;
        int source = local < count ? local : input.lastSpeaking;
        const std::vector<int> empty;
        out = plan(input.views[source], local < count ? input.bodies[local] : empty, local < input.release,
                   index);
        waits = local >= count && out.second.front() != 151680 ? waits + 1 : 0;
        return true;
    });
}

void Pipeline::stream(const NextBlock &next, const fs::path &dir, const Emit &emit) {
    std::vector<float> pending;
    size_t trailingQuiet = 0;
    bool ended = false;
    run(dir, emit, true, [&](int local, int index, Plan &out, const Intent::HiddenReady &waitReady) {
        if (ended)
            return false;
        AVBlock block;
        if (!next(block))
            throw InputError("input stream ended without final block");
        if (block.pcm.size() < 11333 || block.pcm.size() > 11334)
            throw InputError("expected 17 frames of 16 kHz mono audio");
        block.decodeJpeg();
        ended = block.end;
        size_t lastVoiced = 0;
        for (size_t pos = 0; pos < block.pcm.size(); pos += 320) {
            size_t stop = std::min(pos + 320, block.pcm.size());
            double power = 0;
            for (size_t j = pos; j < stop; ++j)
                power += double(block.pcm[j]) * block.pcm[j];
            if (power / (stop - pos) > (32768.0 * .006) * (32768.0 * .006))
                lastVoiced = stop;
        }
        std::string text;
        bool hold = false;
        if (lastVoiced || !pending.empty()) {
            for (int16_t sample : block.pcm)
                pending.push_back(sample / 32768.0f);
            trailingQuiet = lastVoiced ? block.pcm.size() - lastVoiced : trailingQuiet + block.pcm.size();
            // Release the floor after 480 ms of silence or at EOS. All input up to here is committed.
            hold = trailingQuiet < 7680 && !ended;
            if (!hold || pending.size() >= 30 * 16000) {
                text = inputs.recognize(pending);
                pending.clear();
                trailingQuiet = 0;
            }
        }
        emit({{"type", "transcript"}, {"index", local}, {"text", text}});
        auto visual = inputs.vision(block.rgb, block.width, block.height);
        out = plan(visual, tokens.bpe.encode(text), hold, index, waitReady);
        return true;
    });
}

void Pipeline::run(const fs::path &dir, const Emit &emit, bool fresh,
                   const std::function<bool(int, int, Plan &, const Intent::HiddenReady &)> &nextPlan) {
    if (fresh)
        global = 0;
    check();
    const bool resetReal = !real || fresh;
    if (!real && devices.ffnPeer >= 0) {
        c10::cuda::CUDAGuard guard(at::Device(at::kCUDA, 0));
        auto session = std::make_unique<Session>(0, ModelSource(root), h / 16, w / 16);
        if (session->enableFfnPeer(devices.ffnPeer)) {
            std::erase(devices.tilePeers, devices.ffnPeer);
            std::clog << "Duo Realization FFN on CUDA 0 and " << devices.ffnPeer << '\n';
        }
        real = std::move(session);
    }
    Channel<Body> bodies;
    struct Packet {
        ChunkResult media;
        std::shared_future<void> committed;
    };
    Channel<Packet> packets;
    std::mutex errors;
    std::exception_ptr error;
    auto worker = [&](auto fn) {
        return std::async(std::launch::async, [&, fn] {
            c10::InferenceMode mode;
            try {
                fn();
            } catch (...) {
                {
                    std::lock_guard lock(errors);
                    if (!error)
                        error = std::current_exception();
                }
                cancelled.store(true);
                bodies.close();
                packets.close();
            }
        });
    };
    std::promise<std::vector<float>> shared;
    auto sharedFuture = shared.get_future();
    int first = global, produced = 0;
    std::atomic<int> voiceStart{std::numeric_limits<int>::max()};
    auto planner = worker([&] {
        try {
            if (!intent || fresh) {
                if (!intent)
                    intent = std::make_unique<Intent>(root, devices.intent, devices.intent != 0, intentCache);
                shared.set_value(prefix());
            } else
                shared.set_value({});
        } catch (...) {
            shared.set_exception(std::current_exception());
            throw;
        }
        for (int local = 0;; ++local) {
            check();
            int index = first + local;
            Plan pending;
            std::promise<void> intentCommit;
            bool released = false;
            Intent::HiddenReady waitReady;
            if (fresh && devices.intent != 0) {
                waitReady = [&](const at::Tensor &h50) {
                    auto features = floats(h50.narrow(0, h50.size(0) - 1, 1));
                    bodies.put({index, std::move(features), intentCommit.get_future().share()});
                    released = true;
                };
            }
            // A broken promise propagates any incomplete Intent KV update to the
            // downstream worker before it may publish the speculative media.
            if (!nextPlan(local, index, pending, waitReady))
                break;
            if (released) {
                // KV writes from the remaining layers are asynchronous. Resolve
                // the publication fence only after GPU completion on their stream.
                c10::cuda::getCurrentCUDAStream(devices.intent).synchronize();
                intentCommit.set_value();
            }
            auto &[hidden, ids] = pending;
            bool speak = ids.front() == 151680;
            if (speak && voiceStart.load() == std::numeric_limits<int>::max())
                voiceStart.store(index);
            emit({{"type", "plan"},
                  {"index", index},
                  {"action", speak ? "speak" : "wait"},
                  {"text", tokens.text(ids)}});
            if (!released)
                bodies.put({index, std::move(hidden), {}});
            ++produced;
        }
        bodies.close();
    });
    auto realization = worker([&] {
        c10::cuda::CUDAGuard dev(at::Device(at::kCUDA, 0));
        if (resetReal) {
            if (!real)
                real = std::make_unique<Session>(0, ModelSource(root), h / 16, w / 16);
            auto ref = readFile(assets / referencePreset(w, h));
            if (ref.size() != size_t(h / 16) * (w / 16) * 24 * 2)
                throw std::runtime_error("wrong preset geometry");
            auto values = floats(at::from_blob(ref.data(), {int64_t(ref.size() / 2)}, at::kBFloat16));
            auto states = sharedFuture.get();
            check();
            real->reset(values, states, states.size() / 5120, fresh);
        }
        Body b;
        while (bodies.take(b)) {
            check();
            if (devices.codec != 0) {
                std::promise<void> committed;
                auto ready = committed.get_future().share();
                try {
                    real->generate(b.index, b.hidden, b.hidden.size() / 5120, 42 + b.index, {},
                                   [&](const ChunkResult &media) {
                                       if (b.intentCommitted.valid())
                                           b.intentCommitted.get();
                                       packets.put({media, ready});
                                   });
                    committed.set_value();
                } catch (...) {
                    committed.set_exception(std::current_exception());
                    throw;
                }
            } else {
                auto media = real->generate(b.index, b.hidden, b.hidden.size() / 5120, 42 + b.index);
                packets.put({std::move(media), {}});
            }
        }
        packets.close();
    });
    auto codec = worker([&] {
        c10::cuda::CUDAGuard dev(at::Device(at::kCUDA, devices.codec));
        if (!solarx) {
            solarx = std::make_unique<SolarX>(root + "/codec/video_vae/model.safetensors", devices.codec,
                                              devices.tilePeers);
            audioRuntime = std::make_unique<minimax_h3::Runtime>(devices.codec, nomas::DType::FP32);
            audio = std::make_unique<minimax_h3::AudioVAE>(*audioRuntime, root + "/codec/audio_vae");
        }
        solarx->reset();
        std::unique_ptr<AvcEncoder> avc;
        if (streamAvc)
            avc = std::make_unique<AvcEncoder>(w, h);
        std::ofstream rgbFile(dir / "frames.rgb", std::ios::binary),
            pcmFile(dir / "audio.f32", std::ios::binary);
        Packet packet;
        int count = 0;
        while (packets.take(packet)) {
            auto &r = packet.media;
            check();
            if (r.chunk != first + count)
                throw std::logic_error("out-of-order Codec packet");
            int lh = h / 16, lw = w / 16, samples = r.endSample - r.firstSample;
            auto z = at::from_blob(r.video.data(), {5, lh / 2, lw / 2, 24, 2, 2}, at::kFloat)
                         .permute({3, 0, 1, 4, 2, 5})
                         .reshape({24, 5, lh, lw});
            auto deviceRgb = solarx->decode(z);
            auto rgb = deviceRgb.cpu();
            std::vector<float> stereo(2 * samples);
            // AudioVAE decodes each channel independently and has no temporal state.
            // Before a stream opens its voice, the published PCM is exactly zero.
            const bool audible = !fresh || r.chunk >= voiceStart.load();
            for (int ch = 0; audible && ch < 2; ++ch) {
                std::vector<float> latent(29 * 32);
                for (int t = 0; t < 29; ++t)
                    for (int c = 0; c < 32; ++c)
                        latent[t * 32 + c] = float(at::BFloat16(r.audio[(ch * 29 + t) * 32 + c])) *
                                                 constants::audio_latents_std[c] +
                                             constants::audio_latents_mean[c];
                auto wave =
                    audioRuntime->download(audio->decode(audioRuntime->upload(latent.data(), 29, 32)));
                if (wave.size() < size_t(samples))
                    throw std::runtime_error("short decoded PCM");
                for (int i = 0; i < samples; ++i) {
                    if (!std::isfinite(wave[i]))
                        throw std::runtime_error("nonfinite audio output");
                    stereo[2 * i + ch] = wave[i];
                }
            }
            auto name = "chunk-" + std::to_string(count) + (avc ? ".dav" : ".mp4");
            minimax_h3::Generation g;
            g.width = w;
            g.height = h;
            g.frames = 17;
            g.audio = std::move(stereo);
            rgbFile.write(static_cast<const char *>(rgb.data_ptr()), rgb.nbytes());
            pcmFile.write(reinterpret_cast<const char *>(g.audio.data()), g.audio.size() * sizeof(float));
            if (!rgbFile || !pcmFile)
                throw std::runtime_error("media storage full");
            if (avc) {
                auto yuv = at::empty({17, h * w * 3 / 2}, deviceRgb.options());
                duo_rgb_to_i420(deviceRgb.data_ptr<uint8_t>(), yuv.data_ptr<uint8_t>(), w, h, 17,
                                c10::cuda::getCurrentCUDAStream(devices.codec));
                C10_CUDA_KERNEL_LAUNCH_CHECK();
                auto host = yuv.cpu();
                avc->write(dir / name, count, host.data_ptr<uint8_t>(), g.audio);
            } else {
                minimax_h3::MP4Writer mp4(g, (dir / name).string(), "ffmpeg");
                mp4.write(rgb.data_ptr<uint8_t>(), rgb.nbytes(), [&] { check(); });
                mp4.finish([&] { check(); });
            }
            if (packet.committed.valid())
                packet.committed.get();
            emit({{"type", "chunk"},
                  {"file", name},
                  {"index", count++},
                  {"frames", 17},
                  {"samples", samples}});
        }
        rgbFile.close();
        pcmFile.close();
        check();
        {
            std::lock_guard lock(errors);
            if (error)
                return;
        }
        if (!count)
            throw std::runtime_error("no Codec packets");
        command({"ffmpeg",
                 "-nostdin",
                 "-v",
                 "error",
                 "-y",
                 "-f",
                 "rawvideo",
                 "-pix_fmt",
                 "rgb24",
                 "-s",
                 std::to_string(w) + "x" + std::to_string(h),
                 "-r",
                 "24",
                 "-i",
                 dir / "frames.rgb",
                 "-f",
                 "f32le",
                 "-ar",
                 "32000",
                 "-ac",
                 "2",
                 "-i",
                 dir / "audio.f32",
                 "-threads",
                 "4",
                 "-c:v",
                 "libx264",
                 "-preset",
                 "fast",
                 "-crf",
                 "18",
                 "-pix_fmt",
                 "yuv420p",
                 "-c:a",
                 "aac",
                 "-b:a",
                 "192k",
                 "-movflags",
                 "+faststart",
                 dir / "reply.mp4"});
    });
    planner.get();
    realization.get();
    codec.get();
    fs::remove(dir / "frames.rgb");
    fs::remove(dir / "audio.f32");
    if (error)
        std::rethrow_exception(error);
    global += produced;
    emit({{"type", "done"}, {"file", "reply.mp4"}, {"chunks", produced}});
}
} // namespace fastllm::duo::live
