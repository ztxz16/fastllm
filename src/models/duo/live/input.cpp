#include "input.h"
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern "C" {
#include "qwen_asr.h"
}
#include <fstream>
#include <sstream>
#include <cmath>
extern char **environ;
namespace fs = std::filesystem;
namespace fastllm::duo::live {
std::string readFile(const fs::path &p) {
    std::ifstream f(p, std::ios::binary);
    if (!f)
        throw std::runtime_error("cannot open " + p.string());
    return std::string(std::istreambuf_iterator<char>(f), {});
}
void command(std::vector<std::string> args, const fs::path &output) {
    std::vector<char *> av;
    for (auto &s : args)
        av.push_back(s.data());
    av.push_back(nullptr);
    posix_spawn_file_actions_t act;
    int err = posix_spawn_file_actions_init(&act);
    if (err)
        throw std::runtime_error("spawn initialization failed");
    if (!output.empty())
        err = posix_spawn_file_actions_addopen(&act, STDOUT_FILENO, output.c_str(),
                                               O_WRONLY | O_CREAT | O_TRUNC, 0600);
    pid_t child;
    if (!err)
        err = posix_spawnp(&child, av[0], &act, nullptr, av.data(), environ);
    posix_spawn_file_actions_destroy(&act);
    if (err)
        throw std::runtime_error(args[0] + ": " + std::strerror(err));
    int status = 0;
    pid_t done;
    do {
        done = waitpid(child, &status, 0);
    } while (done < 0 && errno == EINTR);
    if (done < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
        throw InputError(args[0] + " failed");
}
namespace {
uint32_t codepoint(const std::string &s, size_t &i) {
    auto c = uint8_t(s.at(i++));
    if (c < 128)
        return c;
    int n = (c & 0xe0) == 0xc0 ? 1 : (c & 0xf0) == 0xe0 ? 2 : 3;
    uint32_t v = c & ((1u << (6 - n)) - 1);
    for (int k = 0; k < n; ++k)
        v = (v << 6) | (uint8_t(s.at(i++)) & 63);
    return v;
}
struct Range {
    uint32_t first, last;
    unsigned char mask;
};
constexpr Range categories[] = {
#include "../../../unicode-categories.inc"
};
struct Word {
    std::string value;
    size_t begin, end;
};
std::vector<Word> words(const std::string &text) {
    std::vector<Word> out;
    bool ongoing = false;
    for (size_t i = 0; i < text.size();) {
        auto start = i;
        auto cp = codepoint(text, i);
        auto r = std::lower_bound(std::begin(categories), std::end(categories), cp,
                                  [](auto &range, uint32_t c) { return range.last < c; });
        bool kept = (r != std::end(categories) && r->first <= cp && (r->mask & 5)) || cp == 39;
        if (!kept) {
            ongoing = false;
            continue;
        }
        bool cjk = (cp >= 0x3400 && cp <= 0x9fff) || (cp >= 0xf900 && cp <= 0xfaff) ||
                   (cp >= 0x20000 && cp <= 0x2ceaf);
        if (cjk || !ongoing)
            out.push_back({text.substr(start, i - start), start, i});
        else {
            out.back().value += text.substr(start, i - start);
            out.back().end = i;
        }
        ongoing = !cjk;
    }
    return out;
}
json11::Json json(const std::string &s) {
    std::string err;
    auto j = json11::Json::parse(s, err);
    if (!err.empty())
        throw std::runtime_error(err);
    return j;
}
} // namespace
Tokenizer::Tokenizer(const std::string &dir) : bpe(dir) {
    auto j = json(readFile(fs::path(dir) / "tokenizer.json"));
    std::map<int, int> inverse;
    int next = 256;
    for (int i = 0; i < 256; ++i)
        inverse[(i >= 33 && i <= 126) || (i >= 161 && i <= 172) || i >= 174 ? i : next++] = i;
    auto store = [&](int id, std::string value) {
        if (id < 0 || id > 200000)
            throw std::runtime_error("invalid vocabulary id");
        if (size_t(id) >= pieces.size())
            pieces.resize(id + 1);
        pieces[id] = std::move(value);
    };
    for (auto &[spelling, id] : j["model"]["vocab"].object_items()) {
        std::string raw;
        for (size_t i = 0; i < spelling.size();)
            raw += char(inverse.at(codepoint(spelling, i)));
        store(id.int_value(), raw);
    }
    for (auto &s : j["added_tokens"].array_items())
        store(s["id"].int_value(), s["content"].string_value());
}
std::string Tokenizer::text(const std::vector<int> &ids) const {
    std::string s;
    for (int id : ids)
        s += pieces.at(id);
    return s;
}
struct ResidentSpeech {
    int device;
    qwen_ctx_t *ctx;
    ResidentSpeech(const std::string &path, int gpu) : device(gpu) {
        native::DeviceGuard guard(device);
        ctx = qwen_load(path.c_str());
        if (!ctx)
            throw std::runtime_error("cannot load Qwen3-ASR recognizer");
    }
    ~ResidentSpeech() {
        native::DeviceGuard guard(device);
        qwen_free(ctx);
    }
};
InputWorker::InputWorker(std::string model, int speech, int vision)
    : root(std::move(model)), speechDevice(speech), visionDevice(vision) {}
InputWorker::~InputWorker() = default;
std::string InputWorker::recognize(const std::vector<float> &pcm) {
    native::DeviceGuard dev(speechDevice);
    if (!recognizer)
        recognizer = std::make_unique<ResidentSpeech>(root + "/speech/recognizer", speechDevice);
    std::unique_ptr<char, decltype(&std::free)> text(
        qwen_transcribe_audio(recognizer->ctx, pcm.data(), pcm.size()), std::free);
    if (!text)
        throw InputError("语音识别失败");
    std::string result = text.get();
    while (!result.empty() && std::isspace(uint8_t(result.back())))
        result.pop_back();
    return result;
}
Input InputWorker::prepare(const fs::path &file, const fs::path &dir, const Tokenizer &tokens,
                           const std::function<void(const std::string &)> &transcribed) {

    command({"ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height",
             "-of", "json", file},
            dir / "probe.json");
    auto info = json(readFile(dir / "probe.json"));
    double duration = std::stod(info["format"]["duration"].string_value());
    int width = 0, height = 0;
    for (auto &s : info["streams"].array_items())
        if (s["codec_type"].string_value() == "video" && !width) {
            width = s["width"].int_value();
            height = s["height"].int_value();
        }
    if (!std::isfinite(duration) || duration <= 0 || duration > 30 || width <= 0 || height <= 0)
        throw InputError("需要 30 秒以内的音视频");
    command({"ffmpeg", "-nostdin", "-v", "error", "-i", file, "-vn", "-ac", "1", "-ar", "16000", "-f",
             "f32le", "pipe:1"},
            dir / "speech.f32");
    auto raw = readFile(dir / "speech.f32");
    if (raw.empty() || raw.size() % 4 || raw.size() > 31 * 16000 * 4)
        throw InputError("没有可用音轨");
    std::vector<float> pcm(raw.size() / 4);
    std::memcpy(pcm.data(), raw.data(), raw.size());
    Input result;
    const std::string transcript = recognize(pcm);
    auto spans = words(transcript);
    if (spans.empty())
        throw InputError("没有识别到语音");
    for (const char *s : {"<|", "<wait>", "<speak>", "<blank>", "<chunk_"})
        if (transcript.find(s) != std::string::npos)
            throw InputError("recognized reserved protocol text");
    transcribed(transcript);
    std::vector<std::vector<int>> ids;
    for (auto &s : spans)
        ids.push_back(tokens.bpe.encode(s.value));
    if (!aligner)
        aligner = std::make_unique<Aligner>(root, speechDevice);
    auto stamps = aligner->align(pcm, ids);
    int start = std::floor(stamps.front() * .024),
        stop = std::min<int>(std::ceil(duration * 24), std::ceil(stamps.back() * .024));
    if (start >= std::ceil(duration * 24))
        throw std::invalid_argument("speech timestamp beyond video");
    int count = (std::max(stop - start, 1) + 16) / 17;
    result.bodies.resize(count);
    auto body = tokens.bpe.encode(transcript);
    size_t byte = 0;
    double prev = stamps.front(), end = prev, gateCommit = 0;
    bool first = true;
    for (int id : body) {
        auto hi = byte + tokens.pieces.at(id).size();
        int loTime = INT_MAX, hiTime = -1;
        for (size_t j = 0; j < spans.size(); ++j)
            if (spans[j].begin < hi && spans[j].end > byte) {
                loTime = std::min(loTime, stamps[j * 2]);
                hiTime = std::max(hiTime, stamps[j * 2 + 1]);
            }
        if (hiTime < 0) {
            loTime = prev;
            hiTime = loTime + 1;
        }
        hiTime = std::max(loTime + 1, hiTime);
        prev = hiTime;
        byte = hi;
        double commit = hiTime - 1 - start * 1000. / 24;
        int block = 0;
        while (block < count && commit >= std::nearbyint((block + 1) * 17000. / 24))
            ++block;
        if (block == count || commit < 0)
            throw std::invalid_argument("speech token outside video");
        result.bodies[block].push_back(id);
        result.lastSpeaking = block;
        if (loTime - end > 17000. / 24)
            first = false;
        if (first) {
            gateCommit = std::max(gateCommit, commit);
            end = std::max<double>(end, hiTime);
        }
    }
    result.release = int(std::floor(gateCommit * 24 / 17000)) + 1;
    auto filter = "trim=start=" + std::to_string(start / 24.) +
                  ":end=" + std::to_string(std::max(start + 1, stop) / 24.) +
                  ",setpts=PTS-STARTPTS,fps=24,scale=320:240:force_original_aspect_ratio=decrease:force_"
                  "divisible_by=2,tpad=stop_mode=clone:stop_duration=1";
    command({"ffmpeg", "-nostdin", "-v", "error", "-y", "-i", file, "-vf", filter, "-an", "-frames:v",
             std::to_string(count * 17), "-c:v", "libx264", "-pix_fmt", "yuv420p", dir / "user.mp4"});
    command(
        {"ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "json", dir / "user.mp4"},
        dir / "video.json");
    auto geom = json(readFile(dir / "video.json"))["streams"][0];
    width = geom["width"].int_value();
    height = geom["height"].int_value();
    command({"ffmpeg", "-nostdin", "-v", "error", "-i", dir / "user.mp4", "-pix_fmt", "rgb24", "-f",
             "rawvideo", "pipe:1"},
            dir / "user.rgb");
    auto pixels = readFile(dir / "user.rgb");
    fs::remove(dir / "user.rgb");
    if (pixels.size() != size_t(count) * 17 * width * height * 3)
        throw std::runtime_error("incomplete normalized video");
    const size_t blockBytes = size_t(17) * width * height * 3;
    for (int k = 0; k < count; ++k) {
        const auto start = pixels.data() + k * blockBytes;
        result.views.push_back(vision({start, start + blockBytes}, width, height));
    }
    return result;
}
namespace {
// Separable antialiased Keys cubic filter with half-pixel centers and uint8
// rounding after each pass, as used by the released vision preprocessing.
struct Tap {
    int index;
    double weight;
};
std::vector<std::vector<Tap>> cubicTaps(int input, int output) {
    std::vector<std::vector<Tap>> taps(output);
    const double scale = double(input) / output, support = std::max(1., scale);
    for (int dst = 0; dst < output; ++dst) {
        const double center = (dst + .5) * scale - .5;
        double sum = 0;
        for (int src = std::max(0, int(std::ceil(center - 2 * support)));
             src <= std::min(input - 1, int(std::floor(center + 2 * support))); ++src) {
            double a = std::abs((src - center) / support);
            double w = a < 1 ? ((1.5 * a - 2.5) * a) * a + 1 : ((-.5 * a + 2.5) * a - 4) * a + 2;
            taps[dst].push_back({src, w});
            sum += w;
        }
        for (auto &tap : taps[dst])
            tap.weight /= sum;
    }
    return taps;
}
std::vector<uint8_t> resizeRGB(const std::vector<uint8_t> &src, int frames, int width, int height, int w,
                               int h) {
    auto xs = cubicTaps(width, w), ys = cubicTaps(height, h);
    std::vector<uint8_t> dst(size_t(frames) * w * h * 3), tmp(size_t(height) * w * 3);
    auto pixel = [](double v) { return uint8_t(std::clamp(int(std::floor(v + .5)), 0, 255)); };
    for (int f = 0; f < frames; ++f) {
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < w; ++x)
                for (int c = 0; c < 3; ++c) {
                    double sum = 0;
                    for (auto tap : xs[x])
                        sum += tap.weight * src[((size_t(f) * height + y) * width + tap.index) * 3 + c];
                    tmp[(y * w + x) * 3 + c] = pixel(sum);
                }
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                for (int c = 0; c < 3; ++c) {
                    double sum = 0;
                    for (auto tap : ys[y])
                        sum += tap.weight * tmp[(tap.index * w + x) * 3 + c];
                    dst[((size_t(f) * h + y) * w + x) * 3 + c] = pixel(sum);
                }
    }
    return dst;
}
} // namespace
Visual InputWorker::vision(const std::vector<uint8_t> &pixels, int width, int height) {
    if (width <= 0 || height <= 0 || width > 4096 || height > 4096 ||
        pixels.size() != size_t(17) * width * height * 3)
        throw InputError("invalid 17-frame video block");
    double scale = std::sqrt(18. * width * height / 16384.);
    int w = std::max(32, int(std::floor(width / scale / 32)) * 32),
        h = std::max(32, int(std::floor(height / scale / 32)) * 32);
    auto resized = resizeRGB(pixels, 17, width, height, w, h);
    native::DeviceGuard dev(visionDevice);
    if (!visionRuntime) {
        visionRuntime = std::make_unique<nomas::Runtime>(visionDevice, nomas::DType::BF16);
        visionWeights = std::make_unique<nomas::Weights>(*visionRuntime, root + "/intent");
    }
    auto rgba = [&](int f) {
        auto ptr = resized.data() + size_t(f) * w * h * 3;
        std::vector<uint8_t> a(w * h * 4, 255);
        for (int i = 0; i < w * h; ++i)
            std::memcpy(a.data() + 4 * i, ptr + 3 * i, 3);
        return a;
    };
    Visual v{h / 32, w / 32, {}};
    for (int j = 0; j < 17; j += 2) {
        auto a = rgba(j), b = rgba(std::min(16, j + 1));
        auto out = nomas::qwenVision(*visionRuntime, *visionWeights, w, h, a, {}, &b);
        std::array<native::Tensor, 4> embeds;
        for (int i = 0; i < 4; ++i) {
            auto host = visionRuntime->download(out[i]);
            embeds[i] = native::from_blob(host.data(), {out[i].rows, out[i].cols}, native::kFloat)
                            .to(visionDevice, native::kBFloat16);
        }
        v.pairs.push_back(std::move(embeds));
    }
    return v;
}
} // namespace fastllm::duo::live
