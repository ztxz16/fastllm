#include "models/minimax_h3/runtime.h"
#include "json11.hpp"
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
namespace fastllm {
namespace minimax_h3 {
namespace {
std::string capture(std::vector<std::string> args, size_t limit) {
    int pipefd[2];
    if (pipe2(pipefd, O_CLOEXEC))
        throw std::runtime_error("cannot create media pipe");
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, pipefd[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&actions, pipefd[0]);
    posix_spawn_file_actions_addclose(&actions, pipefd[1]);
    std::vector<char *> argv;
    for (auto &arg : args)
        argv.push_back(arg.data());
    argv.push_back(nullptr);
    pid_t child = -1;
    int error = posix_spawnp(&child, argv[0], &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    close(pipefd[1]);
    if (error) {
        close(pipefd[0]);
        throw std::runtime_error("cannot launch media decoder: " + std::string(std::strerror(error)));
    }
    try {
        std::string output;
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        for (;;) {
            if (std::chrono::steady_clock::now() > deadline)
                throw std::invalid_argument("media decode timed out");
            pollfd p{pipefd[0], POLLIN, 0};
            int ready = poll(&p, 1, 100);
            if (ready < 0 && errno != EINTR)
                throw std::runtime_error("media read failed");
            if (ready <= 0)
                continue;
            std::array<char, 65536> buffer;
            ssize_t n = read(pipefd[0], buffer.data(), buffer.size());
            if (n < 0 && errno == EINTR)
                continue;
            if (n < 0)
                throw std::runtime_error("media read failed");
            if (!n)
                break;
            if (output.size() + size_t(n) > limit)
                throw std::invalid_argument("decoded media exceeds limit");
            output.append(buffer.data(), n);
        }
        close(pipefd[0]);
        pipefd[0] = -1;
        int status;
        while (waitpid(child, &status, 0) < 0)
            if (errno != EINTR)
                throw std::runtime_error("media wait failed");
        child = -1;
        if (!WIFEXITED(status) || WEXITSTATUS(status))
            throw std::invalid_argument("invalid or unsupported media file");
        return output;
    } catch (...) {
        if (pipefd[0] >= 0)
            close(pipefd[0]);
        if (child > 0) {
            kill(child, SIGKILL);
            while (waitpid(child, nullptr, 0) < 0 && errno == EINTR) {
            }
        }
        throw;
    }
}
const char *formats = "mov,matroska,webm,wav,mp3,flac,ogg,aac,png_pipe,jpeg_pipe";
} // namespace
Reference readReference(const std::string &path, Reference::Kind kind, int maxFrames,
                        const std::string &ffmpeg, const std::string &ffprobe, int videoShortEdge) {
    if (!std::filesystem::is_regular_file(path) || std::filesystem::file_size(path) > (size_t(128) << 20) ||
        maxFrames < 1 || maxFrames > 345 || videoShortEdge < 64 || videoShortEdge > 768 ||
        videoShortEdge % 32)
        throw std::invalid_argument("media must be a local regular file up to 128 MiB");
    auto input = std::filesystem::absolute(path).string();
    Reference ref;
    ref.kind = kind;
    if (kind == Reference::Kind::Image) {
        std::ifstream file(input, std::ios::binary);
        std::string bytes((std::istreambuf_iterator<char>(file)), {});
        if (bytes.size() >= 8 && bytes.compare(0, 8, "\x89PNG\r\n\x1a\n", 8) == 0) {
            try {
                ref.frames.push_back(qwen_image::decodePNG(bytes));
            } catch (const std::runtime_error &e) {
                throw std::invalid_argument(e.what());
            }
            return ref;
        }
    }
    std::string error;
    auto info = json11::Json::parse(
        capture({ffprobe, "-v", "error", "-protocol_whitelist", "file,pipe", "-format_whitelist", formats,
                 "-show_streams", "-show_format", "-of", "json", input},
                1 << 20),
        error);
    if (!error.empty())
        throw std::invalid_argument("cannot read media metadata");
    json11::Json video, audio;
    for (const auto &stream : info["streams"].array_items()) {
        if (stream["codec_type"].string_value() == "video" && video.is_null())
            video = stream;
        if (stream["codec_type"].string_value() == "audio" && audio.is_null())
            audio = stream;
    }
    if (kind != Reference::Kind::Image) {
        std::string duration = info["format"]["duration"].string_value();
        double seconds = duration.empty() ? 0 : std::stod(duration);
        if (!std::isfinite(seconds) || seconds < 2 || seconds > 15.000001)
            throw std::invalid_argument(
                "reference media must last 2..15 seconds with valid duration metadata");
        ref.sourceDuration = seconds;
    }
    auto command = [&] {
        return std::vector<std::string>{ffmpeg,
                                        "-v",
                                        "error",
                                        "-nostdin",
                                        "-threads",
                                        "4",
                                        "-protocol_whitelist",
                                        "file,pipe",
                                        "-format_whitelist",
                                        formats,
                                        "-i",
                                        input};
    };
    if (kind != Reference::Kind::Audio) {
        int width = video["width"].int_value(), height = video["height"].int_value();
        if (width < 1 || height < 1 || int64_t(width) * height > 16777216 || width > 4 * height ||
            height > 4 * width)
            throw std::invalid_argument("invalid reference image/video dimensions");
        auto args = command();
        args.insert(args.end(), {"-map", "0:v:0", "-an"});
        if (kind == Reference::Kind::Video) {
            double scale = std::min(double(videoShortEdge) / std::min(width, height),
                                    std::sqrt(768. * 1344 / (double(width) * height)));
            width = std::max(32, int(std::nearbyint(width * scale / 32)) * 32);
            height = std::max(32, int(std::nearbyint(height * scale / 32)) * 32);
            args.insert(args.end(), {"-vf", "fps=24,scale=" + std::to_string(width) + ":" +
                                                std::to_string(height) + ":flags=lanczos"});
        }
        int count = kind == Reference::Kind::Image ? 1 : maxFrames;
        args.insert(args.end(), {"-frames:v", std::to_string(count), "-threads", "4", "-f", "rawvideo",
                                 "-pix_fmt", "rgba", "pipe:1"});
        size_t frameBytes = size_t(width) * height * 4;
        auto pixels = capture(std::move(args), frameBytes * count);
        if (pixels.empty() || pixels.size() % frameBytes)
            throw std::invalid_argument("incomplete decoded reference video");
        for (size_t i = 0; i < pixels.size(); i += frameBytes)
            ref.frames.push_back(
                {width, height, std::vector<uint8_t>(pixels.begin() + i, pixels.begin() + i + frameBytes)});
    }
    if (kind != Reference::Kind::Image && !audio.is_null()) {
        ref.channels = audio["channels"].int_value();
        ref.sampleRate = std::stoi(audio["sample_rate"].string_value());
        if ((ref.channels != 1 && ref.channels != 2) || ref.sampleRate < 8000 || ref.sampleRate > 192000)
            throw std::invalid_argument("reference audio must be mono or stereo at 8000..192000 Hz");
        auto args = command();
        args.insert(args.end(), {"-map", "0:a:0", "-vn", "-t", std::to_string(std::min(15., maxFrames / 24.)),
                                 "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"});
        auto pcm = capture(std::move(args), size_t(ref.channels) * ref.sampleRate * 15 * sizeof(float));
        if (pcm.empty() || pcm.size() % (ref.channels * sizeof(float)))
            throw std::invalid_argument("incomplete reference audio");
        ref.audio.resize(pcm.size() / sizeof(float));
        std::memcpy(ref.audio.data(), pcm.data(), pcm.size());
    }
    if (kind == Reference::Kind::Audio && ref.audio.empty())
        throw std::invalid_argument("reference has no audio stream");
    return ref;
}
} // namespace minimax_h3
} // namespace fastllm
