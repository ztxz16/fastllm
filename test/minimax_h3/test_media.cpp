// Streaming encoder lifecycle regression. Uses the existing FFmpeg executable; no GPU required.
#include "models/minimax_h3/runtime.h"
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
using namespace fastllm::minimax_h3;
namespace {
std::string read(const std::filesystem::path &p) {
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), {}};
}
int descriptors() {
    return std::distance(std::filesystem::directory_iterator("/proc/self/fd"),
                         std::filesystem::directory_iterator{});
}
void check(bool value, const char *message) {
    if (!value)
        throw std::runtime_error(message);
}
}
int main() {
    if (std::system("ffmpeg -version >/dev/null 2>&1"))
        return 77;
    char name[] = "/tmp/fastllm-h3-media-XXXXXX";
    char *made = mkdtemp(name);
    if (!made)
        return 1;
    struct Cleanup {
        std::filesystem::path dir;
        ~Cleanup() { std::error_code e; std::filesystem::remove_all(dir, e); }
    } cleanup{made};
    try {
        auto dir = cleanup.dir;
        Generation g;
        g.width = g.height = 64;
        g.frames = 42;
        const size_t frameBytes = size_t(g.width) * g.height * 3;
        g.rgb.resize(frameBytes * g.frames);
        for (size_t i = 0; i < g.rgb.size(); ++i)
            g.rgb[i] = uint8_t((i / frameBytes * 13 + i / 7) % 256);
        g.audio.resize(2 * 32000 * 2, 0.f);
        writeMP4(g, (dir / "serial.mp4").string());
        {
            MP4Writer writer(g, (dir / "stream.mp4").string());
            size_t first = 0;
            for (int frames : {1, 16, 17, 8}) {
                writer.write(g.rgb.data() + first, frames * frameBytes);
                first += frames * frameBytes;
            }
            writer.finish();
        }
        check(read(dir / "serial.mp4") == read(dir / "stream.mp4"), "chunking changed MP4 bytes");
        auto destination = (dir / "preserve.mp4").string();
        std::ofstream(destination) << "preserve";
        int initialFDs = descriptors();
        auto sigpipe = signal(SIGPIPE, SIG_DFL);
        signal(SIGPIPE, sigpipe);
        for (int trial = 0; trial < 8; ++trial) {
            bool failed = false;
            try { writeMP4(g, destination, trial % 2 ? "/bin/false" : "/missing/ffmpeg"); }
            catch (const std::runtime_error &) { failed = true; }
            check(failed, "failed encoder accepted");
            check(read(destination) == "preserve", "failure replaced prior output");
            check(descriptors() == initialFDs, "file descriptor leaked");
            int status;
            check(waitpid(-1, &status, WNOHANG) == -1 && errno == ECHILD, "child was not reaped");
        }
        check(signal(SIGPIPE, sigpipe) == sigpipe, "process SIGPIPE policy changed");
        // Cancellation must also unblock a writer whose child never reads stdin.
        auto blocked = (dir / "blocked-encoder").string();
        std::ofstream(blocked) << "#!/bin/sh\nexec sleep 30\n";
        chmod(blocked.c_str(), 0700);
        {
            MP4Writer writer(g, destination, blocked);
            writer.write(g.rgb.data(), g.rgb.size());
        }
        // Cancellation checkpoints must interrupt backpressure and child flushing.
        for (bool backpressure : {false, true}) {
            auto large = g;
            large.frames = backpressure ? 124 : 1;
            large.rgb.resize(frameBytes * large.frames, 128);
            bool cancelled = false;
            auto cancel = [] { throw std::runtime_error("cancelled"); };
            try {
                MP4Writer writer(large, destination, blocked);
                writer.write(large.rgb.data(), large.rgb.size(), cancel);
                writer.finish(cancel);
            } catch (const std::runtime_error &) { cancelled = true; }
            check(cancelled, "blocked encoder did not observe cancellation");
            int status;
            check(waitpid(-1, &status, WNOHANG) == -1 && errno == ECHILD, "cancelled child was not reaped");
        }
        check(read(destination) == "preserve", "cancelled writer published output");
        check(descriptors() == initialFDs, "cancelled writer leaked descriptor");
        bool incomplete = false;
        try {
            MP4Writer writer(g, destination);
            writer.write(g.rgb.data(), frameBytes);
            writer.finish();
        } catch (const std::runtime_error &) { incomplete = true; }
        check(incomplete, "incomplete stream accepted");
        check(read(destination) == "preserve", "incomplete stream replaced destination");
        auto reference = g;
        reference.frames = 48;
        reference.rgb.resize(frameBytes * 48, 96);
        reference.audio.resize(2 * 32000 * 2, 0.f);
        writeMP4(reference, (dir / "reference.mp4").string());
        auto decoded = readReference((dir / "reference.mp4").string(), Reference::Kind::Video, 124);
        check(decoded.frames.size() == 48 && decoded.frames[0].width == 768 &&
                  decoded.frames[0].height == 768,
              "reference video geometry/fps mismatch");
        check(decoded.sampleRate == 32000 && decoded.channels == 2 && !decoded.audio.empty(),
              "missing reference audio");
        auto audio = readReference((dir / "reference.mp4").string(), Reference::Kind::Audio, 124);
        check(audio.frames.empty() && audio.audio == decoded.audio, "audio stream decoding differs");
        bool invalid = false;
        try {
            readReference(destination, Reference::Kind::Video);
        } catch (const std::invalid_argument &) {
            invalid = true;
        }
        check(invalid && descriptors() == initialFDs, "invalid input accepted or decoder leaked descriptor");
        check(waitpid(-1, nullptr, WNOHANG) == -1 && errno == ECHILD, "decoder child not reaped");
        g.audio.resize(32000 * 2 / 4);
        writeMP4(g, (dir / "short-audio.mp4").string());
        for (const auto &entry : std::filesystem::directory_iterator(dir))
            check(entry.path().string().find(".tmp.") == std::string::npos, "temporary directory leaked");
        std::cout << "PASS streaming byte identity, failed encoder, blocked-writer cancellation, retry and short audio\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
