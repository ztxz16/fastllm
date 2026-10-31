#include "models/minimax_h3/runtime.h"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <signal.h>
#include <spawn.h>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

extern char **environ;

namespace fastllm {
namespace minimax_h3 {
struct MP4Writer::Impl {
    std::string folder, video, destination;
    size_t expected = 0, submitted = 0, frameBytes = 0;
    int input = -1;
    pid_t child = -1;
    std::thread worker;
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<std::vector<uint8_t>> queue;
    bool stopping = false, ended = false, drained = false;
    int inputError = 0;

    ~Impl() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        changed.notify_all();
        // Wake a blocked send before joining the writer on cancellation.
        if (input >= 0)
            shutdown(input, SHUT_RDWR);
        if (child > 0)
            kill(child, SIGKILL);
        if (worker.joinable())
            worker.join();
        if (input >= 0)
            close(input);
        if (child > 0) {
            int status;
            while (waitpid(child, &status, 0) < 0 && errno == EINTR) {}
        }
        if (!folder.empty()) {
            std::error_code error;
            std::filesystem::remove_all(folder, error);
        }
    }

    void start(const Generation &g, const std::string &path, const std::string &ffmpeg) {
        if (g.width <= 0 || g.height <= 0 || g.frames <= 0 || g.fps <= 0 ||
            g.sampleRate <= 0 || g.audio.size() % 2)
            throw std::invalid_argument("invalid H3 media output");
        frameBytes = size_t(g.width) * g.height * 3;
        if (size_t(g.frames) > std::numeric_limits<size_t>::max() / frameBytes)
            throw std::invalid_argument("H3 media output is too large");
        expected = frameBytes * g.frames;
        destination = path;
        std::string pattern = path + ".tmp.XXXXXX";
        std::vector<char> name(pattern.begin(), pattern.end());
        name.push_back(0);
        char *created = mkdtemp(name.data());
        if (!created)
            throw std::runtime_error("cannot create temporary media directory: " +
                                     std::string(std::strerror(errno)));
        folder = created;
        std::string audio = folder + "/audio.f32";
        video = folder + "/video.mp4";
        std::ofstream file(audio, std::ios::binary);
        file.write(reinterpret_cast<const char *>(g.audio.data()), g.audio.size() * sizeof(float));
        file.close();
        if (!file)
            throw std::runtime_error("cannot save media input: " + audio);
        std::vector<std::string> args = {
            ffmpeg, "-v", "error", "-nostdin", "-y",
            "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", std::to_string(g.width) + "x" + std::to_string(g.height),
            "-framerate", std::to_string(g.fps), "-i", "pipe:0",
            "-f", "f32le", "-ar", std::to_string(g.sampleRate), "-ac", "2", "-i", audio,
            "-c:v", "libx264", "-threads", "4", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", video};
        std::vector<char *> argv;
        for (auto &arg : args)
            argv.push_back(arg.data());
        argv.push_back(nullptr);
        int videoInput[2];
        if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, videoInput) < 0)
            throw std::runtime_error("cannot create FFmpeg input stream");
        posix_spawn_file_actions_t actions;
        int error = posix_spawn_file_actions_init(&actions);
        if (!error) {
            error = posix_spawn_file_actions_addclose(&actions, videoInput[0]);
            if (!error)
                error = posix_spawn_file_actions_adddup2(&actions, videoInput[1], STDIN_FILENO);
            if (!error && videoInput[1] != STDIN_FILENO)
                error = posix_spawn_file_actions_addclose(&actions, videoInput[1]);
            if (!error)
                error = posix_spawnp(&child, argv[0], &actions, nullptr, argv.data(), environ);
            posix_spawn_file_actions_destroy(&actions);
        }
        if (error) {
            close(videoInput[0]);
            close(videoInput[1]);
            throw std::runtime_error("cannot launch FFmpeg: " + std::string(std::strerror(error)));
        }
        close(videoInput[1]);
        input = videoInput[0];
        worker = std::thread([this] {
            struct Done {
                Impl &self;
                ~Done() {
                    std::lock_guard<std::mutex> lock(self.mutex);
                    self.drained = true;
                    self.changed.notify_all();
                }
            } done{*this};
            for (;;) {
                std::vector<uint8_t> chunk;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    changed.wait(lock, [&] { return stopping || ended || !queue.empty(); });
                    if (stopping || queue.empty())
                        return;
                    chunk = std::move(queue.front());
                    queue.pop_front();
                    changed.notify_all();
                }
                size_t offset = 0;
                while (offset < chunk.size()) {
                    ssize_t sent = send(input, chunk.data() + offset, chunk.size() - offset, MSG_NOSIGNAL);
                    if (sent < 0 && errno == EINTR)
                        continue;
                    if (sent <= 0) {
                        std::lock_guard<std::mutex> lock(mutex);
                        inputError = sent < 0 ? errno : EIO;
                        queue.clear();
                        changed.notify_all();
                        return;
                    }
                    offset += size_t(sent);
                }
            }
        });
    }

    void write(const uint8_t *data, size_t bytes, const std::function<void()> &checkpoint) {
        if (!data || !bytes || bytes % frameBytes || bytes > expected - submitted)
            throw std::invalid_argument("invalid H3 RGB chunk");
        // Keep host staging bounded even for the non-streaming writeMP4 API.
        while (bytes) {
            size_t count = std::min(bytes, frameBytes * 17);
            std::unique_lock<std::mutex> lock(mutex);
            while (queue.size() >= 2 && !inputError && !stopping && !ended) {
                if (changed.wait_for(lock, std::chrono::milliseconds(20)) == std::cv_status::timeout && checkpoint) {
                    lock.unlock();
                    checkpoint();
                    lock.lock();
                }
            }
            if (ended || stopping)
                throw std::runtime_error("H3 encoder is closed");
            // -shortest may close its input before consuming the final video frames.
            // finish() still checks the child status before publishing the result.
            if (inputError && inputError != EPIPE)
                throw std::runtime_error("cannot send RGB to FFmpeg");
            if (!inputError)
                queue.emplace_back(data, data + count);
            submitted += count;
            data += count;
            bytes -= count;
            changed.notify_all();
        }
    }

    void finish(const std::function<void()> &checkpoint) {
        if (submitted != expected)
            throw std::runtime_error("incomplete H3 video stream");
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (ended)
                throw std::runtime_error("H3 encoder is already closed");
            ended = true;
        }
        changed.notify_all();
        {
            std::unique_lock<std::mutex> lock(mutex);
            while (!drained) {
                if (changed.wait_for(lock, std::chrono::milliseconds(20)) == std::cv_status::timeout && checkpoint) {
                    lock.unlock();
                    checkpoint();
                    lock.lock();
                }
            }
        }
        worker.join();
        close(input);
        input = -1;
        int status = 0;
        pid_t waited;
        for (;;) {
            waited = waitpid(child, &status, checkpoint ? WNOHANG : 0);
            if (waited < 0 && errno == EINTR)
                continue;
            if (waited != 0)
                break;
            checkpoint();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        if (waited == child)
            child = -1;
        if ((inputError && inputError != EPIPE) || waited < 0 || !WIFEXITED(status) || WEXITSTATUS(status))
            throw std::runtime_error("FFmpeg failed; install ffmpeg with libx264/AAC or pass --ffmpeg");
        if (checkpoint)
            checkpoint();
        std::filesystem::rename(video, destination);
    }
};
MP4Writer::MP4Writer(const Generation &g, const std::string &path, const std::string &ffmpeg)
    : impl_(std::make_unique<Impl>()) {
    impl_->start(g, path, ffmpeg);
}
MP4Writer::~MP4Writer() = default;
void MP4Writer::write(const uint8_t *rgb, size_t bytes, std::function<void()> checkpoint) {
    impl_->write(rgb, bytes, checkpoint);
}
void MP4Writer::finish(std::function<void()> checkpoint) { impl_->finish(checkpoint); }
void writeMP4(const Generation &g, const std::string &path, const std::string &ffmpeg) {
    if (g.rgb.size() != size_t(g.width) * g.height * g.frames * 3)
        throw std::invalid_argument("invalid H3 media output");
    MP4Writer writer(g, path, ffmpeg);
    writer.write(g.rgb.data(), g.rgb.size());
    writer.finish();
}
} // namespace minimax_h3
} // namespace fastllm
