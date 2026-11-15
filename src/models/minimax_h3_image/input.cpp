#include "models/minimax_h3_image/model.h"
#include "json11.hpp"
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
namespace fastllm::minimax_h3_image {
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
} // namespace
Image readImage(const std::string &path) {
    if (!std::filesystem::is_regular_file(path) || std::filesystem::file_size(path) > (size_t(96) << 20))
        throw std::invalid_argument("reference image must be a regular file up to 96 MiB");
    std::ifstream file(path, std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(file)), {});
    if (bytes.size() >= 8 && bytes.compare(0, 8, "\x89PNG\r\n\x1a\n", 8) == 0) {
        try {
            return qwen_image::decodePNG(bytes, 20 * 1024 * 1024);
        } catch (const std::runtime_error &e) {
            throw std::invalid_argument(e.what());
        }
    }
    bool jpeg = bytes.size() >= 3 && uint8_t(bytes[0]) == 0xff && uint8_t(bytes[1]) == 0xd8 &&
                uint8_t(bytes[2]) == 0xff;
    bool webp = bytes.size() >= 12 && bytes.compare(0, 4, "RIFF") == 0 && bytes.compare(8, 4, "WEBP") == 0;
    if (!jpeg && !webp)
        throw std::invalid_argument("reference images must be PNG, JPEG or WebP");
    const std::string input = std::filesystem::absolute(path).string();
    const std::string formats = "jpeg_pipe,webp_pipe";
    std::string error;
    auto info =
        json11::Json::parse(capture({"ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe",
                                     "-format_whitelist", formats, "-show_streams", "-of", "json", input},
                                    1 << 20),
                            error);
    if (!error.empty() || info["streams"].array_items().size() != 1)
        throw std::invalid_argument("invalid reference image metadata");
    const auto &stream = info["streams"][0];
    int w = stream["width"].int_value(), h = stream["height"].int_value();
    if (w < 32 || h < 32 || w > 8192 || h > 8192 || int64_t(w) * h > 20 * 1024 * 1024)
        throw std::invalid_argument("reference image exceeds supported dimensions");
    // Let FFmpeg apply EXIF orientation and emit dimensions with its PNG output.
    auto png = capture({"ffmpeg",
                        "-v",
                        "error",
                        "-nostdin",
                        "-threads",
                        "2",
                        "-protocol_whitelist",
                        "file,pipe",
                        "-format_whitelist",
                        formats,
                        "-i",
                        input,
                        "-map",
                        "0:v:0",
                        "-an",
                        "-frames:v",
                        "1",
                        "-threads",
                        "2",
                        "-f",
                        "image2pipe",
                        "-vcodec",
                        "png",
                        "-pix_fmt",
                        "rgba",
                        "pipe:1"},
                       size_t(84) << 20);
    try {
        return qwen_image::decodePNG(png, 20 * 1024 * 1024);
    } catch (const std::runtime_error &e) {
        throw std::invalid_argument(e.what());
    }
}
Image decodeImage(const std::string &bytes) {
    if (bytes.empty() || bytes.size() > (size_t(32) << 20))
        throw std::invalid_argument("invalid reference image size");
    if (bytes.size() >= 8 && bytes.compare(0, 8, "\x89PNG\r\n\x1a\n", 8) == 0) {
        try {
            return qwen_image::decodePNG(bytes, 20 * 1024 * 1024);
        } catch (const std::runtime_error &e) {
            throw std::invalid_argument(e.what());
        }
    }
    std::string pattern = (std::filesystem::temp_directory_path() / "ftllm-h3-image-XXXXXX").string();
    int fd = mkstemp(pattern.data());
    if (fd < 0)
        throw std::runtime_error("cannot create image decode temporary file");
    try {
        size_t offset = 0;
        while (offset < bytes.size()) {
            ssize_t n = write(fd, bytes.data() + offset, bytes.size() - offset);
            if (n < 0 && errno == EINTR)
                continue;
            if (n <= 0)
                throw std::runtime_error("cannot write reference image");
            offset += size_t(n);
        }
        close(fd);
        fd = -1;
        auto image = readImage(pattern);
        unlink(pattern.c_str());
        return image;
    } catch (...) {
        if (fd >= 0)
            close(fd);
        unlink(pattern.c_str());
        throw;
    }
}
} // namespace fastllm::minimax_h3_image
