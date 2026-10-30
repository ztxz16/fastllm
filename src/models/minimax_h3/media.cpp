#include "models/minimax_h3/runtime.h"
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>

namespace fastllm {
namespace minimax_h3 {
void writeMP4(const Generation &g, const std::string &path, const std::string &ffmpeg) {
    if (g.width <= 0 || g.height <= 0 || g.frames <= 0 ||
        g.rgb.size() != size_t(g.width) * g.height * g.frames * 3 || g.audio.size() % 2)
        throw std::invalid_argument("invalid H3 media output");
    std::string pattern = path + ".tmp.XXXXXX";
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back(0);
    char *folder = mkdtemp(name.data());
    if (!folder)
        throw std::runtime_error("cannot create temporary media directory: " +
                                 std::string(std::strerror(errno)));
    struct Cleanup {
        std::string folder;
        ~Cleanup() {
            std::error_code ec;
            std::filesystem::remove_all(folder, ec);
        }
    } cleanup{folder};
    std::string rgb = cleanup.folder + "/video.rgb", audio = cleanup.folder + "/audio.f32",
                video = cleanup.folder + "/video.mp4";
    auto save = [](const std::string &p, const void *data, size_t size) {
        std::ofstream f(p, std::ios::binary);
        f.write(static_cast<const char *>(data), size);
        f.close();
        if (!f)
            throw std::runtime_error("cannot save media input: " + p);
    };
    save(rgb, g.rgb.data(), g.rgb.size());
    save(audio, g.audio.data(), g.audio.size() * 4);
    std::vector<std::string> args = {ffmpeg,
                                     "-v",
                                     "error",
                                     "-nostdin",
                                     "-y",
                                     "-f",
                                     "rawvideo",
                                     "-pixel_format",
                                     "rgb24",
                                     "-video_size",
                                     std::to_string(g.width) + "x" + std::to_string(g.height),
                                     "-framerate",
                                     std::to_string(g.fps),
                                     "-i",
                                     rgb,
                                     "-f",
                                     "f32le",
                                     "-ar",
                                     std::to_string(g.sampleRate),
                                     "-ac",
                                     "2",
                                     "-i",
                                     audio,
                                     "-c:v",
                                     "libx264",
                                     "-threads",
                                     "4",
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
                                     "-shortest",
                                     "-movflags",
                                     "+faststart",
                                     video};
    std::vector<char *> argv;
    for (auto &arg : args)
        argv.push_back(arg.data());
    argv.push_back(nullptr);
    pid_t child = fork();
    if (child < 0)
        throw std::runtime_error("cannot launch FFmpeg");
    if (child == 0) {
        execvp(argv[0], argv.data());
        _exit(127);
    }
    int status = 0;
    pid_t waited;
    do {
        waited = waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    if (waited < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
        throw std::runtime_error("FFmpeg failed; install ffmpeg with libx264/AAC or pass --ffmpeg");
    std::filesystem::rename(video, path);
}
} // namespace minimax_h3
} // namespace fastllm
