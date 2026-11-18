#include "pipeline.h"
#include "httplib.h"
#include "avc_http.h"
#include <condition_variable>
#include <fstream>
#include <future>
#include <iostream>
#include <random>
#include <fcntl.h>
#include <spawn.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;

namespace fs = std::filesystem;
using namespace fastllm::duo::live;
using Json = json11::Json;
namespace {
struct Conversation {
    std::string id;
    fs::path dir;
    pid_t child = -1;
    int socket = -1;
    std::mutex mutex;
    std::condition_variable changed;
    std::thread worker;
    std::vector<Json> events;
    bool busy = false, closed = false;
    bool avc = false;
    std::vector<size_t> turnOffsets;
    std::chrono::steady_clock::time_point touched = std::chrono::steady_clock::now();
    ~Conversation() {
        if (worker.joinable())
            worker.join();
    }
};
void response(httplib::Response &r, int status, const Json &j) {
    r.status = status;
    r.set_content(j.dump(), "application/json");
}
void emit(Conversation &c, Json::object event) {
    {
        std::lock_guard lock(c.mutex);
        c.touched = std::chrono::steady_clock::now();
        event["at"] =
            std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
        c.events.emplace_back(event);
        if (event["type"].string_value() == "done" || event["type"].string_value() == "error" ||
            event["type"].string_value() == "input_error")
            c.busy = false;
        if (event["type"].string_value() == "error")
            c.closed = true;
    }
    c.changed.notify_all();
}
constexpr int workerSocket = 3;
std::string receive(int fd) {
    char data[65536];
    ssize_t size;
    do {
        size = recv(fd, data, sizeof(data), MSG_TRUNC);
    } while (size < 0 && errno == EINTR);
    if (size <= 0)
        return {};
    if (size > ssize_t(sizeof(data)))
        throw std::runtime_error("oversized worker event");
    return {data, size_t(size)};
}
void sendPacket(int fd, const std::string &msg) {
    ssize_t n;
    do {
        n = send(fd, msg.data(), msg.size(), MSG_NOSIGNAL);
    } while (n < 0 && errno == EINTR);
    if (n != ssize_t(msg.size()))
        throw std::runtime_error("session worker disconnected");
}
int runWorker(char **argv) {
    // Each session owns its CUDA contexts. Shutdown reaps this process and its FFmpeg children.
    if (prctl(PR_SET_PDEATHSIG, SIGKILL) < 0 || getppid() != std::stoi(argv[8]))
        return 1;
    if (fcntl(workerSocket, F_SETFD, FD_CLOEXEC) < 0)
        return 1;
    try {
        Pipeline pipeline(argv[2], argv[3], std::stoi(argv[5]), std::stoi(argv[6]),
                          std::string(argv[7]) == "four", std::string(argv[9]) == "avc", argv[10]);
        for (;;) {
            auto turn = receive(workerSocket);
            if (turn.empty())
                return 0;
            fs::path dir = fs::path(argv[4]) / turn;
            try {
                pipeline.turn(dir / "input.media", dir, [&](Json::object event) {
                    if (auto it = event.find("file"); it != event.end()) {
                        event["url"] = "/media/" + dir.parent_path().filename().string() + "/" + turn + "/" +
                                       it->second.string_value();
                        event.erase(it);
                    }
                    sendPacket(workerSocket, Json(event).dump());
                });
            } catch (const InputError &e) {
                sendPacket(workerSocket,
                           Json(Json::object{{"type", "input_error"}, {"text", e.what()}}).dump());
            }
        }
    } catch (const std::exception &e) {
        try {
            sendPacket(workerSocket, Json(Json::object{{"type", "error"}, {"text", e.what()}}).dump());
        } catch (...) {
        }
        return 1;
    }
}
void startWorker(Conversation &c, const std::string &model, const fs::path &assets, int w, int h,
                 const std::string &profile, const fs::path &intentCache) {
    int pair[2];
    if (socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, pair))
        throw std::runtime_error("cannot open session socket");
    posix_spawn_file_actions_t actions;
    posix_spawnattr_t attr;
    posix_spawn_file_actions_init(&actions);
    posix_spawnattr_init(&attr);
    posix_spawn_file_actions_addclose(&actions, pair[0]);
    posix_spawn_file_actions_adddup2(&actions, pair[1], workerSocket);
    if (pair[1] != workerSocket)
        posix_spawn_file_actions_addclose(&actions, pair[1]);
    posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP);
    posix_spawnattr_setpgroup(&attr, 0);
    std::vector<std::string> args{"/proc/self/exe",
                                  "--worker",
                                  model,
                                  assets,
                                  c.dir,
                                  std::to_string(w),
                                  std::to_string(h),
                                  profile,
                                  std::to_string(getpid()),
                                  c.avc ? "avc" : "mp4",
                                  intentCache};
    std::vector<char *> av;
    for (auto &s : args)
        av.push_back(s.data());
    av.push_back(nullptr);
    int err = posix_spawn(&c.child, av[0], &actions, &attr, av.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    posix_spawnattr_destroy(&attr);
    close(pair[1]);
    if (err) {
        close(pair[0]);
        throw std::runtime_error("cannot start native session worker");
    }
    c.socket = pair[0];
    c.worker = std::thread([&c] {
        try {
            for (;;) {
                auto msg = receive(c.socket);
                if (msg.empty())
                    break;
                std::string err;
                auto event = Json::parse(msg, err);
                if (!err.empty() || !event.is_object())
                    throw std::runtime_error("invalid native worker event");
                emit(c, event.object_items());
            }
        } catch (const std::exception &e) {
            emit(c, {{"type", "error"}, {"text", e.what()}});
        }
        bool unexpected = false;
        {
            std::lock_guard lock(c.mutex);
            if (c.child > 0) {
                kill(-c.child, SIGKILL);
                int status;
                while (waitpid(c.child, &status, 0) < 0 && errno == EINTR) {
                }
            }
            c.child = -1;
            close(c.socket);
            c.socket = -1;
            unexpected = !c.closed;
            c.closed = true;
            c.busy = false;
        }
        if (unexpected)
            emit(c, {{"type", "error"}, {"text", "原生会话进程已退出"}});
        c.changed.notify_all();
    });
}
std::string uid() {
    static constexpr char hex[] = "0123456789abcdef";
    std::random_device rng;
    std::string out;
    for (int i = 0; i < 32; ++i)
        out += hex[rng() % 16];
    return out;
}
} // namespace
int main(int argc, char **argv) {
    if (argc == 11 && std::string(argv[1]) == "--worker")
        return runWorker(argv);
    if (argc != 6 && (argc != 8 || std::string(argv[6]) != "--intent-cache")) {
        std::cerr << "Usage: duo_live MODEL TOOLS_DUO_DIRECTORY SESSION_DIRECTORY PORT single|four "
                     "[--intent-cache DIRECTORY]\n";
        return 2;
    }
    try {
        const std::string root = argv[1], profile = argv[5];
        const fs::path assets = argv[2], sessions = argv[3];
        const fs::path intentCache = argc == 8 ? argv[7] : "";
        int port = std::stoi(argv[4]);
        if ((profile != "single" && profile != "four") || port < 1 || port > 65535)
            throw std::invalid_argument("invalid serving configuration");
        const std::string html = readFile(assets / "live.html");
        const auto player = readFile(assets / "avc-player.js");
        fs::create_directories(sessions);
        std::mutex state;
        std::shared_ptr<Conversation> current;
        auto lookup = [&](const std::string &id) {
            std::lock_guard lock(state);
            return current && current->id == id ? current : std::shared_ptr<Conversation>{};
        };
        httplib::Server server;
        server.set_payload_max_length(64u << 20);
        server.new_task_queue = [] { return new httplib::ThreadPool(8); };
        server.Get("/", [&](auto &, auto &r) { r.set_content(html, "text/html; charset=utf-8"); });
        server.Get("/avc-player.js", [&](auto &, auto &r) { r.set_content(player, "text/javascript"); });
        server.Get("/resolutions",
                   [](auto &, auto &r) { response(r, 200, Json::array{"512x384", "256x192"}); });
        auto retire = [&](const std::shared_ptr<Conversation> &c) {
            {
                std::lock_guard lock(c->mutex);
                c->closed = true;
                if (c->child > 0)
                    kill(-c->child, SIGKILL);
            }
            c->changed.notify_all();
            if (c->worker.joinable())
                c->worker.join();
            if (!c->turnOffsets.empty()) {
                auto dir = c->dir / ("turn-" + std::to_string(c->turnOffsets.size() - 1));
                std::error_code ec;
                fs::remove(dir / "frames.rgb", ec);
                fs::remove(dir / "audio.f32", ec);
                for (const auto &f : fs::directory_iterator(dir, ec))
                    if (f.is_directory() &&
                        f.path().filename().string().find(".mp4.tmp.") != std::string::npos)
                        fs::remove_all(f.path(), ec);
            }
        };
        server.Post("/sessions", [&](auto &q, auto &r) {
            std::lock_guard lock(state);
            if (current) {
                std::lock_guard live(current->mutex);
                if (!current->closed) {
                    response(r, 409, Json::object{{"detail", "已有活动会话"}});
                    return;
                }
            }
            auto res = q.get_param_value("resolution");
            int w = 512, h = 384;
            if (res == "256x192") {
                w = 256;
                h = 192;
            } else if (!res.empty() && res != "512x384") {
                response(r, 400, Json::object{{"detail", "不支持的回复分辨率"}});
                return;
            }
            if (current)
                retire(current);
            current.reset();
            auto next = std::make_shared<Conversation>();
            next->id = uid();
            next->dir = sessions / next->id;
            fs::create_directories(next->dir);
            next->avc = q.get_param_value("playback") == "avc";
            startWorker(*next, root, assets, w, h, profile, intentCache);
            current = next;
            response(r, 200, Json::object{{"id", next->id}});
        });
        server.Post(R"(/sessions/([a-f0-9]{32})/turns)", [&](const httplib::Request &q, httplib::Response &r,
                                                             const httplib::ContentReader &reader) {
            auto c = lookup(q.matches[1]);
            if (!c) {
                response(r, 404, Json::object{});
                return;
            }
            std::string body;
            bool oversized = false;
            if (!reader([&](const char *data, size_t size) {
                    if (size > (64u << 20) - body.size()) {
                        oversized = true;
                        return false;
                    }
                    body.append(data, size);
                    return true;
                })) {
                response(r, oversized ? 413 : 400, Json::object{{"detail", "无效的视频上传"}});
                return;
            }
            std::lock_guard lock(c->mutex);
            if (c->busy || c->closed) {
                response(r, 409, Json::object{{"detail", "会话忙或已结束"}});
                return;
            }
            if (body.empty()) {
                response(r, 400, Json::object{{"detail", "没有视频内容"}});
                return;
            }
            std::string turn = "turn-" + std::to_string(c->turnOffsets.size());
            auto folder = c->dir / turn;
            fs::create_directories(folder);
            auto path = folder / "input.media";
            std::ofstream file(path, std::ios::binary);
            file.write(body.data(), body.size());
            file.close();
            if (!file)
                throw std::runtime_error("cannot store recording");
            c->turnOffsets.push_back(c->events.size());
            c->busy = true;
            try {
                sendPacket(c->socket, turn);
            } catch (...) {
                c->turnOffsets.pop_back();
                c->busy = false;
                throw;
            }
            response(
                r, 200,
                Json::object{{"accepted", true}, {"stream", "/sessions/" + c->id + "/" + turn + "/avc"}});
        });
        server.Get(R"(/sessions/([a-f0-9]{32})/turn-([0-9]{1,9})/avc)", [&](auto &q, auto &r) {
            auto c = lookup(q.matches[1]);
            const auto turn = std::stoul(q.matches[2]);
            if (!c || !c->avc) {
                r.status = 404;
                return;
            }
            std::lock_guard lock(c->mutex);
            if (turn >= c->turnOffsets.size()) {
                r.status = 404;
                return;
            }
            auto folder = c->dir / ("turn-" + std::to_string(turn));
            serveAvcStream(
                r, [c, folder, cursor = c->turnOffsets[turn]]() mutable -> std::optional<fs::path> {
                    std::unique_lock lock(c->mutex);
                    for (;;) {
                        c->changed.wait(lock, [&] { return cursor < c->events.size() || c->closed; });
                        if (cursor == c->events.size())
                            return {};
                        auto &e = c->events[cursor++];
                        auto type = e["type"].string_value();
                        if (type == "chunk")
                            return folder / ("chunk-" + std::to_string(e["index"].int_value()) + ".dav");
                        if (type == "done" || type == "input_error" || type == "error")
                            return {};
                    }
                });
        });
        server.Get(R"(/sessions/([a-f0-9]{32})/events)", [&](auto &q, auto &r) {
            auto c = lookup(q.matches[1]);
            if (!c) {
                response(r, 404, Json::object{});
                return;
            }
            std::unique_lock lock(c->mutex);
            size_t after = 0;
            try {
                if (q.has_param("after"))
                    after = std::stoul(q.get_param_value("after"));
            } catch (...) {
                after = SIZE_MAX;
            }
            if (after > c->events.size()) {
                response(r, 400, Json::object{});
                return;
            }
            c->changed.wait_for(lock, std::chrono::seconds(8),
                                [&] { return c->events.size() > after || c->closed; });
            response(r, 200,
                     Json::object{{"events", Json::array(c->events.begin() + after, c->events.end())},
                                  {"cursor", int(c->events.size())},
                                  {"closed", c->closed}});
        });
        server.Delete(R"(/sessions/([a-f0-9]{32}))", [&](auto &q, auto &r) {
            auto c = lookup(q.matches[1]);
            if (!c) {
                response(r, 404, Json::object{});
                return;
            }
            {
                std::lock_guard lock(state);
                if (current == c)
                    retire(c);
            }
            response(r, 200, Json::object{{"closed", true}});
        });
        server.Get(R"(/media/([a-f0-9]{32})/(turn-[0-9]+)/(chunk-[0-9]+\.(?:mp4|dav)|reply\.mp4))",
                   [&](auto &q, auto &r) {
                       auto file = sessions / q.matches[1].str() / q.matches[2].str() / q.matches[3].str();
                       if (!fs::is_regular_file(file)) {
                           r.status = 404;
                           return;
                       }
                       r.set_content(readFile(file),
                                     file.extension() == ".dav" ? "application/octet-stream" : "video/mp4");
                   });
        std::cout << "duo_live " << profile << " port " << port << std::endl;
        std::atomic<bool> stopped{false};
        auto reaper = std::async(std::launch::async, [&] {
            while (!stopped.load()) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                std::lock_guard lock(state);
                if (!current)
                    continue;
                bool expired;
                {
                    std::lock_guard live(current->mutex);
                    expired = !current->closed && !current->busy &&
                              std::chrono::steady_clock::now() - current->touched > std::chrono::minutes(10);
                }
                if (expired)
                    retire(current);
            }
        });
        bool listening = server.listen("127.0.0.1", port);
        stopped = true;
        reaper.get();
        if (current)
            retire(current);
        if (!listening)
            throw std::runtime_error("failed to listen");
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
