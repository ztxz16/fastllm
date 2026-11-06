#include "distributed/process_group.h"
#include "process_protocol.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <filesystem>
#include <mutex>
#include <poll.h>
#include <random>
#include <set>
#include <signal.h>
#include <spawn.h>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

extern char **environ;

namespace fastllm {
namespace distributed {
namespace protocol {
namespace {
constexpr uint32_t kMagic = 0x46545031;

/**
 * 在共同截止时间内传输完整帧片段，拒绝 EOF 和超时。
 * @param fd 非阻塞 Unix socket。
 * @param bytes 待发送或接收的字节数组。
 * @param size 数组长度，不解释张量或指针。
 * @param sending true 发送，false 接收。
 * @param deadline 整个操作共用的单调时钟截止时间。
 */
void Transfer(int fd, uint8_t *bytes, size_t size, bool sending, Deadline deadline) {
    size_t done = 0;
    while (done < size) {
        auto now = Clock::now();
        if (now >= deadline) {
            throw std::runtime_error("process TP control timeout");
        }
        auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
        int timeout = static_cast<int>(std::min<int64_t>(INT_MAX, std::max<int64_t>(1, remaining)));
        pollfd item{fd, static_cast<short>(sending ? POLLOUT : POLLIN), 0};
        int ready = poll(&item, 1, timeout);
        if (ready < 0 && errno == EINTR) {
            continue;
        }
        if (ready < 0) {
            throw std::runtime_error(std::string("process TP poll: ") + strerror(errno));
        }
        if (ready == 0) {
            continue;
        }
        ssize_t count = sending ? send(fd, bytes + done, size - done, MSG_NOSIGNAL | MSG_DONTWAIT)
                                : recv(fd, bytes + done, size - done, MSG_DONTWAIT);
        if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) {
            continue;
        }
        if (count <= 0) {
            throw std::runtime_error(count == 0 ? "process TP worker disconnected"
                                               : std::string("process TP socket: ") + strerror(errno));
        }
        done += static_cast<size_t>(count);
    }
}
} // namespace

void Put32(std::vector<uint8_t> &bytes, uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        bytes.push_back(static_cast<uint8_t>(value >> shift));
    }
}

uint32_t Get32(const std::vector<uint8_t> &bytes, size_t offset) {
    if (offset > bytes.size() || bytes.size() - offset < 4) {
        throw std::runtime_error("truncated process TP integer");
    }
    uint32_t value = 0;
    for (int i = 0; i < 4; i++) {
        value |= uint32_t(bytes[offset + i]) << (i * 8);
    }
    return value;
}

void Put64(std::vector<uint8_t> &bytes, uint64_t value) {
    Put32(bytes, static_cast<uint32_t>(value));
    Put32(bytes, static_cast<uint32_t>(value >> 32));
}

uint64_t Get64(const std::vector<uint8_t> &bytes, size_t offset) {
    return uint64_t(Get32(bytes, offset)) | (uint64_t(Get32(bytes, offset + 4)) << 32);
}

void Send(int fd, const Message &message, Deadline deadline) {
    if (message.payload.size() > kMaxPayload) {
        throw std::runtime_error("process TP payload exceeds limit");
    }
    std::vector<uint8_t> header;
    for (uint32_t value : {kMagic, kVersion, static_cast<uint32_t>(message.kind),
                          static_cast<uint32_t>(message.payload.size()),
                          static_cast<uint32_t>(message.session), static_cast<uint32_t>(message.session >> 32),
                          static_cast<uint32_t>(message.sequence), static_cast<uint32_t>(message.sequence >> 32)}) {
        Put32(header, value);
    }
    Transfer(fd, header.data(), header.size(), true, deadline);
    // send 不会修改输入；统一传输函数的接收方向需要可写缓冲。
    Transfer(fd, const_cast<uint8_t*>(message.payload.data()), message.payload.size(), true, deadline);
}

Message Receive(int fd, Deadline deadline) {
    std::vector<uint8_t> header(32);
    Transfer(fd, header.data(), header.size(), false, deadline);
    if (Get32(header, 0) != kMagic || Get32(header, 4) != kVersion) {
        throw std::runtime_error("process TP protocol magic/version mismatch");
    }
    uint32_t length = Get32(header, 12);
    if (length > kMaxPayload) {
        throw std::runtime_error("process TP payload exceeds limit");
    }
    Message message{static_cast<Kind>(Get32(header, 8)),
                    Get32(header, 16) | (uint64_t(Get32(header, 20)) << 32),
                    Get32(header, 24) | (uint64_t(Get32(header, 28)) << 32), {}};
    message.payload.resize(length);
    Transfer(fd, message.payload.data(), message.payload.size(), false, deadline);
    return message;
}

std::string ErrorText(const Message &message) {
    return std::string(message.payload.begin(), message.payload.end());
}

void Validate(const Message &message, Kind kind, uint64_t session, uint64_t sequence) {
    if (message.session != session || message.sequence != sequence) {
        throw std::runtime_error("process TP session/sequence mismatch");
    }
    if (message.kind == Kind::Error) {
        throw std::runtime_error("process TP worker: " + ErrorText(message));
    }
    if (message.kind != kind) {
        throw std::runtime_error("unexpected process TP response");
    }
    if ((kind == Kind::Ready || kind == Kind::Pong || kind == Kind::Bye) && !message.payload.empty()) {
        throw std::runtime_error("unexpected process TP response payload");
    }
}
} // namespace protocol

using namespace protocol;

struct ProcessGroup::Impl {
    struct Child {
        ProcessWorkerInfo info;
        int fd = -1;
        int pidfd = -1;
        bool reaped = false;
        int exitStatus = 0;
    };
    // lifecycle 串行化 Start/Stop；所有 socket 和 waitpid 由 mutex 串行化。
    std::mutex lifecycle;
    mutable std::mutex mutex;
    std::condition_variable wake;
    std::thread monitor;
    ProcessGroupOptions options;
    ProcessGroupState state = ProcessGroupState::Stopped;
    std::string error;
    std::vector<Child> children;
    uint64_t session = 0;
    uint64_t sequence = 0;
    bool stopMonitor = false;

    ~Impl() {
        for (auto &child : children) {
            if (child.fd >= 0) {
                close(child.fd);
            }
            if (child.pidfd >= 0) {
                close(child.pidfd);
            }
        }
    }

    /**
     * 回收已退出的自有子进程，避免向已复用的 PID 发送信号。
     * @param child 本组通过 posix_spawn 创建的子进程。
     * @return true 表示已退出或已由外部 SIGCHLD 处理器回收。
     */
    bool Reap(Child &child) {
        if (child.reaped) {
            return true;
        }
        if (child.pidfd >= 0) {
            pollfd item{child.pidfd, POLLIN, 0};
            if (poll(&item, 1, 0) <= 0 || !(item.revents & (POLLIN | POLLHUP))) {
                return false;
            }
        }
        int status = 0;
        pid_t result = waitpid(child.info.pid, &status, WNOHANG);
        if (result == child.info.pid || (result < 0 && errno == ECHILD)) {
            child.reaped = true;
            child.exitStatus = result == child.info.pid ? status : 0;
        }
        return child.reaped;
    }

    /**
     * 有界等待进程退出，失败时保留未回收 PID，禁止丢失所有权后重启。
     * @param force true 先终止尚未回收的 worker。
     * @return 全部退出并回收返回 true；到期仍未退出返回 false。
     */
    bool ReapAll(bool force) {
        if (force) {
            for (auto &child : children) {
                if (child.fd >= 0) {
                    close(child.fd);
                    child.fd = -1;
                }
                if (!Reap(child) && child.pidfd >= 0) {
                    syscall(SYS_pidfd_send_signal, child.pidfd, SIGKILL, nullptr, 0);
                }
            }
        }
        auto deadline = Clock::now() + std::chrono::milliseconds(options.shutdownTimeoutMs);
        do {
            bool complete = true;
            for (auto &child : children) {
                complete = Reap(child) && complete;
            }
            if (complete) {
                for (auto &child : children) {
                    if (!force && child.exitStatus != 0 && error.empty()) {
                        error = "rank " + std::to_string(child.info.rank) + " exited abnormally after shutdown";
                    }
                    if (child.fd >= 0) {
                        close(child.fd);
                    }
                    if (child.pidfd >= 0) {
                        close(child.pidfd);
                    }
                }
                children.clear();
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        } while (Clock::now() < deadline);
        return false;
    }

    void Fail(const std::string &reason) {
        state = ProcessGroupState::Failed;
        error = reason;
        stopMonitor = true;
        if (!ReapAll(true)) {
            error += "; worker reap timeout";
        }
    }

    /**
     * 向整组发送相同命令并按 rank 收集响应。
     *
     * 先发送全部请求再接收，允许 worker 同时进入 NCCL；整个批次共用截止时间。
     * @param kind 请求类型。
     * @param reply 预期响应类型。
     * @param payload 不含进程指针的序列化数据。
     * @param deadline 整组操作的截止时间。
     * @return 按 rank 排列的响应；失败包含 rank 信息并抛异常。
     */
    std::vector<Message> Exchange(Kind kind, Kind reply, const std::vector<uint8_t> &payload,
                                  Deadline deadline) {
        ++sequence;
        for (auto &child : children) {
            try {
                Send(child.fd, {kind, session, sequence, payload}, deadline);
            } catch (const std::exception &e) {
                throw std::runtime_error("rank " + std::to_string(child.info.rank) + ": " + e.what());
            }
        }
        std::vector<Message> results;
        for (auto &child : children) {
            try {
                auto message = Receive(child.fd, deadline);
                Validate(message, reply, session, sequence);
                results.push_back(std::move(message));
            } catch (const std::exception &e) {
                throw std::runtime_error("rank " + std::to_string(child.info.rank) + ": " + e.what());
            }
        }
        return results;
    }

    /** 串行执行 Prefill/Begin/Decode，复用分片连续性和有限值验证；错误终止整组。 */
    std::vector<float> Logits(Kind kind, const std::vector<uint8_t> &payload, int timeoutMs) {
        std::lock_guard<std::mutex> lock(mutex);
        if (state != ProcessGroupState::Ready || options.controlOnly) {
            throw std::runtime_error("Qwen3 prefill requires a ready CUDA process TP group");
        }
        try {
            auto replies = Exchange(kind, Kind::PrefillLogits, payload,
                                    Clock::now() + std::chrono::milliseconds(timeoutMs));
            std::vector<float> logits;
            uint64_t vocabulary = 0;
            for (const auto &reply : replies) {
                if (reply.payload.size() < 24) {
                    throw std::runtime_error("invalid Qwen3 prefill reply header");
                }
                uint64_t begin = Get64(reply.payload, 0), count = Get64(reply.payload, 8);
                uint64_t total = Get64(reply.payload, 16);
                if (!total || total > 1048576 || (vocabulary && total != vocabulary) ||
                    begin != logits.size() || begin >= total || !count || count > total - begin ||
                    count > (kMaxPayload - 24) / 4 || reply.payload.size() != 24 + count * 4) {
                    throw std::runtime_error("invalid Qwen3 prefill vocabulary slices");
                }
                vocabulary = total;
                for (uint64_t i = 0; i < count; ++i) {
                    uint32_t bits = Get32(reply.payload, 24 + i * 4);
                    float value;
                    memcpy(&value, &bits, sizeof(value));
                    if (!std::isfinite(value)) {
                        throw std::runtime_error("nonfinite Qwen3 prefill logits");
                    }
                    logits.push_back(value);
                }
            }
            if (!vocabulary || logits.size() != vocabulary) {
                throw std::runtime_error("incomplete Qwen3 prefill vocabulary");
            }
            return logits;
        } catch (const std::exception &failure) {
            Fail(failure.what());
            throw std::runtime_error(error);
        }
    }

    /** 查询或释放请求缓存；验证各 rank 的活动状态、长度和容量一致。 */
    std::vector<Qwen3RequestInfo> RequestStates(bool end) {
        std::lock_guard<std::mutex> lock(mutex);
        if (state != ProcessGroupState::Ready || options.controlOnly) {
            throw std::runtime_error("Qwen3 request requires a ready CUDA process TP group");
        }
        try {
            auto replies = Exchange(end ? Kind::EndQwen3Request : Kind::QueryQwen3Request,
                end ? Kind::Qwen3RequestEnded : Kind::Qwen3RequestState, {},
                Clock::now() + std::chrono::milliseconds(options.operationTimeoutMs));
            std::vector<Qwen3RequestInfo> result;
            for (const auto &reply : replies) {
                const auto &bytes = reply.payload;
                if (bytes.size() != 24 || Get32(bytes, 0) > 1) {
                    throw std::runtime_error("invalid Qwen3 request state header");
                }
                Qwen3RequestInfo info{Get32(bytes, 0) != 0, Get32(bytes, 4), Get32(bytes, 8),
                                      Get32(bytes, 12), Get64(bytes, 16)};
                bool invalid = info.active ?
                    (end || !info.tokens || info.tokens > info.capacity || info.capacity > 2048 ||
                     info.positionBegin > 1048576 || info.capacity - 1 > 1048576 - info.positionBegin ||
                     !info.deviceBytes || info.deviceBytes > 256ULL * 1024 * 1024) :
                    (info.tokens || info.capacity || info.positionBegin || info.deviceBytes);
                if (!result.empty()) {
                    const auto &first = result.front();
                    invalid = invalid || info.active != first.active || info.tokens != first.tokens ||
                              info.capacity != first.capacity || info.positionBegin != first.positionBegin;
                }
                if (invalid) {
                    throw std::runtime_error("invalid/inconsistent Qwen3 request state");
                }
                result.push_back(info);
            }
            return result;
        } catch (const std::exception &failure) {
            Fail(failure.what());
            throw std::runtime_error(error);
        }
    }

    /**
     * 创建独立 worker 并重定向指定控制 socket。
     * @param rank 连续的组内编号。
     * @return 含父端 socket 和自有 PID 的子进程记录；失败关闭临时描述符。
     */
    Child Spawn(int rank) {
        int sockets[2];
        if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0, sockets) != 0) {
            throw std::runtime_error(std::string("socketpair: ") + strerror(errno));
        }
        // 先复制到高位 CLOEXEC fd，避免原端点恰好等于固定 worker fd 3。
        int source = fcntl(sockets[1], F_DUPFD_CLOEXEC, 4);
        close(sockets[1]);
        if (source < 0) {
            close(sockets[0]);
            throw std::runtime_error("cannot duplicate worker socket");
        }
        std::vector<std::string> args{options.workerPath, "--control-fd", std::to_string(kWorkerFd),
            "--rank", std::to_string(rank), "--world-size", std::to_string(options.devices.size()),
            "--device", std::to_string(options.devices[rank]), "--session", std::to_string(session),
            "--parent-pid", std::to_string(getpid()), "--control-only", options.controlOnly ? "1" : "0"};
        std::vector<char*> argv;
        for (auto &arg : args) {
            argv.push_back(&arg[0]);
        }
        argv.push_back(nullptr);

        posix_spawn_file_actions_t actions;
        int result = posix_spawn_file_actions_init(&actions);
        bool initialized = result == 0;
        if (result == 0) {
            result = posix_spawn_file_actions_adddup2(&actions, source, kWorkerFd);
        }
        if (result == 0) {
            // 在 exec 前关闭父进程描述符，避免误关动态库初始化中新打开的资源。
            DIR *directory = opendir("/proc/self/fd");
            if (directory == nullptr) {
                result = errno;
            } else {
                int directoryFd = dirfd(directory);
                while (dirent *entry = readdir(directory)) {
                    int fd = std::atoi(entry->d_name);
                    if (fd > kWorkerFd && fd != directoryFd) {
                        result = posix_spawn_file_actions_addclose(&actions, fd);
                        if (result != 0) {
                            break;
                        }
                    }
                }
                closedir(directory);
            }
        }
        pid_t pid = -1;
        if (result == 0) {
            result = posix_spawn(&pid, options.workerPath.c_str(), &actions, nullptr, argv.data(), environ);
        }
        if (initialized) {
            posix_spawn_file_actions_destroy(&actions);
        }
        close(source);
        if (result != 0) {
            close(sockets[0]);
            throw std::runtime_error("spawn rank " + std::to_string(rank) + ": " + strerror(result));
        }
        int pidfd = static_cast<int>(syscall(SYS_pidfd_open, pid, 0));
        if (pidfd < 0) {
            // socket 关闭会触发 worker 的父连接监控退出；不向不稳定的裸 PID 发信号。
            close(sockets[0]);
            children.push_back({{rank, options.devices[rank], static_cast<int>(pid)}, -1, -1, false});
            throw std::runtime_error("pidfd_open failed; process TP requires Linux 5.3+ and pidfd permission");
        }
        return {{rank, options.devices[rank], static_cast<int>(pid)}, sockets[0], pidfd, false};
    }

    void Monitor() {
        std::unique_lock<std::mutex> lock(mutex);
        while (!wake.wait_for(lock, std::chrono::milliseconds(options.heartbeatIntervalMs),
                              [&]() { return stopMonitor; })) {
            try {
                for (auto &child : children) {
                    if (Reap(child)) {
                        throw std::runtime_error("rank " + std::to_string(child.info.rank) + " exited");
                    }
                }
                Exchange(Kind::Ping, Kind::Pong, {},
                         Clock::now() + std::chrono::milliseconds(options.operationTimeoutMs));
            } catch (const std::exception &e) {
                Fail(e.what());
                return;
            }
        }
    }
};

ProcessGroup::ProcessGroup() : impl(new Impl) {}

ProcessGroup::~ProcessGroup() {
    try {
        Stop();
    } catch (...) {
        // 析构不能抛异常；Stop 已尝试 SIGKILL 和有界回收。
    }
}

void ProcessGroup::Start(const ProcessGroupOptions &options) {
    std::lock_guard<std::mutex> lifecycleLock(impl->lifecycle);
    std::lock_guard<std::mutex> lock(impl->mutex);
    if (impl->state != ProcessGroupState::Stopped || impl->monitor.joinable() || !impl->children.empty()) {
        throw std::runtime_error("Stop the previous process TP group before Start");
    }
    std::set<int> unique(options.devices.begin(), options.devices.end());
    if (options.workerPath.empty() || options.devices.empty() || options.devices.size() > 1024 ||
        unique.size() != options.devices.size() || *unique.begin() < 0 ||
        options.startupTimeoutMs <= 0 || options.operationTimeoutMs <= 0 ||
        options.shutdownTimeoutMs <= 0 || options.heartbeatIntervalMs <= 0) {
        throw std::invalid_argument("invalid process TP worker path, devices or timeout");
    }
    impl->options = options;
    impl->state = ProcessGroupState::Starting;
    impl->error.clear();
    impl->stopMonitor = false;
    impl->sequence = 0;

    try {
        std::random_device random;
        impl->session = (uint64_t(random()) << 32) | random();
        if (impl->session == 0) {
            impl->session = 1;
        }
        // 在 spawn 前检查 pidfd 能力，避免旧内核或 seccomp 拒绝后留下无法跟踪的 worker。
        int selfPidFd = static_cast<int>(syscall(SYS_pidfd_open, getpid(), 0));
        if (selfPidFd < 0) {
            throw std::runtime_error("process TP requires Linux 5.3+ with pidfd_open permission");
        }
        int signalResult = static_cast<int>(syscall(SYS_pidfd_send_signal, selfPidFd, 0, nullptr, 0));
        close(selfPidFd);
        if (signalResult != 0) {
            throw std::runtime_error("process TP requires pidfd_send_signal permission");
        }
        auto deadline = Clock::now() + std::chrono::milliseconds(options.startupTimeoutMs);
        impl->children.reserve(options.devices.size());
        for (size_t rank = 0; rank < options.devices.size(); rank++) {
            impl->children.push_back(impl->Spawn(static_cast<int>(rank)));
        }
        std::set<std::vector<uint8_t>> deviceUuids;
        for (auto &child : impl->children) {
            auto hello = Receive(child.fd, deadline);
            Validate(hello, Kind::Hello, impl->session, 0);
            if (hello.payload.size() < 40 || hello.payload.size() > 512 ||
                Get32(hello.payload, 0) != uint32_t(child.info.rank) ||
                Get32(hello.payload, 4) != uint32_t(child.info.device) ||
                Get32(hello.payload, 8) != uint32_t(child.info.pid) ||
                Get32(hello.payload, 12) != uint32_t(!options.controlOnly)) {
                throw std::runtime_error("worker identity/CUDA capability mismatch");
            }
            child.info.cudaRuntimeVersion = static_cast<int>(Get32(hello.payload, 16));
            child.info.ncclVersion = static_cast<int>(Get32(hello.payload, 20));
            child.info.deviceName.assign(hello.payload.begin() + 40, hello.payload.end());
            std::vector<uint8_t> uuid(hello.payload.begin() + 24, hello.payload.begin() + 40);
            if (!options.controlOnly && !deviceUuids.insert(uuid).second) {
                throw std::runtime_error("multiple process TP ranks refer to the same CUDA device UUID");
            }
        }

        std::vector<uint8_t> id;
        if (!options.controlOnly) {
            ++impl->sequence;
            Send(impl->children[0].fd, {Kind::CreateId, impl->session, impl->sequence, {}}, deadline);
            auto reply = Receive(impl->children[0].fd, deadline);
            Validate(reply, Kind::UniqueId, impl->session, impl->sequence);
            id = std::move(reply.payload);
        }
        impl->Exchange(Kind::Initialize, Kind::Ready, id, deadline);
        impl->state = ProcessGroupState::Ready;
        impl->monitor = std::thread([this]() { impl->Monitor(); });
    } catch (const std::exception &e) {
        impl->Fail(e.what());
        throw std::runtime_error(impl->error);
    }
}

void ProcessGroup::Ping() {
    std::lock_guard<std::mutex> lock(impl->mutex);
    if (impl->state != ProcessGroupState::Ready) {
        throw std::runtime_error("process TP group not ready: " + impl->error);
    }
    try {
        impl->Exchange(Kind::Ping, Kind::Pong, {},
                       Clock::now() + std::chrono::milliseconds(impl->options.operationTimeoutMs));
    } catch (const std::exception &e) {
        impl->Fail(e.what());
        throw std::runtime_error(impl->error);
    }
}

std::vector<float> ProcessGroup::CheckAllReduce() {
    std::lock_guard<std::mutex> lock(impl->mutex);
    if (impl->state != ProcessGroupState::Ready || impl->options.controlOnly) {
        throw std::runtime_error("NCCL check requires a ready CUDA process TP group");
    }
    try {
        auto replies = impl->Exchange(Kind::AllReduce, Kind::AllReduceResult, {},
                                      Clock::now() + std::chrono::milliseconds(impl->options.operationTimeoutMs));
        std::vector<float> values;
        for (auto &reply : replies) {
            if (reply.payload.size() != 4) {
                throw std::runtime_error("invalid NCCL check result size");
            }
            uint32_t bits = Get32(reply.payload, 0);
            float value;
            static_assert(sizeof(value) == sizeof(bits), "FP32 required");
            memcpy(&value, &bits, sizeof(value));
            values.push_back(value);
        }
        return values;
    } catch (const std::exception &e) {
        impl->Fail(e.what());
        throw std::runtime_error(impl->error);
    }
}

std::vector<ProcessModelInfo> ProcessGroup::LoadQwen3(const std::string &directory,
        uint64_t maxTensorBytes, uint64_t maxDeviceBytes, int timeoutMs) {
    if (directory.empty() || directory.find('\0') != std::string::npos ||
        !maxTensorBytes || !maxDeviceBytes || timeoutMs <= 0) {
        throw std::invalid_argument("invalid Qwen3 loading options");
    }
    const std::string absolute = std::filesystem::absolute(directory).string();
    if (absolute.size() > kMaxPayload - 16) {
        throw std::invalid_argument("Qwen3 model path exceeds protocol limit");
    }
    std::vector<uint8_t> payload;
    Put64(payload, maxTensorBytes);
    Put64(payload, maxDeviceBytes);
    payload.insert(payload.end(), absolute.begin(), absolute.end());

    std::lock_guard<std::mutex> lock(impl->mutex);
    if (impl->state != ProcessGroupState::Ready || impl->options.controlOnly) {
        throw std::runtime_error("Qwen3 loading requires a ready CUDA process TP group");
    }
    try {
        auto replies = impl->Exchange(Kind::LoadQwen3, Kind::ModelLoaded, payload,
                                      Clock::now() + std::chrono::milliseconds(timeoutMs));
        std::vector<ProcessModelInfo> results;
        for (const auto &reply : replies) {
            if (reply.payload.size() != 20 || !Get32(reply.payload, 0) || !Get64(reply.payload, 4)) {
                throw std::runtime_error("invalid Qwen3 loading result");
            }
            results.push_back({Get32(reply.payload, 0), Get64(reply.payload, 4), Get64(reply.payload, 12)});
        }
        return results;
    } catch (const std::exception &e) {
        impl->Fail(e.what());
        throw std::runtime_error(impl->error);
    }
}

std::vector<uint8_t> ProcessGroup::ReadQwen3Weight(int rank, uint32_t tensorIndex,
                                                uint64_t offset, uint32_t bytes) {
    std::lock_guard<std::mutex> lock(impl->mutex);
    if (rank < 0 || size_t(rank) >= impl->children.size() || !bytes || bytes > kMaxPayload ||
        bytes % 2 || offset % 2) {
        throw std::invalid_argument("invalid Qwen3 readback range");
    }
    if (impl->state != ProcessGroupState::Ready || impl->options.controlOnly) {
        throw std::runtime_error("Qwen3 readback requires a ready CUDA process TP group");
    }
    std::vector<uint8_t> payload;
    Put32(payload, static_cast<uint32_t>(rank));
    Put32(payload, tensorIndex);
    Put64(payload, offset);
    Put32(payload, bytes);
    try {
        auto replies = impl->Exchange(Kind::ReadWeight, Kind::WeightData, payload,
                                      Clock::now() + std::chrono::milliseconds(impl->options.operationTimeoutMs));
        for (size_t i = 0; i < replies.size(); ++i) {
            if (replies[i].payload.size() != (i == size_t(rank) ? bytes : 0)) {
                throw std::runtime_error("invalid Qwen3 readback response size");
            }
        }
        return std::move(replies[rank].payload);
    } catch (const std::exception &e) {
        impl->Fail(e.what());
        throw std::runtime_error(impl->error);
    }
}

void ProcessGroup::UnloadQwen3() {
    std::lock_guard<std::mutex> lock(impl->mutex);
    if (impl->state != ProcessGroupState::Ready || impl->options.controlOnly) {
        throw std::runtime_error("Qwen3 unload requires a ready CUDA process TP group");
    }
    try {
        auto replies = impl->Exchange(Kind::UnloadQwen3, Kind::ModelUnloaded, {},
                                      Clock::now() + std::chrono::milliseconds(impl->options.operationTimeoutMs));
        for (const auto &reply : replies) {
            if (!reply.payload.empty()) {
                throw std::runtime_error("invalid Qwen3 unload response");
            }
        }
    } catch (const std::exception &e) {
        impl->Fail(e.what());
        throw std::runtime_error(impl->error);
    }
}

std::vector<std::vector<float>> ProcessGroup::ForwardQwen3Layer(uint32_t layer, uint32_t tokens,
        uint32_t hidden, uint32_t positionBegin, Qwen3LayerStage stage, const std::vector<float> &input,
        int timeoutMs) {
    if (!tokens || tokens > 128 || !hidden || uint64_t(tokens) * hidden != input.size() ||
        input.size() > (kMaxPayload - 24) / 4 || positionBegin > 1048576 || uint32_t(stage) > 2 || timeoutMs <= 0) {
        throw std::invalid_argument("invalid Qwen3 layer request shape/stage/position/timeout");
    }
    std::vector<uint8_t> payload;
    for (uint32_t field : {layer, tokens, hidden, positionBegin, uint32_t(stage), uint32_t(timeoutMs)}) {
        Put32(payload, field);
    }
    for (float value : input) {
        if (!std::isfinite(value) || std::fabs(value) > 65504) {
            throw std::invalid_argument("Qwen3 layer input outside finite FP16 range");
        }
        uint32_t bits;
        memcpy(&bits, &value, sizeof(bits));
        Put32(payload, bits);
    }
    std::lock_guard<std::mutex> lock(impl->mutex);
    if (impl->state != ProcessGroupState::Ready || impl->options.controlOnly) {
        throw std::runtime_error("Qwen3 forward requires a ready CUDA process TP group");
    }
    try {
        auto replies = impl->Exchange(Kind::ForwardLayer, Kind::LayerOutput, payload,
                                      Clock::now() + std::chrono::milliseconds(timeoutMs));
        std::vector<std::vector<float>> results;
        for (const auto &reply : replies) {
            if (reply.payload.size() != input.size() * 4) {
                throw std::runtime_error("invalid Qwen3 layer output length");
            }
            std::vector<float> output(input.size());
            for (size_t i = 0; i < output.size(); ++i) {
                uint32_t bits = Get32(reply.payload, i * 4);
                memcpy(&output[i], &bits, sizeof(bits));
                if (!std::isfinite(output[i])) {
                    throw std::runtime_error("nonfinite Qwen3 layer output");
                }
            }
            results.push_back(std::move(output));
        }
        return results;
    } catch (const std::exception &error) {
        impl->Fail(error.what());
        throw std::runtime_error(impl->error);
    }
}

std::vector<float> ProcessGroup::PrefillQwen3(const std::vector<uint32_t> &tokenIds,
                                            uint32_t positionBegin, int timeoutMs) {
    if (tokenIds.empty() || tokenIds.size() > 128 || positionBegin > 1048576 || timeoutMs <= 0) {
        throw std::invalid_argument("invalid Qwen3 prefill request count/position/timeout");
    }
    std::vector<uint8_t> payload;
    Put32(payload, positionBegin);
    Put32(payload, uint32_t(timeoutMs));
    for (auto id : tokenIds) {
        Put32(payload, id);
    }

    return impl->Logits(Kind::PrefillQwen3, payload, timeoutMs);
}

std::vector<float> ProcessGroup::BeginQwen3Request(const std::vector<uint32_t> &tokenIds,
                                                 uint32_t capacity, uint32_t positionBegin, int timeoutMs) {
    if (tokenIds.empty() || tokenIds.size() > 128 || !capacity || capacity > 2048 ||
        tokenIds.size() > capacity || positionBegin > 1048576 ||
        capacity - 1 > 1048576 - positionBegin || timeoutMs <= 0) {
        throw std::invalid_argument("invalid Qwen3 request capacity/position/timeout");
    }
    std::vector<uint8_t> payload;
    Put32(payload, positionBegin);
    Put32(payload, uint32_t(timeoutMs));
    Put32(payload, capacity);
    for (auto id : tokenIds) {
        Put32(payload, id);
    }
    return impl->Logits(Kind::BeginQwen3Request, payload, timeoutMs);
}

std::vector<float> ProcessGroup::DecodeQwen3(uint32_t tokenId, int timeoutMs) {
    if (timeoutMs <= 0) {
        throw std::invalid_argument("invalid Qwen3 decode timeout");
    }
    std::vector<uint8_t> payload;
    Put32(payload, tokenId);
    Put32(payload, uint32_t(timeoutMs));
    return impl->Logits(Kind::DecodeQwen3, payload, timeoutMs);
}

void ProcessGroup::EndQwen3Request() {
    impl->RequestStates(true);
}

std::vector<Qwen3RequestInfo> ProcessGroup::Qwen3Requests() {
    return impl->RequestStates(false);
}

void ProcessGroup::Stop() {
    std::lock_guard<std::mutex> lifecycleLock(impl->lifecycle);
    {
        std::lock_guard<std::mutex> lock(impl->mutex);
        impl->stopMonitor = true;
        impl->wake.notify_all();
    }
    if (impl->monitor.joinable()) {
        impl->monitor.join();
    }

    std::lock_guard<std::mutex> lock(impl->mutex);
    bool graceful = impl->state == ProcessGroupState::Ready;
    impl->state = ProcessGroupState::Stopping;
    if (graceful) {
        try {
            impl->Exchange(Kind::Shutdown, Kind::Bye, {},
                           Clock::now() + std::chrono::milliseconds(impl->options.shutdownTimeoutMs));
        } catch (const std::exception &e) {
            impl->error = e.what();
            graceful = false;
        }
    }
    bool reaped = graceful && impl->ReapAll(false);
    if (graceful && !reaped && impl->error.empty()) {
        impl->error = "worker exit timeout after shutdown acknowledgement";
    }
    if (!reaped && !impl->ReapAll(true)) {
        impl->state = ProcessGroupState::Failed;
        impl->error += "; worker reap timeout";
        throw std::runtime_error(impl->error);
    }
    impl->state = ProcessGroupState::Stopped;
}

ProcessGroupState ProcessGroup::State() const {
    std::lock_guard<std::mutex> lock(impl->mutex);
    return impl->state;
}

std::string ProcessGroup::LastError() const {
    std::lock_guard<std::mutex> lock(impl->mutex);
    return impl->error;
}

std::vector<ProcessWorkerInfo> ProcessGroup::Workers() const {
    std::lock_guard<std::mutex> lock(impl->mutex);
    std::vector<ProcessWorkerInfo> result;
    for (auto &child : impl->children) {
        result.push_back(child.info);
    }
    return result;
}

} // namespace distributed
} // namespace fastllm
