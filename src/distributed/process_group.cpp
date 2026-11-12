#include "distributed/qwen3_limits.h"
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
 * 报告实际退出的 worker，并尽力保留 socket 中已经完整到达的错误回复。
 *
 * 只窥读当前事务的完整错误帧，不等待补齐消息、不消费数据，也不回收进程。
 * 错误消息尚未发完或不属于当前事务时，仅报告退出 rank。
 * @param worker 已由 pidfd 确认退出的 worker。
 * @param context 受进程组互斥锁保护的本次事务。
 */
[[noreturn]] void ThrowWorkerExited(const WorkerWatch &worker, const WaitContext &context) {
    std::string reason = "rank " + std::to_string(worker.rank) + " exited during process TP control";
    std::vector<uint8_t> header(32);
    ssize_t count = recv(worker.fd, header.data(), header.size(), MSG_PEEK | MSG_DONTWAIT);
    if (count == static_cast<ssize_t>(header.size()) && Get32(header, 0) == kMagic &&
        Get32(header, 4) == kVersion && Get32(header, 8) == static_cast<uint32_t>(Kind::Error) &&
        Get64(header, 16) == context.session && Get64(header, 24) == context.sequence) {
        uint32_t length = Get32(header, 12);
        // 正式 worker 的错误文本上限为 4096 字节；诊断路径不得分配任意大小的帧。
        if (length <= 4096) {
            std::vector<uint8_t> frame(32 + length);
            count = recv(worker.fd, frame.data(), frame.size(), MSG_PEEK | MSG_DONTWAIT);
            if (count == static_cast<ssize_t>(frame.size())) {
                reason += "; worker: " + std::string(frame.begin() + 32, frame.end());
            }
        }
    }
    throw WorkerExited(reason);
}

/**
 * 检查退出监听事件；句柄异常和进程退出均终止当前事务。
 * @param worker 被监听的 worker，不转移其句柄所有权。
 * @param events poll 返回的事件位。
 * @return true 表示进程已退出；无退出事件返回 false；无效句柄抛出异常。
 */
bool HasExited(const WorkerWatch &worker, short events) {
    if (events & (POLLERR | POLLNVAL)) {
        throw WorkerExited("rank " + std::to_string(worker.rank) + " process TP pidfd poll error");
    }
    return events & (POLLIN | POLLHUP);
}

/**
 * 在共同截止时间内传输完整帧片段，同时响应整组 worker 的退出。
 *
 * 持锁借用退出句柄，所有退出异常仍由上层统一清理。当前接收端退出后允许
 * 读完已经缓冲的数据，以保留错误回复；遇到 EOF 或需要继续等待时立即失败。
 * @param fd 非阻塞 Unix socket。
 * @param bytes 待发送或接收的字节数组。
 * @param size 数组长度，不解释张量或指针。
 * @param sending true 发送，false 接收。
 * @param deadline 整个操作共用的单调时钟截止时间。
 * @param context 父进程的退出监听集合；worker 收发和正常关闭时传 nullptr。
 */
void Transfer(int fd, uint8_t *bytes, size_t size, bool sending, Deadline deadline,
              const WaitContext *context) {
    if (size == 0) {
        return;
    }
    pollfd item{fd, static_cast<short>(sending ? POLLOUT : POLLIN), 0};
    std::vector<pollfd> items;
    if (context) {
        items.reserve(context->workers.size() + 1);
        items.push_back(item);
        for (const auto &worker : context->workers) {
            items.push_back({worker.pidfd, POLLIN, 0});
        }
    }

    size_t done = 0;
    while (done < size) {
        auto now = Clock::now();
        if (now >= deadline) {
            throw std::runtime_error("process TP control timeout");
        }
        auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
        int timeout = static_cast<int>(std::min<int64_t>(INT_MAX, std::max<int64_t>(1, remaining)));
        int ready = poll(context ? items.data() : &item, context ? items.size() : 1, timeout);
        if (ready < 0 && errno == EINTR) {
            continue;
        }
        if (ready < 0) {
            throw std::runtime_error(std::string("process TP poll: ") + strerror(errno));
        }
        if (ready == 0) {
            continue;
        }

        const WorkerWatch *exitedReader = nullptr;
        if (context) {
            for (size_t i = 0; i < context->workers.size(); ++i) {
                const auto &worker = context->workers[i];
                if (HasExited(worker, items[i + 1].revents)) {
                    if (!sending && worker.fd == fd) {
                        exitedReader = &worker;
                    } else {
                        ThrowWorkerExited(worker, *context);
                    }
                }
            }
        }

        ssize_t count = sending ? send(fd, bytes + done, size - done, MSG_NOSIGNAL | MSG_DONTWAIT)
                                : recv(fd, bytes + done, size - done, MSG_DONTWAIT);
        if (count < 0 && errno == EINTR) {
            continue;
        }
        if (count <= 0 && exitedReader) {
            ThrowWorkerExited(*exitedReader, *context);
        }
        if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
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

void Send(int fd, const Message &message, Deadline deadline, const WaitContext *context) {
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
    Transfer(fd, header.data(), header.size(), true, deadline, context);
    // send 不会修改输入；统一传输函数的接收方向需要可写缓冲。
    Transfer(fd, const_cast<uint8_t*>(message.payload.data()), message.payload.size(), true, deadline, context);
}

Message Receive(int fd, Deadline deadline, const WaitContext *context) {
    std::vector<uint8_t> header(32);
    Transfer(fd, header.data(), header.size(), false, deadline, context);
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
    Transfer(fd, message.payload.data(), message.payload.size(), false, deadline, context);
    return message;
}

/**
 * 在事务返回成功前检查已回复的 worker 是否也发生了退出。
 * @param context 当前事务的退出监听集合；必须在进程组锁保护下调用。
 */
void CheckWorkerExits(const WaitContext &context) {
    std::vector<pollfd> items;
    items.reserve(context.workers.size());
    for (const auto &worker : context.workers) {
        items.push_back({worker.pidfd, POLLIN, 0});
    }
    int ready;
    do {
        ready = poll(items.data(), items.size(), 0);
    } while (ready < 0 && errno == EINTR);
    if (ready < 0) {
        throw std::runtime_error(std::string("process TP pidfd poll: ") + strerror(errno));
    }
    for (size_t i = 0; i < items.size(); ++i) {
        if (HasExited(context.workers[i], items[i].revents)) {
            ThrowWorkerExited(context.workers[i], context);
        }
    }
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
     * 在持锁期间为本次事务借用全组退出句柄，清理前必须结束使用。
     * @return 包含 rank、socket、pidfd 及当前消息标识的上下文，不拥有句柄。
     */
    WaitContext WatchWorkers() const {
        WaitContext context;
        context.session = session;
        context.sequence = sequence;
        context.workers.reserve(children.size());
        for (const auto &child : children) {
            context.workers.push_back({child.info.rank, child.fd, child.pidfd});
        }
        return context;
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
        auto context = WatchWorkers();
        // Shutdown 的 Bye 后退出是预期行为，由 Stop 验证回复并回收。
        const auto *watch = kind == Kind::Shutdown ? nullptr : &context;
        for (auto &child : children) {
            try {
                Send(child.fd, {kind, session, sequence, payload}, deadline, watch);
            } catch (const WorkerExited &) {
                throw;
            } catch (const std::exception &e) {
                throw std::runtime_error("rank " + std::to_string(child.info.rank) + ": " + e.what());
            }
        }
        std::vector<Message> results;
        for (auto &child : children) {
            try {
                auto message = Receive(child.fd, deadline, watch);
                Validate(message, reply, session, sequence);
                results.push_back(std::move(message));
            } catch (const WorkerExited &) {
                throw;
            } catch (const std::exception &e) {
                throw std::runtime_error("rank " + std::to_string(child.info.rank) + ": " + e.what());
            }
        }
        if (watch) {
            CheckWorkerExits(*watch);
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
    std::vector<Qwen3RequestInfo> RequestStates(bool end, bool clearPrefix = false) {
        std::lock_guard<std::mutex> lock(mutex);
        if (state != ProcessGroupState::Ready || options.controlOnly) {
            throw std::runtime_error("Qwen3 request requires a ready CUDA process TP group");
        }
        try {
            auto replies = Exchange(clearPrefix ? Kind::ClearQwen3Prefix : end ? Kind::EndQwen3Request : Kind::QueryQwen3Request,
                clearPrefix ? Kind::Qwen3PrefixCleared : end ? Kind::Qwen3RequestEnded : Kind::Qwen3RequestState, {},
                Clock::now() + std::chrono::milliseconds(options.operationTimeoutMs));
            std::vector<Qwen3RequestInfo> result;
            for (const auto &reply : replies) {
                const auto &bytes = reply.payload;
                if (bytes.size() < 184 || Get32(bytes, 176) != 1 || Get32(bytes, 0) > 1 || Get32(bytes, 104) > 1 ||
                    Get32(bytes, 24) > 1 || Get32(bytes, 28) > 1 ||
                    Get32(bytes, 72) > 1 || Get32(bytes, 76) > 1) {
                    throw std::runtime_error("invalid Qwen3 request state header; rebuild coordinator and worker together");
                }
                Qwen3RequestInfo info{Get32(bytes, 0) != 0, Get32(bytes, 4), Get32(bytes, 8),
                                      Get32(bytes, 12), Get64(bytes, 16)};
                info.graph = {Get32(bytes, 24) != 0, Get32(bytes, 28) != 0,
                    Get64(bytes, 32), Get64(bytes, 40), Get64(bytes, 48), Get64(bytes, 56), Get64(bytes, 64)};
                info.allReduce = {Get32(bytes, 72) != 0, Get32(bytes, 76) != 0,
                    Get64(bytes, 80), Get64(bytes, 88), Get64(bytes, 96)};
                info.prefix = {Get32(bytes, 104) != 0, Get32(bytes, 108), Get64(bytes, 112),
                    Get64(bytes, 120), Get64(bytes, 128), Get64(bytes, 136), Get64(bytes, 144),
                    Get64(bytes, 152), Get64(bytes, 160), Get64(bytes, 168)};
                const uint32_t entries = Get32(bytes, 180);
                if (entries > 8 || entries != info.prefix.entries) {
                    throw std::runtime_error("invalid Qwen3 prefix directory count");
                }
                size_t offset = 184;
                for (uint32_t i = 0; i < entries; ++i) {
                    const uint32_t position = Get32(bytes, offset), length = Get32(bytes, offset + 4);
                    offset += 8;
                    if (!length || length > kQwen3MaxPromptTokens || position > 1048576 || length - 1 > 1048576 - position ||
                        offset > bytes.size() || length > (bytes.size() - offset) / 4) {
                        throw std::runtime_error("invalid Qwen3 prefix directory entry");
                    }
                    std::vector<uint32_t> ids;
                    for (uint32_t j = 0; j < length; ++j, offset += 4) ids.push_back(Get32(bytes, offset));
                    auto snapshot = std::make_pair(position, ids);
                    if (std::find(info.prefix.snapshots.begin(), info.prefix.snapshots.end(), snapshot) != info.prefix.snapshots.end()) {
                        throw std::runtime_error("duplicate Qwen3 prefix directory entry");
                    }
                    info.prefix.snapshots.push_back(std::move(snapshot));
                }
                if (offset != bytes.size()) throw std::runtime_error("trailing Qwen3 prefix directory bytes");
                const auto &prefix = info.prefix;
                const auto &ar = info.allReduce;
                const auto &g = info.graph;
                bool invalid = info.active ?
                    (end || !info.tokens || info.tokens > info.capacity || info.capacity > kQwen3MaxContextTokens ||
                     info.positionBegin > 1048576 || info.capacity - 1 > 1048576 - info.positionBegin ||
                     !info.deviceBytes || info.deviceBytes > 256ULL * 1024 * 1024) :
                    (info.tokens || info.capacity || info.positionBegin || info.deviceBytes);
                invalid = invalid || ar.enabled != options.customAllReduce || ar.active != ar.enabled ||
                    !ar.workspaceBytes || ar.workspaceBytes > 3ULL * 1024 * 1024 ||
                    (!ar.enabled && ar.customCalls) || g.enabled != options.cudaGraph ||
                    g.captures != g.warmups || g.replays < g.captures ||
                    g.active != (g.workspaceBytes != 0) || g.workspaceBytes > 256ULL * 1024 * 1024 ||
                    (g.active && (!info.active || !g.captures)) ||
                    (g.enabled ? g.eagerDecodes != 0 : (g.active || g.captures || g.replays || g.warmups));
                invalid = invalid || prefix.enabled != options.prefixCache || prefix.entries > 8 ||
                    prefix.deviceBytes > 64ULL * 1024 * 1024 || bool(prefix.entries) != bool(prefix.deviceBytes) ||
                    prefix.hits > prefix.lookups || prefix.reusedTokens < prefix.hits ||
                    prefix.lastReusedTokens >= kQwen3MaxPromptTokens || prefix.lastComputedTokens > kQwen3MaxPromptTokens ||
                    prefix.lastReusedTokens + prefix.lastComputedTokens > kQwen3MaxPromptTokens ||
                    (prefix.lookups && !prefix.lastComputedTokens) ||
                    (clearPrefix && (prefix.entries || prefix.deviceBytes)) ||
                    (!prefix.enabled && (prefix.entries || prefix.deviceBytes || prefix.lookups || prefix.hits ||
                        prefix.reusedTokens || prefix.lastReusedTokens || prefix.lastComputedTokens || prefix.evictions || prefix.clears));
                if (!result.empty()) {
                    const auto &first = result.front();
                    invalid = invalid || info.active != first.active || info.tokens != first.tokens ||
                              info.capacity != first.capacity || info.positionBegin != first.positionBegin ||
                              g.active != first.graph.active || g.captures != first.graph.captures ||
                              g.replays != first.graph.replays || g.warmups != first.graph.warmups ||
                              g.eagerDecodes != first.graph.eagerDecodes ||
                              ar.customCalls != first.allReduce.customCalls || ar.ncclCalls != first.allReduce.ncclCalls ||
                              ar.workspaceBytes != first.allReduce.workspaceBytes;
                    const auto &f = first.prefix;
                    invalid = invalid || prefix.entries != f.entries || prefix.lookups != f.lookups ||
                        prefix.hits != f.hits || prefix.reusedTokens != f.reusedTokens ||
                        prefix.lastReusedTokens != f.lastReusedTokens || prefix.lastComputedTokens != f.lastComputedTokens ||
                        prefix.evictions != f.evictions || prefix.clears != f.clears || prefix.snapshots != f.snapshots;
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
            "--parent-pid", std::to_string(getpid()), "--control-only", options.controlOnly ? "1" : "0",
            "--cuda-graph", options.cudaGraph ? "1" : "0",
            "--custom-allreduce", options.customAllReduce ? "1" : "0",
            "--prefix-cache", options.prefixCache ? "1" : "0"};
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
        options.shutdownTimeoutMs <= 0 || options.heartbeatIntervalMs <= 0 ||
        (options.controlOnly && (options.cudaGraph || options.customAllReduce || options.prefixCache)) ||
        (options.customAllReduce && options.devices.size() != 2)) {
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
        auto startupWatch = impl->WatchWorkers();
        std::set<std::vector<uint8_t>> deviceUuids;
        for (auto &child : impl->children) {
            auto hello = Receive(child.fd, deadline, &startupWatch);
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
            startupWatch.sequence = impl->sequence;
            Send(impl->children[0].fd, {Kind::CreateId, impl->session, impl->sequence, {}}, deadline, &startupWatch);
            auto reply = Receive(impl->children[0].fd, deadline, &startupWatch);
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

std::vector<float> ProcessGroup::CheckAllReduce(uint32_t count, uint32_t salt) {
    if (!count || count > 131072 || salt > 1024) throw std::invalid_argument("invalid AllReduce check shape/salt");
    std::lock_guard<std::mutex> lock(impl->mutex);
    if (impl->state != ProcessGroupState::Ready || impl->options.controlOnly) {
        throw std::runtime_error("NCCL check requires a ready CUDA process TP group");
    }
    try {
        std::vector<uint8_t> payload;
        Put32(payload, count);
        Put32(payload, salt);
        auto replies = impl->Exchange(Kind::AllReduce, Kind::AllReduceResult, payload,
                                      Clock::now() + std::chrono::milliseconds(impl->options.operationTimeoutMs));
        std::vector<float> values;
        for (auto &reply : replies) {
            if (reply.payload.size() != size_t(count) * 4) {
                throw std::runtime_error("invalid NCCL check result size");
            }
            for (uint32_t i = 0; i < count; ++i) {
                uint32_t bits = Get32(reply.payload, size_t(i) * 4);
                float value;
                static_assert(sizeof(value) == sizeof(bits), "FP32 required");
                memcpy(&value, &bits, sizeof(value));
                values.push_back(value);
            }
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
    if (!tokens || tokens > kQwen3MaxPromptTokens || !hidden || uint64_t(tokens) * hidden != input.size() ||
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
    if (tokenIds.empty() || tokenIds.size() > kQwen3MaxPromptTokens || positionBegin > 1048576 || timeoutMs <= 0) {
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
    if (tokenIds.empty() || tokenIds.size() > kQwen3MaxPromptTokens || !capacity || capacity > kQwen3MaxContextTokens ||
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

void ProcessGroup::ClearQwen3PrefixCache() {
    impl->RequestStates(false, true);
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
