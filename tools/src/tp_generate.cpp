#include "distributed/qwen3_generation.h"
#include "json11.hpp"
#include "tp_graph_report.h"

#include <cmath>
#include <cerrno>
#include <arpa/inet.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <sys/socket.h>
#include <unistd.h>

using namespace fastllm::distributed;
using json11::Json;
namespace fs = std::filesystem;
namespace {
/** 将 JSON 字段读取为 uint32；拒绝隐式类型转换和截断。 */
uint32_t Integer(const Json &value) {
    double x = value.number_value();
    if (!value.is_number() || !std::isfinite(x) || x < 0 || x > 4294967295.0 || x != std::floor(x)) {
        throw std::runtime_error("generation field must be uint32");
    }
    return uint32_t(x);
}

/** 读取有限浮点采样参数；范围由公开采样接口统一验证。 */
double Number(const Json &value) {
    if (!value.is_number() || !std::isfinite(value.number_value())) {
        throw std::runtime_error("generation field must be finite number");
    }
    return value.number_value();
}

/** 读取 token ID 数组；只负责 JSON 类型，词表范围由模型接口检查。 */
std::vector<uint32_t> Ids(const Json &value) {
    if (!value.is_array()) {
        throw std::runtime_error("generation IDs must be array");
    }
    std::vector<uint32_t> ids;
    for (const auto &item : value.array_items()) {
        ids.push_back(Integer(item));
    }
    return ids;
}

/** 读取真实 KV 状态，生成完成和卸载后都要求完全释放。 */
Json::array EmptyCache(ProcessGroup &group) {
    Json::array states;
    for (const auto &info : group.Qwen3Requests()) {
        if (info.active || info.tokens || info.capacity || info.positionBegin || info.deviceBytes ||
            info.graph.active || info.graph.workspaceBytes) {
            throw std::runtime_error("generation leaked request KV");
        }
        states.push_back(Json::object{{"active", info.active}, {"tokens", int(info.tokens)},
            {"capacity", int(info.capacity)}, {"device_bytes", double(info.deviceBytes)}});
    }
    return states;
}

/**
 * 通过正式生成接口处理单个请求；单次命令与常驻会话共用采样和 KV 生命周期。
 * @param group 已加载模型、无活动 KV 的进程组。
 * @param request JSON 对象，token_ids 为一维无符号整数数组。
 * @param trace 是否记录有界 FP32 [vocab] logits。
 * @param report 写入实际输出和缓存状态，失败时保留已完成步骤。
 * @param cancelled 可选取消探针，沿用正式生成接口的同步边界和 KV 清理。
 * @param onToken 可选同步 Token 通知，仅发送已采样的 ID；异常沿用 KV 清理路径。
 * @return 正常返回表示生成及 KV 释放成功；非法输入或执行失败抛异常。
 */
void Generate(ProcessGroup &group, const Json &request, bool trace, Json::object &report,
              const std::function<bool()> &cancelled = {},
              const std::function<void(uint32_t)> &onToken = {}) {
    if (!request.is_object()) throw std::runtime_error("generation request must be object");
    const std::set<std::string> fields{"token_ids", "max_new_tokens", "temperature", "top_k", "top_p", "seed", "eos_token_ids"};
    for (const auto &item : request.object_items()) {
        if (!fields.count(item.first)) throw std::runtime_error("unknown generation field: " + item.first);
    }
    auto prompt = Ids(request["token_ids"]);
    Qwen3GenerationOptions sampling;
    const auto &object = request.object_items();
    if (object.count("max_new_tokens")) sampling.maxNewTokens = Integer(request["max_new_tokens"]);
    if (object.count("temperature")) sampling.temperature = Number(request["temperature"]);
    if (object.count("top_k")) sampling.topK = Integer(request["top_k"]);
    if (object.count("top_p")) sampling.topP = Number(request["top_p"]);
    if (object.count("seed")) sampling.seed = Integer(request["seed"]);
    if (object.count("eos_token_ids")) sampling.eosTokenIds = Ids(request["eos_token_ids"]);
    report["request"] = request;
    report["trace_logits"] = trace;

    Json::array steps;
    uint64_t recorded = 0;
    try {
        auto result = GenerateQwen3(group, prompt, sampling, [&](uint32_t token, const std::vector<float> &logits) {
            Json::object step{{"token_id", static_cast<long long>(token)}};
            if (trace) {
                if (logits.size() > 8388608ULL - recorded) {
                    throw std::runtime_error("generation trace exceeds 8M logit elements");
                }
                Json::array values;
                values.reserve(logits.size());
                for (auto value : logits) values.push_back(value);
                step["logits"] = std::move(values);
                recorded += logits.size();
            }
            steps.push_back(std::move(step));
            if (onToken) onToken(token);
        }, cancelled);
        Json::array ids;
        for (auto token : result.tokenIds) ids.push_back(static_cast<long long>(token));
        report["token_ids"] = std::move(ids);
        report["finish_reason"] = result.finishReason;
        report["cache_after_generate"] = EmptyCache(group);
        report["graph"] = GraphState(group);
    } catch (...) {
        report["steps"] = std::move(steps);
        throw;
    }
    report["steps"] = std::move(steps);
}

/**
 * 常驻协调器的独立控制 socket；stdout/stderr 只承载日志，避免 NCCL 输出污染协议。
 * 每条 JSON 前置网络字节序 uint32 长度；输入上限 1 MiB，断连触发整组关闭。
 */
class SessionSocket {
    int fd;
public:
    /** 接管父进程传入的 socket，并禁止 worker 继承；描述符非法时拒绝启动。 */
    explicit SessionSocket(const std::string &value) : fd(-1) {
        if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
            throw std::runtime_error("invalid session fd");
        }
        int candidate = std::stoi(value), type = 0;
        socklen_t size = sizeof(type);
        if (candidate < 3 || getsockopt(candidate, SOL_SOCKET, SO_TYPE, &type, &size) != 0 ||
            type != SOCK_STREAM || fcntl(candidate, F_SETFD, FD_CLOEXEC) != 0) {
            throw std::runtime_error("session fd must be a stream socket");
        }
        fd = candidate;
    }
    ~SessionSocket() { if (fd >= 0) close(fd); }

    /** 非阻塞检查控制端是否退出；不消费下一条消息，也不改变 socket 阻塞模式。 */
    bool Disconnected() const {
        char byte;
        ssize_t count;
        do { count = recv(fd, &byte, 1, MSG_PEEK | MSG_DONTWAIT); } while (count < 0 && errno == EINTR);
        if (count < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
            throw std::runtime_error("session connection check failed");
        }
        return count == 0;
    }

    /** 精确读取指定字节数；仅帧头起始处允许干净 EOF，其余断连均拒绝。 */
    bool ReadBytes(char *buffer, size_t size, bool allowEof = false) {
        size_t offset = 0;
        while (offset < size) {
            ssize_t count = recv(fd, buffer + offset, size - offset, 0);
            if (count < 0 && errno == EINTR) continue;
            if (count < 0) throw std::runtime_error("session receive failed");
            if (count == 0) {
                if (offset == 0 && allowEof) return false;
                throw std::runtime_error("truncated session request");
            }
            offset += size_t(count);
        }
        return true;
    }

    /** 读取有界 JSON 对象；干净 EOF 返回 false，截断帧和非法 JSON 抛异常。 */
    bool Read(Json &message) {
        uint32_t header = 0;
        if (!ReadBytes(reinterpret_cast<char*>(&header), sizeof(header), true)) return false;
        uint32_t length = ntohl(header);
        if (!length || length > 1024 * 1024) throw std::runtime_error("session request exceeds 1 MiB or is empty");
        std::string text(length, '\0'), error;
        ReadBytes(&text[0], text.size());
        message = Json::parse(text, error);
        if (!error.empty() || !message.is_object()) throw std::runtime_error("invalid session JSON");
        return true;
    }

    /** 完整发送缓冲区，禁用 SIGPIPE；客户端断连由外层统一回收所有 worker。 */
    void WriteBytes(const char *data, size_t size) {
        size_t offset = 0;
        while (offset < size) {
            ssize_t count = send(fd, data + offset, size - offset, MSG_NOSIGNAL);
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) throw std::runtime_error("session send failed");
            offset += size_t(count);
        }
    }

    /** 发送带长度头的 JSON，兼容仓库 json11 的多行输出；响应最多 512 MiB。 */
    void Write(const Json &message) {
        std::string text = message.dump();
        if (text.size() > 512ULL * 1024 * 1024) throw std::runtime_error("session response exceeds 512 MiB");
        uint32_t header = htonl(uint32_t(text.size()));
        WriteBytes(reinterpret_cast<const char*>(&header), sizeof(header));
        WriteBytes(text.data(), text.size());
    }
};

/**
 * 独立数据报取消通道；消息只携带网络字节序 uint32 请求 ID。
 * 旧请求的迟到取消直接丢弃，不能取消下一请求；消息格式错误使整组失败。
 */
class SessionCancellation {
    int fd = -1;
    bool valid = true;
public:
    /** 接管父进程传入的 datagram socket，禁止 worker 继承。 */
    explicit SessionCancellation(const std::string &value) {
        if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
            throw std::runtime_error("invalid cancellation fd");
        }
        int candidate = std::stoi(value), type = 0;
        socklen_t size = sizeof(type);
        if (candidate < 3 || getsockopt(candidate, SOL_SOCKET, SO_TYPE, &type, &size) != 0 ||
            type != SOCK_DGRAM || fcntl(candidate, F_SETFD, FD_CLOEXEC) != 0) {
            throw std::runtime_error("cancellation fd must be a datagram socket");
        }
        fd = candidate;
    }
    ~SessionCancellation() { if (fd >= 0) close(fd); }

    /** 协议损坏后不可继续复用会话。 */
    bool Valid() const { return valid; }

    /** 消费已到达的取消 ID；只有当前 ID 命中时返回 true，未来 ID 属于协议错误。 */
    bool Requested(uint32_t id) {
        bool cancelled = false;
        for (;;) {
            uint32_t message = 0;
            ssize_t count = recv(fd, &message, sizeof(message), MSG_DONTWAIT | MSG_TRUNC);
            if (count < 0 && errno == EINTR) continue;
            if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return cancelled;
            if (count != static_cast<ssize_t>(sizeof(message)) || ntohl(message) > id || ntohl(message) == 0) {
                valid = false;
                throw std::runtime_error("invalid cancellation message");
            }
            cancelled = cancelled || ntohl(message) == id;
        }
    }
};

/**
 * 在同一进程组串行处理多次生成；权重只加载一次，每次生成独立释放 KV。
 * @param group 已加载模型的进程组；退出时卸载并停止。
 * @param socket 独立控制通道，id 必须从 1 开始连续递增。
 * @param trace 是否记录与单请求入口相同的 logits。
 * @param report 会话启动信息与最终清理状态。
 * @param cancellation 可选取消通道；不与控制帧混用，迟到取消按请求 ID 隔离。
 * @return 正常返回表示 EOF 或 shutdown 已完成清理；致命错误抛异常并交由 main 回收。
 */
void Serve(ProcessGroup &group, SessionSocket &socket, bool trace, Json::object &report,
           SessionCancellation *cancellation) {
    auto ready = report;
    ready["event"] = "ready";
    ready["protocol"] = 1;
    ready["status"] = "PASS";
    ready["model_load_count"] = 1;
    ready["cache"] = EmptyCache(group);
    ready["graph"] = GraphState(group);
    ready["cancellation"] = cancellation ? "request_id_datagram" : "none";
    ready["token_events"] = true;
    socket.Write(ready);
    uint64_t expected = 1;
    bool shutdown = false;
    Json message;
    while (socket.Read(message)) {
        uint32_t id = Integer(message["id"]);
        if (id != expected++) throw std::runtime_error("out-of-order session request id");
        const auto op = message["op"].string_value();
        const std::set<std::string> fields = op == "generate" ?
            std::set<std::string>{"id", "op", "request", "stream_tokens"} : std::set<std::string>{"id", "op"};
        for (const auto &item : message.object_items()) {
            if (!fields.count(item.first)) throw std::runtime_error("unknown session field");
        }
        if (op == "shutdown") { shutdown = true; break; }
        if (op != "generate") throw std::runtime_error("unknown session operation");
        if (message.object_items().count("stream_tokens") && !message["stream_tokens"].is_bool()) {
            throw std::runtime_error("stream_tokens must be boolean");
        }
        bool streamTokens = message["stream_tokens"].bool_value();
        uint32_t tokenIndex = 0;

        Json::object result{{"event", "result"}, {"id", static_cast<long long>(id)},
            {"status", "FAIL"}, {"backend", report["backend"]}, {"workers", report["workers"]},
            {"execution", report["execution"]}};
        try {
            Generate(group, message["request"], trace, result, [&]() {
                if (socket.Disconnected()) throw std::runtime_error("session client disconnected");
                return cancellation && cancellation->Requested(id);
            }, [&](uint32_t token) {
                if (streamTokens) {
                    socket.Write(Json::object{{"event", "token"}, {"id", static_cast<long long>(id)},
                        {"index", static_cast<long long>(tokenIndex++)}, {"token_id", static_cast<long long>(token)}});
                }
            });
            result["status"] = "PASS";
            result["reusable"] = true;
        } catch (const std::exception &error) {
            result["error"] = error.what();
            if (dynamic_cast<const Qwen3GenerationCancelled*>(&error)) result["cancelled"] = true;
            bool reusable = false;
            if (group.State() == ProcessGroupState::Ready && (!cancellation || cancellation->Valid())) {
                try {
                    result["cache_after_error"] = EmptyCache(group);
                    result["graph"] = GraphState(group);
                    reusable = true;
                } catch (...) {}
            }
            result["reusable"] = reusable;
            socket.Write(result);
            if (!reusable) throw;
            continue;
        }
        socket.Write(result);
    }

    group.UnloadQwen3();
    report["cache_after_unload"] = EmptyCache(group);
    report["graph_after_unload"] = GraphState(group);
    group.Stop();
    if (!group.LastError().empty() || !group.Workers().empty()) {
        throw std::runtime_error("session shutdown failed: " + group.LastError());
    }
    report["shutdown_complete"] = true;
    report["status"] = "PASS";
    if (shutdown) {
        auto reply = report;
        reply["event"] = "shutdown";
        reply["id"] = static_cast<long long>(expected - 1);
        socket.Write(reply);
    }
}
}

/**
 * 从 token ID 文件或常驻控制通道执行真实生成；文本和聊天模板由 Python 前端负责。
 * @param argc/argv --worker、--model 必填，--request 与 --session-fd 二选一，--devices 默认 0,1；
 *                 --trace-logits 1 开启有界诊断，默认不保留词表历史。
 * @return 0 表示生成、KV 释放和 worker 回收均成功；否则 JSON 含错误及部分输出。
 */
int main(int argc, char **argv) {
    ProcessGroup group;
    Json::object report{{"status", "FAIL"}, {"backend", "qwen3_process_generate_cublas_nccl"}};
    std::unique_ptr<SessionSocket> session;
    std::unique_ptr<SessionCancellation> cancellation;
    try {
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2) {
            const std::string key = argv[i];
            if (i + 1 == argc || (key != "--worker" && key != "--model" && key != "--request" &&
                key != "--devices" && key != "--trace-logits" && key != "--session-fd" &&
                key != "--cancel-fd" && key != "--cuda-graph" && key != "--custom-allreduce") || !args.emplace(key, argv[i+1]).second) {
                throw std::runtime_error("invalid/duplicate generation argument");
            }
        }
        bool trace = false;
        if (args.count("--trace-logits")) {
            if (args.at("--trace-logits") != "1" && args.at("--trace-logits") != "0") {
                throw std::runtime_error("trace-logits must be 0 or 1");
            }
            trace = args.at("--trace-logits") == "1";
        }
        if (args.count("--request") == args.count("--session-fd")) {
            throw std::runtime_error("specify exactly one of --request or --session-fd");
        }
        if (args.count("--cancel-fd")) {
            if (!args.count("--session-fd")) throw std::runtime_error("cancel-fd requires session-fd");
            cancellation.reset(new SessionCancellation(args.at("--cancel-fd")));
        }
        Json request;
        if (args.count("--session-fd")) {
            session.reset(new SessionSocket(args.at("--session-fd")));
        } else {
            auto path = fs::path(args.at("--request"));
            if (fs::file_size(path) > 1024 * 1024) {
                throw std::runtime_error("generation request exceeds 1 MiB");
            }
            std::ifstream file(path);
            std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>()), error;
            request = Json::parse(text, error);
            if (!file || !error.empty() || !request.is_object()) {
                throw std::runtime_error("invalid generation JSON: " + error);
            }
        }

        ProcessGroupOptions options;
        options.cudaGraph = args.count("--cuda-graph") && ParseCudaGraph(args.at("--cuda-graph"));
        options.customAllReduce = args.count("--custom-allreduce") && ParseCustomAllReduce(args.at("--custom-allreduce"));
        report["execution"] = options.cudaGraph ? "decode_cuda_graph" : "eager";
        options.workerPath = fs::absolute(args.at("--worker")).string();
        options.operationTimeoutMs = options.shutdownTimeoutMs = 30000;
        auto devices = args.count("--devices") ? args.at("--devices") : "0,1";
        if (devices.empty() || devices.back() == ',') throw std::runtime_error("invalid devices");
        std::istringstream stream(devices);
        for (std::string item; std::getline(stream, item, ',');) {
            if (item.empty() || item.find_first_not_of("0123456789") != std::string::npos) {
                throw std::runtime_error("invalid device index");
            }
            options.devices.push_back(std::stoi(item));
        }
        group.Start(options);
        Json::array workers;
        for (const auto &worker : group.Workers()) {
            workers.push_back(Json::object{{"rank", worker.rank}, {"device", worker.device}, {"pid", worker.pid},
                {"gpu", worker.deviceName}, {"cuda_runtime_version", worker.cudaRuntimeVersion}, {"nccl_version", worker.ncclVersion}});
        }
        report["workers"] = workers;
        group.LoadQwen3(fs::absolute(args.at("--model")).string());
        if (session) {
            Serve(group, *session, trace, report, cancellation.get());
        } else {
            Generate(group, request, trace, report);
            group.UnloadQwen3();
            report["cache_after_unload"] = EmptyCache(group);
            report["graph_after_unload"] = GraphState(group);
            group.Stop();
            if (!group.LastError().empty() || !group.Workers().empty()) {
                throw std::runtime_error("generation shutdown failed: " + group.LastError());
            }
            report["shutdown_complete"] = true;
            report["status"] = "PASS";
        }
    } catch (const std::exception &error) {
        report["status"] = "FAIL";
        report["error"] = error.what();
        report["group_failed"] = group.State() == ProcessGroupState::Failed;
        auto previous = group.LastError();
        try {
            bool cacheClean = true;
            if (group.State() == ProcessGroupState::Ready) {
                try {
                    report["cache_after_error"] = EmptyCache(group);
                } catch (const std::exception &cleanup) {
                    cacheClean = false;
                    report["request_cleanup_error"] = cleanup.what();
                }
            }
            group.Stop();
            if (!cacheClean || group.State() != ProcessGroupState::Stopped || !group.Workers().empty() || group.LastError() != previous) {
                throw std::runtime_error("generation cleanup failed: " + group.LastError());
            }
            report["shutdown_complete"] = true;
        } catch (const std::exception &cleanup) {
            report["shutdown_complete"] = false;
            report["cleanup_error"] = cleanup.what();
        }
    }
    if (!report.count("steps")) report["steps"] = Json::array{};
    std::cout << Json(report).dump() << std::endl;
    return report["status"].string_value() == "PASS" ? 0 : 1;
}
