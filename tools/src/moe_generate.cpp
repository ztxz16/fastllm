#include "distributed/qwen3_generation.h"
#include "tp_graph_report.h"
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <fcntl.h>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>

using namespace fastllm::distributed;
using json11::Json;

namespace {
/** 读取严格 uint32；在启动或提交 worker 之前拒绝隐式转换。 */
uint32_t Integer(const Json &value) {
    const double x = value.number_value();
    if (!value.is_number() || !std::isfinite(x) || x < 0 || x > 4294967295.0 || std::floor(x) != x) {
        throw std::invalid_argument("MoE protocol expects uint32");
    }
    return uint32_t(x);
}

/** 独立有界帧通道，stdout/stderr 仅用于日志；断连由 main 回收整个进程组。 */
class Channel {
    int fd;
public:
    explicit Channel(int value) : fd(value) {
        int type = 0;
        socklen_t size = sizeof(type);
        if (fd < 3 || getsockopt(fd, SOL_SOCKET, SO_TYPE, &type, &size) || type != SOCK_STREAM ||
            fcntl(fd, F_SETFD, FD_CLOEXEC)) throw std::runtime_error("invalid MoE session socket");
    }
    ~Channel() { close(fd); }

    /** 精确读写一段帧数据；帧中断、超时或对端退出抛异常，不能继续复用。 */
    bool Transfer(char *data, size_t size, bool sendData, bool eof = false) {
        size_t offset = 0;
        while (offset < size) {
            ssize_t count = sendData ? send(fd, data + offset, size - offset, MSG_NOSIGNAL) :
                                      recv(fd, data + offset, size - offset, 0);
            if (count < 0 && errno == EINTR) continue;
            if (!count && !offset && eof) return false;
            if (count <= 0) throw std::runtime_error("MoE session disconnected or truncated");
            offset += size_t(count);
        }
        return true;
    }

    /** 读取不超过 1 MiB 的 JSON 对象；正常 EOF 返回 false。 */
    bool Read(Json &message) {
        uint32_t header = 0;
        if (!Transfer(reinterpret_cast<char*>(&header), 4, false, true)) return false;
        const auto length = ntohl(header);
        if (!length || length > 1024*1024) throw std::runtime_error("invalid MoE request frame size");
        std::string data(length, '\0'), error;
        Transfer(&data[0], length, false);
        message = Json::parse(data, error);
        if (!error.empty() || !message.is_object()) throw std::runtime_error("invalid MoE JSON");
        return true;
    }

    /** 发送结果；诊断 logits 也受 128 MiB 帧上限约束。 */
    void Write(const Json &message) {
        auto data = message.dump();
        if (data.size() > 128ULL*1024*1024) throw std::runtime_error("MoE reply exceeds frame limit");
        uint32_t header = htonl(data.size());
        Transfer(reinterpret_cast<char*>(&header), 4, true);
        Transfer(&data[0], data.size(), true);
    }
};

/**
 * 返回真实请求及设备状态，额外报告持久 KV 预留，不能把预留冒充活动请求。
 * @param group 已就绪的进程组。
 * @return 每个 Attention rank 的 KV、Graph 和前缀快照证据。
 */
Json::array States(ProcessGroup &group) {
    auto states = group.Qwen3Requests();
    auto graph = GraphState(states);
    for (size_t i = 0; i < states.size(); ++i) {
        auto item = graph[i].object_items();
        item["request_active"] = states[i].active;
        item["request_tokens"] = static_cast<long long>(states[i].tokens);
        item["request_capacity"] = static_cast<long long>(states[i].capacity);
        item["request_kv_bytes"] = static_cast<long long>(states[i].deviceBytes);
        item["reserved_kv_bytes"] = static_cast<long long>(states[i].reservedKVBytes);
        graph[i] = item;
    }
    return graph;
}

/**
 * 串行处理联合轮次；准入选择由 Python 队列负责，前向、通信及采样在原生进程执行。
 * @param group 已加载完整 MoE 权重的进程组。
 * @param channel 一个模型对应的独立控制通道。
 * @param world Attention/EP rank 数，最多四个。
 * @param graph 是否允许全组 Decode Graph。
 * @param timeout 每轮包含全部 worker 的截止毫秒数。
 */
void Serve(ProcessGroup &group, Channel &channel, size_t world, bool graph, int timeout) {
    std::vector<Qwen3GenerationOptions> sampling(world);
    std::vector<std::mt19937> random(world);
    Json message;
    uint32_t expected = 1;
    while (channel.Read(message)) {
        const auto id = Integer(message["id"]);
        if (id != expected++) throw std::runtime_error("MoE round ID mismatch");
        auto op = message["op"].string_value();
        if (op == "shutdown") {
            group.EndQwen3Request();
            group.UnloadQwen3();
            auto states = States(group);
            group.Stop();
            if (!group.LastError().empty() || !group.Workers().empty()) throw std::runtime_error("MoE shutdown incomplete");
            channel.Write(Json::object{{"id", static_cast<long long>(id)}, {"status", "PASS"}, {"states", states}, {"shutdown_complete", true}});
            return;
        }
        if (op == "clear_prefix_cache") {
            group.ClearQwen3PrefixCache();
            channel.Write(Json::object{{"id", static_cast<long long>(id)}, {"status", "PASS"}, {"states", States(group)}});
            continue;
        }
        if (op != "step" || !message["steps"].is_array() || message["steps"].array_items().size() != world ||
            !message["trace"].is_bool()) throw std::runtime_error("invalid MoE round operation");
        std::vector<QwenMoeStep> steps(world);
        bool useGraph = graph, hasDecode = false;
        for (size_t rank = 0; rank < world; ++rank) {
            const auto &input = message["steps"][rank];
            auto &step = steps[rank];
            step.action = static_cast<QwenMoeAction>(Integer(input["action"]));
            if (!input["token_ids"].is_array()) throw std::runtime_error("missing MoE token array");
            for (const auto &token : input["token_ids"].array_items()) step.tokenIds.push_back(Integer(token));
            if (step.action == QwenMoeAction::Begin) {
                useGraph = false;
                step.capacity = Integer(input["capacity"]);
                step.positionBegin = Integer(input["position_begin"]);
                const auto &params = input["sampling"];
                auto &s = sampling[rank];
                s.topK = Integer(params["top_k"]);
                s.seed = Integer(params["seed"]);
                if (!params["temperature"].is_number() || !params["top_p"].is_number()) throw std::runtime_error("invalid sampling types");
                s.temperature = params["temperature"].number_value();
                s.topP = params["top_p"].number_value();
                if (!std::isfinite(s.temperature) || s.temperature < 0 || !std::isfinite(s.topP) || s.topP <= 0 || s.topP > 1) {
                    throw std::runtime_error("invalid MoE sampling");
                }
                random[rank].seed(s.seed);
            }
            hasDecode = hasDecode || step.action == QwenMoeAction::Decode;
        }
        useGraph = useGraph && hasDecode;
        const auto start = std::chrono::steady_clock::now();
        auto logits = group.StepQwenMoe(steps, useGraph, timeout);
        const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        Json::array tokens, traces;
        for (size_t rank = 0; rank < world; ++rank) {
            if (logits[rank].empty()) { tokens.push_back(Json()); traces.push_back(Json::array{}); continue; }
            const double uniform = double(random[rank]()) / 4294967296.0;
            tokens.push_back(static_cast<long long>(SampleQwen3(logits[rank], sampling[rank], uniform)));
            Json::array values;
            if (message["trace"].bool_value()) for (float x : logits[rank]) values.push_back(x);
            traces.push_back(values);
        }
        channel.Write(Json::object{{"id", static_cast<long long>(id)}, {"status", "PASS"},
            {"token_ids", tokens}, {"logits", traces}, {"states", States(group)},
            {"execution", useGraph ? "decode_cuda_graph" : "eager"}, {"round_ms", elapsed}});
    }
    group.EndQwen3Request();
    group.UnloadQwen3();
    group.Stop();
}
}

/** 启动一个原生 Attention DP/EP 协调器；失败退出前回收所有 worker，不做自动重试。 */
int main(int argc, char **argv) {
    ProcessGroup group;
    try {
        std::map<std::string, std::string> args;
        const std::set<std::string> allowed{"--model", "--worker", "--devices", "--session-fd", "--cuda-graph", "--prefix-cache", "--timeout-ms"};
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc || !allowed.count(argv[i]) || !args.emplace(argv[i], argv[i+1]).second) throw std::runtime_error("invalid MoE argument");
        }
        auto parse = [&](const std::string &key) {
            const auto &s = args.at(key);
            if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos) throw std::runtime_error("invalid integer argument");
            return std::stoi(s);
        };
        Channel channel(parse("--session-fd"));
        ProcessGroupOptions options;
        options.workerPath = std::filesystem::absolute(args.at("--worker")).string();
        options.cudaGraph = ParseCudaGraph(args.at("--cuda-graph"));
        options.prefixCache = ParsePrefixCache(args.at("--prefix-cache"));
        const int timeout = parse("--timeout-ms");
        if (timeout <= 0) throw std::runtime_error("invalid MoE timeout");
        options.operationTimeoutMs = timeout;
        auto devices = args.at("--devices");
        if (devices.empty() || devices.back() == ',') throw std::runtime_error("invalid MoE devices");
        std::istringstream input(devices);
        for (std::string s; std::getline(input, s, ',');) {
            if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos) throw std::runtime_error("invalid MoE device");
            options.devices.push_back(std::stoi(s));
        }
        if (options.devices.size() != 1 && options.devices.size() != 2 && options.devices.size() != 4) throw std::runtime_error("MoE requires 1/2/4 ranks");
        group.Start(options);
        auto loaded = group.LoadQwenMoe(std::filesystem::absolute(args.at("--model")).string(), 1ULL<<30, 12ULL<<30, timeout);
        Json::array workers;
        for (const auto &w : group.Workers()) workers.push_back(Json::object{{"rank", w.rank}, {"device", w.device},
            {"pid", w.pid}, {"gpu", w.deviceName}, {"cuda_runtime_version", w.cudaRuntimeVersion}, {"nccl_version", w.ncclVersion},
            {"weight_device_bytes", static_cast<long long>(loaded.at(w.rank).deviceBytes)}});
        channel.Write(Json::object{{"event", "ready"}, {"status", "PASS"}, {"protocol", 1},
            {"backend", "qwen_moe_attention_dp_ep"}, {"model_load_count", 1}, {"workers", workers},
            {"cancellation", "round_boundary"}, {"token_events", true}, {"states", States(group)}});
        Serve(group, channel, options.devices.size(), options.cudaGraph, timeout);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "MoE coordinator failed: " << error.what() << std::endl;
        try { group.Stop(); } catch (const std::exception &cleanup) { std::cerr << cleanup.what() << std::endl; }
        return 1;
    }
}
