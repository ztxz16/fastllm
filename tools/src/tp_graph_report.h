#pragma once

#include "distributed/process_group.h"
#include "json11.hpp"
#include <stdexcept>

/**
 * 从 worker 查询实际 Graph 路径及累计次数；不以启动参数代替执行证据。
 * @param group 已就绪的 CUDA 进程组，允许尚未加载或已经卸载模型。
 * @return 按 rank 排列的状态；通信失败或状态不一致时抛异常并终止整组。
 */
inline json11::Json::array GraphState(fastllm::distributed::ProcessGroup &group) {
    json11::Json::array states;
    for (const auto &info : group.Qwen3Requests()) {
        const auto &g = info.graph;
        // json11 的 double 会输出带小数点的数字；计数/字节必须走整数重载，
        // 否则 Python 解码为 float，启动阶段的严格协议检查就会拒绝。
        states.push_back(json11::Json::object{{"enabled", g.enabled}, {"active", g.active},
            {"captures", static_cast<long long>(g.captures)}, {"replays", static_cast<long long>(g.replays)},
            {"warmups", static_cast<long long>(g.warmups)}, {"eager_decodes", static_cast<long long>(g.eagerDecodes)},
            {"workspace_bytes", static_cast<long long>(g.workspaceBytes)},
            {"allreduce", json11::Json::object{{"enabled", info.allReduce.enabled}, {"active", info.allReduce.active},
                {"custom_calls", static_cast<long long>(info.allReduce.customCalls)},
                {"nccl_calls", static_cast<long long>(info.allReduce.ncclCalls)},
                {"workspace_bytes", static_cast<long long>(info.allReduce.workspaceBytes)}}},
            {"prefix_cache", json11::Json::object{{"enabled", info.prefix.enabled},
                {"entries", static_cast<long long>(info.prefix.entries)},
                {"device_bytes", static_cast<long long>(info.prefix.deviceBytes)},
                {"lookups", static_cast<long long>(info.prefix.lookups)},
                {"hits", static_cast<long long>(info.prefix.hits)},
                {"reused_tokens", static_cast<long long>(info.prefix.reusedTokens)},
                {"last_reused_tokens", static_cast<long long>(info.prefix.lastReusedTokens)},
                {"last_computed_tokens", static_cast<long long>(info.prefix.lastComputedTokens)},
                {"evictions", static_cast<long long>(info.prefix.evictions)},
                {"clears", static_cast<long long>(info.prefix.clears)}}}});
    }
    return states;
}

/**
 * 解析原生命令行 Graph 开关，启动 worker 前拒绝非法值。
 * @param value 只接受字符串 0 或 1。
 * @return 是否开启 Graph；其他值抛 invalid_argument。
 */
inline bool ParseCudaGraph(const std::string &value) {
    if (value != "0" && value != "1") throw std::invalid_argument("cuda-graph must be 0 or 1");
    return value == "1";
}

/** 严格解析自定义规约开关；不能把其他字符串静默解释为关闭。 */
inline bool ParseCustomAllReduce(const std::string &value) {
    if (value != "0" && value != "1") throw std::invalid_argument("custom-allreduce must be 0 or 1");
    return value == "1";
}

/** 严格解析前缀快照开关；默认关闭，拒绝未知值。 */
inline bool ParsePrefixCache(const std::string &value) {
    if (value != "0" && value != "1") throw std::invalid_argument("prefix-cache must be 0 or 1");
    return value == "1";
}
