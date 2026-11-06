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
        states.push_back(json11::Json::object{{"enabled", g.enabled}, {"active", g.active},
            {"captures", double(g.captures)}, {"replays", double(g.replays)},
            {"warmups", double(g.warmups)}, {"eager_decodes", double(g.eagerDecodes)},
            {"workspace_bytes", double(g.workspaceBytes)}});
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
