#ifndef FASTLLM_DISTRIBUTED_PD_NATIVE_H
#define FASTLLM_DISTRIBUTED_PD_NATIVE_H

#include <array>
#include <cstdint>
#include <vector>

namespace fastllm { namespace distributed {

enum class PDAction : uint32_t { Prepare = 1, Export = 2, Commit = 3, Release = 4, RestorePrefix = 5, RestoreLocal = 6, Initialize = 7 };

/** 本地 worker 生成的真实 KV 范围；IPC 句柄不能替代请求身份和传输授权。 */
struct PDRegion {
    uint32_t layer = 0, component = 0, tokenBegin = 0, tokenEnd = 0;
    uint32_t headBegin = 0, headEnd = 0, headDim = 0, pageSize = 0;
    uint64_t allocationBytes = 0, offset = 0, tokenStride = 0, headStride = 0;
    std::array<uint8_t, 64> memory{};
    std::array<uint8_t, 16> device{};
};

/** 已经本地授权的控制调用；key 在一次 worker 启动内唯一，不能跨启动复用。 */
struct PDCommand {
    PDAction action = PDAction::Export;
    uint64_t key = 0;
    uint32_t tokens = 0, capacity = 0, position = 0;
    int32_t targetRank = -1; // -1 表示整个 TP 组；Attention DP 只选择一个 rank。
    std::vector<uint32_t> tokenIds;
};

} }
#endif
