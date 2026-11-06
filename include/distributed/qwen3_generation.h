#ifndef FASTLLM_DISTRIBUTED_QWEN3_GENERATION_H
#define FASTLLM_DISTRIBUTED_QWEN3_GENERATION_H

#include "distributed/process_group.h"
#include <functional>
#include <stdexcept>

namespace fastllm {
namespace distributed {

struct Qwen3GenerationOptions {
    uint32_t maxNewTokens = 32;
    double temperature = 0.0;
    uint32_t topK = 0;
    double topP = 1.0;
    uint32_t seed = 0;
    std::vector<uint32_t> eosTokenIds;
};

struct Qwen3GenerationResult {
    // 包含实际采样出的 EOS；最后一个输出 token 不再送入 Decode。
    std::vector<uint32_t> tokenIds;
    std::string finishReason;
};

class Qwen3GenerationCancelled : public std::runtime_error {
public:
    Qwen3GenerationCancelled() : std::runtime_error("Qwen3 generation cancelled") {}
};

/**
 * 在协调进程从完整词表 logits 选择一个 token。
 *
 * 输入为有限 FP32 [vocab]；temperature=0 取最大值，同分取最小 ID。
 * 否则依次执行 temperature、top-k、top-p，并在保留集合重新归一化。
 * @param logits 按全局 token ID 排列的分数，不能为空。
 * @param options temperature>=0，topP 在 (0,1]，topK=0 表示不截断。
 * @param uniform 协调器随机数，范围 [0,1)，贪心也检查范围。
 * @return 全局 token ID；非法参数或非有限 logits 抛异常，不作隐式回退。
 */
uint32_t SampleQwen3(const std::vector<float> &logits,
    const Qwen3GenerationOptions &options, double uniform);

/**
 * 用现有请求 KV 接口完成一次有界生成，采样只在协调进程执行。
 *
 * 已加载的 group 必须没有活动请求；不管理权重和 worker 的启动/停止。
 * Begin 后统一广播每个采样 ID；EOS 或长度上限结束时释放请求 KV。
 * 首个输出来自 Prefill；最后一个输出不再计算，capacity=prompt+maxNewTokens-1。
 * @param group 已就绪、已加载模型的进程组，不允许并发访问。
 * @param prompt Prompt ID，长度 1..128，容量不能超过 2048。
 * @param options 长度必须为正；EOS ID 必须落在模型词表内。
 * @param observer 可选同步观察回调，收到实际选中的 ID 和对应完整 logits；
 *                 引用仅在调用期间有效，异常将中止生成并尝试释放请求。
 * @param cancelled 可选非阻塞取消探针；在 Prefill 前和逐 token 边界检查，
 *                  不打断执行中的 CUDA/NCCL 操作；取消抛 Qwen3GenerationCancelled。
 * @return 输出 ID 和 eos/length 原因；失败抛异常，worker 失败沿用整组失败语义。
 */
Qwen3GenerationResult GenerateQwen3(ProcessGroup &group, const std::vector<uint32_t> &prompt,
    const Qwen3GenerationOptions &options,
    const std::function<void(uint32_t, const std::vector<float>&)> &observer = {},
    const std::function<bool()> &cancelled = {});

} // namespace distributed
} // namespace fastllm
#endif
