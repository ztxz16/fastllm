#ifndef FASTLLM_DISTRIBUTED_QWEN3_LAYER_H
#define FASTLLM_DISTRIBUTED_QWEN3_LAYER_H

#include "distributed/process_group.h"
#include "distributed/process_allreduce.h"
#include "distributed/qwen3_device_weights.h"
#include <nccl.h>

namespace fastllm {
namespace distributed {

class Qwen3LayerExecutor {
public:
    /**
     * 创建单个 worker 的执行器，不取得权重、stream 或通信组所有权。
     * @param weights 已上传的 rank-local FP16 权重，必须比执行器存活更久。
     * @param stream worker 执行 stream；全部算子与 NCCL 使用此 stream。
     * @param comm 已初始化的 TP 通信组。
     * @param allReduce 与执行器共享 stream 的规约器；须晚于全部请求图释放。
     * @param cudaGraph true 时单 token Decode 使用严格 CUDA Graph；Prefill 保持 Eager。
     * @param prefixCache 是否保留跨请求 Prompt KV 快照；模型卸载时释放。
     */
    Qwen3LayerExecutor(const Qwen3DeviceWeights &weights, cudaStream_t stream, ncclComm_t comm, ProcessAllReduce &allReduce, bool cudaGraph = false, bool prefixCache = false);
    ~Qwen3LayerExecutor();
    Qwen3LayerExecutor(const Qwen3LayerExecutor&) = delete;
    Qwen3LayerExecutor &operator=(const Qwen3LayerExecutor&) = delete;

    /**
     * 执行单请求无历史 KV 的 causal prefill，并同步读回输出。
     *
     * FP16 激活与权重，FP32 累加/规约；Attention 和 MLP 分别调用一次生产 Sum。
     * 临时缓冲总上限 256 MiB，每次成功调用后释放；失败后必须终止 worker。
     * 不含 embedding、最终 Norm、lm_head 或采样；独立阶段与 Full 共用全部算子。
     * @param layer 模型层号。
     * @param tokens 序列长度，范围 1..128。
     * @param hidden 模型隐藏维度。
     * @param positionBegin 连续位置起点，仅影响 RoPE，不表示已有 KV。
     * @param stage Full、Attention 或 Mlp，均包含对应残差。
     * @param input 行主序 FP32 [tokens, hidden]，入口转成 FP16。
     * @return 行主序 FP32 输出；越界、配置不支持或非有限值抛异常，无 CPU fallback。
     */
    std::vector<float> Forward(uint32_t layer, uint32_t tokens, uint32_t hidden,
        uint32_t positionBegin, Qwen3LayerStage stage, const std::vector<float> &input);

    /**
     * 运行完整无 KV prefill，复用单层执行路径和同一 stream，仅读回本 rank 的末 token logits。
     *
     * 输入、隐藏状态和工作区在一次请求内复用；成功后释放，失败必须终止 worker。
     * @param tokenIds 非空 token ID，长度至多 128，不能超出词表。
     * @param positionBegin 连续位置起点，范围 0..1048576。
     * @return 本 rank 的 [vocabCount] FP32 logits，词表起点由权重 Plan 指定。
     */
    std::vector<float> Prefill(const std::vector<uint32_t> &tokenIds, uint32_t positionBegin);

    /**
     * 分配单请求各层 FP16 KV；可选复制最长前缀后计算后缀，成功后保存 Prompt 快照。
     * 至少重算末 token，失败必须终止 worker；快照不包含后续 Decode token。
     * @param tokenIds 1..128 个合法 ID。
     * @param capacity 1..2048，覆盖 Prompt 和后续 token；每 rank KV 总上限 256 MiB。
     * @param positionBegin 起点，容量末位置不得超过 1048576。
     * @return 本 rank 末 token 的 FP32 词表分片。
     */
    std::vector<float> BeginRequest(const std::vector<uint32_t> &tokenIds,
        uint32_t capacity, uint32_t positionBegin);

    /** @param tokenId 追加的单个 ID；必须已有请求且尚有容量。@return 本 rank FP32 logits；失败终止 worker。 */
    std::vector<float> Decode(uint32_t tokenId);

    /** 同步销毁请求 Graph 并释放固定工作区及 KV；保留累计执行计数，允许重复调用。 */
    void EndRequest();

    /** 同步清空所有不可变前缀快照；保留命中计数，不影响独立的活动请求 KV 或 Graph。 */
    void ClearPrefixCache();

    /** @return 请求 KV、Graph 计数/工作区及前缀快照计数/显存字节。 */
    Qwen3RequestInfo RequestInfo() const;

    /** 同步释放请求 KV、前缀快照、工作区和 cuBLAS handle；必须在权重卸载前调用，失败须终止 worker。 */
    void Clear();

private:
    /**
     * 复用完整模型计算，成功同步并释放临时工作区后才提交缓存有效长度。
     * @param tokenIds 1..128 个输入 ID，Decode 仅传一个；入口检查词表边界。
     * @param positionBegin 当前输入的 RoPE 起点，缓存模式必须紧接历史位置。
     * @param cached true 时读取并追加各层 FP16 KV；false 不接触活动请求缓存。
     * @return 本 rank 的 FP32 词表分片，出现异常后必须终止 worker。
     */
    std::vector<float> RunTokens(const std::vector<uint32_t> &tokenIds, uint32_t positionBegin, bool cached);
    /**
     * 使用请求内固定地址重放单 token Decode，成功后才提交 KV 长度。
     * @param tokenId 已验证范围的输入 ID；首次调用先预热同一位置，再捕获和重放。
     * @return FP32 本 rank 词表 logits；任何捕获、执行或检查失败均终止 worker，不回退。
     */
    std::vector<float> DecodeGraph(uint32_t tokenId);
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace distributed
} // namespace fastllm
#endif
