#ifndef FASTLLM_DISTRIBUTED_QWEN3_LAYER_H
#define FASTLLM_DISTRIBUTED_QWEN3_LAYER_H

#include "distributed/process_group.h"
#include "distributed/qwen3_device_weights.h"
#include <nccl.h>

namespace fastllm {
namespace distributed {

class Qwen3LayerExecutor {
public:
    /**
     * 创建单个 worker 的 eager 单层执行器，不取得权重、stream 或通信组所有权。
     * @param weights 已上传的 rank-local FP16 权重，必须比执行器存活更久。
     * @param stream worker 执行 stream；全部算子与 NCCL 使用此 stream。
     * @param comm 已初始化的 TP 通信组。
     */
    Qwen3LayerExecutor(const Qwen3DeviceWeights &weights, cudaStream_t stream, ncclComm_t comm);
    ~Qwen3LayerExecutor();
    Qwen3LayerExecutor(const Qwen3LayerExecutor&) = delete;
    Qwen3LayerExecutor &operator=(const Qwen3LayerExecutor&) = delete;

    /**
     * 执行单请求无历史 KV 的 causal prefill，并同步读回输出。
     *
     * FP16 激活与权重，FP32 累加/规约；Attention 和 MLP 分别执行一次 NCCL SUM。
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

    /** 同步并释放工作区和 cuBLAS handle；必须在权重卸载前调用，失败须终止 worker。 */
    void Clear();

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace distributed
} // namespace fastllm
#endif
