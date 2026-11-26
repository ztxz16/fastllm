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
     * @param stream worker 执行 stream；Qwen3分页路径要求cudaStreamPerThread，算子与NCCL共用此流。
     * @param comm 已初始化的 TP 通信组。
     * @param allReduce 与执行器共享 stream 的规约器；须晚于全部请求图释放。
     * @param cudaGraph true 时单 token Decode 使用严格 CUDA Graph；Prefill 保持 Eager。
     * @param prefixCache 是否保留跨请求Prompt前缀；Qwen3使用FT共享页面，MoE保持连续快照。
     * @param batchInvariant 固定 SM70 专家 GEMM；加载时验证支持条件，失败不回退。
     */
    Qwen3LayerExecutor(const Qwen3DeviceWeights &weights, cudaStream_t stream, ncclComm_t comm, ProcessAllReduce &allReduce, bool cudaGraph = false, bool prefixCache = false, bool batchInvariant = false);
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
     * @param tokens 序列长度，范围 1..2048。
     * @param hidden 模型隐藏维度。
     * @param positionBegin 连续位置起点，仅影响 RoPE，不表示已有 KV。
     * @param stage Full、Attention 或 Mlp，均包含对应残差。
     * @param input 行主序 FP32 [tokens, hidden]，入口转成 FP16。
     * @return 行主序 FP32 输出；越界、配置不支持或非有限值抛异常，无 CPU fallback。
     */
    std::vector<float> Forward(uint32_t layer, uint32_t tokens, uint32_t hidden,
        uint32_t positionBegin, Qwen3LayerStage stage, const std::vector<float> &input);

    /**
     * 执行本 rank 的 MoE 单层，与其他 rank 共同进行 EP 集合通信。
     * @param layer 已加载模型的层号。
     * @param tokens 本 rank 有效 Token 数；0 表示空闲。
     * @param capacity 本轮各 rank 最大 Token 数，至少 1；固定 NCCL 分块长度。
     * @param positionBegin 本地序列 RoPE 起点；不表示历史 KV。
     * @param stage Full/Attention/Mlp；同阶段在所有 rank 一致。
     * @param input FP32 [tokens,hidden]；转换后在 worker stream 计算。
     * @return 真实 FP16 输出及中间量；同步失败或非法状态抛异常，必须退出 worker。
     */
    QwenMoeLayerResult ForwardMoe(uint32_t layer, uint32_t tokens, uint32_t capacity,
        uint32_t positionBegin, Qwen3LayerStage stage, const std::vector<float> &input);

    /**
     * 执行完整 MoE 联合轮次；所有 rank 必须共同调用，包括空闲 rank。
     * @param step 本 rank 的准入、Decode、结束或空闲操作。
     * @param graph 使用整组固定地址 Graph，动态位置与活动掩码在设备更新。
     * @return 活动请求的 FP32 完整 logits；空闲返回空，失败后必须终止 worker。
     */
    std::vector<float> StepMoe(const QwenMoeStep &step, bool graph);

    /**
     * 运行完整无 KV prefill，复用单层执行路径和同一 stream，仅读回本 rank 的末 token logits。
     *
     * 输入、隐藏状态和工作区在一次请求内复用；成功后释放，失败必须终止 worker。
     * @param tokenIds 非空 token ID，长度至多 2048，不能超出词表。
     * @param positionBegin 连续位置起点，范围 0..1048576。
     * @return 本 rank 的 [vocabCount] FP32 logits，词表起点由权重 Plan 指定。
     */
    std::vector<float> Prefill(const std::vector<uint32_t> &tokenIds, uint32_t positionBegin);

    /**
     * 建立单请求各层FP16 KV页表，绑定全TP组共同的完整前缀页后计算后缀。
     * 成功后登记完整Prompt页；至少重算一个token，Decode不改写共享前缀页，失败终止worker。
     * @param tokenIds 1..2048 个合法 ID。
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

    /** 准备、导出、提交或解除 KV 占用；调用者必须先验证交接身份，并等待在途访问结束。 */
    std::vector<PDRegion> PDStorage(const PDCommand &command);
    /** 在预先分配的页中生成 KV；层完成通知只携带 key/rank/layer，不承载 KV。 */
    void PDPrefill(const std::vector<uint32_t> &tokens, uint64_t key, const std::string &notifyPath);

    /** 同步清空前缀索引；保留命中计数及活动请求页引用，不破坏正在使用的KV或Graph。 */
    void ClearPrefixCache(int layer = -1);

    /** @return 请求KV、Graph计数/工作区、前缀目录及分页资源统计。 */
    Qwen3RequestInfo RequestInfo() const;

    /** 返回最紧张的分页池计数；不修改缓存、执行GPU操作或扫描前缀目录。 */
    Qwen3KVUsage KVUsage() const;

    /** 同步释放请求KV、前缀索引/页池、Graph私有计划、工作区和cuBLAS handle；必须在权重卸载前调用，失败须终止 worker。 */
    void Clear();

private:
    /**
     * 复用完整模型计算，成功同步并释放临时工作区后才提交缓存有效长度。
     * @param tokenIds 1..2048 个输入 ID，Decode 仅传一个；入口检查词表边界。
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
