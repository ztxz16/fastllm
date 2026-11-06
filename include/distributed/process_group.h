#ifndef FASTLLM_DISTRIBUTED_PROCESS_GROUP_H
#define FASTLLM_DISTRIBUTED_PROCESS_GROUP_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fastllm {
namespace distributed {

enum class ProcessGroupState { Stopped, Starting, Ready, Failed, Stopping };

struct ProcessGroupOptions {
    std::string workerPath;
    // CUDA_VISIBLE_DEVICES 生效后的逻辑设备编号，顺序决定 rank。
    std::vector<int> devices;
    bool controlOnly = false;
    int startupTimeoutMs = 60000;
    int operationTimeoutMs = 10000;
    int shutdownTimeoutMs = 3000;
    int heartbeatIntervalMs = 1000;
};

struct ProcessWorkerInfo {
    int rank;
    int device;
    int pid;
    int cudaRuntimeVersion = 0;
    int ncclVersion = 0;
    std::string deviceName;
};

struct ProcessModelInfo {
    uint32_t tensorCount = 0;
    uint64_t deviceBytes = 0;
    uint64_t sourceBytesRead = 0;
};

enum class Qwen3LayerStage : uint32_t { Full = 0, Attention = 1, Mlp = 2 };

// Linux 单机控制面。Ready 仅代表 worker/通信就绪，不代表模型已加载。
class ProcessGroup {
public:
    ProcessGroup();
    ~ProcessGroup();
    ProcessGroup(const ProcessGroup&) = delete;
    ProcessGroup &operator=(const ProcessGroup&) = delete;

    /**
     * 启动每卡一个独立进程并建立通信组。
     *
     * 使用 posix_spawn 执行独立 worker，不在继承的 CUDA 上下文中运行代码。
     * 组内全部握手成功后才返回；失败会终止整组并抛出 std::runtime_error。
     * 重启前必须调用 Stop；controlOnly 只检查控制面，不初始化 CUDA/NCCL。
     * @param options 可执行文件、逻辑设备列表及各阶段超时，所有超时必须为正。
     */
    void Start(const ProcessGroupOptions &options);

    /**
     * 验证所有 worker 对同一序号的控制消息作出响应。
     * @return 正常返回表示整组响应成功；超时或协议错误抛异常并终止整组。
     */
    void Ping();

    /**
     * 执行通信自检，每个 rank 提交 FP32 标量 rank+1 并进行 NCCL SUM。
     *
     * 此接口用于底座验收，不替代模型推理或性能测试；不支持 controlOnly。
     * @return 每个 rank 实际读回的求和值；运行失败抛异常并终止整组。
     */
    std::vector<float> CheckAllReduce();

    /**
     * 让各 worker 独立读取本 rank 的 Qwen3 权重并以 FP16 驻留 GPU。
     *
     * 只加载权重，不执行前向、不创建 KV。重复加载必须先卸载；任一 worker
     * 失败会终止整组，不能继续使用其他 rank 的部分结果。
     * @param directory 本机各 worker 可访问的 HF 模型目录，调用时固定为绝对路径。
     * @param maxTensorBytes 单个源分片的 CPU 缓冲上限，必须大于零。
     * @param maxDeviceBytes 每个 rank 的 FP16 权重总字节上限，不包含 KV 或 CUDA 上下文。
     * @param timeoutMs 整组加载截止时间；与短控制消息超时独立。
     * @return 按 rank 排列的权重计数与实际读取字节；失败抛异常。
     */
    std::vector<ProcessModelInfo> LoadQwen3(const std::string &directory,
        uint64_t maxTensorBytes = 1073741824ULL, uint64_t maxDeviceBytes = 8589934592ULL,
        int timeoutMs = 300000);

    /**
     * 从指定 worker 的真实 GPU 权重读回连续字节，用于加载正确性验收。
     * @param rank 组内 rank。
     * @param tensorIndex 分片计划中的张量序号。
     * @param offset FP16 字节偏移，必须按两个字节对齐。
     * @param bytes 精确读取字节数，必须为正偶数且不超过 1 MiB，不允许越界。
     * @return 小端 FP16 行主序数据；worker 拒绝或通信失败时终止整组并抛异常。
     */
    std::vector<uint8_t> ReadQwen3Weight(int rank, uint32_t tensorIndex, uint64_t offset, uint32_t bytes);

    /**
     * 同步各 worker 的执行 stream 并释放所有模型权重，可重复调用。
     * 不销毁 NCCL 组；完成后允许重新加载。失败终止整组并抛异常。
     */
    void UnloadQwen3();

    /**
     * 执行已加载模型的单个 Decoder 层，返回所有 rank 的实际输出。
     *
     * 单请求完整 causal prefill，无历史 KV；所有 rank 接收相同 hidden states。
     * 内部 FP16 激活、FP32 GEMM 累加及 NCCL SUM，结果转回 FP32 仅用于控制面传输。
     * Full 执行 Attention+残差和 MLP+残差；另外两个阶段复用相同生产算子。
     * @param layer 从零开始的层号。
     * @param tokens 输入 token 数，范围 1..128。
     * @param hidden 隐藏维度，必须匹配模型。
     * @param positionBegin 连续 RoPE 位置起点，最大 1048576，不表示已有 KV。
     * @param stage 执行阶段。
     * @param input 行主序 [tokens, hidden] FP32 输入，必须可表示为有限 FP16。
     * @param timeoutMs 整组计算和读回的截止时间，默认 60 秒。
     * @return 按 rank 排列的 [tokens, hidden] 输出；请求超过 1 MiB 或无效时拒绝。
     *         worker 失败终止整组；不执行采样，不创建或更新 KV。
     */
    std::vector<std::vector<float>> ForwardQwen3Layer(uint32_t layer, uint32_t tokens, uint32_t hidden,
        uint32_t positionBegin, Qwen3LayerStage stage, const std::vector<float> &input, int timeoutMs = 60000);

    /**
     * 执行完整 Qwen3 causal prefill，拼接各 rank 的词表分片得到末 token logits。
     *
     * Embedding、全部 Decoder 层、最终 RMSNorm 和分片 lm_head 均在 worker GPU 执行。
     * 激活保持 FP16，末层投影输出 FP32；不建立 KV cache、不采样、不返回所有 token。
     * @param tokenIds 单请求 token ID，长度 1..128；每个 ID 必须小于模型词表大小。
     * @param positionBegin 连续位置起点，范围 0..1048576，不表示已有 KV。
     * @param timeoutMs 包括通信与读回的整组截止时间，必须为正。
     * @return 按全局词表顺序排列的有限 FP32 logits；分片不连续或 worker 出错终止整组。
     */
    std::vector<float> PrefillQwen3(const std::vector<uint32_t> &tokenIds,
        uint32_t positionBegin = 0, int timeoutMs = 300000);

    /**
     * 停止监控、请求正常退出并回收全部 worker。
     *
     * 正常退出超时后发送 SIGKILL，在限定时间内轮询回收，不无限 waitpid。
     * 可重复调用；退出异常保留在 LastError，未能回收的进程阻止再次 Start。
     * @return 全部进程已回收时正常返回；仍有进程未回收时抛异常。
     */
    void Stop();

    ProcessGroupState State() const;
    std::string LastError() const;
    std::vector<ProcessWorkerInfo> Workers() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace distributed
} // namespace fastllm
#endif
