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
