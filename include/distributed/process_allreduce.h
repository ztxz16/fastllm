#pragma once

#include "distributed/process_group.h"
#include <cuda_runtime_api.h>
#include <nccl.h>
#include <memory>

namespace fastllm { namespace distributed {

/** 单机进程通信器；固定 IPC 缓冲属于通信组，晚于全部请求图释放。 */
class ProcessAllReduce {
public:
    /**
     * 集体创建通信资源并交换 IPC 句柄；失败由父进程终止整组，不局部回退。
     * @param stream 生产执行 stream；所有调用必须串行提交到它。
     * @param comm 已初始化的 NCCL 组，用于引导、较大消息和关闭屏障。
     * @param custom true 要求 TP=2、SM70+ 且至少 16 SM、双向 P2P 和原生跨卡原子能力。
     */
    ProcessAllReduce(cudaStream_t stream, ncclComm_t comm, bool custom);
    ~ProcessAllReduce();
    ProcessAllReduce(const ProcessAllReduce&) = delete;
    ProcessAllReduce &operator=(const ProcessAllReduce&) = delete;

    /**
     * 原地执行 FP32 SUM；与生产层共用，Eager/Graph 使用同一实现。
     * @param data 当前设备连续 FP32[count]，不能与通信工作区重叠。
     * @param count 正整数；custom 开启时 <=65536 元素走 IPC，其余走 NCCL。
     * 无分配、同步或 host 序号更新；运行失败终止整组，不能重试本次规约。
     */
    void Sum(float *data, size_t count);

    /**
     * 同步生产 stream 后读取 GPU 实际调用次数；不能在 capture 中调用。
     * @return 组级路径、计数和本地分配字节；关闭后或屏障未完成时抛异常。
     */
    ProcessAllReduceInfo Info();

    /** 集体关闭；先同步并关闭对端映射，再经屏障释放所有者分配。 */
    void Close();

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}} // namespace fastllm::distributed
