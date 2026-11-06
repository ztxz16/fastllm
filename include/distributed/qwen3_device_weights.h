#ifndef FASTLLM_DISTRIBUTED_QWEN3_DEVICE_WEIGHTS_H
#define FASTLLM_DISTRIBUTED_QWEN3_DEVICE_WEIGHTS_H

#include "distributed/qwen3_shard_loader.h"
#include <cuda_runtime_api.h>
#include <memory>

namespace fastllm {
namespace distributed {

struct Qwen3DeviceTensor {
    std::string name;
    std::vector<uint64_t> shape;
    void *data = nullptr; // 当前设备上的 FP16 行主序数据；只由所属权重对象释放。
    uint64_t bytes = 0;
};

class Qwen3DeviceWeights {
public:
    /**
     * 将 rank-local 分片转换为 FP16 并上传当前 worker 的设备。
     *
     * 使用调用方执行 stream；每次上传同步后才复用 CPU 缓冲。CPU 仅持有一个源
     * 分片及至多 1 MiB 转换缓冲，不加载完整 CPU 模型。不接受非有限值或超出
     * FP16 有限范围的值。任意失败使本对象失效，必须终止 worker 回收 CUDA 上下文。
     * @param loader 已验证的本 rank 分片读取器，构造完成后可释放。
     * @param stream worker 的非空 CUDA stream，不取得其所有权。
     * @param maxTensorBytes 单个源张量 CPU 分配上限，必须为正。
     * @param maxDeviceBytes FP16 权重总字节上限，不含上下文、KV、分配器开销。
     */
    Qwen3DeviceWeights(const Qwen3ShardLoader &loader, cudaStream_t stream,
                      uint64_t maxTensorBytes, uint64_t maxDeviceBytes);
    // 异常析构不调用 CUDA；正常退出必须先 Clear，失败由 worker 退出回收上下文。
    ~Qwen3DeviceWeights() = default;
    Qwen3DeviceWeights(const Qwen3DeviceWeights&) = delete;
    Qwen3DeviceWeights &operator=(const Qwen3DeviceWeights&) = delete;

    /** @return 指定序号的只读 GPU 张量；越界或设备改变抛异常，Clear 后指针失效。 */
    const Qwen3DeviceTensor &Tensor(uint32_t index) const;
    uint32_t TensorCount() const { return static_cast<uint32_t>(tensors.size()); }
    uint64_t DeviceBytes() const { return deviceBytes; }
    uint64_t SourceBytesRead() const { return sourceBytesRead; }

    /**
     * 同步读回真实 GPU 数据，供正确性检查使用，不参与正式前向。
     * @param index 张量序号。
     * @param offset 偶数字节偏移。
     * @param bytes 正偶数字节长度，不超过 1 MiB 且不能越界。
     * @return FP16 原始字节；使用同一执行 stream，失败抛异常。
     */
    std::vector<uint8_t> Read(uint32_t index, uint64_t offset, uint32_t bytes) const;

    /** 同步 stream 并释放全部 GPU 权重，可重复调用；失败须终止 worker。 */
    void Clear();

private:
    /** 检查当前设备与构造时一致；拒绝隐式设备迁移。 */
    void CheckDevice() const;
    int device = -1;
    cudaStream_t stream;
    uint64_t deviceBytes = 0, sourceBytesRead = 0;
    std::vector<Qwen3DeviceTensor> tensors;
};

} // namespace distributed
} // namespace fastllm
#endif
