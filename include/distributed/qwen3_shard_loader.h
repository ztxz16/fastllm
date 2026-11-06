#ifndef FASTLLM_DISTRIBUTED_QWEN3_SHARD_LOADER_H
#define FASTLLM_DISTRIBUTED_QWEN3_SHARD_LOADER_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fastllm {
namespace distributed {

struct ShardSlice {
    std::string source;
    uint64_t rowBegin = 0, rowEnd = 0;
    uint64_t columnBegin = 0, columnEnd = 0;
};

struct ShardTensorPlan {
    std::string name;
    std::string dtype; // safetensors F16 / BF16 / F32；不在读取器中转换。
    std::vector<uint64_t> shape;
    std::vector<ShardSlice> slices; // 按行拼接；一维张量视为 [N, 1]。
    uint64_t bytes = 0;
    bool zeroFill = false; // 行并行输出偏置仅 rank 0 读取，其他 rank 为零。
};

struct Qwen3ShardPlan {
    int rank = 0, worldSize = 1;
    uint64_t queryHeadBegin = 0, queryHeadCount = 0;
    uint64_t kvHeadBegin = 0, kvHeadCount = 0;
    uint64_t intermediateBegin = 0, intermediateCount = 0;
    uint64_t vocabBegin = 0, vocabCount = 0;
    uint64_t tensorBytes = 0;
    std::vector<ShardTensorPlan> tensors;
};

struct LoadedShardTensor {
    // 行主序、原始小端浮点字节；不含未归本 rank 所有的行/列。
    std::vector<uint8_t> data;
    uint64_t bytesRead = 0; // 请求读取的张量 payload，不是物理磁盘 I/O。
};

class Qwen3ShardLoader {
public:
    /**
     * 根据 HF 配置和 safetensors 元数据创建单个 rank 的稠密 Qwen3 分片计划。
     *
     * 构造阶段只读取配置、索引和文件头，不读取权重 payload。支持未量化
     * F16/BF16/F32、标准分离 Q/K/V 与 Gate/Up、可选 Attention 偏置及 tied
     * embedding。均匀分片允许非整除维度，TP 超过 KV 头数时复制对应 KV 头。
     * 相对路径在构造时固定为绝对路径；输入文件在对象生命周期内必须保持不变。
     * 未知权重、量化、重复 JSON 字段或非法布局抛异常。
     * @param directory 包含 config.json 和 safetensors 文件/索引的本地目录。
     * @param rank 当前进程 rank，范围 [0, worldSize)。
     * @param worldSize TP 进程数；必须能给每个 rank 分配至少一个 Q 头、MLP 通道和词表行。
     */
    Qwen3ShardLoader(const std::string &directory, int rank, int worldSize);
    ~Qwen3ShardLoader();
    Qwen3ShardLoader(const Qwen3ShardLoader&) = delete;
    Qwen3ShardLoader &operator=(const Qwen3ShardLoader&) = delete;

    /** @return 已验证的只读分片计划；生命周期与读取器相同。 */
    const Qwen3ShardPlan &Plan() const;

    /**
     * 直接将源文件行/列片段读入最终 rank-local 缓冲区。
     *
     * QKV 与 Gate/Up 在目标缓冲区内按行合并，不创建完整源张量或临时分片。
     * 不分配 CUDA 内存，不执行 dtype 转换，也不修改现有线程 TP 加载路径。
     * @param name Plan 中的目标权重名。
     * @param maxBytes 单个目标张量允许的最大分配字节数；超限在分配和 payload I/O 前拒绝。
     * @return 本 rank 权重及实际请求读取的字节数；短读、文件变化、未知名字抛异常。
     */
    LoadedShardTensor LoadTensor(const std::string &name, uint64_t maxBytes) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace distributed
} // namespace fastllm
#endif
