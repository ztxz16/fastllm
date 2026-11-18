#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace fastllm {
namespace distributed {
namespace detail {

/**
 * 保存正式 MoE Eager Begin 的隐藏状态轨迹，不改变算子输入或计算策略。
 *
 * 调用方在提交计算前分配 GPU 快照区并计入原有工作区预算。本类预建全部
 * 描述，执行期间只在原 stream 追加 D2D 快照，随后统一读回，调用方完成
 * 原有同步后才写文件。快照区由执行器持有；本对象必须活到读回完成。
 * 仅用于独立数值诊断，不支持 Decode、Graph、Prefix 或性能测量。
 */
class MoeForwardTrace {
    struct Entry {
        std::string name;
        size_t offset, elements;
    };
    std::vector<Entry> entries;
    std::vector<uint32_t> ids;
    std::vector<half> host;
    half *device = nullptr;
    std::string directory;
    uint32_t hidden, layers, rank;
    size_t next = 0;
    size_t limitMiB = 32;
    bool readback = false;

    /** CUDA 失败立即抛出，不改变正式后端或重试计算。 */
    static void Check(cudaError_t status) {
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string("MoE forward trace CUDA: ") + cudaGetErrorString(status));
        }
    }

    /** 创建新文件并完整写入；禁止覆盖历史诊断，失败时保留不完整文件供排查。 */
    static void WriteNew(const std::string &path, const void *data, size_t bytes) {
        const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd < 0) throw std::runtime_error("cannot create MoE trace file: " + path);
        FILE *file = fdopen(fd, "wb");
        if (!file) {
            close(fd);
            throw std::runtime_error("cannot open MoE trace stream: " + path);
        }
        const bool written = fwrite(data, 1, bytes, file) == bytes;
        const bool closed = fclose(file) == 0;
        if (!written || !closed) throw std::runtime_error("cannot write complete MoE trace: " + path);
    }

public:
    /**
     * 预建连续 FP16 快照布局，默认最多使用 32 MiB GPU 和等量 CPU 存储。
     * 独立长Prompt诊断可设置FASTLLM_MOE_TRACE_MAX_MIB=256；只接受32或256，
     * 不改变执行器已有的512 MiB总工作区限制，也不改变正常推理的分配或计算。
     * @param directory 已创建的独立目录，保存时只新增文件。
     * @param ids 当前完整历史 Token ID，非空且从位置零开始。
     * @param hidden 每个 Token 的隐藏宽度。
     * @param layers Decoder 层数。
     * @param rank 当前 DP rank；仅有请求的 rank 创建快照。
     */
    MoeForwardTrace(const std::string &directory, const std::vector<uint32_t> &ids,
                    uint32_t hidden, uint32_t layers, uint32_t rank)
        : ids(ids), directory(directory), hidden(hidden), layers(layers), rank(rank) {
        if (const char *value = std::getenv("FASTLLM_MOE_TRACE_MAX_MIB")) {
            const std::string setting(value);
            if (setting != "32" && setting != "256") {
                throw std::runtime_error("FASTLLM_MOE_TRACE_MAX_MIB must be 32 or 256");
            }
            limitMiB = setting == "256" ? 256 : 32;
        }
        const size_t limit = limitMiB * 1024 * 1024 / sizeof(half);
        const uint16_t endian = 1;
        if (!*reinterpret_cast<const uint8_t*>(&endian) || ids.empty() || !hidden || !layers ||
            hidden > limit || ids.size() > limit / hidden) {
            throw std::runtime_error("invalid MoE forward trace shape/endianness");
        }
        const size_t rows = ids.size() * hidden;
        const uint64_t stages = uint64_t(layers) * 2 + 1;
        if (stages > (limit - hidden) / rows) {
            throw std::runtime_error("MoE forward trace exceeds " + std::to_string(limitMiB) + " MiB");
        }
        entries.push_back({"embedding", 0, rows});
        for (uint32_t layer = 0; layer < layers; ++layer) {
            for (const char *stage : {"attention", "moe"}) {
                entries.push_back({"layer_" + std::to_string(layer) + "_" + stage,
                                   entries.size() * rows, rows});
            }
        }
        entries.push_back({"final_norm", size_t(stages) * rows, hidden});
        host.resize(size_t(stages) * rows + hidden);
    }

    /** 返回预先验证的快照字节数，供执行器从原有工作区池分配。 */
    size_t Bytes() const { return host.size() * sizeof(half); }

    /** 绑定执行器拥有的连续 GPU 区域；只允许在采集前绑定一次。 */
    void Attach(void *buffer) {
        if (!buffer || device || next) throw std::runtime_error("invalid MoE trace buffer binding");
        device = static_cast<half*>(buffer);
    }

    /**
     * 按预定阶段保存连续 FP16 激活，仅读取正式张量，不分配或同步。
     * @param source 当前 stream 已产生的激活。
     * @param elements 普通阶段为 tokens*hidden，final_norm 为 hidden。
     * @param stream 正式执行器 stream，快照必须在下一次覆盖源张量之前提交。
     */
    void Capture(const half *source, size_t elements, cudaStream_t stream) {
        if (!device || !source || next >= entries.size() || entries[next].elements != elements || readback) {
            throw std::runtime_error("invalid MoE forward trace capture order/shape");
        }
        Check(cudaMemcpyAsync(device + entries[next].offset, source, elements * sizeof(half),
                              cudaMemcpyDeviceToDevice, stream));
        ++next;
    }

    /** 在原有最终同步前提交一次整体读回，拒绝缺失阶段或重复提交。 */
    void Readback(cudaStream_t stream) {
        if (next != entries.size() || readback) throw std::runtime_error("incomplete MoE forward trace");
        Check(cudaMemcpyAsync(host.data(), device, Bytes(), cudaMemcpyDeviceToHost, stream));
        readback = true;
    }

    /**
     * 原有 stream 同步成功后写入 FP16 小端二进制及 JSON 描述，最后写描述表示完成。
     * @param round 执行器单调增加的采集轮次，与 PID、rank 一起保证文件名可区分。
     */
    void Save(uint64_t round) const {
        if (!readback) throw std::runtime_error("MoE trace readback not submitted");
        const std::string base = "rank-" + std::to_string(rank) + "-pid-" + std::to_string(getpid()) +
                                 "-round-" + std::to_string(round);
        const std::string binary = base + ".f16";
        std::string json = "{\"version\":1,\"dtype\":\"float16\",\"byte_order\":\"little\",\"rank\":" +
            std::to_string(rank) + ",\"tokens\":" + std::to_string(ids.size()) + ",\"hidden\":" +
            std::to_string(hidden) + ",\"layers\":" + std::to_string(layers) + ",\"ids\":[";
        for (size_t i = 0; i < ids.size(); ++i) json += (i ? "," : "") + std::to_string(ids[i]);
        json += "],\"round\":" + std::to_string(round) + ",\"max_mib\":" + std::to_string(limitMiB) +
                ",\"binary\":\"" + binary + "\",\"snapshots\":[";
        for (size_t i = 0; i < entries.size(); ++i) {
            const auto &entry = entries[i];
            json += std::string(i ? "," : "") + "{\"name\":\"" + entry.name + "\",\"offset\":" +
                    std::to_string(entry.offset) + ",\"elements\":" + std::to_string(entry.elements) + "}";
        }
        json += "]}\n";
        WriteNew(directory + "/" + binary, host.data(), Bytes());
        WriteNew(directory + "/" + base + ".json", json.data(), json.size());
    }
};

} // namespace detail
} // namespace distributed
} // namespace fastllm
