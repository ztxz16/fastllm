#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace fastllm {
namespace distributed {
namespace detail {

/**
 * 观察正式第0层MoE内部张量，包含空闲请求rank上实际执行的专家。
 *
 * 在原有路由计数同步后预建有界布局；GPU空间由执行器原工作区池分配。
 * 只在原stream追加D2D复制，最终统一读回；不更换算子、通信或路由。
 * 专家快照保留实际压紧行数及槽位索引，跨组合比较必须按槽位匹配。
 */
class MoeBlockTrace {
    struct Entry {
        std::string name, dtype;
        size_t offset, bytes;
        std::vector<size_t> shape;
    };
    std::vector<Entry> entries;
    std::vector<uint8_t> host;
    std::vector<int> counts;
    uint8_t *device = nullptr;
    std::string directory;
    uint32_t tokens, capacity, hidden, width, sharedWidth, experts, topk, rank, world;
    size_t total = 0, next = 0;
    bool readback = false;
    static constexpr size_t limit = 256ULL * 1024 * 1024;

    /** 预建无padding的连续字节布局，形状乘法溢出或超出256 MiB立即拒绝。 */
    void Add(const std::string &name, const std::string &dtype, std::vector<size_t> shape) {
        size_t bytes = dtype == "float16" ? 2 : 4;
        for (size_t n : shape) {
            if (!n || n > limit / bytes) throw std::runtime_error("invalid MoE block trace shape");
            bytes *= n;
        }
        if (bytes > limit - total) throw std::runtime_error("MoE block trace exceeds 256 MiB");
        entries.push_back({name, dtype, total, bytes, shape});
        total += bytes;
    }

    /** CUDA失败直接终止本次诊断，不改变正式执行策略。 */
    static void Check(cudaError_t status) {
        if (status != cudaSuccess) throw std::runtime_error(std::string("MoE block trace: ") + cudaGetErrorString(status));
    }

    /** 创建全新文件；短写和关闭失败均报错，禁止覆盖历史证据。 */
    static void Write(const std::string &path, const void *data, size_t bytes) {
        int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd < 0) throw std::runtime_error("cannot create MoE block trace: " + path);
        FILE *file = fdopen(fd, "wb");
        if (!file) { close(fd); throw std::runtime_error("cannot open MoE block trace stream"); }
        const bool written = fwrite(data, 1, bytes, file) == bytes;
        const bool closed = fclose(file) == 0;
        if (!written || !closed) throw std::runtime_error("incomplete MoE block trace write");
    }

public:
    /**
     * 根据正式路由计数安排完整快照，不修改任何计数或槽位。
     * @param directory 本次会话独占目录。
     * @param tokens 当前rank有效Token数，可为零。
     * @param capacity 全组每rank的padded行数；仅支持大于一的Prefill。
     * @param hidden 隐藏宽度H。
     * @param width 单专家中间宽度M，Gate/Up为[count,2M]。
     * @param sharedWidth 共享专家中间宽度。
     * @param experts 全局专家数。
     * @param topk 每Token路由槽位数。
     * @param rank 当前专家owner rank。
     * @param world 进程组rank数。
     * @param counts 原有同步后取得的int32[experts]，包括全组有效Token。
     */
    MoeBlockTrace(const std::string &directory, uint32_t tokens, uint32_t capacity,
                  uint32_t hidden, uint32_t width, uint32_t sharedWidth, uint32_t experts,
                  uint32_t topk, uint32_t rank, uint32_t world, const std::vector<int> &counts)
        : counts(counts), directory(directory), tokens(tokens), capacity(capacity), hidden(hidden),
          width(width), sharedWidth(sharedWidth), experts(experts), topk(topk), rank(rank), world(world) {
        const uint16_t endian = 1;
        const size_t rows = size_t(capacity) * world;
        if (!*reinterpret_cast<const uint8_t*>(&endian) || capacity <= 1 || tokens > capacity ||
            !hidden || !width || !sharedWidth || !experts || !topk || topk > experts || rank >= world ||
            counts.size() != experts) throw std::runtime_error("invalid MoE block trace metadata");
        for (int count : counts) if (count < 0 || size_t(count) > rows) throw std::runtime_error("invalid expert count");

        if (tokens) Add("input", "float16", {tokens, hidden});
        Add("norm", "float16", {capacity, hidden});
        Add("router_logits", "float16", {capacity, experts});
        Add("selected_experts", "int32", {capacity, topk});
        Add("routing_weights", "float16", {capacity, topk});
        Add("gathered_norm", "float16", {rows, hidden});
        Add("gathered_experts", "int32", {rows, topk});
        Add("gathered_weights", "float16", {rows, topk});
        Add("expert_counts", "int32", {experts});
        for (uint32_t e = rank; e < experts; e += world) {
            if (!counts[e]) continue;
            const size_t n = counts[e];
            const std::string prefix = "expert_" + std::to_string(e) + "_";
            Add(prefix + "indices", "int32", {n});
            Add(prefix + "packed", "float16", {n, hidden});
            Add(prefix + "gateup", "float16", {n, size_t(2)*width});
            Add(prefix + "activated", "float16", {n, width});
            Add(prefix + "down", "float16", {n, hidden});
        }
        Add("local_contributions", "float16", {capacity, topk, hidden});
        Add("routed", "float32", {capacity, hidden});
        if (tokens) {
            Add("shared_gateup", "float16", {tokens, size_t(2)*sharedWidth});
            Add("shared_activated", "float16", {tokens, sharedWidth});
            Add("shared_down", "float16", {tokens, hidden});
            Add("shared_gate", "float16", {tokens, 1});
            Add("shared_weighted", "float16", {tokens, hidden});
            Add("output", "float16", {tokens, hidden});
        }
        host.resize(total);
    }

    /** 返回有界快照大小；执行器仍执行原有512 MiB总工作区检查。 */
    size_t Bytes() const { return total; }

    /** 绑定执行器拥有的设备缓冲，仅允许采集前调用一次。 */
    void Attach(void *buffer) {
        if (!buffer || device || next) throw std::runtime_error("invalid MoE block trace binding");
        device = static_cast<uint8_t*>(buffer);
    }

    /**
     * 在原stream复制下一阶段完整张量，复制后原缓冲可继续复用。
     * @param name 预建阶段名称，严格验证调用顺序。
     * @param source 连续张量地址，类型与形状由布局固定。
     * @param stream 正式计算stream；不增加同步或执行额外算子。
     */
    void Capture(const std::string &name, const void *source, cudaStream_t stream) {
        if (!device || !source || readback || next >= entries.size() || entries[next].name != name)
            throw std::runtime_error("invalid MoE block trace capture: " + name);
        const auto &entry = entries[next++];
        Check(cudaMemcpyAsync(device + entry.offset, source, entry.bytes, cudaMemcpyDeviceToDevice, stream));
    }

    /** 在原最终同步前统一读回，拒绝漏阶段或重复读回。 */
    void Readback(cudaStream_t stream) {
        if (next != entries.size() || readback) throw std::runtime_error("incomplete MoE block trace");
        Check(cudaMemcpyAsync(host.data(), device, total, cudaMemcpyDeviceToHost, stream));
        readback = true;
    }

    /** 原stream已同步后保存小端原始数据，最后写manifest作为完成标记。 */
    void Save(uint64_t round) const {
        if (!readback) throw std::runtime_error("MoE block trace readback missing");
        const std::string base = "rank-" + std::to_string(rank) + "-pid-" + std::to_string(getpid()) +
                                 "-round-" + std::to_string(round);
        std::string json = "{\"version\":1,\"byte_order\":\"little\",\"layer\":0,\"max_mib\":256";
        auto field = [&](const char *name, uint64_t value) { json += ",\"" + std::string(name) + "\":" + std::to_string(value); };
        field("rank", rank); field("world", world); field("round", round); field("tokens", tokens);
        field("capacity", capacity); field("hidden", hidden); field("width", width);
        field("shared_width", sharedWidth); field("experts", experts); field("topk", topk); field("bytes", total);
        json += ",\"counts\":[";
        for (size_t i = 0; i < counts.size(); ++i) json += (i ? "," : "") + std::to_string(counts[i]);
        json += "],\"binary\":\"" + base + ".bin\",\"snapshots\":[";
        for (size_t i = 0; i < entries.size(); ++i) {
            const auto &entry = entries[i];
            json += std::string(i ? "," : "") + "{\"name\":\"" + entry.name + "\",\"dtype\":\"" + entry.dtype +
                "\",\"offset_bytes\":" + std::to_string(entry.offset) + ",\"bytes\":" + std::to_string(entry.bytes) + ",\"shape\":[";
            for (size_t j = 0; j < entry.shape.size(); ++j) json += (j ? "," : "") + std::to_string(entry.shape[j]);
            json += "]}";
        }
        json += "]}\n";
        Write(directory + "/" + base + ".bin", host.data(), total);
        Write(directory + "/" + base + ".json", json.data(), json.size());
    }
};

} // namespace detail
} // namespace distributed
} // namespace fastllm
