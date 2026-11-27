#ifndef FASTLLM_PD_CONTIGUOUS_CUH
#define FASTLLM_PD_CONTIGUOUS_CUH

#include "distributed/pd_native.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <map>
#include <stdexcept>

namespace fastllm { namespace distributed { namespace detail {

/**
 * 连续 KV 的独立交接缓冲；计算保留原有固定地址，避免破坏 EP Decode Graph。
 *
 * 导出和导入仅使用 GPU 内复制。所有权由 key 管理；Release 只能在服务层确认
 * 在途访问结束后调用，析构不隐式释放可能仍被远端引用的显存。
 */
class PDContiguous {
    struct Entry { uint32_t tokens; uint64_t bytes; std::vector<void*> layers; };
    std::map<uint64_t, Entry> entries;
    uint64_t used = 0;
    uint32_t layerCount, heads, dim;
    static constexpr uint64_t budget = 512ULL * 1024 * 1024;

    static void Check(cudaError_t status) {
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    }
public:
    PDContiguous(uint32_t layers, uint32_t kvHeads, uint32_t headDim)
        : layerCount(layers), heads(kvHeads), dim(headDim) {}

    /** @return 在固定交接预算内可分配的 128 Token 单位数；不包含正在解码的固定 KV。 */
    uint32_t Free() const {
        size_t available, total;
        Check(cudaMemGetInfo(&available, &total));
        return std::min<uint64_t>(budget - used, available) / (uint64_t(layerCount) * 128 * 2 * heads * dim * 2);
    }
    uint32_t Total() const { return budget / (uint64_t(layerCount) * 128 * 2 * heads * dim * 2); }

    /**
     * 分配请求独占的连续 FP16 [token,2,head,dim] 交接缓冲。
     *
     * 容量检查通过后先读取设备属性中的 UUID，再登记交接和分配显存；
     * 设备查询失败时不留下交接记录。UUID 沿用分页 KV 的 16 字节表示。
     *
     * @param key 本地唯一交接编号。
     * @param tokens 有效 Token 数，调用方检查模型容量。
     * @return IPC 描述；不足时返回空，不修改既有交接。CUDA 失败终止 worker。
     */
    std::vector<PDRegion> Prepare(uint64_t key, uint32_t tokens) {
        if (!key || !tokens || entries.count(key)) throw std::runtime_error("invalid contiguous PD allocation");
        const uint64_t bytes = uint64_t(tokens) * 2 * heads * dim * 2;
        if (bytes > budget / layerCount || bytes * layerCount > budget - used) return {};
        size_t available, total;
        Check(cudaMemGetInfo(&available, &total));
        if (bytes * layerCount > available) return {};

        int device = 0;
        cudaDeviceProp property{};
        Check(cudaGetDevice(&device));
        Check(cudaGetDeviceProperties(&property, device));

        auto &entry = entries.emplace(key, Entry{tokens, bytes, std::vector<void*>(layerCount, nullptr)}).first->second;
        std::vector<PDRegion> result;
        for (uint32_t layer = 0; layer < layerCount; ++layer) {
            const auto status = cudaMalloc(&entry.layers[layer], bytes);
            if (status == cudaErrorMemoryAllocation) {
                (void)cudaGetLastError();
                Release(key);
                return {};
            }
            Check(status);
            used += bytes;
            cudaIpcMemHandle_t handle;
            Check(cudaIpcGetMemHandle(&handle, entry.layers[layer]));
            for (uint32_t component = 0; component < 2; ++component) {
                PDRegion r;
                r.layer = layer; r.component = component; r.tokenEnd = tokens;
                r.headEnd = heads; r.headDim = dim; r.pageSize = 0;
                r.allocationBytes = bytes; r.offset = uint64_t(component) * heads * dim * 2;
                r.tokenStride = uint64_t(2) * heads * dim * 2; r.headStride = dim * 2;
                std::memcpy(r.memory.data(), &handle, 64);
                std::memcpy(r.device.data(), property.uuid.bytes, r.device.size());
                result.push_back(r);
            }
        }
        return result;
    }

    /** 同一 worker stream 上复制一层；完成事件必须排在本调用之后。 */
    void Copy(uint64_t key, uint32_t layer, void *compute, uint32_t tokens, bool exporting, cudaStream_t stream) {
        auto &entry = entries.at(key);
        if (layer >= layerCount || tokens > entry.tokens) throw std::runtime_error("contiguous PD copy outside range");
        if (!tokens) return;
        void *snapshot = entry.layers[layer];
        Check(cudaMemcpyAsync(exporting ? snapshot : compute, exporting ? compute : snapshot,
            uint64_t(tokens) * 2 * heads * dim * 2, cudaMemcpyDeviceToDevice, stream));
    }

    /** 在调用方已排空本地及远端访问后幂等解除占用。 */
    void Release(uint64_t key) {
        auto it = entries.find(key);
        if (it == entries.end()) return;
        for (auto memory : it->second.layers) if (memory) { Check(cudaFree(memory)); used -= it->second.bytes; }
        entries.erase(it);
    }

    /** @return 是否仍有交接持有内存；卸载不得隐式越过在途保护。 */
    bool Empty() const { return entries.empty(); }
};
} } }
#endif
