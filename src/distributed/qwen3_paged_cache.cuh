#ifndef FASTLLM_PROCESS_QWEN3_PAGED_CACHE_CUH
#define FASTLLM_PROCESS_QWEN3_PAGED_CACHE_CUH

#include <cuda_fp16.h>
#include "fastllm.h"
#include "devices/cpu/cpudevice.h"
#include "devices/cuda/cudadevice.h"
#include "devices/cuda/attention/fastllm-attention-backend.h"
#include <set>
#include <map>
#include <cstring>

namespace fastllm { namespace distributed { namespace detail {

/** 将CUDA错误交给worker的统一失败路径。 */
inline void CheckPagedCuda(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(std::string("FT paged cache CUDA: ") + cudaGetErrorString(error));
}

/** 将融合FP16 QKV[T,Hq+2Hkv,D]拆成FT使用的连续HND布局，不改变数值。 */
__global__ void ProcessUnpackQkv(const half *input, half *q, half *k, half *v,
                               int tokens, int qHeads, int kvHeads, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int heads = qHeads + 2 * kvHeads;
    if (i >= tokens * heads * dim) return;
    int d = i % dim, head = i / dim % heads, token = i / (heads * dim);
    if (head < qHeads) q[(head * tokens + token) * dim + d] = input[i];
    else if (head < qHeads + kvHeads) k[((head - qHeads) * tokens + token) * dim + d] = input[i];
    else v[((head - qHeads - kvHeads) * tokens + token) * dim + d] = input[i];
}

/**
 * 适配进程TP与FT正式分页算子；页分配、前缀索引和引用计数完全由FT页管理器持有。
 * FP16页池为[pages,pageLen,localKVHeads,headDim]，请求描述符为[heads,tokens,dim]。
 * Graph前在CPU准备页引用与固定地址元数据，捕获时只执行已准备的CUDA操作。
 */
class ProcessPagedKV {
    struct Layer {
        std::unique_ptr<PagedCacheManager> km, vm;
        Data k{DataType::FLOAT16}, v{DataType::FLOAT16};
        Data qs{DataType::INT32}, ps{DataType::INT32}, pi{DataType::INT32}, last{DataType::INT32};
        Data insert{DataType::INT32}, offset{DataType::INT32};
    };
    struct Scratch {
        Data q{DataType::FLOAT16}, k{DataType::FLOAT16}, v{DataType::FLOAT16};
        int capacity = 0;
    };
    int device, qHeads, kvHeads, dim;
    std::vector<std::unique_ptr<Layer>> layers;
    Scratch eager, graph;
    CudaAppendPagedCacheOp append;
    CudaAppendPagedCacheBatchOp appendBatch;
    std::string backend;
    std::map<uint64_t, std::vector<std::vector<int>>> pdPins;
    bool pdPrepared = false;

    /** 在当前设备分配持久张量；不得在Graph捕获期间调用。 */
    void Allocate(Data &data, const std::vector<int> &shape) {
        data.directMemory = true;
        data.ToDevice(DataDevice::CUDA, {device}, false);
        data.Resize(shape);
        data.Allocate();
    }

    /** 按调用规模准备共享Q/K/V缓冲；Decode缓冲与Eager缓冲分开保证捕获地址稳定。 */
    void PrepareScratch(Scratch &scratch, int tokens) {
        if (scratch.capacity >= tokens) return;
        Allocate(scratch.q, {qHeads, tokens, dim});
        Allocate(scratch.k, {kvHeads, tokens, dim});
        Allocate(scratch.v, {kvHeads, tokens, dim});
        scratch.capacity = tokens;
    }

    /** 创建独占页池和固定容量元数据；临时无状态计算不占用请求的页面。 */
    std::unique_ptr<Layer> NewLayer(int maxPages = -1) {
        auto layer = std::unique_ptr<Layer>(new Layer());
        Data desc(DataType::FLOAT16, {kvHeads, 0, dim});
        desc.dataDevice = DataDevice::CUDA;
        desc.dataDeviceIds = {device};
        layer->km.reset(CreatePagedCacheManager(PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, desc, -1, maxPages));
        layer->vm.reset(CreatePagedCacheManager(PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, desc, -1, maxPages));
        RestorePagedCachePages(layer->k, *layer->km, {}, 0);
        RestorePagedCachePages(layer->v, *layer->vm, {}, 0);
        Allocate(layer->qs, {2}); Allocate(layer->ps, {2});
        Allocate(layer->pi, {layer->km->maxPages}); Allocate(layer->last, {1});
        Allocate(layer->insert, {1}); Allocate(layer->offset, {1});
        return layer;
    }

    /** 上传页表；K/V使用相同分配顺序，必须保持物理页编号配对。 */
    void Metadata(Layer &layer, int tokens) {
        if (layer.k.pageIndex != layer.v.pageIndex || layer.k.lastPageLen != layer.v.lastPageLen)
            throw std::runtime_error("FT paged K/V allocation diverged");
        auto copy = [](Data &to, const std::vector<int> &values) {
            to.cpuIntDatas = values;
            CheckPagedCuda(cudaMemcpyAsync(to.cudaData, to.cpuIntDatas.data(), values.size() * sizeof(int), cudaMemcpyHostToDevice, cudaStreamPerThread));
        };
        copy(layer.qs, {0, tokens});
        copy(layer.ps, {0, int(layer.k.pageIndex.size())});
        copy(layer.pi, layer.k.pageIndex);
        copy(layer.last, {layer.k.lastPageLen});
        copy(layer.insert, {layer.k.pageIndex.back()});
        copy(layer.offset, {layer.k.lastPageLen - 1});
    }

    /** 构造正式批量追加算子的单请求参数；准备阶段和执行阶段共享同一参数约束。 */
    DataDict AppendArgs(Layer &layer, Data &input, bool key, Data **cache) {
        return {{"pagedCacheManager", key ? layer.km.get() : layer.vm.get()},
                {"currentCaches", reinterpret_cast<Data*>(cache)}, {"input", &input},
                {"insertIndexs", &layer.insert}, {"insertPositions", &layer.offset}};
    }

public:
    /** 创建本rank的所有层页池；由执行器保证设备和线程默认流已经就绪。 */
    ProcessPagedKV(int count, int qh, int kh, int headDim) : qHeads(qh), kvHeads(kh), dim(headDim) {
        CheckPagedCuda(cudaGetDevice(&device));
        RegisterBuiltinCudaAttentionBackends();
        for (int i = 0; i < count; ++i) layers.push_back(NewLayer());
    }

    int PageLen() const { return layers.front()->km->pageLen; }
    int MaxPages() const { return layers.front()->km->maxPages; }

    /**
     * 为接收分配独占页；只分配收到的 Prompt 范围，不提前分配增长预算。
     * @param tokens 接收 Token 数，必须由服务准入预算批准。
     * @return false 表示页池暂时不足，保持原状态；true 表示全部层空间已准备。
     */
    bool PrepareReceive(int tokens) {
        const int pages = (tokens + PageLen() - 1) / PageLen();
        if (tokens <= 0 || pages > MaxPages()) throw std::runtime_error("invalid PD receive length");
        for (const auto &layer : layers) {
            if (!layer->k.pageIndex.empty() || !layer->v.pageIndex.empty())
                throw std::runtime_error("PD receive requires no active pages");
            if (layer->km->FreePageCount() < pages || layer->vm->FreePageCount() < pages) return false;
        }
        for (auto &layer : layers) for (bool key : {true, false}) {
            auto &cache = key ? layer->k : layer->v;
            auto &manager = key ? *layer->km : *layer->vm;
            for (int i = 0; i < pages; ++i) cache.pageIndex.push_back(manager.GetUnusedPageIndex(true));
            cache.lastPageLen = (tokens - 1) % PageLen() + 1;
            cache.Resize({kvHeads, tokens, dim});
        }
        return true;
    }

    /** 接收完成后缩短有效长度；丢弃的页解除请求引用，传输 pin 由调用者单独解除。 */
    void CommitReceive(int tokens) {
        const size_t count = (tokens + PageLen() - 1) / PageLen();
        for (const auto &layer : layers) for (auto *cache : {&layer->k, &layer->v})
            if (tokens < 0 || count > cache->pageIndex.size()) throw std::runtime_error("invalid PD commit length");
        for (auto &layer : layers) for (auto *cache : {&layer->k, &layer->v}) {
            while (cache->pageIndex.size() > count) {
                cache->pagedKVCacheData->ReleasePageIndex(cache->pageIndex.back());
                cache->pageIndex.pop_back();
            }
            cache->lastPageLen = tokens ? (tokens - 1) % PageLen() + 1 : 0;
            cache->Resize({kvHeads, tokens, dim});
        }
    }

    /**
     * 从真实页池生成 IPC 描述，并持有额外页引用；不复制 KV。
     * @param key 本地唯一交接 key；重复导出同 key 必须由服务层复用原描述。
     * @param tokens 有效 Token 数，必须覆盖已有页面范围。
     * @param headBegin 本 rank 第一个全局 KV head。
     * @return 独立 K/V、逐物理页的 region，最后一维为连续 FP16 head_dim。
     */
    std::vector<PDRegion> Export(uint64_t key, int tokens, int headBegin) {
        if (!key || tokens <= 0 || pdPins.count(key)) throw std::runtime_error("invalid or reused PD export key");
        cudaDeviceProp property{};
        CheckPagedCuda(cudaGetDeviceProperties(&property, device));
        std::vector<PDRegion> result;
        std::vector<std::vector<int>> pages;
        const uint64_t stride = uint64_t(kvHeads) * dim * sizeof(half);
        for (size_t index = 0; index < layers.size(); ++index) for (bool keyPart : {true, false}) {
            auto &layer = *layers[index];
            auto &manager = keyPart ? *layer.km : *layer.vm;
            auto &cache = keyPart ? layer.k : layer.v;
            if (cache.dims.size() != 3 || cache.dims[1] < tokens)
                throw std::runtime_error("PD export exceeds valid KV length");
            cudaIpcMemHandle_t handle{};
            CheckPagedCuda(cudaIpcGetMemHandle(&handle, manager.cudaData));
            const size_t count = (tokens + PageLen() - 1) / PageLen();
            pages.emplace_back(cache.pageIndex.begin(), cache.pageIndex.begin() + count);
            for (size_t page = 0; page < count; ++page) {
                PDRegion region;
                region.layer = index; region.component = keyPart ? 0 : 1;
                region.tokenBegin = page * PageLen(); region.tokenEnd = std::min(tokens, int((page + 1) * PageLen()));
                region.headBegin = headBegin; region.headEnd = headBegin + kvHeads;
                region.headDim = dim; region.pageSize = PageLen();
                region.allocationBytes = uint64_t(MaxPages()) * PageLen() * stride;
                region.offset = uint64_t(cache.pageIndex[page]) * PageLen() * stride;
                region.tokenStride = stride; region.headStride = dim * sizeof(half);
                std::memcpy(region.memory.data(), &handle, sizeof(handle));
                std::memcpy(region.device.data(), property.uuid.bytes, 16);
                result.push_back(region);
            }
        }
        for (size_t i = 0; i < layers.size(); ++i) {
            layers[i]->km->Pick(pages[2 * i]); layers[i]->vm->Pick(pages[2 * i + 1]);
        }
        pdPins.emplace(key, std::move(pages));
        return result;
    }

    /** 传输结束后幂等解除额外 pin；不会释放整个 KV 池或活动请求引用。 */
    void ReleaseExport(uint64_t key) {
        auto it = pdPins.find(key);
        if (it == pdPins.end()) return;
        for (size_t i = 0; i < layers.size(); ++i) {
            layers[i]->km->ReleasePageIndices(it->second[2 * i]);
            layers[i]->vm->ReleasePageIndices(it->second[2 * i + 1]);
        }
        pdPins.erase(it);
    }

    bool HasExports() const { return !pdPins.empty(); }
    void SetPreparedPrefill(bool enabled) { pdPrepared = enabled; }

    /** 读取各层K/V池可用页计数，返回占用率最高者；可淘汰前缀页仍可分配。 */
    Qwen3KVUsage Usage() const {
        Qwen3KVUsage result;
        for (const auto &layer : layers) for (auto *manager : {layer->km.get(), layer->vm.get()}) {
            const int total = manager->maxPages, available = manager->FreePageCount();
            if (total <= 0 || available < 0 || available > total)
                throw std::runtime_error("invalid FT paged free-page count");
            if (!result.totalPages || uint64_t(available) * result.totalPages < uint64_t(result.freePages) * total) {
                result.totalPages = uint32_t(total);
                result.freePages = uint32_t(available);
            }
        }
        return result;
    }

    /** 查询所有层K/V的共同完整前缀；至少留一个输入token重算输出。 */
    uint32_t Query(const std::vector<int> &ids, uint32_t position) {
        std::vector<Data*> caches;
        for (auto &layer : layers) { caches.push_back(&layer->k); caches.push_back(&layer->v); }
        return QueryPagedCachePrefix(caches, ids, position);
    }

    /** 绑定跨rank协商后的页数；不进行GPU KV拷贝。 */
    void Restore(const std::vector<int> &ids, uint32_t position, uint32_t tokens) {
        for (auto &layer : layers) {
            std::vector<int> kp, vp;
            layer->km->Query(ids, kp, position); layer->vm->Query(ids, vp, position);
            RestorePagedCachePages(layer->k, *layer->km, kp, tokens / PageLen());
            RestorePagedCachePages(layer->v, *layer->vm, vp, tokens / PageLen());
        }
    }

    /** 把本地完整前缀页复制到已登记的接收页；保持导出地址不变，调用方同步后再发布就绪。 */
    void RestorePrepared(const std::vector<int> &ids, uint32_t position, uint32_t tokens) {
        const uint64_t bytes = uint64_t(PageLen()) * kvHeads * dim * sizeof(half);
        for (auto &layer : layers) for (bool key : {true, false}) {
            auto &manager = key ? *layer->km : *layer->vm;
            auto &cache = key ? layer->k : layer->v;
            std::vector<int> source;
            manager.Query(ids, source, position);
            if (source.size() < tokens / PageLen() || cache.pageIndex.size() < tokens / PageLen())
                throw std::runtime_error("prepared local prefix lost source or target pages");
            auto memory = static_cast<uint8_t*>(manager.cudaData);
            for (size_t page = 0; page < tokens / PageLen(); ++page)
                CheckPagedCuda(cudaMemcpyAsync(memory + uint64_t(cache.pageIndex[page]) * bytes,
                    memory + uint64_t(source[page]) * bytes, bytes, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        }
    }

    /** Prefill成功后仅记录完整Prompt页，Decode不向前缀索引登记生成内容。 */
    void Record(const std::vector<int> &ids, uint32_t position) {
        for (auto &layer : layers) {
            RecordPagedCachePrefix(layer->k, ids, position);
            RecordPagedCachePrefix(layer->v, ids, position);
        }
    }

    /** 返回本执行器持有的固定Decode Q/K/V工作区字节数。 */
    uint64_t GraphBytes() const {
        return uint64_t(graph.capacity) * (qHeads + 2 * kvHeads) * dim * sizeof(half);
    }

    /** Graph已销毁后释放捕获专用工作区；页表和页池属于模型生命周期。 */
    void ReleaseGraphScratch() {
        ReleaseCudaAttentionGraphResources(this);
        graph.q.FreeSpace(); graph.k.FreeSpace(); graph.v.FreeSpace();
        graph.capacity = 0;
    }

    /** 释放请求引用而保留页池和有效前缀；调用方先同步并销毁请求Graph。 */
    void End() {
        for (auto &layer : layers) {
            ReleasePagedCacheReferences(layer->k, true);
            ReleasePagedCacheReferences(layer->v, true);
        }
    }

    /** 清空索引但不释放活动请求正在引用的页。 */
    void ClearPrefix(int selected = -1) {
        if (selected < -1 || selected >= int(layers.size())) throw std::runtime_error("invalid paged cache layer");
        for (size_t i = 0; i < layers.size(); ++i) if (selected < 0 || i == size_t(selected)) {
            layers[i]->km->ClearPrefix(); layers[i]->vm->ClearPrefix();
        }
    }

    /** 在Graph外执行正式追加算子的形状/页面准备，每次Decode仅推进一次元数据。 */
    void PrepareDecode(int past) {
        PrepareScratch(graph, 1);
        for (auto &layer : layers) {
            for (bool key : {true, false}) {
                Data input(DataType::FLOAT16, {1, kvHeads, dim}, DataDevice::CUDA, key ? graph.k.cudaData : graph.v.cudaData);
                Data *cache = key ? &layer->k : &layer->v;
                auto args = AppendArgs(*layer, input, key, &cache);
                appendBatch.Reshape("AppendPagedCacheBatch", args, {}, {});
                cache->Resize({kvHeads, past + 1, dim});
            }
            Metadata(*layer, 1);
        }
    }

    /**
     * 将本rank的融合QKV适配到FT正式分页追加及Attention后端。
     *
     * Eager阶段通过正式追加算子分配页面；Graph阶段只写预先分配的槽位。
     * 后端选择、失败处理和数值语义沿用FT；无状态调用使用独立临时页池。
     * @param index 当前模型层索引。
     * @param tokens 本次新增token数量；prepared为true时必须为1。
     * @param qkv FP16行主序[tokens,qHeads+2*kvHeads,dim]，已完成Norm和RoPE。
     * @param output FP16行主序[tokens,qHeads,dim]，由执行器提供固定地址。
     * @param cached true时使用活动请求的历史页面；false时没有历史KV。
     * @param prepared true时页面和固定地址元数据已在捕获外准备。
     */
    void Run(int index, int tokens, const half *qkv, half *output, bool cached, bool prepared) {
        std::unique_ptr<Layer> temporary;
        if (!cached) temporary = NewLayer((tokens + PageLen() - 1) / PageLen());
        Layer &layer = cached ? *layers.at(index) : *temporary;
        Scratch &scratch = prepared ? graph : eager;
        if (!prepared) PrepareScratch(scratch, tokens);
        int count = tokens * (qHeads + 2 * kvHeads) * dim;
        ProcessUnpackQkv<<<(count + 255) / 256, 256, 0, cudaStreamPerThread>>>(qkv,
            static_cast<half*>(scratch.q.cudaData), static_cast<half*>(scratch.k.cudaData),
            static_cast<half*>(scratch.v.cudaData), tokens, qHeads, kvHeads, dim);
        for (bool key : {true, false}) {
            Data input(DataType::FLOAT16, prepared ? std::vector<int>{1, kvHeads, dim} : std::vector<int>{kvHeads, tokens, dim},
                       DataDevice::CUDA, key ? scratch.k.cudaData : scratch.v.cudaData);
            input.dataDeviceIds = {device};
            Data *cache = key ? &layer.k : &layer.v;
            if (pdPrepared) {
                if (!cached || prepared || cache->dims.size() != 3 || cache->dims[1] < tokens)
                    throw std::runtime_error("invalid PD prepared Prefill append");
                auto &manager = key ? *layer.km : *layer.vm;
                const int past = cache->dims[1] - tokens;
                auto *source = static_cast<const half*>(key ? scratch.k.cudaData : scratch.v.cudaData);
                auto *destination = static_cast<half*>(manager.cudaData);
                for (int begin = past; begin < past + tokens;) {
                    const int page = begin / PageLen(), offset = begin % PageLen();
                    const int count = std::min(PageLen() - offset, past + tokens - begin);
                    for (int head = 0; head < kvHeads; ++head) {
                        CheckPagedCuda(cudaMemcpy2DAsync(
                            destination + (size_t(cache->pageIndex.at(page)) * PageLen() + offset) * kvHeads * dim + head * dim,
                            size_t(kvHeads) * dim * sizeof(half), source + (size_t(head) * tokens + begin - past) * dim,
                            size_t(dim) * sizeof(half), size_t(dim) * sizeof(half), count,
                            cudaMemcpyDeviceToDevice, cudaStreamPerThread));
                    }
                    begin += count;
                }
            } else if (prepared) {
                auto args = AppendArgs(layer, input, key, &cache);
                appendBatch.Run("AppendPagedCacheBatch", args, {}, {});
            } else {
                DataDict args{{"pagedCacheManager", key ? layer.km.get() : layer.vm.get()}, {"cache", cache}, {"input", &input}};
                append.Reshape("AppendPagedCache", args, {}, {});
                append.Run("AppendPagedCache", args, {}, {});
            }
        }
        if (!prepared) Metadata(layer, tokens);
        Data query(DataType::FLOAT16, {qHeads, tokens, dim}, DataDevice::CUDA, scratch.q.cudaData);
        Data result(DataType::FLOAT16, {tokens, qHeads, dim}, DataDevice::CUDA, output);
        query.dataDeviceIds = result.dataDeviceIds = {device};
        CudaAttentionInvocation call;
        call.cacheLayout = CudaAttentionCacheLayout::PAGED;
        call.phase = tokens == 1 ? CudaAttentionPhase::DECODE : CudaAttentionPhase::PREFILL;
        call.dataType = DataType::FLOAT16; call.deviceId = device; call.batch = 1;
        call.group = qHeads / kvHeads; call.headDim = dim; call.scale = 1.0f / sqrtf(float(dim));
        call.attentionType = 1; call.sync = false; call.enableCudaGraph = prepared;
        call.graphResourceOwner = this;
        call.q = &query; call.kCaches = &layer.k; call.vCaches = &layer.v;
        call.qSizes = &layer.qs; call.pageSizes = &layer.ps; call.pageIndexs = &layer.pi;
        call.lastPageLens = &layer.last; call.output = &result;
        std::string error;
        if (!RunCudaAttentionBackend(call, backend, error)) throw std::runtime_error("FT paged attention: " + error);
        CheckPagedCuda(cudaGetLastError());
        if (temporary) CheckPagedCuda(cudaStreamSynchronize(cudaStreamPerThread));
    }

    /** 汇总真实页面资源及所有层共同有效的目录；页池占用与前缀占用分开统计。 */
    Qwen3PrefixInfo Info(const Qwen3PrefixInfo &counters) {
        auto info = counters;
        info.directoryVersion = 2; info.pageLen = PageLen(); info.maxPages = MaxPages(); info.backend = backend;
        info.poolBytes = info.activePages = info.cachedPages = info.deviceBytes = info.evictions = 0;
        info.snapshots.clear();
        for (auto &layer : layers) for (auto *manager : {layer->km.get(), layer->vm.get()}) {
            uint64_t bytes = uint64_t(PageLen()) * kvHeads * dim * sizeof(half);
            info.poolBytes += bytes * manager->maxPages;
            info.cachedPages += manager->pageToTrieNode.size();
            info.deviceBytes += bytes * manager->pageToTrieNode.size();
            info.evictions += manager->evictedPages;
            for (int refs : manager->pageRefCount) info.activePages += refs > 0;
        }
        for (auto entry : layers.front()->km->PrefixDirectory()) {
            size_t pages = entry.second.size() / PageLen();
            for (auto &layer : layers) for (auto *manager : {layer->km.get(), layer->vm.get()}) {
                std::vector<int> found;
                manager->Query(entry.second, found, entry.first);
                pages = std::min(pages, found.size());
            }
            if (pages) info.snapshots.emplace_back(entry.first,
                std::vector<uint32_t>(entry.second.begin(), entry.second.begin() + pages * PageLen()));
        }
        std::sort(info.snapshots.begin(), info.snapshots.end());
        info.snapshots.erase(std::unique(info.snapshots.begin(), info.snapshots.end()), info.snapshots.end());
        info.entries = info.snapshots.size();
        return info;
    }
};

}}}
#endif
