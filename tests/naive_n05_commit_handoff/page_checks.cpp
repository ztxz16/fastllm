// Diagnostic interposer for real verification requests, never a serving library.
// Retain an actual tail page without changing the binding epoch: a fresh COW
// check must notice the additional owner even when all descriptors are reused.
#include "naive_n05_tp.h"
#include <cuda_runtime_api.h>
#include <atomic>
#include <cstring>
#include <dlfcn.h>
#include <stdexcept>
#include <vector>

namespace {
std::atomic<bool> enabled{false};
std::atomic<unsigned long long> checked[65]{}, copied[65]{};
void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
std::vector<unsigned char> ReadBytes(const void *pointer, size_t bytes, int device) {
    int previous;
    Require(cudaGetDevice(&previous) == cudaSuccess, "page audit: current device");
    Require(cudaSetDevice(device) == cudaSuccess, "page audit: select device");
    std::vector<unsigned char> result(bytes);
    auto status = cudaMemcpy(result.data(), pointer, bytes, cudaMemcpyDeviceToHost);
    auto restored = cudaSetDevice(previous);
    Require(status == cudaSuccess && restored == cudaSuccess, "page audit: copy/restore");
    return result;
}
struct RetainedTail {
    fastllm::PagedCacheManager *manager = nullptr;
    fastllm::Data *local = nullptr;
    int oldPage = -1, slot = -1, device = -1;
    size_t bytes = 0, sideBytes = 0;
    unsigned char *side = nullptr;
    std::vector<unsigned char> contents, sideContents;
    ~RetainedTail() { if (manager) manager->ReleasePageIndex(oldPage); }
};
}
extern "C" void fastllm_page_checks_enable(int value) { enabled = value != 0; }
extern "C" unsigned long long fastllm_page_checks_count(int batch, int cow) {
    return batch > 0 && batch <= 64 ? (cow ? copied[batch].load() : checked[batch].load()) : 0;
}

namespace fastllm {
bool NaiveN05FlashModel::PrepareTensorParallelDecode(const Data &ids,
        std::vector<std::pair<Data, Data>> &kv, bool verify, const LogitsSelection &selection,
        std::vector<int *> &lengths, const TargetBatch *batch, bool hidden, bool reuseOnly) {
    using Function = bool (*)(NaiveN05FlashModel *, const Data &, std::vector<std::pair<Data, Data>> &,
        bool, const LogitsSelection &, std::vector<int *> &, const TargetBatch *, bool, bool);
    static auto original = reinterpret_cast<Function>(dlsym(RTLD_NEXT,
        "_ZN7fastllm18NaiveN05FlashModel27PrepareTensorParallelDecodeERKNS_4DataERSt6vectorISt4pairIS1_S1_ESaIS6_EEbRKNS0_15LogitsSelectionERS4_IPiSaISD_EEPKNS0_11TargetBatchEbb"));
    Require(original != nullptr, "page audit: missing real prepare");
    const int count = batch ? batch->Size() : 0;
    const bool audit = enabled && verify && reuseOnly && count > 0 && count <= 64 &&
        !preparingGraphs && mainLoop && mainLoop->get_id() == std::this_thread::get_id() &&
        !tpPagedCaches.empty();
    RetainedTail tail;
    if (audit && !copied[count]) {
        // Different batches probe different ranks, K/V parts and request rows.
        const int rank = count % tpDevices.size(), part = count % 2, row = count - 1;
        for (int layer = 0; layer < block_cnt && !tail.manager; ++layer) if (!slidingLayers[layer]) {
            Data &root = part ? batch->Value(row, layer) : batch->Key(row, layer);
            auto found = root.multiDeviceDatas.find(tpDevices[rank]);
            if (found == root.multiDeviceDatas.end() || !found->second) continue;
            Data &local = *found->second;
            auto &manager = *tpPagedCaches[rank][layer * 2 + part];
            if (!local.isPagedKVCache || local.pagedKVCacheData != &manager ||
                local.dims.size() != 3 || local.dims[1] <= 0 || local.pageLen <= 0 ||
                local.dims[1] % local.pageLen == 0) continue;
            const int slot = (local.dims[1] - 1) / local.pageLen;
            if (slot >= (int)local.pageIndex.size()) continue;
            const int page = local.pageIndex[slot];
            {
                std::lock_guard<std::mutex> guard(manager.pageIndexLocker);
                if (manager.pageRefCount[page] != 1) continue;
            }
            tail.local = &local; tail.oldPage = page; tail.slot = slot; tail.device = tpDevices[rank];
            tail.bytes = GetDataBytes(manager.dataType, manager.pageLen, manager.dims[3]);
            tail.contents = ReadBytes((unsigned char *)manager.cudaData + page * tail.bytes, tail.bytes, tail.device);
            auto sidecar = tpIndexerSidecars.find(&manager);
            if (sidecar != tpIndexerSidecars.end()) {
                auto &side = *sidecar->second;
                tail.side = (unsigned char *)side.cudaData;
                tail.sideBytes = size_t(side.dims[1]) * side.dims[3] * side.unitSize;
                tail.sideContents = ReadBytes(tail.side + page * tail.sideBytes, tail.sideBytes, tail.device);
            }
            std::vector<int> retain{page};
            manager.Pick(retain);
            tail.manager = &manager;
        }
    }
    bool ok = original(this, ids, kv, verify, selection, lengths, batch, hidden, reuseOnly);
    if (ok && audit) {
        for (int rank = 0; rank < (int)tpDevices.size(); ++rank)
        for (int layer = 0; layer < block_cnt; ++layer) if (!slidingLayers[layer])
        for (int part = 0; part < 2; ++part) {
            auto &manager = *tpPagedCaches[rank][layer * 2 + part];
            std::lock_guard<std::mutex> guard(manager.pageIndexLocker);
            for (int row = 0; row < count; ++row) {
                Data &root = part ? batch->Value(row, layer) : batch->Key(row, layer);
                Data &local = *root.multiDeviceDatas.at(tpDevices[rank]);
                const int end = batch->Key(row, 0).dims[1] + batch->lengths[row];
                Require((int64_t)local.pageIndex.size() * local.pageLen >= end, "page audit: missing physical capacity");
                const int previous = local.dims[1];
                if (previous && previous % local.pageLen)
                    Require(manager.pageRefCount[local.pageIndex[(previous - 1) / local.pageLen]] == 1,
                            "page audit: shared writable tail");
            }
        }
        ++checked[count];
        if (tail.manager) {
            const int page = tail.local->pageIndex[tail.slot];
            Require(page != tail.oldPage, "page audit: retained tail was not copied");
            Require(ReadBytes((unsigned char *)tail.manager->cudaData + page * tail.bytes,
                              tail.bytes, tail.device) == tail.contents, "page audit: copied KV differs");
            if (tail.side)
                Require(ReadBytes(tail.side + page * tail.sideBytes, tail.sideBytes, tail.device) == tail.sideContents,
                        "page audit: copied indexer sidecar differs");
            const auto table = ReadBytes((int *)tail.local->cudaData + tail.slot, sizeof(int), tail.device);
            int uploaded;
            std::memcpy(&uploaded, table.data(), sizeof(uploaded));
            Require(uploaded == page, "page audit: stale GPU page table");
            ++copied[count];
        }
    }
    return ok;
}
}
