#pragma once
#include "models/naive_n05_flash.h"
#ifdef USE_CUDA
#include "parallel/cuda_runtime.h"
#include "utils/cuda_decode_sampling.h"
#include <algorithm>
#include <condition_variable>

namespace fastllm {
struct NaiveN05FlashModel::AttentionParallelState {
    // Only the owned scheduler may reuse descriptors. Identities change before
    // cache mutation and on request recycling, including reuse of the same address.
    struct RequestPlan {
        bool valid = false;
        DataType kvType = DATA_AUTO_NONE;
        std::weak_ptr<TPDecodeState> graph;
        uint64_t epoch = 0;
        struct Request {
            const Data *key = nullptr;
            uint64_t identity = 0;
            int length = 0;
            int reservedUntil = 0;
        };
        std::vector<Request> requests;
        std::map<PagedCacheManager *, std::vector<std::pair<const Data *, int>>> pageTails;
        struct LengthField { int *length, *lastPage; int pageLen; };
        std::vector<std::vector<LengthField>> lengthFields;
        void AdvanceRow(size_t row, int count = 1) const {
            const int length = requests[row].length + count;
            for (const auto &field : lengthFields[row]) {
                *field.length = length;
                if (field.lastPage) *field.lastPage = (length - 1) % field.pageLen + 1;
            }
        }
        bool AdvanceMetadata(int window) const {
            if (lengthFields.size() != requests.size() || lengthFields.empty() ||
                std::any_of(requests.begin(), requests.end(),
                    [window](const Request &r) { return r.length < window - 1; })) return false;
            for (size_t row = 0; row < requests.size(); ++row) AdvanceRow(row);
            return true;
        }
        // Caller holds the page pool's lock.
        static bool HasSharedTail(const Data &cache, int length) {
            return length > 0 && length % cache.pageLen &&
                cache.pagedKVCacheData->pageRefCount[cache.pageIndex[(length - 1) / cache.pageLen]] > 1;
        }
        bool CanAppend(const std::vector<int> *counts = nullptr) const {
            if (requests.empty() || (counts && counts->size() != requests.size())) return false;
            for (size_t i = 0; i < requests.size(); ++i)
                if (requests[i].reservedUntil - requests[i].length < (counts ? (*counts)[i] : 1)) return false;
            // Sharing can change without replacing a descriptor (prefix/PD
            // snapshots call Pick). Check the physical tail under each pool's
            // lock, once per pool rather than once per request/layer.
            for (const auto &group : pageTails) {
                std::lock_guard<std::mutex> guard(group.first->pageIndexLocker);
                if (!group.first->HasSharedPages()) continue;
                for (const auto &entry : group.second)
                    if (HasSharedTail(*entry.first, requests[entry.second].length)) return false;
            }
            return true;
        }
        // Request identities and order have already matched. Keep the plan and
        // its per-pool descriptor lists in place across page growth or tail COW.
        std::vector<int> RetainInOrder(const std::vector<int> *counts) {
            std::vector<int> source(requests.size());
            for (size_t i = 0; i < requests.size(); ++i) {
                const int append = counts ? (*counts)[i] : 1;
                source[i] = requests[i].reservedUntil - requests[i].length >= append ? int(i) : -1;
            }
            for (const auto &group : pageTails) {
                std::lock_guard<std::mutex> guard(group.first->pageIndexLocker);
                if (!group.first->HasSharedPages()) continue;
                for (const auto &entry : group.second) {
                    const int row = entry.second;
                    if (source[row] >= 0 && HasSharedTail(*entry.first, requests[row].length)) source[row] = -1;
                }
            }
            if (!lengthFields.empty())
                for (size_t i = 0; i < source.size(); ++i)
                    if (source[i] < 0) lengthFields[i].clear();
            return source;
        }
        // Move surviving request metadata into the new batch order. Old roots
        // may have been freed or recycled, so only inspect live batch roots.
        std::vector<int> Reorder(const TargetBatch &batch, int count,
                const std::unordered_map<const Data *, uint64_t> &live, const std::vector<int> *append) {
            std::unordered_map<const Data *, int> oldRows;
            for (size_t i = 0; i < requests.size(); ++i) oldRows.emplace(requests[i].key, i);
            std::vector<int> source(count, -1), destination(requests.size(), -1);
            std::vector<Request> next(count);
            std::vector<std::vector<LengthField>> fields(count);
            for (int row = 0; row < count; ++row) {
                const Data &key = batch.Key(row, 0);
                auto old = oldRows.find(&key);
                auto id = live.find(&key);
                if (old == oldRows.end() || id == live.end() || key.dims.size() != 3) continue;
                const int index = old->second;
                const auto &request = requests[index];
                if (request.identity != id->second || request.length != key.dims[1] ||
                    request.reservedUntil - request.length < (append ? (*append)[row] : 1)) continue;
                source[row] = index;
                destination[index] = row;
                next[row] = request;
                if (!lengthFields.empty()) fields[row] = std::move(lengthFields[index]);
            }
            // Another request may have acquired a reference to our last page
            // without changing the descriptor. Only that row needs reservation.
            for (auto &group : pageTails) {
                std::lock_guard<std::mutex> guard(group.first->pageIndexLocker);
                for (auto &tail : group.second) {
                    const int row = destination[tail.second];
                    tail.second = row;
                    if (row >= 0 && source[row] >= 0 && HasSharedTail(*tail.first, next[row].length))
                        source[row] = -1;
                }
            }
            for (auto &group : pageTails) {
                auto &tails = group.second;
                tails.erase(std::remove_if(tails.begin(), tails.end(),
                    [&](const auto &tail) { return tail.second < 0 || source[tail.second] < 0; }), tails.end());
            }
            requests.swap(next);
            lengthFields.swap(fields);
            for (int row = 0; row < count; ++row)
                if (source[row] < 0) lengthFields[row].clear();
            return source;
        }
        bool Matches(const TargetBatch &batch, uint64_t currentEpoch,
                const std::unordered_map<const Data *, uint64_t> &live, int liveRows) const {
            if (epoch != currentEpoch || requests.size() != (size_t)liveRows) return false;
            for (size_t row = 0; row < requests.size(); ++row) {
                const Data &root = batch.Key(row, 0);
                auto id = live.find(&root);
                const auto &request = requests[row];
                if (&root != request.key || id == live.end() || id->second != request.identity ||
                    root.dims.size() != 3 || root.dims[1] != request.length) return false;
            }
            return true;
        }
        bool Remember(const TargetBatch &batch, uint64_t currentEpoch,
                const std::unordered_map<const Data *, uint64_t> &live, int liveRows,
                const std::vector<int> *limits = nullptr) {
            if (limits && limits->size() != (size_t)liveRows) return false;
            const bool sameEpoch = epoch == currentEpoch;
            epoch = currentEpoch;
            requests.resize(liveRows);
            for (size_t row = 0; row < requests.size(); ++row) {
                const Data &root = batch.Key(row, 0);
                auto id = live.find(&root);
                if (id == live.end() || root.dims.size() != 3) return false;
                const auto &old = requests[row];
                const int limit = limits ? (*limits)[row] :
                    sameEpoch && old.key == &root && old.identity == id->second ? old.reservedUntil : 0;
                requests[row] = {&root, id->second, root.dims[1], limit};
            }
            return true;
        }
    };
    RequestPlan plan;
    bool reuseBindings = false;
    // Previous physical row on the same attention owner, or -1 to validate.
    std::vector<int> previousBindingRows;
    struct GroupBatch {
        std::vector<std::pair<Data *, Data *>> kv;
        std::vector<GenerationConfig> configs;
        std::vector<int> lengths;
        std::vector<float> positions;
        std::vector<int> sequenceIndices;
        std::unique_ptr<TargetBatch> batch;
    };
    struct Rank {
        Data hidden, normalized, projected, reduction, positions, liveKeys;
        TargetWorkspace workspace;
    };
    parallel::CudaRuntime runtime;
    std::unique_ptr<parallel::CudaRowTransfer> transfer;
    std::vector<GroupBatch> groups;
    std::vector<Rank> ranks;
    parallel::RowPartition rows;
    // Outside the captured model graph. Only each attention group's leader
    // owns full-vocabulary logits; peers write their disjoint vocab columns.
    std::vector<Data> samplingLogits;
    struct SamplingShard {
        Data workspace, selected, received;
    };
    std::vector<SamplingShard> samplingShards;
    std::vector<std::shared_ptr<CudaDecodeSampling>> sampling;
    int samplingCapacity = 0;
    // Serving graphs own separate DP scratch: a large prefill must not move
    // buffers retained by a decode graph. All serving shapes share this arena.
    std::unique_ptr<parallel::CudaRowTransfer> servingTransfer;
    std::vector<GroupBatch> servingGroups;
    std::vector<Rank> servingRanks;
    parallel::RowPartition servingRows;
    RequestPlan servingPlan;
    std::vector<std::pair<Data, Data>> padding;
    bool servingPadded = false;
    bool stablePrefillReduction = false;
    int servingLiveRows = 0;
    const std::vector<int> *verifyLengths = nullptr;
    struct UnsupportedServingLayout {};
    static bool SupportsStablePrefillReduction(const std::vector<int> &architectures) {
        return !architectures.empty() && std::all_of(architectures.begin(), architectures.end(),
            [](int arch) { return arch == 120; });
    }
    explicit AttentionParallelState(parallel::Topology topology)
        : runtime(topology), groups(topology.attentionDP), ranks(topology.Size()),
          rows(topology.attentionDP, {}), servingGroups(topology.attentionDP), servingRanks(topology.Size()),
          servingRows(topology.attentionDP, {}) {}
};
}
#endif
