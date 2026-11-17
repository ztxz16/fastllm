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
        bool AdvanceMetadata(int window) const {
            if (lengthFields.size() != requests.size() || lengthFields.empty() ||
                std::any_of(requests.begin(), requests.end(),
                    [window](const Request &r) { return r.length < window - 1; })) return false;
            for (size_t row = 0; row < requests.size(); ++row) {
                const int length = requests[row].length + 1;
                for (const auto &field : lengthFields[row]) {
                    *field.length = length;
                    if (field.lastPage) *field.lastPage = (length - 1) % field.pageLen + 1;
                }
            }
            return true;
        }
        bool CanAppend() const {
            if (requests.empty() || !std::all_of(requests.begin(), requests.end(),
                [](const Request &r) { return r.length < r.reservedUntil; })) return false;
            // Sharing can change without replacing a descriptor (prefix/PD
            // snapshots call Pick). Check the physical tail under each pool's
            // lock, once per pool rather than once per request/layer.
            for (const auto &group : pageTails) {
                std::lock_guard<std::mutex> guard(group.first->pageIndexLocker);
                for (const auto &entry : group.second) {
                    const auto &cache = *entry.first;
                    const int length = requests[entry.second].length;
                    if (length && length % cache.pageLen &&
                        group.first->pageRefCount[cache.pageIndex[(length - 1) / cache.pageLen]] > 1) return false;
                }
            }
            return true;
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
    int servingLiveRows = 0;
    struct UnsupportedServingLayout {};
    explicit AttentionParallelState(parallel::Topology topology)
        : runtime(topology), groups(topology.attentionDP), ranks(topology.Size()),
          rows(topology.attentionDP, {}), servingGroups(topology.attentionDP), servingRanks(topology.Size()),
          servingRows(topology.attentionDP, {}) {}
};
}
#endif
