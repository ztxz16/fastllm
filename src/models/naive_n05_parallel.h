#pragma once
#include "models/naive_n05_flash.h"
#ifdef USE_CUDA
#include "parallel/cuda_runtime.h"
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
            const Data *key;
            uint64_t identity;
            int length;
        };
        std::vector<Request> requests;
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
                const std::unordered_map<const Data *, uint64_t> &live, int liveRows) {
            epoch = currentEpoch;
            requests.resize(liveRows);
            for (size_t row = 0; row < requests.size(); ++row) {
                const Data &root = batch.Key(row, 0);
                auto id = live.find(&root);
                if (id == live.end() || root.dims.size() != 3) return false;
                requests[row] = {&root, id->second, root.dims[1]};
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
    std::atomic<size_t> nextGroup{0};
    explicit AttentionParallelState(parallel::Topology topology)
        : runtime(topology), groups(topology.attentionDP), ranks(topology.Size()),
          rows(topology.attentionDP, {}), servingGroups(topology.attentionDP), servingRanks(topology.Size()),
          servingRows(topology.attentionDP, {}) {}
};
}
#endif
