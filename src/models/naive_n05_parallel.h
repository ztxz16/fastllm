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
        uint64_t epoch = 0;
        struct Request {
            const Data *key;
            uint64_t identity;
            int length;
        };
        std::vector<Request> requests;
        bool Matches(const TargetBatch &batch, uint64_t currentEpoch,
                const std::unordered_map<const Data *, uint64_t> &live) const {
            if (epoch != currentEpoch || requests.size() != (size_t)batch.Size()) return false;
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
                const std::unordered_map<const Data *, uint64_t> &live) {
            epoch = currentEpoch;
            requests.resize(batch.Size());
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
    bool planValid = false, reuseBindings = false;
    std::weak_ptr<TPDecodeState> validatedGraph;
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
    std::atomic<size_t> nextGroup{0};
    DataType planKVType = DATA_AUTO_NONE;
    explicit AttentionParallelState(parallel::Topology topology)
        : runtime(topology), groups(topology.attentionDP), ranks(topology.Size()),
          rows(topology.attentionDP, {}) {}
};
}
#endif
