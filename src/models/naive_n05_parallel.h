#pragma once
#include "models/naive_n05_flash.h"
#ifdef USE_CUDA
#include "parallel/cuda_runtime.h"
#include <condition_variable>

namespace fastllm {
struct NaiveN05FlashModel::AttentionParallelState {
    struct GroupBatch {
        std::vector<std::pair<Data *, Data *>> kv;
        std::vector<GenerationConfig> configs;
        std::vector<int> lengths;
        std::vector<float> positions;
        std::vector<int> sequenceIndices;
        std::unique_ptr<TargetBatch> batch;
    };
    struct Rank {
        Data normalized, projected, reduction, positions, liveKeys;
        TargetWorkspace workspace;
    };
    parallel::CudaRuntime runtime;
    std::unique_ptr<parallel::CudaRowTransfer> transfer;
    std::vector<GroupBatch> groups;
    std::vector<Rank> ranks;
    parallel::RowPartition rows;
    std::atomic<size_t> nextGroup{0};
    explicit AttentionParallelState(parallel::Topology topology)
        : runtime(topology), groups(topology.attentionDP), ranks(topology.Size()),
          rows(topology.attentionDP, {}) {}
};
}
#endif
