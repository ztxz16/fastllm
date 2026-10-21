#pragma once
#include "models/naive_n05_flash.h"
#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda-kvmem.cuh"

namespace fastllm {
struct NaiveN05FlashModel::SparseKvState {
    struct Rank {
        std::vector<std::unique_ptr<KvMemSparseCache>> layers;
        std::vector<std::unique_ptr<PagedCacheManager>> indexers;
        std::vector<std::pair<Data, Data>> windows;
        Data stageKey, stageValue;
    };
    struct Lease {
        std::shared_ptr<SparseKvState> storage;
        int slot = -1;
        ~Lease() {
            if (!storage || slot < 0) return;
            std::lock_guard<std::mutex> lock(storage->mutex);
            storage->free.push_back(slot);
        }
    };
    std::mutex mutex;
    std::vector<int> free, devices;
    std::vector<std::unique_ptr<Rank>> ranks;
    int requests = 0, tokens = 0;
    std::map<PagedCacheManager *, Data *> indexSidecars;
    ~SparseKvState();
};
}
#endif
