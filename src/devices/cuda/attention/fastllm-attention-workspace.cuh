#pragma once
#include "fastllm-attention-common.cuh"

struct FastllmCudaTempDeviceBuffer {
    int device = -1;
    void *data = nullptr;
    size_t size = 0;

    explicit FastllmCudaTempDeviceBuffer(int device) : device(device) {}

    ~FastllmCudaTempDeviceBuffer();
};

struct FlashInferWorkSpaceManager {
    const size_t float_workspace_size;
    size_t int_workspace_size = 1024 * 1024;

    std::mutex plan_mutex;
    void* d_float_workspace = nullptr;
    void* d_int_workspace = nullptr;
    void* h_page_locked_int_workspace = nullptr;

#ifdef FASTLLM_ENABLE_FLASHINFER
    struct MLADecodePlan : FastllmCudaTempDeviceBuffer {
        flashinfer::MLAPlanInfo info;
        uint64_t last_used = 0;
        explicit MLADecodePlan(int device) : FastllmCudaTempDeviceBuffer(device) {}
    };
    // Sparse decode keys contain the head shape and ordered request KV lengths.
    // Token indices and tensor addresses remain per-call kernel arguments.
    std::map<std::vector<int>, std::unique_ptr<MLADecodePlan>> mla_decode_plans;
    uint64_t mla_plan_clock = 0;
#endif

    // Integer schedules are usually only a few KiB. Size the staging arena
    // from the counting planner; keep the float arena (kernel split policy)
    // unchanged. Call under plan_mutex, outside stream capture.
    void EnsureIntCapacity(size_t required);

    FlashInferWorkSpaceManager();

    ~FlashInferWorkSpaceManager();
};

FlashInferWorkSpaceManager &getFastllmFlashInferWorkSpace();
