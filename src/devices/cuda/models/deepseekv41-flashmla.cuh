#pragma once
#include "devices/cuda/fastllm-v41-full-decode.h"
#include <cuda_runtime_api.h>
#include <memory>

// One graph-owned workspace per distinct shape. The parent only shares it
// between layers ordered by a hidden-state dependency.
struct FastllmV41MlaWorkspace {
    void *storage = nullptr, *query = nullptr, *keys = nullptr, *codes = nullptr;
    float *angles = nullptr, *maxima = nullptr, *lse = nullptr;
    int *positions = nullptr, *indices = nullptr, *lengths = nullptr;
    uint32_t *scales = nullptr;
    bool nativeQuery = false, stageNative = false;
    bool paged = false, outputOnly = false, bf16Rounding = false, headMajor = false;
    int batch = 0, width = 0, pitch = 0, scalePitch = 0, device = -1;
    ~FastllmV41MlaWorkspace();
};

std::shared_ptr<FastllmV41MlaWorkspace> FastllmV41CreateMla(const FastllmV41FullDecodeConfig &c,
                                                            bool outputOnly = false);
cudaError_t FastllmV41RunMla(const FastllmV41MlaWorkspace &w, const FastllmV41FullDecodeConfig &c,
                             const FastllmV41FullDecodeBuffers &buffers,
                             const FastllmV41FullDecodeRequest *requests, cudaStream_t stream, bool unpack);

bool FastllmV41IsMlaHistoryKernel(const void *func);
cudaError_t FastllmV41RunMla(const FastllmV41MlaWorkspace &w, const FastllmV41FullDecodeConfig &c,
                             const FastllmV41FullDecodeBuffers &buffers, const FastllmV41FullDecodeRequest *requests,
                             cudaStream_t stream, bool unpack, bool historyOnly, bool allIndices = false);
bool FastllmV41IsMlaPageKernel(const void *func);

void FastllmV41MlaCore(const FastllmV41MlaWorkspace &w, const FastllmV41FullDecodeConfig &c,
                       const FastllmV41FullDecodeBuffers &b, cudaStream_t stream);
