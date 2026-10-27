#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>

// Reuses the Dots3 per-128 E4M3 quantizer; rows are 128-element BF16 groups.
cudaError_t FastllmCudaGlm5NextQuantizeLatentRaw(const void *input,
    void *bytes, float *scales, int rows, cudaStream_t stream);

// Compiled separately for sm_120a. Cache rows contain 512 E4M3 bytes and
// four FP32 scales; indices are padded to a multiple of 64 with -1.
cudaError_t FastllmCudaGlm5NextDsaPrefillSm120Raw(const void *query,
    const void *cache, const int32_t *indices, void *output, float *lse,
    int queries, int width, float scale, cudaStream_t stream);
