#pragma once
#include "devices/cuda/fastllm-v41-full-decode.h"
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <cstring>

// Default on for eligible inputs; =0 disables both ordinary and full-plan paths.
inline bool FastllmV41FlashInferEnabled() {
    const char *value = std::getenv("FASTLLM_DSV41_FLASHINFER");
    return value == nullptr || std::strcmp(value, "1") == 0;
}

// Optional SM120 implementation. A null plan leaves native attention selected.
// The caller owns operands and destroys graphs before releasing this workspace.
void *FastllmV41FlashInferCreate(const FastllmV41FullDecodeConfig &config);
void FastllmV41FlashInferDestroy(void *plan);
cudaError_t FastllmV41FlashInferRun(void *plan, const FastllmV41FullDecodeBuffers &buffers,
                                    const FastllmV41FullDecodeRequest *requests, cudaStream_t stream);

// Ordinary sparse-attention entry. Indices are already causally filtered by
// the model's indexer, exactly as for the native sparse-attention operator.
// Workspace belongs to the caller's stream/graph-aware allocation pool.
struct FastllmV41FlashInferChunkConfig {
    int batches = 1, length = 1, heads = 64, topK = 0, compressedRows = 0, startPosition = 0;
    float scale = 1.f;
};
// Returns zero when unsupported or when kernel setup cannot run during capture.
size_t FastllmV41FlashInferChunkWorkspace(const FastllmV41FlashInferChunkConfig &config);
cudaError_t FastllmV41FlashInferChunkRun(const FastllmV41FlashInferChunkConfig &config,
    const FastllmV41FullDecodeBuffers &buffers, const void *ring, const void *compressed,
    void *workspace, cudaStream_t stream);
