#pragma once
#include "devices/cuda/fastllm-v41-full-decode.h"
#include <cuda_runtime_api.h>

// Optional SM120 implementation. A null plan leaves native attention selected.
// The caller owns operands and destroys graphs before releasing this workspace.
void *FastllmV41FlashInferCreate(const FastllmV41FullDecodeConfig &config);
void FastllmV41FlashInferDestroy(void *plan);
cudaError_t FastllmV41FlashInferRun(void *plan, const FastllmV41FullDecodeBuffers &buffers,
                                    const FastllmV41FullDecodeRequest *requests, cudaStream_t stream);
