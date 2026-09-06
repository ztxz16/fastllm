#pragma once

#include "fastllm.h"

// Native HIP implementation of the 256-expert fused routers. The caller owns
// the device buffers and checks the asynchronous launch status.
void FastllmRocmFusedSelectExpert256(
    const void *logits, fastllm::DataType dataType, const void *bias,
    int biasType, int32_t *index, float *score, int tokens, int topk,
    bool sigmoid, bool needNorm, float routeScale);
