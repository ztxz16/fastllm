#pragma once

#include "devices/cuda/fastllm-mxfp8-projection.h"

struct FastllmNaiveRopeCacheRow;

// One graph slot contains consecutive candidates for the same request.
// The device binding and live lengths may change at every replay.
bool FastllmCudaTryPackedSwaVerify(const fastllm::Data &qkv, const fastllm::Data &positions,
        const fastllm::Data &live, const FastllmNaiveRopeCacheRow &binding,
        const fastllm::Data &sink, int heads, int kvHeads, float theta, float valueScale,
        FastllmMxfp8Activation packed, fastllm::Data &attention, bool overlapQkv = false);
