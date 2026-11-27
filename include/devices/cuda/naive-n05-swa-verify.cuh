#pragma once
#include <vector>

#include "devices/cuda/fastllm-mxfp8-projection.h"

struct FastllmNaiveRopeCacheRow;

// One graph slot contains consecutive candidates for the same request.
// The device binding and live lengths may change at every replay.
bool FastllmCudaTryPackedSwaVerify(const fastllm::Data &qkv, const fastllm::Data &positions,
        const fastllm::Data &live, const FastllmNaiveRopeCacheRow &binding,
        const fastllm::Data &sink, int heads, int kvHeads, float theta, float valueScale,
        FastllmMxfp8Activation packed, fastllm::Data &attention, bool overlapQkv = false);

bool FastllmCudaTryPackedSwaVerify(const fastllm::Data &qkv, const fastllm::Data &positions,
        const fastllm::Data &live, const FastllmNaiveRopeCacheRow *bindings,
        const fastllm::Data &sink, int heads, int kvHeads, float theta, float valueScale,
        FastllmMxfp8Activation packed, fastllm::Data &attention, bool overlapQkv,
        const int *requestLengths, int owners);

#ifdef FASTLLM_ENABLE_NAIVE_HOPPER_VERIFY
const FastllmMxfp8ProjectionBackend *FastllmCudaSm90SwaOutputProjection();
#endif

bool FastllmCudaTryPackedSwaVerify(const fastllm::Data &qkv, const fastllm::Data &positions,
        const fastllm::Data &live, const FastllmNaiveRopeCacheRow &binding,
        const fastllm::Data &sink, int heads, int kvHeads, float theta, float valueScale,
        FastllmMxfp8Activation packed, fastllm::Data &attention, bool overlapQkv,
        int requestRows, int bindingStride = 0, const std::vector<int> *requestLengths = nullptr);

bool FastllmCudaCanPackedSwaVerify(const FastllmNaiveRopeCacheRow &binding, int rows, int heads, int kvHeads, int requests = 1, int maxWidth = 0);
