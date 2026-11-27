#pragma once
#include "devices/cuda/fastllm-mxfp8-projection.h"
struct FastllmNaiveDecodeScratch;

// One local graph slot. The cache binding and visible lengths are device inputs.
// Opt in to overlap only when normalized input, live rows and cache bindings
// are ready independently of the preceding QKV projection.
bool FastllmCudaTryDpDSAVerify(const fastllm::Data &normalized, const fastllm::Data &qkv,
        const fastllm::Data &wk, const fastllm::Data &wq, const fastllm::Data &weights,
        const fastllm::Data &keyNorm, const fastllm::Data &keyBias, const fastllm::Data &sink,
        const fastllm::Data &positions, const fastllm::Data &live, const fastllm::Data &key,
        const fastllm::Data &value, const void *const *binding, int capacity, float theta,
        float valueScale, FastllmMxfp8Activation packed, FastllmNaiveDecodeScratch &scratch,
        fastllm::Data &attention, bool overlapQkv = false);

struct FastllmNaiveRopeCacheRow;
bool FastllmCudaTryDpDSAVerify(const fastllm::Data &normalized, const fastllm::Data &qkv,
        const fastllm::Data &wk, const fastllm::Data &wq, const fastllm::Data &weights,
        const fastllm::Data &keyNorm, const fastllm::Data &keyBias, const fastllm::Data &sink,
        const fastllm::Data &positions, const fastllm::Data &live, const fastllm::Data &key,
        const fastllm::Data &value, const void *const *binding, int capacity, float theta,
        float valueScale, FastllmMxfp8Activation packed, FastllmNaiveDecodeScratch &scratch,
        fastllm::Data &attention, bool overlapQkv, const FastllmNaiveRopeCacheRow *second, int firstRows);

struct FastllmNaiveVerifyAttentionInput;
bool FastllmCudaSm90VerifyDsaBatch(const FastllmNaiveVerifyAttentionInput &input, FastllmMxfp8Activation packed);
bool FastllmCudaSm90ProjectVerifyIndexer(const fastllm::Data &input, const fastllm::Data &wq,
        const fastllm::Data &wk, const fastllm::Data &weights, fastllm::Data &out, bool &planar);

bool FastllmCudaTryDpDSAVerify(const fastllm::Data &normalized, const fastllm::Data &qkv,
        const fastllm::Data &wk, const fastllm::Data &wq, const fastllm::Data &weights,
        const fastllm::Data &keyNorm, const fastllm::Data &keyBias, const fastllm::Data &sink,
        const fastllm::Data &positions, const fastllm::Data &live, const fastllm::Data &key,
        const fastllm::Data &value, const void *const *binding, int capacity, float theta,
        float valueScale, FastllmMxfp8Activation packed, FastllmNaiveDecodeScratch &scratch,
        fastllm::Data &attention, bool overlapQkv, int requestRows,
        const fastllm::Data *scoreArena = nullptr, fastllm::Data *queries = nullptr,
        const fastllm::Data *indexPool = nullptr, const std::vector<int> *requestLengths = nullptr);

// Checks the grouped block-scaled decoder on the current device.
bool FastllmCudaHasGroupedFP4DSADecode();

bool FastllmCudaCanBatchDpDSAVerify(const std::vector<FastllmNaiveRopeCacheRow> &caches, int width,
        int capacity, const fastllm::Data &scores, const fastllm::Data *query, const fastllm::Data *indexPool = nullptr, const std::vector<int> *requestLengths = nullptr);
