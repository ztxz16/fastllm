#pragma once
#include "fastllm.h"

// Single-request GLM hybrid DSA. Position stores the current logical token
// index on the device, so graph replay can advance without host parameters.
bool FastllmCudaHybridDsaSupported();
// Single-token FP32 projections retain the original 256-thread accumulation
// tree and BF16 boundaries, without intermediate activation conversions.
bool FastllmCudaHybridDsaProjections(const fastllm::Data &input, const fastllm::Data &normalized,
    const fastllm::Data &keyWeight, const fastllm::Data &gateWeight,
    const fastllm::Data &queryWeight, const fastllm::Data &headWeight,
    const fastllm::Data &gamma, const fastllm::Data &beta,
    fastllm::Data &key, fastllm::Data &gate, fastllm::Data &query, fastllm::Data &head);
// Preserve the BF16 input/scale boundaries of the indexer's FP8 round trip.
bool FastllmCudaHybridDsaQuantizeRotated(const fastllm::Data &rotated, fastllm::Data &output);
bool FastllmCudaHybridDsaAppendPool(const fastllm::Data &key, const fastllm::Data &gate,
    const fastllm::Data &ape, const fastllm::Data &position,
    fastllm::Data &tailKeys, fastllm::Data &tailGates, fastllm::Data &pooled);
bool FastllmCudaHybridDsaStorePool(const fastllm::Data &pooled,
    const fastllm::Data &position, fastllm::Data &keys);
bool FastllmCudaHybridDsaAdvance(fastllm::Data &position);
bool FastllmCudaHybridDsaScore(const fastllm::Data &query, const fastllm::Data &weights,
    const fastllm::Data &keys, const fastllm::Data &position, fastllm::Data &scores);
bool FastllmCudaHybridDsaSelect(const fastllm::Data &scores, const fastllm::Data &position,
    const fastllm::Data &pageTable, int pageLen, fastllm::Data &indices);
