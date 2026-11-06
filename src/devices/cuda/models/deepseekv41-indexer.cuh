#pragma once

#include <cstdint>

namespace fastllm_v41_indexer {
// SM120/121 implementation; the portable entry point below validates inputs.
bool Fp4Prefill(const void *q, const float *weights, const uint8_t *keys,
               float *output, void *scratch, int batch, int tokens, int count,
               int ratio, int position);
}

namespace fastllm { class Data; }

// Q must already contain the model's E2M1/UE8M0 block-32 round trip.
// Unsupported devices, formats and scratch budgets use the generic scorer.
extern "C" bool FastllmCudaDeepSeekV41IndexerScoreFp4Query(
    const fastllm::Data &q, const fastllm::Data &weights, const fastllm::Data &k,
    int ratio, int startPos, fastllm::Data &output);
