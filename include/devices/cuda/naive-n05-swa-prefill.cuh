#pragma once
#include "fastllm.h"
#include "fastllm-mxfp8-projection.h"

// All pointers refer to disjoint request slices that live through the layer.
// The caller stages the chronological KV before the rank's rolling buffer is reused.
struct FastllmNaiveStagedSwa {
    const void *query, *key, *value;
    void *output;
    int rows, stride, keys, past, endTile;
};

void FastllmCudaNaivePrefillAttention(const fastllm::Data &query, const fastllm::Data &key,
    const fastllm::Data &value, const fastllm::Data &indices, const fastllm::Data &sink, int heads,
    int kvHeads, int dim, int valueDim, int pastLength, int window, fastllm::Data &output,
    bool causal, const fastllm::Data *scratch, std::vector<FastllmNaiveStagedSwa> *pending,
    FastllmMxfp8PrefillOutput packed = {});

void FastllmCudaNaiveFlushSwa(const std::vector<FastllmNaiveStagedSwa> &pending,
    const fastllm::Data &sink, int heads, int kvHeads);
