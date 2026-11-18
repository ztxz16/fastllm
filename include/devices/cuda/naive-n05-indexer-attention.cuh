#pragma once
#include "devices/cuda/naive-n05-cuda.cuh"

// True only if this build contains a cubin for the active CUDA device.
bool FastllmCudaHasBlackwellIndexerAttention();

// Only the model Indexer guarantees unique logical positions. Arbitrary index
// lists keep their original multiplicity in FastllmCudaNaiveBatchAttention.
// Scratch is dead on the calling stream and is never resized or retained.
bool FastllmCudaTryNaiveIndexerAttention(const FastllmNaivePrefillAttentionItem &item,
    const fastllm::Data &sink, int heads, int kvHeads, int dim, int valueDim, const fastllm::Data &scratch);
void FastllmCudaNaiveBatchIndexerAttention(const std::vector<FastllmNaivePrefillAttentionItem> &items,
    const fastllm::Data &sink, int heads, int kvHeads, int dim, int valueDim, const fastllm::Data &scratch);
