#pragma once
#include "devices/cuda/naive-n05-kv.cuh"

// Hopper prefill scoring for sixteen E4M3-rounded Q heads and unchanged K.
// No global scratch beyond the caller's existing score matrix.
namespace naive_indexer {
// False leaves outputs untouched; the caller retains its generic CUDA path.
bool Prepare();
void Launch(const __nv_bfloat16 *q, FastllmNaiveKV keys, const float *scales,
            const __nv_bfloat16 *weights, float *scores,
            int queries, int keyCount, int past, int stride);
}
