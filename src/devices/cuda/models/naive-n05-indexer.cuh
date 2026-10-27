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

// Temporary opt-in FP8 experiment. The accepted source tree is untouched.
namespace naive_indexer {
struct ExperimentalEntry {
    FastllmNaiveKV key;
    const __nv_bfloat16 *q, *weights;
    const float *scales;
    float *scores;
    int rows, keys, past, stride, part = 0, gap = 0;
};
// Bitmask of entries dispatched; zero preserves the caller's original path.
unsigned LaunchExperimentalFP8(const ExperimentalEntry *entries, int count);
unsigned long long ExperimentalFP8Launches();
}
