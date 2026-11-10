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

// Native Hopper FP8 prefill. Call only for explicitly enabled FP8 compute.
struct FP8Entry {
    FastllmNaiveKV key;
    const __nv_bfloat16 *q, *weights;
    const float *scales;
    float *scores;
    int rows, keys, past, stride, part = 0, gap = 0;
};
// Bitmask of entries dispatched; zero preserves the caller's original path.
// Keep staged local K alive across coarse scoring and candidate refinement.
unsigned LaunchFP8(const FP8Entry *entries, int count, fastllm::Data &scratch, FP8Entry *staged);
// Reuse consumed coarse-score rows for exact BF16 K expansion when it fits.
// Remaining score rows must not overlap this region; no allocation is made.
bool StageRefinementKeys(const FP8Entry &entry, int consumedRows);
// Rescore the selected candidate IDs with the original BF16 arithmetic.
void RefineFP8(const FP8Entry &entry, unsigned long long *candidates, const int *ends, int count,
              const __nv_bfloat16 *expanded);
}
