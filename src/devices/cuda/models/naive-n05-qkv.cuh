#pragma once

#include <cuda_bf16.h>

// TP8 decode: 8 Q heads, one K/V head, head widths 192/128 and rotary width 64.
// All mutable cache bindings and lengths are read from device memory on replay.
namespace naive_qkv {
struct Params {
    __nv_bfloat16 *q;
    const __nv_bfloat16 *indexKey;
    const float *positions;
    const int *lengths;
    const void *const *cache;
    float theta;
    __nv_bfloat16 valueScale;
    int pageLen, keyRows, valueRows;
    bool ring;
    bool fp8 = false, scaledIndexer = false;
};

// Initializes per-device launch attributes before capture; false leaves outputs untouched.
bool Prepare();
void Launch(const __nv_bfloat16 *input, const __nv_bfloat16 *weight,
            Params params, int rows, float *scratch);
}
