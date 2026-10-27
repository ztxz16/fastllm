#pragma once
#include "gguf.h"

// Canonical IQ2 rows with independently addressed Q8_K activations. Reuse the
// decoded weight fragment while retaining each single-row accumulation order.
void ggml_vec_dot_iq2_q8_K_batch(ggml_type type, int n, const void *weight,
    const void *const *inputs, float *outputs, int rows);
