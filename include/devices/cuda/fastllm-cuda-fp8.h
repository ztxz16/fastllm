#pragma once

namespace fastllm {
class Data;
}

// Prepare an already materialized FP8 Linear weight for repeated small-row
// SM70 GEMM calls. This uses the existing in-place layout/FP16-scale conversion.
// The caller must select the weight's CUDA device and ensure serving is idle.
// Returns false for ineligible weights/row counts or a deferred conversion,
// without changing the weight storage.
bool FastllmCudaWarmupFp8E4M3Sm70(fastllm::Data &weight, int rows);
