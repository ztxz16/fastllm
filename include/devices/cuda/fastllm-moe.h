#pragma once
#include <cstddef>

// Resident packed 16-bit [E,2I,H] / [E,H,I] experts. Scratch uses the same type,
// [M,K,I] and [M,K,H]. Each down contribution rounds AFTER route weighting.
// Optional workspace is sized by WorkspaceBytes below (256-byte alignment).
// It enables grouped Tensor Cores or a cuBLAS fallback with GPU pointer tables.
// Routes must contain unique expert ids in [0,E) for each input row.
std::size_t FastllmCudaMoe16WorkspaceBytes(int rows, int hidden, int intermediate,
    int experts, int topk, bool fp16 = false);
// Optional 128-byte-aligned packed scratch enables native SM90 TMA for
// aligned blocks. The caller may omit it for small eager blocks; zero means
// the shape/compiled architecture requires the generic path.
std::size_t FastllmCudaMoe16PackedBytes(int rows, int hidden, int intermediate, int experts, int topk, bool fp16 = false);
// Fused shared expert epilogue. shared is ungated [rows,hidden], sharedGate
// is [rows], optionally strided; both use the selected 16-bit type and must
// be null together. Retain sigmoid, product, routed-sum and final-add rounding.
bool FastllmCudaMoe16(const void *input, const void *gateUp, const void *down, const int *indices,
    const float *scores, void *middle, void *parts, void *output, int rows, int hidden,
    int intermediate, int experts, int topk, void *stream = nullptr, int *workspace = nullptr,
    void *packed = nullptr, const void *shared = nullptr, const void *sharedGate = nullptr, int sharedGateStride = 1, bool fp16 = false);
