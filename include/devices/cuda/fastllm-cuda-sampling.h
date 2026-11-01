#ifndef FASTLLM_CUDA_SAMPLING_H
#define FASTLLM_CUDA_SAMPLING_H
#include <cstddef>

struct FastllmCudaSamplingRow {
    const float *logits = nullptr;
    double uniform = 0.; // Request-owned RNG draw in [0, 1).
    float temperature = 1.f, topP = 1.f;
    int topK = 0;
};
struct FastllmCudaSamplingResult { int token, error; };

// Stable descending score order, ascending token-id ties. Apply top-k first
// (retaining cutoff ties), then top-p, retaining its boundary token. Logits
// remain unchanged. Invalid logits produce error=1, token=-1 for that row.
// Rows live on the host; their logits, output and workspace live on the GPU.
// Output holds batch results. Output, logits and scratch must not overlap.
// Caller-owned, 256-byte-aligned scratch permits stream/graph reuse; a null
// stream selects cudaStreamPerThread. The size query returns zero on failure.
std::size_t FastllmCudaSampleLogitsWorkspaceBytes(int batch, int vocab);
bool FastllmCudaSampleLogits(const FastllmCudaSamplingRow *rows, int batch,
    int vocab, FastllmCudaSamplingResult *output, void *workspace,
    std::size_t workspaceBytes, void *stream = nullptr);
#endif
