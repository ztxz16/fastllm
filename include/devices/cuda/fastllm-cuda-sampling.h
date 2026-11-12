#ifndef FASTLLM_CUDA_SAMPLING_H
#define FASTLLM_CUDA_SAMPLING_H
#include <cstddef>

enum class FastllmCudaSamplingPolicy {
    TopKWithTies,
    LLMSampling // Exact-k and reciprocal-temperature scaling; probability sums stay FP64.
};

struct FastllmCudaSamplingRow {
    const float *logits = nullptr;
    double uniform = 0.; // Request-owned RNG draw in [0, 1).
    float temperature = 1.f, topP = 1.f;
    int topK = 0;
    // Optional sorted device-side token lists. An empty allow-list is unmasked.
    const int *allowedTokens = nullptr;
    int allowedCount = 0;
    FastllmCudaSamplingPolicy policy = FastllmCudaSamplingPolicy::TopKWithTies;
    const int *repeatTokens = nullptr; // Sorted; duplicates apply repeated penalties.
    int repeatCount = 0;
    float repeatPenalty = 1.f;
};
struct FastllmCudaSamplingResult { int token, error; };

// Exact small-top-k selection for ordinary, unmasked sampling. Each row
// returns 64 candidates, descending score then ascending token ID. Padding
// has token=-1. A negative first token reports invalid/all-infinite logits.
// Scaling uses the caller's FP32 reciprocal to preserve CPU sampling scores.
// Only the physical work tile is bounded; logical batches are unrestricted.
constexpr int FastllmCudaLogitsTopKCapacity = 64;
struct FastllmCudaTopKRow { const float *logits; float inverseTemperature; };
struct FastllmCudaTopKCandidate { float score; int token; };
std::size_t FastllmCudaLogitsTopKWorkspaceBytes(int batch, int vocab);
bool FastllmCudaLogitsTopK(const FastllmCudaTopKRow *rows, int batch, int vocab,
    FastllmCudaTopKCandidate *output, void *workspace, std::size_t workspaceBytes,
    void *stream = nullptr);

// Stable descending score order, ascending token-id ties. Apply top-k first
// (optionally retaining cutoff ties), then top-p, retaining its boundary token. Logits
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
