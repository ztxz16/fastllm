#pragma once

#include <cstddef>

// Fixed tensor addresses and launch shapes; request-dependent values are
// uploaded separately before replay. The caller owns all tensor buffers.
struct FastllmV41FullDecodeConfig {
    int batch = 1, heads = 8, indexHeads = 32;
    int ratio = 0, window = 128, maxKeys = 131072, topK = 512;
    int blockSize = 64, candidateTopK = 32;
    int ropeDim = 64, originalSeqLen = 4096, betaFast = 32, betaSlow = 1;
    float ropeBase = 10000.f, ropeFactor = 1.f, normEps = 1e-6f, softmaxScale = 1.f;
    bool kvSource = false, indexSource = false;
    bool writeCandidates = false, useCandidates = false;
};

struct FastllmV41FullDecodeBuffers {
    void *q = nullptr, *kv = nullptr, *output = nullptr;
    void *rawKV = nullptr, *rawScore = nullptr;
    void *joined = nullptr, *scores = nullptr, *latents = nullptr, *newIndexKeys = nullptr;
    void *indexQ = nullptr, *indexWeights = nullptr, *topk = nullptr, *candidates = nullptr;
    const void *compressNorm = nullptr, *sink = nullptr;
};

struct FastllmV41FullDecodeRequest {
    void *windowKV = nullptr, *compressedKV = nullptr, *indexK = nullptr;
    void *rawTailKV = nullptr, *rawTailScore = nullptr;
    int position = 0;
};

// Create/destroy and upload run outside capture. Prepare/Finish contain only
// fixed-shape GPU work, and can be captured exactly once during startup.
// Between Prepare and Finish the caller projects/normalizes newIndexKeys.
void *FastllmCudaV41CreateFullDecode(const FastllmV41FullDecodeConfig &config,
                                     const FastllmV41FullDecodeBuffers &buffers);
void FastllmCudaV41DestroyFullDecode(void *plan);
bool FastllmCudaV41UploadFullDecode(void *plan, const FastllmV41FullDecodeRequest *requests,
                                    size_t count);
bool FastllmCudaV41PrepareFullDecode(void *plan);
bool FastllmCudaV41FinishFullDecode(void *plan);
