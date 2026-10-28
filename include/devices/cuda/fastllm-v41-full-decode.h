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
    // A verification plan treats batch rows as one causal chunk. Ring writes
    // remain deferred until acceptance; the saved raw stream supports rollback.
    bool verifying = false;
    bool kvSource = false, indexSource = false;
    bool writeCandidates = false, useCandidates = false;
};

struct FastllmV41FullDecodeBuffers {
    void *q = nullptr, *kv = nullptr, *output = nullptr;
    void *rawKV = nullptr, *rawScore = nullptr;
    void *joined = nullptr, *scores = nullptr, *latents = nullptr, *newIndexKeys = nullptr;
    void *indexQ = nullptr, *indexWeights = nullptr, *topk = nullptr, *candidates = nullptr;
    const void *compressNorm = nullptr, *sink = nullptr;
    // Optional caller-owned request slice. It must outlive the plan and graphs.
    void *requests = nullptr;
    void *savedKV = nullptr, *savedRawKV = nullptr, *savedRawScore = nullptr;
    // Optional caller-owned [batch, ceil(maxKeys / blockSize)] block IDs and
    // [batch] counts. A consumer may reuse a preceding producer's list only
    // for the same mask, per-row visible range and stream-ordered replay.
    // The producer must run again after every mask/position update.
    void *candidateBlocks = nullptr, *candidateCounts = nullptr;
    bool reuseCandidates = false;
};

struct FastllmV41FullDecodeRequest {
    void *windowKV = nullptr, *compressedKV = nullptr, *indexK = nullptr;
    void *rawTailKV = nullptr, *rawTailScore = nullptr;
    int position = 0;
};

// Immutable per-layer pointers to a full verifier's saved rows and dynamic
// request metadata. Descriptors remain valid for the lifetime of that graph.
struct FastllmV41VerifyCommitLayer {
    const FastllmV41FullDecodeRequest *requests = nullptr;
    const void *savedKV = nullptr, *savedRawKV = nullptr, *savedRawScore = nullptr;
    int window = 128, ratio = 0;
};
bool FastllmCudaV41CommitVerifiedPrefix(const FastllmV41VerifyCommitLayer *layers,
                                      int layerCount, int accepted, int forwarded);

// Create/destroy and upload run outside capture. Prepare/Finish contain only
// fixed-shape GPU work, and can be captured exactly once during startup.
// Between Prepare and Finish the caller projects/normalizes newIndexKeys.
void *FastllmCudaV41CreateFullDecode(const FastllmV41FullDecodeConfig &config,
                                     const FastllmV41FullDecodeBuffers &buffers);
void FastllmCudaV41DestroyFullDecode(void *plan);
bool FastllmCudaV41UploadFullDecode(void *plan, const FastllmV41FullDecodeRequest *requests,
                                    size_t count);
// Plans use consecutive slices of storage. Null slots are allowed while the
// caller builds the graph; initialized plans retain the ordinary validation.
bool FastllmCudaV41UploadFullDecodeBatch(void *storage, void *const *plans, size_t layers,
    const FastllmV41FullDecodeRequest *requests, size_t batch);
bool FastllmCudaV41PrepareFullDecode(void *plan);
bool FastllmCudaV41FinishFullDecode(void *plan);

// Source bytes are staged in pinned storage until all asynchronous copies finish.
struct FastllmV41HostCopy { void *dst; const void *src; size_t bytes; };
bool FastllmCudaV41UploadFullDecodeInputs(void *storage, void *const *plans, size_t layers,
    const FastllmV41FullDecodeRequest *requests, size_t batch,
    const FastllmV41HostCopy *extra, size_t extraCount);
