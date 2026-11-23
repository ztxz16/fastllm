#pragma once

#include <cstddef>
#include <atomic>
#include <cstdint>
#include <vector>

// Separate causal layouts without a fixed batch-size offset or key collision.
inline int64_t FastllmV41VerificationGraphKey(int rows, bool segmented) {
    return -(int64_t(rows) * 2 + int(segmented));
}

// PP stages retain independent workspaces for decode and both causal layouts.
// Reserve all positive int row counts instead of a fixed stride per layer.
inline int64_t FastllmV41PipelineGraphKey(int firstLayer, int rows, bool verifying, bool segmented) {
    return (int64_t(1) << 62) | (int64_t(firstLayer) << 33) |
        (int64_t(rows) << 2) | (verifying ? 1 + int(segmented) : 0);
}

// Fixed tensor addresses and launch shapes; request-dependent values are
// uploaded separately before replay. The caller owns all tensor buffers.
struct FastllmV41FullDecodeConfig {
    int batch = 1, heads = 8, indexHeads = 32;
    int ratio = 0, window = 128, maxKeys = 131072, topK = 512;
    int blockSize = 64, candidateTopK = 32;
    // Storage modes match QuantizeKV: 0 BF16, 1 FP8, 2/3 FP4.
    int ringMode = 0, cmpMode = 0, indexMode = 0;
    int ringRowBytes = 1024, cmpRowBytes = 1024, indexRowBytes = 256;
    int ropeDim = 64, originalSeqLen = 4096, betaFast = 32, betaSlow = 1;
    float ropeBase = 10000.f, ropeFactor = 1.f, normEps = 1e-6f, softmaxScale = 1.f;
    // Verification rows belong to explicit causal chunks. Ring writes remain
    // deferred until each request accepts its prefix.
    bool verifying = false;
    bool segmentedVerify = false; // disables optimizations that share one request's keys
    bool vectorUnpack = false; // resolved from the existing unpack policy at plan creation
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
    const float *indexNorm = nullptr;
};

struct FastllmV41FullDecodeRequest {
    void *windowKV = nullptr, *compressedKV = nullptr, *indexK = nullptr;
    void *rawTailKV = nullptr, *rawTailScore = nullptr;
    int position = 0;
    // Row range of this request in a verification batch. A zero length retains
    // the original single-chunk API. Saved raw streams start at 2 * chunkBegin.
    int chunkBegin = 0, chunkLength = 0;
};

// Immutable per-layer pointers to a full verifier's saved rows and dynamic
// request metadata. Descriptors remain valid for the lifetime of that graph.
struct FastllmV41VerifyCommitLayer {
    const FastllmV41FullDecodeRequest *requests = nullptr;
    const void *savedKV = nullptr, *savedRawKV = nullptr, *savedRawScore = nullptr;
    int window = 128, ratio = 0, ringRowBytes = 1024;
};
bool FastllmCudaV41CommitVerifiedPrefix(const FastllmV41VerifyCommitLayer *layers,
                                      int layerCount, int accepted, int forwarded, int rowOffset = 0, int batchRows = 0);
// Device spans are [request, {accepted, rowOffset}]. The caller validates
// disjoint request ranges and keeps both descriptors alive until completion.
bool FastllmCudaV41CommitVerifiedBatch(const FastllmV41VerifyCommitLayer *layers,
                                     int layerCount, const int *spans, int requests, int batchRows);

// Create/destroy and upload run outside capture. Prepare/Finish contain only
// fixed-shape GPU work, and can be captured exactly once during startup.
// Between Prepare and Finish the caller projects/normalizes newIndexKeys.
void *FastllmCudaV41CreateFullDecode(const FastllmV41FullDecodeConfig &config,
                                     const FastllmV41FullDecodeBuffers &buffers);
void FastllmCudaV41DestroyFullDecode(void *plan);
// Before capture, share attention scratch between strictly sequential layers.
// Different devices/layouts and unavailable backends keep independent storage.
bool FastllmCudaV41ShareAttentionWorkspace(void *plan, void *source);
bool FastllmCudaV41UploadFullDecode(void *plan, const FastllmV41FullDecodeRequest *requests,
                                    size_t count);
// Plans use consecutive slices of storage. Null slots are allowed while the
// caller builds the graph; initialized plans retain the ordinary validation.
bool FastllmCudaV41UploadFullDecodeBatch(void *storage, void *const *plans, size_t layers,
    const FastllmV41FullDecodeRequest *requests, size_t batch);
bool FastllmCudaV41PrepareFullDecode(void *plan);
bool FastllmCudaV41CanPublishSource(const void *plan);
bool FastllmCudaV41PrepareAllIndices(void *plan, bool publishSource);
bool FastllmCudaV41FinishSource(void *plan);
void *FastllmCudaV41ForkSourceStore(const void *graph);
void *FastllmCudaV41MoveSourceToPrelude(void *prefix, const void *compressorReady, const void *core);
bool FastllmCudaV41FinishFullDecode(void *plan);
bool FastllmCudaV41FinishAllIndices(void *plan);
std::vector<void *> FastllmCudaV41CaptureNodes();
void *FastllmCudaV41WithoutIndexProjections(const void *graph, const std::vector<void *> &nodes);
void *FastllmCudaV41CaptureFrontier();
bool FastllmCudaV41FuseExpertBoundary(const void *previous, const void *attention,
        const void *shortAttention, const void *postKernel, int rows, bool packOnly, void **result);
void *FastllmCudaV41GraphPrefix(const void *graph, const void *frontier, const void *compressorInput = nullptr,
                                const void *beforeCompressor = nullptr, bool trim = true);

// Source bytes are staged in pinned storage until all asynchronous copies finish.
struct FastllmV41HostCopy { void *dst; const void *src; size_t bytes; };
// A distributed coordinator owns only Engram and token metadata; request
// descriptors belong to the attention ranks.
bool FastllmCudaV41UploadAuxiliaryInputs(const FastllmV41HostCopy *copies, size_t count);
bool FastllmCudaV41UploadFullDecodeInputs(void *storage, void *const *plans, size_t layers,
    const FastllmV41FullDecodeRequest *requests, size_t batch,
    const FastllmV41HostCopy *extra, size_t extraCount);

// A graph-owned wait/copy pair for CPU-produced rows. The caller owns the
// mapped ready allocation (free with FastllmCudaHostFree) and the graph.
// Before replay: reset ready, fill hostData, release-store ready = 1.
// Do not reuse either buffer until the previous graph has consumed it.
bool FastllmCudaV41CreateHostCopy(void *dst, size_t bytes, void **hostData,
    std::atomic<unsigned> **ready, void **graph);

bool FastllmCudaV41MlaOutput(void *plan, void *&codes, void *&scales, int &pitch, bool &headMajor);
void FastllmCudaV41UseMlaOutput(void *plan);

// Validated, fixed independent decode rows on the same rank stream.
bool FastllmCudaV41AdvancePositions(void *storage, size_t count);

bool FastllmCudaV41PublishDraftFeature(const void *hidden, void *packed, const int *globalRows,
                                      int rows, int dim, int slot, int features);

bool FastllmCudaV41OwnerEmbedding(const void *weight, bool fp32, const int *ids, void *hidden,
                                  float *firstMix, int rows, int dim, int vocab);
bool FastllmCudaV41OwnerEmbedding(const void *weight, bool fp32, const int *ids, void *hidden,
                                  float *firstMix, int rows, int dim, int vocab,
                                  const int *positions, FastllmV41FullDecodeRequest *requests, int layers);
