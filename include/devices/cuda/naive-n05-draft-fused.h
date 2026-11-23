#pragma once
#include <cstdint>

// Raw CUDA ABI also used by the operator-level validation harness. Weights
// are already TP-sharded; callers retain ownership of all source tensors.
struct FastllmDraftFusedWeight {
    const void *qkv, *o, *inputNorm, *qNorm, *kNorm;
    const void *gateUp, *down, *postNorm;
};
struct FastllmDraftContextRequest {
    const void *features;
    const void *const *cacheTable;
    int rows, position, oldLength, tableRequests, tableRequest;
};
struct FastllmDraftHeadRequest {
    int anchor, position;
    uint64_t seed;
    float temperature;
};
extern "C" {
bool FastllmDraftFusedSupported(const int *devices, int ranks);
void *FastllmDraftFusedCreate(const int *devices, int ranks, int layers,
    const FastllmDraftFusedWeight *weights, int rows, int window, float eps, float theta);
bool FastllmDraftFusedPrepare(void *state, int requests);
void FastllmDraftFusedDestroy(void *state);
const char *FastllmDraftFusedError();
// Each request owns a mailbox, epoch, barriers and intermediate tensors.
// Calls on participating ranks must use a common request/layer order. Prepared
// counts 1..64 use bounded request tiles; scalar tails and unsupported batch
// kernel geometries retain the original per-request path.
bool FastllmDraftFusedRun(void *state, int rank, int requests, void *hidden,
    const int *live, const void *const *cacheTable, void *stream);
// Retains the raw body output and writes an independent normalized output.
// Hidden, normalized and FP32 gamma must be 16-byte aligned and non-aliasing.
// Unsupported layouts keep Run followed by the ordinary normalization path.
// Builds the first padded draft block from device token IDs and shared embeddings.
bool FastllmDraftFusedRunEmbedded(void *state, int rank, int requests, void *hidden,
    const float *ids, const void *embedding, const void *mask, const int *live,
    const void *const *cacheTable, const float *gamma, void *normalized, void *stream);
bool FastllmDraftFusedRunNormalized(void *state,int rank,int requests,void *hidden,
    const int *live,const void *const *cacheTable,const float *gamma,void *normalized,void *stream);
bool FastllmDraftFusedMlpBatchProbe(void *state,int rank,int requests,int layer,
    const void *input,void *output,void *stream);
bool FastllmDraftFusedMlpProbe(void *state, int rank, int request, int layer,
    const void *input, void *output, void *stream);
bool FastllmDraftFusedAttentionBatchProbe(void *state,int rank,int requests,int layer,
    const void *input,void *output,const int *live,const void *const *table,void *stream);
bool FastllmDraftFusedAttentionProbe(void *state, int rank, int request, int layer,
    const void *input, void *output, const int *live,
    const void *const *cacheTable, int requests, void *stream);
unsigned FastllmDraftFusedWatchdog(void *state);
// In-place FP32 sum in the serialized draft domain; borrows context scratch.
// All ranks must submit the same shape and retain data through stream completion.
bool FastllmDraftFusedReduceSupported(void *state,int requests);
bool FastllmDraftFusedReduce(void *state,int rank,int requests,void *data,int elements,void *stream);
bool FastllmDraftFusedContextPrepare(void *state, const void *fc, const void *norm, int owner);
bool FastllmDraftFusedContextRun(void *state, int rank, int request, const void *features,
    int rows, int position, int oldLength, const void *const *cacheTable,
    int tableRequests, int tableRequest, void *stream);
// Prepared at startup; unavailable plans retain ContextRun. A failed submission
// must be drained and propagated, never replayed through another path.
// One rank-major descriptor upload serves all GPUs. A domain is serialized:
// all rank consumers must finish before the next upload or destruction.
bool FastllmDraftFusedContextSharedSupported(void *state,int count);
bool FastllmDraftFusedContextSharedUpload(void *state,const FastllmDraftContextRequest *requests,int count);
// Accepted counts are device-resident on the descriptor owner and ordered on
// producerStream. Each context kernel consumes its request's count directly;
// callers retain the counts until all rank consumers complete.
bool FastllmDraftFusedContextSharedUploadAccepted(void *state,const FastllmDraftContextRequest *requests,
    int count,const int *accepted,void *producerStream);
struct FastllmDraftNextInputs {
    const int *tokens;
    int stride;
    float *anchors[8];
    int *lengths[8];
};
// The context graph binds the next body inputs before its completion event.
bool FastllmDraftFusedContextUploadNext(void *state,const FastllmDraftContextRequest *requests,
    int count,const int *accepted,const FastllmDraftNextInputs &next,void *producerStream);
bool FastllmDraftFusedContextSharedRun(void *state,int rank,int count,void *stream);
bool FastllmDraftFusedHeadPrepare(void *state,const void *const *heads,const void *w1,const void *w2,
    const float *confidenceWeight,const float *confidenceBias,int owner);
// Called after KV allocation. All ranks must fit the head while retaining
// the configured fraction of total device memory for runtime use.
bool FastllmDraftFusedHeadPrepareWithBudget(void *state,const void *const *heads,const void *w1,const void *w2,
    const float *confidenceWeight,const float *confidenceBias,int owner,double gpuMemRatio);
bool FastllmDraftFusedHeadRun(void *state,int rank,int request,const void *hidden,
    int anchor,uint64_t seed,float temperature,int steps,void *stream);
const int64_t *FastllmDraftFusedHeadTokens(void *state, int rank, int request);
bool FastllmDraftFusedVerificationInput(const int64_t *tokens, void *packedInput,
    int stride, int offset, int position, int rows, void *stream);
// Startup-only composition. The caller owns the target graph, hidden/input
// buffers and completion event; all remain live until this plan is retired.
void *FastllmDraftFusedVerificationCreate(void *state,int rank,int count,const void *hidden,
    void *packedInput,int stride,void *headReady,void *targetGraph,int verifyRows);
void FastllmDraftFusedVerificationDestroy(void *plan);
// The same executable serves ordinary verification with its head prefix disabled.
void *FastllmDraftFusedVerificationExec(void *plan,bool head);
bool FastllmDraftFusedVerificationUpload(void *plan,const FastllmDraftHeadRequest *requests,
    int count,void *stream);
// Join all rank completion events on the owner stream before Submit or Finish.
// The complete temperature-softmax q is retained for each requested step.
// Steps are a prefix of the trained seven-row head; a null confidence output
// skips confidence computation/export when truncation is disabled.
bool FastllmDraftFusedHeadFinish(void *state,int request,const void *hidden,float temperature,
    float *probabilities,int64_t *tokens,float *confidence,int steps,void *stream);
// Async variant: outputs must be pinned host memory retained until the owner
// stream completes. Callers may submit all requests before one stream join.
bool FastllmDraftFusedHeadSubmit(void *state,int request,const void *hidden,float temperature,
    float *probabilities,int64_t *tokens,float *confidence,int steps,void *stream);
// Dense batch outputs reserve seven rows per request. Probability rows beyond
// each prefix are untouched; unused host token/confidence slots are unspecified.
bool FastllmDraftFusedHeadSubmitBatch(void *state,int count,const void *hidden,
    const float *temperatures,const int *steps,float *probabilities,int64_t *tokens,float *confidence,void *stream);
bool FastllmDraftFusedHeadProbe(void *state,int rank,int request,void *base,void *corrected,
    void *ids,void *biasTableRows,const int64_t *previous,int rows,void *stream);
}
