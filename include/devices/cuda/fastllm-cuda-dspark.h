#pragma once
#include "fastllm.h"
// A shard's stable softmax statistics and Gumbel winner, gathered as 16 bytes.
// best is relative to maximum, with temperature already applied.
struct FastllmDsparkSamplePartial { float maximum, sum, best; int token; };
static_assert(sizeof(FastllmDsparkSamplePartial) == 16, "Markov sampling gather packet must be 16 bytes");
extern "C" {
// requestWindows is request-major with three ring pointers per request.
void FastllmCudaDsparkTransferWindows(void *const *requestWindows, void *const *batchWindows,
    int batch, size_t bytesPerRing, bool scatter);
bool FastllmCudaDsparkInverseRotaryBatch(fastllm::Data &x, int ropeDim, float ropeBase,
    int originalSeqLen, float ropeFactor, int betaFast, int betaSlow, const int *positions);
bool FastllmCudaDsparkDraftSparseAttention(const fastllm::Data &q, const fastllm::Data &kv,
    const fastllm::Data &table, const fastllm::Data &indices, fastllm::Data &sink,
    float scale, fastllm::Data &scratch, fastllm::Data &output);
// Dense BF16 rank-256 shards. Caller validates shapes, zeros keys before the
// chain, and performs an unsigned MAX all-reduce between Score and Finish.
bool FastllmCudaDsparkMarkovShardScore(const float *logits, const void *embed,
    const void *head, const int *tokens, unsigned long long *keys,
    int step, int vocab, int start, int count);
bool FastllmCudaDsparkMarkovShardFinish(const unsigned long long *keys,
    const void *embed, int *tokens, float *outEmbeds, int step);
bool FastllmCudaDsparkMarkovShardScoreBatch(const float *logits, const void *embed,
    const void *head, const int *tokens, unsigned long long *keys,
    int step, int vocab, int start, int count, int batch, int steps);
bool FastllmCudaDsparkMarkovShardFinishBatch(const unsigned long long *keys,
    const void *embed, int *tokens, float *outEmbeds, int step, int batch, int steps);
// BF16/rank-256 sampling shards. partials has ceil(count/64) entries; packet
// contains one entry per rank. Gather packet after Sample, then call Normalize.
bool FastllmCudaDsparkMarkovShardSample(const float *logits, const void *embed,
    const void *head, const int *tokens, float *scores, float *outEmbeds,
    FastllmDsparkSamplePartial *partials, FastllmDsparkSamplePartial *packet,
    unsigned long long seed, unsigned long long *counter,
    int step, int start, int count, int stride, float temperature);
bool FastllmCudaDsparkMarkovShardNormalize(float *scores,
    const FastllmDsparkSamplePartial *gathered, int *tokens,
    int ranks, int step, int count, int stride, float temperature);

// Full-window graph inputs. metadata is INT32 [positions for B, anchors for B];
// the caller supplies valid token IDs and start position >= windowSize.
bool FastllmCudaDeepSeekV41DraftInput(const fastllm::Data &embedding,
    const fastllm::Data &metadata, int noiseToken, int block, int hcMult, int windowSize,
    fastllm::Data &hidden, fastllm::Data &preMix, fastllm::Data &indices, fastllm::Data &tokens);
bool FastllmCudaDeepSeekV41WindowStorePosition(const fastllm::Data &chunk, fastllm::Data &ring,
                                              int startPos, int windowSize, const int *devicePosition);
// Fixed-capacity [batch, rows, dim] input; only each device-side accepted
// prefix is stored. Positions and counts have batch entries and may change
// between graph replays without changing the input/window addresses.
void FastllmCudaDsparkPackAttention(const fastllm::Data &query, const fastllm::Data &kv,
    const fastllm::Data &window, const fastllm::Data &indices, const fastllm::Data &position,
    fastllm::Data &shadow, fastllm::Data &valid, const size_t *offsets, fastllm::Data &packet);
// For a peer-visible window, changed identifies rows to refresh in this rank's persistent table.
bool FastllmCudaDsparkDraftRotaryTable(fastllm::Data &q, fastllm::Data &kv, const fastllm::Data &window,
    int ropeDim, float ropeBase, int startPos, int originalSeqLen, float ropeFactor, int betaFast,
    int betaSlow, const int *devicePosition, fastllm::Data &table, const int *changed);
// Three [512,5120] W8A16 projections with independent packed weights and scales.
bool FastllmCudaDsparkContextProjection(const void *input, const void *weights, void *output, int rows);
bool FastllmCudaDsparkContextStore(const fastllm::Data *const projected[3], const fastllm::Data *const norms[3],
    fastllm::Data *const rings[3], bool stacked, int batch, int rows, const int *positions, float eps,
    int ropeDim, float ropeBase, float ropeFactor, int betaFast, int betaSlow);
bool FastllmCudaDeepSeekV41WindowStoreBatchPrefix(const fastllm::Data &chunk, fastllm::Data &ring,
    int windowSize, const int *devicePositions, const int *deviceCounts);
// tokens[0] is the device-side anchor; tokens[1..block] are produced in order.
bool FastllmCudaDeepSeekV41MarkovChainDevice(const fastllm::Data &logits,
    const fastllm::Data &embedWeight, const fastllm::Data &headWeight, int block,
    fastllm::Data &tokens, fastllm::Data &partialValue, fastllm::Data &partialIndex,
    fastllm::Data &outEmbeds);
// Full-support temperature sampling. Retains the actual normalized proposal q
// for chain rejection. The caller supplies a valid anchor in tokens[0]; all
// outputs stay on device. Scratch can be reused between sequential steps.
// Like the Qwen Gumbel primitive, warm the fixed vocabulary before capture.
bool FastllmCudaDeepSeekV41MarkovSampleDevice(const fastllm::Data &logits,
    const fastllm::Data &embedWeight, const fastllm::Data &headWeight,
    int block, float temperature, fastllm::Data &tokens, fastllm::Data &probs,
    fastllm::Data &scratch, fastllm::Data &outEmbeds);
// Filtered proposal q with graph-stable scratch and a replay-advancing RNG.
bool FastllmCudaDeepSeekV41MarkovFilteredSampleDevice(const fastllm::Data &logits,
    const fastllm::Data &embedWeight, const fastllm::Data &headWeight,
    int block, float temperature, int topK, float topP,
    fastllm::Data &tokens, fastllm::Data &probs, fastllm::Data &outEmbeds,
    fastllm::Data &samplingWorkspace);
// Batched chains share launches at each dependent token step. Input/output
// layout is request-major; token anchors have stride block+1, q has block rows.
bool FastllmCudaDeepSeekV41MarkovFilteredSampleBatchDevice(const fastllm::Data &logits,
    const fastllm::Data &embedWeight, const fastllm::Data &headWeight,
    int block, float temperature, int topK, float topP,
    fastllm::Data &tokens, fastllm::Data &probs, fastllm::Data &outEmbeds,
    fastllm::Data &scores, fastllm::Data &samplingWorkspace);
}
