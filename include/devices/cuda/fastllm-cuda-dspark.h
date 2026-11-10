#pragma once
#include "fastllm.h"
// A shard's stable softmax statistics and Gumbel winner, gathered as 16 bytes.
// best is relative to maximum, with temperature already applied.
struct FastllmDsparkSamplePartial { float maximum, sum, best; int token; };
static_assert(sizeof(FastllmDsparkSamplePartial) == 16, "Markov sampling gather packet must be 16 bytes");
extern "C" {
// Dense BF16 rank-256 shards. Caller validates shapes, zeros keys before the
// chain, and performs an unsigned MAX all-reduce between Score and Finish.
bool FastllmCudaDsparkMarkovShardScore(const float *logits, const void *embed,
    const void *head, const int *tokens, unsigned long long *keys,
    int step, int vocab, int start, int count);
bool FastllmCudaDsparkMarkovShardFinish(const unsigned long long *keys,
    const void *embed, int *tokens, float *outEmbeds, int step);
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

// Full-window graph inputs. metadata is INT32 [start position, anchor token];
// the caller supplies valid token IDs and start position >= windowSize.
bool FastllmCudaDeepSeekV41DraftInput(const fastllm::Data &embedding,
    const fastllm::Data &metadata, int noiseToken, int block, int hcMult, int windowSize,
    fastllm::Data &hidden, fastllm::Data &preMix, fastllm::Data &indices, fastllm::Data &tokens);
bool FastllmCudaDeepSeekV41WindowStorePosition(const fastllm::Data &chunk, fastllm::Data &ring,
                                              int startPos, int windowSize, const int *devicePosition);
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
