#pragma once
#include "fastllm.h"
extern "C" {
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
}
