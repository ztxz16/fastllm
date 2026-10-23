#pragma once
#include "fastllm.h"
extern "C" {
bool FastllmCudaDeepSeekV41WindowStorePosition(const fastllm::Data &chunk, fastllm::Data &ring,
                                              int startPos, int windowSize, const int *devicePosition);
// tokens[0] is the device-side anchor; tokens[1..block] are produced in order.
bool FastllmCudaDeepSeekV41MarkovChainDevice(const fastllm::Data &logits,
    const fastllm::Data &embedWeight, const fastllm::Data &headWeight, int block,
    fastllm::Data &tokens, fastllm::Data &partialValue, fastllm::Data &partialIndex,
    fastllm::Data &outEmbeds);
}
