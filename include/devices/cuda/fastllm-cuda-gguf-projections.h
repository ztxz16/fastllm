#pragma once

namespace fastllm {
class Data;
}

// Resident, bias-free FP16/BF16 projections of the same input (1..8 rows).
// Any number of projections >= 2 may share the call-local Q8 workspace.
// Unsupported types/layouts and MMQ batches return false before any writes.
bool FastllmCudaGGUFLinearShared(const fastllm::Data &input, fastllm::Data *const *weights,
                                 fastllm::Data *const *outputs, int count);
bool FastllmCudaGGUFMixedGateUp(const fastllm::Data &input, fastllm::Data &gate, fastllm::Data &up,
                                fastllm::Data &output);

// Read [keyHeads, valueHeads/keyHeads, headDim] directly while producing Q8
// in [valueHeads/keyHeads, keyHeads, headDim] order for the output projection.
bool FastllmCudaGGUFLinearAddPermuted(const fastllm::Data &input, fastllm::Data &weight,
                                      const fastllm::Data &bias, fastllm::Data &output, int keyHeads,
                                      int valueHeads, int headDim);

// Internal bridge: both GGUF MMVQ families use the same block_q8_1 layout.
bool FastllmCudaGGUFExtendedFromQ8(const void *input, const void *weight, void *output, int type, int rows,
                                   int columns, int outputRows, int storeMode, void *stream);
