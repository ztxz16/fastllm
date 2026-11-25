#pragma once
#include <functional>
#include <cstddef>
#include <cstdint>

namespace fastllm {
class Data;
}

// Project packed Q8_0 weights with float activations and caller-owned Q8_1
// scratch (rows * columns / 32 blocks). Scratch may be retained in a graph.
std::size_t FastllmCudaQ8ProjectionScratchBytes(int rows, int columns);
void FastllmCudaProjectQ8_0(const float *input, const void *weight, float *output, void *scratch,
                            int rows, int columns, int outputs);

// Resident, bias-free FP16/BF16 projections of the same input.
// Any number of projections >= 2 may share the call-local Q8 workspace.
// Compatible same-format Q5/Q6/Q8/IQ4_NL jobs also share one MMVQ launch
// (up to 8 jobs per launch); specialized IQ/planar kernels keep their dispatch.
// Larger batches share the MMQ quantizer by activation layout and reuse its
// bounded workspace. Unsupported types/layouts return false before any writes.
bool FastllmCudaGGUFLinearShared(const fastllm::Data &input, fastllm::Data *const *weights,
                                 fastllm::Data *const *outputs, int count);

// Call-local scheduling hooks. Incremental backends prepare shared activations
// before prepared() and retain them through all projected(index) callbacks.
// A backend may instead finish the whole group before notifying consumers.
// Each output is available to its callback. Callbacks must join their parallel
// consumers before the final notification returns, and must not change the
// input, weights or other outputs.
// Unsupported calls return false before callbacks or output writes.
struct FastllmCudaGGUFProjectionHooks {
    std::function<void()> prepared;
    std::function<void(int)> projected;
    // Optional independent work after validation/scratch reservation and before
    // quantization. May start a branch joined by the final projected callback.
    // Non-incremental backends notify after computing the entire group.
    std::function<void()> preparing;
};
bool FastllmCudaGGUFLinearShared(const fastllm::Data &input, fastllm::Data *const *weights,
                                 fastllm::Data *const *outputs, int count,
                                 const FastllmCudaGGUFProjectionHooks &hooks);
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

// Small-batch grouped Q5_K WoA, with the ordinary four-warp GEMV reduction order.
bool FastllmCudaGGUFGroupedGemv(const fastllm::Data &input, fastllm::Data &weight,
                              int groups, int rows, fastllm::Data &output);

// Internal MMQ bridge; caller validates tensor layout, aliases and device ownership.
bool FastllmCudaGGUFMMQShared(const fastllm::Data &input, fastllm::Data *const *weights,
                              fastllm::Data *const *outputs, int count);

// Project each GPU route against its expert's contiguous GGUF matrix. Consecutive
// routes may share an input row. Outer scales, when present, are per expert.
using FastllmGGUFExpertProjection = void (*)(const void *weights, const float *outer, const int *experts,
        const float *input, float *output, int routes, int routesPerInput, int cols, int rows, size_t rowBytes);
FastllmGGUFExpertProjection FastllmCudaGetGGUFExpertProjection(int type);

// One input shared by all top-k routes. Uses caller-owned Q8_1 scratch.
bool FastllmCudaGGUFSingleExpertProjection(int type, const uint8_t *weights, const float *outer, const int *ids,
        const float *input, float *output, void *scratch, int topk, int cols, int width, size_t rowBytes);

// Resident gate/up and optional down projection. The device pointer table has
// two slots per expert (repeat the gate pointer when down runs on another backend).
// The gate builds a route map that the down projection reuses in the same workspace.
// The row limit bounds transient storage; longer requests can use consecutive slices.
int FastllmCudaGGUFExpertMMQMaxRows(int gateType, int downType, int hidden, int inter, int experts, int topk);
size_t FastllmCudaGGUFExpertMMQBytes(int rows, int hidden, int inter, int experts, int topk);
void FastllmCudaGGUFExpertMMQ(bool gate, int type, const void *weights, const float *outer,
        const int *indices, const float *input, float *output, void *workspace,
        int rows, int hidden, int inter, int experts, int topk);
