#ifndef FASTLLM_EXL3_H
#define FASTLLM_EXL3_H

#include "fastllm.h"
#include <functional>

namespace fastllm {
// An EXL3 part stores the original trellis followed by FP16 input and output
// scales. Logical weights are [outputs, inputs]. blockK = twice the bit rate;
// blockM = number of equal-sized, independently rotated output projections.
// Grouped weights are [groups, outputs, inputs], blockM = groups; each group
// has its own input slice and independent rotation scales.
// Keeping projections separate is essential when gate and up have different suh.
bool Exl3BitRateSupported(int twiceBits);
uint64_t Exl3PartBytes(int outputs, int inputs, int twiceBits);
uint64_t Exl3StorageBytes(const Data &weight);
void Exl3Hadamard128(float *values, size_t count);
void Exl3DecodeInner(const uint8_t *packed, int outputs, int inputs, int twiceBits,
                     float *decoded);
void Exl3Reconstruct(const uint8_t *packed, int outputs, int inputs, int twiceBits,
                     float *decoded);
// NUMA format primitives. Shards retain every trellis/scale byte exactly once.
// Dot rows are independent 16-column tiles; output transforms still own 128.
void Exl3CopyRowShard(const Data &weight, uint8_t *destination, int node, int nodes);
void Exl3CopyCanonical(const Data &weight, uint8_t *destination);
void Exl3PrepareInputBlock(const Data &weight, int part, const float *input,
    float *rotated, int first, bool roundBfloat);
// Large x86 decode matrices use blockwise 12-bit activations and the affine
// mul1 codebook. Short matrices, other ISAs and reconstruction stay exact.
// Nonfinite inputs or an unrepresentable quantization scale use the exact decoder.
void Exl3DotRows(const Data &weight, int part, int node, int nodes,
    const float *rotated, float *output, int first, int count);
void Exl3FinishOutputBlock(const Data &weight, int part, float *output, int first);
void Exl3MoeCPU(const Data &input, Data &output, const Data &index, const Data &score,
                Data **weights, int weightsBatch, float sharedScale, float swigluLimit);
void Exl3GroupedCPU(const Data &input, const Data &weight, Data &output);
void Exl3LinearCPU(const Data &input, const Data &weight, const Data &bias, Data &output);
// Per-route BF16 expert boundaries, weighted FP32 results. GPU routes are skipped.
// The callback runs after the gate/up workers have been submitted; it must not
// submit work to the same CPU pool. Registered hybrid decode uses the common
// NUMA worker interface; this remains the portable standalone fallback.
void Exl3MoeCPUExperts(const float *input, float *output, int rows, Data **weights,
    const int32_t *indices, const int32_t *gpuIndices, const float *scores,
    int topk, float limit, const std::function<void()> &submitGpu = {});
}

#ifdef USE_CUDA
void FastllmCudaRestoreExl3Weight(const fastllm::Data &weight, void *destination, void *stream = nullptr);
bool FastllmCudaExl3Grouped(const fastllm::Data &input, const fastllm::Data &weight, fastllm::Data &output);
void FastllmCudaReleaseExl3(const fastllm::Data *owner);
bool FastllmCudaExl3Moe(const fastllm::Data &input, fastllm::Data &output,
    const fastllm::Data &index, const fastllm::Data &score, fastllm::Data **weights, int count, float limit);
bool FastllmCudaExl3LinearShared(const fastllm::Data &input, fastllm::Data *const *weights,
                               fastllm::Data *const *outputs, int count);
bool FastllmCudaExl3Linear(const fastllm::Data &input, fastllm::Data &weight,
                          const fastllm::Data &bias, fastllm::Data &output);
// Opaque cache records retain the original trellis and scale bytes. Inactive
// route slots are -1; each output row is the weighted result of that route.
// Two bounded workspaces isolate parallel resident/staged graph branches.
// Optional routeMap maps compact selected slots to original row/top-k routes;
// preserveMissing leaves outputs belonging to another launch untouched.
bool FastllmCudaExl3Cache(const fastllm::Data &input, const uint8_t *records,
    const int32_t *slots, const uint64_t *offsets, size_t stride, size_t downOffset,
    int inter, int gateBits, int downBits, const float *scores, int topk,
    float limit, float *perExpert, int workspace = 0,
    const int32_t *routeMap = nullptr, int routeCount = 0, bool preserveMissing = false);
#endif
#endif
