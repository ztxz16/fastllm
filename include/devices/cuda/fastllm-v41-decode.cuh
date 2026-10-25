#pragma once

#include "fastllm.h"
#include <vector>

struct FastllmCudaV41DecodeTransform {
    fastllm::Data *input;
    fastllm::Data *window;
    int startPos;
    int quantMode;
    bool inverse;
    int quantBlock = 32;
};

// Single-token BF16 rows, with independent request positions and ring buffers.
// Rejection does not mutate inputs; callers retain the ordinary CUDA fallback.
extern "C" bool
FastllmCudaV41DecodeTransformsBatch(const std::vector<FastllmCudaV41DecodeTransform> &items,
                                    int ropeDim, float ropeBase, int originalSeqLen,
                                    float ropeFactor, int betaFast, int betaSlow, int windowSize);

struct FastllmCudaV41DecodeIndexTask {
    fastllm::Data *query, *weights, *keys, *candidates, *topk;
    int startPos;
    bool writeCandidates, useCandidates;
};

extern "C" bool
FastllmCudaV41DecodeIndexerBatch(const std::vector<FastllmCudaV41DecodeIndexTask> &items, int ratio,
                                 int topK, int blockSize, int candidateTopK, int ropeDim,
                                 float ropeBase, int originalSeqLen, float ropeFactor, int betaFast,
                                 int betaSlow, bool quantizeQuery);

// Small-batch BF16 input, FP32 router weights/output. The original FP32
// reduction and sqrt(softplus) score transform are preserved.
extern "C" bool FastllmCudaV41RouterScores(const fastllm::Data &input,
    fastllm::Data &weight, float temperature, fastllm::Data &scores);

// Finish the current mix while normalizing with the previous sublayer coefficients.
// When provided, quantizedOutput receives the same BF16 block-32 fake-quantized
// values as FastllmCudaDeepSeekV41QuantizeActivation, without another kernel.
extern "C" bool FastllmCudaDeepSeekV41HcMixPreNorm(const fastllm::Data &x, const fastllm::Data &previousPre,
    fastllm::Data &fn, fastllm::Data &scale, fastllm::Data &base, fastllm::Data &normWeight,
    int sinkhornIters, float mixEps, float normEps, fastllm::Data &pre, fastllm::Data &post,
    fastllm::Data &comb, fastllm::Data &output, fastllm::Data *quantizedOutput = nullptr);

// Small BF16 multi-row attention post and next projection. Inputs must be
// local CUDA tensors; outputs must be distinct, owned tensors.
extern "C" bool FastllmCudaDeepSeekV41HcPostMixPreNorm(const fastllm::Data &input,
    const fastllm::Data &residual, const fastllm::Data &previousPre,
    const fastllm::Data &previousPost, const fastllm::Data &previousComb,
    const fastllm::Data &fn, const fastllm::Data &scale, const fastllm::Data &base,
    const fastllm::Data &normWeight, int sinkhornIters, float mixEps, float normEps,
    fastllm::Data &hidden, fastllm::Data &pre, fastllm::Data &post,
    fastllm::Data &comb, fastllm::Data &output);

// Also emits a distinct block-32 fake-FP8 BF16 copy for the expert branches.
extern "C" bool FastllmCudaDeepSeekV41HcPostMixPreNormQuantized(const fastllm::Data &input,
    const fastllm::Data &residual, const fastllm::Data &previousPre,
    const fastllm::Data &previousPost, const fastllm::Data &previousComb,
    const fastllm::Data &fn, const fastllm::Data &scale, const fastllm::Data &base,
    const fastllm::Data &normWeight, int sinkhornIters, float mixEps, float normEps,
    fastllm::Data &hidden, fastllm::Data &pre, fastllm::Data &post,
    fastllm::Data &comb, fastllm::Data &output, fastllm::Data &quantizedOutput);

// Full-K output-channel slice; no partial dot products or reduction across GPUs.
// For M > 1 writes packed [M, rowCount] at offset M * rowStart in output.
extern "C" bool FastllmCudaBFloat16FP8ReplicatedRowSlice(const fastllm::Data &input,
    fastllm::Data &weight, fastllm::Data &output, int rowStart, int rowCount);

// Local shared-expert intermediate TP partition. Input is already block-32
// quantized BF16. Scratch/output are preallocated on the current device.
// Produces FP32 partials; the caller merges them before its existing reduce.
extern "C" bool FastllmCudaV41SharedExpertTP(const fastllm::Data &input,
    fastllm::Data &gateUpWeight, fastllm::Data &downWeight,
    fastllm::Data &gateUp, fastllm::Data &activation, fastllm::Data &output,
    int rank, int ranks, float swigluLimit);

// Concatenate native row-major FP8 matrices with matching block scales.
// Destination is caller-owned and must be empty; setup runs before capture.
extern "C" bool FastllmCudaConcatFp8Rows(const fastllm::Data &first,
    const fastllm::Data &second, fastllm::Data &output);

// Packed BF16 RMSNorm; an optional fake-FP8 query output serves W8A8 emulation.
extern "C" bool FastllmCudaV41QKVNormPacked(const fastllm::Data &qkv,
    const fastllm::Data &qw, const fastllm::Data &kw, float eps,
    fastllm::Data &qn, fastllm::Data &kn, fastllm::Data *qq = nullptr);

// Compatibility wrapper retaining the original fake-FP8 query output.
extern "C" bool FastllmCudaV41QKVNormQuantPacked(const fastllm::Data &qkv,
    const fastllm::Data &qWeight, const fastllm::Data &kvWeight, float eps,
    fastllm::Data &qNorm, fastllm::Data &qQuant, fastllm::Data &kvNorm);

// Independent B1 block-scaled FP8 projections, retaining each full-K reduction.
// Inputs, weights and preallocated outputs must reside on the current device.
extern "C" bool FastllmCudaBFloat16FP8LinearPair(const fastllm::Data &input,
    fastllm::Data &firstWeight, fastllm::Data &secondWeight,
    fastllm::Data &firstOutput, fastllm::Data &secondOutput);

// BF16 Q/KV RMSNorm with the original 512-thread reduction. Q is rounded to
// BF16 before block-32 FP8 round-trip; both rounded and quantized Q are retained.
// KV output may alias KV input. Other aliases and unsupported shapes reject.
extern "C" bool FastllmCudaV41QKVNormQuant(const fastllm::Data &q, const fastllm::Data &kv,
    const fastllm::Data &qWeight, const fastllm::Data &kvWeight, float eps,
    fastllm::Data &qNorm, fastllm::Data &qQuant, fastllm::Data &kvNorm);

extern "C" bool FastllmCudaV41RouterSelect(const fastllm::Data &input, fastllm::Data &weight,
    fastllm::Data &bias, float temperature, float routeScale, fastllm::Data &logits,
    fastllm::Data &indices, fastllm::Data &scores);

extern "C" bool FastllmCudaV41RouterSelectTopK(const fastllm::Data &input, fastllm::Data &weight,
    fastllm::Data &bias, float temperature, float routeScale, fastllm::Data &logits,
    fastllm::Data &indices, fastllm::Data &scores, int topk);
