#pragma once
// Stable softmax routing for FP16/BF16 logits, E<=256 and K<=16. Equal logits
// choose the lower expert id; FP32 score sums follow expert-id order. This raw
// overload leaves the Data API's legacy tie policy unchanged. Outputs are
// [rows,K] INT32/FP32. rowStride=0 means E; larger strides accept fused gates.
// Optional [ceil(rows/maskRows),E] byte masks share allowed experts across a
// group of rows. At least one expert per group must be allowed; logits must
// be finite before masking. The stream defaults to cudaStreamPerThread.
bool FastllmCudaFusedSoftmaxSelectExpert(const void *logits, int *indices, float *scores,
    int rows, int experts, int topk, bool normalizeTopK, int rowStride = 0,
    const unsigned char *allowed = nullptr, int maskRows = 1,
    void *stream = nullptr, bool fp16 = false);
