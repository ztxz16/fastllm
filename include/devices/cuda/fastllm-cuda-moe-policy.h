#pragma once

namespace fastllm { class Data; }

// Decode-only mode selection; the ordinary cache and hybrid capability APIs
// keep their existing meaning. Begin/End bracket a complete sampled token.
bool FastllmCudaUseMoeHybrid(fastllm::Data **weights, int weightsBatch);
void *FastllmCudaBeginMoeDecode(fastllm::Data **weights, int weightsBatch, int topk);
void FastllmCudaEndMoeDecode(void *state);
