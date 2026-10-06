#pragma once

#include <functional>
#include <vector>

namespace fastllm { class Data; }

// Decode-only mode selection; the ordinary cache and hybrid capability APIs
// keep their existing meaning. Begin/End bracket a complete sampled token.
bool FastllmCudaUseMoeHybrid(fastllm::Data **weights, int weightsBatch);
void *FastllmCudaBeginMoeDecode(fastllm::Data **weights, int weightsBatch, int topk);
void FastllmCudaEndMoeDecode(void *state);

// The caller hands off these devices' compute streams before borrowing them.
// A nonempty list includes the current device and bounds every GPU used,
// including fallback. Empty retains ordinary hybrid behavior. Cache admission
// remains independent of temporary expert execution.
bool FastllmCudaMergeMOEHybridOnDevices(const fastllm::Data &input,
    const fastllm::Data &index, const fastllm::Data &score, fastllm::Data &output,
    fastllm::Data **weights, int weightsBatch, int layer,
    const std::vector<int> &devices, const std::function<void()> &launchParallel);
