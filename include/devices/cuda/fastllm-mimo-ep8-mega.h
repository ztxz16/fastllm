#pragma once
#include "fastllm.h"
#include <memory>

// Opt-in MiMo complete-expert EP8 backend. All handles and symmetric buffers
// for decode are prepared before worker execution / CUDA Graph capture.
struct FastllmCudaMimoEp8Mega;
bool FastllmCudaMimoEp8MegaSupported(const std::vector<int> &devices);
// Required backing for an eligible eager prefill chunk, or zero for the
// existing small-chunk path / builds without the SM90 backend.
size_t FastllmCudaMimoEp8PrefillWorkspaceBytes(int rows);
std::shared_ptr<FastllmCudaMimoEp8Mega> FastllmCudaCreateMimoEp8Mega(
    const std::vector<int> &devices,
    const std::vector<std::vector<std::vector<fastllm::Data *>>> &weights);
// Eager-only, in-place gather of contiguous row shards. Reuses the existing
// EP8 peer publication barriers; all data stays on GPUs, with no extra arena.
void FastllmCudaMimoEp8PrefillAllGather(FastllmCudaMimoEp8Mega &state, int rank,
                                      fastllm::Data &output);
// Equal prefill shards only. The reduced slice remains at its original
// row offset; other rows are not valid collective outputs.
void FastllmCudaMimoEp8PrefillReduceScatter(FastllmCudaMimoEp8Mega &state, int rank,
                                          fastllm::Data &output);
// Routing may cover the full input or this rank's equal contiguous token shard.
// Sharded results are gathered in-place, unless the optional residual/RMSNorm
// fusion succeeds. True means residual and normOutput are ready; in that case
// output only contains this rank's slice and must not be consumed as a full tensor.
bool FastllmCudaRunMimoEp8Mega(FastllmCudaMimoEp8Mega &state, int rank, int layer,
    const fastllm::Data &input, const fastllm::Data &index, const fastllm::Data &score,
    fastllm::Data &output, bool graphDecode, fastllm::Data *residual = nullptr,
    const fastllm::Data *normWeight = nullptr, fastllm::Data *normOutput = nullptr, float eps = 0,
    fastllm::Data *prefillScratch = nullptr, const int *liveKeys = nullptr);
