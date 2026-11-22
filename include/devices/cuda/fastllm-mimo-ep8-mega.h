#pragma once
#include "fastllm.h"
#include <memory>

// MiMo complete-expert EP4/EP8 backend. All handles and symmetric buffers
// for decode are prepared before worker execution / CUDA Graph capture.
// The backend is selected from the group architecture and expert weight format.
struct FastllmCudaMimoEp8Mega;
bool FastllmCudaMimoEp8MegaSupported(const std::vector<int> &devices);
// One row-major E2M1 plane followed by one UE8M0 byte per complete K32 group.
inline bool FastllmCudaMimoIsMxfp4Expert(const fastllm::Data &w) {
    return w.dataType == fastllm::NVFP4 && w.blockK == 1 && w.blockM == 32 &&
           !w.IsRepacked && !w.cudaNativeNvfp4Layout && w.scales.empty() &&
           w.dims.size() == 2 && w.dims[0] > 0 && w.dims[1] > 0 && w.dims[1] % 32 == 0 &&
           w.GetBytes() == fastllm::GetNVFP4StorageBytes(w.dims[0], w.dims[1], 1, 32);
}
// Query only after group admission; preserves the calling CUDA device.
bool FastllmCudaMimoEp8MegaOwnsPackedWeights(int device);
// Required backing for an eligible eager prefill chunk, or zero for the
// existing small-chunk path / builds without the SM90 backend.
size_t FastllmCudaMimoEp8PrefillWorkspaceBytes(int rows, int ranks);
std::shared_ptr<FastllmCudaMimoEp8Mega> FastllmCudaCreateMimoEp8Mega(
    const std::vector<int> &devices,
    const std::vector<std::vector<std::vector<fastllm::Data *>>> &weights);
std::shared_ptr<FastllmCudaMimoEp8Mega> FastllmCudaCreateMimoEp8Mega(
    const std::vector<int> &devices,
    const std::vector<std::vector<std::vector<fastllm::Data *>>> &weights, int topk);
// Coordinator only, before rank workers submit collectives. Prepare native
// graph capacities up to the requested per-rank row bound; cached calls are cheap.
void FastllmCudaPrepareMimoEp8Mega(FastllmCudaMimoEp8Mega &state, int maxRows);
// Eager-only, in-place gather of contiguous row shards. Reuses the existing
// EP8 peer publication barriers; all data stays on GPUs, with no extra arena.
void FastllmCudaMimoEp8PrefillAllGather(FastllmCudaMimoEp8Mega &state, int rank,
                                      fastllm::Data &output);
// Equal prefill shards only. The reduced slice remains at its original
// row offset; other rows are not valid collective outputs.
void FastllmCudaMimoEp8PrefillReduceScatter(FastllmCudaMimoEp8Mega &state, int rank,
                                          fastllm::Data &output);
// Routing may cover the full input or this rank's contiguous token shard;
// owner boundaries are floor(total * rank / ranks), including uneven batches.
// Sharded results are gathered in-place, unless the optional residual/RMSNorm
// fusion succeeds. True means residual and normOutput are ready; in that case
// output only contains this rank's slice and must not be consumed as a full tensor.
bool FastllmCudaRunMimoEp8Mega(FastllmCudaMimoEp8Mega &state, int rank, int layer,
    const fastllm::Data &input, const fastllm::Data &index, const fastllm::Data &score,
    fastllm::Data &output, bool graphDecode, fastllm::Data *residual = nullptr,
    const fastllm::Data *normWeight = nullptr, fastllm::Data *normOutput = nullptr, float eps = 0,
    fastllm::Data *prefillScratch = nullptr, const int *liveKeys = nullptr);

bool FastllmCudaMimoEp8MegaWeights(FastllmCudaMimoEp8Mega&,int,int,const void**);
bool FastllmCudaMimoEp8MegaWeights(FastllmCudaMimoEp8Mega&,int,int,const void**,bool *mxfp4);
// Dispatch rank-local token rows and return them in the same local order,
// without an output AllGather. Every EP rank must pass the same maxRows;
// local row counts may differ, including zero. Existing world-row dispatch
// remains available for layouts whose consumers need replicated activations.
void FastllmCudaRunMimoEp8MegaLocal(FastllmCudaMimoEp8Mega &state, int rank, int layer,
    const fastllm::Data &input, const fastllm::Data &index, const fastllm::Data &score,
    fastllm::Data &output, int maxRows, const int *liveKeys = nullptr,
    fastllm::Data *prefillScratch = nullptr);
// Scheduler-provided lower bound on the number of real, unmasked rows on every rank.
void FastllmCudaRunMimoEp8MegaLocal(FastllmCudaMimoEp8Mega &state, int rank, int layer,
    const fastllm::Data &input, const fastllm::Data &index, const fastllm::Data &score,
    fastllm::Data &output, int maxRows, const int *liveKeys,
    fastllm::Data *prefillScratch, int densePrefixRows);
