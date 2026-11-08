#ifndef FASTLLM_TP2_PREFILL_H
#define FASTLLM_TP2_PREFILL_H
#include <functional>
namespace fastllm {
class Data;
}
bool FastllmCanUseTP2WHT6AllReduceAdd(int count, int dataType, int deviceId);
// FASTLLM_TP2_WHT6_ALLREDUCE=1; default off, SM80/86 only.
// Opt-in lossy WHT6 transport for large eager FP16 TP=2 contributions.
// Both ranks retain the same uncompressed residual and add it after reduction.
// Returns false without changing tensors if the path is not applicable.
bool FastllmTryTP2WHT6AllReduceAdd(const fastllm::Data &partial, fastllm::Data &residual, int deviceId);
// FASTLLM_TP2_MLP_OVERLAP=1; default off, independent of the WHT6 flag.
// Opt-in eager SM80/86 MLP pipeline, 2048 tokens split into four 512-token chunks.
// Requires Marlin FP8 weights. Uses WHT6 when enabled, otherwise native NCCL FP16.
// False means ineligible with no work submitted; execution errors never fall back.
// FASTLLM_QWEN35_TP4_MLP_OVERLAP: same pipeline on four FP8 ranks, any rows.
bool FastllmTryTP4MlpOverlap(const fastllm::Data &input, fastllm::Data &gateUp,
                             const fastllm::Data &gateUpBias, fastllm::Data &down,
                             const fastllm::Data &downBias, fastllm::Data &residual,
                             int tokens, int deviceId,
                             const std::function<void(int, int)> *normChunk = nullptr);
// FASTLLM_QWEN35_TP4_ATTN_OVERLAP: while armed, this thread's next in-place FP16
// reduce is held instead of run, for the pipeline above; flush runs it in place.
void FastllmTP4ArmDeferredReduce(bool armed);
bool FastllmTP4HasDeferredReduce();
void FastllmTP4FlushDeferredReduce(int deviceId);
bool FastllmTryTP2MlpOverlap(const fastllm::Data &input, fastllm::Data &gateUp,
                             const fastllm::Data &gateUpBias, fastllm::Data &down,
                             const fastllm::Data &downBias, fastllm::Data &residual, int deviceId);
#endif
