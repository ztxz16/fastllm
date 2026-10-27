#pragma once
#include "device.h"
namespace fastllm {
// One decode token per independent request. Cache is [slots, channels, 4];
// slotIds is optional (identity mapping), and must name distinct valid slots.
// Common CUDA Block contract, also required by the unfused fallback. Dense
// input/cache/conv/slot tensors on one device; FP16/BF16 activation and history,
// FP32 convolution and biases; input/weight base pointers are 16-byte aligned.
// Outputs and scratch must not alias inputs/cache.
// Fusion supports SM75+ (software FP8 conversion below SM89). The build must
// include a compatible image; only SM120 performance has been measured.
bool FastllmCudaGdnInputConvValidInputs(const Data &input, const Data &weight, const Data &bias,
                                        const Data &convWeight, const Data &convBias, const Data &cache,
                                        const Data *slotIds, int batch);
bool FastllmCudaGdnInputConvCanRun(const Data &input, const Data &weight, const Data &bias,
                                   const Data &convWeight, const Data &convBias, const Data &cache,
                                   const Data *slotIds, int batch);
void FastllmCudaGdnInputConv(Data &input, Data &weight, const Data &bias, const Data &convWeight,
                             const Data &convBias, Data &cache, const Data *slotIds, Data &convOutput,
                             Data &z, int batch);
void FastllmCudaGdnProjectedConv(const Data &projected, const Data &convWeight, const Data &convBias,
                                 Data &cache, const Data *slotIds, Data &convOutput, Data &z, int batch);

class CudaGdnInputConvOp : public BaseOperator {
  public:
    bool CanRun(const std::string &, const DataDict &, const FloatDict &, const IntDict &) override;
    void Reshape(const std::string &, const DataDict &, const FloatDict &, const IntDict &) override;
    void Run(const std::string &, const DataDict &, const FloatDict &, const IntDict &) override;
};
// Returns whether fusion was selected. Both branches produce identical layouts
// and update cache exactly once. Scratch is only used by the fallback.
bool CudaGdnInputConvBlock(Data &input, Data &weight, const Data &bias, const Data &convWeight,
                           const Data &convBias, Data &cache, const Data *slotIds, Data &convOutput, Data &z,
                           Data &scratch, int batch);
} // namespace fastllm

// Hopper BF16 decode fusions; unsupported shapes retain the caller fallback.
extern "C" {
bool FastllmCudaRMSNormSiluMulBFloat16(
    const fastllm::Data &input, fastllm::Data &weight,
    const fastllm::Data &gateInput, fastllm::Data &output, float eps);
bool FastllmRecurrentGatedDeltaRuleNormBaBFloat16(
    fastllm::Data &q, fastllm::Data &k, fastllm::Data &v,
    fastllm::Data &a, fastllm::Data &b, fastllm::Data &normWeight,
    fastllm::Data &aLog, fastllm::Data &dtBias, fastllm::Data &state,
    fastllm::Data &output, float eps, float qScale);
// BF16 graph state is [slots,Hv,K,V]. Slot IDs must be distinct and in range;
// the caller owns the pools and keeps their device addresses stable for replay.
bool FastllmCudaBFloat16GdnGraphSupported();
bool FastllmRecurrentGatedDeltaRuleBFloat16Slots(
    const fastllm::Data &conv, const fastllm::Data &ba,
    const fastllm::Data &norm, const fastllm::Data &aLog, const fastllm::Data &dtBias,
    fastllm::Data &statePool, const fastllm::Data &slotIds, fastllm::Data &output,
    int batch, int keyHeads, int valueHeads, float eps, float qScale);
bool FastllmCudaBFloat16ConvSiluSlots(
    fastllm::Data &pool, const fastllm::Data &slots, const fastllm::Data &input,
    const fastllm::Data &weight, const fastllm::Data &bias, fastllm::Data &output, int batch);
}
