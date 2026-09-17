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
