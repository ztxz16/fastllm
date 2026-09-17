#pragma once
#include "device.h"
namespace fastllm {
// Dense CUDA RMSNorm + a small unquantized Linear. Also emits the normalized
// tensor for other consumers. Outputs must be distinct and not alias inputs.
// Fusion: FP16/BF16, batch 1..8, K=1024..8192 in steps of 1024, N=1..256.
// Uses ordinary SIMT operations; CUDA supplies software BF16 conversion where
// needed. A loadable image for the selected specialization is required. SM75+
// compile coverage; only SM120 performance has been measured.
// Other valid Linear/RMSNorm configurations use the complete original path.
// Aliasing violates both paths' contract and is rejected before Reshape writes.
bool FastllmCudaRMSNormSmallLinearCanRun(const Data &input, const Data &norm, const Data &weight,
                                         const Data &bias, const Data &normalized, const Data &output);
void FastllmCudaRMSNormSmallLinear(const Data &input, const Data &norm, const Data &weight, const Data &bias,
                                   Data &normalized, Data &output, float eps);
class CudaRMSNormSmallLinearOp : public BaseOperator {
  public:
    bool CanRun(const std::string &, const DataDict &, const FloatDict &, const IntDict &) override;
    void Reshape(const std::string &, const DataDict &, const FloatDict &, const IntDict &) override;
    void Run(const std::string &, const DataDict &, const FloatDict &, const IntDict &) override;
};
bool CudaRMSNormSmallLinearBlock(Data &input, Data &norm, Data &weight, const Data &bias, Data &normalized,
                                 Data &output, float eps);
} // namespace fastllm
