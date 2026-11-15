#pragma once
#include "models/minimax_h3/runtime.h"

namespace fastllm::minimax_h3 {
// Comfy video projections use the shared image CUDA backends. Auxiliary
// conditioning/modulation and output heads still come from the matched base.
class VideoProjections {
public:
    VideoProjections(Runtime &, const std::string &, nomas::ParallelContext, bool sequenceParallel);
    ~VideoProjections();
    Tensor linear(const Tensor &, const std::string &diffusersName, DType output = DType::BF16);
    void erasePrefix(const std::string &);
    void clear();
    bool collectiveCalibration() const;
    // Packed QKV/FC1 and fused LoRA accumulation form one projection path.
    // NVFP4 keeps separate BF16 delta rounding before activation calibration.
    bool fusedProjections() const;
    void setHostMemoryBudget(const std::shared_ptr<nomas::HostMemoryBudget> &);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace fastllm::minimax_h3
