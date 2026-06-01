// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch.h"
#include "src/turbomind/kernels/gemm/registry.h"
#include "src/turbomind/kernels/gemm/utils.h"

namespace turbomind::gemm {

Registry::Registry(std::shared_ptr<cudaDeviceProp> device_prop):
    device_prop_{std::move(device_prop)}, arch_{device_prop_->major * 100 + device_prop_->minor * 10}
{
    // Register the V100 kernels we actually use in this build:
    // AWQ uint4, FP8/E4M3, and dense fp16 Tensor Core paths.
    sm70_884_4();
    sm70_884_8();
    sm70_884_16();
}

bool Registry::Add(std::unique_ptr<Kernel> kernel)
{
    bool is_valid = true;

    if (!is_arch_compatible(kernel->arch(), arch_)) {
        is_valid = false;
    }

    if ((int)device_prop_->sharedMemPerBlockOptin < kernel->smem_size()) {
        is_valid = false;
    }

    if (is_valid) {
        ptrs_.push_back(kernels_.emplace_back(transpose(*kernel)).get());
        ptrs_.push_back(kernels_.emplace_back(std::move(kernel)).get());
    }

    return true;
}

}  // namespace turbomind::gemm
