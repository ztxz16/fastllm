// Test-only adapter for the production FP8 Data dispatch path.
#include "fp8Sm90TestBridge.cpp"

namespace fastllm {
extern "C" int FastllmCudaGetLinearExactBatchThreshold();
extern "C" void FastllmCudaSetLinearExactBatchThreshold(int threshold);
}

extern "C" int HopperTestSetExactThreshold(int threshold) {
    const int previous = fastllm::FastllmCudaGetLinearExactBatchThreshold();
    fastllm::FastllmCudaSetLinearExactBatchThreshold(threshold);
    return previous;
}
