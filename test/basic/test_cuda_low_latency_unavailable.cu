#include "devices/cuda/low_latency/dense.h"
#include <cstdio>

int main() {
    int devices=0;
    if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    namespace ll=fastllm::cuda::low_latency;
    // This executable intentionally contains only compute_75 PTX. JIT on a
    // newer physical GPU does not turn the guarded BF16 MMA stub into a kernel.
    // Admission must check the source image as well as the physical device.
    ll::DensePlan plan;
    const auto error=ll::PrepareDense({2,4096,2048},plan,cudaStreamPerThread);
    if(!ll::BackendUnavailable(error) || plan.Ready()) {
        std::fprintf(stderr,"FAIL unsupported image admitted: %s\n",cudaGetErrorString(error));
        return 1;
    }
    std::puts("PASS unavailable image rejected before launch");
    return 0;
}
