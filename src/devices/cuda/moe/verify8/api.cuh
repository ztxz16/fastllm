#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace fastllm_verify8 {
constexpr int GateCodeBytes = 32 * 1024 / 2;
constexpr int GatePageBytes = GateCodeBytes + 32 * (1024 / 32);
} // namespace fastllm_verify8
struct FastllmVerify8Args {
    CUtensorMap gateMap, downMap;
    const uint8_t *gateScales, *downScales, *xq, *xs;
    const float *router, *bias;
    int *ids;
    float *prob;
    void *out;
    const uint8_t *gatePages;
    int rank, rows;
    const uint64_t *epoch;
    unsigned long long *tagged_up, *peers[8];
};
bool FastllmVerify8Supported(int device);
bool FastllmVerify8PagesSupported(int device);
cudaError_t FastllmVerify8Launch(const FastllmVerify8Args &args, cudaStream_t stream);
