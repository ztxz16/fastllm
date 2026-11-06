#include "devices/cuda/naive-n05-kv.cuh"
#include <cuda_runtime.h>
#include <cstdio>

// Exhaust all E2M1 pairs and E4M3 scales, including signed zeros, subnormal
// scales and NaNs. Also cover unaligned addresses and vectors crossing blocks.
__global__ void CheckPairs(unsigned *errors) {
    unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= 65536) return;
    __align__(16) uint8_t bytes[48];
    uint8_t scales[4];
    for (int i = 0; i < 48; ++i) bytes[i] = id & 255;
    for (int i = 0; i < 4; ++i) scales[i] = (id >> 8) ^ (i & 1);
    for (int alignment : {0, 4, 8})
    for (int column : {0, 16, 32}) {
        FastllmNaiveKVRow row(bytes + alignment, false, scales, column);
        unsigned pairs[8];
        row.LoadFP4Pairs<8>(pairs);
        for (int i = 0; i < 16; ++i) {
            __nv_bfloat16 scalar = row[i];
            if (((unsigned short *)pairs)[i] != *(unsigned short *)&scalar)
                atomicAdd(errors, 1);
        }
    }
    for (int alignment = 0; alignment < 2; ++alignment)
    for (int column : {0, 1, 4, 8, 12, 15, 16, 24}) {
        FastllmNaiveKVRow row(bytes + alignment, false, scales, column);
        uint4 eight = row.Load8();
        uint2 four = row.Load4();
        for (int i = 0; i < 8; ++i) {
            __nv_bfloat16 scalar = row[i];
            const unsigned short expected = *(unsigned short *)&scalar;
            const unsigned short a = ((unsigned short *)&eight)[i];
            const unsigned short b = i < 4 ? ((unsigned short *)&four)[i] : expected;
            if (a != expected || b != expected) atomicAdd(errors, 1);
        }
    }
}
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    unsigned *device = nullptr, errors = 1;
    if (cudaMalloc(&device, sizeof(unsigned)) != cudaSuccess ||
        cudaMemset(device, 0, sizeof(unsigned)) != cudaSuccess) return 1;
    CheckPairs<<<256, 256>>>(device);
    cudaError_t status = cudaMemcpy(&errors, device, sizeof(unsigned), cudaMemcpyDeviceToHost);
    cudaFree(device);
    if (status != cudaSuccess || errors) {
        std::fprintf(stderr, "FP4 vector decode: %u mismatches, %s\n", errors, cudaGetErrorString(status));
        return 1;
    }
    std::puts("FP4 scalar/vector decode: all 65536 pairs/scales and alignments bitwise equal");
    return 0;
}
