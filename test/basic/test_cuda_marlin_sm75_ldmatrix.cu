#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include "marlin_template.cuh"

template<int Count>
__global__ void LoadFragments(uint32_t *output) {
#if __CUDA_ARCH__ >= 750
    extern __shared__ uint16_t tile[];
    const int lane = threadIdx.x;
    for (int i = lane; i < Count * 64; i += 32) tile[i] = uint16_t(i + 1);
    __syncthreads();
    constexpr auto type = fastllm_marlin_moe_types::kFloat16.id();
    marlin_moe_wna16::MarlinScalarType<type>::FragA fragment;
    // Only the contributing lanes address the allocation. On SM75, ldsm
    // must also supply valid addresses for the unused upper lanes.
    marlin_moe_wna16::ldsm<Count, type>(fragment, tile + lane * 8);
    for (int i = 0; i < Count; ++i) output[lane * Count + i] = ((uint32_t *)&fragment)[i];
#endif
}

void CheckCuda(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

int main() {
    cudaDeviceProp properties;
    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess ||
        properties.major != 7 || properties.minor != 5) {
        std::puts("SKIP: SM75 device required");
        return 77;
    }
    try {
        uint32_t *output;
        CheckCuda(cudaMalloc(&output, 128 * sizeof(uint32_t)));
        for (int count : {1, 2, 4}) {
            if (count == 1) LoadFragments<1><<<1, 32, 128>>>(output);
            if (count == 2) LoadFragments<2><<<1, 32, 256>>>(output);
            if (count == 4) LoadFragments<4><<<1, 32, 512>>>(output);
            CheckCuda(cudaDeviceSynchronize());
            std::vector<uint32_t> values(32 * count);
            CheckCuda(cudaMemcpy(values.data(), output, values.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            for (int lane = 0; lane < 32; ++lane) {
                for (int matrix = 0; matrix < count; ++matrix) {
                    const int index = matrix * 64 + (lane / 4) * 8 + (lane % 4) * 2;
                    const uint32_t expected = (index + 1) | ((index + 2) << 16);
                    if (values[lane * count + matrix] != expected) {
                        throw std::runtime_error("ldmatrix fragment mismatch");
                    }
                }
            }
            std::printf("PASS ldmatrix x%d\n", count);
        }
        CheckCuda(cudaFree(output));
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
