#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
void CheckCuda(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

void FillWeight(fastllm::Data &weight) {
    weight.isModelWeight = true;
    weight.Allocate();
    for (int row = 0; row < weight.dims[0]; ++row) {
        const float value = (row % 7 - 3) * 0.25f;
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        for (int col = 0; col < weight.dims[1]; ++col) {
            ((uint16_t *)weight.cpuData)[row * weight.dims[1] + col] = bits >> 16;
        }
    }
}

void CheckLinear(fastllm::Data &weight, const fastllm::Data &bias, int device, int rows) {
    using namespace fastllm;
    const int m = weight.dims[1], k = weight.dims[0];
    std::vector<float> values(rows * m);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < m; ++col) values[row * m + col] = (row % 5 + 1) * 0.125f;
    }
    Data input(FLOAT32, {rows, m}, values);
    Data output(FLOAT32, {rows, k});
    input.ToDevice(DataDevice::CUDA, std::vector<int>{device});
    output.ToDevice(DataDevice::CUDA, std::vector<int>{device});
    output.Allocate();
    FastllmCudaMatMulBFloat16(input, weight, bias, output, rows, m, k);
    CheckCuda(cudaDeviceSynchronize());
    output.ToDevice(DataDevice::CPU);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < k; ++col) {
            float expected = m * (row % 5 + 1) * 0.125f * (col % 7 - 3) * 0.25f;
            if (!bias.dims.empty()) expected += (col % 11 - 5) * 0.0625f;
            float actual = ((float *)output.cpuData)[row * k + col];
            if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-5f) {
                std::fprintf(stderr, "GPU %d rows=%d [%d,%d]: %.9g != %.9g\n",
                             device, rows, row, col, actual, expected);
                throw std::runtime_error("BF16 linear output mismatch");
            }
        }
    }
}
} // namespace

int main() {
    using namespace fastllm;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) {
        std::puts("SKIP: two CUDA devices required");
        return 77;
    }
    try {
        // Leave peer access disabled in this fresh process: every upload and
        // input/output tensor must use the selected GPU's own allocation.
        Data weight(BFLOAT16, {1280, 2560}), emptyBias;
        FillWeight(weight);
        // The same CPU expert is assigned to different GPUs across requests.
        for (int device : {0, 1, 0, 1}) {
            CheckCuda(cudaSetDevice(device));
            weight.ToCudaTemporary({}, true);
            for (int rows : {1, 2, 3, 4, 7, 8, 17}) CheckLinear(weight, emptyBias, device, rows);
            weight.FreeCudaTemporary({}, false);
            std::printf("PASS bias-free expert migration to GPU %d\n", device);
        }
        // Exercise the unchanged real-bias path on each device as well.
        for (int device : {0, 1}) {
            CheckCuda(cudaSetDevice(device));
            Data biasedWeight(BFLOAT16, {1280, 2560});
            FillWeight(biasedWeight);
            std::vector<float> values(1280);
            for (int col = 0; col < 1280; ++col) values[col] = (col % 11 - 5) * 0.0625f;
            Data bias(FLOAT32, {1280}, values);
            bias.ToDevice(DataDevice::CUDA, std::vector<int>{device});
            biasedWeight.ToCudaTemporary({}, true);
            for (int rows : {1, 4, 8, 17}) CheckLinear(biasedWeight, bias, device, rows);
            biasedWeight.FreeCudaTemporary({}, false);
            for (void *ptr : biasedWeight.extraCudaData) CheckCuda(cudaFree(ptr));
            biasedWeight.extraCudaData.clear();
            std::printf("PASS nonzero bias on GPU %d\n", device);
        }
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    } catch (const char *error) {
        std::fprintf(stderr, "%s\n", error);
        return 1;
    }
    return 0;
}
