#include "fastllm-cuda.cuh"
#include "gguf.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>

static void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

static void Cuda(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

template<typename T>
static void Test(int rows, int columns, int outputs) {
    constexpr bool bf16 = std::is_same<T, __nv_bfloat16>::value;
    const auto dtype = bf16 ? fastllm::BFLOAT16 : fastllm::FLOAT16;
    fastllm::Data input(dtype, {rows, columns}), output(dtype, {rows, outputs});
    fastllm::Data weight(fastllm::DATA_GGUF_FORMAT), bias;
    weight.ggmlType = GGML_TYPE_Q8_0;
    weight.isGGUFData = true;
    weight.Resize({outputs, columns});
    weight.Allocate();
    input.Allocate();
    std::vector<float> sums(outputs, 0);
    auto *blocks = reinterpret_cast<block_q8_0 *>(weight.cpuData);
    for (int r = 0; r < outputs; ++r) {
        for (int c = 0; c < columns; ++c) {
            auto &block = blocks[(r * columns + c) / 32];
            block.d = __half_as_ushort(__float2half_rn(1.0f / 128));
            block.qs[c % 32] = (r * 3 + c * 5) % 17 - 8;
            sums[r] += block.qs[c % 32] / 128.0f;
        }
    }
    auto *values = reinterpret_cast<T *>(input.cpuData);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < columns; ++c)
            values[r * columns + c] = T((r % 7 - 3) / 8.0f);
    input.ToDevice(fastllm::CUDA, {0}, true);
    weight.ToDevice(fastllm::CUDA, {0}, true);
    output.ToDevice(fastllm::CUDA, {0}, false);
    output.Allocate();

    // Test both admission and the caller's numerical fallback. K = 128 is
    // used by GLM-5.3 KDA; K = 384 catches partial tiles beyond the first one.
    bool accepted;
    if constexpr (bf16) {
        accepted = FastllmCudaBFloat16MatMulGGUFMMQ(input.cudaData, weight.cudaData,
            output.cudaData, GGML_TYPE_Q8_0, rows, columns, outputs, cudaStreamPerThread);
    } else {
        accepted = FastllmCudaHalfMatMulGGUFMMQ(input.cudaData, weight.cudaData,
            output.cudaData, GGML_TYPE_Q8_0, rows, columns, outputs, cudaStreamPerThread);
    }
    Check(accepted == (columns % 256 == 0 && rows <= 1024),
        "MMQ partial K tile admission is incorrect");
    Cuda(cudaDeviceSynchronize());
    for (int pass = 0; pass < 2; ++pass) {
        if constexpr (bf16) {
            Check(FastllmCudaBFloat16MatMulGGUF(input, weight, bias, output,
                rows, columns, outputs), "BF16 GGUF linear failed");
        } else {
            Check(FastllmCudaHalfMatMulGGUF(input, weight, bias, output,
                rows, columns, outputs), "FP16 GGUF linear failed");
        }
        Cuda(cudaDeviceSynchronize());
        std::vector<T> actual(rows * outputs);
        Cuda(cudaMemcpy(actual.data(), output.cudaData, actual.size() * sizeof(T), cudaMemcpyDeviceToHost));
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < outputs; ++c) {
                const float expected = float(T(sums[c] * (r % 7 - 3) / 8.0f));
                const float observed = float(actual[r * outputs + c]);
                if (!std::isfinite(observed) || std::fabs(observed - expected) >= 0.001f) {
                    std::fprintf(stderr, "bf16=%d rows=%d K=%d N=%d pass=%d row=%d col=%d got=%g expected=%g\n",
                        bf16, rows, columns, outputs, pass, r, c, observed, expected);
                    throw std::runtime_error("GGUF linear disagrees with CPU Q8 reference");
                }
            }
        }
    }
}

int main() {
    try {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
            std::puts("FASTLLM_TEST_SKIP_NO_DEVICE: no CUDA device");
            return 0;
        }
        cudaDeviceProp prop;
        Cuda(cudaGetDeviceProperties(&prop, 0));
        if (prop.major * 10 + prop.minor < 80) {
            std::puts("FASTLLM_TEST_SKIP_NO_DEVICE: BF16 test requires SM80+");
            return 0;
        }
        for (int rows : {33, 129, 2048}) {
            for (int columns : {128, 256, 384}) {
                Test<half>(rows, columns, 129);
                Test<__nv_bfloat16>(rows, columns, 129);
            }
        }
        Test<__nv_bfloat16>(33, 128, 8192);
        std::puts("PASS: GGUF MMQ K alignment and narrow projection fallback");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
