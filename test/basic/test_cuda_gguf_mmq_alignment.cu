#include "fastllm-cuda.cuh"
#define GGML_COMMON_DECL_CUDA
#include "gguf.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include "cuda_gguf_weights_test.cuh"

static void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

static void Cuda(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

template<typename T>
static void Test(int rows, int columns, int outputs, ggml_type type = GGML_TYPE_Q8_0) {
    constexpr bool bf16 = std::is_same<T, __nv_bfloat16>::value;
    constexpr bool fp32 = std::is_same<T, float>::value;
    const auto dtype = fp32 ? fastllm::FLOAT32 : bf16 ? fastllm::BFLOAT16 : fastllm::FLOAT16;
    // One extra output row catches tail writes past the requested range.
    fastllm::Data input(dtype, {rows, columns}), output(dtype, {rows + 1, outputs});
    fastllm::Data weight(fastllm::DATA_GGUF_FORMAT), bias;
    weight.ggmlType = type;
    weight.isGGUFData = true;
    weight.Resize({outputs, columns});
    weight.Allocate();
    input.Allocate();
    std::vector<float> sums(outputs, 0);
    if (type == GGML_TYPE_Q8_0) {
        auto *blocks = reinterpret_cast<block_q8_0 *>(weight.cpuData);
        for (int r = 0; r < outputs; ++r) {
            for (int c = 0; c < columns; ++c) {
                auto &block = blocks[(r * columns + c) / 32];
                block.d = __float2half_rn(1.0f / 128);
                block.qs[c % 32] = (r * 3 + c * 5) % 17 - 8;
                sums[r] += block.qs[c % 32] / 128.0f;
            }
        }
    } else {
        const auto packed = Weights(type, outputs, columns);
        std::memcpy(weight.cpuData, packed.data(), packed.size());
        std::vector<float> decoded(size_t(outputs) * columns);
        ggml_type_to_float(type)(packed.data(), decoded.data(), decoded.size());
        for (int r = 0; r < outputs; ++r)
            for (int c = 0; c < columns; ++c)
                sums[r] += decoded[size_t(r) * columns + c];
    }
    // Exact Q8 scales keep this a test of row offsets and output conversion,
    // without adding activation quantization error to the CPU reference.
    const float rowValues[] = {-127.f/128, -127.f/256, -127.f/512,
                               0, 127.f/512, 127.f/256, 127.f/128};
    auto *values = reinterpret_cast<T *>(input.cpuData);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < columns; ++c)
            values[r * columns + c] = T(rowValues[r % 7]);
    input.ToDevice(fastllm::CUDA, {0}, true);
    weight.ToDevice(fastllm::CUDA, {0}, true);
    output.ToDevice(fastllm::CUDA, {0}, false);
    output.Allocate();
    Cuda(cudaMemset(output.cudaData, 0x5a, output.GetBytes()));

    auto validate = [&]() {
        std::vector<T> actual(size_t(rows + 1) * outputs);
        Cuda(cudaMemcpy(actual.data(), output.cudaData, actual.size() * sizeof(T), cudaMemcpyDeviceToHost));
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < outputs; ++c) {
                const float expected = float(T(sums[c] * rowValues[r % 7]));
                const float observed = float(actual[size_t(r) * outputs + c]);
                if (!std::isfinite(observed) || std::fabs(observed - expected) >= 0.001f) {
                    std::fprintf(stderr, "dtype=%d rows=%d K=%d N=%d row=%d col=%d got=%g expected=%g\n",
                        int(dtype), rows, columns, outputs, r, c, observed, expected);
                    throw std::runtime_error("GGUF linear disagrees with CPU dequantized reference");
                }
            }
        }
        const auto *tail = reinterpret_cast<const unsigned char *>(actual.data() + size_t(rows) * outputs);
        for (size_t i = 0; i < outputs * sizeof(T); ++i)
            Check(tail[i] == 0x5a, "GGUF linear wrote beyond the output tail");
    };

    // Test both admission and the caller's numerical fallback. K = 128 is
    // used by GLM-5.3 KDA; K = 384 catches partial tiles beyond the first one.
    bool accepted;
    if constexpr (fp32) {
        accepted = FastllmCudaFloatMatMulGGUFMMQ(input.cudaData, weight.cudaData,
            output.cudaData, type, rows, columns, outputs, cudaStreamPerThread);
    } else if constexpr (bf16) {
        accepted = FastllmCudaBFloat16MatMulGGUFMMQ(input.cudaData, weight.cudaData,
            output.cudaData, type, rows, columns, outputs, cudaStreamPerThread);
    } else {
        accepted = FastllmCudaHalfMatMulGGUFMMQ(input.cudaData, weight.cudaData,
            output.cudaData, type, rows, columns, outputs, cudaStreamPerThread);
    }
    Check(accepted == (columns % 256 == 0),
        "MMQ partial K tile admission is incorrect");
    Cuda(cudaDeviceSynchronize());
    if (accepted) validate();
    for (int pass = 0; pass < 2; ++pass) {
        if constexpr (fp32) {
            Check(FastllmCudaMatMulFloatGGUF(input, weight, bias, output,
                rows, columns, outputs), "FP32 GGUF linear failed");
        } else if constexpr (bf16) {
            Check(FastllmCudaBFloat16MatMulGGUF(input, weight, bias, output,
                rows, columns, outputs), "BF16 GGUF linear failed");
        } else {
            Check(FastllmCudaHalfMatMulGGUF(input, weight, bias, output,
                rows, columns, outputs), "FP16 GGUF linear failed");
        }
        Cuda(cudaDeviceSynchronize());
        validate();
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
        if (prop.major * 10 + prop.minor < 75) {
            std::puts("FASTLLM_TEST_SKIP_NO_DEVICE: MMQ requires SM75+");
            return 0;
        }
        for (int rows : {33, 129, 1024, 1025, 1031, 2000, 2048, 2057}) {
            for (int columns : {128, 256, 384}) {
                Test<float>(rows, columns, 129);
                Test<half>(rows, columns, 129);
                Test<__nv_bfloat16>(rows, columns, 129);
            }
        }
        Test<__nv_bfloat16>(33, 128, 8192);
        // A 32768-column FP32 scratch row makes the bounded workspace split
        // these wide projections, including a final 1-row or 7-row chunk.
        Test<float>(1025, 256, 32768);
        Test<half>(1031, 256, 32768);
        Test<__nv_bfloat16>(1025, 256, 32768);
        Test<__nv_bfloat16>(2055, 256, 32768);
        // A wide input reaches the quantized-activation budget first, while
        // the narrow output keeps the FP32 scratch well below its limit.
        Test<__nv_bfloat16>(4097, 32768, 129);
        Test<__nv_bfloat16>(65537, 256, 129);
        for (auto type : {GGML_TYPE_Q4_K, GGML_TYPE_IQ4_XS, GGML_TYPE_Q2_K}) {
            std::fprintf(stderr, "CHECK: long and chunked packed type=%d\n", int(type));
            Test<float>(2057, 256, 129, type);
            Test<half>(2057, 256, 129, type);
            Test<__nv_bfloat16>(2057, 256, 129, type);
            Test<__nv_bfloat16>(1025, 256, 32768, type);
        }
        std::puts("PASS: GGUF MMQ K alignment, chunk tails and narrow projection fallback");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
