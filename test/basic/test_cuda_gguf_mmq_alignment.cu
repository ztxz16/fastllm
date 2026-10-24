#include "fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-gguf-projections.h"
#include "executor.h"
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

template<typename T>
static void TestGroupedWoA(int tokens) {
    using namespace fastllm;
    constexpr bool bf16 = std::is_same<T, __nv_bfloat16>::value;
    const auto dtype = std::is_same<T, float>::value ? FLOAT32 : bf16 ? BFLOAT16 : FLOAT16;
    constexpr int groups = 8, columns = 256, outputs = 129;
    Data input(dtype, {1, tokens, groups, columns});
    Data weight(DATA_GGUF_FORMAT, GGML_TYPE_Q5_K, {groups, outputs, columns}), output;
    input.Allocate(false); weight.Allocate(false);
    const auto packed = Weights(GGML_TYPE_Q5_K, groups * outputs, columns);
    memcpy(weight.cpuData, packed.data(), packed.size());
    std::vector<float> decoded(groups * outputs * columns), sums(groups * outputs, 0);
    ggml_type_to_float(GGML_TYPE_Q5_K)(packed.data(), decoded.data(), decoded.size());
    for (int i = 0; i < groups * outputs; ++i)
        for (int c = 0; c < columns; ++c) sums[i] += decoded[i*columns+c];
    for (int t = 0; t < tokens; ++t) for (int g = 0; g < groups; ++g)
        for (int c = 0; c < columns; ++c)
            reinterpret_cast<T *>(input.cpuData)[(t*groups+g)*columns+c] =
                T((t%3-1) * (127.f/128) / (1 << (g%3)));
    input.ToDevice(CUDA, {0}, true); weight.ToDevice(CUDA, {0}, true);
    weight.strides.clear(); // Direct GGUF imports carry ggml strides, not Data strides.
    auto &executor = *static_cast<Executor *>(GetExecutor());
    executor.RunOnDevice("cuda", "DeepSeekV4WoA", {{"input", &input}, {"weight", &weight}, {"output", &output}},
                         {}, {{"groups", groups}, {"oRank", outputs}});
    Cuda(cudaDeviceSynchronize()); output.ToDevice(CPU);
    Check(output.dims == std::vector<int>({1,tokens,groups*outputs}) && output.dataType == dtype,
          "GGUF grouped WoA shape/type");
    double error = 0, norm = 0;
    for (int t = 0; t < tokens; ++t) for (int g = 0; g < groups; ++g) for (int o = 0; o < outputs; ++o) {
        const float expected = (t%3-1) * (127.f/128) / (1 << (g%3)) * sums[g*outputs+o];
        const float actual = float(reinterpret_cast<T *>(output.cpuData)[(t*groups+g)*outputs+o]);
        Check(std::isfinite(actual), "GGUF grouped WoA nonfinite output");
        error += double(actual-expected)*(actual-expected); norm += double(expected)*expected;
    }
    Check(std::sqrt(error / std::max(norm,1e-30)) < .004, "GGUF grouped WoA differs from scalar Q5_K oracle");
}

template<typename T>
static void TestGroupedGemvExact(int groups, int columns, int outputs) {
    using namespace fastllm;
    const auto dtype = std::is_same<T,float>::value ? FLOAT32 :
        std::is_same<T,half>::value ? FLOAT16 : BFLOAT16;
    Data input(dtype, {1,1,groups,columns});
    Data weight(DATA_GGUF_FORMAT, GGML_TYPE_Q5_K, {groups,outputs,columns});
    Data output, reference(dtype, {1,1,groups*outputs}), bias;
    input.Allocate(false); weight.Allocate(false);
    const auto packed = Weights(GGML_TYPE_Q5_K, groups*outputs, columns);
    memcpy(weight.cpuData, packed.data(), packed.size());
    input.ToDevice(CUDA, {0}, false); weight.ToDevice(CUDA, {0}, true);
    reference.ToDevice(CUDA, {0}, false); reference.Allocate(false);
    const size_t groupBytes = weight.Count(0) / groups;
    weight.strides.clear();
    auto run = [&] { Check(FastllmCudaGGUFGroupedGemv(input, weight, groups, outputs, output),
                          "grouped GEMV rejected supported shape"); };
    run(); Cuda(cudaDeviceSynchronize());
    cudaGraph_t graph; cudaGraphExec_t executable;
    Check(FastllmCudaGraphMemoryPoolBegin(), "grouped graph pool begin");
    Check(FastllmCudaGraphBeginCapture(), "grouped graph begin");
    run();
    void *captured=nullptr;
    Check(FastllmCudaGraphEndCapture(&captured), "grouped graph end");
    graph=static_cast<cudaGraph_t>(captured);
    std::vector<void *> reserved;
    Check(FastllmCudaGraphMemoryPoolEnd(reserved), "grouped graph pool end");
    Cuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int pass=0; pass<3; ++pass) {
        std::vector<T> values(groups*columns), actual(groups*outputs), expected(groups*outputs);
        for (size_t i=0; i<values.size(); ++i) values[i]=T(std::sin(float(i*17+pass*13))*.25f);
        Cuda(cudaMemcpy(input.cudaData, values.data(), values.size()*sizeof(T), cudaMemcpyHostToDevice));
        if (pass) Cuda(cudaGraphLaunch(executable, cudaStreamPerThread)); else run();
        for (int g=0; g<groups; ++g) {
            Data x,y,w;
            x.FakeFrom(input, size_t(g)*columns*sizeof(T)); x.Resize({1,columns});
            y.FakeFrom(reference, size_t(g)*outputs*sizeof(T)); y.Resize({1,outputs});
            w.FakeFrom(weight, g*groupBytes); w.ggmlType=GGML_TYPE_Q5_K; w.dims={outputs,columns};
            if constexpr (std::is_same<T,float>::value)
                Check(FastllmCudaMatMulFloatGGUF(x,w,bias,y,1,columns,outputs), "ordinary GEMV");
            else if constexpr (std::is_same<T,half>::value)
                Check(FastllmCudaHalfMatMulGGUF(x,w,bias,y,1,columns,outputs), "ordinary GEMV");
            else Check(FastllmCudaBFloat16MatMulGGUF(x,w,bias,y,1,columns,outputs), "ordinary GEMV");
        }
        Cuda(cudaStreamSynchronize(cudaStreamPerThread));
        Cuda(cudaMemcpy(actual.data(), output.cudaData, actual.size()*sizeof(T), cudaMemcpyDeviceToHost));
        Cuda(cudaMemcpy(expected.data(), reference.cudaData, expected.size()*sizeof(T), cudaMemcpyDeviceToHost));
        Check(!memcmp(actual.data(), expected.data(), actual.size()*sizeof(T)),
              "grouped GEMV must be bitwise equal to independent GEMVs, including graph replay");
    }
    Cuda(cudaGraphExecDestroy(executable)); Cuda(cudaGraphDestroy(graph));
    FastllmCudaGraphMemoryPoolRelease(reserved);
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
        for (int groups : {1,3,8}) {
            TestGroupedGemvExact<float>(groups, 256, 129);
            TestGroupedGemvExact<half>(groups, 4096, 1024);
            TestGroupedGemvExact<__nv_bfloat16>(groups, 4096, 1024);
        }
        for (int tokens : {1, 33, 1025}) {
            TestGroupedWoA<float>(tokens);
            TestGroupedWoA<__half>(tokens);
            TestGroupedWoA<__nv_bfloat16>(tokens);
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
        for (auto type : {GGML_TYPE_Q4_K, GGML_TYPE_IQ4_XS, GGML_TYPE_Q2_K, GGML_TYPE_Q5_K}) {
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
