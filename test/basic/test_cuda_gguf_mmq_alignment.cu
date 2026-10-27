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
static void TestGroupedGemvExact(int groups, int columns, int outputs, int tokens) {
    using namespace fastllm;
    const auto dtype = std::is_same<T,float>::value ? FLOAT32 :
        std::is_same<T,half>::value ? FLOAT16 : BFLOAT16;
    const int batch = tokens%2 == 0 ? 2 : 1;
    Data input(dtype, {batch,tokens/batch,groups,columns});
    Data weight(DATA_GGUF_FORMAT, GGML_TYPE_Q5_K, {groups,outputs,columns});
    Data output, reference(dtype, {tokens+1,outputs}), groupedInput(dtype,{tokens,columns}), bias;
    input.Allocate(false); weight.Allocate(false);
    const auto packed = Weights(GGML_TYPE_Q5_K, groups*outputs, columns);
    memcpy(weight.cpuData, packed.data(), packed.size());
    input.ToDevice(CUDA, {0}, false); weight.ToDevice(CUDA, {0}, true);
    reference.ToDevice(CUDA, {0}, false); reference.Allocate(false);
    groupedInput.ToDevice(CUDA, {0}, false); groupedInput.Allocate(false);
    Cuda(cudaMemset(input.cudaData,0,size_t(tokens)*groups*columns*sizeof(T)));
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
        std::vector<T> values(tokens*groups*columns), actual(tokens*groups*outputs), expected(actual.size());
        for (size_t i=0; i<values.size(); ++i) values[i]=T(std::sin(float(i*17+pass*13))*.25f);
        Cuda(cudaMemcpy(input.cudaData, values.data(), values.size()*sizeof(T), cudaMemcpyHostToDevice));
        if (pass) Cuda(cudaGraphLaunch(executable, cudaStreamPerThread)); else run();
        for (int g=0; g<groups; ++g) {
            std::vector<T> groupValues(size_t(tokens)*columns), groupOutput(size_t(tokens)*outputs);
            for (int r=0; r<tokens; ++r)
                memcpy(groupValues.data()+size_t(r)*columns,values.data()+size_t(r*groups+g)*columns,columns*sizeof(T));
            Cuda(cudaMemcpy(groupedInput.cudaData,groupValues.data(),groupValues.size()*sizeof(T),cudaMemcpyHostToDevice));
            Data w;
            w.FakeFrom(weight, g*groupBytes); w.ggmlType=GGML_TYPE_Q5_K; w.dims={outputs,columns};
            Cuda(cudaMemset(reference.cudaData,0x5a,reference.GetBytes()));
            if constexpr (std::is_same<T,float>::value)
                Check(FastllmCudaMatMulFloatGGUF(groupedInput,w,bias,reference,tokens,columns,outputs), "ordinary GEMV");
            else if constexpr (std::is_same<T,half>::value)
                Check(FastllmCudaHalfMatMulGGUF(groupedInput,w,bias,reference,tokens,columns,outputs), "ordinary GEMV");
            else Check(FastllmCudaBFloat16MatMulGGUF(groupedInput,w,bias,reference,tokens,columns,outputs), "ordinary GEMV");
            Cuda(cudaStreamSynchronize(cudaStreamPerThread));
            Cuda(cudaMemcpy(groupOutput.data(),reference.cudaData,groupOutput.size()*sizeof(T),cudaMemcpyDeviceToHost));
            std::vector<unsigned char> guard(size_t(outputs)*sizeof(T));
            Cuda(cudaMemcpy(guard.data(),static_cast<char *>(reference.cudaData)+groupOutput.size()*sizeof(T),
                            guard.size(),cudaMemcpyDeviceToHost));
            for (auto value : guard) Check(value==0x5a, "ordinary GEMV overwrote output guard");
            for (int r=0; r<tokens; ++r)
                memcpy(expected.data()+size_t(r*groups+g)*outputs,groupOutput.data()+size_t(r)*outputs,outputs*sizeof(T));
        }
        Cuda(cudaStreamSynchronize(cudaStreamPerThread));
        Cuda(cudaMemcpy(actual.data(), output.cudaData, actual.size()*sizeof(T), cudaMemcpyDeviceToHost));
        Check(!memcmp(actual.data(), expected.data(), actual.size()*sizeof(T)),
              "grouped GEMV must be bitwise equal to independent GEMVs, including graph replay");
    }
    Cuda(cudaGraphExecDestroy(executable)); Cuda(cudaGraphDestroy(graph));
    FastllmCudaGraphMemoryPoolRelease(reserved);
}

template<typename T, ggml_type Type = GGML_TYPE_Q8_0>
static void TestQuantizedBatchExact(int rows,int columns,int outputs) {
    using namespace fastllm;
    constexpr bool bf16=std::is_same<T,__nv_bfloat16>::value;
    const auto dtype=std::is_same<T,float>::value?FLOAT32:bf16?BFLOAT16:FLOAT16;
    Data x(dtype,{rows,columns}),w(DATA_GGUF_FORMAT,Type,{outputs,columns}),y(dtype,{rows+1,outputs}),bias;
    x.Allocate();w.Allocate();
    if constexpr (Type == GGML_TYPE_Q8_0) {
        auto *blocks=reinterpret_cast<block_q8_0 *>(w.cpuData);
        for(size_t i=0;i<w.GetBytes()/sizeof(block_q8_0);++i) {
            blocks[i].d=__float2half_rn(std::ldexp(.007f,int(i%9)-4));
            for(int j=0;j<32;++j)blocks[i].qs[j]=int((i*17+j*23)%256)-128;
        }
    } else {
        const auto packed=Weights(Type,outputs,columns);
        memcpy(w.cpuData,packed.data(),packed.size());
    }
    for(int i=0;i<rows*columns;++i)reinterpret_cast<T *>(x.cpuData)[i]=T(std::sin(i*.37f)*std::ldexp(1.f,(i/32)%7-3));
    x.ToDevice(CUDA,{0},true);w.ToDevice(CUDA,{0},true);y.ToDevice(CUDA,{0},false);y.Allocate();
    Cuda(cudaMemset(y.cudaData,0x5a,y.GetBytes()));
    auto run=[&](Data &a,Data &b,int n) {
        if constexpr(std::is_same<T,float>::value)Check(FastllmCudaMatMulFloatGGUF(a,w,bias,b,n,columns,outputs),"GGUF FP32 failed");
        else if constexpr(bf16)Check(FastllmCudaBFloat16MatMulGGUF(a,w,bias,b,n,columns,outputs),"GGUF BF16 failed");
        else Check(FastllmCudaHalfMatMulGGUF(a,w,bias,b,n,columns,outputs),"GGUF half failed");
    };
    run(x,y,rows);
    std::vector<unsigned char> actual(y.GetBytes()),expected(size_t(rows)*outputs*sizeof(T));
    Cuda(cudaMemcpy(actual.data(),y.cudaData,actual.size(),cudaMemcpyDeviceToHost));
    for(int row=0;row<rows;++row) {
        Data xi,yi(dtype,{1,outputs});xi.FakeFrom(x,size_t(row)*columns*sizeof(T));xi.Resize({1,columns});
        yi.ToDevice(CUDA,{0},false);yi.Allocate();run(xi,yi,1);
        Cuda(cudaMemcpy(expected.data()+size_t(row)*outputs*sizeof(T),yi.cudaData,yi.GetBytes(),cudaMemcpyDeviceToHost));
    }
    if (memcmp(actual.data(),expected.data(),expected.size())) {
        fprintf(stderr,"GGUF exact mismatch type=%d dtype=%d rows=%d K=%d N=%d threshold=%d\n",int(Type),int(dtype),rows,columns,outputs,FastllmCudaGetLinearExactBatchThreshold());
        throw std::runtime_error("GGUF batch differs from single-row arithmetic");
    }
    for(size_t i=expected.size();i<actual.size();++i)Check(actual[i]==0x5a,"GGUF batch overwrote output guard");
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
        const int savedThreshold = fastllm::FastllmCudaGetLinearExactBatchThreshold();
        // Explicit exact-batch mode retains the single-token reduction tree.
        fastllm::FastllmCudaSetLinearExactBatchThreshold(9);
        for (int rows : {1,2,3,4,5,6,7,8}) for (int columns : {256,768,5120}) for (int outputs : {129,512,1280}) {
            TestQuantizedBatchExact<half>(rows,columns,outputs);
            TestQuantizedBatchExact<__nv_bfloat16>(rows,columns,outputs);
        }
        for (int rows : {1,2,3,4,5,6,7,8})
            for (int columns : {1024,1280,1536,1792,2048,2304,2560,2816})
                for (int outputs : {129,2049}) {
                    TestQuantizedBatchExact<float,GGML_TYPE_Q5_K>(rows,columns,outputs);
                    TestQuantizedBatchExact<half,GGML_TYPE_Q5_K>(rows,columns,outputs);
                    TestQuantizedBatchExact<__nv_bfloat16,GGML_TYPE_Q5_K>(rows,columns,outputs);
                }
        // The independent grouped GEMV keeps the legacy reduction tree.
        // Check ordinary B5-B8 as well as exact-batch mode and its boundary,
        // including odd output tails and replay with different activations.
        for (int threshold : {0,6,9}) {
            fastllm::FastllmCudaSetLinearExactBatchThreshold(threshold);
            for (int tokens : {1,2,3,4,5,6,7,8}) for (int columns : {1280,2304}) {
                TestGroupedGemvExact<float>(3, columns, 2049, tokens);
                TestGroupedGemvExact<half>(3, columns, 2049, tokens);
                TestGroupedGemvExact<__nv_bfloat16>(3, columns, 2049, tokens);
            }
        }
        fastllm::FastllmCudaSetLinearExactBatchThreshold(savedThreshold);
        for (int tokens : {1,2,3,4,5,6,7,8}) for (int groups : {1,3,8}) {
            TestGroupedGemvExact<float>(groups, 256, 129, tokens);
            TestGroupedGemvExact<half>(groups, 768, 257, tokens);
            TestGroupedGemvExact<__nv_bfloat16>(groups, 4096, 1024, tokens);
        }
        for (int tokens : {1, 5, 8, 9, 17, 33, 1025}) {
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
