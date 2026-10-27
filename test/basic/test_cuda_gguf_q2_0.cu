#include <cuda_fp16.h>
#include <cuda_bf16.h>
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "gguf.h"
#include "fastllm-gguf-dequant.cuh"
#include "fastllm-gguf-gemv.cuh"
#include "moe/fastllm-moe-gguf-q8.cuh"
#include "fastllm-cuda.cuh"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

static void Check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
static void Cuda(cudaError_t code) { if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code)); }
template<typename T> static T Cast(float v) { return static_cast<T>(v); }
template<> half Cast(float v) { return __float2half_rn(v); }
template<> __nv_bfloat16 Cast(float v) { return __float2bfloat16_rn(v); }

struct Fixture {
    std::vector<block_q2_0> weight;
    std::vector<float> reference;
    Fixture(int rows, int columns) : weight(rows * columns / QK2_0), reference(rows * columns) {
        uint32_t random = 937;
        const float scales[] = {-0.5f, 0.0f, 0.03125f, 0.25f, 1.0f};
        for (size_t b = 0; b < weight.size(); ++b) {
            const float d = scales[b % 5];
            weight[b].d = __float2half_rn(d);
            for (int byte = 0; byte < QK2_0 / 4; ++byte) {
                uint8_t packed = 0;
                for (int p = 0; p < 4; ++p) {
                    random = random * 1664525U + 1013904223U;
                    const int code = (random >> 24) & 3;
                    packed |= code << (2 * p);
                    reference[b * QK2_0 + 4 * byte + p] = d * (code - 1);
                }
                weight[b].qs[byte] = packed;
            }
        }
    }
};

static void TestCpu(int columns) {
    Fixture f(1, columns);
    Check(ggml_blck_size(GGML_TYPE_Q2_0) == 64 && ggml_row_size(GGML_TYPE_Q2_0, columns) == f.weight.size() * 18,
          "Q2_0 GGUF type metadata");
    std::vector<float> decoded(columns);
    ggml_type_to_float(GGML_TYPE_Q2_0)(f.weight.data(), decoded.data(), columns);
    Check(decoded == f.reference, "Q2_0 CPU reference decoding");
    std::vector<block_q8_0> activation(columns / QK8_0);
    const float scales[] = {0.03125f, -0.125f, 0.5f};
    double expected = 0;
    for (size_t b = 0; b < activation.size(); ++b) {
        const float d = scales[b % 3];
        activation[b].d = __float2half_rn(d);
        for (int j = 0; j < QK8_0; ++j) {
            const int q = int((b * 97 + j * 47) % 256) - 128;
            activation[b].qs[j] = q;
            expected += double(f.reference[b * QK8_0 + j]) * (d * q);
        }
    }
    float result = 0;
    ggml_vec_dot_q2_0_q8_0(columns, &result, 0, f.weight.data(), 0, activation.data(), 0, 1);
    Check(result == float(expected), "Q2_0 AVX2/scalar dot disagrees with decoded Q8 activation reference");
}

template<typename T> static void TestGpu(int rows, int columns, to_t_cuda_t<T> decode) {
    Fixture f(rows, columns);
    Check(decode != nullptr, "Q2_0 CUDA decoder missing");
    void *weight = nullptr; T *output = nullptr, *input = nullptr;
    const size_t bytes = f.weight.size() * sizeof(block_q2_0);
    Cuda(cudaMalloc(&weight, bytes));
    Cuda(cudaMalloc(reinterpret_cast<void **>(&output), f.reference.size() * sizeof(T)));
    Cuda(cudaMalloc(reinterpret_cast<void **>(&input), columns * sizeof(T)));
    Cuda(cudaMemcpy(weight, f.weight.data(), bytes, cudaMemcpyHostToDevice));
    decode(weight, output, rows, columns, cudaStreamPerThread);
    Cuda(cudaGetLastError()); Cuda(cudaDeviceSynchronize());
    std::vector<T> actual(f.reference.size());
    Cuda(cudaMemcpy(actual.data(), output, actual.size() * sizeof(T), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < actual.size(); ++i) {
        Check(float(actual[i]) == float(Cast<T>(f.reference[i])), "Q2_0 CUDA dequantization mismatch");
    }
    std::vector<T> values(columns);
    for (int c = 0; c < columns; ++c) values[c] = Cast<T>(float((c * 19) % 33 - 16) / 32);
    Cuda(cudaMemcpy(input, values.data(), columns * sizeof(T), cudaMemcpyHostToDevice));
    Check(FastllmGgufDirectGemv(input, weight, output, GGML_TYPE_Q2_0, columns, rows, cudaStreamPerThread),
          "Q2_0 direct GEMV rejected valid shape");
    Cuda(cudaGetLastError()); Cuda(cudaDeviceSynchronize());
    Cuda(cudaMemcpy(actual.data(), output, rows * sizeof(T), cudaMemcpyDeviceToHost));
    for (int r = 0; r < rows; ++r) {
        double expected = 0;
        for (int c = 0; c < columns; ++c) expected += double(float(Cast<T>(f.reference[r * columns + c]))) * float(values[c]);
        Check(float(actual[r]) == float(Cast<T>(float(expected))), "Q2_0 direct GEMV mismatch");
    }
    Cuda(cudaFree(weight)); Cuda(cudaFree(output)); Cuda(cudaFree(input));
}

template<int Lanes>
__global__ void Q2DotRows(const block_q2_0 *weight, const block_q8_1 *input,
                           float *output, int rows, int columns) {
    const int row = blockIdx.x*(256/Lanes) + threadIdx.x/Lanes;
    if (row >= rows) return;
    const float value = gguf_cache_q8::RowDot<GGML_TYPE_Q2_0, Lanes>(
        weight + size_t(row)*columns/QK2_0, input, columns, nullptr);
    if (threadIdx.x%Lanes == 0) output[row] = value;
}

static void TestGroupedQ8Dot(int rows, int columns) {
    Fixture f(rows, columns);
    std::vector<block_q8_1> input(columns/32);
    for (size_t b=0; b<input.size(); ++b) {
        input[b].ds = __floats2half2_rn(float(int(b%7)-3)*0.0137f, 0);
        for (int i=0; i<32; ++i) input[b].qs[i]=int((b*97+i*47)%256)-128;
    }
    void *weights=nullptr;block_q8_1 *activation=nullptr;float *original=nullptr,*grouped=nullptr;
    Cuda(cudaMalloc(&weights,f.weight.size()*sizeof(block_q2_0)));
    Cuda(cudaMalloc(&activation,input.size()*sizeof(block_q8_1)));
    Cuda(cudaMalloc(&original,rows*sizeof(float)));Cuda(cudaMalloc(&grouped,rows*sizeof(float)));
    Cuda(cudaMemcpy(weights,f.weight.data(),f.weight.size()*sizeof(block_q2_0),cudaMemcpyHostToDevice));
    Cuda(cudaMemcpy(activation,input.data(),input.size()*sizeof(block_q8_1),cudaMemcpyHostToDevice));
    Q2DotRows<32><<<(rows+7)/8,256,0,cudaStreamPerThread>>>((block_q2_0*)weights,activation,original,rows,columns);
    Q2DotRows<8><<<(rows+31)/32,256,0,cudaStreamPerThread>>>((block_q2_0*)weights,activation,grouped,rows,columns);
    Cuda(cudaGetLastError());Cuda(cudaStreamSynchronize(cudaStreamPerThread));
    std::vector<float> oldResult(rows),newResult(rows);
    Cuda(cudaMemcpy(oldResult.data(),original,rows*sizeof(float),cudaMemcpyDeviceToHost));
    Cuda(cudaMemcpy(newResult.data(),grouped,rows*sizeof(float),cudaMemcpyDeviceToHost));
    Check(std::memcmp(oldResult.data(),newResult.data(),rows*sizeof(float))==0,
          "Q2_0 grouped Q8 dot changed full-warp reduction arithmetic");
    Cuda(cudaFree(weights));Cuda(cudaFree(activation));Cuda(cudaFree(original));Cuda(cudaFree(grouped));
}

template<typename T> static void TestMmvq(int batch, int columns) {
    const int rows = 13; // Exercise a partial eight-row CUDA block.
    Fixture f(rows, columns);
    std::vector<T> values(batch*columns), actual(batch*rows);
    std::vector<float> quantized(batch*columns);
    for (int i = 0; i < batch*columns; ++i)
        values[i] = Cast<T>(i%columns < 32 ? 0 :
            .47f*std::sin(i*.713f)+.031f*std::cos(i*1.37f));
    for (int b = 0; b < batch*columns; b += 32) {
        float maximum = 0;
        for (int i = 0; i < 32; ++i) maximum = std::max(maximum, std::fabs(float(values[b+i])));
        const float scale = maximum/127.0f;
        const float stored = __half2float(__float2half_rn(scale));
        for (int i = 0; i < 32; ++i)
            quantized[b+i] = maximum == 0 ? 0 : std::round(float(values[b+i])/scale)*stored;
    }
    void *weight = nullptr; T *input = nullptr, *output = nullptr;
    Cuda(cudaMalloc(&weight, f.weight.size()*sizeof(block_q2_0)));
    Cuda(cudaMalloc(reinterpret_cast<void **>(&input), values.size()*sizeof(T)));
    Cuda(cudaMalloc(reinterpret_cast<void **>(&output), actual.size()*sizeof(T)));
    Cuda(cudaMemcpy(weight, f.weight.data(), f.weight.size()*sizeof(block_q2_0), cudaMemcpyHostToDevice));
    Cuda(cudaMemcpy(input, values.data(), values.size()*sizeof(T), cudaMemcpyHostToDevice));
    auto launch = [&](int n, int m) {
        if constexpr (std::is_same<T, float>::value)
            return FastllmCudaFloatMatMulGGUFMMVQ(input,weight,output,GGML_TYPE_Q2_0,n,m,rows,cudaStreamPerThread);
        else if constexpr (std::is_same<T, half>::value)
            return FastllmCudaHalfMatMulGGUFMMVQ(input,weight,output,GGML_TYPE_Q2_0,n,m,rows,cudaStreamPerThread);
        else
            return FastllmCudaBFloat16MatMulGGUFMMVQ(input,weight,output,GGML_TYPE_Q2_0,n,m,rows,cudaStreamPerThread);
    };
    Check(!launch(9, columns) && !launch(batch, 96) && !launch(batch, 32768),
          "Q2 MMVQ admitted unsupported shape");
    Check(launch(batch, columns), "Q2 MMVQ rejected supported shape");
    Cuda(cudaDeviceSynchronize());
    if (batch == 3 && columns == 320) {
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        Check(launch(batch, columns), "Q2 MMVQ graph capture rejected");
        Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
        Cuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        Cuda(cudaMemset(output, 0xff, actual.size()*sizeof(T)));
        Cuda(cudaGraphLaunch(executable, cudaStreamPerThread));
        Cuda(cudaDeviceSynchronize());
        Cuda(cudaGraphExecDestroy(executable));
        Cuda(cudaGraphDestroy(graph));
    }
    Cuda(cudaMemcpy(actual.data(), output, actual.size()*sizeof(T), cudaMemcpyDeviceToHost));
    for (int b = 0; b < batch; ++b) for (int r = 0; r < rows; ++r) {
        double reference = 0, magnitude = 0;
        for (int c = 0; c < columns; ++c) {
            const double term = double(f.reference[r*columns+c])*quantized[b*columns+c];
            reference += term; magnitude += std::fabs(term);
        }
        const float expected = float(Cast<T>(float(reference)));
        const float rounding = std::is_same<T, float>::value ? 0.0f :
            std::is_same<T, half>::value ? .001f : .008f;
        Check(std::isfinite(float(actual[b*rows+r])) &&
              std::fabs(float(actual[b*rows+r])-expected) <=
                  1e-5*std::max(1.0, magnitude)+rounding*std::max(1.0f, std::fabs(expected)),
              "Q2 MMVQ disagrees with independent CPU Q8 reference");
    }
    Cuda(cudaFree(weight)); Cuda(cudaFree(input)); Cuda(cudaFree(output));
}

static void TestMmvqAdmission() {
    // Invalid calls must reject before reading any device buffer. IQ formats
    // require 256-column weight blocks even though Q8 input uses blocks of 32.
    for (auto type : {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
                      GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M}) {
        for (int columns : {32, 64, 192, 288}) {
            Check(!FastllmCudaFloatMatMulGGUFMMVQ(nullptr, nullptr, nullptr,
                      type, 1, columns, 128, cudaStreamPerThread) &&
                  !FastllmCudaHalfMatMulGGUFMMVQ(nullptr, nullptr, nullptr,
                      type, 1, columns, 128, cudaStreamPerThread) &&
                  !FastllmCudaBFloat16MatMulGGUFMMVQ(nullptr, nullptr, nullptr,
                      type, 1, columns, 128, cudaStreamPerThread),
                  "MMVQ accepted a partial weight block");
            Check(!FastllmCudaHalfGgufGateUpSiluMulMMVQ(nullptr, nullptr, nullptr,
                      nullptr, type, 1, columns, 128, cudaStreamPerThread),
                  "Fused gate/up accepted a partial or unsupported weight block");
        }
    }
    for (auto type : {GGML_TYPE_Q2_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
                      GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M}) {
        Check(!FastllmCudaHalfGgufGateUpSiluMulMMVQ(nullptr, nullptr, nullptr,
                  nullptr, type, 1, 256, 128, cudaStreamPerThread),
              "Fused gate/up accepted a format without an implementation");
    }
    Cuda(cudaGetLastError());
}

int main() {
    try {
        for (int columns : {64, 128, 192, 640, 2560}) TestCpu(columns);
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
        Cuda(cudaSetDevice(0));
        TestMmvqAdmission();
        for (int columns : {64,128,192,256,320,448,512})
            for (int rows : {7,33}) TestGroupedQ8Dot(rows,columns);
        for (int columns : {64, 128, 192, 640, 2560}) {
            TestGpu<float>(7, columns, ggml_get_to_fp32_cuda(GGML_TYPE_Q2_0));
            TestGpu<half>(7, columns, ggml_get_to_fp16_cuda(GGML_TYPE_Q2_0));
            TestGpu<__nv_bfloat16>(7, columns, ggml_get_to_bf16_cuda(GGML_TYPE_Q2_0));
        }
        for (int columns : {64, 320, 2560}) for (int batch : {1, 3, 8}) {
            TestMmvq<float>(batch, columns);
            TestMmvq<half>(batch, columns);
            TestMmvq<__nv_bfloat16>(batch, columns);
        }
        std::cout << "PASS: Q2_0 CPU decoding/dot and CUDA FP32/FP16/BF16 dequantization/GEMV/MMVQ\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
