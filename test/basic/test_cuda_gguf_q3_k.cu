#include "fastllm-cuda.cuh"
#include "fastllm-gguf-mmq-common.cuh"
#include "vecdotq.cuh"
#include "fastllm-gguf-q3-k-gemv.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

static void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
static void Cuda(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

// Retain the previous four-warp reduction as a bitwise compatibility oracle.
template<typename T>
__global__ void LegacyQ3(const block_q3_K *weight, const block_q8_1 *input,
                         T *output, int columns) {
    const int lane = threadIdx.x, warp = threadIdx.y, tid = lane + 32*warp;
    float sum = 0;
    for (int block = tid/16; block < columns/QK_K; block += 8)
        sum += vec_dot_q3_K_q8_1(weight, input + 8*block,
                                blockIdx.x*(columns/QK_K) + block, tid%16);
    __shared__ float partial[3][32];
    if (warp) partial[warp-1][lane] = sum;
    __syncthreads();
    if (warp) return;
#pragma unroll
    for (int i = 0; i < 3; ++i) sum += partial[i][lane];
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, mask);
    if (lane == 0) output[blockIdx.x] = static_cast<T>(sum);
}

template<typename T>
static void Test(int rows, int columns, bool zero = false) {
    std::vector<block_q3_K> weights(size_t(rows)*columns/QK_K);
    std::vector<block_q8_1> inputs(columns/QK8_1);
    uint32_t random = 937;
    auto next = [&]() { random = random*1664525U + 1013904223U; return random >> 24; };
    for (auto &block : weights) {
        // Exercise the full quant, high-mask and six-bit scale code space.
        for (auto &v : block.qs) v = next();
        for (auto &v : block.hmask) v = next();
        for (auto &v : block.scales) v = next();
        block.d = __float2half_rn((int(next()%7)-3)*0.00137f);
    }
    for (auto &block : inputs) {
        const float d = zero ? 0 : (int(next()%7)-3)*0.0137f;
        block.ds = __floats2half2_rn(d, 0);
        for (auto &v : block.qs) v = zero ? 0 : int(next())-128;
    }
    block_q3_K *w = nullptr;
    block_q8_1 *x = nullptr;
    T *output = nullptr, *reference = nullptr;
    Cuda(cudaMalloc(&w, weights.size()*sizeof(block_q3_K)));
    Cuda(cudaMalloc(&x, inputs.size()*sizeof(block_q8_1)));
    Cuda(cudaMalloc(&output, (rows+8)*sizeof(T)));
    Cuda(cudaMalloc(&reference, rows*sizeof(T)));
    Cuda(cudaMemcpy(w, weights.data(), weights.size()*sizeof(block_q3_K), cudaMemcpyHostToDevice));
    Cuda(cudaMemcpy(x, inputs.data(), inputs.size()*sizeof(block_q8_1), cudaMemcpyHostToDevice));
    Cuda(cudaMemset(output, 0x5a, (rows+8)*sizeof(T)));
    Check(fastllm_gguf_q3_k::Supports(x, columns, rows), "Q3_K valid shape rejected");
    Check(!fastllm_gguf_q3_k::Supports(nullptr, columns, rows) &&
          !fastllm_gguf_q3_k::Supports(reinterpret_cast<char *>(x)+4, columns, rows) &&
          !fastllm_gguf_q3_k::Supports(x, columns, 127) &&
          !fastllm_gguf_q3_k::Supports(x, 255, rows) &&
          !fastllm_gguf_q3_k::Supports(x, 18688, rows), "Q3_K admission guard failed");
    LegacyQ3<<<rows, dim3(32,4), 0, cudaStreamPerThread>>>(w, x, reference, columns);
    fastllm_gguf_q3_k::Launch(w, x, output, columns, rows, cudaStreamPerThread);
    Cuda(cudaGetLastError());
    Cuda(cudaStreamSynchronize(cudaStreamPerThread));
    if ((rows == 129 || rows == 1025) && columns == 2560) {
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        fastllm_gguf_q3_k::Launch(w, x, output, columns, rows, cudaStreamPerThread);
        Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
        Cuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        Cuda(cudaMemset(output, 0x5a, (rows+8)*sizeof(T)));
        Cuda(cudaGraphLaunch(executable, cudaStreamPerThread));
        Cuda(cudaStreamSynchronize(cudaStreamPerThread));
        Cuda(cudaGraphExecDestroy(executable));
        Cuda(cudaGraphDestroy(graph));
    }
    std::vector<T> actual(rows+8), expected(rows);
    Cuda(cudaMemcpy(actual.data(), output, actual.size()*sizeof(T), cudaMemcpyDeviceToHost));
    Cuda(cudaMemcpy(expected.data(), reference, expected.size()*sizeof(T), cudaMemcpyDeviceToHost));
    Check(std::memcmp(actual.data(), expected.data(), rows*sizeof(T)) == 0,
          "Q3_K result differs from the original four-warp reduction");
    const auto *tail = reinterpret_cast<const unsigned char *>(actual.data()+rows);
    for (size_t i = 0; i < 8*sizeof(T); ++i) Check(tail[i] == 0x5a, "Q3_K output tail overwritten");
    // Independent CPU dequantization, followed by a double-precision dot
    // against the stored Q8 activation (no reuse of the GPU scale decoder).
    std::vector<float> decoded(size_t(rows)*columns);
    ggml_type_to_float(GGML_TYPE_Q3_K)(weights.data(), decoded.data(), decoded.size());
    for (int row = 0; row < rows; ++row) {
        double sum = 0, magnitude = 0;
        for (int col = 0; col < columns; ++col) {
            const auto &block = inputs[col/QK8_1];
            const double term = double(decoded[size_t(row)*columns+col])*
                (__low2float(block.ds)*block.qs[col%QK8_1]);
            sum += term; magnitude += std::fabs(term);
        }
        const float rounded = float(static_cast<T>(float(sum)));
        const float epsilon = std::is_same<T,float>::value ? 0 :
            std::is_same<T,half>::value ? 0.001f : 0.008f;
        Check(std::isfinite(float(actual[row])) && std::fabs(float(actual[row])-rounded) <=
              2e-6*std::max(1.0,magnitude) + epsilon*std::max(1.0f,std::fabs(rounded)),
              "Q3_K result disagrees with the CPU dequantized dot");
    }
    Cuda(cudaFree(w)); Cuda(cudaFree(x)); Cuda(cudaFree(output)); Cuda(cudaFree(reference));
}

int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
        Cuda(cudaSetDevice(0));
        for (int columns : {256, 512, 768, 2304, 2560, 4352, 18432}) {
            for (int rows : {128,129}) {
                Test<float>(rows, columns);
                Test<half>(rows, columns);
                Test<__nv_bfloat16>(rows, columns);
            }
        }
        for (int columns : {2560, 8192}) {
            Test<float>(1025, columns);
            Test<half>(1025, columns);
            Test<__nv_bfloat16>(1025, columns);
        }
        Test<float>(1025, 2560, true);
        Test<float>(129, 2560, true);
        Test<half>(129, 2560, true);
        Test<__nv_bfloat16>(129, 2560, true);
        std::cout << "PASS: Q3_K CUDA GEMV CPU reference, bitwise reduction and graph replay\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
