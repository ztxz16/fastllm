#include "fastllm-cuda.cuh"
#include "fastllm-gguf-mmq-common.cuh"
#include "vecdotq.cuh"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

using namespace fastllm;
static void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
static void Cuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

// Independent activation quantization and the unchanged MMQ-header Q6 dot
// retain the old saturating-byte subtraction and four-warp summation order.
template<typename T>
__global__ void QuantizeReference(const T *input, block_q8_1 *output, int columns) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= columns) return;
    const float value = float(input[i]);
    float maximum = fabsf(value), sum = value;
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, mask));
        sum += __shfl_xor_sync(0xffffffffu, sum, mask);
    }
    const float scale = maximum / 127;
    output[i / 32].qs[i % 32] = maximum == 0 ? 0 : roundf(value / scale);
    if (i % 32 == 0) output[i / 32].ds = __floats2half2_rn(scale, sum);
}

template<typename T>
__global__ void LegacyQ6(const block_q6_K *weight, const block_q8_1 *input,
                         T *output, int columns) {
    const int lane = threadIdx.x, warp = threadIdx.y;
    float sum = 0;
    for (int block = warp; block < columns / QK_K; block += 4)
        sum += vec_dot_q6_K_q8_1(weight, input + 8 * block,
                                blockIdx.x * (columns / QK_K) + block, lane);
    __shared__ float partial[3][32];
    if (warp) partial[warp - 1][lane] = sum;
    __syncthreads();
    if (warp) return;
#pragma unroll
    for (int i = 0; i < 3; ++i) sum += partial[i][lane];
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, mask);
    if (lane == 0) output[blockIdx.x] = static_cast<T>(sum);
}

template<typename T>
static void Test(int device, int columns, int rows, bool zero = false) {
    constexpr DataType dtype = std::is_same<T, float>::value ? FLOAT32 :
        std::is_same<T, half>::value ? FLOAT16 : BFLOAT16;
    auto allocate = [&](Data &data) {
        data.dataDevice = DataDevice::CUDA;
        data.dataDeviceIds = {device};
        data.Allocate();
    };
    Data w(DATA_GGUF_FORMAT, int(GGML_TYPE_Q6_K), {rows, columns});
    Data x(dtype, {1, columns}), y(dtype, {1, rows + 8}), reference(dtype, {1, rows});
    Data q(FLOAT32, {columns / 32 * int(sizeof(block_q8_1) / sizeof(float))});
    Data bias(FLOAT32);
    for (Data *data : {&w, &x, &y, &reference, &q}) allocate(*data);
    std::vector<block_q6_K> weights(size_t(rows) * columns / QK_K);
    uint32_t random = 937;
    auto next = [&]() { random = random * 1664525u + 1013904223u; return random >> 24; };
    for (auto &block : weights) {
        for (auto &v : block.ql) v = next();
        for (auto &v : block.qh) v = next();
        for (auto &v : block.scales) v = int(next()) - 128;
        block.d = __float2half_rn((int(next() % 7) - 3) * .00137f);
    }
    std::vector<T> values(columns);
    for (auto &v : values) v = static_cast<T>(zero ? 0.f : (int(next()) - 128) / 256.f);
    Cuda(cudaMemcpy(w.cudaData, weights.data(), weights.size() * sizeof(block_q6_K), cudaMemcpyHostToDevice));
    Cuda(cudaMemcpy(x.cudaData, values.data(), columns * sizeof(T), cudaMemcpyHostToDevice));
    Cuda(cudaMemset(y.cudaData, 0x5a, (rows + 8) * sizeof(T)));
    QuantizeReference<<<columns / 256, 256, 0, cudaStreamPerThread>>>(
        static_cast<T *>(x.cudaData), static_cast<block_q8_1 *>(q.cudaData), columns);
    LegacyQ6<<<rows, dim3(32, 4), 0, cudaStreamPerThread>>>(
        static_cast<block_q6_K *>(w.cudaData), static_cast<block_q8_1 *>(q.cudaData),
        static_cast<T *>(reference.cudaData), columns);
    auto run = [&]() {
        if constexpr (std::is_same<T, float>::value)
            Check(FastllmCudaMatMulFloatGGUF(x, w, bias, y, 1, columns, rows), "FP32 entry rejected");
        else if constexpr (std::is_same<T, half>::value)
            Check(FastllmCudaHalfMatMulGGUF(x, w, bias, y, 1, columns, rows), "FP16 entry rejected");
        else
            Check(FastllmCudaBFloat16MatMulGGUF(x, w, bias, y, 1, columns, rows), "BF16 entry rejected");
    };
    run();
    Cuda(cudaStreamSynchronize(cudaStreamPerThread));
    std::vector<T> actual(rows + 8), expected(rows);
    Cuda(cudaMemcpy(expected.data(), reference.cudaData, rows * sizeof(T), cudaMemcpyDeviceToHost));
    auto compare = [&]() {
        Cuda(cudaMemcpy(actual.data(), y.cudaData, actual.size() * sizeof(T), cudaMemcpyDeviceToHost));
        Check(!std::memcmp(actual.data(), expected.data(), rows * sizeof(T)), "Q6_K changed legacy output bits");
        const auto *tail = reinterpret_cast<const unsigned char *>(actual.data() + rows);
        for (size_t i = 0; i < 8 * sizeof(T); ++i) Check(tail[i] == 0x5a, "Q6_K overwrote output tail");
    };
    compare();
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    run();
    Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Cuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    Cuda(cudaGraphLaunch(executable, cudaStreamPerThread));
    Cuda(cudaStreamSynchronize(cudaStreamPerThread));
    compare();
    Cuda(cudaGraphExecDestroy(executable));
    Cuda(cudaGraphDestroy(graph));

    // CPU GGUF decoding + FP64 dot verifies the represented Q8 activation
    // independently of both GPU implementations, including negative scales.
    std::vector<block_q8_1> quantized(columns / 32);
    Cuda(cudaMemcpy(quantized.data(), q.cudaData, quantized.size() * sizeof(block_q8_1), cudaMemcpyDeviceToHost));
    std::vector<float> decoded(columns);
    for (int row : {0, rows / 2, rows - 1}) {
        ggml_type_to_float(GGML_TYPE_Q6_K)(weights.data() + size_t(row) * columns / QK_K, decoded.data(), columns);
        double sum = 0, magnitude = 0;
        for (int i = 0; i < columns; ++i) {
            const auto &block = quantized[i / 32];
            const double term = double(decoded[i]) * (__low2float(block.ds) * block.qs[i % 32]);
            sum += term;
            magnitude += std::fabs(term);
        }
        const double rounded = float(static_cast<T>(float(sum)));
        constexpr double epsilon = dtype == FLOAT32 ? 0 : dtype == FLOAT16 ? .001 : .008;
        Check(std::isfinite(float(actual[row])) && std::fabs(float(actual[row]) - rounded) <=
                  2e-6 * std::max(1.0, magnitude) + epsilon * std::max(1.0, std::fabs(rounded)),
              "Q6_K disagrees with CPU dequantized dot");
    }
}

int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
        for (int device = 0; device < std::min(devices, 2); ++device) {
            FastllmCudaSetDevice(device);
            for (int columns : {256, 768, 1536, 2560, 3072, 6144, 8192})
                for (int rows : {13, 257}) {
                    Test<float>(device, columns, rows);
                    Test<half>(device, columns, rows);
                    Test<__nv_bfloat16>(device, columns, rows);
                }
            for (int columns : {1536, 3072, 6144}) Test<half>(device, columns, 2560);
            Test<half>(device, 6144, 2560, true);
        }
        std::cout << "PASS: Q6_K legacy bits, CPU reference, graph replay and TP shard shapes\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
