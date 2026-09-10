#include "../../src/devices/cuda/linear/fastllm-linear-fp8-quant.cuh"

#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace fastllm_cuda_cutlass_fp8_sm89;

namespace {
void Check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

struct Buffer {
    void *data = nullptr;
    explicit Buffer(size_t bytes) { Check(cudaMalloc(&data, bytes)); }
    ~Buffer() { cudaFree(data); }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
};

template <typename T>
void RunCase(int rows, int cols, int pattern, int inputOffset,
             int outputOffset, bool capture, cudaStream_t stream) {
    size_t count = (size_t)rows * cols;
    std::vector<T> values(count + inputOffset);
    std::mt19937 random(123);
    for (size_t i = 0; i < values.size(); ++i) {
        if (pattern == 0) {
            values[i] = T(float(int(random() % 20001) - 10000) / 997.0f);
        } else {
            // Raw patterns cover signed zero, subnormal, finite extremes,
            // infinities and both signs/payloads of NaNs for FP16 and BF16.
            uint16_t bits = pattern == 1 ? ((i & 1) ? 0x8000 : 0) : uint16_t(i);
            std::memcpy(&values[i], &bits, sizeof(bits));
        }
    }
    if (pattern == 3) {
        // Include isolated non-finite values without making every row's max
        // infinite; the all-NaN row must retain the scalar scale of 1.
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                values[inputOffset + (size_t)row * cols + col] = T(float(col % 31 - 15));
            }
            uint16_t bits = row % 2 ? 0x7fff : 0xffff;
            if (row == 0) {
                for (int col = 0; col < cols; ++col)
                    std::memcpy(&values[inputOffset + col], &bits, sizeof(bits));
            } else {
                std::memcpy(&values[inputOffset + (size_t)row * cols], &bits, sizeof(bits));
            }
        }
    }
    if (pattern == 4) {
        // Pin the row scale to 1 and exercise both sides of every positive
        // E4M3 rounding midpoint, plus its negative counterpart.
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols - 1; ++col) {
                __nv_fp8_e4m3 low, high;
                low.__x = uint8_t((col / 3) % 126);
                high.__x = low.__x + 1;
                T midpoint = T((float(low) + float(high)) * 0.5f);
                uint16_t bits;
                std::memcpy(&bits, &midpoint, sizeof(bits));
                bits = uint16_t(bits + col % 3 - 1);
                if (row & 1) bits |= 0x8000;
                std::memcpy(&values[inputOffset + (size_t)row * cols + col], &bits, sizeof(bits));
            }
            values[inputOffset + (size_t)row * cols + cols - 1] = T(448.0f);
        }
    }
    constexpr size_t guard = 16;
    size_t outputBytes = count + outputOffset + guard;
    Buffer input(values.size() * sizeof(T)), actual(outputBytes), expected(count),
           scales((rows + 2) * sizeof(float)), expectedScales(rows * sizeof(float));
    Check(cudaMemcpyAsync(input.data, values.data(), values.size() * sizeof(T),
                          cudaMemcpyHostToDevice, stream));
    Check(cudaMemsetAsync(actual.data, 0xa5, outputBytes, stream));
    Check(cudaMemsetAsync(scales.data, 0xa5, (rows + 2) * sizeof(float), stream));
    auto *a = static_cast<const T *>(input.data) + inputOffset;
    auto *q = static_cast<uint8_t *>(actual.data) + outputOffset;
    auto *s = static_cast<float *>(scales.data) + 1;
    // The established scalar implementation is the bitwise contract. Compare
    // both FP8 bytes and FP32 scales, including dispatch fallbacks and graphs.
    FastllmSm89QuantPerRowKernel<T, false><<<rows, 256, 0, stream>>>(
        a, static_cast<uint8_t *>(expected.data),
        static_cast<float *>(expectedScales.data), rows, cols);
    if (capture) {
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        FastllmSm89LaunchQuantPerRow(a, q, s, rows, cols, stream);
        Check(cudaStreamEndCapture(stream, &graph));
        Check(cudaGraphInstantiate(&executable, graph, 0));
        for (int repeat = 0; repeat < 3; ++repeat) Check(cudaGraphLaunch(executable, stream));
        Check(cudaStreamSynchronize(stream));
        Check(cudaGraphExecDestroy(executable));
        Check(cudaGraphDestroy(graph));
    } else {
        FastllmSm89LaunchQuantPerRow(a, q, s, rows, cols, stream);
    }
    Check(cudaGetLastError());
    Check(cudaStreamSynchronize(stream));
    std::vector<uint8_t> observed(outputBytes), reference(count);
    std::vector<uint32_t> observedScales(rows + 2), referenceScales(rows);
    Check(cudaMemcpy(observed.data(), actual.data, outputBytes, cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(reference.data(), expected.data, count, cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(observedScales.data(), scales.data, (rows + 2) * 4, cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(referenceScales.data(), expectedScales.data, rows * 4, cudaMemcpyDeviceToHost));
    if (std::memcmp(observed.data() + outputOffset, reference.data(), count) ||
        std::memcmp(observedScales.data() + 1, referenceScales.data(), rows * 4)) {
        std::fprintf(stderr, "rows=%d cols=%d pattern=%d inputOffset=%d outputOffset=%d graph=%d\n",
                     rows, cols, pattern, inputOffset, outputOffset, capture);
        throw std::runtime_error("FP8 quantization differs from scalar reference");
    }
    for (size_t i = 0; i < observed.size(); ++i) {
        if ((i < (size_t)outputOffset || i >= outputOffset + count) && observed[i] != 0xa5)
            throw std::runtime_error("FP8 output guard overwritten");
    }
    if (observedScales.front() != 0xa5a5a5a5 || observedScales.back() != 0xa5a5a5a5)
        throw std::runtime_error("FP8 scale guard overwritten");
}

template <typename T>
int RunSuite(cudaStream_t stream) {
    int cases = 0;
    for (int rows : {1, 4, 7, 32, 2048}) {
        for (int cols : {16, 128, 1008, 1024, 1040, 3072, 5120, 8704, 16384}) {
            RunCase<T>(rows, cols, 0, 0, 0, rows == 4, stream);
            ++cases;
        }
    }
    for (int pattern : {1, 2, 3, 4}) {
        for (int cols : {1024, 1040, 8704, 32768}) {
            RunCase<T>(8, cols, pattern, 0, 0, true, stream);
            ++cases;
        }
    }
    // Dynamic dimensions and offsets exercise the scalar fallback, including
    // vectors' alignment requirements rather than only CUDA allocation bases.
    for (auto offsets : {std::pair<int, int>{1, 0}, {0, 1}, {1, 1}}) {
        for (int cols : {1023, 1024, 1025, 1040}) {
            RunCase<T>(3, cols, 0, offsets.first, offsets.second, true, stream);
            ++cases;
        }
    }
    return cases;
}
} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        int tested = 0;
        for (int device = 0; device < devices; ++device) {
            cudaDeviceProp props;
            Check(cudaGetDeviceProperties(&props, device));
            // This helper is currently used only by the SM89 CUTLASS path.
            if (props.major != 8 || props.minor != 9) continue;
            Check(cudaSetDevice(device));
            cudaStream_t stream;
            Check(cudaStreamCreate(&stream));
            int cases = RunSuite<half>(stream) + RunSuite<__nv_bfloat16>(stream);
            Check(cudaStreamDestroy(stream));
            std::printf("device=%d FP16/BF16 %d cases: bytes/scales/guards/graphs PASS\n", device, cases);
            ++tested;
        }
        if (!tested) return 77;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
