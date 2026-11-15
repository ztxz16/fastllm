#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1
#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
void Check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

uint16_t Encode(float value, fastllm::DataType type) {
    uint16_t bits;
    if (type == fastllm::BFLOAT16) {
        auto v = __float2bfloat16(value);
        std::memcpy(&bits, &v, sizeof(bits));
    } else {
        auto v = __float2half_rn(value);
        std::memcpy(&bits, &v, sizeof(bits));
    }
    return bits;
}

void Case(fastllm::DataType type, int rows, bool hasBias) {
    using namespace fastllm;
    constexpr int m = 256, k = 128;
    Data weight(BFLOAT16, {k, m}), input(type, {rows, m}), output(type, {rows, k});
    weight.Allocate();
    for (int j = 0; j < k; ++j)
        for (int i = 0; i < m; ++i)
            ((uint16_t *)weight.cpuData)[j * m + i] = Encode((j % 7 - 3) / 32.f, BFLOAT16);
    weight.ToDevice(CUDA, std::vector<int>{0});
    input.ToDevice(CUDA, std::vector<int>{0});
    output.ToDevice(CUDA, std::vector<int>{0});
    input.Allocate();
    output.Allocate();
    Data empty, bias(FLOAT32, {k});
    if (hasBias) {
        bias.Allocate();
        for (int j = 0; j < k; ++j) ((float *)bias.cpuData)[j] = (j % 11 - 5) / 32.f;
        bias.ToDevice(CUDA, std::vector<int>{0});
    }
    // Real bias keeps its normal warmed cache. Bias-free grouped projections
    // instead reconstruct this view on every invocation, including capture.
    Data biasedView;
    biasedView.FakeFrom(weight, 0);
    biasedView.Resize({k, m});
    auto invoke = [&]() {
        Data freshView;
        freshView.FakeFrom(weight, 0);
        freshView.Resize({k, m});
        Data &view = hasBias ? biasedView : freshView;
        bool ok = type == FLOAT16
            ? FastllmCudaHalfMatMulBFloat16(input, view, hasBias ? bias : empty, output, rows, m, k)
            : FastllmCudaBFloat16MatMulBFloat16(input, view, hasBias ? bias : empty, output, rows, m, k);
        if (!ok) throw std::runtime_error("BF16 linear declined");
        return view.extraCudaData.empty() && view.extraCudaHalfData.empty();
    };
    std::vector<uint16_t> values(rows * m), actual(rows * k);
    auto upload = [&](int sign) {
        for (int r = 0; r < rows; ++r)
            for (int i = 0; i < m; ++i) values[r * m + i] = Encode(sign * (r % 5 + 1) / 64.f, type);
        Check(cudaMemcpy(input.cudaData, values.data(), values.size() * 2, cudaMemcpyHostToDevice));
    };
    const char *stage = "eager";
    auto verify = [&](int sign) {
        Check(cudaMemcpy(actual.data(), output.cudaData, actual.size() * 2, cudaMemcpyDeviceToHost));
        for (int r = 0; r < rows; ++r) for (int j = 0; j < k; ++j) {
            float expected = m * sign * (r % 5 + 1) / 64.f * (j % 7 - 3) / 32.f;
            if (hasBias) expected += (j % 11 - 5) / 32.f;
            if (actual[r * k + j] != Encode(expected, type) &&
                !(expected == 0.f && (actual[r * k + j] & 0x7fff) == 0)) {
                std::fprintf(stderr, "stage=%s dtype=%d rows=%d bias=%d sign=%d at [%d,%d]: bits %04x expected %04x (%g)\n", stage, int(type), rows, int(hasBias), sign, r, j, actual[r*k+j], Encode(expected,type), expected);
                throw std::runtime_error("BF16 linear graph/reference mismatch");
            }
        }
    };
    upload(1);
    invoke();
    Check(cudaDeviceSynchronize());
    verify(1);
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    bool noAuxiliary = invoke();
    Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    if (!hasBias && !noAuxiliary) throw std::runtime_error("bias-free projection allocated bias storage");
    stage = "replay";
    for (int sign : {-1, 1}) {
        upload(sign);
        Check(cudaGraphLaunch(exec, cudaStreamPerThread));
        Check(cudaStreamSynchronize(cudaStreamPerThread));
        verify(sign);
    }
    Check(cudaGraphExecDestroy(exec));
    Check(cudaGraphDestroy(graph));
    for (void *p : biasedView.extraCudaData) Check(cudaFree(p));
    for (void *p : biasedView.extraCudaHalfData) Check(cudaFree(p));
    biasedView.extraCudaData.clear();
    biasedView.extraCudaHalfData.clear();
    std::printf("PASS dtype=%d rows=%d bias=%d eager + graph replay\n", int(type), rows, int(hasBias));
}
}

int main() {
    try {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
        Check(cudaSetDevice(0));
        for (auto type : {fastllm::FLOAT16, fastllm::BFLOAT16})
            for (int rows : {1, 4, 8, 17})
                for (bool bias : {false, true}) Case(type, rows, bias);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    } catch (const char *error) {
        std::fprintf(stderr, "%s\n", error);
        return 1;
    }
    return 0;
}
