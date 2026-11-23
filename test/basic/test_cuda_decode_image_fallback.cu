#include "devices/cuda/fastllm-cuda-decode-dense.h"
#include "devices/cuda/fastllm-cuda-decode-linear.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/low_latency/execution.h"
#include "devices/cuda/naive-n05-cuda.cuh"
#include <cuda_bf16.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace fastllm;

// Only this test TU is built as sm_75 cubin without PTX. The library and the
// reference operators retain their normal build targets.
static __global__ void ForeignImage() {}
static bool probeForeign = false;
extern "C" cudaError_t __real_cudaFuncGetAttributes(cudaFuncAttributes *, const void *);
extern "C" cudaError_t __wrap_cudaFuncGetAttributes(cudaFuncAttributes *attrs, const void *kernel) {
    return __real_cudaFuncGetAttributes(attrs, probeForeign ? (const void *)ForeignImage : kernel);
}

static void Check(bool ok, const char *why) {
    if (!ok) throw std::runtime_error(why);
}
static void Cuda(cudaError_t status) { Check(status == cudaSuccess, cudaGetErrorString(status)); }
static void Upload(Data &data, DataType type, int rows, int width, float first) {
    std::vector<float> values(size_t(rows) * width, 0.0f);
    values.front() = first;
    Data host(type, {rows, width}, values);
    data.CopyFrom(host);
    data.ToDevice(DataDevice::CUDA, {0}, true);
}
// Capture the first call for each kernel variant, as a newly started rank worker can.
static void CheckColdRouterCapture(Data &weight, int rows, bool hasImage) {
    const int experts = weight.dims[0];
    Data x, fp32, scores, empty, bias, biased;
    Upload(x, BFLOAT16, rows, 4096, 1);
    Upload(fp32, FLOAT32, rows, 4096, 1);
    Upload(scores, FLOAT32, rows, experts, 0);
    Upload(bias, FLOAT32, 1, experts, 3);
    Upload(biased, FLOAT32, rows, experts, 0);
    Cuda(cudaDeviceSynchronize());
    probeForeign = !hasImage;
    Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    const bool fused = FastllmCudaNaiveRouterSigmoid(x, weight, scores);
    if (!hasImage && !fused) {
        Check(FastllmCudaMatMulFloat32(fp32, weight, empty, scores, rows, 4096, experts) &&
              FastllmCudaSigmoid(scores, scores), "bias-free Router fallback failed");
        Check(FastllmCudaMatMulFloat32(fp32, weight, bias, biased, rows, 4096, experts),
              "biased Linear failed");
    }
    cudaGraph_t graph;
    Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    probeForeign = false;
    Check(fused == hasImage, "cold Router image admission during capture");
    cudaGraphExec_t exec;
    Cuda(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
    Cuda(cudaStreamSynchronize(cudaStreamPerThread));
    std::vector<float> host(rows * experts);
    Cuda(cudaMemcpy(host.data(), scores.cudaData, scores.GetBytes(), cudaMemcpyDeviceToHost));
    for (size_t j = 0; j < host.size(); ++j)
        Check(std::abs(host[j] - (j == 0 ? 1.f / (1.f + std::exp(-2.f)) : .5f)) < 1e-6f,
              "captured Router output mismatch");
    if (!hasImage) {
        Cuda(cudaMemcpy(host.data(), biased.cudaData, biased.GetBytes(), cudaMemcpyDeviceToHost));
        for (size_t j = 0; j < host.size(); ++j)
            Check(host[j] == (j == 0 ? 5.f : j % experts == 0 ? 3.f : 0.f),
                  "captured Linear bias mismatch");
    }
    Cuda(cudaGraphExecDestroy(exec));
    Cuda(cudaGraphDestroy(graph));
}
int main() {
    try {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
        Cuda(cudaSetDevice(0));
        cudaDeviceProp prop{};
        Cuda(cudaGetDeviceProperties(&prop, 0));
        if (prop.major < 8) return 77;
        namespace ll = fastllm::cuda::low_latency;
        cudaFuncAttributes attr{};
        Check(ll::BackendUnavailable(__real_cudaFuncGetAttributes(&attr, (const void *)ForeignImage)),
              "test requires an incompatible native cubin");
        cudaGetLastError();

        constexpr int h = 4096, i = 2048;
        Data x, gate, up, down, gamma, router, mid, out, empty;
        Upload(x, BFLOAT16, 1, h, 1);
        Upload(gate, BFLOAT16, i, h, 1);
        Upload(up, BFLOAT16, i, h, 1);
        Upload(down, BFLOAT16, h, i, 1);
        Upload(router, FLOAT32, 16, h, 2);
        Data scale(FLOAT32, {h}, std::vector<float>(h, 1));
        gamma.CopyFrom(scale); gamma.ToDevice(DataDevice::CUDA, {0}, true);

        FastllmCudaSetLinearExactBatchThreshold(8);
        CheckColdRouterCapture(router, 2, true);
        CheckColdRouterCapture(router, 1, false);

        probeForeign = true;
        for (int attempt = 0; attempt < 2; ++attempt) {
            Check(!FastllmCudaNaiveRouterSigmoid(x, router, out), "Router accepted a missing image");
            Check(!FastllmCudaDecodeDense(x, gate, up, down, mid, out), "Dense accepted a missing image");
            Check(!FastllmCudaDecodeNormLinear(x, nullptr, gamma, gate, out, 1e-5f),
                  "projection accepted a missing image");
            Check(!mid.cudaData && !out.cudaData && mid.dims.empty() && out.dims.empty(),
                  "fallback mutated outputs");
            Cuda(cudaPeekAtLastError());
        }
        probeForeign = false;

        Data gateResult, upResult, result;
        Upload(gateResult, BFLOAT16, 1, i, 0);
        Upload(upResult, BFLOAT16, 1, i, 0);
        Upload(result, BFLOAT16, 1, h, 0);
        Check(FastllmCudaBFloat16MatMulBFloat16(x, gate, empty, gateResult, 1, h, i) &&
              FastllmCudaBFloat16MatMulBFloat16(x, up, empty, upResult, 1, h, i) &&
              FastllmCudaSilu(gateResult, gateResult) && FastllmCudaMulTo(gateResult, upResult, 1) &&
              FastllmCudaBFloat16MatMulBFloat16(gateResult, down, empty, result, 1, i, h),
              "ordinary Dense path failed");
        std::vector<__nv_bfloat16> host(h);
        Cuda(cudaMemcpy(host.data(), result.cudaData, result.GetBytes(), cudaMemcpyDeviceToHost));
        Check(float(host[0]) == float(__float2bfloat16_rn(1.0f / (1.0f + std::exp(-1.0f)))),
              "ordinary Dense produced an incorrect active channel");
        for (int j = 1; j < h; ++j) Check(float(host[j]) == 0, "ordinary Dense changed a zero channel");
        Data fp32Input, probabilities, normalized, head;
        Upload(fp32Input, FLOAT32, 1, h, 0);
        Upload(probabilities, FLOAT32, 1, 16, 0);
        Check(FastllmBF16ToFloat(x.cudaData, fp32Input.cudaData, h) &&
              FastllmCudaMatMulFloat32(fp32Input, router, empty, probabilities, 1, h, 16) &&
              FastllmCudaSigmoid(probabilities, probabilities), "ordinary Router path failed");
        std::vector<float> scores(16);
        Cuda(cudaMemcpy(scores.data(), probabilities.cudaData, probabilities.GetBytes(), cudaMemcpyDeviceToHost));
        for (int j = 0; j < 16; ++j)
            Check(std::abs(scores[j] - (j == 0 ? 1.f / (1.f + std::exp(-2.f)) : .5f)) < 1e-6f,
                  "ordinary Router produced an incorrect probability");

        Upload(normalized, BFLOAT16, 1, h, 0);
        Upload(head, FLOAT32, 1, i, 0);
        Check(FastllmCudaRMSNorm(x, gamma, normalized, 1e-5f) &&
              FastllmCudaBFloat16MatMulBFloat16(normalized, gate, empty, gateResult, 1, h, i) &&
              FastllmBF16ToFloat(gateResult.cudaData, head.cudaData, i), "ordinary projection path failed");
        scores.resize(i);
        Cuda(cudaMemcpy(scores.data(), head.cudaData, head.GetBytes(), cudaMemcpyDeviceToHost));
        const float expected = float(__float2bfloat16_rn(1.f / std::sqrt(1.f / h + 1e-5f)));
        for (int j = 0; j < i; ++j)
            Check(scores[j] == (j == 0 ? expected : 0.f), "ordinary projection produced an incorrect channel");
        std::puts("PASS decode image fallback");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
