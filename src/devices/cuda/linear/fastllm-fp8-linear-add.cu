#include "devices/cuda/fastllm-cuda-fp8-linear-add.h"
#include "devices/cuda/fastllm-fp8-linear-add.cuh"
#include "devices/cuda/fastllm-fp8-row.cuh"
#include "fastllm-cuda.cuh"
#include "utils.h"
#include <cstdlib>
#include <cstring>
#include <map>
extern void FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(fastllm::Data &, const fastllm::Data &, int);
namespace fastllm {
namespace {
bool DenseCuda(const Data &x, int device, size_t alignment) {
    if (x.dataDevice != DataDevice::CUDA || !x.cudaData || x.multiDeviceData ||
        reinterpret_cast<uintptr_t>(x.cudaData) % alignment ||
        (!x.dataDeviceIds.empty() && (x.dataDeviceIds.size() != 1 || x.dataDeviceIds[0] != device)) ||
        x.strides.size() != x.dims.size())
        return false;
    uint64_t stride = 1;
    for (int i = int(x.dims.size()) - 1; i >= 0; --i) {
        if (x.dims[i] <= 0 || x.strides[i] != stride)
            return false;
        stride *= x.dims[i];
    }
    return true;
}
bool Overlap(const Data &a, size_t asize, const Data &b, size_t bsize) {
    uintptr_t ap = reinterpret_cast<uintptr_t>(a.cudaData), bp = reinterpret_cast<uintptr_t>(b.cudaData);
    return ap < bp + bsize && bp < ap + asize;
}
template <class T, int K> void Launch(Data &input, Data &weight, const Data &bias, Data &output) {
    fp8add::Kernel<T, K><<<320, 256, 0, cudaStreamPerThread>>>(
        (const T *)input.cudaData, (const uint8_t *)weight.cudaData, (const float *)weight.extraCudaData[0],
        bias.dims.empty() ? nullptr : (const float *)bias.cudaData, (T *)output.cudaData);
}
template <class T> void Dispatch(Data &input, Data &weight, const Data &bias, Data &output) {
    if (weight.dims[0] != 5120 || (weight.dims[1] != 6144 && weight.dims[1] != 17408)) {
        const int K = weight.dims[1], N = weight.dims[0];
        const auto *x = (const T *)input.cudaData;
        const auto *w = (const uint8_t *)weight.cudaData;
        const auto *scales = (const float *)weight.extraCudaData[0];
        const auto *b = bias.dims.empty() ? nullptr : (const float *)bias.cudaData;
        auto *y = (T *)output.cudaData;
        if (K % 512 == 0)
            fp8row::Kernel<T, 8, 1, 2, 16, true, float>
                <<<(N + 7) / 8, 256, 0, cudaStreamPerThread>>>(x, w, scales, b, y, K, N);
        else if (K % 256 == 0)
            fp8row::Kernel<T, 8, 1, 2, 8, true, float>
                <<<(N + 7) / 8, 256, 0, cudaStreamPerThread>>>(x, w, scales, b, y, K, N);
        else
            fp8row::TailKernel<T, true, float>
                <<<(N + 7) / 8, 256, 0, cudaStreamPerThread>>>(x, w, scales, b, y, K, N);
    } else if (weight.dims[1] == 6144)
        Launch<T, 6144>(input, weight, bias, output);
    else
        Launch<T, 17408>(input, weight, bias, output);
}
} // namespace
bool FastllmCudaFP8LinearAddCanRun(const Data &input, const Data &weight, const Data &bias,
                                   const Data &output) {
    const char *enabled = std::getenv("FASTLLM_CUDA_FP8_LINEAR_ADD");
    if (enabled && (!std::strcmp(enabled, "0") || !std::strcmp(enabled, "false")))
        return false;
    if ((input.dataType != DataType::FLOAT16 && input.dataType != DataType::BFLOAT16) ||
        output.dataType != input.dataType || input.dims.empty() || output.dims.empty() ||
        weight.dataType != DataType::FP8_E4M3 || weight.dims.size() != 2 ||
        weight.dims[0] < 512 || weight.dims[0] > 65536 || weight.dims[1] < 512 || weight.dims[1] > 32768 ||
        weight.IsRepacked || weight.blockK != 1 ||
        weight.blockM < weight.dims[1] || weight.scales.size() != size_t(weight.dims[0]) ||
        input.dims.back() != weight.dims[1] || input.Count(0) != uint64_t(weight.dims[1]) ||
        output.dims.back() != weight.dims[0] || output.Count(0) != uint64_t(weight.dims[0]))
        return false;
    const int N = weight.dims[0];
    const char *generic = std::getenv("FASTLLM_CUDA_TP_FUSIONS");
    if (generic && (!std::strcmp(generic, "0") || !std::strcmp(generic, "false")) &&
        (N != 5120 || (weight.dims[1] != 6144 && weight.dims[1] != 17408) || weight.blockM != weight.dims[1]))
        return false;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || !DenseCuda(input, device, 16) ||
        !DenseCuda(weight, device, 16) || !DenseCuda(output, device, 2))
        return false;
    if (!bias.dims.empty() && (bias.dataType != DataType::FLOAT32 || bias.dims != std::vector<int>{N} ||
                               !DenseCuda(bias, device, 4)))
        return false;
    if (Overlap(input, input.Count(0) * 2, output, size_t(N) * 2) ||
        Overlap(weight, weight.Count(0), output, size_t(N) * 2) ||
        (!bias.dims.empty() && Overlap(bias, size_t(N) * 4, output, size_t(N) * 2)))
        return false;
    static thread_local std::map<int, bool> supported;
    auto it = supported.find(device);
    if (it == supported.end()) {
        int major = 0, minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess)
            return false;
        if (major * 10 + minor < 75)
            return false;
        // All dispatch variants are compiled together for the same architecture list.
        cudaFuncAttributes attributes{};
        auto status = cudaFuncGetAttributes(&attributes, fp8row::TailKernel<half, true, float>);
        if (status != cudaSuccess)
            cudaGetLastError();
        it = supported.emplace(device, status == cudaSuccess && attributes.maxThreadsPerBlock >= 256).first;
    }
    if (!it->second)
        return false;
    // Allocation/copies are forbidden during capture; warmup prepares the scale cache.
    cudaStreamCaptureStatus capture;
    if (cudaStreamIsCapturing(cudaStreamPerThread, &capture) != cudaSuccess)
        return false;
    return capture == cudaStreamCaptureStatusNone ||
           (!weight.extraCudaData.empty() && weight.extraCudaData[0]);
}
void FastllmCudaFP8LinearAdd(Data &input, Data &weight, const Data &bias, Data &output) {
    FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(weight, bias, weight.dims[0]);
    if (input.dataType == DataType::FLOAT16)
        Dispatch<half>(input, weight, bias, output);
    else
        Dispatch<__nv_bfloat16>(input, weight, bias, output);
    AssertInFastLLM(cudaGetLastError() == cudaSuccess, "FP8 LinearAdd CUDA launch failed.\n");
}
} // namespace fastllm
