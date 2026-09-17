#include "devices/cuda/fastllm-cuda-nvfp4-fused.h"
#include "devices/cuda/fastllm-nvfp4-fused.cuh"
#include "fastllm-cuda.cuh"
#include "utils.h"
#include <cstdlib>
#include <cstring>
#include <map>
namespace fastllm {
namespace {
bool Dense(const Data &d, int device, size_t align) {
    if (d.dataDevice != DataDevice::CUDA || !d.cudaData || d.multiDeviceData ||
        reinterpret_cast<uintptr_t>(d.cudaData) % align || d.dims.empty() ||
        d.strides.size() != d.dims.size() ||
        (!d.dataDeviceIds.empty() && (d.dataDeviceIds.size() != 1 || d.dataDeviceIds[0] != device)))
        return false;
    uint64_t stride = 1;
    for (int i = int(d.dims.size()) - 1; i >= 0; --i) {
        if (d.dims[i] <= 0 || d.strides[i] != stride)
            return false;
        stride *= d.dims[i];
    }
    return true;
}
bool Overlap(const Data &a, size_t as, const Data &b, size_t bs) {
    auto ap = reinterpret_cast<uintptr_t>(a.cudaData), bp = reinterpret_cast<uintptr_t>(b.cudaData);
    return ap < bp + bs && bp < ap + as;
}
struct DeviceInfo {
    int sms;
    bool supported;
};
DeviceInfo Info(int device) {
    static thread_local std::map<int, DeviceInfo> cache;
    auto it = cache.find(device);
    if (it != cache.end())
        return it->second;
    int major = 0, minor = 0, sms = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess)
        return {0, false};
    cudaFuncAttributes attr{}, downAttr{};
    auto status = cudaFuncGetAttributes(&attr, nvfused::GemvEpilogue<half, 34816, 5120, true, 8, 1, 2, 4>);
    auto downStatus =
        cudaFuncGetAttributes(&downAttr, nvfused::GemvEpilogue<half, 5120, 17408, false, 16, 2, 2, 2>);
    if (status != cudaSuccess || downStatus != cudaSuccess)
        cudaGetLastError();
    DeviceInfo result{sms, major * 10 + minor >= 75 && sms > 0 && status == cudaSuccess &&
                               attr.maxThreadsPerBlock >= 256 && downStatus == cudaSuccess &&
                               downAttr.maxThreadsPerBlock >= 512};
    cache.emplace(device, result);
    return result;
}
} // namespace
bool FastllmCudaNvfp4FusedCanRun(const Data &input, const Data &weight, const Data &bias, const Data &output,
                                 bool gate) {
    const char *flag = std::getenv(gate ? "FASTLLM_CUDA_NVFP4_SWIGLU" : "FASTLLM_CUDA_NVFP4_ADD");
    if (flag && (!std::strcmp(flag, "0") || !std::strcmp(flag, "false")))
        return false;
    const int n = gate ? 34816 : 5120, k = gate ? 5120 : 17408, out = gate ? n / 2 : n;
    if (input.dataType != DataType::FLOAT16 || output.dataType != input.dataType || input.dims.empty() ||
        output.dims.empty() || weight.dataType != DataType::NVFP4_BLOCK_16 ||
        weight.dims.size() != 2 || weight.dims[0] != n || weight.dims[1] != k || weight.blockK != 1 || weight.blockM != 16 ||
        !weight.IsRepacked || !bias.dims.empty() || input.dims.back() != k || input.Count(0) != uint64_t(k) ||
        output.dims.back() != out || output.Count(0) != uint64_t(out))
        return false;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess || !Dense(input, device, 16) || !Dense(weight, device, 16) ||
        !Dense(output, device, 2))
        return false;
    auto info = Info(device);
    if (!info.supported || !FastllmCudaHasNVFP4MarlinLayout(weight))
        return false;
    // The existing Marlin conversion stores its tensor scale after the packed codes,
    // byte scales and per-SM lock workspace in the source allocation's reclaimed tail.
    size_t required =
        size_t(n) * k / 2 + size_t(n) * k / 16 + size_t(info.sms) * 4 * sizeof(int) + sizeof(float);
    if (weight.GetBytes() < required || Overlap(input, k * 2, weight, weight.GetBytes()) ||
        Overlap(input, k * 2, output, out * 2) || Overlap(weight, weight.GetBytes(), output, out * 2))
        return false;
    return true;
}
void FastllmCudaNvfp4Fused(Data &input, Data &weight, Data &output, bool gate) {
    int device = 0;
    cudaGetDevice(&device);
    auto info = Info(device);
    int n = weight.dims[0], k = weight.dims[1];
    auto q = (const uint32_t *)weight.cudaData;
    auto scales = (const uint8_t *)weight.cudaData + size_t(n) * k / 2;
    auto global = (const float *)(scales + size_t(n) * k / 16 + size_t(info.sms) * 4 * sizeof(int));
    if (gate)
        nvfused::GemvEpilogue<half, 34816, 5120, true, 8, 1, 2, 4><<<544, 256, 0, cudaStreamPerThread>>>(
            (const half *)input.cudaData, q, scales, global, (half *)output.cudaData);
    else
        nvfused::GemvEpilogue<half, 5120, 17408, false, 16, 2, 2, 2><<<320, 512, 0, cudaStreamPerThread>>>(
            (const half *)input.cudaData, q, scales, global, (half *)output.cudaData);
    AssertInFastLLM(cudaGetLastError() == cudaSuccess, "NVFP4 fused CUDA launch failed.\n");
}
} // namespace fastllm
