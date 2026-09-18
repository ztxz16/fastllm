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
    auto status = cudaFuncGetAttributes(&attr, nvfused::GemvEpilogue<half, 0, 0, true, 8, 2, 2, 1>);
    auto downStatus =
        cudaFuncGetAttributes(&downAttr, nvfused::GemvEpilogue<half, 0, 0, false, 8, 2, 2, 1>);
    if (status != cudaSuccess || downStatus != cudaSuccess)
        cudaGetLastError();
    DeviceInfo result{sms, major * 10 + minor >= 75 && sms > 0 && status == cudaSuccess &&
                               attr.maxThreadsPerBlock >= 256 && downStatus == cudaSuccess &&
                               downAttr.maxThreadsPerBlock >= 256};
    cache.emplace(device, result);
    return result;
}
bool ShapeTuningEnabled() {
    const char *flag = std::getenv("FASTLLM_CUDA_NVFP4_SHAPE_TUNING");
    return !flag || (std::strcmp(flag, "0") && std::strcmp(flag, "false"));
}
} // namespace
bool FastllmCudaNvfp4FusedCanRun(const Data &input, const Data &weight, const Data &bias, const Data &output,
                                 bool gate) {
    const char *flag = std::getenv(gate ? "FASTLLM_CUDA_NVFP4_SWIGLU" : "FASTLLM_CUDA_NVFP4_ADD");
    if (flag && (!std::strcmp(flag, "0") || !std::strcmp(flag, "false")))
        return false;
    if (weight.dims.size() != 2) return false;
    const int n = weight.dims[0], k = weight.dims[1], out = gate ? n / 2 : n;
    if (n < 128 || n > 65536 || n % 128 || k < 256 || k > 32768 || k % 128) return false;
    // These uneven gate/up shards are faster with Marlin + SwiGLU in
    // end-to-end TP decode, despite the extra launch. Keep that complete
    // fallback; do not select a fusion merely because its shape is valid.
    if (gate && k == 5120 && (n == 11520 || n == 11776))
        return false;
    const char *generic = std::getenv("FASTLLM_CUDA_TP_FUSIONS");
    if (generic && (!std::strcmp(generic, "0") || !std::strcmp(generic, "false")) &&
        (n != (gate ? 34816 : 5120) || k != (gate ? 5120 : 17408))) return false;
    if (input.dims.empty() || input.strides.size() != input.dims.size() ||
        output.dims.empty() || output.strides.size() != output.dims.size()) return false;
    const uint64_t count = input.Count(0);
    const uint64_t batch = count / k;
    const bool smallBatch = !gate && n == 5120 && k == 8704 && ShapeTuningEnabled();
    if (count % k || batch < 1 || batch > (smallBatch ? 8 : 1)) return false;
    if (input.dataType != DataType::FLOAT16 || output.dataType != input.dataType || input.dims.empty() ||
        output.dims.empty() || weight.dataType != DataType::NVFP4_BLOCK_16 ||
        weight.dims.size() != 2 || weight.dims[0] != n || weight.dims[1] != k || weight.blockK != 1 || weight.blockM != 16 ||
        !weight.IsRepacked || !bias.dims.empty() || input.dims.back() != k || input.Count(0) != batch * uint64_t(k) ||
        output.dims.back() != out || output.Count(0) != batch * uint64_t(out))
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
    if (batch > 1) {
        // Small-batch Tensor Core reduction uses scratch already reclaimed by
        // the Marlin repack. Admission must never allocate during capture.
        const size_t scratchEnd = ((required + 15) & ~size_t(15)) + size_t(info.sms) * 8 * 256 * sizeof(float);
        if (!Dense(output, device, 16) || weight.GetBytes() < scratchEnd ||
            !FastllmCudaMarlinNVFP4AddSupported(n, k))
            return false;
    }
    if (weight.GetBytes() < required || Overlap(input, batch * k * 2, weight, weight.GetBytes()) ||
        Overlap(input, batch * k * 2, output, batch * out * 2) || Overlap(weight, weight.GetBytes(), output, batch * out * 2))
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
    const bool tuneShape = info.supported && ShapeTuningEnabled();
    // Specializing both matrix dimensions removes runtime packed-weight
    // and scale address arithmetic. Keep the same warp layout and FP32 reduction
    // order as the generic fusion. Select by local shape, not by TP rank count.
    if (tuneShape && !gate && n == 5120 && k == 8704) {
        const int batch = input.Count(0) / k;
        if (batch > 1) {
            auto *workspace = (int *)(scales + size_t(n) * k / 16);
            auto *scratch = (void *)((reinterpret_cast<uintptr_t>(global) + sizeof(float) + 15) & ~uintptr_t(15));
            AssertInFastLLM(FastllmCudaMarlinHalfNVFP4Add(input.cudaData, q, scales, global,
                output.cudaData, batch, n, k, workspace, scratch), "NVFP4 residual GEMM unavailable after admission.\n");
        } else {
            nvfused::GemvEpilogue<half, 5120, 8704, false, 8, 2, 2, 1>
                <<<640, 256, 0, cudaStreamPerThread>>>(
                    (const half *)input.cudaData, q, scales, global, (half *)output.cudaData);
        }
    } else if (gate && n != 34816 && k == 5120)
        nvfused::GemvEpilogue<half, 0, 5120, true, 8, 2, 2, 1><<<n / 16, 256, 0, cudaStreamPerThread>>>(
            (const half *)input.cudaData, q, scales, global, (half *)output.cudaData, n, k);
    else if (gate && (n != 34816 || k != 5120))
        nvfused::GemvEpilogue<half, 0, 0, true, 8, 2, 2, 1><<<n / 16, 256, 0, cudaStreamPerThread>>>(
            (const half *)input.cudaData, q, scales, global, (half *)output.cudaData, n, k);
    else if (!gate && (n != 5120 || k != 17408))
        nvfused::GemvEpilogue<half, 0, 0, false, 8, 2, 2, 1><<<n / 8, 256, 0, cudaStreamPerThread>>>(
            (const half *)input.cudaData, q, scales, global, (half *)output.cudaData, n, k);
    else if (gate)
        nvfused::GemvEpilogue<half, 34816, 5120, true, 8, 1, 2, 4><<<544, 256, 0, cudaStreamPerThread>>>(
            (const half *)input.cudaData, q, scales, global, (half *)output.cudaData);
    else
        nvfused::GemvEpilogue<half, 5120, 17408, false, 16, 2, 2, 2><<<320, 512, 0, cudaStreamPerThread>>>(
            (const half *)input.cudaData, q, scales, global, (half *)output.cudaData);
    AssertInFastLLM(cudaGetLastError() == cudaSuccess, "NVFP4 fused CUDA launch failed.\n");
}
bool FastllmCudaNvfp4ShapeGemvCanRun(const Data &input, const Data &weight, const Data &bias,
                                   const Data &output) {
    const char *flag = std::getenv("FASTLLM_CUDA_NVFP4_SHAPE_TUNING");
    if ((flag && (!std::strcmp(flag, "0") || !std::strcmp(flag, "false"))) ||
        weight.dims.size() != 2 || weight.dims[0] != 5120 || weight.dims[1] != 8704 ||
        input.dims.empty() || input.strides.size() != input.dims.size() || input.Count(0) != 8704 ||
        !FastllmCudaNvfp4FusedCanRun(input, weight, bias, output, false))
        return false;
    int device = 0;
    return cudaGetDevice(&device) == cudaSuccess && Info(device).supported;
}
void FastllmCudaNvfp4ShapeGemv(const Data &input, const Data &weight, Data &output) {
    int device = 0;
    cudaGetDevice(&device);
    const auto info = Info(device);
    const auto *q = (const uint32_t *)weight.cudaData;
    const auto *scales = (const uint8_t *)weight.cudaData + size_t(5120) * 8704 / 2;
    const auto *global = (const float *)(scales + size_t(5120) * 8704 / 16 +
                                       size_t(info.sms) * 4 * sizeof(int));
    nvfused::GemvEpilogue<half, 5120, 8704, false, 8, 2, 2, 1, false>
        <<<640, 256, 0, cudaStreamPerThread>>>(
            (const half *)input.cudaData, q, scales, global, (half *)output.cudaData);
    AssertInFastLLM(cudaGetLastError() == cudaSuccess, "NVFP4 shape GEMV launch failed.\n");
}
} // namespace fastllm
