#include "devices/cuda/fastllm-cuda-gdn.h"
#include "devices/cuda/fastllm-gdn-input.cuh"
#include "fastllm-cuda.cuh"
#include "utils.h"
#include <cstdlib>
#include <cstring>
#include <map>
extern void FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(fastllm::Data &, const fastllm::Data &, int);
namespace fastllm {
namespace {
bool DenseCuda(const Data &x, int device, size_t alignment = 16) {
    if (x.dataDevice != DataDevice::CUDA || !x.cudaData || x.multiDeviceData ||
        reinterpret_cast<uintptr_t>(x.cudaData) % alignment ||
        (!x.dataDeviceIds.empty() && (x.dataDeviceIds.size() != 1 || x.dataDeviceIds[0] != device)) ||
        x.strides.size() != x.dims.size())
        return false;
    uint64_t stride = 1;
    for (int i = int(x.dims.size()) - 1; i >= 0; --i) {
        if (x.strides[i] != stride || x.dims[i] <= 0)
            return false;
        stride *= x.dims[i];
    }
    return true;
}
bool Bias(const Data &b, int n, int device) {
    return b.dims.empty() ||
           (b.dataType == DataType::FLOAT32 && b.dims == std::vector<int>{n} && DenseCuda(b, device, 4));
}
template <class T, int BT, int W, int R, int V, int Chains = 4, bool Dynamic = false>
void Launch(Data &input, Data &weight, const Data &bias, const Data &cw, const Data &cb, Data &cache,
            const Data *slots, Data &out, Data &z, int batch) {
    gdn::InputConvKernel<T, BT, W, R, V, Chains, Dynamic>
        <<<dim3((weight.dims[0] + W * R - 1) / (W * R), (batch + BT - 1) / BT), W * 32, 0, cudaStreamPerThread>>>(
            (const T *)input.cudaData, (const uint8_t *)weight.cudaData,
            (const float *)weight.extraCudaData[0],
            bias.dims.empty() ? nullptr : (const float *)bias.cudaData, (const float *)cw.cudaData,
            cb.dims.empty() ? nullptr : (const float *)cb.cudaData, (T *)cache.cudaData,
            slots ? (const int *)slots->cudaData : nullptr, (T *)out.cudaData, (T *)z.cudaData, batch, weight.dims[1], cw.dims[0], weight.dims[0] - cw.dims[0]);
}
template <class T, bool Dynamic = false>
void Dispatch(Data &input, Data &weight, const Data &bias, const Data &cw, const Data &cb, Data &cache,
              const Data *slots, Data &out, Data &z, int batch) {
    // All tiles are exact: never dispatch a rounded-up partial batch.
    switch (batch) {
    case 1:
        Launch<T, 1, 8, 2, 8, 4, Dynamic>(input, weight, bias, cw, cb, cache, slots, out, z, batch);
        break;
    case 2:
        Launch<T, 2, 4, 2, 8, 4, Dynamic>(input, weight, bias, cw, cb, cache, slots, out, z, batch);
        break;
    case 3:
        Launch<T, 3, 4, 2, 8, 4, Dynamic>(input, weight, bias, cw, cb, cache, slots, out, z, batch);
        break;
    case 4:
        Launch<T, 4, 4, 2, 8, 4, Dynamic>(input, weight, bias, cw, cb, cache, slots, out, z, batch);
        break;
    case 5:
        Launch<T, 5, 4, 2, 8, 4, Dynamic>(input, weight, bias, cw, cb, cache, slots, out, z, batch);
        break;
    case 6:
        Launch<T, 6, 4, 2, 8, 4, Dynamic>(input, weight, bias, cw, cb, cache, slots, out, z, batch);
        break;
    case 7:
        Launch<T, 7, 4, 2, 8, 4, Dynamic>(input, weight, bias, cw, cb, cache, slots, out, z, batch);
        break;
    case 8:
        // More rows share each activation load; one chain limits register use.
        Launch<T, 8, 8, 4, 8, 1, Dynamic>(input, weight, bias, cw, cb, cache, slots, out, z, batch);
        break;
    default:
        AssertInFastLLM(false, "Unsupported fused GDN batch.\n");
    }
}
template <class T>
__global__ void ProjectedConv(const T *projected, const float *cw, const float *bias, T *cache,
                              const int *slots, T *out, T *z, int batch, int channels, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch * width)
        return;
    int b = i / width, c = i % width;
    if (c >= channels) {
        z[(size_t)b * (width - channels) + c - channels] = projected[i];
        return;
    }
    int slot = slots ? slots[b] : b;
    T *h = cache + ((size_t)slot * channels + c) * 4;
    T a = h[1], d = h[2], e = h[3], p = projected[i];
    h[0] = a;
    h[1] = d;
    h[2] = e;
    h[3] = p;
    float v = bias ? bias[c] : 0.f;
    const float *w = cw + c * 4;
    v = fmaf(float(a), w[0], v);
    v = fmaf(float(d), w[1], v);
    v = fmaf(float(e), w[2], v);
    v = fmaf(float(p), w[3], v);
    float r = float(T(v));
    out[(size_t)b * channels + c] = T(r / (1.f + expf(-r)));
}
} // namespace
bool FastllmCudaGdnInputConvValidInputs(const Data &input, const Data &weight, const Data &bias,
                                        const Data &cw, const Data &cb, const Data &cache, const Data *slots,
                                        int batch) {
    if (batch <= 0 || (input.dataType != DataType::FLOAT16 && input.dataType != DataType::BFLOAT16) ||
        input.dims.empty() || weight.dims.size() != 2 || weight.dims[0] <= 0 || weight.dims[1] <= 0 ||
        input.dims.back() != weight.dims[1] || input.Count(0) != uint64_t(batch) * weight.dims[1] ||
        cw.dataType != DataType::FLOAT32 || cw.dims.empty() || cw.dims[0] <= 0)
        return false;
    int channels = cw.dims[0];
    if ((cw.dims != std::vector<int>({channels, 4}) && cw.dims != std::vector<int>({channels, 1, 4})) ||
        weight.dims[0] <= channels || cache.dataType != input.dataType ||
        cache.Count(0) < uint64_t(batch) * channels * 4 ||
        !((cache.dims.size() == 3 && cache.dims[1] == channels && cache.dims[2] == 4) ||
          (cache.dims.size() == 4 && cache.dims[1] == 1 && cache.dims[2] == channels && cache.dims[3] == 4)))
        return false;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
        return false;
    // A repacked Linear weight may have its own physical layout; only the fusion
    // requires ordinary dense row-major weights. The other tensors are dense in
    // both branches. Linear vector-loads its input and weights, so their base
    // pointers must be 16-byte aligned even when the fusion is disabled.
    if (!DenseCuda(input, device, 16) || !DenseCuda(cw, device, 4) || !DenseCuda(cache, device, 2) ||
        !Bias(bias, weight.dims[0], device) || !Bias(cb, channels, device) ||
        weight.dataDevice != DataDevice::CUDA || !weight.cudaData || weight.multiDeviceData ||
        reinterpret_cast<uintptr_t>(weight.cudaData) % 16 ||
        (!weight.dataDeviceIds.empty() &&
         (weight.dataDeviceIds.size() != 1 || weight.dataDeviceIds[0] != device)))
        return false;
    return !slots || (slots->dataType == DataType::INT32 && slots->Count(0) >= uint64_t(batch) &&
                      DenseCuda(*slots, device, 4));
}

bool FastllmCudaGdnInputConvCanRun(const Data &input, const Data &weight, const Data &bias, const Data &cw,
                                   const Data &cb, const Data &cache, const Data *slots, int batch) {
    const char *enabled = std::getenv("FASTLLM_CUDA_GDN_INPUT_CONV");
    if (enabled && (!std::strcmp(enabled, "0") || !std::strcmp(enabled, "false")))
        return false;
    if (batch > 8 || !FastllmCudaGdnInputConvValidInputs(input, weight, bias, cw, cb, cache, slots, batch) ||
        weight.dataType != DataType::FP8_E4M3 || weight.dims[1] < 256 || weight.dims[1] > 32768 ||
        weight.dims[1] % 256 != 0 || weight.dims[0] > 65536 ||
        weight.blockK != 1 || weight.blockM != weight.dims[1] || weight.IsRepacked ||
        weight.scales.size() != size_t(weight.dims[0]))
        return false;
    const char *generic = std::getenv("FASTLLM_CUDA_TP_FUSIONS");
    if (generic && (!std::strcmp(generic, "0") || !std::strcmp(generic, "false")) &&
        (weight.dims != std::vector<int>({16384, 5120}) || cw.dims[0] != 10240))
        return false;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
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
        // All dispatch specializations live in this translation unit and use
        // the same architecture list. Querying one verifies a loadable image,
        // including builds whose CUDA_ARCH does not cover this device.
        cudaFuncAttributes attributes{};
        cudaError_t status = cudaFuncGetAttributes(&attributes, gdn::InputConvKernel<half, 1, 8, 2, 8, 4, true>);
        if (status != cudaSuccess) {
            // An unsupported image is a capability miss, not a failed launch.
            cudaGetLastError();
        }
        it = supported.emplace(device, status == cudaSuccess && attributes.maxThreadsPerBlock >= 256).first;
    }
    if (!it->second || !DenseCuda(input, device) || !DenseCuda(weight, device) ||
        !DenseCuda(cw, device, 16) || !DenseCuda(cache, device))
        return false;
    // No tensor contents are copied: callers own valid, distinct device-side slot IDs.
    cudaStreamCaptureStatus capture;
    if (cudaStreamIsCapturing(cudaStreamPerThread, &capture) != cudaSuccess)
        return false;
    return capture == cudaStreamCaptureStatusNone ||
           (!weight.extraCudaData.empty() && weight.extraCudaData[0]);
}
void FastllmCudaGdnInputConv(Data &input, Data &weight, const Data &bias, const Data &cw, const Data &cb,
                             Data &cache, const Data *slots, Data &out, Data &z, int batch) {
    FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(weight, bias, weight.dims[0]);
    if (weight.dims != std::vector<int>({16384, 5120}) || cw.dims[0] != 10240) {
        if (input.dataType == DataType::FLOAT16)
            Dispatch<half, true>(input, weight, bias, cw, cb, cache, slots, out, z, batch);
        else
            Dispatch<__nv_bfloat16, true>(input, weight, bias, cw, cb, cache, slots, out, z, batch);
    } else if (input.dataType == DataType::FLOAT16)
        Dispatch<half>(input, weight, bias, cw, cb, cache, slots, out, z, batch);
    else
        Dispatch<__nv_bfloat16>(input, weight, bias, cw, cb, cache, slots, out, z, batch);
    AssertInFastLLM(cudaGetLastError() == cudaSuccess, "GDN input/conv CUDA launch failed.\n");
}
void FastllmCudaGdnProjectedConv(const Data &p, const Data &cw, const Data &cb, Data &cache,
                                 const Data *slots, Data &out, Data &z, int batch) {
    int channels = cw.dims[0], width = p.dims.back(), blocks = (batch * width + 255) / 256;
#define LAUNCH(T)                                                                                            \
    ProjectedConv<T><<<blocks, 256, 0, cudaStreamPerThread>>>(                                               \
        (const T *)p.cudaData, (const float *)cw.cudaData,                                                   \
        cb.dims.empty() ? nullptr : (const float *)cb.cudaData, (T *)cache.cudaData,                         \
        slots ? (const int *)slots->cudaData : nullptr, (T *)out.cudaData, (T *)z.cudaData, batch, channels, \
        width)
    if (p.dataType == DataType::FLOAT16) {
        LAUNCH(half);
    } else {
        LAUNCH(__nv_bfloat16);
    }
#undef LAUNCH
    AssertInFastLLM(cudaGetLastError() == cudaSuccess, "GDN projected conv CUDA launch failed.\n");
}
} // namespace fastllm
