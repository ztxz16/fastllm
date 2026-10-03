#include "devices/cuda/fastllm-cuda-rmsnorm-small-linear.h"
#include "devices/cuda/fastllm-rmsnorm-small-linear.cuh"
#include "devices/cuda/fastllm-rmsnorm-small-linear-mixed.cuh"
#include "gguf.h"
#include "fastllm-cuda.cuh"
#include "utils.h"
#include <map>
#include <cstdlib>
#include <cstring>
namespace fastllm {
namespace {
bool Dense(const Data &x, int dev, size_t align) {
    if (x.dataDevice != DataDevice::CUDA || !x.cudaData || x.multiDeviceData ||
        reinterpret_cast<uintptr_t>(x.cudaData) % align || x.dims.empty() ||
        x.strides.size() != x.dims.size() ||
        (!x.dataDeviceIds.empty() && (x.dataDeviceIds.size() != 1 || x.dataDeviceIds[0] != dev)))
        return false;
    uint64_t stride = 1;
    for (int i = int(x.dims.size()) - 1; i >= 0; --i) {
        if (x.dims[i] <= 0 || x.strides[i] != stride)
            return false;
        stride *= x.dims[i];
    }
    return true;
}
bool MixedWeight(const Data &x, const Data &w) {
    return x.dataType == FLOAT16 && (w.dataType == BFLOAT16 ||
        (w.dataType == DATA_GGUF_FORMAT && w.ggmlType == GGML_TYPE_BF16));
}
bool DenseMixedWeight(const Data &w, int dev) {
    if (w.dataType == BFLOAT16) return Dense(w, dev, 16);
    if (w.dataDevice != DataDevice::CUDA || !w.cudaData || w.multiDeviceData ||
        reinterpret_cast<uintptr_t>(w.cudaData)%16 ||
        (!w.dataDeviceIds.empty() && (w.dataDeviceIds.size() != 1 || w.dataDeviceIds[0] != dev))) return false;
    const auto *t = static_cast<const ggml_tensor *>(w.ggmlTensor);
    return t && t->type == GGML_TYPE_BF16 && t->ne[0] == w.dims[1] && t->ne[1] == w.dims[0] &&
        t->nb[0] == 2 && t->nb[1] == size_t(w.dims[1])*2;
}
bool MixedAvailable() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    static thread_local std::map<int, bool> cache;
    auto it = cache.find(dev);
    if (it == cache.end()) {
        cudaFuncAttributes a{};
        const auto status = cudaFuncGetAttributes(&a, rmssmall::Mixed5120<false>);
        if (status != cudaSuccess) cudaGetLastError();
        it = cache.emplace(dev, status == cudaSuccess && a.maxThreadsPerBlock >= 512).first;
    }
    return it->second;
}
template <class T, int D> bool Available() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess)
        return false;
    static thread_local std::map<int, bool> cache;
    auto it = cache.find(dev);
    if (it == cache.end()) {
        cudaFuncAttributes a{};
        auto e = cudaFuncGetAttributes(&a, rmssmall::Kernel<T, D>);
        if (e != cudaSuccess)
            cudaGetLastError();
        it = cache.emplace(dev, e == cudaSuccess && a.maxThreadsPerBlock >= 256).first;
    }
    return it->second;
}
template <class T> bool Available(int D) {
#define CHECK(DIM)                                                                                           \
    case DIM:                                                                                                \
        return Available<T, DIM>();
    switch (D) {
        CHECK(1024) CHECK(2048) CHECK(3072) CHECK(4096) CHECK(5120) CHECK(6144) CHECK(7168) CHECK(8192)
    }
#undef CHECK
    return false;
}
template <class T>
void Launch(const Data &x, const Data &g, const Data &w, const Data &b, Data &y, Data &o, float eps) {
    int D = w.dims[1], N = w.dims[0], batch = x.Count(0) / D;
#define RUN(DIM)                                                                                             \
    case DIM:                                                                                                \
        rmssmall::Kernel<T, DIM><<<dim3((N + 1) / 2, batch), 256, 0, cudaStreamPerThread>>>(                 \
            (const T *)x.cudaData, (const float *)g.cudaData, (const T *)w.cudaData,                         \
            b.dims.empty() ? nullptr : (const float *)b.cudaData, (T *)y.cudaData, (T *)o.cudaData, N, eps); \
        break;
    switch (D) {
        RUN(1024) RUN(2048) RUN(3072) RUN(4096) RUN(5120) RUN(6144) RUN(7168) RUN(8192)
    default:
        AssertInFastLLM(false, "Unsupported RMSNormSmallLinear width.\n");
    }
#undef RUN
}
} // namespace
bool FastllmCudaRMSNormSmallLinearCanRun(const Data &x, const Data &g, const Data &w, const Data &b,
                                         const Data &y, const Data &o) {
    const char *flag = std::getenv("FASTLLM_CUDA_RMSNORM_SMALL_LINEAR");
    if (flag && (!std::strcmp(flag, "0") || !std::strcmp(flag, "false")))
        return false;
    const bool mixed = MixedWeight(x, w);
    if ((x.dataType != FLOAT16 && x.dataType != BFLOAT16) || (!mixed && w.dataType != x.dataType) ||
        g.dataType != FLOAT32 || w.IsRepacked || w.dims.size() != 2 || x.dims.empty())
        return false;
    int D = w.dims[1], N = w.dims[0];
    if (mixed && (D != 5120 || !b.dims.empty())) return false;
    if (D < 1024 || D > 8192 || D % 1024 || N < 1 || N > 256 || x.dims.back() != D || x.Count(0) % D ||
        x.Count(0) / D < 1 || x.Count(0) / D > 8 || g.dims != std::vector<int>{D})
        return false;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || !Dense(x, dev, mixed ? 16 : 4) || !Dense(g, dev, mixed ? 16 : 8) ||
        !(mixed ? DenseMixedWeight(w, dev) : Dense(w, dev, 4)) ||
        (!b.dims.empty() && (b.dataType != FLOAT32 || b.dims != std::vector<int>{N} || !Dense(b, dev, 4))))
        return false;
    // Reject aliases before Reshape/Allocate can change any caller-owned tensor.
    for (const Data *out : {&y, &o}) {
        if (out->multiDeviceData ||
            (out->cudaData && (!out->dataDeviceIds.empty() &&
                               (out->dataDeviceIds.size() != 1 || out->dataDeviceIds[0] != dev))))
            return false;
        for (const Data *in : {&x, &g, &w, &b}) {
            if (out == in)
                return false;
            if (out->cudaData && in->cudaData) {
                uintptr_t a = (uintptr_t)out->cudaData, c = (uintptr_t)in->cudaData;
                if (a < c + in->GetBytes() && c < a + out->GetBytes())
                    return false;
            }
        }
    }
    if (&y == &o)
        return false;
    if (y.cudaData && o.cudaData) {
        uintptr_t a = (uintptr_t)y.cudaData, c = (uintptr_t)o.cudaData;
        if (a < c + o.GetBytes() && c < a + y.GetBytes())
            return false;
    }
    return mixed ? MixedAvailable() : (x.dataType == FLOAT16 ? Available<half>(D) : Available<__nv_bfloat16>(D));
}
void FastllmCudaRMSNormSmallLinear(const Data &x, const Data &g, const Data &w, const Data &b, Data &y,
                                   Data &o, float eps) {
    if (MixedWeight(x, w)) {
        const bool bf16Dot = x.Count(0)/5120 == 8 && FastllmCudaGetLinearExactBatchThreshold() <= 8;
        auto kernel = bf16Dot ? rmssmall::Mixed5120<true> : rmssmall::Mixed5120<false>;
        kernel<<<dim3((w.dims[0]+1)/2, x.Count(0)/5120), 512, 0, cudaStreamPerThread>>>(
            static_cast<const half *>(x.cudaData), static_cast<const float *>(g.cudaData),
            static_cast<const __nv_bfloat16 *>(w.cudaData), static_cast<half *>(y.cudaData),
            static_cast<half *>(o.cudaData), w.dims[0], eps);
    } else if (x.dataType == FLOAT16)
        Launch<half>(x, g, w, b, y, o, eps);
    else
        Launch<__nv_bfloat16>(x, g, w, b, y, o, eps);
    AssertInFastLLM(cudaGetLastError() == cudaSuccess, "RMSNormSmallLinear launch failed.\n");
}
} // namespace fastllm
