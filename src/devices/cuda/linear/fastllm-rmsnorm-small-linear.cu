#include "devices/cuda/fastllm-cuda-rmsnorm-small-linear.h"
#include "devices/cuda/fastllm-rmsnorm-small-linear.cuh"
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
    if ((x.dataType != FLOAT16 && x.dataType != BFLOAT16) || w.dataType != x.dataType ||
        g.dataType != FLOAT32 || w.IsRepacked || w.dims.size() != 2 || x.dims.empty())
        return false;
    int D = w.dims[1], N = w.dims[0];
    if (D < 1024 || D > 8192 || D % 1024 || N < 1 || N > 256 || x.dims.back() != D || x.Count(0) % D ||
        x.Count(0) / D < 1 || x.Count(0) / D > 8 || g.dims != std::vector<int>{D})
        return false;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || !Dense(x, dev, 4) || !Dense(g, dev, 8) || !Dense(w, dev, 4) ||
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
    return x.dataType == FLOAT16 ? Available<half>(D) : Available<__nv_bfloat16>(D);
}
void FastllmCudaRMSNormSmallLinear(const Data &x, const Data &g, const Data &w, const Data &b, Data &y,
                                   Data &o, float eps) {
    if (x.dataType == FLOAT16)
        Launch<half>(x, g, w, b, y, o, eps);
    else
        Launch<__nv_bfloat16>(x, g, w, b, y, o, eps);
    AssertInFastLLM(cudaGetLastError() == cudaSuccess, "RMSNormSmallLinear launch failed.\n");
}
} // namespace fastllm
