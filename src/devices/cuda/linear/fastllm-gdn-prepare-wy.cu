#include "devices/cuda/fastllm-cuda-gdn-prepare.h"
#include "fastllm-gdn-prepare-wy.cuh"
#include "fastllm.h"
#include "fastllm-native-prefill-policy.cuh"
#include <climits>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>

namespace fastllm {
namespace {
bool Dense(const Data &d, int device, size_t alignment) {
    if (d.dataType != DataType::FLOAT16 || d.dataDevice != DataDevice::CUDA || !d.cudaData ||
        d.multiDeviceData || d.dims.empty() || uintptr_t(d.cudaData) % alignment ||
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
bool Overlap(const Data &a, const Data &b) {
    auto ap = uintptr_t(a.cudaData), bp = uintptr_t(b.cudaData);
    return ap < bp + b.GetBytes() && bp < ap + a.GetBytes();
}
bool ExactOrDisjoint(const Data &a, const Data &b) {
    return !Overlap(a, b) || (a.cudaData == b.cudaData && a.GetBytes() == b.GetBytes());
}
bool BaseCanRun(const Data &v, const Data &k, const Data &g, int &device) {
    const char *enabled = std::getenv("FASTLLM_CUDA_GDN_PREPARE_WY");
    // The Triton route owns separate cached decay scales; leave its lifecycle
    // entirely with the existing path instead of reusing stale scale metadata.
    if ((enabled && !fastllm_native_prefill::ResolvePrefillSwitch(enabled, false)) || GetFastllmEnv().cudaTriton)
        return false;
    cudaStreamCaptureStatus capture;
    if (cudaStreamIsCapturing(cudaStreamPerThread, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone)
        return false;
    int major = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess || major < 8)
        return false;
    if (!Dense(v, device, 32) || !Dense(k, device, 32) || !Dense(g, device, 2) || v.dims.size() != 5 ||
        k.dims != v.dims || g.dims.size() != 4 || v.dims[3] != 64 || v.dims[4] != 128)
        return false;
    if (!enabled) {
        int minor = 0;
        if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess ||
            !fastllm_native_prefill::AutomaticGdnPrepare(major, minor, v.dims[0], v.dims[1],
                                                        v.dims[2], v.dims[3], v.dims[4]))
            return false;
    }
    auto shape = v.dims;
    shape.pop_back();
    if (g.dims != shape || v.Count(0) / 8192 > INT_MAX)
        return false;
    static thread_local std::map<int, bool> available;
    auto it = available.find(device);
    if (it == available.end()) {
        cudaFuncAttributes attr{}, keyAttr{};
        auto error = cudaFuncGetAttributes(&attr, fastllm_gdn_wy::Prepare<8>);
        auto keyError = cudaFuncGetAttributes(&keyAttr, fastllm_gdn_wy::Prepare<8, true>);
        if (error != cudaSuccess || keyError != cudaSuccess)
            cudaGetLastError();
        it = available
                 .emplace(device, error == cudaSuccess && keyError == cudaSuccess &&
                                      attr.binaryVersion >= 80 && keyAttr.binaryVersion >= 80 &&
                                      attr.maxThreadsPerBlock >= 512 && keyAttr.maxThreadsPerBlock >= 512)
                 .first;
    }
    return it->second;
}
void PrepareOutput(Data &output, const Data &reference, const std::vector<int> &shape) {
    output.dataType = DataType::FLOAT16;
    output.dataDevice = DataDevice::CUDA;
    output.dataDeviceIds = reference.dataDeviceIds;
    output.Resize(shape);
    output.Allocate(false);
}
bool OutputsValid(const Data &v, const Data &k, const Data &g, const Data &vo, const Data &ko, int device) {
    return Dense(vo, device, 32) && Dense(ko, device, 32) && !Overlap(vo, ko) && !Overlap(vo, k) &&
           !Overlap(vo, g) && !Overlap(ko, g) && ExactOrDisjoint(vo, v) && ExactOrDisjoint(ko, v) &&
           ExactOrDisjoint(ko, k);
}
void CheckLaunch() {
    auto error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("GDN prepare WY launch: ") + cudaGetErrorString(error));
}
} // namespace
bool FastllmCudaGdnPrepareWyCanRun(const Data &attn, const Data &v, const Data &k, const Data &g) {
    int device = 0;
    if (!BaseCanRun(v, k, g, device) || !Dense(attn, device, 2))
        return false;
    auto shape = v.dims;
    shape.back() = 64;
    return attn.dims == shape;
}
bool FastllmCudaTryGdnPrepareWy(const Data &attn, const Data &v, const Data &k, const Data &g, Data &vo,
                                Data &ko) {
    if (!FastllmCudaGdnPrepareWyCanRun(attn, v, k, g))
        return false;
    if (&vo == &ko || &vo == &attn || &vo == &g || &vo == &k || &ko == &attn || &ko == &g)
        return false;
    PrepareOutput(vo, v, v.dims);
    PrepareOutput(ko, k, k.dims);
    int device = 0;
    cudaGetDevice(&device);
    if (!OutputsValid(v, k, g, vo, ko, device) || Overlap(vo, attn) || Overlap(ko, attn))
        return false;
    fastllm_gdn_wy::Prepare<8><<<v.Count(0) / 8192, 512, 43136>>>(
        static_cast<const half *>(attn.cudaData), static_cast<const half *>(v.cudaData),
        static_cast<const half *>(k.cudaData), static_cast<const half *>(g.cudaData),
        static_cast<half *>(vo.cudaData), static_cast<half *>(ko.cudaData));
    CheckLaunch();
    return true;
}
bool FastllmCudaGdnPrepareFromKeyCanRun(const Data &key, const Data &v, const Data &k, const Data &g) {
    int device = 0;
    return BaseCanRun(v, k, g, device) && Dense(key, device, 32) && key.dims == k.dims && !Overlap(g, key) &&
           !Overlap(g, k) && !Overlap(g, v);
}
bool FastllmCudaTryGdnPrepareFromKey(const Data &key, const Data &v, const Data &k, Data &g, Data &decay,
                                     Data &vo, Data &ko) {
    if (!FastllmCudaGdnPrepareFromKeyCanRun(key, v, k, g))
        return false;
    if (&vo == &ko || &vo == &key || &vo == &g || &vo == &k || &ko == &key || &ko == &g || &decay == &key ||
        &decay == &v || &decay == &k || &decay == &g || &decay == &vo || &decay == &ko)
        return false;
    PrepareOutput(vo, v, v.dims);
    PrepareOutput(ko, k, k.dims);
    auto shape = g.dims;
    shape.push_back(64);
    PrepareOutput(decay, g, shape);
    int device = 0;
    cudaGetDevice(&device);
    if (!OutputsValid(v, k, g, vo, ko, device) || !Dense(decay, device, 2) || Overlap(vo, key) ||
        Overlap(ko, key) || Overlap(decay, key) || Overlap(decay, v) || Overlap(decay, k) ||
        Overlap(decay, g) || Overlap(decay, vo) || Overlap(decay, ko))
        return false;
    fastllm_gdn_wy::Prepare<8, true><<<v.Count(0) / 8192, 512, 43136>>>(
        nullptr, static_cast<const half *>(v.cudaData), static_cast<const half *>(k.cudaData),
        static_cast<const half *>(g.cudaData), static_cast<half *>(vo.cudaData),
        static_cast<half *>(ko.cudaData), static_cast<const half *>(key.cudaData),
        static_cast<half *>(g.cudaData), static_cast<half *>(decay.cudaData));
    CheckLaunch();
    return true;
}
} // namespace fastllm
