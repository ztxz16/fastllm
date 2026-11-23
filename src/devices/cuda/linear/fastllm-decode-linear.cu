#include "devices/cuda/fastllm-cuda-decode-linear.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/low_latency/norm_linear.h"
#include "../low_latency/data_layout.h"
#include <tuple>
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <map>

namespace fastllm {
namespace {
using cuda::low_latency::DenseData;
using cuda::low_latency::DataOverlap;

// Check the address from the producing device. Peer capability alone does
// not tell us whether this allocation is currently accessible. Graph replay
// uses the captured binding; the owner keeps it accessible until completion.
bool CanMirror(int source, const Data &mirror) {
    if (source == mirror.dataDeviceIds[0]) return true;
    cudaPointerAttributes attributes{};
    const auto error = cudaPointerGetAttributes(&attributes, mirror.cudaData);
    AssertInFastLLM(error == cudaSuccess, cudaGetErrorString(error));
    return attributes.devicePointer == mirror.cudaData;
}

template<bool Mirror = false>
bool Dispatch(const Data &input, const Data *branch, const Data &gamma,
        const Data &weight, const Data &output, float eps, int device, bool launch,
        const Data *mirror = nullptr, int mirrorOffset = 0) {
    namespace ll = cuda::low_latency;
    const int inner = weight.dims[1], channels = weight.dims[0];
    const int rows = input.Count(0) / inner;
    // Plans contain no tensor addresses. Preparing a new shape is restricted
    // to startup/warmup; request-time capture cannot mutate kernel resources.
    using Key = std::tuple<int, int, int, int>;
    static thread_local std::map<Key, ll::NormLinearPlan> plans;
    const Key key{device, rows, inner, channels};
    auto found = plans.find(key);
    if (found == plans.end()) {
        if (FastllmCudaGraphIsCapturingFast()) return false;
        ll::NormLinearPlan plan;
        const ll::NormLinearShape shape{rows, inner, channels, inner, inner, inner, channels};
        const auto error = ll::PrepareNormLinear(shape, Mirror, plan, cudaStreamPerThread);
        if (error == cudaErrorStreamCaptureUnsupported) return false; // Transient: never cache.
        if (ll::BackendUnavailable(error)) {
            const auto pending = cudaGetLastError();
            AssertInFastLLM(pending == cudaSuccess || ll::BackendUnavailable(pending),
                           "Pending CUDA failure during norm/projection admission.");
            plans.emplace(key, plan);
            return false;
        }
        AssertInFastLLM(error == cudaSuccess, "BF16 norm/projection preparation failed.");
        found = plans.emplace(key, plan).first;
    }
    if (!found->second.Ready()) return false;
    if (!launch) return true;
    ll::NormLinearBindings bindings;
    bindings.input = {input.cudaData, input.GetBytes()};
    bindings.gamma = {gamma.cudaData, gamma.GetBytes()};
    bindings.weight = {weight.cudaData, weight.GetBytes()};
    if (branch) bindings.branch = {branch->cudaData, branch->GetBytes()};
    bindings.output = {output.cudaData, output.GetBytes()};
    bindings.epsilon = eps;
    if constexpr (Mirror) {
        bindings.mirror = {(float *)mirror->cudaData + mirrorOffset,
            mirror->GetBytes() - size_t(mirrorOffset) * sizeof(float)};
        bindings.mirrorStride = mirror->dims.back();
    }
    const auto error = ll::RunNormLinear(found->second, bindings, cudaStreamPerThread);
    AssertInFastLLM(error == cudaSuccess, "BF16 norm/projection launch failed.");
    return true;
}
} // namespace

bool FastllmCudaDecodeNormLinearCanRun(const Data &input, const Data &norm,
        const Data &weight, const Data &output) {
    if (input.dataType != BFLOAT16 || weight.dataType != BFLOAT16 || norm.dataType != FLOAT32 ||
        weight.dims.size() != 2 || input.dims.empty() || input.dims.back() != weight.dims[1] ||
        weight.dims[1] < 256 || weight.dims[1] > 8192 || weight.dims[1] % 256 ||
        weight.dims[0] < 16 || weight.dims[0] % 16 ||
        norm.dims != std::vector<int>{weight.dims[1]}) return false;
    const int inner = weight.dims[1], channels = weight.dims[0];
    const uint64_t rows = input.Count(0) / inner;
    int device = 0;
    AssertInFastLLM(cudaGetDevice(&device) == cudaSuccess, "Cannot query norm/projection device.");
    if (rows < 1 || rows > 65535u * 8 ||
        !DenseData(input, device) || !DenseData(weight, device) || !DenseData(norm, device) ||
        output.multiDeviceData || DataOverlap(output, input) || DataOverlap(output, weight) || DataOverlap(output, norm) ||
        (output.cudaData && output.dataDeviceIds != std::vector<int>{device})) return false;
    if ((output.isFake || output.cudaDataBorrowed) &&
        (output.dataType != FLOAT32 || !DenseData(output, device) || output.Count(0) != rows * channels)) return false;
    return Dispatch(input, nullptr, norm, weight, output, 0, device, false);
}

bool FastllmCudaDecodeNormLinearPreferred(const Data &input, const Data &norm,
        const Data &weight, const Data &output) {
    if (input.dims.empty() || weight.dims.size()!=2 || weight.dims[1]<=0) return false;
    const auto rows=input.Count(0)/weight.dims[1];
    if (rows<1 || rows>7) return false;
    // A performance choice shared across architectures, separate from raw
    // capability. Preserve the measured B300 wide-head Lt path. Hopper also
    // keeps Lt for wide multi-row heads; narrow row tiles can use the fusion.
    if (rows>=3 && weight.dims[1]>=4096 && weight.dims[0]>=16384) return false;
    return FastllmCudaDecodeNormLinearCanRun(input,norm,weight,output);
}

bool FastllmCudaDecodeNormLinear(const Data &input, const Data *branch, const Data &norm,
        const Data &weight, Data &output, float eps, const Data *mirror, int mirrorOffset) {
    if (!std::isfinite(eps) || eps <= 0 || !FastllmCudaDecodeNormLinearPreferred(input, norm, weight, output))
        return false;
    int device = 0;
    cudaGetDevice(&device);
    if (branch && (branch->dataType != BFLOAT16 || branch->dims != input.dims ||
        !DenseData(*branch, device) || DataOverlap(output, *branch))) return false;
    if (mirror) {
        if (mirror->dataType != FLOAT32 || mirror->dims.size() != 2 || mirror->dataDeviceIds.size() != 1 ||
            !DenseData(*mirror, mirror->dataDeviceIds[0]) || !CanMirror(device, *mirror) || mirrorOffset < 0 ||
            mirrorOffset > mirror->dims[1] - weight.dims[0] ||
            mirror->dims[0] < input.Count(0) / weight.dims[1] || DataOverlap(*mirror, input) ||
            DataOverlap(*mirror, weight) || DataOverlap(*mirror, norm) || DataOverlap(*mirror, output) ||
            (branch && DataOverlap(*mirror, *branch)) ||
            !Dispatch<true>(input, branch, norm, weight, output, eps, device, false, mirror, mirrorOffset)) return false;
    }
    auto shape = input.dims;
    shape.back() = weight.dims[0];
    output.dataType = FLOAT32;
    output.UpdateUnitSize();
    output.Resize(shape);
    output.ToDevice(input.dataDevice, input.dataDeviceIds, false);
    output.Allocate(false);
    if (mirror) return Dispatch<true>(input, branch, norm, weight, output, eps, device, true, mirror, mirrorOffset);
    return Dispatch(input, branch, norm, weight, output, eps, device, true);
}
} // namespace fastllm
