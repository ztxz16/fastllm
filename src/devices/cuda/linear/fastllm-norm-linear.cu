#include "devices/cuda/fastllm-cuda-norm-linear.h"
#include <limits>

namespace fastllm {
namespace {
bool Dense(const Data &d, int device) {
    if (d.dataDevice != DataDevice::CUDA || !d.cudaData || d.multiDeviceData || d.IsRepacked ||
        d.dataDeviceIds != std::vector<int>{device} || d.dims.empty() ||
        d.strides.size() != d.dims.size()) return false;
    uint64_t stride = 1;
    for (int i = int(d.dims.size()) - 1; i >= 0; --i) {
        if (d.dims[i] <= 0 || d.strides[i] != stride ||
            stride > std::numeric_limits<uint64_t>::max() / d.dims[i]) return false;
        stride *= d.dims[i];
    }
    return true;
}
bool Bind(const Data &x, const Data *branch, const Data &g, const Data &w, const Data &out,
          int device, cuda::low_latency::NormLinearShape &shape,
          cuda::low_latency::NormLinearBindings &bindings) {
    if (x.dataType != BFLOAT16 || w.dataType != BFLOAT16 || g.dataType != FLOAT32 ||
        out.dataType != FLOAT32 || w.dims.size() != 2 || x.dims.empty() ||
        x.dims.back() != w.dims[1] || g.dims != std::vector<int>{w.dims[1]} ||
        !Dense(x,device) || !Dense(w,device) || !Dense(g,device) || !Dense(out,device) ||
        (branch && (branch->dataType != BFLOAT16 || branch->dims != x.dims || !Dense(*branch,device))))
        return false;
    auto expected = x.dims; expected.back() = w.dims[0];
    if (out.dims != expected) return false;
    const uint64_t rows = x.Count(0) / w.dims[1];
    if (rows > std::numeric_limits<int>::max()) return false;
    const int inner = w.dims[1], channels = w.dims[0];
    shape = {int(rows),inner,channels,inner,inner,inner,channels};
    bindings.input = {x.cudaData,size_t(x.GetBytes())};
    bindings.gamma = {g.cudaData,size_t(g.GetBytes())};
    bindings.weight = {w.cudaData,size_t(w.GetBytes())};
    bindings.output = {out.cudaData,size_t(out.GetBytes())};
    if (branch) bindings.branch = {branch->cudaData,size_t(branch->GetBytes())};
    return true;
}
} // namespace

cudaError_t FastllmCudaPrepareNormLinear(const Data &x, const Data *branch, const Data &g,
    const Data &w, const Data &out, cuda::low_latency::NormLinearPlan &plan, cudaStream_t stream) {
    int device;
    auto error = cudaGetDevice(&device); if (error != cudaSuccess) return error;
    cuda::low_latency::NormLinearShape shape;
    cuda::low_latency::NormLinearBindings bindings;
    if (!Bind(x,branch,g,w,out,device,shape,bindings)) return cudaErrorNotSupported;
    cuda::low_latency::NormLinearPlan prepared;
    error = cuda::low_latency::PrepareNormLinear(shape,false,prepared,stream);
    if (error != cudaSuccess) return error;
    if (!cuda::low_latency::CanRunNormLinear(prepared,bindings)) return cudaErrorInvalidValue;
    plan = prepared;
    return cudaSuccess;
}

cudaError_t FastllmCudaRunNormLinear(const cuda::low_latency::NormLinearPlan &plan,
    const Data &x, const Data *branch, const Data &g, const Data &w, Data &out,
    float epsilon, cudaStream_t stream) {
    int device;
    auto error = cudaGetDevice(&device); if (error != cudaSuccess) return error;
    cuda::low_latency::NormLinearShape shape;
    cuda::low_latency::NormLinearBindings bindings;
    if (!Bind(x,branch,g,w,out,device,shape,bindings)) return cudaErrorInvalidValue;
    const auto &prepared = plan.Shape();
    if (prepared.rows != shape.rows || prepared.inner != shape.inner || prepared.channels != shape.channels ||
        prepared.inputStride != shape.inputStride || prepared.branchStride != shape.branchStride ||
        prepared.weightStride != shape.weightStride || prepared.outputStride != shape.outputStride)
        return cudaErrorInvalidValue;
    bindings.epsilon = epsilon;
    return cuda::low_latency::RunNormLinear(plan,bindings,stream);
}
} // namespace fastllm
