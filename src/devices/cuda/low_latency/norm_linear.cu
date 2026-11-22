#include "devices/cuda/low_latency/norm_linear.h"
#include "norm_linear.cuh"
#include <algorithm>
#include <cmath>
#include <limits>

namespace fastllm::cuda::low_latency {
namespace {
bool Span(int64_t rows, int64_t width, int64_t stride, size_t unit, size_t &bytes) {
    if (rows < 1 || width < 1 || stride < width) return false;
    const uint64_t limit = std::numeric_limits<int64_t>::max() / unit;
    if (uint64_t(width) > limit || uint64_t(rows - 1) > (limit - width) / stride) return false;
    bytes = (uint64_t(rows - 1) * stride + width) * unit;
    return true;
}
bool Fits(const void *ptr, size_t capacity, size_t bytes, size_t alignment) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
    return ptr && address % alignment == 0 && capacity >= bytes &&
           bytes <= std::numeric_limits<uintptr_t>::max() - address;
}
bool Overlap(const void *a, size_t na, const void *b, size_t nb) {
    if (!a || !b) return false;
    const uintptr_t x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    return x <= y ? y - x < na : x - y < nb;
}
template<bool Mirror, bool Dense> const void *Kernel(int tile) {
#define TILE(N) case N: return reinterpret_cast<const void *>(detail::NormLinearSimt<N, Mirror, Dense>);
    switch (tile) { TILE(1) TILE(2) TILE(3) TILE(4) TILE(5) TILE(6) TILE(7) TILE(8) }
#undef TILE
    return nullptr;
}
} // namespace

bool SupportsNormLinearShape(const NormLinearShape &s) {
    // The vector-load specialization requires K%256; callers retain their
    // original operation for other widths. N and logical rows may have tails.
    if (s.rows < 1 || s.rows > 65535 * 8 || s.inner < 256 || s.inner > 8192 ||
        s.inner % 256 || s.channels < 1 || s.weightStride % 8) return false;
    size_t bytes = 0;
    return Span(s.rows, s.inner, s.inputStride, 2, bytes) &&
           Span(s.rows, s.inner, s.branchStride, 2, bytes) &&
           Span(s.channels, s.inner, s.weightStride, 2, bytes) &&
           Span(s.rows, s.channels, s.outputStride, 4, bytes);
}

cudaError_t PrepareNormLinear(const NormLinearShape &s, bool mirror, NormLinearPlan &out,
                             cudaStream_t stream, PrepareOptions options) {
    if (!SupportsNormLinearShape(s)) return cudaErrorNotSupported;
    cudaStreamCaptureStatus capturing;
    cudaError_t error = cudaStreamIsCapturing(stream, &capturing);
    if (error != cudaSuccess) return error;
    if (capturing != cudaStreamCaptureStatusNone) return cudaErrorStreamCaptureUnsupported;
    NormLinearPlan plan;
    if ((error = cudaGetDevice(&plan.device_)) != cudaSuccess) return error;
    cudaDeviceProp properties{};
    if ((error = cudaGetDeviceProperties(&properties, plan.device_)) != cudaSuccess) return error;
    // This is a SIMT kernel: BF16 storage/conversion does not require BF16 MMA.
    if (properties.major < 7) return cudaErrorNotSupported;
    const size_t sharedLimit = std::min(options.sharedMemoryLimit, properties.sharedMemPerBlockOptin);
    cudaFuncAttributes attributes{};
    for (plan.tile_ = std::min(s.rows, 8); plan.tile_ > 0; --plan.tile_) {
        // The dense variant omits grid.y addressing, so it is valid only when
        // the entire logical batch fits in the selected physical row tile.
        const bool dense = s.inputStride == s.inner && s.branchStride == s.inner &&
                           s.weightStride == s.inner && s.outputStride == s.channels &&
                           s.rows <= plan.tile_ &&
                           s.channels <= std::numeric_limits<int>::max() - properties.multiProcessorCount * 64;
        plan.kernel_ = mirror ? (dense ? Kernel<true,true>(plan.tile_) : Kernel<true,false>(plan.tile_))
                              : (dense ? Kernel<false,true>(plan.tile_) : Kernel<false,false>(plan.tile_));
        if ((error = cudaFuncGetAttributes(&attributes, plan.kernel_)) != cudaSuccess) return error;
        if (attributes.maxThreadsPerBlock < 256 || attributes.ptxVersion < 70) return cudaErrorNotSupported;
        plan.shared_ = size_t(plan.tile_) * s.inner * 2;
        if (plan.shared_ + attributes.sharedSizeBytes <= sharedLimit) break;
    }
    if (!plan.tile_ || (s.rows + plan.tile_ - 1) / plan.tile_ > properties.maxGridSize[1])
        return cudaErrorNotSupported;
    if (plan.shared_ + attributes.sharedSizeBytes > properties.sharedMemPerBlock) {
        // Fixed per-image ceiling: preparing a smaller K cannot shrink the
        // shared-memory allowance of graphs already captured for a larger K.
        const int ceiling = std::min(size_t(plan.tile_) * 8192 * 2,
                                    properties.sharedMemPerBlockOptin - attributes.sharedSizeBytes);
        if ((error = cudaFuncSetAttribute(plan.kernel_, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         ceiling)) != cudaSuccess) return error;
    }
    plan.blocks_ = int(std::min((int64_t(s.channels) + 15) / 16,
                               int64_t(properties.multiProcessorCount) * 4));
    plan.shape_ = s;
    plan.mirror_ = mirror;
    out = plan;
    return cudaSuccess;
}

bool CanRunNormLinear(const NormLinearPlan &p, const NormLinearBindings &b) {
    if (!p.Ready() || !std::isfinite(b.epsilon) || b.epsilon <= 0) return false;
    const auto &s = p.shape_;
    size_t input, branch, weight, output, mirror = 0;
    if (!Span(s.rows,s.inner,s.inputStride,2,input) ||
        !Span(s.rows,s.inner,s.branchStride,2,branch) ||
        !Span(s.channels,s.inner,s.weightStride,2,weight) ||
        !Span(s.rows,s.channels,s.outputStride,4,output)) return false;
    const size_t gamma = size_t(s.inner) * 4;
    if (!Fits(b.input.data,b.input.bytes,input,2) || !Fits(b.gamma.data,b.gamma.bytes,gamma,4) ||
        !Fits(b.weight.data,b.weight.bytes,weight,16) || !Fits(b.output.data,b.output.bytes,output,4) ||
        (b.branch.data && !Fits(b.branch.data,b.branch.bytes,branch,2))) return false;
    if (p.mirror_ != bool(b.mirror.data)) return false;
    if (p.mirror_ && (!Span(s.rows,s.channels,b.mirrorStride,4,mirror) ||
                     !Fits(b.mirror.data,b.mirror.bytes,mirror,4))) return false;
    const ConstBuffer inputs[] = {{b.input.data,input},{b.gamma.data,gamma},
                                 {b.weight.data,weight},{b.branch.data,branch}};
    for (const auto &in : inputs) {
        if (Overlap(in.data,in.bytes,b.output.data,output) ||
            Overlap(in.data,in.bytes,b.mirror.data,mirror)) return false;
    }
    return !Overlap(b.output.data,output,b.mirror.data,mirror);
}

cudaError_t RunNormLinear(const NormLinearPlan &p, const NormLinearBindings &b, cudaStream_t stream) {
    if (!CanRunNormLinear(p,b)) return cudaErrorInvalidValue;
    int device = -1;
    cudaError_t error = cudaGetDevice(&device);
    if (error != cudaSuccess) return error;
    if (device != p.device_) return cudaErrorInvalidDevice;
    const auto &s = p.shape_;
    const void *input = b.input.data, *branch = b.branch.data, *gamma = b.gamma.data, *weight = b.weight.data;
    void *output = b.output.data, *mirror = b.mirror.data;
    int inner = s.inner, channels = s.channels, rows = s.rows;
    float epsilon = b.epsilon;
    int64_t mirrorStride = b.mirrorStride, inputStride = s.inputStride, branchStride = s.branchStride,
            weightStride = s.weightStride, outputStride = s.outputStride;
    void *args[] = {&input,&branch,&gamma,&weight,&output,&inner,&channels,&epsilon,&mirror,
                   &mirrorStride,&rows,&inputStride,&branchStride,&weightStride,&outputStride};
    return cudaLaunchKernel(p.kernel_, dim3(p.blocks_,(s.rows+p.tile_-1)/p.tile_), dim3(256),
                            args,p.shared_,stream);
}
} // namespace fastllm::cuda::low_latency
