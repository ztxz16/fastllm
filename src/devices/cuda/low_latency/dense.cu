#include "devices/cuda/low_latency/dense.h"
#include "devices/cuda/low_latency/dense.cuh"
#include <algorithm>
#include <cstdint>
#include <limits>

namespace fastllm::cuda::low_latency {
namespace {
template<int H, int I> void Kernels(int tile, const void *&gate, const void *&down) {
#define TILE(R) case R: gate = (const void *)DenseGateUp<H,I,false,R>; down = (const void *)DenseDown<I,H,false,R>; break;
    switch (tile) { TILE(8) TILE(4) TILE(2) TILE(1) }
#undef TILE
}
void Select(const DenseShape &s, int tile, const void *&gate, const void *&down) {
    if (s.hidden == 1024 && s.intermediate == 256) Kernels<1024,256>(tile,gate,down);
    if (s.hidden == 4096 && s.intermediate == 2048) Kernels<4096,2048>(tile,gate,down);
    if (s.hidden == 4096 && s.intermediate == 4096) Kernels<4096,4096>(tile,gate,down);
    if (s.hidden == 8192 && s.intermediate == 2048) Kernels<8192,2048>(tile,gate,down);
}
bool Fits(const void *p, size_t capacity, size_t needed) {
    const auto address = uintptr_t(p);
    return p && address % 16 == 0 && capacity >= needed && needed <= UINTPTR_MAX-address;
}
bool Overlap(ConstBuffer a, ConstBuffer b) {
    const auto x = uintptr_t(a.data), y = uintptr_t(b.data);
    return x <= y ? y-x < a.bytes : x-y < b.bytes;
}
} // namespace

bool SupportsDenseShape(const DenseShape &s) {
    return s.rows > 0 && s.rows <= 65535*8 &&
        ((s.hidden == 1024 && s.intermediate == 256) ||
         (s.hidden == 4096 && (s.intermediate == 2048 || s.intermediate == 4096)) ||
         (s.hidden == 8192 && s.intermediate == 2048));
}

cudaError_t PrepareDense(const DenseShape &s, DensePlan &out, cudaStream_t stream, PrepareOptions options) {
#ifdef CUDA_NO_TENSOR_CORE
    return cudaErrorNotSupported;
#endif
    if (!SupportsDenseShape(s)) return cudaErrorNotSupported;
    cudaStreamCaptureStatus capture;
    auto error = cudaStreamIsCapturing(stream,&capture);
    if (error != cudaSuccess) return error;
    if (capture != cudaStreamCaptureStatusNone) return cudaErrorStreamCaptureUnsupported;
    DensePlan p;
    if ((error = cudaGetDevice(&p.device_)) != cudaSuccess) return error;
    cudaDeviceProp properties{};
    if ((error = cudaGetDeviceProperties(&properties,p.device_)) != cudaSuccess) return error;
    if (properties.major < 8) return cudaErrorNotSupported;
    const size_t limit = std::min(options.sharedMemoryLimit,properties.sharedMemPerBlockOptin);
    cudaFuncAttributes gate{},down{};
    for (p.tile_ = 8; p.tile_ > 0; p.tile_ /= 2) {
        Select(s,p.tile_,p.gate_,p.down_);
        if ((error = cudaFuncGetAttributes(&gate,p.gate_)) != cudaSuccess) return error;
        if ((error = cudaFuncGetAttributes(&down,p.down_)) != cudaSuccess) return error;
        if (gate.binaryVersion < 80 || down.binaryVersion < 80 ||
            gate.ptxVersion < 80 || down.ptxVersion < 80 ||
            gate.maxThreadsPerBlock < 256 || down.maxThreadsPerBlock < 256) return cudaErrorNotSupported;
        p.gateShared_ = size_t(p.tile_)*s.hidden*2+8*128*4+32*p.tile_*4;
        p.downShared_ = size_t(p.tile_)*s.intermediate*2+8*128*4;
        if (p.gateShared_+gate.sharedSizeBytes <= limit && p.downShared_+down.sharedSizeBytes <= limit) break;
    }
    if (!p.tile_ || (s.rows+p.tile_-1)/p.tile_ > properties.maxGridSize[1]) return cudaErrorNotSupported;
    for (auto pair : {std::make_pair(p.gate_,p.gateShared_),std::make_pair(p.down_,p.downShared_)}) {
        // Size is fixed for a given kernel template, so subsequent preparations
        // cannot shrink an image's shared-memory allowance during graph replay.
        if (pair.second > properties.sharedMemPerBlock &&
            (error = cudaFuncSetAttribute(pair.first,cudaFuncAttributeMaxDynamicSharedMemorySize,pair.second)) != cudaSuccess)
            return error;
    }
    p.shape_ = s;
    out = p;
    return cudaSuccess;
}

bool CanRunDense(const DensePlan &p, const DenseBindings &b) {
    if (!p.Ready()) return false;
    const auto &s = p.shape_;
    const size_t x = size_t(s.rows)*s.hidden*2, w = size_t(s.hidden)*s.intermediate*2,
                 mid = size_t(s.rows)*s.intermediate*2;
    const ConstBuffer buffers[] = {b.input,b.gate,b.up,b.down,
        {b.intermediate.data,b.intermediate.bytes},{b.output.data,b.output.bytes}};
    const size_t required[] = {x,w,w,w,mid,x};
    for (int i=0; i<6; ++i) if (!Fits(buffers[i].data,buffers[i].bytes,required[i])) return false;
    // Inputs may share storage; either output must be disjoint from all live
    // inputs and from the other output. Validate everything before launch #1.
    for (int i=4; i<6; ++i) for (int j=0; j<i; ++j)
        if (Overlap({buffers[i].data,required[i]},{buffers[j].data,required[j]})) return false;
    return true;
}

cudaError_t RunDense(const DensePlan &p, const DenseBindings &b, cudaStream_t stream) {
    if (!CanRunDense(p,b)) return cudaErrorInvalidValue;
    int device = -1;
    auto error = cudaGetDevice(&device);
    if (error != cudaSuccess) return error;
    if (device != p.device_) return cudaErrorInvalidDevice;
    const void *input=b.input.data,*gate=b.gate.data,*up=b.up.data,*down=b.down.data,*gamma=nullptr,*residual=nullptr;
    void *mid=b.intermediate.data,*output=b.output.data;
    int rows=p.shape_.rows;
    float eps=0;
    void *gateArgs[]={&input,&gate,&up,&gamma,&eps,&mid,&rows};
    void *downArgs[]={&mid,&down,&residual,&output,&rows};
    const unsigned tiles=(rows+p.tile_-1)/p.tile_;
    error = cudaLaunchKernel(p.gate_,dim3(p.shape_.intermediate/16,tiles),dim3(256),gateArgs,p.gateShared_,stream);
    if (error != cudaSuccess) return error;
    return cudaLaunchKernel(p.down_,dim3(p.shape_.hidden/64,tiles),dim3(256),downArgs,p.downShared_,stream);
}
} // namespace fastllm::cuda::low_latency
