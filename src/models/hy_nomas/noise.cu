#include "models/hy_nomas/runtime.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <curand_kernel.h>
#include <algorithm>
#include <stdexcept>

namespace fastllm { namespace nomas {
namespace {
void checkNoise(cudaError_t status) {
    if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<class T> __device__ T noiseCast(float x) { return x; }
template<> __device__ __nv_bfloat16 noiseCast(float x) { return __float2bfloat16_rn(x); }

// Match CUDA torch.randn(float32): Philox subsequence = thread index, four
// normals per draw, grid-stride output. Each latent frame has its own seed.
// Fuse [3,4,H,W] -> [(H/16)*(W/16),3072] and the decoder input cast, avoiding
// an intermediate FP32 video, concatenation, upload and patchify pass.
template<class T> __global__ void pixelNoiseKernel(T *output,uint64_t seed,int firstFrame,
                                                  int height,int width,size_t count) {
    const size_t thread=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const size_t stride=size_t(gridDim.x)*blockDim.x;
    const size_t plane=size_t(height)*width;
    curandStatePhilox4_32_10_t state;
    curand_init(seed*1000003ULL+uint64_t(firstFrame)+blockIdx.y,thread,0,&state);
    for(size_t base=thread;base<count;base+=stride*4) {
        float4 normal=curand_normal4(&state);
        float values[]={normal.x,normal.y,normal.z,normal.w};
        #pragma unroll
        for(int j=0;j<4;++j) {
            size_t index=base+stride*j;
            if(index>=count) continue;
            int c=index/(4*plane),frame=index/plane%4;
            int y=index/width%height,x=index%width;
            size_t token=(size_t(blockIdx.y)*(height/16)+y/16)*(width/16)+x/16;
            size_t channel=((frame*16+y%16)*16+x%16)*3+c;
            // randn applies std * normal + mean even for std=1, mean=0;
            // keep that operation so negative zero matches the reference too.
            float value=__fmaf_rn(values[j],1.f,0.f);
            output[token*3072+channel]=noiseCast<T>(value);
        }
    }
}
}

Tensor Runtime::pixelNoise(uint64_t seed,int firstFrame,int frames,int height,int width) {
    if(dtype()==DType::FP16) throw std::invalid_argument("PixelUMM noise supports FP32/BF16");
    if(firstFrame<0 || frames<=0 || frames>65535 || height<=0 || width<=0 || height%16 || width%16)
        throw std::invalid_argument("invalid PixelUMM noise shape");
    const int64_t spatial=int64_t(height/16)*(width/16);
    // Match the reference's single 32-bit-indexed randn launch per frame.
    if(spatial>INT32_MAX/3072 || spatial*frames>INT32_MAX)
        throw std::invalid_argument("PixelUMM noise shape exceeds supported indexing");
    checkNoise(cudaSetDevice(device_));
    cudaDeviceProp props{}; checkNoise(cudaGetDeviceProperties(&props,device_));
    const size_t count=size_t(spatial)*3072;
    const int blocks=std::min<size_t>((count+255)/256,
        props.multiProcessorCount*(props.maxThreadsPerMultiProcessor/256));
    Tensor out(int(spatial*frames),3072,isFloat32());
    if(isFloat32()) pixelNoiseKernel<<<dim3(blocks,frames),256,0,cudaStreamPerThread>>>(
        reinterpret_cast<float *>(out.data),seed,firstFrame,height,width,count);
    else pixelNoiseKernel<<<dim3(blocks,frames),256,0,cudaStreamPerThread>>>(
        reinterpret_cast<__nv_bfloat16 *>(out.data),seed,firstFrame,height,width,count);
    checkNoise(cudaGetLastError());
    return out;
}
}} // namespace fastllm::nomas
