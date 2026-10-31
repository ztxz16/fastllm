#include "models/lingbot_world/runtime.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fastllm { namespace lingbot {
namespace {
float *ptr(const Tensor &t) {
    if(!t.isFloat32()) throw std::invalid_argument("LingBot sampler requires FP32 tensors");
    return reinterpret_cast<float *>(t.data);
}
int grid(size_t n) { return int(std::min<size_t>((n+255)/256,4096)); }
void check() { auto s=cudaGetLastError(); if(s!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
__global__ void normalKernel(float *out,size_t n,uint64_t seed,uint64_t sequence) {
    for(size_t i=size_t(blockIdx.x)*256+threadIdx.x;i<(n+3)/4;i+=size_t(gridDim.x)*256) {
        curandStatePhilox4_32_10_t rng;
        curand_init(seed,i,sequence*4,&rng);
        float4 values=curand_normal4(&rng);
        float v[4]={values.x,values.y,values.z,values.w};
        for(int j=0;j<4 && i*4+j<n;++j) out[i*4+j]=v[j];
    }
}
__global__ void flowKernel(const float *x,const float *pred,float *out,size_t n,float sigma) {
    for(size_t i=size_t(blockIdx.x)*256+threadIdx.x;i<n;i+=size_t(gridDim.x)*256)
        out[i]=float(double(x[i])-double(sigma)*double(pred[i]));
}
__global__ void noiseKernel(const float *x,const float *noise,float *out,size_t n,float sigma) {
    for(size_t i=size_t(blockIdx.x)*256+threadIdx.x;i<n;i+=size_t(gridDim.x)*256)
        out[i]=(1.f-sigma)*x[i]+sigma*noise[i];
}
}
Tensor Runtime::normal(int rows,int cols,uint64_t seed,uint64_t sequence) {
    if(rows<=0 || cols<=0) throw std::invalid_argument("invalid LingBot noise shape");
    Tensor out(rows,cols,true);
    normalKernel<<<grid((out.size()+3)/4),256,0,cudaStreamPerThread>>>(ptr(out),out.size(),seed,sequence);
    check(); return out;
}
Tensor Runtime::flowToClean(const Tensor &x,const Tensor &pred,float sigma) {
    if(x.rows!=pred.rows || x.cols!=pred.cols || !std::isfinite(sigma)) throw std::invalid_argument("invalid LingBot flow inputs");
    Tensor out(x.rows,x.cols,true);
    flowKernel<<<grid(out.size()),256,0,cudaStreamPerThread>>>(ptr(x),ptr(pred),ptr(out),out.size(),sigma);
    check(); return out;
}
Tensor Runtime::addNoise(const Tensor &x,const Tensor &noise,float sigma) {
    if(x.rows!=noise.rows || x.cols!=noise.cols || !std::isfinite(sigma)) throw std::invalid_argument("invalid LingBot noise inputs");
    Tensor out(x.rows,x.cols,true);
    noiseKernel<<<grid(out.size()),256,0,cudaStreamPerThread>>>(ptr(x),ptr(noise),ptr(out),out.size(),sigma);
    check(); return out;
}
}} // namespace fastllm::lingbot
