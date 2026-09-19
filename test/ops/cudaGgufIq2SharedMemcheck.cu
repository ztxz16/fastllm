// Standalone memory checking of the production kernels. Linking the full
// runtime initializes CUDA before Compute Sanitizer can install its hooks.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>
#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "gguf.h"
static __device__ __forceinline__ int get_int_b2(const void *p,const int &i) {
    const auto *q=static_cast<const uint16_t*>(p);return int(q[2*i]) | (int(q[2*i+1])<<16);
}
static __device__ __forceinline__ int get_int_b4(const void *p,const int &i) {
    return static_cast<const int*>(p)[i];
}
static __device__ __forceinline__ int ggml_cuda_dp4a(int a,int b,int c) {return __dp4a(a,b,c);}
static __device__ __forceinline__ float warp_reduce_sum(float value) {
    #pragma unroll
    for(int d=16;d>0;d>>=1)value+=__shfl_xor_sync(0xffffffff,value,d);
    return value;
}
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#define WARP_SIZE 32
#include <fastllm-gguf-iq2-gemv.cuh>
#define CUDA(x) do { auto error=(x); if(error!=cudaSuccess) { fprintf(stderr,"%d: %s\n",__LINE__,cudaGetErrorString(error)); exit(1); } } while(0)
template<ggml_type T,typename Out>void Check(const void*w,const void*u,const block_q8_1*x,int K,int M) {
    std::vector<Out> h(M+2,Out(-17.f));Out*y;
    CUDA(cudaMalloc(&y,h.size()*sizeof(Out)));
    CUDA(cudaMemcpy(y,h.data(),h.size()*sizeof(Out),cudaMemcpyHostToDevice));
    iq2_decode::Launch<T,false>(w,nullptr,x,y,K,M,cudaStreamPerThread);
    CUDA(cudaDeviceSynchronize());CUDA(cudaMemcpy(h.data(),y,h.size()*sizeof(Out),cudaMemcpyDeviceToHost));
    if(float(h[M])!=-17.f || float(h[M+1])!=-17.f)exit(2);
    if constexpr(std::is_same<Out,half>::value) {
        iq2_decode::Launch<T,true>(w,u,x,y,K,M,cudaStreamPerThread);
        CUDA(cudaDeviceSynchronize());CUDA(cudaMemcpy(h.data(),y,h.size()*sizeof(Out),cudaMemcpyDeviceToHost));
        if(float(h[M])!=-17.f || float(h[M+1])!=-17.f)exit(3);
    }
    CUDA(cudaFree(y));
}
int main(int argc,char**argv) {
    if(argc!=2)return 1;
    std::ifstream f(argv[1],std::ios::binary);uint32_t cases;f.read((char*)&cases,4);
    int checks=0;
    for(unsigned ci=0;ci<cases;++ci) {
        uint32_t h[4];f.read((char*)h,sizeof(h));std::vector<char>packed(h[3]);f.read(packed.data(),packed.size());
        f.seekg(size_t(h[1])*h[2]*4,std::ios::cur);
        if(h[0]!=16 && h[0]!=17 && h[0]!=22)continue;
        if(h[2]>18432)continue;
        const int K=h[2];const size_t rowBytes=h[3]/h[1];
        for(int M : {1,7,33,127,128,129,1031}) {
            std::vector<char>weights(rowBytes*M);
            for(int i=0;i<M;++i)std::copy_n(packed.data()+(i%h[1])*rowBytes,rowBytes,weights.data()+i*rowBytes);
            void*w,*u;block_q8_1*x;
            CUDA(cudaMalloc(&w,weights.size()));CUDA(cudaMalloc(&u,weights.size()));
            CUDA(cudaMemcpy(w,weights.data(),weights.size(),cudaMemcpyHostToDevice));
            CUDA(cudaMemcpy(u,weights.data(),weights.size(),cudaMemcpyHostToDevice));
            CUDA(cudaMalloc(&x,K/32*sizeof(block_q8_1)));
            std::vector<half>values(K);for(int j=0;j<K;++j)values[j]=half((j%11-5)*.02f);
            std::vector<block_q8_1> quantized(K/32);
            for(int j=0;j<K;j+=32) {
                float maxabs=0,sum=0;
                for(int z=0;z<32;++z){maxabs=std::max(maxabs,std::abs(float(values[j+z])));sum+=float(values[j+z]);}
                float d=maxabs/127.f;half scale=half(d),total=half(sum);
                auto &q=quantized[j/32];std::memcpy(&q.ds,&scale,2);std::memcpy(reinterpret_cast<char*>(&q.ds)+2,&total,2);
                for(int z=0;z<32;++z)q.qs[z]=maxabs==0?0:std::round(float(values[j+z])/d);
            }
            CUDA(cudaMemcpy(x,quantized.data(),K/32*sizeof(block_q8_1),cudaMemcpyHostToDevice));
            if(h[0]==16) {
                Check<GGML_TYPE_IQ2_XXS,half>(w,u,x,K,M);Check<GGML_TYPE_IQ2_XXS,float>(w,u,x,K,M);Check<GGML_TYPE_IQ2_XXS,__nv_bfloat16>(w,u,x,K,M);
            } else if(h[0]==17) {
                Check<GGML_TYPE_IQ2_XS,half>(w,u,x,K,M);Check<GGML_TYPE_IQ2_XS,float>(w,u,x,K,M);Check<GGML_TYPE_IQ2_XS,__nv_bfloat16>(w,u,x,K,M);
            } else {
                Check<GGML_TYPE_IQ2_S,half>(w,u,x,K,M);Check<GGML_TYPE_IQ2_S,float>(w,u,x,K,M);Check<GGML_TYPE_IQ2_S,__nv_bfloat16>(w,u,x,K,M);
            }
            CUDA(cudaFree(w));CUDA(cudaFree(u));CUDA(cudaFree(x));checks+=4;
        }
    }
    printf("PRODUCTION_KERNEL_MEMORY_CHECKS %d passed\n",checks);
}
