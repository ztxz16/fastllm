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
static __device__ __forceinline__ half FastllmGgufHalfSiluMulValue(half gate,half up) {
    return __hmul(__hdiv(gate,__hadd(__float2half(1.f),hexp(-gate))),up);
}
#include <fastllm-gguf-iq3-gemv.cuh>
#include <fastllm-gguf-small-mmvq.cuh>
#define CUDA(x) do { auto error=(x); if(error!=cudaSuccess) { fprintf(stderr,"%d: %s\n",__LINE__,cudaGetErrorString(error)); exit(1); } } while(0)
template<ggml_type T,typename Out>void Single(const void*w,const block_q8_1*x,Out*y,int K,int M) {
    if constexpr(T==GGML_TYPE_IQ3_S || T==GGML_TYPE_IQ3_XXS) {
        if (K<=18432) {
            fastllm_gguf_iq3::Launch<T,false>(w,nullptr,x,y,K,M,cudaStreamPerThread);
            return;
        }
    }
    fastllm_gguf_small_mmvq::LaunchRows<T,1,8>(w,x,y,K,M,K,M,cudaStreamPerThread);
}
__global__ void CheckIQ4LookupKernel(int2 *out) {
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
    // The first 256 cover every low/high nibble pair; the rest mix positions.
    const uint32_t word=i<256?i*0x01010101u:i*0x9e3779b9u;
    out[i]=fastllm_gguf_small_mmvq::LookupIQ4(word);
}
static void CheckIQ4Lookup() {
    constexpr int count=4096;
    const int8_t table[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    int2 *device;CUDA(cudaMalloc(&device,count*sizeof(int2)));
    CheckIQ4LookupKernel<<<count/256,256,0,cudaStreamPerThread>>>(device);
    CUDA(cudaDeviceSynchronize());std::vector<int2> values(count);
    CUDA(cudaMemcpy(values.data(),device,count*sizeof(int2),cudaMemcpyDeviceToHost));
    for(uint32_t i=0;i<count;++i) {
        uint32_t word=i<256?i*0x01010101u:i*0x9e3779b9u,lo=0,hi=0;
        for(int byte=0;byte<4;++byte){lo|=uint32_t(uint8_t(table[(word>>(8*byte))&15]))<<(8*byte);hi|=uint32_t(uint8_t(table[(word>>(8*byte+4))&15]))<<(8*byte);}
        if(uint32_t(values[i].x)!=lo || uint32_t(values[i].y)!=hi)exit(7);
    }
    CUDA(cudaFree(device));printf("IQ4_LOOKUP_CHECKS %d passed\n",count);
}
template<ggml_type T,typename Out>void Check(const void*w,const void*u,const block_q8_1*x,int K,int M) {
    std::vector<Out> h(M+2,Out(-17.f));Out*y;
    CUDA(cudaMalloc(&y,h.size()*sizeof(Out)));
    CUDA(cudaMemcpy(y,h.data(),h.size()*sizeof(Out),cudaMemcpyHostToDevice));
    Single<T>(w,x,y,K,M);
    CUDA(cudaDeviceSynchronize());CUDA(cudaMemcpy(h.data(),y,h.size()*sizeof(Out),cudaMemcpyDeviceToHost));
    if(float(h[M])!=-17.f || float(h[M+1])!=-17.f)exit(2);
    if constexpr(std::is_same<Out,half>::value && (T==GGML_TYPE_IQ3_S || T==GGML_TYPE_IQ3_XXS)) {
        if (K<=18432) {
            fastllm_gguf_iq3::Launch<T,true>(w,u,x,y,K,M,cudaStreamPerThread);
            CUDA(cudaDeviceSynchronize());CUDA(cudaMemcpy(h.data(),y,h.size()*sizeof(Out),cudaMemcpyDeviceToHost));
            if(float(h[M])!=-17.f || float(h[M+1])!=-17.f)exit(3);
        }
    }
    CUDA(cudaFree(y));
    // Padded input and output strides, all input token counts, and partial
    // output tiles exercise the batch launch independently of public routing.
    std::vector<Out> reference(8*M);
    CUDA(cudaMalloc(&y,reference.size()*sizeof(Out)));
    for(int token=0;token<8;++token)
        Single<T>(w,x+token*(K+256)/32,y+token*M,K,M);
    CUDA(cudaDeviceSynchronize());CUDA(cudaMemcpy(reference.data(),y,reference.size()*sizeof(Out),cudaMemcpyDeviceToHost));
    CUDA(cudaFree(y));
    for (int tokens=2; tokens<=8; ++tokens) {
        const int outputStride=M+3;
        std::vector<Out> batch(tokens*outputStride+2,Out(-17.f));
        CUDA(cudaMalloc(&y,batch.size()*sizeof(Out)));
        CUDA(cudaMemcpy(y,batch.data(),batch.size()*sizeof(Out),cudaMemcpyHostToDevice));
        if(tokens==8) {
            cudaGraph_t graph; cudaGraphExec_t executable;
            CUDA(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
            fastllm_gguf_small_mmvq::LaunchBatch<T>(w,x,y,K,M,tokens,K+256,outputStride,cudaStreamPerThread);
            CUDA(cudaStreamEndCapture(cudaStreamPerThread,&graph));
            CUDA(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
            CUDA(cudaGraphLaunch(executable,cudaStreamPerThread));
            CUDA(cudaDeviceSynchronize());
            CUDA(cudaGraphExecDestroy(executable));CUDA(cudaGraphDestroy(graph));
        } else {
            fastllm_gguf_small_mmvq::LaunchBatch<T>(w,x,y,K,M,tokens,K+256,outputStride,cudaStreamPerThread);
        }
        CUDA(cudaDeviceSynchronize());CUDA(cudaMemcpy(batch.data(),y,batch.size()*sizeof(Out),cudaMemcpyDeviceToHost));
        for(int token=0;token<tokens;++token) {
            for(int row=M;row<outputStride;++row)if(float(batch[token*outputStride+row])!=-17.f)exit(4);
            for(int row=0;row<M;++row) {
                float value=float(batch[token*outputStride+row]);
                if(!std::isfinite(value) || value!=float(reference[token*M+row]))exit(5);
            }
        }
        if(float(batch[tokens*outputStride])!=-17.f || float(batch[tokens*outputStride+1])!=-17.f)exit(6);
        CUDA(cudaFree(y));
    }
}
int main(int argc,char**argv) {
    if(argc!=2)return 1;
    CheckIQ4Lookup();
    std::ifstream f(argv[1],std::ios::binary);uint32_t cases;f.read((char*)&cases,4);
    int checks=0;
    for(unsigned ci=0;ci<cases;++ci) {
        uint32_t h[4];f.read((char*)h,sizeof(h));std::vector<char>packed(h[3]);f.read(packed.data(),packed.size());
        f.seekg(size_t(h[1])*h[2]*4,std::ios::cur);
        if(h[0]!=18 && h[0]!=21 && h[0]!=23 && h[0]!=16 && h[0]!=17 && h[0]!=22 && h[0]!=12 && h[0]!=10 && h[0]!=29)continue;
        const int K=h[2];const size_t rowBytes=h[3]/h[1];
        for(int M : {1,7,33,129,1031,4097}) {
            if(K>18432 && M!=129 && M!=4097)continue;
            if(M==4097 && K!=256 && K!=18688)continue;
            std::vector<char>weights(rowBytes*M);
            for(int i=0;i<M;++i)std::copy_n(packed.data()+(i%h[1])*rowBytes,rowBytes,weights.data()+i*rowBytes);
            void*w,*u;block_q8_1*x;
            CUDA(cudaMalloc(&w,weights.size()));CUDA(cudaMalloc(&u,weights.size()));
            CUDA(cudaMemcpy(w,weights.data(),weights.size(),cudaMemcpyHostToDevice));
            CUDA(cudaMemcpy(u,weights.data(),weights.size(),cudaMemcpyHostToDevice));
            const int elements=8*(K+256);
            CUDA(cudaMalloc(&x,elements/32*sizeof(block_q8_1)));
            std::vector<half>values(elements);for(int j=0;j<elements;++j)values[j]=half((j%11-5)*.02f);
            std::vector<block_q8_1> quantized(elements/32);
            for(int j=0;j<elements;j+=32) {
                float maxabs=0,sum=0;
                for(int z=0;z<32;++z){maxabs=std::max(maxabs,std::abs(float(values[j+z])));sum+=float(values[j+z]);}
                float d=maxabs/127.f;half scale=half(d),total=half(sum);
                auto &q=quantized[j/32];std::memcpy(&q.ds,&scale,2);std::memcpy(reinterpret_cast<char*>(&q.ds)+2,&total,2);
                for(int z=0;z<32;++z)q.qs[z]=maxabs==0?0:std::round(float(values[j+z])/d);
            }
            CUDA(cudaMemcpy(x,quantized.data(),elements/32*sizeof(block_q8_1),cudaMemcpyHostToDevice));
            if(h[0]==18) {
                Check<GGML_TYPE_IQ3_XXS,half>(w,u,x,K,M);Check<GGML_TYPE_IQ3_XXS,float>(w,u,x,K,M);Check<GGML_TYPE_IQ3_XXS,__nv_bfloat16>(w,u,x,K,M);
            } else if(h[0]==16) {
                Check<GGML_TYPE_IQ2_XXS,half>(w,u,x,K,M);Check<GGML_TYPE_IQ2_XXS,float>(w,u,x,K,M);Check<GGML_TYPE_IQ2_XXS,__nv_bfloat16>(w,u,x,K,M);
            } else if(h[0]==17) {
                Check<GGML_TYPE_IQ2_XS,half>(w,u,x,K,M);Check<GGML_TYPE_IQ2_XS,float>(w,u,x,K,M);Check<GGML_TYPE_IQ2_XS,__nv_bfloat16>(w,u,x,K,M);
            } else if(h[0]==22) {
                Check<GGML_TYPE_IQ2_S,half>(w,u,x,K,M);Check<GGML_TYPE_IQ2_S,float>(w,u,x,K,M);Check<GGML_TYPE_IQ2_S,__nv_bfloat16>(w,u,x,K,M);
            } else if(h[0]==29) {
                Check<GGML_TYPE_IQ1_M,half>(w,u,x,K,M);Check<GGML_TYPE_IQ1_M,float>(w,u,x,K,M);Check<GGML_TYPE_IQ1_M,__nv_bfloat16>(w,u,x,K,M);
            } else if(h[0]==10) {
                Check<GGML_TYPE_Q2_K,half>(w,u,x,K,M);Check<GGML_TYPE_Q2_K,float>(w,u,x,K,M);Check<GGML_TYPE_Q2_K,__nv_bfloat16>(w,u,x,K,M);
            } else if(h[0]==12) {
                Check<GGML_TYPE_Q4_K,half>(w,u,x,K,M);Check<GGML_TYPE_Q4_K,float>(w,u,x,K,M);Check<GGML_TYPE_Q4_K,__nv_bfloat16>(w,u,x,K,M);
            } else if(h[0]==23) {
                Check<GGML_TYPE_IQ4_XS,half>(w,u,x,K,M);Check<GGML_TYPE_IQ4_XS,float>(w,u,x,K,M);Check<GGML_TYPE_IQ4_XS,__nv_bfloat16>(w,u,x,K,M);
            } else {
                Check<GGML_TYPE_IQ3_S,half>(w,u,x,K,M);Check<GGML_TYPE_IQ3_S,float>(w,u,x,K,M);Check<GGML_TYPE_IQ3_S,__nv_bfloat16>(w,u,x,K,M);
            }
            CUDA(cudaFree(w));CUDA(cudaFree(u));CUDA(cudaFree(x));checks+=((h[0]==18 || h[0]==21) && K<=18432?4:3)+3*7;
        }
    }
    printf("PRODUCTION_KERNEL_MEMORY_CHECKS %d passed\n",checks);
}
