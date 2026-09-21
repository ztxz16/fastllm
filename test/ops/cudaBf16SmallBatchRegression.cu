#include "fastllm-cuda.cuh"
#include "fastllm.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

static float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
static float fp(float x) { return __half2float(__float2half_rn(x)); }
static void cudaCheck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static std::vector<float> read(fastllm::Data x) {
    x.ToDevice(fastllm::DataDevice::CPU);
    fastllm::ToDataTypeForceCPU(x,fastllm::DataType::FLOAT32);
    float *p=(float*)x.cpuData; return {p,p+x.Count(0)};
}
int main(int argc,char**argv) try {
    using namespace fastllm;
    FastllmCudaSetDevice(0);
    const bool micro=argc>1;
    int checks=0;
    for(bool halfInput:{true,false}) for(int K: (micro?std::vector<int>{5120}:std::vector<int>{33,256,5120}))
    for(int M: (micro?std::vector<int>{96}:std::vector<int>{1,96,129,1024,1025}))
    for(int n: (micro?std::vector<int>{1,2,4,6,8,9}:std::vector<int>{1,6,7,8,9}))
    for(int pattern=0;pattern<(micro?1:3);++pattern) {
        DataType dtype=halfInput?DataType::FLOAT16:DataType::BFLOAT16;
        std::mt19937 rng(211+K+M+n);
        std::normal_distribution<float> normal(0,.1f);
        std::vector<float>w(M*K),x(n*K),b(M);
        for(auto &v:w)v=bf(normal(rng));
        for(int i=0;i<n*K;++i)x[i]=(halfInput?fp:bf)(pattern==0?normal(rng):pattern==1?((i%3)-1)*.125f:0.f);
        for(auto &v:b)v=(halfInput?fp:bf)(normal(rng));
        Data input(dtype,{n,K},x);
        input.ToDevice(DataDevice::CUDA,{0},true);
        for(bool useBias: (micro?std::vector<bool>{false}:std::vector<bool>{false,true})) {
            Data weight(DataType::BFLOAT16,{M,K},w);
            weight.ToDevice(DataDevice::CUDA,{0},true);
            Data bias(DataType::FLOAT32);
            if(useBias) {
                bias.Resize({M});bias.Allocate();
                std::memcpy(bias.cpuData,b.data(),M*sizeof(float));
                bias.ToDevice(DataDevice::CUDA,{0},true);
            }
            Data output(dtype,{1,n*M+2},std::vector<float>(n*M+2,-17.f));
            output.ToDevice(DataDevice::CUDA,{0},true);
            auto run=[&](){
                if(halfInput)FastllmCudaHalfMatMulBFloat16(input,weight,bias,output,n,K,M);
                else FastllmCudaBFloat16MatMulBFloat16(input,weight,bias,output,n,K,M);
            };
            run();cudaCheck(cudaDeviceSynchronize());auto actual=read(output);
            if(actual[n*M]!=-17.f || actual[n*M+1]!=-17.f)throw std::runtime_error("output guard overwritten");
            double err2=0,ref2=0,maxErr=0;
            for(int t=0;t<n;++t)for(int row=0;row<M;++row) {
                double sum=0;
                for(int k=0;k<K;++k)sum+=double(w[row*K+k])*(halfInput&&n>=8?bf(x[t*K+k]):x[t*K+k]);
                float expected=(float)sum;
                if(n>=8)expected=bf(expected);
                if(useBias)expected+=b[row];
                expected=(halfInput?fp:bf)(expected);
                double error=actual[t*M+row]-expected;
                if(!std::isfinite(actual[t*M+row]))throw std::runtime_error("nonfinite output");
                err2+=error*error;ref2+=double(expected)*expected;maxErr=std::max(maxErr,std::abs(error));
            }
            double rel=std::sqrt(err2/std::max(ref2,1e-30));
            if(rel>0.002) {
                std::cerr<<"FAIL half="<<halfInput<<" n="<<n<<" K="<<K<<" M="<<M<<" bias="<<useBias<<" pattern="<<pattern<<" rel="<<rel<<"\n";
                return 3;
            }
            ++checks;
            if(micro) {
                for(int i=0;i<50;++i)run();
                cudaEvent_t a,z;cudaCheck(cudaEventCreate(&a));cudaCheck(cudaEventCreate(&z));
                std::vector<float> times;
                for(int s=0;s<5;++s){cudaCheck(cudaEventRecord(a,cudaStreamPerThread));for(int i=0;i<100;++i)run();cudaCheck(cudaEventRecord(z,cudaStreamPerThread));cudaCheck(cudaEventSynchronize(z));float ms;cudaCheck(cudaEventElapsedTime(&ms,a,z));times.push_back(ms*10);}
                std::sort(times.begin(),times.end());
                std::cout<<"{\"half\":"<<halfInput<<",\"n\":"<<n<<",\"us\":"<<times[2]<<",\"rel\":"<<rel<<",\"maxerr\":"<<maxErr<<"}\n";
                cudaEventDestroy(a);cudaEventDestroy(z);
            }
        }
    }
    std::cout<<"CHECKS "<<checks<<" passed\n";
    return 0;
} catch(const std::exception&e) {std::cerr<<e.what()<<"\n";return 2;}
