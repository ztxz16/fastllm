#include "utils/hybrid_moe.h"
#include "moe_cache_config.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
using namespace fastllm;
static uint16_t B(float x) { uint32_t u; std::memcpy(&u,&x,4); u+=0x7fff+((u>>16)&1); return u>>16; }
static float F(uint16_t x) { uint32_t u=uint32_t(x)<<16; float f; std::memcpy(&f,&u,4); return f; }
static void Require(bool v,const char *s) { if(!v) throw std::runtime_error(s); }
static void Gpu(Data &x) { x.ToDevice(DataDevice::CUDA,std::vector<int>{0}); }
static void Run(int hidden,int inter,int experts) {
    std::mt19937 rng(137+hidden); std::normal_distribution<float> normal(0,.04f);
    Data gate(BFLOAT16,{experts,2*inter,hidden}),down(BFLOAT16,{experts,hidden,inter});
    gate.Allocate();down.Allocate();
    auto *g=(uint16_t*)gate.cpuData,*d=(uint16_t*)down.cpuData;
    for(size_t i=0;i<gate.Count(0);++i)g[i]=B(normal(rng));
    for(size_t i=0;i<down.Count(0);++i)d[i]=B(normal(rng));
    HybridMoeBF16 backend({{&gate,&down}},0);
    for(int rows:{1,2,8,16,32,64}) for(int repeated:{0,1}) {
        const int topk=4;
        Data x(BFLOAT16,{rows,hidden}),ids(INT32,{rows,topk}),scores(FLOAT32,{rows,topk}),out;
        x.Allocate();ids.Allocate();scores.Allocate();
        std::vector<uint16_t> input(rows*hidden);std::vector<int> idx(rows*topk);std::vector<float> weights(rows*topk);
        for(auto &v:input)v=B(normal(rng)*8);
        std::vector<int> pool(experts);std::iota(pool.begin(),pool.end(),0);
        for(int r=0;r<rows;++r) {std::shuffle(pool.begin(),pool.end(),rng);for(int k=0;k<topk;++k){
            idx[r*topk+k]=repeated ? (experts-1-k) : pool[k];
            weights[r*topk+k]=k==0 ? 0.f : (k==1?.18732f:.412345f);
        }}
        std::memcpy(x.cpuData,input.data(),input.size()*2);std::memcpy(ids.cpuData,idx.data(),idx.size()*4);
        std::memcpy(scores.cpuData,weights.data(),weights.size()*4);Gpu(x);Gpu(ids);Gpu(scores);
        // Repeated calls exercise cache warmup and transitions as well as CPU-only execution.
        for(int call=0;call<3;++call)backend.Run(x,ids,scores,out,0);
        out.ToDevice(DataDevice::CPU);auto *actual=(uint16_t*)out.cpuData;
        double error=0,energy=0,maxError=0;
        for(int row=0;row<rows;++row){
            std::vector<std::vector<float>> parts(topk,std::vector<float>(hidden));
            for(int k=0;k<topk;++k){int e=idx[row*topk+k];std::vector<float> middle(inter);
                for(int j=0;j<inter;++j){double a=0,b=0;for(int h=0;h<hidden;++h){
                    a+=double(F(input[row*hidden+h]))*F(g[((size_t)e*2*inter+j)*hidden+h]);
                    b+=double(F(input[row*hidden+h]))*F(g[((size_t)e*2*inter+inter+j)*hidden+h]);
                }float ga=F(B(a)),up=F(B(b));middle[j]=F(B(ga/(1.f+std::exp(-ga))*up));}
                for(int h=0;h<hidden;++h){double sum=0;for(int j=0;j<inter;++j)sum+=double(middle[j])*F(d[((size_t)e*hidden+h)*inter+j]);
                    parts[k][h]=F(B(float(sum)*weights[row*topk+k]));}
            }
            std::vector<int> order(topk);std::iota(order.begin(),order.end(),0);std::stable_sort(order.begin(),order.end(),[&](int a,int b){return idx[row*topk+a]<idx[row*topk+b];});
            for(int h=0;h<hidden;++h){float sum=0;for(int k:order)sum+=parts[k][h];float expected=F(B(sum)),value=F(actual[row*hidden+h]);
                Require(std::isfinite(value),"nonfinite output");double delta=value-expected;error+=delta*delta;energy+=double(expected)*expected;maxError=std::max(maxError,std::abs(delta));}
        }
        double rel=std::sqrt(error/std::max(energy,1.e-30));
        std::cout<<"shape="<<rows<<","<<hidden<<","<<inter<<" repeated="<<repeated<<" rel_l2="<<rel<<" max_abs="<<maxError<<std::endl;
        Require(rel<.006,"BF16 hybrid relative error");
    }
    auto stats = backend.GetStats();
    std::cout << "cpu_routes=" << stats.cpuRoutes << " gpu_routes=" << stats.gpuRoutes
              << " cache_hit_routes=" << stats.cacheHits << " uploads=" << stats.uploads << std::endl;
    if (GetMoeCudaCacheBytes() >= size_t(hidden) * inter * 12) {
        Require(stats.gpuRoutes > 0 && stats.cacheHits > 0, "cache path not exercised");
    }
}
static void Exact() {
    constexpr int h=128,i=64,e=8,k=4,rows=16;
    Data gate(BFLOAT16,{e,2*i,h}),down(BFLOAT16,{e,h,i});gate.Allocate(0.f);down.Allocate(0.f);
    auto *g=(uint16_t*)gate.cpuData,*d=(uint16_t*)down.cpuData;
    for(int expert=0;expert<e;++expert) {
        for(int j=0;j<i;++j) {
            g[(expert*2*i+j)*h+j]=B(.5f);
            g[(expert*2*i+i+j)*h+j]=B(1.5f);
        }
        for(int n=0;n<h;++n)d[(expert*h+n)*i+n%i]=B(float(expert+1)/16);
    }
    HybridMoeBF16 backend({{&gate,&down},{nullptr,nullptr},{&gate,&down}},0);
    Data x(BFLOAT16,{rows,h}),ids(INT32,{rows,k}),scores(FLOAT32,{rows,k}),out;
    x.Allocate();ids.Allocate();scores.Allocate();
    const float route[k]={0,.18732f,.412345f,-.2937f};
    for(int r=0;r<rows;++r)for(int n=0;n<h;++n)((uint16_t*)x.cpuData)[r*h+n]=B(float(1+r%4)/8);
    for(int r=0;r<rows;++r)for(int t=0;t<k;++t){((int*)ids.cpuData)[r*k+t]=7-t;((float*)scores.cpuData)[r*k+t]=route[t];}
    Gpu(x);Gpu(ids);Gpu(scores);
    for(int turn=0;turn<8;++turn)for(int layer:{0,2}) {
        backend.SetPrefill(turn == 0 || turn == 4);
        backend.Run(x,ids,scores,out,layer);
        Data host;host.CopyFrom(out);host.ToDevice(DataDevice::CPU);
        for(int r=0;r<rows;++r) {
            float input=float(1+r%4)/8,ga=F(B(input*.5f)),up=F(B(input*1.5f));
            float middle=F(B(ga/(1.f+std::exp(-ga))*up)),sum=0;
            for(int t=k-1;t>=0;--t)sum+=F(B(middle*(8-t)/16*route[t]));
            for(int n=0;n<h;++n)Require(((uint16_t*)host.cpuData)[r*h+n]==B(sum),"exact BF16 rounding/score contract");
        }
    }
    std::cout<<"exact multi-layer rounding PASS"<<std::endl;
}
int main(int argc,char **argv){try{
    int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||!count)return 77;
    SetThreads(8);SetDeviceMap({{"cuda:0",1}});
    SetMoeCudaCacheBytes(argc>1?std::stoull(argv[1]):0);
    Exact();Run(128,64,8);Run(2048,512,16);
    std::cout<<"ALL_PASS"<<std::endl;
}catch(const std::exception &e){std::cerr<<e.what()<<std::endl;return 1;}}
