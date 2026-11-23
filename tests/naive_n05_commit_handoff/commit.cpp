#include "fastllm.h"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>
using namespace fastllm;
static void cudaCheck(cudaError_t e) { if(e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Cache {
    std::unique_ptr<Data> key, value;
    std::vector<unsigned char> k, v, kr, vr;
    int past, count, kc, vc, bytes;
    Cache(int rank, int i, bool uniform=false) {
        static const int positions[]={0,1,126,127,128,129,255,256,8191,8192,65529};
        past=positions[i%11]; count=uniform?8:i%9;
        kc=uniform?192:std::vector<int>{1,17,64,192,257}[i%5];
        vc=uniform?128:std::vector<int>{3,128,7,256,31}[i%5];
        auto type=(!uniform && i%2)?FP8_E4M3:BFLOAT16;bytes=type==BFLOAT16?2:1;
        auto init=[&](int width,std::vector<unsigned char>&host,std::vector<unsigned char>&ref){
            auto d=std::make_unique<Data>(type,std::vector<int>{1,144,width});
            d->ToDevice(DataDevice::CUDA,{rank},false);d->Allocate(false);
            d->expansionDims={1,144,width};
            host.resize(size_t(144)*width*bytes);
            for(size_t j=0;j<host.size();++j)host[j]=(j*29+i*71+rank*13+(j>>8))&255;
            ref=host;
            for(int row=0;row<count;++row)
                std::copy_n(host.data()+size_t(128+row)*width*bytes,width*bytes,
                    ref.data()+size_t((past+row)&127)*width*bytes);
            cudaCheck(cudaMemcpy(d->cudaData,host.data(),host.size(),cudaMemcpyHostToDevice));
            d->Resize({1,std::min(past,127)+8,width});return d;
        };
        key=init(kc,k,kr);value=init(vc,v,vr);
    }
    void reset() {
        cudaCheck(cudaMemcpy(key->cudaData,k.data(),k.size(),cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(value->cudaData,v.data(),v.size(),cudaMemcpyHostToDevice));
    }
    void check() {
        for(auto pair:{std::make_pair(key.get(),&kr),std::make_pair(value.get(),&vr)}) {
            std::vector<unsigned char> got(pair.second->size());
            cudaCheck(cudaMemcpy(got.data(),pair.first->cudaData,got.size(),cudaMemcpyDeviceToHost));
            if(got!=*pair.second)throw std::runtime_error("KV bitwise/guard mismatch");
            if(pair.first->dims[1]!=std::min(past+count,127))throw std::runtime_error("KV length mismatch");
        }
    }
};
int main() {
    try {
        int checked=0;
        for(int rank=0;rank<8;++rank) {
            FastllmCudaSetDevice(rank);
            for(int n:{1,39,78,111,112,113,117,156,224,225,312}) {
                std::vector<std::unique_ptr<Cache>> storage;
                std::vector<FastllmNaiveCacheCommit> desc;
                for(int i=0;i<n;++i) {
                    storage.emplace_back(new Cache(rank,i));auto &c=*storage.back();
                    desc.push_back({c.key.get(),c.value.get(),c.past,c.count});
                }
                FastllmCudaNaiveBatchCommitSwaVerify(desc);cudaCheck(cudaStreamSynchronize(cudaStreamPerThread));
                for(auto &c:storage)c->check();
                for(auto &c:storage) {c->reset();FastllmCudaNaiveCommitSwaVerify(*c->key,*c->value,c->past,c->count);}
                cudaCheck(cudaStreamSynchronize(cudaStreamPerThread));
                for(auto &c:storage)c->check();
                checked+=n;
            }
        }
        printf("BITWISE_PASS entries=%d ranks=8 types=BF16,FP8 counts=0..8 guards=full_allocation\n",checked);
        FastllmCudaSetDevice(0);
        for(int requests:{1,2,3,4,8}) {
            std::vector<std::unique_ptr<Cache>> storage;std::vector<FastllmNaiveCacheCommit> desc;
            for(int i=0;i<39*requests;++i) {storage.emplace_back(new Cache(0,i,true));auto &c=*storage.back();desc.push_back({c.key.get(),c.value.get(),c.past,c.count});}
            for(bool batch:{false,true}) {
                auto run=[&]{if(batch)FastllmCudaNaiveBatchCommitSwaVerify(desc);else for(auto &c:storage)FastllmCudaNaiveCommitSwaVerify(*c->key,*c->value,c->past,c->count);};
                for(int i=0;i<5;++i)run();cudaCheck(cudaStreamSynchronize(cudaStreamPerThread));
                double host=0,gpu=0;
                cudaEvent_t begin,end;cudaCheck(cudaEventCreate(&begin));cudaCheck(cudaEventCreate(&end));
                for(int i=0;i<30;++i) {
                    auto t=std::chrono::steady_clock::now();
                    cudaCheck(cudaEventRecord(begin,cudaStreamPerThread));run();
                    cudaCheck(cudaEventRecord(end,cudaStreamPerThread));cudaCheck(cudaEventSynchronize(end));
                    host+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t).count();
                    float elapsed;cudaCheck(cudaEventElapsedTime(&elapsed,begin,end));gpu+=elapsed*1000;
                }
                cudaCheck(cudaEventDestroy(begin));cudaCheck(cudaEventDestroy(end));
                printf("BENCH requests=%d batched=%d host_us=%.3f gpu_event_us=%.3f\n",requests,batch,host/30,gpu/30);
            }
        }
        puts("ALL_PASS");return 0;
    }catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
