#include "exl3.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include "utils/hybrid_moe.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-moe-policy.h"
#include "devices/cuda/fastllm-cuda-moe-cache-stats.h"
#include <algorithm>
#include <map>
#include <tuple>
#include "pure_moe_graph_test.h"

using namespace fastllm;
static void Require(bool ok,const char *message){if(!ok)throw std::runtime_error(message);}
static void Fill(Data &weight,std::mt19937 &rng){
    weight.Allocate(false);
    const int n=weight.dims[0]/weight.blockM,k=weight.dims[1];
    const size_t bytes=Exl3PartBytes(n,k,weight.blockK),trellis=size_t(n)*k*weight.blockK/16;
    for(int part=0;part<weight.blockM;++part){
        auto *p=weight.cpuData+part*bytes;
        for(size_t i=0;i<trellis;i+=4){uint32_t value=rng();std::memcpy(p+i,&value,4);}
        auto *scales=reinterpret_cast<uint16_t *>(p+trellis);
        for(int i=0;i<n+k;++i)scales[i]=float_to_half((int(rng()%63)-31)/512.f);
    }
}

int main() { try {
    if (FastllmCudaGetDeviceCount() < 2) return 77;
    SetThreads(4);
    constexpr int layers=2, experts=24, hidden=256, inter=128;
    std::mt19937 rng(20261117);
    std::vector<std::unique_ptr<Data>> owned, gpuOwned;
    std::vector<std::vector<Data *>> weights(layers,std::vector<Data *>(2+experts*2));
    auto gpuWeights=weights;
    FastllmCudaSetDevice(0);
    for (int l=0;l<layers;++l) for (int e=0;e<experts;++e) for (int p=0;p<2;++p) {
        auto w=std::make_unique<Data>(EXL3);w->blockM=p?1:2;w->blockK=4+2*l;
        w->Resize(p?std::vector<int>{hidden,inter}:std::vector<int>{2*inter,hidden});Fill(*w,rng);
        auto g=std::make_unique<Data>();g->CopyFrom(*w);g->ToDevice(CUDA,std::vector<int>{0});
        weights[l][2+2*e+p]=w.get();gpuWeights[l][2+2*e+p]=g.get();
        owned.push_back(std::move(w));gpuOwned.push_back(std::move(g));
    }
    setenv("FASTLLM_DSV41_MOE_CACHE_MODE","gpu",1);
    SetMoeCudaCacheBytes(16*(weights[1][2]->GetBytes()+weights[1][3]->GetBytes()+256));
    HybridMoeCachePlan plan(HybridMoeMath::DeepSeekV41,.01f);
    for (auto &w:weights) plan.Add(w,"cpu");
    Require(plan.Prepare([]{},true),"EXL3 pure cache preparation");
    std::map<std::tuple<int,int,int,int>, std::unique_ptr<PureMoeGraphTest>> graphs;
    int checks=0;
    double maxRelative=0;
    for (int topk:{3,6}) for (int rows:{1,2,5,6,8,9,10,12,17}) for (int turn=0;turn<3;++turn) {
        Data x(BFLOAT16,{rows,hidden}),ids(INT32,{rows,topk}),scores(FLOAT32,{rows,topk});
        for (auto *d:{&x,&ids,&scores})d->Allocate(false);
        for (int i=0;i<rows*hidden;++i) ((uint16_t*)x.cpuData)[i]=Float32ToBFloat16RNEBits((int(rng()%97)-48)/32.f);
        for (int r=0;r<rows;++r) for (int k=0;k<topk;++k) {
            ((int*)ids.cpuData)[r*topk+k]=(7*turn+3*r+(k==1 && r%2==0?0:k))%experts;
            ((float*)scores.cpuData)[r*topk+k]=float(int(rng()%17)-4)/16.f;
        }
        Data gx,gi,gs;
        gx.CopyFrom(x);gi.CopyFrom(ids);gs.CopyFrom(scores);
        for (auto *d:{&gx,&gi,&gs})d->ToDevice(CUDA,std::vector<int>{0});
        for (int l=0;l<layers;++l) {
            FastllmCudaSetDevice(0);
            std::vector<float> route(size_t(rows)*topk*hidden);
            for (int r=0;r<rows;++r) for (int k=0;k<topk;++k) {
                Data xi,ii,ss,y;
                xi.FakeFrom(gx,size_t(r)*hidden*2);xi.Resize({1,hidden});
                ii.FakeFrom(gi,size_t(r*topk+k)*4);ii.Resize({1,1});
                ss.FakeFrom(gs,size_t(r*topk+k)*4);ss.Resize({1,1});
                Require(FastllmCudaExl3Moe(xi,y,ii,ss,gpuWeights[l].data(),gpuWeights[l].size(),.01f),"EXL3 resident oracle");
                FastllmCudaCopyFromDeviceToHost(route.data()+size_t(r*topk+k)*hidden,y.cudaData,hidden*4);
            }
            for (int device:{0,1}) {
                FastllmCudaSetDevice(device);
                Data dx,di,ds,y;dx.CopyFrom(x);di.CopyFrom(ids);ds.CopyFrom(scores);
                for (auto *d:{&dx,&di,&ds})d->ToDevice(CUDA,std::vector<int>{device});
                uint64_t before[8]{},after[8]{};
                Require(fastllm_moe_cuda_cache_route_stats(device,before),"EXL3 stats before");
                int callbacks=0;
                FastllmCudaMoeRouteScope scope(turn%2?FastllmMoeRoutePhase::Draft:FastllmMoeRoutePhase::Verify);
                Require(RunHybridMoe(dx,di,ds,y,weights[l].data(),weights[l].size(),l,[&]{++callbacks;}),"EXL3 pure verifier");
                // Retain each shape across turns, while other shapes/layers
                // evict its records and share the bounded graph workspace.
                auto &graph = graphs[{topk,rows,l,device}];
                if (!graph) graph = std::make_unique<PureMoeGraphTest>();
                auto &replayed = graph->Run(dx,di,ds,weights[l].data(),weights[l].size(),l);
                y.CopyFrom(replayed);
                y.ToDevice(CPU);
                Require(callbacks==1 && y.dataType==BFLOAT16 && y.dims==x.dims,"EXL3 pure output contract");
                Require(fastllm_moe_cuda_cache_route_stats(device,after),"EXL3 stats after");
                Require(before[5]==after[5],"EXL3 pure verifier executed CPU experts");
                double error=0,norm=0;
                for (int r=0;r<rows;++r) {
                    std::vector<int> order(topk);for(int k=0;k<topk;++k)order[k]=k;
                    std::stable_sort(order.begin(),order.end(),[&](int a,int b){return ((int*)ids.cpuData)[r*topk+a]<((int*)ids.cpuData)[r*topk+b];});
                    for (int c=0;c<hidden;++c) {
                        float sum=0;
                        for(int k:order) sum+=route[size_t(r*topk+k)*hidden+c];
                        uint32_t av=uint32_t(((uint16_t*)y.cpuData)[r*hidden+c])<<16;
                        uint32_t bv=uint32_t(Float32ToBFloat16RNEBits(sum))<<16;
                        float a,b;std::memcpy(&a,&av,4);std::memcpy(&b,&bv,4);
                        Require(std::isfinite(a)&&std::isfinite(b),"EXL3 nonfinite verifier output");
                        error+=double(a-b)*(a-b);norm+=double(b)*b;
                    }
                }
                const double relative = std::sqrt(error/std::max(norm,1e-30));
                maxRelative=std::max(maxRelative,relative);
                if (!(relative < .008)) std::cerr << "EXL3 rows=" << rows << " topk=" << topk
                    << " turn=" << turn << " layer=" << l << " device=" << device << " relative=" << relative << std::endl;
                Require(relative<.008,"EXL3 pure verifier/oracle mismatch");
                ++checks;
            }
        }
    }
    graphs.clear();
    FastllmCudaReleaseMoeCache(weights[0].data(),weights[0].size());SetMoeCudaCacheBytes(0);
    std::cout<<"EXL3 PURE CACHE PASS: "<<checks<<" checks, max relative L2 "<<maxRelative<<", two GPUs, mixed bitrates, topk 3/6, bounded chunks and zero CPU experts"<<std::endl;
    return 0;
} catch(const std::exception &e) { std::cerr<<e.what()<<std::endl;return 1; } }
