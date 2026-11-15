#define main ExistingExl3MoeTests
#include "test_exl3_moe.cpp"
#undef main
#include "utils/hybrid_moe.h"
#include <atomic>
#include <thread>
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-moe-policy.h"
#include "devices/cuda/fastllm-cuda-moe-cache-stats.h"
#include "moe_cache_config.h"
#include "devices/numas/numasdevice.h"

static float ReadBf(const Data &d,int i) {
    uint32_t word=uint32_t(reinterpret_cast<const uint16_t *>(d.cpuData)[i])<<16;
    float f;std::memcpy(&f,&word,4);return f;
}
// Validate approximate CPU arithmetic independently, then use its per-route
// outputs for the strict CPU/GPU mixture check. This keeps stale cache records,
// wrong scores and graph workspace reuse visible at the original tolerance.
static void DecodeCpuOracle(const Data &input,float *routes,Data **weights,int weightCount,
        const int *ids,const float *scores,int topk,int layer) {
#ifdef USE_NUMAS
    if(!CanRunNumasMoeDecodeExperts(weights,weightCount))return;
    const int k=input.dims.back();std::vector<float> x(k),actual(topk*k);
    for(int i=0;i<k;++i)x[i]=ReadBf(input,i);
    NumasMoeDecodeExperts(x.data(),actual.data(),weights,ids,nullptr,topk,layer,scores,.01f);
    for(int at=0;at<topk;++at) {
        double error=0,norm=0;
        for(int c=0;c<k;++c) {
            const float a=routes[at*k+c],b=actual[at*k+c];
            Require(std::isfinite(a)&&std::isfinite(b),"Nonfinite CPU expert output");
            error+=double(a-b)*(a-b);norm+=double(a)*a;
        }
        Require(std::sqrt(error/std::max(norm,1e-30))<.02,"Approximate CPU expert exceeds L2 limit");
    }
    std::copy(actual.begin(),actual.end(),routes);
#endif
}
static void CheckVerifierRows(std::vector<std::vector<Data *>> &weights,
        std::vector<std::vector<Data *>> &gpuWeights, int k, int mid) {
    const int layers=weights.size(), experts=(weights[0].size()-2)/2, topk=3;
    std::mt19937 rng(981);
    for (int rows : {2, 3, 5, 7, 9, 12, 17, 24, 33, 65}) {
        Data x(BFLOAT16,{rows,k}),ids(INT32,{rows,topk}),scores(FLOAT32,{rows,topk});
        for (auto *d:{&x,&ids,&scores}) d->Allocate(false);
        std::memset(x.cpuData,0,x.GetBytes());std::memset(ids.cpuData,0,ids.GetBytes());std::memset(scores.cpuData,0,scores.GetBytes());
        Data gx,gi,gs;gx.CopyFrom(x);gi.CopyFrom(ids);gs.CopyFrom(scores);
        for(auto *d:{&gx,&gi,&gs})d->ToDevice(CUDA,{0});
        std::vector<Data> outputs(layers);
        std::vector<const void *> children;std::vector<void *> handles;
        for(int l=0;l<layers;++l) {
            void *child=nullptr;
            auto *handle=FastllmCudaCreateNativeMoeFullGraph(gx,gi,gs,outputs[l],weights[l].data(),weights[l].size(),l,nullptr,&child);
            Require(handle&&child,"EXL3 multirow full graph creation");children.push_back(child);handles.push_back(handle);
        }
        HybridMoeGraph graph;Require(graph.Compose(children,handles),"EXL3 multirow composition");
        double maxRelativeL2=0;
        uint64_t before[8]{};fastllm_moe_cuda_cache_route_stats(0,before);
        for(int turn=0;turn<10;++turn) {
            for(int i=0;i<rows*k;++i)reinterpret_cast<uint16_t *>(x.cpuData)[i]=Float32ToBFloat16RNEBits((int(rng()%97)-48)/32.f);
            for(int row=0;row<rows;++row)for(int at=0;at<topk;++at) {
                const int route=row*topk+at;
                // Alternate complete reuse and different experts across rows.
                reinterpret_cast<int *>(ids.cpuData)[route]=(turn+(turn%2?row:0)+at*3)%experts;
                reinterpret_cast<float *>(scores.cpuData)[route]=float(int(rng()%17)-4)/16.f;
            }
            FastllmCudaCopyFromHostToDevice(gx.cudaData,x.cpuData,x.GetBytes());
            FastllmCudaCopyFromHostToDevice(gi.cudaData,ids.cpuData,ids.GetBytes());
            FastllmCudaCopyFromHostToDevice(gs.cudaData,scores.cpuData,scores.GetBytes());
            HybridMoeDecodeStep step;step.Begin(weights[0].data(),weights[0].size(),topk);
            Require(graph.Launch(),"EXL3 multirow full graph replay");
            Require(cudaDeviceSynchronize()==cudaSuccess,"EXL3 multirow graph completion");step.Finish();
            for(int l=0;l<layers;++l) {
                std::vector<uint16_t> actual(rows*k);FastllmCudaCopyFromDeviceToHost(actual.data(),outputs[l].cudaData,rows*k*2);
                std::vector<float> cpu(rows*topk*k),gpu(rows*topk*k);
                std::vector<int> cpuOwners(rows*topk,-1);
                NumasMoeVerifyExpertsWithOverlap(reinterpret_cast<uint16_t *>(x.cpuData),cpu.data(),rows,
                    weights[l].data(),weights[l].size(),reinterpret_cast<int *>(ids.cpuData),cpuOwners.data(),
                    reinterpret_cast<float *>(scores.cpuData),topk,l,.01f,true,32,{},nullptr,true);
                for(int at=0;at<topk;++at) {
                    // The independent projection entry accepts compute tiles;
                    // compare every row of the larger scheduling batch.
                    for (int first=0;first<rows;first+=8) {
                        const int count=std::min(8,rows-first);
                        Data tile,oneId(INT32,{count,1}),oneScore(FLOAT32,{count,1}),one;
                        tile.FakeFrom(gx,size_t(first)*k*sizeof(uint16_t));tile.Resize({count,k});
                        oneId.Allocate(false);oneScore.Allocate(false);
                        for(int row=0;row<count;++row) {
                            reinterpret_cast<int *>(oneId.cpuData)[row]=reinterpret_cast<int *>(ids.cpuData)[(first+row)*topk+at];
                            reinterpret_cast<float *>(oneScore.cpuData)[row]=reinterpret_cast<float *>(scores.cpuData)[(first+row)*topk+at];
                        }
                        oneId.ToDevice(CUDA,{0});oneScore.ToDevice(CUDA,{0});
                        Require(FastllmCudaExl3Moe(tile,one,oneId,oneScore,gpuWeights[l].data(),gpuWeights[l].size(),.01f),"EXL3 multirow GPU route oracle");
                        one.ToDevice(CPU);
                        for(int row=0;row<count;++row)std::memcpy(gpu.data()+size_t((first+row)*topk+at)*k,reinterpret_cast<float *>(one.cpuData)+row*k,k*4);
                    }

                }
                auto check=[&](const uint16_t *observed) {
                    for(int row=0;row<rows;++row) {
                        std::vector<int> order(topk);for(int at=0;at<topk;++at)order[at]=row*topk+at;
                        std::stable_sort(order.begin(),order.end(),[&](int a,int b){return reinterpret_cast<int *>(ids.cpuData)[a]<reinterpret_cast<int *>(ids.cpuData)[b];});
                        double best=1e100,norm=0,bestNorm=0;
                        for(int mask=0;mask<(1<<topk);++mask) {
                            double error=0;norm=0;
                            for(int c=0;c<k;++c) {
                                float sum=0;
                                for(int route:order)sum+=(mask&(1<<(route%topk))?gpu:cpu)[size_t(route)*k+c];
                                uint32_t a=uint32_t(observed[row*k+c])<<16,b=uint32_t(Float32ToBFloat16RNEBits(sum))<<16;
                                float av,bv;std::memcpy(&av,&a,4);std::memcpy(&bv,&b,4);
                                Require(std::isfinite(av)&&std::isfinite(bv),"Nonfinite EXL3 multirow output");
                                error+=double(av-bv)*(av-bv);norm+=double(bv)*bv;
                            }
                            if(error<best){best=error;bestNorm=norm;}
                        }
                        const double relativeL2=std::sqrt(best/std::max(bestNorm,1e-30));
                        maxRelativeL2=std::max(maxRelativeL2,relativeL2);
                        // Grouped-window and single-expert projections have
                        // different BF16 rounding. The expanded deterministic
                        // inputs reach 1.232% in the pre-batching eight-row graph
                        // too; bound this existing difference at 1.5%.
                        Require(relativeL2<.015,"EXL3 multirow result differs from valid CPU/GPU routes");
                    }
                };
                check(actual.data());
                // The noncaptured verify/draft path must use the same batched
                // backend and preserve partials across sparse resident/staged calls.
                if(turn<2) {
                    Data eager;int callbacks=0;
                    FastllmCudaMoeRouteScope scope(turn?FastllmMoeRoutePhase::Draft:FastllmMoeRoutePhase::Verify);
                    HybridMoeDecodeStep eagerStep;eagerStep.Begin(weights[0].data(),weights[0].size(),topk);
                    Require(RunHybridMoe(gx,gi,gs,eager,weights[l].data(),weights[l].size(),l,[&]{++callbacks;}),"EXL3 multirow eager hybrid");
                    eager.ToDevice(CPU);eagerStep.Finish();Require(callbacks==1&&eager.dataType==BFLOAT16,"EXL3 multirow eager semantics");
                    check(reinterpret_cast<uint16_t *>(eager.cpuData));
                }
            }
        }
        uint64_t after[8]{};fastllm_moe_cuda_cache_route_stats(0,after);
        Require(after[4]>before[4]&&after[5]>before[5],"Multirow graph must exercise CPU and GPU routes");
        std::cout<<"EXL3 MULTIROW GRAPH PASS rows="<<rows<<" hidden="<<k<<" inter="<<mid<<" max_l2="<<maxRelativeL2<<" staged_routes="<<(after[4]-before[4]-(after[2]-before[2]))<<std::endl;
        graph.Reset();
    }
}

static void CheckHybrid() {
    constexpr int layers=2,experts=8,topk=3;
    const bool large=std::getenv("EXL3_TEST_LARGE")!=nullptr;
    const int k=large?5120:256,mid=large?2304:128;
    std::mt19937 rng(891);
    std::vector<std::unique_ptr<Data>> owned;
    std::vector<std::vector<Data *>> weights(layers,std::vector<Data *>(2+experts*2,nullptr));
    for(int l=0;l<layers;++l) for(int e=0;e<experts;++e) for(int part=0;part<2;++part) {
        auto w=std::make_unique<Data>(EXL3);w->blockM=part?1:2;w->blockK=4+2*l;
        w->Resize(part?std::vector<int>{k,mid}:std::vector<int>{2*mid,k});Fill(*w,rng);
        weights[l][2+e*2+part]=w.get();owned.push_back(std::move(w));
    }
    std::vector<std::unique_ptr<Data>> gpuOwned;
    std::vector<std::vector<Data *>> gpuWeights(layers,std::vector<Data *>(2+experts*2,nullptr));
    for(int l=0;l<layers;++l) for(int at=2;at<2+experts*2;++at) {
        auto w=std::make_unique<Data>();w->CopyFrom(*weights[l][at]);
        w->ToDevice(CUDA,std::vector<int>{0});gpuWeights[l][at]=w.get();gpuOwned.push_back(std::move(w));
    }
    // Exercise the common pageable arena/cache fallback without depending on
    // the address of canonical storage that registration is allowed to retire.
    if (std::getenv("EXL3_TEST_PAGEABLE")) setenv("FT_PINNED_WEIGHT", "0", 1);
    SetMoeCudaCacheBytes(6*(weights[1][2]->GetBytes()+weights[1][3]->GetBytes()+256));
    MoeCacheConfig config;config.minHeat=0;config.replacementMargin=0;config.maxReplacements=8;
    SetMoeCacheConfig(config);
    HybridMoeCachePlan plan(HybridMoeMath::DeepSeekV41,.01f);
    const bool numa=std::getenv("EXL3_TEST_NUMA")!=nullptr;
    for(auto &table:weights)plan.Add(table,numa?"numa":"cpu");
    auto registerWeights=[&] {
        if (!numa) return;
        std::vector<std::pair<Data*,std::string>> pending;
        for(auto &table:weights)for(int i=2;i<int(table.size());++i)
            pending.push_back({table[i],i%2?"linear":"linearSwiglu"});
        RegisterNumasMoeWeightBatch(pending);
    };
    Require(plan.Prepare(registerWeights,true),"EXL3 hybrid prepare");
    Data input(BFLOAT16,{1,k}),index(INT32,{1,topk}),score(FLOAT32,{1,topk});
    input.Allocate(false);index.Allocate(false);score.Allocate(false);
    std::vector<float> x(k);std::vector<float> perRoute(topk*k,-99);
    int callbacks=0;
    for(int turn=0;turn<40;++turn) {
        for(int c=0;c<k;++c) {
            float v=(int(rng()%63)-31)/32.f;
            reinterpret_cast<uint16_t *>(input.cpuData)[c]=Float32ToBFloat16RNEBits(v);x[c]=ReadBf(input,c);
        }
        for(int at=0;at<topk;++at) {
            reinterpret_cast<int *>(index.cpuData)[at]=(turn<20?at:experts-1-at);
            reinterpret_cast<float *>(score.cpuData)[at]=.2f+.15f*at;
        }
        HybridMoeDecodeStep step;step.Begin(weights[0].data(),weights[0].size(),topk);
        Require(bool(step),"EXL3 decode policy start");
        for(int l=0;l<layers;++l) {
            Data reference(BFLOAT16,{1,k});Exl3MoeCPU(input,reference,index,score,weights[l].data(),weights[l].size(),0,.01f);
            int skip[topk]={-1,-1,-1};
            Exl3MoeCPUExperts(x.data(),perRoute.data(),1,weights[l].data(),
                reinterpret_cast<int *>(index.cpuData),skip,reinterpret_cast<float *>(score.cpuData),topk,.01f,[&]{++callbacks;});
            // Check each weighted expert exactly, before reduction can contract
            // score*y+sum into a different FP32 rounding than a per-route buffer.
            for(int at=0;at<topk;++at) {
                Data oneIndex(INT32,{1,1}),oneScore(FLOAT32,{1,1}),one(FLOAT32,{1,k});
                oneIndex.Allocate(false);oneScore.Allocate(false);
                *reinterpret_cast<int *>(oneIndex.cpuData)=reinterpret_cast<int *>(index.cpuData)[at];
                *reinterpret_cast<float *>(oneScore.cpuData)=reinterpret_cast<float *>(score.cpuData)[at];
                Exl3MoeCPU(input,one,oneIndex,oneScore,weights[l].data(),weights[l].size(),0,.01f);
                for(int c=0;c<k;++c) {
                    float a=reinterpret_cast<float *>(one.cpuData)[c],b=perRoute[at*k+c];
                    if(a!=b) std::cout<<"route mismatch "<<turn<<"/"<<l<<"/"<<at<<"/"<<c<<" "<<std::hexfloat<<a<<" "<<b<<std::defaultfloat<<std::endl;
                    Require(a==b,"CPU per-route boundaries");
                }
            }
            DecodeCpuOracle(input,perRoute.data(),weights[l].data(),weights[l].size(),
                reinterpret_cast<int *>(index.cpuData),reinterpret_cast<float *>(score.cpuData),topk,l);
            Data gx,gi,gs,result;gx.CopyFrom(input);gi.CopyFrom(index);gs.CopyFrom(score);
            for(auto *v:{&gx,&gi,&gs})v->ToDevice(CUDA,std::vector<int>{0});
            { FastllmCudaMoeRouteScope scope(FastllmMoeRoutePhase::Decode);
              Require(RunHybridMoe(gx,gi,gs,result,weights[l].data(),weights[l].size(),l,[&]{++callbacks;}),"EXL3 hybrid execution"); }
            result.ToDevice(CPU);
            std::vector<float> gpuRoute(topk*k);
            for(int at=0;at<topk;++at) {
                Data ii(INT32,{1,1}),ss(FLOAT32,{1,1}),oo;
                ii.Allocate(false);ss.Allocate(false);
                *reinterpret_cast<int *>(ii.cpuData)=reinterpret_cast<int *>(index.cpuData)[at];
                *reinterpret_cast<float *>(ss.cpuData)=reinterpret_cast<float *>(score.cpuData)[at];
                ii.ToDevice(CUDA,std::vector<int>{0});ss.ToDevice(CUDA,std::vector<int>{0});
                Require(FastllmCudaExl3Moe(gx,oo,ii,ss,gpuWeights[l].data(),gpuWeights[l].size(),.01f),"GPU route oracle");
                oo.ToDevice(CPU);std::memcpy(gpuRoute.data()+at*k,oo.cpuData,k*sizeof(float));
            }
            double err=0,norm=0;
            for(int c=0;c<k;++c){double a=ReadBf(reference,c),b=ReadBf(result,c);err+=(a-b)*(a-b);norm+=a*a;}
            // Each route may use either backend. Enumerate the valid mixtures
            // so BF16 projection ties cannot mask a wrong cached record/score.
            double best=1e100;
            for(int mask=0;mask<(1<<topk);++mask) {
                double mixedError=0;
                for(int c=0;c<k;++c) {
                    float sum=0;
                    for(int order=0;order<topk;++order) {
                        int at=turn<20?order:topk-1-order;
                        sum+=(mask&(1<<at)?gpuRoute:perRoute)[at*k+c];
                    }
                    uint32_t word=uint32_t(Float32ToBFloat16RNEBits(sum))<<16;
                    float expect;std::memcpy(&expect,&word,4);
                    double delta=expect-ReadBf(result,c);mixedError+=delta*delta;
                }
                best=std::min(best,mixedError);
            }
            Require(std::isfinite(err)&&std::sqrt(best/std::max(norm,1e-30))<1e-5,"Hybrid output differs from all valid CPU/GPU mixtures");
        }
        step.Finish();
    }
    uint64_t stats[8]{};Require(fastllm_moe_cuda_cache_route_stats(0,stats),"hybrid stats");
    std::cout<<"HYBRID_STATS";for(auto v:stats)std::cout<<" "<<v;std::cout<<std::endl;
    Require(stats[2]>0 && stats[4]>0 && stats[5]>0,"Expected cache hits and both CPU/GPU execution");
    Require(callbacks==160,"CPU/GPU callback must run exactly once");
    for(int rows:{2,9,64}) for(int l=0;l<layers;++l) {
        Data x(BFLOAT16,{rows,k}),ids(INT32,{rows,topk}),scores(FLOAT32,{rows,topk}),reference(FLOAT32,{rows,k}),actual;
        x.Allocate(false);ids.Allocate(false);scores.Allocate(false);
        for(int i=0;i<rows*k;++i) reinterpret_cast<uint16_t *>(x.cpuData)[i]=Float32ToBFloat16RNEBits((int(rng()%63)-31)/32.f);
        for(int i=0;i<rows*topk;++i) { reinterpret_cast<int *>(ids.cpuData)[i]=(i+i/topk)%experts;reinterpret_cast<float *>(scores.cpuData)[i]=.2f+.15f*(i%topk); }
        x.ToDevice(CUDA,std::vector<int>{0});ids.ToDevice(CUDA,std::vector<int>{0});scores.ToDevice(CUDA,std::vector<int>{0});
        DoCudaMergeMOEV41Fallback(x,reference,ids,scores,gpuWeights[l].data(),gpuWeights[l].size(),0,.01f);
        int called=0;
        Require(RunHybridMoe(x,ids,scores,actual,weights[l].data(),weights[l].size(),l,[&]{++called;}),"EXL3 streamed prefill");
        reference.ToDevice(CPU);actual.ToDevice(CPU);double err=0,norm=0;
        for(int i=0;i<rows*k;++i) {double a=reinterpret_cast<float *>(reference.cpuData)[i],b=reinterpret_cast<float *>(actual.cpuData)[i];err+=(a-b)*(a-b);norm+=a*a;}
        Require(called==1&&std::sqrt(err/std::max(norm,1e-30))<1e-5,"Prefill differs from original CUDA projections");
    }
    {
        Data gx,gi,gs;gx.CopyFrom(input);gi.CopyFrom(index);gs.CopyFrom(score);
        for(auto *v:{&gx,&gi,&gs})v->ToDevice(CUDA,std::vector<int>{0});
        std::array<Data,layers> outputs;
        std::vector<const void *> children;std::vector<void *> handles;
        uint64_t before[8]{};fastllm_moe_cuda_cache_route_stats(0,before);
        for(int l=0;l<layers;++l) {
            void *child=nullptr;
            void *handle=FastllmCudaCreateNativeMoeFullGraph(gx,gi,gs,outputs[l],weights[l].data(),weights[l].size(),l,nullptr,&child);
            Require(handle&&child,"EXL3 full graph creation");children.push_back(child);handles.push_back(handle);
        }
        HybridMoeGraph graph;Require(graph.Compose(children,handles),"EXL3 full graph composition");
        for(int turn=0;turn<24;++turn) {
            for(int c=0;c<k;++c) reinterpret_cast<uint16_t *>(input.cpuData)[c]=Float32ToBFloat16RNEBits((int(rng()%63)-31)/32.f);
            for(int at=0;at<topk;++at) reinterpret_cast<int *>(index.cpuData)[at]=(turn+at)%experts;
            FastllmCudaCopyFromHostToDevice(gx.cudaData,input.cpuData,input.GetBytes());
            FastllmCudaCopyFromHostToDevice(gi.cudaData,index.cpuData,index.GetBytes());
            HybridMoeDecodeStep step;step.Begin(weights[0].data(),weights[0].size(),topk);
            Require(graph.Launch(),"EXL3 full graph replay");
            Require(cudaDeviceSynchronize()==cudaSuccess,"EXL3 full graph completion");
            step.Finish();
            std::vector<int> order(topk);for(int at=0;at<topk;++at)order[at]=at;
            std::sort(order.begin(),order.end(),[&](int a,int b){return reinterpret_cast<int *>(index.cpuData)[a]<reinterpret_cast<int *>(index.cpuData)[b];});
            for(int l=0;l<layers;++l) {
                std::vector<uint16_t> actual(k);FastllmCudaCopyFromDeviceToHost(actual.data(),outputs[l].cudaData,k*2);
                std::vector<float> cpu(topk*k),gpu(topk*k);
                for(int at=0;at<topk;++at) {
                    Data ii(INT32,{1,1}),ss(FLOAT32,{1,1}),one(FLOAT32,{1,k}),other;
                    ii.Allocate(false);ss.Allocate(false);
                    *reinterpret_cast<int *>(ii.cpuData)=reinterpret_cast<int *>(index.cpuData)[at];
                    *reinterpret_cast<float *>(ss.cpuData)=reinterpret_cast<float *>(score.cpuData)[at];
                    Exl3MoeCPU(input,one,ii,ss,weights[l].data(),weights[l].size(),0,.01f);
                    std::memcpy(cpu.data()+at*k,one.cpuData,k*4);
                    DecodeCpuOracle(input,cpu.data()+at*k,weights[l].data(),weights[l].size(),
                        reinterpret_cast<int *>(ii.cpuData),reinterpret_cast<float *>(ss.cpuData),1,l);

                    ii.ToDevice(CUDA,std::vector<int>{0});ss.ToDevice(CUDA,std::vector<int>{0});
                    Require(FastllmCudaExl3Moe(gx,other,ii,ss,gpuWeights[l].data(),gpuWeights[l].size(),.01f),"Graph GPU route oracle");
                    other.ToDevice(CPU);std::memcpy(gpu.data()+at*k,other.cpuData,k*4);
                }
                double best=1e100,norm=0;
                for(int mask=0;mask<(1<<topk);++mask) {
                    double error=0;norm=0;
                    for(int c=0;c<k;++c) {
                        float sum=0;for(int at:order)sum+=(mask&(1<<at)?gpu:cpu)[at*k+c];
                        uint32_t a=uint32_t(actual[c])<<16,b=uint32_t(Float32ToBFloat16RNEBits(sum))<<16;
                        float actualValue,expected;std::memcpy(&actualValue,&a,4);std::memcpy(&expected,&b,4);
                        error+=double(actualValue-expected)*(actualValue-expected);norm+=double(expected)*expected;
                    }
                    best=std::min(best,error);
                }
                Require(std::sqrt(best/std::max(norm,1e-30))<1e-5,"Graph replay reused stale inputs, routes or scratch");
            }
        }
        uint64_t after[8]{};fastllm_moe_cuda_cache_route_stats(0,after);
        Require(after[4]>before[4]&&after[5]>before[5],"Full graph must exercise both CPU and GPU");
        const auto staged=after[4]-before[4]-(after[2]-before[2]);
        std::cout<<"FULLGRAPH_STAGED_ROUTES "<<staged<<std::endl;
        if(large)Require(staged>0,"Large full graph must exercise staged DMA experts");
        // A finished request releases its private graph on a consumer thread
        // while the executor uses other handles in the shared expert mailbox.
        for(int turn=0;turn<32;++turn) {
            std::array<Data,layers> retiringOutputs;
            std::vector<const void *> retiringChildren;
            std::vector<void *> retiringHandles;
            for(int l=0;l<layers;++l) {
                void *child=nullptr;
                auto *handle=FastllmCudaCreateNativeMoeFullGraph(gx,gi,gs,retiringOutputs[l],
                    weights[l].data(),weights[l].size(),l,nullptr,&child);
                Require(handle&&child,"Concurrent graph retirement setup");
                retiringChildren.push_back(child);retiringHandles.push_back(handle);
            }
            HybridMoeGraph retiring;
            Require(retiring.Compose(retiringChildren,retiringHandles),"Concurrent graph retirement composition");
            std::atomic<bool> start{false};
            std::thread consumer([&] {
                FastllmCudaSetDevice(0);
                while(!start.load(std::memory_order_acquire))std::this_thread::yield();
                retiring.Reset();
            });
            HybridMoeDecodeStep step;step.Begin(weights[0].data(),weights[0].size(),topk);
            start.store(true,std::memory_order_release);
            const bool launched=graph.Launch();
            consumer.join();
            Require(launched&&cudaDeviceSynchronize()==cudaSuccess,"Replay while another request retires");
            step.Finish();
        }
        std::cout<<"EXL3 CONCURRENT GRAPH RETIREMENT PASS: 32 retire/replay races"<<std::endl;
        graph.Reset();std::cout<<"EXL3 FULL GRAPH PASS: two mixed-bitrate layers, changing inputs/routes, CPU/GPU mixtures"<<std::endl;
    }
    CheckVerifierRows(weights,gpuWeights,k,mid);
    FastllmCudaReleaseMoeCache(weights[0].data(),weights[0].size());
    std::cout<<"EXL3 HYBRID PASS: heterogeneous layers, cold/hot cache, changing routes, exact CPU boundaries"<<std::endl;
}
int main(){SetThreads(8);CheckHybrid();return 0;}
