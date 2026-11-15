#include "fastllm.h"
#include "utils.h"
#include "gguf.h"
#include "utils/hybrid_moe.h"
#include "moe_cache_config.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cpu/cpudevice.h"
#include "devices/cuda/cudadevice.h"
#include "devices/cuda/fastllm-cuda-v41-gguf.h"
#include "devices/cuda/fastllm-cuda-moe-policy.h"
#include "devices/cuda/fastllm-cuda-moe-cache-stats.h"
#include "devices/numas/numasdevice.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>
using namespace fastllm;
static void Need(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
static void Cu(cudaError_t e) { Need(e==cudaSuccess,cudaGetErrorString(e)); }
static void Gpu(Data &d) { d.ToDevice(CUDA,{0}); d.Allocate(false); }
static double L2(const std::vector<float> &a,const std::vector<float> &b) {
    double diff=0,norm=0;
    for(size_t i=0;i<a.size();++i) { Need(std::isfinite(a[i])&&std::isfinite(b[i]),"nonfinite output");diff+=double(a[i]-b[i])*(a[i]-b[i]);norm+=double(b[i])*b[i]; }
    return std::sqrt(diff/std::max(norm,1e-20));
}
struct Layer {
    static constexpr int hidden=512, inter=256, experts=8;
    int gt,dt;
    std::vector<std::unique_ptr<Data>> owned;
    std::vector<Data *> host,gpu;
    std::vector<std::vector<uint8_t>> canonical;
    Layer(int g,int d):gt(g),dt(d),host(2*(experts+1)),gpu(host.size()),canonical(host.size()) {
        std::mt19937 rng(517+g*37+d);
        for(int e=0;e<experts;++e)for(int part=0;part<2;++part) {
            const int at=2+2*e+part,type=part?dt:gt;
            const std::vector<int> dims=part?std::vector<int>{hidden,inter}:std::vector<int>{2*inter,hidden};
            auto packed=std::make_unique<Data>(DATA_GGUF_FORMAT,type,dims);
            if(type==GGML_TYPE_Q3_K) {
                std::vector<float> values(size_t(dims[0])*dims[1]);
                for(auto &v:values)v=(int(rng()%101)-50)/1024.f;
                packed->CreateFromOriData(WeightType::LINEAR,FLOAT32,(uint8_t*)values.data(),nullptr,nullptr);
            } else {
                packed->Allocate(false);
                for(size_t i=0;i<packed->GetBytes();++i)packed->cpuData[i]=rng()>>24;
                const size_t block=ggml_type_size(ggml_type(type));
                for(size_t i=0;i<packed->GetBytes();i+=block) {
                    if(type==GGML_TYPE_MXFP4)packed->cpuData[i]=120+rng()%3;
                    else {const uint16_t scale=float_to_half(.0005f*(1+rng()%4));std::memcpy(packed->cpuData+i,&scale,2);}
                }
            }
            canonical[at].assign(packed->cpuData,packed->cpuData+packed->GetBytes());
            std::unique_ptr<Data> weight;
            if(type==GGML_TYPE_MXFP4) {
                weight=std::make_unique<Data>(NVFP4);weight->blockK=1;weight->blockM=32;
                weight->Resize(dims);weight->UpdateUnitSize();weight->Allocate(false);
                auto *scales=GetNVFP4ScaleData(*weight);Need(scales,"native FP4 scale storage");
                for(size_t b=0;b<canonical[at].size()/17;++b) {
                    const auto *src=canonical[at].data()+17*b;auto *dst=weight->cpuData+16*b;scales[b]=src[0];
                    for(int j=0;j<8;++j) {dst[j]=(src[1+2*j]&15)|((src[2+2*j]&15)<<4);dst[8+j]=(src[1+2*j]>>4)|(src[2+2*j]&240);}
                }
            } else weight=std::move(packed);
            weight->isModelWeight=true;
            auto device=std::make_unique<Data>();
            device->blockK=weight->blockK;device->blockM=weight->blockM;
            device->CopyFrom(*weight);device->ToDevice(CUDA,{0},true);
            host[at]=weight.get();gpu[at]=device.get();owned.push_back(std::move(weight));owned.push_back(std::move(device));
        }
    }
    ~Layer(){FastllmCudaReleaseMoeCache(host.data(),host.size());}
    std::vector<float> Oracle(Data &x,Data &ids,Data &scores,int rows,int topk) {
        std::vector<float> out(size_t(rows)*topk*hidden);
        for(int row=0;row<rows;++row)for(int k=0;k<topk;++k) {
            Data xi,ii,ss,y;
            xi.FakeFrom(x,size_t(row)*hidden*2);xi.Resize({1,hidden});
            ii.FakeFrom(ids,size_t(row*topk+k)*4);ii.Resize({1,1});
            ss.FakeFrom(scores,size_t(row*topk+k)*4);ss.Resize({1,1});
            DoCudaMergeMOEV41Fallback(xi,y,ii,ss,gpu.data(),gpu.size(),0,.8f);
            Cu(cudaMemcpy(out.data()+size_t(row*topk+k)*hidden,y.cudaData,hidden*sizeof(float),cudaMemcpyDeviceToHost));
        }
        return out;
    }
    void CheckResidentBytes(bool &seen) {
        FastllmCudaMoePrefillResidents view;
        Need(FastllmCudaGetMoePrefillResidents(host.data(),experts,view,false),"resident inspection");
        if(gt==GGML_TYPE_MXFP4&&dt==GGML_TYPE_MXFP4) {
            Need(view.weightType==NVFP4_BLOCK_32_E8M0,"pure FP4 lost its native backend");return;
        }
        Need(view.weightType==DATA_GGUF_FORMAT&&view.gateType==gt&&view.downType==dt,"mixed cache format");
        for(size_t i=0;i<view.weights.size();++i)if(view.weights[i]) {
            std::vector<uint8_t> bytes(canonical[i+2].size());
            Cu(cudaMemcpy(bytes.data(),view.weights[i],bytes.size(),cudaMemcpyDeviceToHost));
            Need(bytes==canonical[i+2],"cache admission changed quantized bytes");seen=true;
        }
    }
};

// Compare each new canonical GPU projection with the established native
// per-expert path, independently of routing policy and NUMA arithmetic.
static void CheckProjection(Layer &l) {
    if(l.gt==GGML_TYPE_MXFP4&&l.dt==GGML_TYPE_MXFP4)return;
    const size_t gate=l.canonical[2].size(),down=l.canonical[3].size(),stride=gate+down;
    Data records(INT8,{int(stride*Layer::experts)});records.Allocate(false);
    for(int e=0;e<Layer::experts;++e) {
        std::memcpy(records.cpuData+size_t(e)*stride,l.canonical[2+2*e].data(),gate);
        std::memcpy(records.cpuData+size_t(e)*stride+gate,l.canonical[3+2*e].data(),down);
    }
    records.ToDevice(CUDA,{0},true);
    double worst=0;
    for(int rows:{1,3,17}) {
        constexpr int topk=3;const int routes=rows*topk;
        Data x(BFLOAT16,{rows,Layer::hidden}),ids(INT32,{rows,topk}),scores(FLOAT32,{rows,topk});
        x.Allocate(false);ids.Allocate(false);scores.Allocate(false);
        for(int i=0;i<rows*Layer::hidden;++i)((uint16_t*)x.cpuData)[i]=Float32ToBFloat16RNEBits(std::sin(i*.31f)*.5f);
        for(int i=0;i<routes;++i){((int*)ids.cpuData)[i]=(i*3)%Layer::experts;((float*)scores.cpuData)[i]=i%3==0?0:i%3==1?-.17f:.37f;}
        x.ToDevice(CUDA,{0},true);ids.ToDevice(CUDA,{0},true);scores.ToDevice(CUDA,{0},true);
        const auto reference=l.Oracle(x,ids,scores,rows,topk);
        const size_t bytes=FastllmCudaV41CanonicalGGUFCacheWorkspaceBytes(Layer::hidden,Layer::inter,rows,topk);
        Data workspace(INT8,{int(bytes)}),out(FLOAT32,{routes,Layer::hidden});Gpu(workspace);Gpu(out);
        FastllmCudaMoeGGUFCacheView view{};view.records=(uint8_t*)records.cudaData;view.routeSlots=(int32_t*)ids.cudaData;
        view.recordStride=stride;view.downOffset=gate;view.gateType=l.gt;view.downType=l.dt;
        view.hidden=Layer::hidden;view.inter=Layer::inter;view.workspace=workspace.cudaData;view.workspaceBytes=bytes;
        auto run=[&]{Need(FastllmCudaMoeV41CanonicalGGUFCacheCompute(x,view,(float*)scores.cudaData,topk,.8f,(float*)out.cudaData),"mixed projection dispatch");};
        run();Cu(cudaDeviceSynchronize());
        cudaGraph_t graph;cudaGraphExec_t exec;
        Cu(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));run();Cu(cudaStreamEndCapture(cudaStreamPerThread,&graph));
        Cu(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));Cu(cudaGraphLaunch(exec,cudaStreamPerThread));
        std::vector<float> actual(reference.size());Cu(cudaMemcpy(actual.data(),out.cudaData,actual.size()*4,cudaMemcpyDeviceToHost));
        const double error=L2(actual,reference);worst=std::max(worst,error);
        std::printf("PROJECTION gate=%d down=%d rows=%d L2=%.8g\n",l.gt,l.dt,rows,error);std::fflush(stdout);
        Need(error<.02,"mixed projection L2 exceeds 0.02");
        Cu(cudaGraphExecDestroy(exec));Cu(cudaGraphDestroy(graph));
    }
    std::printf("PASS projection gate=%d down=%d maxL2=%.8g\n",l.gt,l.dt,worst);
}

static void CheckGraphs(std::vector<std::unique_ptr<Layer>> &layers, bool unbound) {
    HostMoePlacementScope placement(unbound);
    std::vector<std::pair<Data*,std::string>> pending;
    HybridMoeCachePlan plan(HybridMoeMath::DeepSeekV41,.8f);
    uint64_t budget=0;
    for(auto &l:layers) {
        plan.Add(l->host,unbound ? "cpu" : "numa");budget+=2*(l->canonical[2].size()+l->canonical[3].size());
        for(size_t i=2;i<l->host.size();++i)pending.push_back({l->host[i],i%2?"linear":"linearSwiglu"});
    }
    SetMoeCudaCacheBytes(budget);
    Need(plan.Prepare([&]{RegisterHostMoeWeightBatch(pending,unbound);},true),"mixed cache family preparation");
    std::vector<const Data *> keys;
    for (const auto &layer : layers) keys.push_back(layer->host[2]);
    std::reverse(keys.begin(), keys.end());
    keys.push_back(keys.front());
    auto &root=*layers.front();bool seen=false;double worst=0;int replays=0;
    for(int rows:{1,2,4,7,17}) {
        constexpr int topk=3;
        Data x(BFLOAT16,{rows,Layer::hidden}),ids(INT32,{rows,topk}),scores(FLOAT32,{rows,topk});Gpu(x);Gpu(ids);Gpu(scores);
        std::vector<std::unique_ptr<Data>> out;
        for(size_t i=0;i<layers.size();++i){out.push_back(std::make_unique<Data>(BFLOAT16,std::vector<int>{rows,Layer::hidden}));Gpu(*out.back());}
        std::vector<uint16_t> hostX(rows*Layer::hidden),actual(hostX.size());
        std::vector<int> selected(rows*topk),cpu(selected.size(),-1);std::vector<float> scale(selected.size());
        auto update=[&](int pass){
            for(size_t i=0;i<hostX.size();++i)hostX[i]=Float32ToBFloat16RNEBits(std::sin(i*.17f+pass*.1f)*.5f);
            for(size_t i=0;i<selected.size();++i){selected[i]=((pass/4)*3+(i%3==2?0:i%3)+i/3)%Layer::experts;scale[i]=i%5==0?0.f:i%3==1?-.17f:.37f;}
            Cu(cudaMemcpy(x.cudaData,hostX.data(),hostX.size()*2,cudaMemcpyHostToDevice));
            Cu(cudaMemcpy(ids.cudaData,selected.data(),selected.size()*4,cudaMemcpyHostToDevice));
            Cu(cudaMemcpy(scores.cudaData,scale.data(),scale.size()*4,cudaMemcpyHostToDevice));
        };update(0);
        HybridMoeDecodeStep step;step.Begin(root.host.data(),root.host.size(),topk);Need(bool(step),"family step");step.Finish();
        uint64_t cacheStats[5]{};Need(fastllm_moe_cuda_cache_stats(0,cacheStats,false),"cache stats");
        Need(cacheStats[2]<=budget,"storage groups duplicated the GPU budget");
        Need(cacheStats[4]==layers.size()*Layer::experts,"not every group allocated a cache");
        Need(FastllmCudaLinkMoeCacheGroups(keys), "repeated family link after cache allocation");
        Need(!FastllmCudaLinkMoeCacheGroups({root.host[2]}), "live family membership changed");
        std::vector<void*> handles(layers.size()),children(layers.size());void *graph=nullptr,*exec=nullptr,*replay=nullptr;
        cudaGraph_t shared;cudaGraphNode_t empty;Cu(cudaGraphCreate(&shared,0));Cu(cudaGraphAddEmptyNode(&empty,shared,nullptr,0));
        Need(FastllmCudaGraphPrepareCaptureDevice()&&FastllmCudaGraphMemoryPoolBegin(),"parent graph pool");
        for(size_t i=0;i<layers.size();++i){auto &l=*layers[i];handles[i]=FastllmCudaCreateNativeMoeFullGraph(x,ids,scores,*out[i],l.host.data(),l.host.size(),int(i),shared,&children[i]);Need(handles[i]&&children[i],"mixed full graph creation");}
        std::vector<void*> reserved;Need(FastllmCudaGraphMemoryPoolEnd(reserved),"parent graph pool end");Cu(cudaGraphDestroy(shared));
        Need(FastllmCudaGraphCompose(const_cast<const void*const*>(children.data()),children.size(),&graph,&exec),"mixed graph compose");
        Need(FastllmCudaCreateNativeMoeReplayGraph(graph,handles.data(),handles.size(),&replay),"mixed graph replay preparation");
        for(int pass=0;pass<8;++pass) {
            update(pass);std::vector<std::vector<float>> expected;
            for(auto &l:layers) {
                std::vector<uint16_t> result(hostX.size());
                NumasMoeVerifyExperts(hostX.data(),result.data(),rows,l->host.data(),l->host.size(),selected.data(),cpu.data(),scale.data(),topk,0,.8f,false,32);
                expected.emplace_back(result.size());for(size_t i=0;i<result.size();++i)expected.back()[i]=BFloat16BitsToFloat32(result[i]);
            }
            step.Begin(root.host.data(),root.host.size(),topk);Need(bool(step),"replay family step");
            Need(FastllmCudaLaunchNativeMoeFullGraph(pass%2?replay:exec,handles.data(),handles.size(),pass%2==0),"mixed full graph launch");step.WaitCpu();step.Finish();
            for(size_t i=0;i<layers.size();++i) {
                Cu(cudaMemcpy(actual.data(),out[i]->cudaData,actual.size()*2,cudaMemcpyDeviceToHost));
                std::vector<float> result(actual.size());for(size_t j=0;j<actual.size();++j)result[j]=BFloat16BitsToFloat32(actual[j]);
                const double error=L2(result,expected[i]);worst=std::max(worst,error);
                if(error>=.03) { double a=0,b=0;for(size_t j=0;j<result.size();++j){a+=double(result[j])*result[j];b+=double(expected[i][j])*expected[i][j];}
                    std::fprintf(stderr,"GRAPH layer=%zu rows=%d pass=%d L2=%.9g norm=%.9g/%.9g first=%.9g/%.9g\n",i,rows,pass,error,a,b,result[0],expected[i][0]); }
                Need(error<.03,"hybrid graph/NUMA L2 exceeds 0.03");
                layers[i]->CheckResidentBytes(seen);++replays;
            }
        }
        FastllmCudaGraphExecDestroy(replay);FastllmCudaGraphExecDestroy(exec);FastllmCudaGraphDestroy(graph);
        for(auto *handle:handles)FastllmCudaReleaseNativeMoeFullGraph(handle);
        FastllmCudaGraphMemoryPoolRelease(reserved);
        std::printf("PASS full graph rows=%d, both storage groups, budget=%llu\n",rows,(unsigned long long)budget);std::fflush(stdout);
    }
    uint64_t routes[8]{};Need(fastllm_moe_cuda_cache_route_stats(0,routes),"route stats");
    Need(seen&&routes[4]>0&&routes[5]>0&&routes[6]>0,"missing GPU/CPU routes, cache hits or admitted payload");
    std::printf("PASS hybrid %d outputs maxL2=%.8g gpu=%llu cpu=%llu hits=%llu shards=%zu\n",replays,worst,(unsigned long long)routes[4],(unsigned long long)routes[5],(unsigned long long)routes[6],root.host[2]->numasData.size());
    // Removing one backend must not leave a dangling family reference.
    layers.back().reset();HybridMoeDecodeStep surviving;surviving.Begin(root.host.data(),root.host.size(),3);Need(bool(surviving),"surviving cache family");surviving.Finish();
}
int main(int argc, char **argv){try{
    if(FastllmCudaGetDeviceCount()<1)return 77;
    FastllmCudaSetDevice(0);SetThreads(4);
    std::vector<std::unique_ptr<Layer>> layers;
    for(auto pair:{std::pair<int,int>{GGML_TYPE_IQ3_XXS,GGML_TYPE_MXFP4},{GGML_TYPE_MXFP4,GGML_TYPE_IQ3_S},{GGML_TYPE_IQ3_S,GGML_TYPE_Q3_K},{GGML_TYPE_MXFP4,GGML_TYPE_MXFP4}}){layers.push_back(std::make_unique<Layer>(pair.first,pair.second));CheckProjection(*layers.back());}
    CheckGraphs(layers,argc>1&&std::strcmp(argv[1],"cpu")==0);std::puts("ALL PASS mixed expert cache");return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}catch(const char*e){std::fprintf(stderr,"FAIL %s\n",e);return 1;}}
