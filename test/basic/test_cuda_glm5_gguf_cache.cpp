#include "fastllm.h"
#include "executor.h"
#include "utils.h"
#include "gguf.h"
#include "../../src/devices/cuda/gguf_mmq/fastllm-gguf-moe-glm5.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-moe-cache-stats.h"
#include "devices/cuda/fastllm-cuda-moe-policy.h"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include "devices/numas/numasdevice.h"
#include "devices/numas/numas.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>
namespace fastllm {
void RegisterNumas(Data *, std::string);
void DoCudaMergeMOEFromCPU(Data &, Data &, Data &, Data &, Data &, Data &, Data &,
    Data **, Data **, float, bool, const std::unordered_set<int> &, bool, MoeGateType,
    bool, float, int, bool, bool);
}
using namespace fastllm;
static void Require(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
static void Cuda(cudaError_t e) { Require(e == cudaSuccess, cudaGetErrorString(e)); }
static float Bf(float x) { return RoundFloat32ToBFloat16RNE(x); }
static void Q8(std::vector<float> &v) {
    std::vector<block_q8_K> q(v.size() / QK_K);
    iqk_quantize_row_q8_K(v.data(), q.data(), v.size(), GGML_TYPE_Q8_K, GGML_TYPE_IQ2_XXS);
    for (size_t c = 0; c < v.size(); ++c) v[c] = q[c / QK_K].d * q[c / QK_K].qs[c % QK_K];
}
static float Compare(const std::vector<float> &a, const std::vector<float> &b, const char *label) {
    double error = 0, norm = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        Require(std::isfinite(a[i]), "nonfinite result");
        error += std::pow(double(a[i]) - b[i], 2); norm += double(b[i]) * b[i];
    }
    const float relative = std::sqrt(error / std::max(norm, 1e-15));
    if (relative > .008f) {
        std::fprintf(stderr, "%s relative L2 = %.8f\n", label, relative);
        throw std::runtime_error(label);
    }
    return relative;
}
static void OrdinaryNumas(Data &x, Data &ids, Data &scores, Data &out,
                          std::vector<Data *> &weights, int layer) {
    Data a,b,c,d,e;
    std::vector<Data *> biases(weights.size());
    static_cast<Executor *>(GetExecutor())->RunOnDevice("numa", "MergeMOE",
        {{"input",&x},{"index",&ids},{"score",&scores},{"output",&out},
         {"weights",reinterpret_cast<Data *>(weights.data())},
         {"biass",reinterpret_cast<Data *>(biases.data())},
         {"w1",&a},{"w2",&b},{"w3",&c},{"curInput",&d},{"curOutput",&e}},
        {{"swigluLimit",.125f}},{{"weights___batch",int(weights.size())},
         {"biass___batch",int(weights.size())},{"layer",layer},
         {"deepSeekV4Mode",1},{"activationQuantBlock",128}});
}

struct DirectLayoutFixture {
    int gateType, downType;
    size_t gateBytes, downBytes;
    std::vector<uint8_t> canonical;
    explicit DirectLayoutFixture(const std::vector<Data *> &weights) {
        gateType=weights[2]->ggmlType; downType=weights[3]->ggmlType;
        gateBytes=weights[2]->GetBytes(); downBytes=weights[3]->GetBytes();
        canonical.resize(2*(gateBytes+downBytes));
        for (int e=0;e<2;++e) {
            memcpy(canonical.data()+e*(gateBytes+downBytes),weights[2+2*e]->cpuData,gateBytes);
            memcpy(canonical.data()+e*(gateBytes+downBytes)+gateBytes,weights[3+2*e]->cpuData,downBytes);
        }
    }
    void Check(const std::vector<Data *> &weights, int hidden, int inter) const {
        const size_t stride=gateBytes+downBytes;
        std::vector<uint8_t> packed(canonical.size());
        for (int e=0;e<2;++e) for (int part=0;part<2;++part) {
            const auto &shards=weights[2+2*e+part]->numasData;
            const size_t bytes=(part ? downBytes : gateBytes)/shards.size();
            for (size_t n=0;n<shards.size();++n)
                memcpy(packed.data()+e*stride+(part ? gateBytes : 0)+n*bytes,shards[n],bytes);
        }
        const int gateStorage=weights[2]->ggmlType, downStorage=weights[3]->ggmlType;
        Require(FastllmCudaMoeGlm5GGUFCacheNumaSupported(gateStorage,downStorage,hidden,inter),
                "GLM NUMA layout capability missing");
        Data ordinary(INT8,{int(canonical.size())},CPU,const_cast<uint8_t *>(canonical.data()));
        Data native(INT8,{int(packed.size())},CPU,packed.data());
        ordinary.ToDevice(CUDA); native.ToDevice(CUDA);
        constexpr int topk=3;
        for (int rows : {1,2,3,4,7,9,17}) for (bool compact : {false,true}) {
            const int routes=rows*topk;
            std::vector<uint16_t> x(rows*hidden);
            std::vector<int32_t> slots, map;
            std::vector<float> scores(routes);
            for (int i=0;i<rows*hidden;++i) x[i]=Float32ToBFloat16RNEBits(float((i*17)%61-30)/23.f);
            for (int r=0;r<routes;++r) {
                scores[r]=r%3==0 ? 0.f : r%3==1 ? -.125f : .3125f;
                if (!compact || r%3!=1) {map.push_back(r);slots.push_back((r+r/topk)%2);}
            }
            std::reverse(map.begin(),map.end());
            Data input(BFLOAT16,{rows,hidden},CPU,x.data()), ds(FLOAT32,{routes},CPU,scores.data());
            Data di(INT32,{int(slots.size())},CPU,slots.data()), dm(INT32,{int(map.size())},CPU,map.data());
            input.ToDevice(CUDA);ds.ToDevice(CUDA);di.ToDevice(CUDA);dm.ToDevice(CUDA);
            std::vector<float> output[2];
            for (int variant=0;variant<2;++variant) {
                Data scratch(FLOAT32,{rows*(hidden+topk*inter)}), result(FLOAT32,{routes,hidden}), activation;
                scratch.ToDevice(CUDA);scratch.Allocate(false);result.ToDevice(CUDA);result.Allocate(false);
                Cuda(cudaMemset(result.cudaData,0,result.GetBytes()));
                FastllmCudaMoeGGUFCacheView view{
                    static_cast<const uint8_t *>(variant ? native.cudaData : ordinary.cudaData),
                    static_cast<const int32_t *>(di.cudaData),stride,gateBytes,gateType,downType,hidden,inter,
                    scratch.cudaData,size_t(scratch.GetBytes())};
                if (compact) {view.routeMap=static_cast<const int32_t *>(dm.cudaData);view.routeCount=map.size();}
                if (variant) {view.numaGateType=gateStorage;view.numaDownType=downStorage;}
                for (int reuse=0;reuse<2;++reuse) {
                    view.q8InputPrepared=reuse!=0;
                    Require(FastllmCudaMoeGlm5GGUFCacheCompute(input,activation,view,
                        static_cast<const float *>(ds.cudaData),topk,.125f,static_cast<float *>(result.cudaData)),
                        "GLM direct layout computation rejected");
                }
                result.ToDevice(CPU);
                output[variant].assign(reinterpret_cast<float *>(result.cpuData),
                    reinterpret_cast<float *>(result.cpuData)+routes*hidden);
            }
            Require(output[0]==output[1],"GLM NUMA layout differs from canonical GPU arithmetic");
        }
    }
};

static void CheckVerifyCpu(std::vector<Data *> &weights, int hidden, int layer) {
    constexpr int topk=3;
    Require(CanRunNumasMoeDecodeExperts(weights.data(), weights.size()),
            "registered verifier weights must support exact decode");
    for (int rows : {1,2,3,4,7,9,17,32}) for (int mode=0;mode<3;++mode) {
        std::vector<uint16_t> x(rows*hidden);
        std::vector<float> fx(rows*hidden), scores(rows*topk), expected(rows*topk*hidden,123.f), actual(expected);
        std::vector<int32_t> ids(rows*topk), owners(rows*topk);
        for (int i=0;i<rows*hidden;++i) {fx[i]=Bf(float((i*13)%67-33)/19.f);x[i]=Float32ToBFloat16RNEBits(fx[i]);}
        for (int r=0;r<rows*topk;++r) {
            ids[r]=r%2; scores[r]=r%3==0 ? 0.f : r%3==1 ? -.125f : .3125f;
            owners[r]=(mode==2 || (mode==1 && (r/topk==0 || ids[r]==0))) ? 0 : -1;
        }
        for (int row=0;row<rows;++row) NumasMoeDecodeExperts(fx.data()+row*hidden,
            expected.data()+row*topk*hidden,weights.data(),ids.data()+row*topk,
            owners.data()+row*topk,topk,layer,scores.data()+row*topk,.125f,128);
        const auto caller = std::this_thread::get_id();
        int submits=0;double cpuUs=-1;
        const bool delay=rows==3 && mode==1;
        const auto start=std::chrono::steady_clock::now();
        NumasMoeVerifyExpertsWithOverlap(x.data(),actual.data(),rows,weights.data(),weights.size(),
            ids.data(),owners.data(),scores.data(),topk,layer,.125f,true,128,[&] {
                Require(std::this_thread::get_id()==caller,"GPU submission moved off the caller thread");
                ++submits;if(delay) std::this_thread::sleep_for(std::chrono::milliseconds(40));
            },&cpuUs);
        const double wallUs=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
        const bool anyCpu=std::any_of(owners.begin(),owners.end(),[](int owner){return owner<0;});
        Require(submits==1 && (anyCpu ? cpuUs>0 : cpuUs==0),"verify callback/timing missing");
        Require(!delay || cpuUs<wallUs*.5,"verify CPU estimate includes callback-only stall");
        Require(actual==expected,"multi-row scored CPU differs from sequential decode or overwrites GPU routes");
        std::fill(actual.begin(),actual.end(),123.f);
        submits=0;
        NumasMoeVerifyExpertsWithOverlap(x.data(),actual.data(),rows,weights.data(),weights.size(),
            ids.data(),owners.data(),scores.data(),topk,layer,.125f,true,128,[&] {
                Require(std::this_thread::get_id()==caller,"prepared GPU submission moved threads");
                ++submits;
            },&cpuUs,true);
        Require(submits==1 && actual==expected,"prepared verifier changes CPU arithmetic or route ownership");
        if (rows>1 && mode==1) {
            std::fill(actual.begin(),actual.end(),123.f);
            bool caught=false;
            try {
                NumasMoeVerifyExpertsWithOverlap(x.data(),actual.data(),rows,weights.data(),weights.size(),
                    ids.data(),owners.data(),scores.data(),topk,layer,.125f,true,128,
                    [] {throw std::runtime_error("expected submit failure");},&cpuUs,true);
            } catch (const std::runtime_error &error) {
                caught=std::string(error.what())=="expected submit failure";
            }
            Require(caught && actual==expected,"GPU failure did not drain all CPU rows before returning");
        }
    }
}

enum class ExpertCacheTestMode { Parallel, Verify, VerifyNoCache, VerifyFullCache };

static void CheckExpertCache(std::vector<Data *> *weights, int tables, int hidden, int inter,
                             ExpertCacheTestMode mode) {
    constexpr int topk=6;
    const bool single=mode!=ExpertCacheTestMode::Parallel;
    const bool noCache=mode==ExpertCacheTestMode::VerifyNoCache;
    const bool fullCache=mode==ExpertCacheTestMode::VerifyFullCache;
    const int experts=(weights[0].size()-2)/2;
    const std::vector<int> devices=single ? std::vector<int>{0} : std::vector<int>{1,0};
    const std::vector<int> noDevices;
    std::vector<std::unique_ptr<Data>> records;
    std::vector<size_t> strides, gateBytes;
    for(int t=0;t<tables;++t) {
        const size_t gate=weights[t][2]->GetBytes(), down=weights[t][3]->GetBytes(), stride=gate+down;
        std::vector<uint8_t> packed(experts*stride);
        for(int e=0;e<experts;++e) for(int part=0;part<2;++part) {
            const auto &shards=weights[t][2+2*e+part]->numasData;
            const size_t bytes=(part ? down : gate)/shards.size();
            for(size_t n=0;n<shards.size();++n)
                std::memcpy(packed.data()+e*stride+(part ? gate : 0)+n*bytes,shards[n],bytes);
        }
        auto data=std::make_unique<Data>(INT8,std::vector<int>{int(packed.size())},CPU,packed.data());
        data->ToDevice(CUDA,std::vector<int>{0}); records.push_back(std::move(data));
        strides.push_back(stride);gateBytes.push_back(gate);
    }
    bool allHit=false,mixed=false,evicted=false,sawCpu=false,sawStaged=false;
    uint64_t previousUploads=0;
    for(int step=0;step<96;++step) {
        const int t=step%tables, rows=1+(step/tables)%9;
        const int origin=single ? 0 : step%2, count=rows*topk, first=step<48 ? 0 : 12;
        std::vector<uint16_t> x(rows*hidden);
        std::vector<int32_t> ids(count);
        std::vector<float> scores(count);
        for(int i=0;i<rows*hidden;++i)x[i]=Float32ToBFloat16RNEBits((i/hidden)%3==1 ? 0.f : float((i*17)%67-33)/21.f);
        for(int row=0;row<rows;++row)for(int k=0;k<topk;++k) {
            ids[row*topk+k]=first+(k+row)%8;
            scores[row*topk+k]=k==0 ? 0.f : k==1 ? -.125f : .3125f;
        }
        for(int row=0;row<rows;++row) ids[row*topk+2]=ids[row*topk+3];
        // Same packed records and complete reduction dimension as the ordinary
        // unsharded GPU adapter. All-hit output must be bitwise identical.
        Cuda(cudaSetDevice(0));
        Data rx(BFLOAT16,{rows,hidden},CPU,x.data()),ri(INT32,{count},CPU,ids.data()),
             rs(FLOAT32,{count},CPU,scores.data()),scratch(FLOAT32,{rows*(hidden+topk*inter)}),
             per(FLOAT32,{count,hidden}),activation;
        rx.ToDevice(CUDA,std::vector<int>{0});
        ri.ToDevice(CUDA,std::vector<int>{0});
        rs.ToDevice(CUDA,std::vector<int>{0});
        scratch.ToDevice(CUDA,std::vector<int>{0});scratch.Allocate(false);
        per.ToDevice(CUDA,std::vector<int>{0});per.Allocate(false);
        const int gate=weights[t][2]->ggmlType,down=weights[t][3]->ggmlType;
        auto ordinary=[](int type) {
            switch(type){case GGML_TYPE_IQ2_XXS_R4:return int(GGML_TYPE_IQ2_XXS);
                case GGML_TYPE_IQ2_S_R4:return int(GGML_TYPE_IQ2_S);
                case GGML_TYPE_IQ3_XXS_R4:return int(GGML_TYPE_IQ3_XXS);default:return type;}
        };
        FastllmCudaMoeGGUFCacheView v{static_cast<uint8_t *>(records[t]->cudaData),
            static_cast<int32_t *>(ri.cudaData),strides[t],gateBytes[t],ordinary(gate),ordinary(down),
            hidden,inter,scratch.cudaData,size_t(scratch.GetBytes())};
        v.numaGateType=gate;v.numaDownType=down;
        Require(FastllmCudaMoeGlm5GGUFCacheCompute(rx,activation,v,static_cast<float *>(rs.cudaData),topk,.125f,
            static_cast<float *>(per.cudaData)),"GGUF oracle rejected");
        per.ToDevice(CPU);std::vector<uint16_t> expected(rows*hidden);
        for(int row=0;row<rows;++row) {
            std::vector<int> order(topk);for(int k=0;k<topk;++k)order[k]=k;
            std::stable_sort(order.begin(),order.end(),[&](int a,int b){return ids[row*topk+a]<ids[row*topk+b];});
            for(int c=0;c<hidden;++c) {
                float sum=0;for(int k:order)sum+=reinterpret_cast<float *>(per.cpuData)[(row*topk+k)*hidden+c];
                expected[row*hidden+c]=Float32ToBFloat16RNEBits(sum);
            }
        }
        Cuda(cudaSetDevice(origin));
        Data input(BFLOAT16,{rows,hidden},CPU,x.data()),index(INT32,{rows,topk},CPU,ids.data()),
             score(FLOAT32,{rows,topk},CPU,scores.data()),output;
        input.ToDevice(CUDA,std::vector<int>{origin});
        index.ToDevice(CUDA,std::vector<int>{origin});
        score.ToDevice(CUDA,std::vector<int>{origin});
        // Exercise both implicit and explicit device selection for every quantization pair.
        const auto &requestedDevices=single && (step/tables)%2 ? noDevices : devices;
        void *state=FastllmCudaBeginMoeDecode(weights[t].data(),weights[t].size(),topk,
            requestedDevices.empty() ? nullptr : &requestedDevices);
        Require(state!=nullptr,"GGUF cache scope unavailable");
        uint64_t before[2][8]{},after[2][8]{},beforeCache[2][5]{};
        for(int d:devices) {
            Require(fastllm_moe_cuda_cache_route_stats(d,before[d]),"GGUF before routes");
            Require(fastllm_moe_cuda_cache_stats(d,beforeCache[d],false),"GGUF before cache counters");
        }
        Cuda(cudaSetDevice(origin));int callbacks=0;
        Require(FastllmCudaMergeMOEHybridOnDevices(input,index,score,output,weights[t].data(),weights[t].size(),
            t,requestedDevices,
            [&]{++callbacks;Cuda(cudaSetDevice(single ? origin : 1-origin));}),"GGUF cache dispatch rejected");
        Require(callbacks==1 && FastllmCudaGetDevice()==origin,"GGUF callback or device restoration");
        FastllmCudaEndMoeDecode(state);
        uint64_t routes=0,hits=0,computed=0,queries=0,queryHits=0;
        for(int d:devices) {
            Require(fastllm_moe_cuda_cache_route_stats(d,after[d]),"GGUF after routes");
            routes+=after[d][1]-before[d][1];hits+=after[d][2]-before[d][2];
            computed+=after[d][4]-before[d][4]+after[d][5]-before[d][5];
            sawCpu|=after[d][5]>before[d][5];
            sawStaged|=after[d][4]-before[d][4]>after[d][2]-before[d][2];
            uint64_t cache[5]{};
            Require(fastllm_moe_cuda_cache_stats(d,cache,false),"GGUF cache counters");
            queryHits+=cache[0]-beforeCache[d][0];
            queries+=cache[0]-beforeCache[d][0]+cache[1]-beforeCache[d][1];
            if(single) {
                if(noCache) Require(cache[2]==0 && cache[3]==0 && cache[0]==0,
                    "zero-cache verify allocated resident experts or reported hits");
                else {
                    Require(cache[2]>0 && cache[3]>0,"verify cache is empty");
                    Require(fullCache ? cache[3]==cache[4] : cache[3]<cache[4],
                        "verify cache capacity does not match the test mode");
                }
            }
        }
        Require(routes==count && computed==count && hits<=routes,"GGUF duplicated or lost logical routes");
        Require(queries==routes && queryHits==hits,"GGUF cache query counters differ from logical routes");
        output.ToDevice(CPU);
        if(hits==routes) {
            allHit=true;
            Require(std::memcmp(output.cpuData,expected.data(),expected.size()*2)==0,"GGUF all-hit differs from unsharded GPU output");
        } else {
            mixed|=hits>0;
            std::vector<float> a(expected.size()),b(expected.size());
            for(size_t i=0;i<a.size();++i){a[i]=BFloat16BitsToFloat32(reinterpret_cast<uint16_t *>(output.cpuData)[i]);b[i]=BFloat16BitsToFloat32(expected[i]);}
            Compare(a,b,"GGUF CPU/miss/cached mixture vs GPU oracle");
        }
        if(single) continue;
        uint64_t stats[2][6]{};
        for(int d=0;d<2;++d)Require(fastllm_moe_cuda_cache_ep_stats(d,stats[d]),"EP physical counters");
        Require(stats[0][0]==2 && stats[1][0]==2 && stats[0][1]==stats[1][1] &&
            stats[0][3]==stats[1][3],"EP cache capacities diverged");
        for (int d=0;d<2;++d) {
            Cuda(cudaSetDevice(d));
            FastllmCudaMoePrefillResidents residents;
            Require(FastllmCudaGetMoePrefillResidents(weights[t].data(),experts,residents,false) &&
                residents.ownerCount==2 && residents.ownerRank==1-d && residents.nativeGlm,
                "EP prefill view lost native ownership");
            for(int e=0;e<experts;++e) if(residents.weights[e*2])
                Require(e%2==1-d && residents.weights[e*2+1],"EP expert stored on the wrong device");
        }
        Require(stats[0][1]<uint64_t(tables*experts),"EP eviction test unexpectedly fits all experts");
        if(step==47) {
            Require(stats[0][2]==stats[0][1],"EP replacement test did not fill the cache");
            previousUploads=stats[0][5];
        }
        if(step==95)evicted=stats[0][5]>previousUploads && stats[0][4]>0;
    }
    if(single) {
        if(!noCache) Require(allHit,"verify test missed all-hit execution");
        if(!fullCache) Require(sawCpu && sawStaged,"verify test missed CPU or streamed GPU routes");
        if(!noCache && !fullCache) Require(mixed,"verify test missed mixed hit/miss execution");
    } else Require(allHit && mixed && evicted,"EP test missed hits, mixed routes or replacement");
    for(int d:devices) {
        uint64_t cache[5]{};
        Require(fastllm_moe_cuda_cache_stats(d,cache,true),"GGUF counter reset");
        Require(fastllm_moe_cuda_cache_stats(d,cache,false) && cache[0]==0 && cache[1]==0 &&
            (noCache ? cache[2]==0 && cache[3]==0 : cache[2]>0 && cache[3]>0),
            "cache reset changed capacity or left stale counts");
    }
    std::puts(single ? "PASS: GLM GGUF verify, rows 1-9, output, routing and cache capacity" :
        "PASS: GLM GGUF EP cache, whole experts, modulo ownership, frequency, replacement, rows 1-9, bitwise all-hit");
}

static void CheckExpertPrefill(std::vector<Data *> *tables, int count, int hidden, bool numa=false) {
    constexpr int topk=6;
    const int experts=(tables[0].size()-2)/2;
    const std::vector<int> devices{1,0};
    for(int rows : (numa ? std::vector<int>{33,129} : std::vector<int>{2,33,129})) {
        std::vector<std::vector<uint16_t>> expected(2*count);
        uint64_t uploads[2]{};
        for(int pass=0;pass<3;++pass) {
            if(pass==1) {
                Require(FastllmCudaPrepareMoeExpertCache(tables[0].data(),tables[0].size(),devices),
                    "EP prefill ownership setup failed");
            }
            for(int t=0;t<count;++t) for(int d=0;d<2;++d) {
                Cuda(cudaSetDevice(d));
                Data input(BFLOAT16,{rows,hidden}),ids(INT32,{rows,topk}),scores(FLOAT32,{rows,topk});
                input.Allocate();ids.Allocate();scores.Allocate();
                for(int i=0;i<rows*hidden;++i)
                    reinterpret_cast<uint16_t *>(input.cpuData)[i]=Float32ToBFloat16RNEBits(float(i%37-18)/23.f);
                std::unordered_set<int> selected;
                for(int row=0;row<rows;++row)for(int k=0;k<topk;++k) {
                    const int e=(row+k)%8;
                    reinterpret_cast<int *>(ids.cpuData)[row*topk+k]=e;
                    reinterpret_cast<float *>(scores.cpuData)[row*topk+k]=k==0 ? 0.f : k==1 ? -.125f : .3125f;
                    selected.insert(e+1);
                }
                input.ToDevice(CUDA,std::vector<int>{d});
                if(pass==1 && t==0) {
                    // The prefill speed estimator invokes ordinary GGUF math
                    // on the same table. Its restored payload must not replace
                    // native records, whether the expert is cold or resident.
                    FastllmCudaMoePrefillResidents view;
                    Require(FastllmCudaGetMoePrefillResidents(tables[t].data(),experts,view,true),"probe cache unavailable");
                    uint64_t before[6]{},after[6]{};
                    Require(fastllm_moe_cuda_cache_ep_stats(d,before),"probe before stats");
                    Data px(BFLOAT16,{65,hidden}),pi(INT32,{65,1}),ps(FLOAT32,{65,1}),po(BFLOAT16,{65,hidden}),pa,pb,pc;
                    px.Allocate();pi.Allocate();ps.Allocate();
                    for(int j=0;j<65*hidden;++j)reinterpret_cast<uint16_t *>(px.cpuData)[j]=Float32ToBFloat16RNEBits(float(j%17-8)/23.f);
                    for(int j=0;j<65;++j){reinterpret_cast<int *>(pi.cpuData)[j]=0;reinterpret_cast<float *>(ps.cpuData)[j]=1;}
                    px.ToDevice(CUDA,std::vector<int>{d});std::vector<Data*> biases(tables[t].size());
                    DoCudaMergeMOEFromCPU(px,po,pi,ps,pa,pb,pc,tables[t].data(),biases.data(),0,true,{1},true,
                        MoeGateSwiglu,false,0,128,false,true);
                    Require(fastllm_moe_cuda_cache_ep_stats(d,after) && before[5]==after[5],
                        "generic GGUF probe polluted native expert cache");
                }
                Data output(BFLOAT16,{rows,hidden}),w1,w2,w3;
                std::vector<Data *> biases(tables[t].size());
                // An unregistered table key bypasses cache lookup for the
                // reference while borrowing the same original NUMA weights.
                const auto &source=*tables[t][2];
                Data key(DATA_GGUF_FORMAT,source.ggmlType,source.dims);
                key.isFake=true;key.IsRepacked=source.IsRepacked;
                key.isGGUFData=source.isGGUFData;key.isModelWeight=source.isModelWeight;
                key.forceGGUFFp32Dequant=source.forceGGUFFp32Dequant;
                key.numasData=source.numasData;
                auto uncached=tables[t];uncached[2]=&key;
                if(numa) {
                    NumasMoeCudaAssistScope assist(&devices);
                    OrdinaryNumas(input,ids,scores,output,pass ? tables[t] : uncached,t);
                }
                else DoCudaMergeMOEFromCPU(input,output,ids,scores,w1,w2,w3,
                    pass ? tables[t].data() : uncached.data(),biases.data(),
                    0,true,selected,true,MoeGateSwiglu,true,.125f,128,false,true);
                Cuda(cudaStreamSynchronize(cudaStreamPerThread));
                std::vector<uint16_t> actual(rows*hidden);
                output.ToDevice(CPU);
                std::memcpy(actual.data(),output.cpuData,actual.size()*2);
                Cuda(cudaSetDevice(d));
                for(auto value:actual) Require((value & 0x7f80)!=0x7f80,"nonfinite EP prefill output");
                if(!pass) {expected[t*2+d]=actual;continue;}
                if(numa) {
                    std::vector<float> a(actual.size()),b(actual.size());
                    for(size_t i=0;i<a.size();++i){a[i]=BFloat16BitsToFloat32(actual[i]);b[i]=BFloat16BitsToFloat32(expected[t*2+d][i]);}
                    Compare(a,b,"EP NUMA prefill differs from streamed GPU output");
                } else Require(actual==expected[t*2+d],"EP prefill cache changed CUDA output");
                FastllmCudaMoePrefillResidents view;
                Require(FastllmCudaGetMoePrefillResidents(tables[t].data(),experts,view,false) && view.nativeGlm,
                    "EP prefill residents missing");
                int resident=0;
                for(int e=0;e<experts;++e)if(view.weights[e*2]) {
                    ++resident;Require(e%2==1-d,"EP prefill admitted another GPU's expert");
                    for(int part=0;part<2;++part) {
                        const auto &w=*tables[t][(e+1)*2+part];
                        std::vector<uint8_t> bytes(w.GetBytes());
                        Cuda(cudaMemcpy(bytes.data(),view.weights[e*2+part],bytes.size(),cudaMemcpyDeviceToHost));
                        const size_t shardBytes=bytes.size()/w.numasData.size();
                        for(size_t n=0;n<w.numasData.size();++n)
                            Require(!memcmp(bytes.data()+n*shardBytes,w.numasData[n],shardBytes),
                                "EP prefill cache changed native NUMA storage");
                    }
                }
                Require(resident>0,"EP prefill admitted no experts");
            }
            if(pass)for(int d=0;d<2;++d) {
                uint64_t stats[6]{};Require(fastllm_moe_cuda_cache_ep_stats(d,stats),"EP prefill stats");
                if(pass==1){uploads[d]=stats[5];Require(uploads[d]>0,"EP prefill uploaded no bytes");}
                else Require(stats[5]==uploads[d],"EP prefill reuploaded resident weights");
            }
        }
    }
    std::puts(numa ?
        "PASS: GLM GGUF EP prefill, NUMA/two-GPU integration, native bytes, rows 33/129" :
        "PASS: GLM GGUF EP prefill, reversed devices, native bytes, bitwise cached output, rows 2/33/129");
}

int main(int argc, char **argv) {
    try {
        const std::string option=argc>1 ? argv[1] : "";
        const bool numaPrefill=option=="--ep-numa-prefill";
        const bool expertPrefill=numaPrefill || option=="--ep-prefill";
        const bool expertCache=expertPrefill || option=="--ep-cache";
        const bool resident=option=="--resident";
        ExpertCacheTestMode cacheMode=ExpertCacheTestMode::Parallel;
        if(option=="--verify") cacheMode=ExpertCacheTestMode::Verify;
        else if(option=="--verify-no-cache") cacheMode=ExpertCacheTestMode::VerifyNoCache;
        else if(option=="--verify-full-cache") cacheMode=ExpertCacheTestMode::VerifyFullCache;
        const bool verify=cacheMode!=ExpertCacheTestMode::Parallel;
        const bool noCache=option=="--no-cache" || cacheMode==ExpertCacheTestMode::VerifyNoCache;
        const bool frequency=noCache || option=="--frequency";
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices || (expertCache && devices<2)) {
            std::puts("FASTLLM_TEST_SKIP_NO_DEVICE: CUDA unavailable"); return 0;
        }
        constexpr int hidden=512, experts=24, topk=6, tables=4;
        const int inter=resident ? 512 : 256;
        SetThreads(8);
        // Exercise both prefill GPU workers without relying on timing-based
        // decisions from the speed estimator for these small test matrices.
        if(numaPrefill) setenv("FT_EXPERT_LIMIT","1",1);
        setenv("FASTLLM_GLM5_MOE_CACHE_PREFETCH",frequency ? "1" : "0",1);
        std::mt19937 rng(71443);
        std::vector<std::unique_ptr<Data>> owned;
        std::vector<Data *> weights[tables];
        std::vector<std::vector<float>> dense[tables];
        FastllmCudaMoeCacheLayer layers[tables];
        std::vector<std::unique_ptr<Data>> tpOwned;
        std::vector<Data *> tpWeights[tables][2];
        size_t stride=0;
        std::vector<DirectLayoutFixture> directFixtures;
        for (int t=0;t<tables;++t) {
            weights[t].resize(2*(experts+1)); dense[t].resize(2*experts);
            for (int e=0;e<experts;++e) for (int part=0;part<2;++part) {
                const auto type=part ? (t%2 ? GGML_TYPE_IQ4_XS : GGML_TYPE_IQ3_XXS) :
                                      (t<2 ? GGML_TYPE_IQ2_XXS : GGML_TYPE_IQ2_S);
                const int rows=part ? hidden : 2*inter, cols=part ? inter : hidden;
                auto w=std::make_unique<Data>(DATA_GGUF_FORMAT,type,std::vector<int>{rows,cols});
                w->isModelWeight=true; w->isGGUFData=true;
                w->forceGGUFFp32Dequant=expertPrefill && t%2; w->Allocate(false);
                for (size_t i=0;i<w->GetBytes();++i) w->cpuData[i]=rng()>>24;
                const size_t block=ggml_type_size(type);
                for (size_t off=0;off<w->GetBytes();off+=block) {
                    const uint16_t scale=float_to_half(std::ldexp(resident ? (1.f + float(rng()%37)/64.f) : 1.f,-12+int(rng()%3)));
                    memcpy(w->cpuData+off,&scale,2);
                }
                auto &matrix=dense[t][2*e+part];matrix.resize(rows*cols);
                ggml_type_to_float(type)(w->cpuData,matrix.data(),matrix.size());
                if (type==GGML_TYPE_IQ4_XS) for (auto &v:matrix) v=Bf(v);
                weights[t][2*(e+1)+part]=w.get();owned.push_back(std::move(w));
            }
            stride=std::max(stride,weights[t][2]->GetBytes()+weights[t][3]->GetBytes());
            layers[t]={weights[t].data(),int(weights[t].size()),false,.125f,true};
            if (!resident) directFixtures.emplace_back(weights[t]);
        }
        stride=(stride+127)/128*128;
        unsetenv("FASTLLM_MOE_CUDA_CACHE_BYTES_0");
        unsetenv("FASTLLM_MOE_CUDA_CACHE_BYTES_1");
        int cacheSlots=frequency ? 64 : 16;
        if(verify) cacheSlots=32;
        if(cacheMode==ExpertCacheTestMode::VerifyFullCache) cacheSlots=tables*experts;
        SetMoeCudaCacheBytes(noCache ? 0 : stride*cacheSlots);
        if (resident) {
            for (int t=0;t<tables;++t) for (size_t i=2;i<weights[t].size();++i) {
                auto &w = *weights[t][i];
                w.ToDevice(CUDA, std::vector<int>{(t/2) % std::min(2, devices)});
                Require(w.cudaData && !w.cpuData && w.numasData.empty(),
                        "resident expert retained host storage");
            }
            if (devices >= 2) for (int t=0;t<tables;++t) {
                for (int r=0;r<2;++r) tpWeights[t][r].resize(weights[t].size());
                for (size_t i=2;i<weights[t].size();++i) {
                    auto source=std::make_unique<Data>();
                    source->CopyFrom(*weights[t][i]); source->ToDevice(CPU);
                    source->isModelWeight=true;
                    const int axis=i%2 ? 1 : 0;
                    DivisionScheme scheme;
                    for (int r=0;r<2;++r) for (int part=0;part<(axis==0 ? 2 : 1);++part)
                        scheme[r].push_back({part*inter+r*inter/2,part*inter+(r+1)*inter/2});
                    std::vector<int> ranks{0,1}; Data bias;
                    Require(SplitMultiCudaWeight(*source,bias,ranks,scheme,axis,true),"GLM TP expert split failed");
                    Require(!source->cpuData && source->numasData.empty(),"GLM TP expert retained CPU copy");
                    for (int r=0;r<2;++r) {
                        tpWeights[t][r][i]=source->multiDeviceDatas.at(r);
                        tpWeights[t][r][i]->ClearTensorParallelLayout();
                    }
                    tpOwned.push_back(std::move(source));
                }
            }
        } else {
            if (noCache) Require(!FastllmCudaPrepareMoeCache(layers,tables,[] {}),
                "zero-cache streaming must be explicitly prepared");
            Require(FastllmCudaPrepareMoeCache(layers,tables,[&] {
                for (auto &table:weights) for (size_t i=2;i<table.size();++i)
                    RegisterNumas(table[i],i%2 ? "linearColumn" : "linearSwiglu");
            },noCache),"GLM GGUF registration rejected");
            Require(FastllmCudaCanRunMoeHybrid(weights[0].data(),weights[0].size()),"hybrid unavailable");
            Require(!FastllmCudaCanRunMoeCache(weights[0].data(),weights[0].size()),"generic unscored math admitted");
            Require(!FastllmCudaMoeGlm5GGUFCacheSupported(GGML_TYPE_Q4_K,GGML_TYPE_IQ4_XS,hidden,inter),
                    "unsupported GLM pair admitted");
            for (int t=0;t<tables;++t) {
                directFixtures[t].Check(weights[t],hidden,inter);
                CheckVerifyCpu(weights[t],hidden,t);
            }
            std::puts("PASS: GLM direct NUMA layout and scored CPU verify rows 1/2/3/4/7/9/17");
        }
        if(expertCache || verify) {
            if(expertPrefill) CheckExpertPrefill(weights,tables,hidden,numaPrefill);
            else CheckExpertCache(weights,tables,hidden,inter,cacheMode);
            FastllmCudaReleaseMoeCache(weights[0].data(),weights[0].size());
            ClearNumasMoeRuntimeCache();return 0;
        }
        float worst=0;
        bool sawCpu=false, sawMixed=false, sawGpu=false, sawStaged=false;
        bool sawCooperative[2]{};
        for (int step=0;step<(frequency ? 32 : 8);++step) {
            const int t=step%tables;
            std::vector<float> x(hidden),scores(topk),expected(hidden),per(topk*hidden);
            std::vector<uint16_t> bx(hidden);
            std::vector<int32_t> ids(topk);
            for (int c=0;c<hidden;++c) {
                x[c]=Bf(std::ldexp(float(int(rng()%97)-48)/19.f,c/32%4-2));
                bx[c]=Float32ToBFloat16RNEBits(x[c]);
            }
            for (int r=0;r<topk;++r) {
                ids[r]=(step*7+r*3)%experts;
                scores[r]=r==0 ? 0 : r==1 ? -.125f : float(r+1)/16;
            }
            ids[2]=ids[3]; // duplicate expert with distinct route scores
            auto qx=x;Q8(qx);
            for (int r=0;r<topk;++r) {
                const auto &g=dense[t][2*ids[r]], &d=dense[t][2*ids[r]+1];
                std::vector<float> mid(inter);
                for (int row=0;row<inter;++row) {
                    double gate=0,up=0;
                    for (int c=0;c<hidden;++c) {
                        gate+=double(g[row*hidden+c])*qx[c];
                        up+=double(g[(row+inter)*hidden+c])*qx[c];
                    }
                    const float a=std::min(Bf(gate),.125f), b=std::clamp(Bf(up),-.125f,.125f);
                    mid[row]=Bf((a/(1+std::exp(-a)))*b*scores[r]);
                }
                if (t%2==0) Q8(mid);
                for (int row=0;row<hidden;++row) {
                    double v=0;for(int c=0;c<inter;++c) v+=double(d[row*inter+c])*mid[c];
                    per[r*hidden+row]=Bf(v);
                }
            }
            // GLM reduces in increasing expert-id order, preserving duplicates.
            std::vector<int> order(topk);for(int r=0;r<topk;++r)order[r]=r;
            std::stable_sort(order.begin(),order.end(),[&](int a,int b){return ids[a]<ids[b];});
            for(int c=0;c<hidden;++c) {float v=0;for(int r:order)v+=per[r*hidden+c];expected[c]=Bf(v);}
            if (resident) {
                const int device = (t/2) % std::min(2, devices);
                Cuda(cudaSetDevice(device));
                for (int rows : {1, 3, 17, 32, 33, 65, 257, 1025}) {
                    std::vector<uint16_t> batchX(rows * hidden);
                    std::vector<int32_t> batchIds(rows * topk);
                    std::vector<float> batchScores(rows * topk), reference(rows * hidden);
                    for (int row=0; row<rows; ++row) {
                        for (int c=0; c<hidden; ++c) {
                            batchX[row*hidden+c] = row%2 ? 0 : bx[c];
                            reference[row*hidden+c] = row%2 ? 0 : expected[c];
                        }
                        for (int k=0; k<topk; ++k) {
                            batchIds[row*topk+k] = ids[(k+row)%topk];
                            batchScores[row*topk+k] = scores[(k+row)%topk];
                        }
                    }
                    Data input(BFLOAT16,{rows,hidden},CPU,batchX.data()), index(INT32,{rows,topk},CPU,batchIds.data()),
                         score(FLOAT32,{rows,topk},CPU,batchScores.data()), activation, workspace, out;
                    input.ToDevice(CUDA,std::vector<int>{device}); index.ToDevice(CUDA,std::vector<int>{device}); score.ToDevice(CUDA,std::vector<int>{device});
                    Require(FastllmCudaMergeMOEGlm5GGUFResident(input,activation,workspace,out,
                        weights[t].data(),weights[t].size(),static_cast<const int32_t *>(index.cudaData),
                        static_cast<const float *>(score.cudaData),topk,.125f), "resident GLM rejected");
                    const size_t grouped = fastllm_gguf_mmq::Glm5GroupedWorkspaceBytes(
                        weights[t][2]->ggmlType,weights[t][3]->ggmlType,std::min(rows,1024),
                        hidden,inter,experts,topk);
                    if (rows>32 && grouped) Require(workspace.GetBytes()==grouped,
                        "supported prefill did not select grouped workspace");
                    if (t%2) Require(grouped==0,"IQ4_XS BF16 down admitted to Q8 grouped path");
                    out.ToDevice(CPU);
                    std::vector<float> actual(rows*hidden);
                    for (int c=0;c<rows*hidden;++c)
                        actual[c]=BFloat16BitsToFloat32(reinterpret_cast<uint16_t *>(out.cpuData)[c]);
                    worst=std::max(worst,Compare(actual,reference,"resident vs scalar oracle"));
                    if (devices >= 2) {
                        std::vector<float> sum(rows*hidden,0);
                        for (int rank=0;rank<2;++rank) {
                            Cuda(cudaSetDevice(rank));
                            Data tx(BFLOAT16,{rows,hidden},CPU,batchX.data()), ti(INT32,{rows,topk},CPU,batchIds.data()),
                                ts(FLOAT32,{rows,topk},CPU,batchScores.data()), ta, tw, to;
                            tx.ToDevice(CUDA,std::vector<int>{rank}); ti.ToDevice(CUDA,std::vector<int>{rank});
                            ts.ToDevice(CUDA,std::vector<int>{rank});
                            Require(FastllmCudaMergeMOEGlm5GGUFResident(tx,ta,tw,to,
                                tpWeights[t][rank].data(),tpWeights[t][rank].size(),
                                static_cast<const int32_t *>(ti.cudaData),static_cast<const float *>(ts.cudaData),
                                topk,.125f),"GLM TP resident shard rejected");
                            to.ToDevice(CPU);
                            for (int c=0;c<rows*hidden;++c)
                                sum[c]+=BFloat16BitsToFloat32(reinterpret_cast<uint16_t *>(to.cpuData)[c]);
                        }
                        for (auto &v:sum) v=Bf(v);
                        worst=std::max(worst,Compare(sum,reference,"resident TP sum vs full scalar oracle"));
                        Cuda(cudaSetDevice(device));
                    }
                }
                continue;
            }
            Data hx(BFLOAT16,{1,hidden},CPU,bx.data()),hi(INT32,{1,topk},CPU,ids.data()),
                 hs(FLOAT32,{1,topk},CPU,scores.data()),ordinary;
            OrdinaryNumas(hx,hi,hs,ordinary,weights[t],t);
            ordinary.ToDevice(CPU);
            std::vector<float> cpu(hidden);
            for(int c=0;c<hidden;++c) cpu[c]=BFloat16BitsToFloat32(reinterpret_cast<uint16_t*>(ordinary.cpuData)[c]);
            worst=std::max(worst,Compare(cpu,expected,"ordinary NUMA vs scalar oracle"));
            // Exercise scored CPU subsets directly, including duplicate routes
            // with different scores and the all-GPU callback-only case.
            std::vector<int32_t> mask(topk, -1);
            const int subsetGpu = step % (topk + 1);
            for (int r=0;r<subsetGpu;++r) mask[r]=ids[r];
            std::vector<float> subset(topk*hidden, 123.f), subsetExpected(subset);
            for (int r=subsetGpu;r<topk;++r)
                std::copy_n(per.data()+r*hidden,hidden,subsetExpected.data()+r*hidden);
            auto checkSubset=[&](const char *label) {
                Require(std::all_of(subset.begin(),subset.begin()+subsetGpu*hidden,
                    [](float v){return v==123.f;}),"GPU-owned route modified by CPU");
                const std::vector<float> actual(subset.begin()+subsetGpu*hidden,subset.end());
                const std::vector<float> reference(subsetExpected.begin()+subsetGpu*hidden,subsetExpected.end());
                worst=std::max(worst,Compare(actual,reference,label));
            };
            double cpuUs=-1;
            int submitted=0;
            NumasMoeDecodeExpertsWithOverlap(x.data(),subset.data(),weights[t].data(),
                ids.data(),mask.data(),topk,t,[&]{++submitted;},
                scores.data(),.125f,128,&cpuUs);
            Require(submitted==1 && (subsetGpu==topk ? cpuUs==0 : cpuUs>0),
                    "scored subset callback or CPU timing missing");
            checkSubset("scored CPU subset vs scalar oracle");
            if (step==0) {
                auto begin=std::chrono::steady_clock::now();
                NumasMoeDecodeExpertsWithOverlap(x.data(),subset.data(),weights[t].data(),
                    ids.data(),mask.data(),topk,t,[]{std::this_thread::sleep_for(std::chrono::milliseconds(20));},
                    scores.data(),.125f,128,&cpuUs);
                const double wallUs=std::chrono::duration<double,std::micro>(
                    std::chrono::steady_clock::now()-begin).count();
                Require(cpuUs>0 && cpuUs<wallUs*.5,
                        "scored CPU timing included the GPU callback stall");
                checkSubset("timed scored CPU subset");
                bool caught=false;
                try {
                    NumasMoeDecodeExpertsWithOverlap(x.data(),subset.data(),weights[t].data(),
                        ids.data(),mask.data(),topk,t,[]{throw std::runtime_error("expected submit failure");},
                        scores.data(),.125f,128,&cpuUs);
                } catch(const std::runtime_error &) {caught=true;}
                Require(caught,"scored subset swallowed callback failure");
                NumasMoeDecodeExpertsWithOverlap(x.data(),subset.data(),weights[t].data(),
                    ids.data(),mask.data(),topk,t,[]{},scores.data(),.125f,128,&cpuUs);
                checkSubset("scored CPU callback recovery");
            }
            for(int device=0;device<std::min(2,devices);++device) {
                Cuda(cudaSetDevice(device));
                Data input(BFLOAT16,{1,hidden},CPU,bx.data()),index(INT32,{1,topk},CPU,ids.data()),
                     score(FLOAT32,{1,topk},CPU,scores.data()),out;
                input.ToDevice(CUDA,std::vector<int>{device});
                index.ToDevice(CUDA,std::vector<int>{device});score.ToDevice(CUDA,std::vector<int>{device});
                // Refill outside Begin/End, then resume frequency admission.
                // This also verifies that End restores ordinary routing.
                for(int split : noCache ? std::vector<int>{-1,-1,-1} :
                                frequency ? std::vector<int>{-1,-1,-1,6,-1,0} :
                                            std::vector<int>{0,2,6,6,0}) {
                    setenv("FASTLLM_GLM5_MOE_CACHE_GPU_EXPERTS",std::to_string(split).c_str(),1);
                    void *state=nullptr;
                    if (split < 0) {
                        // A stale calibration override must not refill misses.
                        setenv("FASTLLM_GLM5_MOE_CACHE_GPU_EXPERTS","0",1);
                        state=FastllmCudaBeginMoeDecode(weights[t].data(),weights[t].size(),topk);
                        Require(state!=nullptr,"GLM frequency policy unavailable");
                    }
                    uint64_t before[8]={},after[8]={};
                    Require(fastllm_moe_cuda_cache_route_stats(device,before),"before counters");
                    int callbacks=0;
                    Require(FastllmCudaMergeMOEHybrid(input,index,score,out,weights[t].data(),weights[t].size(),t,[&]{
                        ++callbacks;if(devices>1)Cuda(cudaSetDevice(1-device));
                    }),"GGUF GLM hybrid rejected");
                    Require(callbacks==1,"shared callback missing");
                    FastllmCudaEndMoeDecode(state);
                    Require(fastllm_moe_cuda_cache_route_stats(device,after),"after counters");
                    Require(after[0]-before[0]==1 && after[1]-before[1]==topk,"missing routes");
                    const auto gpu=after[4]-before[4];
                    if (noCache) Require(after[2]==0 && after[6]==0 && after[7]==0,
                        "zero-cache dispatch retained or prefetched experts");
                    if (split < 0) {
                        const auto resident=after[2]-before[2];
                        Require(gpu>=resident && after[6]-before[6]==resident &&
                                gpu+after[5]-before[5]==topk,
                                "frequency staged dispatch lost resident or CPU routes");
                        sawStaged|=gpu>resident;
                        Require(after[7]==before[7],"frequency also ran legacy prefetch");
                        sawCpu|=gpu<topk; sawMixed|=gpu>0 && gpu<topk; sawGpu|=gpu==topk;
                    } else Require(gpu==split && after[5]-before[5]==topk-split,"wrong dispatch");
                    Require(after[2]-before[2]+after[3]-before[3]==topk,"wrong residency count");
                    out.ToDevice(CPU);std::vector<float> actual(hidden);
                    for(int c=0;c<hidden;++c)actual[c]=BFloat16BitsToFloat32(reinterpret_cast<uint16_t*>(out.cpuData)[c]);
                    if(gpu==0) Require(std::memcmp(out.cpuData,ordinary.cpuData,hidden*2)==0,
                        "CPU-only cache route changed ordinary GLM GGUF arithmetic");
                    worst=std::max(worst,Compare(actual,expected,"hybrid vs scalar oracle"));
                }
            }
            if (frequency && devices > 1) {
                // Exercise arbitrary verifier widths as well as decode; rotate
                // routes and alternate zero/nonzero inputs between rows to catch wrong maps.
                for (int rows : {1, 2, 5}) {
                    const int device = step % 2;
                    Cuda(cudaSetDevice(device));
                    std::vector<uint16_t> batchX(rows * hidden);
                    std::vector<int32_t> batchIds(rows * topk);
                    std::vector<float> batchScores(rows * topk), batchExpected(rows * hidden);
                    for (int row=0;row<rows;++row) {
                        // A zero activation row must produce zero regardless of its
                        // score/route order, while other rows retain the oracle.
                        for (int c=0;c<hidden;++c) {
                            batchX[row*hidden+c]=row%2 ? 0 : bx[c];
                            batchExpected[row*hidden+c]=row%2 ? 0 : expected[c];
                        }
                        for (int k=0;k<topk;++k) {
                            batchIds[row*topk+k]=ids[(k+row)%topk];
                            batchScores[row*topk+k]=scores[(k+row)%topk];
                        }
                    }
                    Data input(BFLOAT16,{rows,hidden},CPU,batchX.data()), index(INT32,{rows,topk},CPU,batchIds.data()),
                         score(FLOAT32,{rows,topk},CPU,batchScores.data()), out;
                    input.ToDevice(CUDA,std::vector<int>{device}); index.ToDevice(CUDA,std::vector<int>{device}); score.ToDevice(CUDA,std::vector<int>{device});
                    void *state = FastllmCudaBeginMoeDecode(weights[t].data(),weights[t].size(),topk);
                    Require(state != nullptr, "cooperative policy unavailable");
                    uint64_t before[2][8]{}, after[2][8]{};
                    for (int d=0;d<2;++d) Require(fastllm_moe_cuda_cache_route_stats(d,before[d]),"cooperative before");
                    Cuda(cudaSetDevice(device));
                    int callbacks=0;
                    Require(FastllmCudaMergeMOEHybridOnDevices(input,index,score,out,
                        weights[t].data(),weights[t].size(),t,{0,1},[&]{++callbacks; Cuda(cudaSetDevice(1-device));}),
                        "cooperative GGUF rejected");
                    Require(callbacks==1 && FastllmCudaGetDevice()==device,"cooperative callback/device restore");
                    FastllmCudaEndMoeDecode(state);
                    uint64_t total=0, computed=0;
                    for (int d=0;d<2;++d) {
                        Require(fastllm_moe_cuda_cache_route_stats(d,after[d]),"cooperative after");
                        total+=after[d][1]-before[d][1];
                        computed+=after[d][4]-before[d][4]+after[d][5]-before[d][5];
                        sawCooperative[d]|=after[d][4]>before[d][4];
                        if(noCache) Require(after[d][2]==0 && after[d][6]==0,"cooperative unexpectedly cached weights");
                    }
                    Require(total==rows*topk && computed==rows*topk,"cooperative routes lost or duplicated");
                    out.ToDevice(CPU); std::vector<float> actual(rows*hidden);
                    for(int c=0;c<rows*hidden;++c) actual[c]=BFloat16BitsToFloat32(reinterpret_cast<uint16_t*>(out.cpuData)[c]);
                    worst=std::max(worst,Compare(actual,batchExpected,"cooperative vs scalar oracle"));
                }
            }
        }
        if (resident) {
            std::printf("PASS: GLM GGUF resident experts, freed host weights, both GPUs, GEMV/grouped rows 1/3/17/32/33/65/257/1025; max relative L2 %.8f\n",worst);
            return 0;
        }
        for(int device=0;device<std::min(2,devices);++device) {
            uint64_t stats[5]={};Require(fastllm_moe_cuda_cache_stats(device,stats,false),"query counters");
            Require(noCache ? stats[0]==0 && stats[1]>0 && stats[2]==0 && stats[3]==0 :
                stats[0] && stats[1] && (frequency ? stats[3]>=64 && stats[3]<tables*experts : stats[3]==16),
                    "cache cold/hot/eviction or compact slots missing");
        }
        if (frequency) Require(sawCpu && sawMixed && (noCache || sawGpu) && sawStaged,
                               "frequency did not exercise CPU, resident and staged GPU routes");
        if (frequency && devices > 1) Require(sawCooperative[0] && sawCooperative[1], "cooperative path did not exercise both GPUs");
        FastllmCudaReleaseMoeCache(weights[0].data(),weights[0].size());
        ClearNumasMoeRuntimeCache();
        std::printf("PASS: GLM GGUF cache CPU/mixed/GPU, scalar oracle, duplicates, scores, clamp, eviction, counters; max relative L2 %.8f\n",worst);
        return 0;
    } catch(const std::exception &e) { std::fprintf(stderr,"FAIL: %s\n",e.what());return 1; }
}
