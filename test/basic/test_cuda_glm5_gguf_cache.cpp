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
namespace fastllm { void RegisterNumas(Data *, std::string); }
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

int main(int argc, char **argv) {
    try {
        const bool resident = argc > 1 && std::string(argv[1]) == "--resident";
        const bool noCache = argc > 1 && std::string(argv[1]) == "--no-cache";
        const bool frequency = noCache || (argc > 1 && std::string(argv[1]) == "--frequency");
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) {
            std::puts("FASTLLM_TEST_SKIP_NO_DEVICE: CUDA unavailable"); return 0;
        }
        constexpr int hidden=512, experts=24, topk=6, tables=4;
        const int inter=resident ? 512 : 256;
        SetThreads(8);
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
                w->isModelWeight=true; w->isGGUFData=true; w->Allocate(false);
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
        SetMoeCudaCacheBytes(noCache ? 0 : stride*(frequency ? 64 : 16));
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
            }
            std::puts("PASS: GLM direct NUMA layout rows 1/2/3/4/7/9/17");
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
