#include "fastllm.h"
#include "executor.h"
#include "utils.h"
#include "gguf.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-moe-cache-stats.h"
#include "devices/cuda/fastllm-cuda-moe-policy.h"
#include "devices/numas/numasdevice.h"
#include "devices/numas/numas.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
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
int main(int argc, char **argv) {
    try {
        const bool frequency = argc > 1 && std::string(argv[1]) == "--frequency";
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) {
            std::puts("FASTLLM_TEST_SKIP_NO_DEVICE: CUDA unavailable"); return 0;
        }
        constexpr int hidden=512, inter=256, experts=24, topk=6, tables=4;
        SetThreads(8);
        setenv("FASTLLM_GLM5_MOE_CACHE_PREFETCH",frequency ? "1" : "0",1);
        std::mt19937 rng(71443);
        std::vector<std::unique_ptr<Data>> owned;
        std::vector<Data *> weights[tables];
        std::vector<std::vector<float>> dense[tables];
        FastllmCudaMoeCacheLayer layers[tables];
        size_t stride=0;
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
                    const uint16_t scale=float_to_half(std::ldexp(1.f,-12+int(rng()%3)));
                    memcpy(w->cpuData+off,&scale,2);
                }
                auto &matrix=dense[t][2*e+part];matrix.resize(rows*cols);
                ggml_type_to_float(type)(w->cpuData,matrix.data(),matrix.size());
                if (type==GGML_TYPE_IQ4_XS) for (auto &v:matrix) v=Bf(v);
                weights[t][2*(e+1)+part]=w.get();owned.push_back(std::move(w));
            }
            stride=std::max(stride,weights[t][2]->GetBytes()+weights[t][3]->GetBytes());
            layers[t]={weights[t].data(),int(weights[t].size()),false,.125f,true};
        }
        stride=(stride+127)/128*128;
        SetMoeCudaCacheBytes(stride*(frequency ? 64 : 16));
        Require(FastllmCudaPrepareMoeCache(layers,tables,[&] {
            for (auto &table:weights) for (size_t i=2;i<table.size();++i)
                RegisterNumas(table[i],i%2 ? "linearColumn" : "linearSwiglu");
        }),"GLM GGUF registration rejected");
        Require(FastllmCudaCanRunMoeHybrid(weights[0].data(),weights[0].size()),"hybrid unavailable");
        Require(!FastllmCudaCanRunMoeCache(weights[0].data(),weights[0].size()),"generic unscored math admitted");
        Require(!FastllmCudaMoeGlm5GGUFCacheSupported(GGML_TYPE_Q4_K,GGML_TYPE_IQ4_XS,hidden,inter),
                "unsupported GLM pair admitted");
        float worst=0;
        bool sawCpu=false, sawMixed=false, sawGpu=false, sawStaged=false;
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
            Data hx(BFLOAT16,{1,hidden},CPU,bx.data()),hi(INT32,{1,topk},CPU,ids.data()),
                 hs(FLOAT32,{1,topk},CPU,scores.data()),ordinary;
            OrdinaryNumas(hx,hi,hs,ordinary,weights[t],t);
            ordinary.ToDevice(CPU);
            std::vector<float> cpu(hidden);
            for(int c=0;c<hidden;++c) cpu[c]=BFloat16BitsToFloat32(reinterpret_cast<uint16_t*>(ordinary.cpuData)[c]);
            worst=std::max(worst,Compare(cpu,expected,"ordinary NUMA vs scalar oracle"));
            for(int device=0;device<std::min(2,devices);++device) {
                Cuda(cudaSetDevice(device));
                Data input(BFLOAT16,{1,hidden},CPU,bx.data()),index(INT32,{1,topk},CPU,ids.data()),
                     score(FLOAT32,{1,topk},CPU,scores.data()),out;
                input.ToDevice(CUDA,std::vector<int>{device});
                index.ToDevice(CUDA,std::vector<int>{device});score.ToDevice(CUDA,std::vector<int>{device});
                // Refill outside Begin/End, then resume frequency admission.
                // This also verifies that End restores ordinary routing.
                for(int split : frequency ? std::vector<int>{-1,-1,-1,6,-1,0} :
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
        }
        for(int device=0;device<std::min(2,devices);++device) {
            uint64_t stats[5]={};Require(fastllm_moe_cuda_cache_stats(device,stats,false),"query counters");
            Require(stats[0] && stats[1] && (frequency ? stats[3]>=64 && stats[3]<tables*experts : stats[3]==16),
                    "cache cold/hot/eviction or compact slots missing");
        }
        if (frequency) Require(sawCpu && sawMixed && sawGpu && sawStaged,
                               "frequency did not exercise CPU, resident and staged GPU routes");
        FastllmCudaReleaseMoeCache(weights[0].data(),weights[0].size());
        ClearNumasMoeRuntimeCache();
        std::printf("PASS: GLM GGUF cache CPU/mixed/GPU, scalar oracle, duplicates, scores, clamp, eviction, counters; max relative L2 %.8f\n",worst);
        return 0;
    } catch(const std::exception &e) { std::fprintf(stderr,"FAIL: %s\n",e.what());return 1; }
}
