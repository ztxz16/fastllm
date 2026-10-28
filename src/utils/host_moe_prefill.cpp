#include "utils/host_moe_prefill.h"
#if defined(USE_CUDA) && !defined(USE_ROCM)
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cpu/cpudevice.h"
#include "devices/cuda/cudadevice.h"
#include "devices/cpu/computeutils.h"
#include "utils/moeexpertpartition.h"
#include "utils/nsys.h"
#include "utils.h"
#include "gguf.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <tuple>
#include <unordered_set>
#endif

namespace fastllm {
#if defined(USE_CUDA) && !defined(USE_ROCM)
namespace {
using Clock = std::chrono::steady_clock;
static double Ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now()-start).count();
}
static bool Enabled() {
    const char *value = std::getenv("FT_GPU_PREFILL");
    return !value || (std::string(value)!="0" && std::string(value)!="false" &&
                      std::string(value)!="off");
}
static int OrdinaryType(int type) {
    // CPU's in-place R4 packing is restored by the existing CUDA streamer.
    static const std::map<int,int> ordinary=[] {
        std::map<int,int> value;
        for(int candidate=0;candidate<200;++candidate)
            if(const auto *repack=get_repack_info(static_cast<ggml_type>(candidate)))
                value[int(repack->new_type)]=candidate;
        return value;
    }();
    auto it=ordinary.find(type);
    return it==ordinary.end()?type:it->second;
}
struct CpuSubset : CpuMergeMOE { using CpuMergeMOE::Run; };
struct Scratch {
    Data partial;
    bool used=false;
};
struct Costs {
    double cpuUsPerRoute = 80.0;
    double gpuUsPerByte = 1.0/10000.0; // Conservative initial PCIe estimate.
};
using CostKey = std::tuple<int,int,int,int,int>;
static std::mutex costMutex;
static std::map<CostKey, Costs> costs;
static void RepackBeforeSplit(Data **weights, int count) {
    std::vector<Data *> pending;
    std::set<Data *> unique;
    for (int i=2;i<count;++i)
        if (!weights[i]->IsRepacked && !weights[i]->disableGGUFRepack && unique.insert(weights[i]).second)
            pending.push_back(weights[i]);
    if (pending.empty()) return;
    auto *pool=GetAlivePool();
    const int threads=std::min(int(pool->threads.size()),int(pending.size()));
    std::vector<std::unique_ptr<MultiThreadRepackWeightsOp>> jobs;
    for (int t=0;t<threads;++t)
        jobs.emplace_back(new MultiThreadRepackWeightsOp(pending.data(),
            int(pending.size())*t/threads,int(pending.size())*(t+1)/threads));
    for (int t=0;t<threads;++t) pool->PushOp(t,jobs[t].get());
    for (int t=0;t<threads;++t) pool->Wait(t);
}
}
#endif

bool RunHostMoePrefill(const Data &input, const Data &index, const Data &score,
                      Data &output, Data **weights, int count, int layer,
                      HostMoePrefillStats *statistics) {
#if !defined(USE_CUDA) || defined(USE_ROCM)
    return false;
#else
    if (!Enabled() || input.dataDevice!=CUDA || !input.cudaData ||
        input.dims.size()!=2 || input.dims[0]<=32 || input.dims[0]>4096 ||
        (input.dataType!=FLOAT32 && input.dataType!=FLOAT16 && input.dataType!=BFLOAT16) ||
        index.dataType!=INT32 || score.dataType!=FLOAT32 || index.dims.size()!=2 ||
        index.dims[0]!=input.dims[0] || index.dims!=score.dims ||
        index.dims[1]<=0 || index.dims[1]>32 || !weights || count<4 || count%2 ||
        weights[0] || weights[1] || FastllmCudaGraphIsCapturing()) return false;
    const int experts=count/2-1, rows=input.dims[0], hidden=input.dims[1], topk=index.dims[1];
    if (experts>1024 || !weights[2] || !weights[3] || weights[3]->dims.size()!=2) return false;
    const int inter=weights[3]->dims[1];
    const int gateType=OrdinaryType(weights[2]->ggmlType), downType=OrdinaryType(weights[3]->ggmlType);
    if (!FastllmCudaGGUFPrefillSupported(input.dataType,gateType) ||
        !FastllmCudaGGUFPrefillSupported(input.dataType,downType)) return false;
    for (int i=2;i<count;++i) {
        const Data *w=weights[i];
        if (!w || w->dataType!=DATA_GGUF_FORMAT || w->dataDevice!=CPU ||
            !w->cpuData || w->isDiskWeight || !w->numasData.empty() || w->dims.size()!=2 ||
            w->dims!=std::vector<int>({i%2?hidden:2*inter,i%2?inter:hidden}) ||
            OrdinaryType(w->ggmlType)!=(i%2?downType:gateType)) return false;
    }
    Data cpuIndex, cpuScore, cpuInput;
    cpuIndex.CopyFrom(index); cpuIndex.ToDevice(CPU);
    cpuScore.CopyFrom(score); cpuScore.ToDevice(CPU);
    auto *ids=reinterpret_cast<const int32_t *>(cpuIndex.cpuData);
    auto *scores=reinterpret_cast<float *>(cpuScore.cpuData);
    std::vector<int> routes(experts,0);
    for (int r=0;r<rows*topk;++r) {
        if (ids[r]<0 || ids[r]>=experts || !std::isfinite(scores[r])) return false;
        ++routes[ids[r]];
    }
    const CostKey key={gateType,downType,hidden,inter,GetThreads()};
    Costs estimate;
    { std::lock_guard<std::mutex> lock(costMutex); estimate=costs[key]; }
    const size_t expertBytes=weights[2]->GetBytes()+weights[3]->GetBytes();
    std::vector<detail::MoeExpertCost> jobs;
    for (int r:routes) if(r) jobs.push_back({r,r*estimate.cpuUsPerRoute,
        expertBytes*estimate.gpuUsPerByte+r*5.0});
    int threshold=detail::SelectMoeExpertLimit(jobs,128,1);
    if (const char *limit=std::getenv("FT_EXPERT_LIMIT")) threshold=std::max(1,std::atoi(limit));
    std::unordered_set<int> gpuExperts;
    HostMoePrefillStats stats;
    for (int e=0;e<experts;++e) if(routes[e]) {
        if(routes[e]>=threshold) {gpuExperts.insert(e+1);++stats.gpuExperts;stats.gpuRoutes+=routes[e];}
        else {++stats.cpuExperts;stats.cpuRoutes+=routes[e];}
    }
    if (gpuExperts.empty()) return false;
    NsysRange range(NsysEnabled()?"host_moe_prefill/layer/"+std::to_string(layer):std::string());
    // Packing is shared by CPU and CUDA. Finish it before publishing pointers
    // to the upload worker; it must never race CPU MergeMOE's lazy repack.
    RepackBeforeSplit(weights,count);
    cpuInput.CopyFrom(input); cpuInput.ToDevice(CPU);
    std::vector<float> gpuScores(scores,scores+rows*topk);
    for (int r=0;r<rows*topk;++r) if(gpuExperts.count(ids[r]+1)) scores[r]=0.0f;
    else gpuScores[r]=0.0f;
    const int device=input.dataDeviceIds.empty()?FastllmCudaGetDevice():input.dataDeviceIds[0];
    FastllmCudaSetDevice(device);
    FastllmCudaSyncCurrentThreadStream();
    // One serial inference thread owns its scratch across layers/requests.
    // The worker borrows it only until join; the existing CUDA adapter owns
    // bounded upload/compute scratch and receives cacheAdmission=false.
    static thread_local std::map<int,std::unique_ptr<Scratch>> buffers;
    auto &slot=buffers[device]; if(!slot) slot.reset(new Scratch());
    Scratch &scratch=*slot;
    const bool coldGpu=!scratch.used;
    bool gpuAccepted=false;
    std::exception_ptr gpuError, cpuError;
    std::thread worker([&] {
        try {
            FastllmCudaSetDevice(device);
            NsysRange gpuRange("host_moe_prefill/gpu");
            const auto start=Clock::now();
            Data alias(input.dataType,input.dims,CUDA,input.cudaData);
            alias.cudaDataBorrowed=true;alias.dataDeviceIds={device};
            Data gpuScore(FLOAT32,{rows,topk},gpuScores),w1,w2,w3;
            std::vector<Data *> bias(count,nullptr);
            scratch.partial.dataType=input.dataType;scratch.partial.Resize(input.dims);
            scratch.partial.ToDevice(CUDA,std::vector<int>{device},false);
            DoCudaMergeMOEFromCPU(alias,scratch.partial,cpuIndex,gpuScore,w1,w2,w3,
                weights,bias.data(),1.0f,true,gpuExperts,false,MoeGateSwiglu,false,0.0f,128,false,false);
            FastllmCudaSyncCurrentThreadStream();
            gpuAccepted=true;scratch.used=true;
            stats.gpuMs=Ms(start);
        } catch(...) { gpuError=std::current_exception(); }
    });
    Data cpuOutput(input.dataType,{rows,hidden});
    try {
        if(stats.cpuRoutes) {
            NsysRange cpuRange("host_moe_prefill/cpu");
            std::vector<Data *> bias(count,nullptr);
            Data w1,w2,w3,tmpInput,tmpOutput;
            DataDict data={{"input",&cpuInput},{"output",&cpuOutput},{"index",&cpuIndex},{"score",&cpuScore},
                {"weights",reinterpret_cast<Data *>(weights)},{"biass",reinterpret_cast<Data *>(bias.data())},
                {"w1",&w1},{"w2",&w2},{"w3",&w3},{"temporaryInput",&tmpInput},{"temporaryOutput",&tmpOutput}};
            CpuSubset cpu;
            const auto start=Clock::now();
            cpu.Run("MergeMOE",data,{},{{"weights___batch",count},{"biass___batch",count},{"hostMoePrefillSubset",1}});
            stats.cpuMs=Ms(start);
        }
    } catch(...) { cpuError=std::current_exception(); }
    worker.join();
    if(gpuError) std::rethrow_exception(gpuError);
    if(cpuError) std::rethrow_exception(cpuError);
    if(!gpuAccepted) return false;
    output.CopyFrom(scratch.partial);
    if(stats.cpuRoutes) {
        cpuOutput.ToDevice(CUDA,std::vector<int>{device});
        FastllmCudaAddTo(output,cpuOutput,1.0f);
        FastllmCudaSyncCurrentThreadStream();
    }
    // Whole-call feedback includes packing/dispatch costs instead of measuring
    // a synthetic resident expert that hides weight-transfer cost.
    { std::lock_guard<std::mutex> lock(costMutex); auto &value=costs[key];
      if(stats.cpuRoutes) value.cpuUsPerRoute=.7*value.cpuUsPerRoute+.3*(stats.cpuMs*1000/stats.cpuRoutes);
      if(!coldGpu) value.gpuUsPerByte=.7*value.gpuUsPerByte+.3*(stats.gpuMs*1000/(expertBytes*double(stats.gpuExperts)));
    }
    if(statistics) *statistics=stats;
    if(std::getenv("FASTLLM_PROFILE_HOST_MOE")) {
        FILE *profile=stderr;
        if(const char *path=std::getenv("FASTLLM_PROFILE_HOST_MOE_PATH")) {
            if(FILE *file=std::fopen(path,"ab")) profile=file;
        }
        fprintf(profile,"[host-moe-prefill] layer=%d rows=%d cpu_experts=%d gpu_experts=%d cpu_routes=%d gpu_routes=%d cpu_ms=%.3f gpu_ms=%.3f numa=0\n",
            layer,rows,stats.cpuExperts,stats.gpuExperts,stats.cpuRoutes,stats.gpuRoutes,stats.cpuMs,stats.gpuMs);
        fflush(profile);
        if(profile!=stderr) std::fclose(profile);
    }
    return true;
#endif
}
}
