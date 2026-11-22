#include "devices/cuda/fastllm-cuda-decode-dense.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/low_latency/dense.h"
#include "../low_latency/data_layout.h"
#include "utils.h"
#include <map>
#include <tuple>

namespace fastllm {
bool FastllmCudaDecodeDense(const Data &input, const Data &gate, const Data &up,
        const Data &down, Data &intermediate, Data &output) {
    namespace ll = cuda::low_latency;
    if (gate.dims.size()!=2 || up.dims!=gate.dims ||
        down.dims!=std::vector<int>{gate.dims[1],gate.dims[0]} || input.dims.empty() ||
        input.dims.back()!=gate.dims[1] || gate.dims[1]<=0) return false;
    const uint64_t logicalRows=input.Count(0)/gate.dims[1];
    // Performance policy, not a kernel limitation. Larger GEMMs keep their
    // existing implementation; the raw plan supports arbitrary logical tiles.
    if (logicalRows<1 || logicalRows>8) return false;
    const int rows=static_cast<int>(logicalRows);
    int device=-1;
    AssertInFastLLM(cudaGetDevice(&device)==cudaSuccess,"Cannot query Dense device.");
    for (const Data *p : {&input,&gate,&up,&down})
        if (p->dataType!=BFLOAT16 || !ll::DenseData(*p,device)) return false;
    if (ll::DataOverlap(intermediate,output)) return false;
    for (const Data *p : {&intermediate,&output}) {
        if (p->isFake || p->cudaDataBorrowed || p->multiDeviceData ||
            (p->cudaData && p->dataDeviceIds!=std::vector<int>{device})) return false;
        for (const Data *in : {&input,&gate,&up,&down}) if (ll::DataOverlap(*p,*in)) return false;
    }
    using Key=std::tuple<int,int,int,int>;
    static thread_local std::map<Key,ll::DensePlan> plans;
    const Key key{device,rows,gate.dims[1],gate.dims[0]};
    auto found=plans.find(key);
    if (found==plans.end()) {
        if (FastllmCudaGraphIsCapturingFast()) return false;
        ll::DensePlan plan;
        auto error=ll::PrepareDense({rows,gate.dims[1],gate.dims[0]},plan,cudaStreamPerThread);
        if (error==cudaErrorStreamCaptureUnsupported) return false; // Transient: never cache.
        if (ll::BackendUnavailable(error)) {
            const auto pending=cudaGetLastError();
            AssertInFastLLM(pending==cudaSuccess || ll::BackendUnavailable(pending),"Pending CUDA failure during Dense admission.");
            plans.emplace(key,plan); // Cache only stable capability misses.
            return false;
        }
        AssertInFastLLM(error==cudaSuccess,"Dense preparation failed.");
        found=plans.emplace(key,plan).first;
    }
    if (!found->second.Ready()) return false;
    for (Data *p : {&intermediate,&output}) {
        auto dims=input.dims;
        dims.back()=p==&intermediate?gate.dims[0]:gate.dims[1];
        p->dataType=BFLOAT16;p->UpdateUnitSize();p->Resize(dims);
        p->ToDevice(input.dataDevice,input.dataDeviceIds,false);p->Allocate(false);
    }
    const ll::DenseBindings b{{input.cudaData,input.GetBytes()},{gate.cudaData,gate.GetBytes()},
        {up.cudaData,up.GetBytes()},{down.cudaData,down.GetBytes()},
        {intermediate.cudaData,intermediate.GetBytes()},{output.cudaData,output.GetBytes()}};
    AssertInFastLLM(ll::RunDense(found->second,b,cudaStreamPerThread)==cudaSuccess,"Dense launch failed.");
    return true;
}
}
