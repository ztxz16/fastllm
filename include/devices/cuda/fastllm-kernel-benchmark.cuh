#pragma once
#include "utils/kernel_registry.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-graph-workspace.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <vector>

namespace fastllm::kernels {
struct CudaCase {
    void *context;
    bool (*run)(void *, int);
    std::string (*unsupported)(void *, int);
    void *output;
    size_t elements;
    bool bfloat16;
};
inline void CudaCheck(cudaError_t code) {
    if(code!=cudaSuccess)throw std::runtime_error(std::string("kernel benchmark: ")+cudaGetErrorString(code));
}
inline std::string CudaHardware(void *) {
    static thread_local std::map<int,std::string> ids;
    const int device=FastllmCudaGetDevice();
    auto found=ids.find(device);if(found!=ids.end())return found->second;
    cudaDeviceProp prop{};int driver=0,runtime=0;
    CudaCheck(cudaGetDeviceProperties(&prop,device));CudaCheck(cudaDriverGetVersion(&driver));CudaCheck(cudaRuntimeGetVersion(&runtime));
    return ids.emplace(device,std::string(prop.name)+":"+std::to_string(prop.major*10+prop.minor)+":"+
        std::to_string(prop.multiProcessorCount)+":"+std::to_string(prop.totalGlobalMem)+":"+
        std::to_string(driver)+":"+std::to_string(runtime)).first->second;
}
inline std::string CudaUnsupported(void *opaque,int implementation) {
    auto &c=*static_cast<CudaCase *>(opaque);return c.unsupported(c.context,implementation);
}
inline double CudaMeasure(void *opaque,int implementation,int repeats) {
    auto &c=*static_cast<CudaCase *>(opaque);
    cudaEvent_t begin=nullptr,end=nullptr;
    cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;bool capturing=false;
    auto cleanup=[&]() {
        if(capturing) {
            if(cudaStreamEndCapture(cudaStreamPerThread,&graph)!=cudaSuccess)cudaGetLastError();
        }
        if(exec)cudaGraphExecDestroy(exec);
        if(graph)cudaGraphDestroy(graph);
        if(begin)cudaEventDestroy(begin);
        if(end)cudaEventDestroy(end);
    };
    try {
        CudaCheck(cudaEventCreate(&begin));CudaCheck(cudaEventCreate(&end));
        CudaCheck(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));capturing=true;
        const bool accepted=c.run(c.context,implementation);
        const auto ended=cudaStreamEndCapture(cudaStreamPerThread,&graph);capturing=false;
        if(ended!=cudaSuccess)cudaGetLastError();
        CudaCheck(ended);
        if(!accepted){cleanup();return -1;}
        CudaCheck(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        CudaCheck(cudaGraphLaunch(exec,cudaStreamPerThread));
        CudaCheck(cudaStreamSynchronize(cudaStreamPerThread));
        CudaCheck(cudaEventRecord(begin,cudaStreamPerThread));
        for(int i=0;i<repeats;++i)CudaCheck(cudaGraphLaunch(exec,cudaStreamPerThread));
        CudaCheck(cudaEventRecord(end,cudaStreamPerThread));CudaCheck(cudaEventSynchronize(end));
        float ms=0;CudaCheck(cudaEventElapsedTime(&ms,begin,end));
        cleanup();return ms/repeats;
    } catch(...) {cleanup();throw;}
}
// Single-output and fusion families share numeric validation. Keep
// each tensor's relative L2 separate so a small gate cannot hide behind a
// larger projection output. Snapshot and restore every user-owned output,
// including inactive MoE routes and writes made before a callback rejects.
struct CudaOutput {
    void *data;
    size_t elements;
    fastllm::DataType type;
};
struct CudaOutputsCase {
    CudaCase operation;
    std::vector<CudaOutput> outputs;
};
inline double CudaOutputValue(const unsigned char *p, fastllm::DataType type) {
    if (type == fastllm::FLOAT32) { float v; std::memcpy(&v,p,4); return v; }
    uint16_t h; std::memcpy(&h,p,2);
    if (type == fastllm::BFLOAT16) {
        uint32_t bits=uint32_t(h)<<16; float v; std::memcpy(&v,&bits,4); return v;
    }
    const int exponent=(h>>10)&31, fraction=h&1023;
    double v=exponent==0 ? std::ldexp(double(fraction),-24) :
        exponent==31 ? (fraction ? NAN : INFINITY) : std::ldexp(double(1024+fraction),exponent-25);
    return (h&0x8000) ? -v : v;
}
inline bool ValidateCudaOutputs(CudaCase &operation,const CudaOutput *outputs,size_t count,
        int implementation,double &l2,std::string &reason) {
    std::vector<std::vector<unsigned char>> original,reference,actual;
    for (size_t i=0;i<count;++i) {
        const auto &v=outputs[i];
        const size_t bytes=v.elements*(v.type==fastllm::FLOAT32?4:2);
        original.emplace_back(bytes);reference.emplace_back(bytes);actual.emplace_back(bytes);
        CudaCheck(cudaMemcpyAsync(original.back().data(),v.data,bytes,cudaMemcpyDeviceToHost,cudaStreamPerThread));
    }
    CudaCheck(cudaStreamSynchronize(cudaStreamPerThread));
    auto restore=[&] {
        for(size_t i=0;i<count;++i) CudaCheck(cudaMemcpyAsync(outputs[i].data,
            original[i].data(),original[i].size(),cudaMemcpyHostToDevice,cudaStreamPerThread));
        CudaCheck(cudaStreamSynchronize(cudaStreamPerThread));
    };
    auto run=[&](int impl,std::vector<std::vector<unsigned char>> &values) {
        for(size_t i=0;i<count;++i) CudaCheck(cudaMemsetAsync(outputs[i].data,0,original[i].size(),cudaStreamPerThread));
        if(!operation.run(operation.context,impl))return false;
        for(size_t i=0;i<count;++i) CudaCheck(cudaMemcpyAsync(values[i].data(),
            outputs[i].data,values[i].size(),cudaMemcpyDeviceToHost,cudaStreamPerThread));
        CudaCheck(cudaStreamSynchronize(cudaStreamPerThread));return true;
    };
    bool ok=false;
    try {ok=run(0,reference)&&run(implementation,actual);} catch(...) {restore();throw;}
    restore();if(!ok){reason="prepare or execution rejected";return false;}
    l2=0;
    for(size_t t=0;t<count;++t) {
        const auto &v=outputs[t];const size_t stride=v.type==fastllm::FLOAT32?4:2;
        double square=0,error=0;
        for(size_t i=0;i<v.elements;++i) {
            double x=CudaOutputValue(reference[t].data()+i*stride,v.type),
                y=CudaOutputValue(actual[t].data()+i*stride,v.type);
            if(!std::isfinite(x)||!std::isfinite(y)){reason="non-finite output";return false;}
            square+=x*x;error+=(x-y)*(x-y);
        }
        l2=std::max(l2,square>0?std::sqrt(error/square):std::sqrt(error));
    }
    return true;
}
inline bool CudaValidate(void *opaque,int implementation,double &l2,std::string &reason) {
    auto &c=*static_cast<CudaCase *>(opaque);
    const CudaOutput output{c.output,c.elements,c.bfloat16?fastllm::BFLOAT16:fastllm::FLOAT32};
    return ValidateCudaOutputs(c,&output,1,implementation,l2,reason);
}
inline bool CudaOutputsValidate(void *opaque,int implementation,double &l2,std::string &reason) {
    auto &c=*static_cast<CudaOutputsCase *>(opaque);
    return ValidateCudaOutputs(c.operation,c.outputs.data(),c.outputs.size(),implementation,l2,reason);
}
inline bool LaunchCuda(const Family &family,CudaCase &context,int implementation) {
    if(context.run(context.context,implementation))return true;
    if(implementation==0)return false;
    // A later tensor may violate an optional candidate's pointer/layout contract.
    // Keep the original implementation available without altering a captured plan.
    std::fprintf(stderr,"[kernels] %s rejected %s at execution; using %s\n",family.name,
        family.implementations[implementation].id,family.implementations[0].id);
    return context.run(context.context,0);
}
inline int SelectCuda(const Family &family,const Signature &signature,CudaCase &context) {
    // Child-graph builders can use CUDA capture directly, outside the fast
    // wrapper bookkeeping. Always query the stream before admitting a trial.
    Call call{&family,signature,&context,
        FastllmCudaGraphIsCapturing() || FastllmCudaGraphMemoryPoolActive(),CudaUnsupported,
        CudaMeasure,CudaValidate,CudaHardware};
    return Select(call);
}
inline double CudaOutputsMeasure(void *opaque,int implementation,int repeats) {
    auto &c=static_cast<CudaOutputsCase *>(opaque)->operation;
    // A fused operation may allocate scratch between kernels. Reserve and pin
    // it for the benchmark graph exactly as for a model graph: a raw capture
    // cannot guarantee that allocator reuse keeps those addresses alive.
    std::vector<size_t> requests;
    { FastllmCudaAllocationRecorder record(&requests);
      FastllmCudaGraphPreparation prepare;
      if(!c.run(c.context,implementation))return -1; }
    CudaCheck(cudaStreamSynchronize(cudaStreamPerThread));
    if(!FastllmCudaGraphPrepareCaptureDevice() || !FastllmCudaReserveGraphWorkspace(requests))return -1;
    std::vector<void *> reserved;
    void *graph=nullptr,*exec=nullptr;
    bool pool=false,capturing=false;
    cudaEvent_t begin=nullptr,end=nullptr;
    auto cleanup=[&] {
        if(capturing)FastllmCudaGraphEndCapture(&graph);
        if(exec)FastllmCudaGraphExecDestroy(exec);
        if(graph)FastllmCudaGraphDestroy(graph);
        if(pool)FastllmCudaGraphMemoryPoolAbort();
        FastllmCudaGraphMemoryPoolRelease(reserved);
        if(begin)cudaEventDestroy(begin);
        if(end)cudaEventDestroy(end);
    };
    try {
        CudaCheck(cudaEventCreate(&begin));CudaCheck(cudaEventCreate(&end));
        pool=FastllmCudaGraphMemoryPoolBegin();
        if(!pool){cleanup();return -1;}
        capturing=FastllmCudaGraphBeginCapture();
        if(!capturing){cleanup();return -1;}
        const bool accepted=c.run(c.context,implementation);
        const bool ended=FastllmCudaGraphEndCapture(&graph);capturing=false;
        if(!accepted || !ended || !graph){cleanup();return -1;}
        if(!FastllmCudaGraphMemoryPoolEnd(reserved)){cleanup();return -1;}
        pool=false;
        if(!FastllmCudaGraphInstantiate(graph,&exec)){cleanup();return -1;}
        CudaCheck(cudaGraphLaunch(static_cast<cudaGraphExec_t>(exec),cudaStreamPerThread));
        CudaCheck(cudaStreamSynchronize(cudaStreamPerThread));
        CudaCheck(cudaEventRecord(begin,cudaStreamPerThread));
        for(int i=0;i<repeats;++i)CudaCheck(cudaGraphLaunch(static_cast<cudaGraphExec_t>(exec),cudaStreamPerThread));
        CudaCheck(cudaEventRecord(end,cudaStreamPerThread));CudaCheck(cudaEventSynchronize(end));
        float ms=0;CudaCheck(cudaEventElapsedTime(&ms,begin,end));cleanup();return ms/repeats;
    } catch(...) {cleanup();throw;}
}
inline std::string CudaOutputsUnsupported(void *opaque,int implementation) {
    return CudaUnsupported(&static_cast<CudaOutputsCase *>(opaque)->operation,implementation);
}
inline int SelectCudaOutputs(const Family &family,const Signature &signature,CudaOutputsCase &context,int fallback=0) {
    try {
        return Select(Call{&family,signature,&context,
            FastllmCudaGraphIsCapturing() || FastllmCudaGraphMemoryPoolActive(),CudaOutputsUnsupported,
            CudaOutputsMeasure,CudaOutputsValidate,CudaHardware,fallback});
    } catch(const std::exception &e) {
        throw std::runtime_error(std::string(family.name)+": "+e.what());
    }
}
inline std::string TritonSupport(int rows,int limit,const char *opFlag) {
    const int arch=FastllmCudaRuntimeArch();
    if(arch!=120&&arch!=121)return "requires SM120 or SM121";
    if(rows<2||rows>limit)return "row capacity exceeded";
    for(const char *key:{"FASTLLM_CUDA_TRITON",opFlag}) {
        const char *value=std::getenv(key);
        if(value&&value[0]) {
            std::string s=value;std::transform(s.begin(),s.end(),s.begin(),[](unsigned char c){return std::tolower(c);});
            if(s=="0"||s=="off"||s=="false")return std::string("disabled by ")+key;
        }
    }
    return "";
}
}
