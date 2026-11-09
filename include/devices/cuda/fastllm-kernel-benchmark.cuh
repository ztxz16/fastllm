#pragma once
#include "utils/kernel_registry.h"
#include "devices/cuda/fastllm-cuda.cuh"
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
        if(capturing)cudaStreamEndCapture(cudaStreamPerThread,&graph);
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
inline bool CudaValidate(void *opaque,int implementation,double &l2,std::string &reason) {
    auto &c=*static_cast<CudaCase *>(opaque);
    const size_t bytes=c.elements*(c.bfloat16?2:4);
    // A partial MoE route invocation shares output with other expert work.
    // Numeric trials must preserve all pre-existing entries, including inactive routes.
    std::vector<unsigned char> original(bytes),reference(bytes),actual(bytes);
    CudaCheck(cudaMemcpyAsync(original.data(),c.output,bytes,cudaMemcpyDeviceToHost,cudaStreamPerThread));
    CudaCheck(cudaStreamSynchronize(cudaStreamPerThread));
    auto restore=[&]() {
        CudaCheck(cudaMemcpyAsync(c.output,original.data(),bytes,cudaMemcpyHostToDevice,cudaStreamPerThread));
        CudaCheck(cudaStreamSynchronize(cudaStreamPerThread));
    };
    auto run=[&](int impl,std::vector<unsigned char>&host) {
        CudaCheck(cudaMemsetAsync(c.output,0,bytes,cudaStreamPerThread));
        if(!c.run(c.context,impl))return false;
        CudaCheck(cudaMemcpyAsync(host.data(),c.output,bytes,cudaMemcpyDeviceToHost,cudaStreamPerThread));
        CudaCheck(cudaStreamSynchronize(cudaStreamPerThread));return true;
    };
    bool accepted;
    try {
        accepted=run(0,reference)&&run(implementation,actual);
    } catch(...) {
        // A callback can reject by throwing after writing only part of the
        // output. Restore the shared buffer before propagating that error.
        restore();
        throw;
    }
    restore();
    if(!accepted){reason="prepare or execution rejected";return false;}
    auto value=[&](const std::vector<unsigned char>&data,size_t i) {
        float x;
        if(c.bfloat16){uint16_t h;memcpy(&h,data.data()+2*i,2);uint32_t u=uint32_t(h)<<16;memcpy(&x,&u,4);}
        else memcpy(&x,data.data()+4*i,4);
        return double(x);
    };
    double square=0,error=0;
    for(size_t i=0;i<c.elements;++i){double x=value(reference,i),y=value(actual,i);
        if(!std::isfinite(x)||!std::isfinite(y)){reason="non-finite output";return false;}
        square+=x*x;error+=(x-y)*(x-y);
    }
    l2=square>0?std::sqrt(error/square):std::sqrt(error);return true;
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
    Call call{&family,signature,&context,FastllmCudaGraphIsCapturingFast(),CudaUnsupported,
        CudaMeasure,CudaValidate,CudaHardware};
    return Select(call);
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
