#ifndef USE_ROCM
#include "fastllm-cuda-disk-moe-async.h"
#include "fastllm-cuda.cuh"
#include "utils.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <set>
#include <tuple>

namespace {
struct alignas(64) DiskMailboxControl {
    alignas(64) unsigned published;
    alignas(64) unsigned completed;
    alignas(64) unsigned failed;
    int32_t ids[16], gpu[16];
    float scores[16];
};
__global__ void DiskMailboxPublish(DiskMailboxControl *control, float *hostInput,
        const float *input, const int32_t *ids, const float *scores,
        const uint8_t *const *records, const uint8_t **localRecords,
        int32_t *localSlots, int32_t *localIds, float *localScores,
        int hidden, int experts, int topk, unsigned sequence) {
    for (int c=threadIdx.x;c<hidden;c+=blockDim.x) hostInput[c]=input[c];
    if (threadIdx.x<topk) {
        const int r=threadIdx.x, e=max(0,min(ids[r],experts-1));
        const auto *record=records[e];
        localRecords[r]=record; localSlots[r]=record?r:-1;
        localIds[r]=e; localScores[r]=scores[r];
        control->ids[r]=e; control->scores[r]=scores[r]; control->gpu[r]=record?1:0;
    }
    __threadfence_system();
    __syncthreads();
    if (!threadIdx.x) *reinterpret_cast<volatile unsigned *>(&control->published)=sequence;
}
__global__ void DiskMailboxWait(const DiskMailboxControl *control, volatile unsigned *failed,
                               unsigned sequence, unsigned long long timeoutCycles) {
    if (threadIdx.x) return;
    const auto begin=clock64();
    while (*reinterpret_cast<const volatile unsigned *>(&control->completed)!=sequence) {
        if(*failed || clock64()-begin>timeoutCycles){*failed=1;__threadfence_system();return;}
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__>=700
        __nanosleep(200);
#endif
    }
    __threadfence_system();
}
__global__ void DiskMailboxMerge(const float *gpu, const float *host, float *output,
        const int32_t *slots,const int32_t *ids,const float *scores,int hidden,int topk) {
    __shared__ int order[16];
    if (!threadIdx.x) {
        for(int r=0;r<topk;++r) {
            int at=r;
            while(at && ids[order[at-1]]>ids[r]) {order[at]=order[at-1];--at;}
            order[at]=r;
        }
    }
    __syncthreads();
    const int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(col>=hidden)return;
    float sum=0;
    for(int j=0;j<topk;++j) {
        const int r=order[j];
        const float value=slots[r]>=0?__fmul_rn(gpu[size_t(r)*hidden+col],scores[r]):host[size_t(r)*hidden+col];
        sum=__fadd_rn(sum,value);
    }
    output[col]=sum;
}
size_t Align(size_t n){return (n+255)&~size_t(255);}
}
struct FastllmCudaDiskMoeAsync::Impl {
    int device,hidden,maxInter;
    unsigned long long timeoutCycles=0;
    std::vector<FastllmCudaDiskMoeAsyncLayout> layers;
    void *host=nullptr,*mapped=nullptr,*gpu=nullptr;
    size_t rowBytes=0,inputOffset=0,outputOffset=0,recordsOffset=0;
    const uint8_t **localRecords=nullptr;
    int32_t *slots=nullptr,*ids=nullptr;
    float *scores=nullptr,*partial=nullptr;
    void *scratch=nullptr;size_t scratchBytes=0;
    fastllm::Data activation,result,warmInput,warmIndex,warmScore;
    ~Impl(){
        const int previous=FastllmCudaGetDevice();FastllmCudaSetDevice(device);
        if(gpu)FastllmCudaDirectFree(gpu);
        if(host)cudaFreeHost(host);
        FastllmCudaSetDevice(previous);
    }
    DiskMailboxControl *Control(int layer,bool deviceSide=false)const{
        return reinterpret_cast<DiskMailboxControl *>(static_cast<uint8_t *>(deviceSide?mapped:host)+rowBytes*layer);
    }
    uint8_t *Row(int layer,bool deviceSide=false)const{
        return static_cast<uint8_t *>(deviceSide?mapped:host)+rowBytes*layer;
    }
    bool Launch(int layer,const fastllm::Data &x,const fastllm::Data &index,
                const fastllm::Data &score,unsigned sequence,const std::function<void()> &shared,float *target=nullptr){
        const auto &l=layers[layer];
        const int k=index.dims[1];
        DiskMailboxPublish<<<1,256,0,cudaStreamPerThread>>>(Control(layer,true),
            reinterpret_cast<float *>(Row(layer,true)+inputOffset),static_cast<const float *>(x.cudaData),
            static_cast<const int32_t *>(index.cudaData),static_cast<const float *>(score.cudaData),
            reinterpret_cast<const uint8_t *const *>(Row(layer,true)+recordsOffset),
            localRecords,slots,ids,scores,hidden,l.experts,k,sequence);
        if(shared)shared();
        FastllmCudaMoeGGUFCacheView view{nullptr,slots,l.recordBytes,l.gateBytes,
            l.gateType,l.downType,hidden,l.inter,scratch,scratchBytes};
        view.recordPointers=localRecords;
        if(!FastllmCudaMoeGGUFCacheCompute(x,activation,result,view,scores,k,partial))return false;
        DiskMailboxWait<<<1,32,0,cudaStreamPerThread>>>(Control(layer,true),&Control(0,true)->failed,sequence,timeoutCycles);
        DiskMailboxMerge<<<(hidden+255)/256,256,0,cudaStreamPerThread>>>(partial,
            reinterpret_cast<const float *>(Row(layer,true)+outputOffset),
            target?target:static_cast<float *>(result.cudaData),slots,ids,scores,hidden,k);
        return cudaPeekAtLastError()==cudaSuccess;
    }
};
FastllmCudaDiskMoeAsync::FastllmCudaDiskMoeAsync(std::unique_ptr<Impl> state):impl(std::move(state)){}
FastllmCudaDiskMoeAsync::~FastllmCudaDiskMoeAsync()=default;
std::unique_ptr<FastllmCudaDiskMoeAsync> FastllmCudaDiskMoeAsync::Create(
        const std::vector<FastllmCudaDiskMoeAsyncLayout> &layers,int device){
    if(layers.empty())return {};
    cudaDeviceProp prop{};
    if(cudaGetDeviceProperties(&prop,device)!=cudaSuccess || !prop.canMapHostMemory || !prop.unifiedAddressing)return {};
    auto p=std::make_unique<Impl>();p->device=device;
    p->timeoutCycles=static_cast<unsigned long long>(prop.clockRate)*1000*10;p->layers=layers;p->hidden=layers[0].hidden;p->maxInter=0;
    int experts=0;
    for(const auto &l:layers){
        if(l.hidden!=p->hidden || l.hidden<=0 || l.inter<=0 || l.experts<=0 ||
           !FastllmCudaMoeGGUFCacheQ8Supported(l.gateType,l.downType,l.hidden,l.inter))return {};
        p->maxInter=std::max(p->maxInter,l.inter);experts=std::max(experts,l.experts);
        p->scratchBytes=std::max(p->scratchBytes,FastllmCudaMoeGGUFCacheBatchWorkspaceBytes(l.hidden,l.inter,1,16));
    }
    p->inputOffset=Align(sizeof(DiskMailboxControl));
    p->outputOffset=p->inputOffset+Align(size_t(p->hidden)*4);
    p->recordsOffset=p->outputOffset+Align(size_t(16)*p->hidden*4);
    p->rowBytes=p->recordsOffset+Align(size_t(experts)*sizeof(void *));
    const size_t hostBytes=p->rowBytes*layers.size();
    const int previous=FastllmCudaGetDevice();FastllmCudaSetDevice(device);
    struct Restore{int device;~Restore(){FastllmCudaSetDevice(device);}}restore{previous};
    if(cudaHostAlloc(&p->host,hostBytes,cudaHostAllocPortable|cudaHostAllocMapped)!=cudaSuccess){cudaGetLastError();return {};}
    if(cudaHostGetDevicePointer(&p->mapped,p->host,0)!=cudaSuccess){cudaGetLastError();return {};}
    std::memset(p->host,0,hostBytes);
    size_t size=0;
    auto take=[&](size_t n){const auto at=size;size+=Align(n);return at;};
    const size_t pointers=take(16*sizeof(void *)),slots=take(16*4),ids=take(16*4),scores=take(16*4);
    const size_t activation=take(size_t(16)*p->maxInter*4),partial=take(size_t(16)*p->hidden*4);
    const size_t result=take(size_t(p->hidden)*4),scratch=take(p->scratchBytes),warm=take(size_t(p->hidden)*4);
    if(FastllmCudaTryDirectMalloc(&p->gpu,size)!=FASTLLM_CUDA_TRY_MALLOC_SUCCESS)return {};
    auto *base=static_cast<uint8_t *>(p->gpu);
    p->localRecords=reinterpret_cast<const uint8_t **>(base+pointers);
    p->slots=reinterpret_cast<int32_t *>(base+slots);p->ids=reinterpret_cast<int32_t *>(base+ids);
    p->scores=reinterpret_cast<float *>(base+scores);p->partial=reinterpret_cast<float *>(base+partial);
    p->scratch=base+scratch;
    auto borrow=[&](fastllm::Data &data,fastllm::DataType type,std::vector<int> dims,size_t offset){
        data.dataType=type;data.UpdateUnitSize();data.Resize(dims);data.dataDevice=fastllm::CUDA;
        data.dataDeviceIds={device};data.cudaData=base+offset;data.cudaDataBorrowed=true;
    };
    borrow(p->activation,fastllm::FLOAT32,{1,16,p->maxInter},activation);
    borrow(p->result,fastllm::FLOAT32,{1,p->hidden},result);
    borrow(p->warmInput,fastllm::FLOAT32,{1,p->hidden},warm);
    borrow(p->warmIndex,fastllm::INT32,{1,16},ids);
    borrow(p->warmScore,fastllm::FLOAT32,{1,16},scores);
    if(cudaMemsetAsync(p->gpu,0,size,cudaStreamPerThread)!=cudaSuccess)return {};
    // Load every kernel before a real GPU waiter can be queued. The dummy
    // sequence is already completed, so warmup never needs a CPU coordinator.
    std::set<std::tuple<int,int,int>> warmed;
    for(size_t i=0;i<layers.size();++i)if(warmed.emplace(layers[i].gateType,layers[i].downType,layers[i].inter).second){
        if(!p->Launch(i,p->warmInput,p->warmIndex,p->warmScore,0,{}))return {};
    }
    if(cudaStreamSynchronize(cudaStreamPerThread)!=cudaSuccess)return {};
    return std::unique_ptr<FastllmCudaDiskMoeAsync>(new FastllmCudaDiskMoeAsync(std::move(p)));
}
const uint8_t **FastllmCudaDiskMoeAsync::Records(int layer){return reinterpret_cast<const uint8_t **>(impl->Row(layer)+impl->recordsOffset);}
unsigned FastllmCudaDiskMoeAsync::Published(int layer)const{return __atomic_load_n(&impl->Control(layer)->published,__ATOMIC_ACQUIRE);}
bool FastllmCudaDiskMoeAsync::Failed()const{return __atomic_load_n(&impl->Control(0)->failed,__ATOMIC_ACQUIRE)!=0;}
void FastllmCudaDiskMoeAsync::ResetFailure(){__atomic_store_n(&impl->Control(0)->failed,0u,__ATOMIC_RELEASE);}
FastllmCudaDiskMoeAsync::HostRequest FastllmCudaDiskMoeAsync::Request(int layer)const{
    auto *c=impl->Control(layer);auto *row=impl->Row(layer);
    return {reinterpret_cast<float *>(row+impl->inputOffset),c->ids,c->scores,c->gpu,reinterpret_cast<float *>(row+impl->outputOffset)};
}
void FastllmCudaDiskMoeAsync::Complete(int layer,unsigned sequence){__atomic_store_n(&impl->Control(layer)->completed,sequence,__ATOMIC_RELEASE);}
bool FastllmCudaDiskMoeAsync::CanRun(int layer,const fastllm::Data &input,const fastllm::Data &index,
        const fastllm::Data &score,const fastllm::Data &output)const{
    if(layer<0 || layer>=int(impl->layers.size()) || input.dataType!=fastllm::FLOAT32 ||
       (input.dims.size()!=2 || input.dims[0]!=1 || input.dims[1]!=impl->hidden) || input.dataDevice!=fastllm::CUDA ||
       (index.dataType!=fastllm::INT32 && index.dataType!=fastllm::INT32PARAM) || score.dataType!=fastllm::FLOAT32 ||
       index.dims!=score.dims || index.dims.size()!=2 || index.dims[0]!=1 || index.dims[1]<1 || index.dims[1]>16 ||
       index.dataDevice!=fastllm::CUDA || score.dataDevice!=fastllm::CUDA ||
       !input.cudaData || !index.cudaData || !score.cudaData ||
       (!input.dataDeviceIds.empty() && input.dataDeviceIds[0]!=impl->device) ||
       (!index.dataDeviceIds.empty() && index.dataDeviceIds[0]!=impl->device) ||
       (!score.dataDeviceIds.empty() && score.dataDeviceIds[0]!=impl->device) ||
       input.multiDeviceData || index.multiDeviceData || score.multiDeviceData || output.multiDeviceData ||
       output.isFake || output.cudaDataBorrowed)return false;
    return true;
}
bool FastllmCudaDiskMoeAsync::Run(int layer,const fastllm::Data &input,const fastllm::Data &index,
        const fastllm::Data &score,fastllm::Data &output,unsigned sequence,const std::function<void()> &shared){
    if(!CanRun(layer,input,index,score,output))return false;
    if(output.dataType!=fastllm::FLOAT32)output.FreeSpace();
    output.dataType=fastllm::FLOAT32;output.UpdateUnitSize();output.Resize(input.dims);
    output.ToDevice(fastllm::CUDA,{impl->device},false);output.Allocate(false);
    const bool ok=impl->Launch(layer,input,index,score,sequence,shared,static_cast<float *>(output.cudaData));
    fastllm::AssertInFastLLM(ok,"Disk asynchronous GGUF launch failed.\n");
    return true;
}
#endif
