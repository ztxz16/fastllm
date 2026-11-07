#define main ExistingExl3MoeTests
#include "test_exl3_moe.cpp"
#undef main
#include "devices/numas/numasdevice.h"
#include "devices/numas/numas.h"
#include <unistd.h>
#include <set>
#include "devices/cuda/fastllm-cuda.cuh"

static bool DecodeClose(const float *a,const float *b,size_t size,int width) {
    for(size_t first=0;first<size;first+=width) {
        double error=0,norm=0;
        for(size_t i=first;i<std::min(size,first+width);++i) {
            if(!std::isfinite(a[i]) || !std::isfinite(b[i]))return false;
            error+=double(a[i]-b[i])*(a[i]-b[i]);norm+=double(a[i])*a[i];
        }
        const double rel=std::sqrt(error/std::max(norm,1e-30));
        std::cout<<"EXL3 decoded expert relative L2 "<<rel<<std::endl;
        if(rel>.02)return false;
    }
    return true;
}

int main() {
    SetThreads(8);
    constexpr int experts=11,k=1024,mid=1024,topk=4;
    std::mt19937 rng(32411);
    std::vector<std::unique_ptr<Data>> owner,copy;
    std::vector<Data*> weights(2+2*experts,nullptr),original(weights.size(),nullptr);
    std::vector<std::pair<Data*,std::string>> pending;
    const int bits[]={2,3,4,5,6,7,8,10,12,14,16};
    for(int e=0;e<experts;++e)for(int part=0;part<2;++part) {
        auto w=std::make_unique<Data>(EXL3);w->blockM=part?1:2;w->blockK=bits[e];
        w->Resize(part?std::vector<int>{k,mid}:std::vector<int>{2*mid,k});Fill(*w,rng);
        auto old=std::make_unique<Data>();old->CopyFrom(*w);
        weights[2+2*e+part]=w.get();original[2+2*e+part]=old.get();
        pending.push_back({w.get(),part?"linear":"linearSwiglu"});
        owner.push_back(std::move(w));copy.push_back(std::move(old));
    }
    HostMoePlacementScope scope(false);
    RegisterNumasMoeWeightBatch(pending);
    const auto workers=GetNumasExpertWorkerIds();
    std::set<int> usedNodes;
    int sampled=0;
    for(size_t i=0;i<owner.size();++i) {
        auto &w=*owner[i];
        Require(w.cpuData==nullptr && w.numasData.size()==workers.size(),"Common NUMA shard storage");
        Require(IsNumasLinearWeightRegistered(&w),"Common weight capability");
        std::vector<uint8_t> canonical(w.GetBytes());Exl3CopyCanonical(w,canonical.data());
        Require(std::memcmp(canonical.data(),copy[i]->cpuData,w.GetBytes())==0,"Shard roundtrip changed packed bytes");
        for(size_t node=0;node<w.numasData.size();++node) {
            usedNodes.insert(node);
            const uintptr_t page=sysconf(_SC_PAGESIZE),base=uintptr_t(w.numasData[node]);
            const size_t bytes=w.GetBytes()/w.numasData.size();
            for(uintptr_t at=(base+page-1)/page*page;at+page<=base+bytes;at+=page*13) {
                void *address=reinterpret_cast<void*>(at);int status=-1;
                Require(move_pages(0,1,&address,nullptr,&status,0)==0,"Cannot query physical NUMA page");
                Require(status==GetNumasPhysicalNode(node),"Row shard is on wrong NUMA node");++sampled;
            }
        }
        // Both registered mapped and deliberately unpinned fallback restores.
        for(bool pinned:{true,false}) {
            bool saved=w.isPinned;if(!pinned)w.isPinned=false;
            w.ToCudaTemporary({},true);
            std::vector<uint8_t> gpu(w.GetBytes());FastllmCudaCopyFromDeviceToHost(gpu.data(),w.cudaData,gpu.size());
            Require(gpu==canonical,"GPU restore changed packed bytes");
            w.FreeCudaTemporary({},false);w.isPinned=saved;
        }
    }
    Require(CanRunNumasMoeDecodeExperts(weights.data(),weights.size()),"Common decode capability");
    if(workers.size()>1)Require(usedNodes.size()>1,"Every expert must use multiple NUMA nodes");
    int cases=0;
    for(int e=0;e<experts;++e)for(int part=0;part<2;++part)for(int rows:{1,3,9})for(auto type:{FLOAT32,FLOAT16,BFLOAT16}) {
        auto &w=*weights[2+2*e+part];Data x(type,{rows,w.dims[1]}),bias,out(type,{rows,w.dims[0]}),expected(type,{rows,w.dims[0]});
        x.Allocate(false);
        for(int j=0;j<rows*w.dims[1];++j){float v=(int(rng()%127)-63)/128.f;
            if(type==FLOAT32)reinterpret_cast<float*>(x.cpuData)[j]=v;
            else reinterpret_cast<uint16_t*>(x.cpuData)[j]=type==FLOAT16?float_to_half(v):Float32ToBFloat16RNEBits(v);}
        Exl3LinearCPU(x,*original[2+2*e+part],bias,expected);Exl3LinearCPU(x,w,bias,out);
        Require(std::memcmp(out.cpuData,expected.cpuData,out.GetBytes())==0,"NUMA projection is not bit exact");++cases;
    }
    std::vector<float> input(k),a(topk*k,-123),b(a);
    for(int i=0;i<k;++i)input[i]=(int(rng()%127)-63)/128.f;
    for(int turn=0;turn<16;++turn) {
        int ids[topk],gpu[topk];float scores[topk];
        for(int j=0;j<topk;++j){ids[j]=(turn+j)%experts;gpu[j]=(turn&(1<<j))?ids[j]:-1;scores[j]=.15f+.1f*j;}
        int callbacks=0;
        Exl3MoeCPUExperts(input.data(),a.data(),1,original.data(),ids,gpu,scores,topk,.01f);
        NumasMoeDecodeExpertsWithOverlap(input.data(),b.data(),weights.data(),ids,gpu,topk,0,[&]{++callbacks;},scores,.01f,32);
        Require(DecodeClose(a.data(),b.data(),a.size(),k),"NUMA route/BF16/clamp mismatch");
        Require(callbacks==1,"NUMA callback count");++cases;
    }
    // Reuse placements after resizing the worker pool and with one active worker.
    for (int threads : {4,12,8}) {
        SetThreads(threads);
        for (bool limited : {false,true}) {
            auto *pool=GetAlivePool();const auto saved=pool->curActivateThreadInterval;
            if(limited) pool->curActivateThreadInterval={0,1};
            int ids[topk]={0,3,5,7},gpu[topk]={-1,-1,-1,-1};float scores[topk]={.1f,.2f,.3f,.4f};
            Exl3MoeCPUExperts(input.data(),a.data(),1,original.data(),ids,gpu,scores,topk,.01f);
            NumasMoeDecodeExperts(input.data(),b.data(),weights.data(),ids,gpu,topk,0,scores,.01f);
            Require(DecodeClose(a.data(),b.data(),a.size(),k),"NUMA resized/restricted worker mismatch");
            pool->curActivateThreadInterval=saved;++cases;
        }
    }
    // Exercise the device fallback directly, including lazy registration and
    // CUDA input copies. The reference is computed before registration.
    for (auto type : {FLOAT32,BFLOAT16}) for (bool cudaInput : {false,true}) {
        Data x(type,{3,k}),ids(INT32,{3,topk}),scores(FLOAT32,{3,topk}),expected(type,{3,k}),actual(type,{3,k});
        x.Allocate(false);ids.Allocate(false);scores.Allocate(false);
        for(int i=0;i<3*k;++i) {
            const float v=input[i%k];
            if(type==FLOAT32)reinterpret_cast<float*>(x.cpuData)[i]=v;
            else reinterpret_cast<uint16_t*>(x.cpuData)[i]=Float32ToBFloat16RNEBits(v);
        }
        for(int i=0;i<3*topk;++i){reinterpret_cast<int*>(ids.cpuData)[i]=i%experts;reinterpret_cast<float*>(scores.cpuData)[i]=.1f*(1+i%topk);}
        Exl3MoeCPU(x,expected,ids,scores,original.data(),original.size(),0,.01f);
        if(cudaInput)for(auto *v:{&x,&ids,&scores})v->ToDevice(CUDA,std::vector<int>{0});
        NumasMergeMOE operation;
        operation.Run("MergeMOE",{{"input",&x},{"index",&ids},{"score",&scores},{"output",&actual},{"weights",reinterpret_cast<Data*>(original.data())}},
            {{"sharedScale",0},{"swigluLimit",.01f}},{{"weights___batch",int(original.size())}});
        Require(std::memcmp(actual.cpuData,expected.cpuData,actual.GetBytes())==0,"NUMA device fallback mismatch");++cases;
    }
    // A cloned format restores canonical bytes and owns no borrowed arena.
    const auto shards=weights[2]->numasData;weights[2]->CopyFrom(*weights[2]);
    Require(weights[2]->numasData==shards,"Self-copy changed NUMA placement");
    Data cloned;cloned.CopyFrom(*weights[2]);Require(cloned.numasData.empty(),"Copy inherited borrowed NUMA storage");
    std::vector<uint8_t> clonedExpected(cloned.GetBytes());Exl3CopyCanonical(*copy[0],clonedExpected.data());
    Require(std::memcmp(cloned.cpuData,clonedExpected.data(),cloned.GetBytes())==0,"Clone changed canonical bytes");
    // Callback failures must drain workers before the shared workspace is reused.
    int ids[topk]={0,1,2,3},gpu[topk]={-1,-1,-1,-1};float scores[topk]={.1f,.2f,.3f,.4f};
    bool caught=false;
    try { NumasMoeDecodeExpertsWithOverlap(input.data(),b.data(),weights.data(),ids,gpu,topk,0,
        []{throw std::runtime_error("test callback");},scores,.01f,32); } catch(const std::runtime_error &){caught=true;}
    Require(caught,"Callback exception was lost");
    Exl3MoeCPUExperts(input.data(),a.data(),1,original.data(),ids,gpu,scores,topk,.01f);
    NumasMoeDecodeExperts(input.data(),b.data(),weights.data(),ids,gpu,topk,0,scores,.01f);
    Require(DecodeClose(a.data(),b.data(),a.size(),k),"Workspace reuse after callback exception");
    std::cout<<"EXL3 NUMA PASS: "<<cases<<" storage/projection/decode cases, "<<sampled<<" physical pages, "<<usedNodes.size()<<" nodes"<<std::endl;
    return 0;
}
