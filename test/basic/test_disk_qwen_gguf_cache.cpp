#include "fastllm.h"
#include "executor.h"
#include "devices/cpu/cpudevice.h"
#include "utils.h"
#include "gguf.h"
#include "devices/disk/diskdevice.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-v41-gguf.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <unistd.h>
using namespace fastllm;
struct CpuReference : CpuMergeMOE { using CpuMergeMOE::Run; };
static void Check(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
static std::string currentCase;
static float ReadValue(const Data &data, size_t index) {
    if (data.dataType == FLOAT32) return reinterpret_cast<const float *>(data.cpuData)[index];
    if (data.dataType == FLOAT16) return half_to_float(reinterpret_cast<const uint16_t *>(data.cpuData)[index]);
    Check(data.dataType == BFLOAT16, "floating output type");
    uint32_t bits = uint32_t(reinterpret_cast<const uint16_t *>(data.cpuData)[index]) << 16;
    float value; memcpy(&value, &bits, sizeof(value)); return value;
}
struct Fixture {
    int hidden, inter, experts;
    std::string path;
    std::vector<std::unique_ptr<Data>> owned;
    std::vector<Data *> memory, disk;
    uint64_t bytes=0;
    Fixture(int variant,int hidden=512,int inter=256,int experts=12) : hidden(hidden),inter(inter),experts(experts),memory(2*(experts+1)),disk(memory.size()) {
        char name[]="/tmp/fastllm-disk-gguf-XXXXXX";
        int fd=mkstemp(name); Check(fd>=0,"tempfile"); path=name;
        FILE *file=fdopen(fd,"wb"); std::mt19937 rng(712+variant);
        for(int e=0;e<experts;++e) for(int part=0;part<2;++part) {
            const auto type=variant>=4 ? (part ? GGML_TYPE_Q2_0 :
                variant==4 ? GGML_TYPE_IQ2_S : variant==5 ? GGML_TYPE_IQ2_XXS :
                variant==6 ? GGML_TYPE_IQ2_XS : GGML_TYPE_IQ1_M) :
                part ? (variant%2 ? GGML_TYPE_IQ2_XS : GGML_TYPE_IQ3_XXS) :
                       (variant<2 ? GGML_TYPE_IQ2_XXS : GGML_TYPE_IQ2_XS);
            auto dims=part ? std::vector<int>{hidden,inter} : std::vector<int>{2*inter,hidden};
            auto w=std::make_unique<Data>(DATA_GGUF_FORMAT,type,dims);
            w->disableGGUFRepack=true; w->isGGUFData=true; w->Allocate(false);
            for(size_t i=0;i<w->GetBytes();++i) w->cpuData[i]=rng()>>24;
            for(size_t i=0;i<w->GetBytes();i+=ggml_type_size(type)) {
                uint16_t scale=float_to_half(std::ldexp(1.f,-12+int(rng()%3)));
                if(type==GGML_TYPE_IQ1_M){
                    auto *b=reinterpret_cast<block_iq1_m*>(w->cpuData+i);
                    for(int j=0;j<4;++j){
                        uint16_t bits;memcpy(&bits,b->scales+j*2,2);
                        bits=(bits&0x0fff)|(((scale>>(j*4))&15)<<12);
                        memcpy(b->scales+j*2,&bits,2);
                    }
                }else memcpy(w->cpuData+i,&scale,2);
            }
            auto lazy=std::make_unique<Data>(DATA_GGUF_FORMAT,type,dims);
            lazy->isDiskWeight=true; lazy->isModelWeight=true; lazy->isGGUFData=true;
            char padding[37]={}; fwrite(padding,1,sizeof(padding),file);
            DiskWeightPart p; p.fileName=path; p.fileOffset=ftell(file);
            p.bytes=w->GetBytes(); p.sourceDataType=DATA_GGUF_FORMAT; p.dims=dims;
            lazy->diskWeightParts.push_back(p);
            Check(fwrite(w->cpuData,1,p.bytes,file)==p.bytes,"write");
            if(e==0) bytes+=w->GetBytes();
            memory[2+2*e+part]=w.get(); disk[2+2*e+part]=lazy.get();
            owned.push_back(std::move(w)); owned.push_back(std::move(lazy));
        }
        fclose(file);
    }
    ~Fixture(){owned.clear();unlink(path.c_str());}
    std::vector<float> Run(bool lazy,int rows,int offset,int device,int topk=3,bool handoff=false,DataType inputType=FLOAT16,bool staleCuda=false,bool asynchronous=false) {
        static int call = 0;
        currentCase = "gate=" + std::to_string(disk[2]->ggmlType) + " down=" + std::to_string(disk[3]->ggmlType) +
            " rows=" + std::to_string(rows) + " topk=" + std::to_string(topk) + " offset=" + std::to_string(offset) +
            " device=" + std::to_string(device) + " call=" + std::to_string(++call);
        Data x(inputType,{rows,hidden}), y(FLOAT32,{rows,hidden});
        Data ids(INT32PARAM,{rows,topk}), scores(FLOAT32,{rows,topk}),a,b,c;
        x.Allocate(false);ids.Allocate(false);scores.Allocate(false);
        for(int r=0;r<rows;++r) {
            for(int j=0;j<hidden;++j) {
                const float value=std::sin((r*hidden+j)*.71f)*3;
                const size_t at=size_t(r)*hidden+j;
                if(inputType==FLOAT32) reinterpret_cast<float*>(x.cpuData)[at]=value;
                else if(inputType==FLOAT16) reinterpret_cast<uint16_t*>(x.cpuData)[at]=float_to_half(value);
                else { uint32_t bits;memcpy(&bits,&value,4);reinterpret_cast<uint16_t*>(x.cpuData)[at]=bits>>16; }
            }
            for(int k=0;k<topk;++k) {
                ((int32_t*)ids.cpuData)[r*topk+k]=(offset+(k==2 && r%3==0?0:k*2)+(experts>12?r%experts:r%2))%experts;
                ((float*)scores.cpuData)[r*topk+k]=k==0?0:k==1?.375f:-.125f;
            }
        }
        auto &table=lazy?disk:memory;std::vector<Data*> biases(table.size());
        DataDict data={{"input",&x},{"output",&y},{"index",&ids},{"score",&scores},
            {"weights",(Data*)table.data()},{"biass",(Data*)biases.data()},{"w1",&a},{"w2",&b},{"w3",&c}};
        IntDict ip={{"weights___batch",int(table.size())},{"biass___batch",int(table.size())},
            {"gateType",int(MoeGateSwiglu)}};
        FloatDict fp={{"sharedScale",0}};
        if(lazy) {
            if(device>=0){
                FastllmCudaSetDevice(device);x.ToDevice(CUDA,{device},true);
                if(!handoff){
                    x.ToDevice(CPU);
                    if(staleCuda) FastllmCudaMemset0(x.cudaData,x.GetBytes());
                }
            }
            DiskMergeMOE op;
            if(handoff) {
                ids.dataType=INT32;ids.UpdateUnitSize();ids.ToDevice(CUDA,{device},true);
                scores.ToDevice(CUDA,{device},true);
                void *xp=x.cudaData,*ip=ids.cudaData,*sp=scores.cudaData;
                int callbacks=0;
                if(asynchronous) {
                    DiskMoeAsyncScope scope({disk},device,true);
                    Check(DiskMoeAsyncScope::TryRun(0,x,ids,scores,y,[&]{++callbacks;}),"async ordinary GPU handoff rejected");
                    scope.Finish();
                } else Check(op.RunGpuGGUFDecode(x,ids,scores,y,table,[&]{++callbacks;}),"ordinary GPU handoff rejected");
                Check(callbacks==1,"shared callback must run exactly once");
                Check(x.dataDevice==CUDA && ids.dataDevice==CUDA && scores.dataDevice==CUDA &&
                    x.cudaData==xp && ids.cudaData==ip && scores.cudaData==sp,"captured inputs changed storage/placement");
            } else op.Run("MergeMOE",data,fp,ip);
        } else {
            // Independent CPU expert math, one token at a time, is the numeric oracle.
            y.Allocate(false);
            for(int r=0;r<rows;++r){
                const size_t rowBytes=x.GetBytes()/rows;
                Data xr(inputType,{1,hidden},CPU,x.cpuData+size_t(r)*rowBytes),yr(FLOAT32,{1,hidden});
                Data ir(INT32PARAM,{1,topk},CPU,ids.cpuData+r*topk*4),sr(FLOAT32,{1,topk},CPU,scores.cpuData+r*topk*4);
                auto one=data;one["input"]=&xr;one["output"]=&yr;one["index"]=&ir;one["score"]=&sr;
                Data curInput, curOutput;
                one["curInput"]=&curInput;one["curOutput"]=&curOutput;
                CpuReference reference; reference.Run("MergeMOE",one,fp,ip);
                for (int j = 0; j < hidden; ++j)
                    reinterpret_cast<float *>(y.cpuData)[size_t(r)*hidden+j] = ReadValue(yr, j);
            }
        }
        y.ToDevice(CPU);std::vector<float> out(rows*hidden);
        for(size_t j=0;j<out.size();++j) out[j]=ReadValue(y, j);
        return out;
    }
};
static void Compare(const std::vector<float>&a,const std::vector<float>&b) {
    double err=0,norm=0;
    for(size_t i=0;i<a.size();++i){Check(std::isfinite(a[i]),"finite");err+=(a[i]-b[i])*(a[i]-b[i]);norm+=b[i]*b[i];}
    double rel=std::sqrt(err/std::max(norm,1e-20));
    if(rel>.035){fprintf(stderr,"%s relative L2 %.8g; actual %.8g %.8g reference %.8g %.8g\n",currentCase.c_str(),rel,a[0],a[1],b[0],b[1]);throw std::runtime_error("Qwen GGUF disk numerics");}
}
static void TestPrefillMemoryPressure() {
    Fixture f(4);
    constexpr int rows = 64, topk = 3;
    const auto expected = f.Run(false,rows,0,-1,topk,false,FLOAT32);
    Data input(FLOAT32,{rows,f.hidden}), output, workspace;
    input.Allocate(false);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < f.hidden; ++c)
            reinterpret_cast<float *>(input.cpuData)[r*f.hidden+c] = std::sin((r*f.hidden+c)*.71f)*3;
    input.ToDevice(CUDA,{0},true);
    std::vector<int32_t> ids(rows*topk);
    std::vector<float> scores(ids.size());
    for (int r = 0; r < rows; ++r) for (int k = 0; k < topk; ++k) {
        ids[r*topk+k] = ((k==2 && r%3==0 ? 0 : k*2)+r%2)%f.experts;
        scores[r*topk+k] = k==0 ? 0 : k==1 ? .375f : -.125f;
    }
    FastllmCudaV41DiskStaging staging;
    FastllmCudaV41DiskExperts source;
    source.staging = &staging;
    source.residents.resize(2*f.experts,nullptr);
    std::unique_ptr<FILE,int(*)(FILE *)> checkpoint(std::fopen(f.path.c_str(),"rb"),std::fclose);
    Check(checkpoint!=nullptr,"pressure-test checkpoint");
    int reads = 0, releases = 0;
    source.read = [&](int e, int part, uint8_t *storage, size_t capacity) {
        const auto &weight = *f.disk[2*e+part];
        Check(storage && weight.GetBytes()<=capacity,"pressure-test staging capacity");
        Check(fseeko(checkpoint.get(),weight.diskWeightParts[0].fileOffset,SEEK_SET)==0 &&
              std::fread(storage,1,weight.GetBytes(),checkpoint.get())==weight.GetBytes(),
              "pressure-test checkpoint read");
        ++reads;
        std::promise<const uint8_t *> ready;
        ready.set_value(storage);
        return ready.get_future();
    };
    source.release = [&](int) { ++releases; };
    auto run = [&] {
        return FastllmCudaMergeMOEGGUFDisk(input,workspace,output,f.disk.data(),
            f.experts,ids.data(),scores.data(),topk,source);
    };
    auto compare = [&] {
        std::vector<float> actual(rows*f.hidden);
        FastllmCudaCopyFromDeviceToHost(actual.data(),output.cudaData,actual.size()*sizeof(float));
        Compare(actual,expected);
    };
    // Warm kernels and persistent staging before deliberately reducing driver
    // free memory. Both phases exercise the actual disk stream admission.
    Check(run(),"pressure-test prefill warmup");
    compare();
    using Allocation = std::unique_ptr<void,void(*)(void *)>;
    for (bool idle : {true,false}) {
        FastllmCudaSyncCurrentThreadStream();
        FastllmCudaClearBigBufferAll();
        Allocation pool(FastllmCudaMalloc(512ULL<<20),FastllmCudaFree);
        uint32_t marker = 0x12345678;
        FastllmCudaCopyFromHostToDevice(pool.get(),&marker,sizeof(marker));
        uint64_t free = 0, total = 0;
        Check(FastllmCudaGetMemoryInfo(0,free,total) && free>(512ULL<<20),
              "pressure test requires 1 GiB available GPU memory");
        void *held = nullptr;
        Check(FastllmCudaTryDirectMalloc(&held,free-(128ULL<<20)) == FASTLLM_CUDA_TRY_MALLOC_SUCCESS,
              "pressure-test live allocation");
        Allocation pressure(held,FastllmCudaForceFree);
        if (idle) pool.reset();
        uint64_t before = 0, after = 0;
        Check(FastllmCudaGetMemoryInfo(0,before,total) && before<(192ULL<<20),
              "pressure test did not reduce driver free memory");
        const int readsBefore = reads, releasesBefore = releases;
        void *outputBefore = output.cudaData, *workspaceBefore = workspace.cudaData;
        Check(run()==idle,idle ? "disk prefill rejected reclaimable idle memory" :
                                "disk prefill consumed its low-memory safety reserve");
        Check(!FastllmCudaGetThreadError(),"prefill rejection poisoned CUDA state");
        Check(FastllmCudaGetMemoryInfo(0,after,total),"pressure-test memory query");
        if (idle) {
            Check(after>before+(384ULL<<20) && reads>readsBefore && releases>releasesBefore,
                  "disk prefill did not reclaim idle storage and execute");
        } else {
            uint32_t observed = 0;
            FastllmCudaCopyFromDeviceToHost(&observed,pool.get(),sizeof(observed));
            Check(observed==marker && reads==readsBefore && releases==releasesBefore &&
                  output.cudaData==outputBefore && workspace.cudaData==workspaceBefore,
                  "rejected prefill changed live buffers or consumed expert sources");
        }
        compare();
        pressure.reset();
        pool.reset();
        Check(run(),"prefill did not recover after memory pressure");
        compare();
        std::printf("PASS prefill pressure idle=%d free_before=%llu free_after=%llu\n",
            idle,(unsigned long long)before,(unsigned long long)after);
    }
}

static void FillPackedCache(Fixture &f) {
    // Three 64-MiB arenas hold 44 records each for this fixture. Populate
    // through real decode admissions, keeping the final sub-record holes.
    for (int expert = 0; expert < 132; ++expert)
        for (int pass = 0; pass < 3; ++pass)
            f.Run(true,1,expert,0,1,true,FLOAT32);
    const auto stats = GetDiskMoeCacheStats();
    Check(stats.cudaBytes == 132 * f.bytes, "three fragmented CUDA arenas were not populated");
}
static void TestPromotionPreflight() {
    // Three full 64-MiB arenas each retain a sub-record hole. Together these
    // holes exceed a record, so logical bytes alone predict a false admission.
    Fixture f(4,2560,640,144);
    SetMoeCpuCacheBytes(f.bytes*6);SetMoeCudaCacheBytes(192ULL<<20);
    FillPackedCache(f);
    const auto ready=GetDiskMoeCacheStats();
    Check(ready.cudaBytes+f.bytes<=GetMoeCudaCacheBytes(),"fragmented test needs logical spare capacity");
    Compare(f.Run(true,1,133,0,3,true,FLOAT32),f.Run(false,1,133,-1,3,false,FLOAT32));
    const auto before=GetDiskMoeCacheStats();
    Compare(f.Run(true,1,133,0,3,true,FLOAT32),f.Run(false,1,133,-1,3,false,FLOAT32));
    const auto after=GetDiskMoeCacheStats();
    Check(after.cpuHits-before.cpuHits==3 && after.diskBytes==before.diskBytes,"preflight changed CPU routes");
    Check(after.uploads==before.uploads && after.cudaBytes==before.cudaBytes,"cold GPU victim protection");
    Check(after.cpuEvictions==before.cpuEvictions && after.asyncMaintenance==before.asyncMaintenance,
        "rejected promotion took CPU ownership or queued maintenance");
    // Streamed prefill can also propose a record while arena holes leave
    // logical capacity. A rejected upload must remain a valid CPU expert;
    // the next prefill must never consume it as a GPU-resident record.
    for (int pass = 0; pass < 2; ++pass) {
        const auto start = GetDiskMoeCacheStats();
        Compare(f.Run(true,40,133,0,3,false,FLOAT32),f.Run(false,40,133,-1,3,false,FLOAT32));
        const auto done = GetDiskMoeCacheStats();
        Check(done.cpuHits-start.cpuHits+done.cudaHits-start.cudaHits+done.misses-start.misses==120,
            "fragmented prefill route counts");
        Check(done.cpuBytes<=GetMoeCpuCacheBytes() && done.cudaBytes<=GetMoeCudaCacheBytes() &&
            !done.cpuCudaOverlapBytes,"fragmented prefill cache budgets");
        Check(done.uploads==start.uploads && done.cudaBytes==start.cudaBytes,
            "failed prefill upload published a GPU record");
    }
}
static void TestPackedCudaTrim() {
    Fixture f(4,2560,640,144);
    SetMoeCpuCacheBytes(f.bytes*6);
    SetMoeCudaCacheBytes(192ULL<<20);
    FillPackedCache(f);
    // Keep hot records in each of the three physical blocks. Logical-only
    // eviction leaves all three allocations pinned after either shrink.
    for (int expert : {0,44,88}) f.Run(true,1,expert,0,1,true,FLOAT32,false,true);
    FastllmCudaSyncCurrentThreadStream();
    uint64_t initialFree = 0, total = 0;
    Check(FastllmCudaGetMemoryInfo(0,initialFree,total),"initial trim memory query");
    for (uint64_t budget : {uint64_t(128)<<20,3*f.bytes}) {
        SetMoeCudaCacheBytes(budget);
        FastllmCudaSyncCurrentThreadStream();
        uint64_t free = 0;
        Check(FastllmCudaGetMemoryInfo(0,free,total),"trim memory query");
        const auto stats = GetDiskMoeCacheStats();
        Check(stats.cudaBytes<=budget && stats.cpuBytes<=GetMoeCpuCacheBytes() &&
            !stats.cpuCudaOverlapBytes,"trim tier budgets");
        // Allow driver allocation granularity, rather than equating payload
        // counters with physical VRAM. Unchanged model workspaces stay live.
        Check(free>=initialFree && free-initialFree+budget+(2ULL<<20)>=(192ULL<<20),
            "GPU shrink retained oversized physical blocks");
        std::printf("PASS packed trim budget=%llu logical=%llu freed=%llu\n",
            (unsigned long long)budget,(unsigned long long)stats.cudaBytes,
            (unsigned long long)(free-initialFree));
    }
    // Reuse the existing mailbox after its GPU pointers were invalidated,
    // then grow the budget again. Both directions must preserve numerics.
    for (int pass = 0; pass < 3; ++pass)
        Compare(f.Run(true,1,0,0,3,true,FLOAT32,false,true),f.Run(false,1,0,-1,3,false,FLOAT32));
    SetMoeCudaCacheBytes(192ULL<<20);
    for (int pass = 0; pass < 3; ++pass)
        Compare(f.Run(true,1,0,0,3,true,FLOAT32,false,true),f.Run(false,1,0,-1,3,false,FLOAT32));
    SetMoeCudaCacheBytes(0);
    Check(!GetDiskMoeCacheStats().cudaBytes,"zero GPU cache after regrowth");
}
static void TestAsyncReleasesPrefillWorkspace() {
    Fixture f(4,2560,640);
    SetMoeCpuCacheBytes(f.bytes*8);
    SetMoeCudaCacheBytes(f.bytes*4);
    const auto expected = f.Run(false,1,0,-1,10,false,FLOAT32);
    // Warm decode kernels before measuring driver memory, so their lazy
    // loading cannot hide the workspace released at the phase boundary.
    Compare(f.Run(true,1,0,0,10,true,FLOAT32,false,true),expected);
    for (int pass = 0; pass < 2; ++pass) {
        f.Run(true,4096,0,0,10,false,FLOAT32);
        FastllmCudaSyncCurrentThreadStream();
        uint64_t before = 0, after = 0, total = 0;
        Check(FastllmCudaGetMemoryInfo(0,before,total),"prefill workspace memory query");
        {
            DiskMoeAsyncScope warmup({f.disk},0,true,true);
            warmup.Finish();
        }
        uint64_t retained = 0;
        Check(FastllmCudaGetMemoryInfo(0,retained,total) && retained <= before+(2ULL<<20),
              "warmup lost the workspace before KV planning");
        Compare(f.Run(true,1,0,0,10,true,FLOAT32,false,true),expected);
        FastllmCudaSyncCurrentThreadStream();
        Check(FastllmCudaGetMemoryInfo(0,after,total),"decode workspace memory query");
        // This prompt's FP32 per-route output alone exceeds 400 MiB.
        // Leave room for driver granularity and ordinary cache admissions.
        Check(after > before && after-before > (256ULL<<20),
              "decode retained the idle large-prefill workspace");
    }
}
static void TestAsyncWithoutRam() {
    for (int variant : {4,7}) for (uint64_t ram : {0ULL,1ULL}) {
        Fixture f(variant,2560,640);
        SetMoeCpuCacheBytes(ram);
        SetMoeCudaCacheBytes(f.bytes*3);
        const auto before = GetDiskMoeCacheStats();
        const auto expected = f.Run(false,1,0,-1,3,false,FLOAT32);
        for (int pass = 0; pass < 6; ++pass) {
            Compare(f.Run(true,1,0,0,3,true,FLOAT32,false,true),expected);
            const auto stats = GetDiskMoeCacheStats();
            Check(stats.cpuBytes<=ram && stats.cudaBytes<=GetMoeCudaCacheBytes() &&
                !stats.cpuCudaOverlapBytes,"DDR-independent promotion budgets");
        }
        const auto warm = GetDiskMoeCacheStats();
        Check(warm.cudaBytes>0 && warm.uploads>before.uploads && warm.cudaHits>before.cudaHits,
            "async GPU admission depends on DDR residency");
        Check(warm.cpuHits-before.cpuHits+warm.cudaHits-before.cudaHits+warm.misses-before.misses==18,
            "DDR-independent route counts");
        std::printf("PASS no-DDR variant=%d ram=%llu uploads=%llu gpu_hits=%llu misses=%llu\n",
            variant,(unsigned long long)ram,(unsigned long long)(warm.uploads-before.uploads),
            (unsigned long long)(warm.cudaHits-before.cudaHits),(unsigned long long)(warm.misses-before.misses));
        Compare(f.Run(true,1,0,0,3,true,FLOAT32,false,true),expected);
        const auto reused = GetDiskMoeCacheStats();
        Check(reused.cudaHits-warm.cudaHits==3 && reused.diskBytes==warm.diskBytes &&
            reused.misses==warm.misses,"DDR-independent GPU reuse reread checkpoint");
    }
    {
        // Host padding can exceed a GPU budget of exactly one payload.
        Fixture f(4,2560,640);
        SetMoeCpuCacheBytes(0);
        SetMoeCudaCacheBytes(f.bytes);
        for (int pass = 0; pass < 4; ++pass)
            Compare(f.Run(true,1,0,0,1,true,FLOAT32,false,true),f.Run(false,1,0,-1,1,false,FLOAT32));
        Check(GetDiskMoeCacheStats().cudaBytes==f.bytes,"exact-payload GPU budget rejected padded source");
    }
    {
        Fixture f(4,2560,640);
        SetMoeCpuCacheBytes(0);SetMoeCudaCacheBytes(f.bytes*3);
        // The first step reaches the admission visit threshold on the next
        // step. Abort that step after a queued layer can retain an upload
        // source; cancellation must release all transient mapped weights.
        const auto expected = f.Run(true,1,0,0,3,true,FLOAT32);
        Data x(FLOAT32,{1,f.hidden}),ids(INT32,{1,3}),scores(FLOAT32,{1,3}),a,b;
        x.Allocate(false);ids.Allocate(false);scores.Allocate(false);
        for (int c = 0; c < f.hidden; ++c) reinterpret_cast<float *>(x.cpuData)[c]=std::sin(c*.71f)*3;
        for (int k = 0; k < 3; ++k) {
            reinterpret_cast<int32_t *>(ids.cpuData)[k]=k==1?2:0;
            reinterpret_cast<float *>(scores.cpuData)[k]=k==0?0:k==1?.375f:-.125f;
        }
        x.ToDevice(CUDA,{0},true);ids.ToDevice(CUDA,{0},true);scores.ToDevice(CUDA,{0},true);
        const auto before = GetDiskMoeCacheStats();
        bool aborted = false;
        try {
            DiskMoeAsyncScope scope({f.disk,f.disk},0,true);
            Check(DiskMoeAsyncScope::TryRun(0,x,ids,scores,a,{}),"queued no-DDR layer rejected");
            DiskMoeAsyncScope::TryRun(1,x,ids,scores,b,[]{throw std::runtime_error("abort staged upload");});
        } catch (const std::runtime_error &error) { aborted = std::string(error.what())=="abort staged upload"; }
        const auto after = GetDiskMoeCacheStats();
        Check(aborted && !after.cpuBytes && !after.cudaBytes,"cancelled no-DDR step admitted a source");
        Check(after.cpuBufferMmaps-before.cpuBufferMmaps==after.cpuBufferMunmaps-before.cpuBufferMunmaps,
            "cancelled no-DDR step retained mapped upload sources");
        for (int pass = 0; pass < 4; ++pass) {
            {
                DiskMoeAsyncScope scope({f.disk,f.disk},0,true);
                Check(DiskMoeAsyncScope::TryRun(0,x,ids,scores,a,{}),"first recovered no-DDR layer rejected");
                Check(DiskMoeAsyncScope::TryRun(1,x,ids,scores,b,{}),"second recovered no-DDR layer rejected");
                scope.Finish();
            }
            for (Data *result : {&a,&b}) {
                result->ToDevice(CPU);
                std::vector<float> actual(f.hidden);
                std::memcpy(actual.data(),result->cpuData,f.hidden*sizeof(float));
                Compare(actual,expected);
            }
        }
        Check(GetDiskMoeCacheStats().cudaBytes>0,"no-DDR cache did not recover after cancellation");
    }
}
static void TestAsyncDiskDecode() {
    {
        // The GPU record view has one quantization layout per layer. Mixed
        // formats within a layer must use the existing per-expert fallback.
        Fixture base(4),gateOther(7),downOther(0);
        SetMoeCpuCacheBytes(base.bytes*6);SetMoeCudaCacheBytes(base.bytes*3);
        Data x(FLOAT32,{1,base.hidden}),ids(INT32,{1,3}),score(FLOAT32,{1,3}),output;
        x.Allocate(0.f);ids.Allocate(false);score.Allocate(0.f);
        std::memset(ids.cpuData,0,ids.GetBytes());
        x.ToDevice(CUDA,{0},true);ids.ToDevice(CUDA,{0},true);score.ToDevice(CUDA,{0},true);
        for(bool gate:{true,false}){
            auto mixed=base.disk;
            mixed[gate?4:5]=gate?gateOther.disk[4]:downOther.disk[5];
            bool accepted;
            {
                DiskMoeAsyncScope scope({mixed},0,true);
                accepted=DiskMoeAsyncScope::TryRun(0,x,ids,score,output,{});
                scope.Finish();
            }
            Check(!accepted,"heterogeneous ordinary GGUF layer accepted");
        }
    }
    // Also exercise the generic per-route CPU fallback (non-Q2_0 down),
    // not only the fused Q2_0 worker used by this checkpoint.
    for (int type = 0; type < 8; ++type) {
        Fixture f(type, type < 4 ? 512 : 2560, type < 4 ? 256 : 640);
        SetMoeCpuCacheBytes(f.bytes*8);SetMoeCudaCacheBytes(f.bytes*4);
        for(int pass=0;pass<4;++pass)for(int topk:{3,10,16}) {
            if(pass==2 && topk==3)SetMoeCudaCacheBytes(f.bytes*2);
            auto before=GetDiskMoeCacheStats();
            Compare(f.Run(true,1,pass,0,topk,true,FLOAT32,false,true),f.Run(false,1,pass,-1,topk,false,FLOAT32));
            auto after=GetDiskMoeCacheStats();
            Check(after.cpuHits-before.cpuHits+after.cudaHits-before.cudaHits+after.misses-before.misses==uint64_t(topk),"async route counts");
            Check(after.cpuBytes<=GetMoeCpuCacheBytes() && after.cudaBytes<=GetMoeCudaCacheBytes() && !after.cpuCudaOverlapBytes,"async cache budgets");
        }
    }
    {
        // Distinct layer mailboxes may refer to the same canonical weights.
        // An admission/eviction must update every alias of a resident record.
        Fixture f(4);SetMoeCpuCacheBytes(f.bytes*6);SetMoeCudaCacheBytes(f.bytes*3);
        const auto expected=f.Run(false,1,0,-1,3,false,FLOAT32);
        for(int pass=0;pass<3;++pass)Compare(f.Run(true,1,0,0,3,true,FLOAT32),expected);
        Data x(FLOAT32,{1,f.hidden}),ids(INT32,{1,3}),scores(FLOAT32,{1,3}),a,b;
        x.Allocate(false);ids.Allocate(false);scores.Allocate(false);
        for(int c=0;c<f.hidden;++c)reinterpret_cast<float *>(x.cpuData)[c]=std::sin(c*.71f)*3;
        for(int k=0;k<3;++k){
            reinterpret_cast<int32_t *>(ids.cpuData)[k]=k==1?2:0;
            reinterpret_cast<float *>(scores.cpuData)[k]=k==0?0:k==1?.375f:-.125f;
        }
        x.ToDevice(CUDA,{0},true);ids.ToDevice(CUDA,{0},true);scores.ToDevice(CUDA,{0},true);
        for(int records:{3,1}){
            SetMoeCudaCacheBytes(f.bytes*records);
            const auto before=GetDiskMoeCacheStats();
            {
                DiskMoeAsyncScope scope({f.disk,f.disk},0,true);
                Check(DiskMoeAsyncScope::TryRun(0,x,ids,scores,a,{}),"first aliased mailbox rejected");
                Check(DiskMoeAsyncScope::TryRun(1,x,ids,scores,b,{}),"second aliased mailbox rejected");
                scope.Finish();
            }
            for(auto *result:{&a,&b}){
                result->ToDevice(CPU);std::vector<float> actual(f.hidden);
                std::memcpy(actual.data(),result->cpuData,f.hidden*4);Compare(actual,expected);
            }
            const auto hits=GetDiskMoeCacheStats().cudaHits-before.cudaHits;
            // Physical trim may retire the whole packed block. Every alias
            // must then see no GPU record, rather than its old freed pointer.
            Check(records==3?hits==6:(!before.cudaBytes?hits==0:(hits==2||hits==4)),
                "aliased GPU record snapshot");
        }
    }
    {
        // A callback can abort after publication but before the GPU waiter.
        // Drain and cancel that scope, then reuse the same model runtime.
        Fixture f(4);SetMoeCpuCacheBytes(f.bytes*6);SetMoeCudaCacheBytes(f.bytes*3);
        Data x(FLOAT32,{1,f.hidden}),ids(INT32,{1,3}),scores(FLOAT32,{1,3}),output;
        x.Allocate(0.f);ids.Allocate(false);scores.Allocate(0.f);
        std::memset(ids.cpuData,0,ids.GetBytes());
        x.ToDevice(CUDA,{0},true);ids.ToDevice(CUDA,{0},true);scores.ToDevice(CUDA,{0},true);
        bool threw=false;
        try {
            DiskMoeAsyncScope scope({f.disk},0,true);
            DiskMoeAsyncScope::TryRun(0,x,ids,scores,output,[]{throw std::runtime_error("test callback abort");});
        } catch(const std::runtime_error &) {threw=true;}
        Check(threw,"async callback exception was swallowed");
        Compare(f.Run(true,1,0,0,3,true,FLOAT32,false,true),f.Run(false,1,0,-1,3,false,FLOAT32));
        {
            DiskMoeAsyncScope scope({f.disk},0,true);
            Check(!DiskMoeAsyncScope::TryRun(-1,x,ids,scores,output,{}),"invalid async layer accepted");
        }
        // Rejected asynchronous dispatch must release the cache before an
        // ordinary synchronous fallback tries to acquire it.
        Compare(f.Run(true,1,0,0,3,true,FLOAT32),f.Run(false,1,0,-1,3,false,FLOAT32));
    }
    std::vector<std::unique_ptr<Fixture>> fixtures;
    std::vector<std::vector<Data *>> tables;
    for(int type:{4,6,7}){fixtures.emplace_back(new Fixture(type,2560,640));tables.push_back(fixtures.back()->disk);}
    const int hidden=fixtures[0]->hidden,topk=10;
    uint64_t bytes=0;for(auto &f:fixtures)bytes+=f->bytes;
    for (uint64_t ram : {bytes*6,uint64_t(0)}) {
        SetMoeCudaCacheBytes(0);
        SetMoeCpuCacheBytes(ram);SetMoeCudaCacheBytes(bytes*3);
        for(int pass=0;pass<5;++pass) {
            std::vector<std::unique_ptr<Data>> inputs,indices,scores,outputs;
            for(size_t layer=0;layer<fixtures.size();++layer) {
                inputs.emplace_back(new Data(FLOAT32,{1,hidden}));indices.emplace_back(new Data(INT32,{1,topk}));
                scores.emplace_back(new Data(FLOAT32,{1,topk}));outputs.emplace_back(new Data(FLOAT32,{1,hidden}));
                auto &x=*inputs.back(),&ids=*indices.back(),&sc=*scores.back();
                x.Allocate(false);ids.Allocate(false);sc.Allocate(false);
                for(int c=0;c<hidden;++c)reinterpret_cast<float *>(x.cpuData)[c]=std::sin(c*.71f)*3;
                for(int k=0;k<topk;++k){
                    reinterpret_cast<int32_t *>(ids.cpuData)[k]=(pass+(k==2?0:k*2))%fixtures[layer]->experts;
                    reinterpret_cast<float *>(sc.cpuData)[k]=k==0?0:k==1?.375f:-.125f;
                }
                x.ToDevice(CUDA,{0},true);ids.ToDevice(CUDA,{0},true);sc.ToDevice(CUDA,{0},true);
            }
            auto before=GetDiskMoeCacheStats();int callbacks=0;
            {
                DiskMoeAsyncScope scope(tables,0,true);
                for(size_t layer=0;layer<fixtures.size();++layer)
                    Check(DiskMoeAsyncScope::TryRun(layer,*inputs[layer],*indices[layer],*scores[layer],*outputs[layer],[&]{++callbacks;}),"multi-layer async rejection");
                scope.Finish();
            }
            Check(callbacks==int(fixtures.size()),"async shared callback count");
            for(size_t layer=0;layer<fixtures.size();++layer){
                outputs[layer]->ToDevice(CPU);std::vector<float> actual(hidden);
                memcpy(actual.data(),outputs[layer]->cpuData,hidden*4);
                Compare(actual,fixtures[layer]->Run(false,1,pass,-1,topk,false,FLOAT32));
            }
            auto after=GetDiskMoeCacheStats();
            Check(after.cpuHits-before.cpuHits+after.cudaHits-before.cudaHits+after.misses-before.misses==fixtures.size()*topk,"queued async routes");
            Check(after.cpuBytes<=GetMoeCpuCacheBytes() && after.cudaBytes<=GetMoeCudaCacheBytes() && !after.cpuCudaOverlapBytes,"queued async budgets");
        }
    }
    {
        // Partial 32-value gate blocks and route/task splits must stay valid
        // when the configured pool changes between model steps.
        Fixture f(4,2560,640);
        SetMoeCpuCacheBytes(f.bytes*8);SetMoeCudaCacheBytes(f.bytes*3);
        for(int threads:{1,3,8,16}){
            SetThreads(threads);
            for(int topk:{3,16})for(int pass=0;pass<2;++pass)
                Compare(f.Run(true,1,pass,0,topk,true,FLOAT32,false,true),
                        f.Run(false,1,pass,-1,topk,false,FLOAT32));
        }
        SetThreads(8);
    }

}
static void TestMixedDiskPlacement() {
    Fixture resident(4), host(4), first(4), second(6);
    for (size_t i=2;i<resident.memory.size();++i)
        resident.memory[i]->ToDevice(CUDA,{0},true);
    Check(DiskMoeLayerIndices({resident.memory}).empty(),"pure CUDA changed disk eligibility");
    Check(DiskMoeLayerIndices({first.disk,host.memory}).empty(),"CPU layer accepted as CUDA resident");
    auto partial=resident.memory;
    partial.back()=first.disk.back();
    Check(DiskMoeLayerIndices({first.disk,partial}).empty(),"partially placed layer accepted");
    partial=first.disk;partial.back()=nullptr;
    Check(DiskMoeLayerIndices({partial,resident.memory}).empty(),"missing expert accepted");
    resident.memory[2]->dataDeviceIds={1};
    Check(DiskMoeLayerIndices({resident.memory,first.disk},0).empty(),"foreign CUDA device accepted");
    resident.memory[2]->dataDeviceIds={0};
    Check(DiskMoeLayerIndices({first.disk,second.disk})==std::vector<int>({0,1}),"all-disk eligibility changed");

    SetMoeCpuCacheBytes((first.bytes+second.bytes)*8);
    SetMoeCudaCacheBytes((first.bytes+second.bytes)*3);
    const int hidden=first.hidden,topk=3;
    // Reuse the same disk runtime with CUDA prefix, middle and suffix layers.
    // Different quantizers make a wrong model-to-mailbox mapping observable.
    for (int residentLayer : {0,1,2}) {
        std::vector<std::vector<Data *>> tables{first.disk,second.disk};
        tables.insert(tables.begin()+residentLayer,resident.memory);
        std::vector<int> expected;
        for(int l=0;l<3;++l)if(l!=residentLayer)expected.push_back(l);
        Check(DiskMoeLayerIndices(tables,0)==expected,"mixed disk graph layer selection");
        for(int pass=0;pass<4;++pass) {
            Data x(FLOAT32,{1,hidden}),ids(INT32,{1,topk}),scores(FLOAT32,{1,topk});
            Data outputs[2],untouched(FLOAT32,{1,hidden}),residentWork(FLOAT32,{1,hidden});
            x.Allocate(false);ids.Allocate(false);scores.Allocate(false);untouched.Allocate(7.f);
            for(int c=0;c<hidden;++c)reinterpret_cast<float *>(x.cpuData)[c]=std::sin(c*.71f)*3;
            for(int k=0;k<topk;++k){
                reinterpret_cast<int32_t *>(ids.cpuData)[k]=(pass+(k==2?0:k*2))%first.experts;
                reinterpret_cast<float *>(scores.cpuData)[k]=k==0?0:k==1?.375f:-.125f;
            }
            x.ToDevice(CUDA,{0},true);ids.ToDevice(CUDA,{0},true);scores.ToDevice(CUDA,{0},true);
            residentWork.Allocate(1.f);residentWork.ToDevice(CUDA,{0},true);
            auto before=GetDiskMoeCacheStats();int callbacks=0;
            {
                DiskMoeAsyncScope scope(tables,0,true);
                int diskLayer=0;
                for(int layer=0;layer<3;++layer) {
                    if(layer==residentLayer) {
                        Check(!DiskMoeAsyncScope::TryRun(layer,x,ids,scores,untouched,[&]{++callbacks;}),
                              "resident layer dispatched to disk mailbox");
                        Check(untouched.dataDevice==CPU && reinterpret_cast<float *>(untouched.cpuData)[0]==7.f,
                              "resident rejection changed output");
                        // CUDA work may sit between two queued disk layers.
                        FastllmCudaMemset0(residentWork.cudaData,residentWork.GetBytes());
                    } else {
                        Check(DiskMoeAsyncScope::TryRun(layer,x,ids,scores,outputs[diskLayer++],[&]{++callbacks;}),
                              "CUDA resident layer disabled disk async dispatch");
                    }
                }
                scope.Finish();
            }
            Check(callbacks==2,"mixed-placement shared callback count");
            Fixture *fixtures[]={&first,&second};
            for(int i=0;i<2;++i) {
                outputs[i].ToDevice(CPU);
                std::vector<float> actual(hidden);
                memcpy(actual.data(),outputs[i].cpuData,hidden*4);
                Compare(actual,fixtures[i]->Run(false,1,pass,-1,topk,false,FLOAT32));
            }
            auto after=GetDiskMoeCacheStats();
            Check(after.cpuHits-before.cpuHits+after.cudaHits-before.cudaHits+after.misses-before.misses==2*topk,
                  "mixed placement counted resident routes as disk");
            Check(after.cpuBytes<=GetMoeCpuCacheBytes() && after.cudaBytes<=GetMoeCudaCacheBytes() &&
                  !after.cpuCudaOverlapBytes,"mixed placement cache budgets");
        }
    }
}

int main(int argc,char **argv){try{
    if(FastllmCudaGetDeviceCount()<1){puts("FASTLLM_TEST_SKIP_NO_DEVICE");return 77;}
    SetThreads(8);
    if(argc==2 && !std::strcmp(argv[1],"--prefill-memory-pressure")) {
        TestPrefillMemoryPressure();puts("PASS: disk prefill memory pressure");return 0;
    }
    if(argc==2 && !std::strcmp(argv[1],"--mixed-disk")) {
        TestMixedDiskPlacement();puts("PASS: mixed CUDA and disk decode");return 0;
    }
    if(argc==2 && !std::strcmp(argv[1],"--promotion-preflight")) {
        TestPromotionPreflight();puts("PASS: fragmented GPU promotion preflight");return 0;
    }
    if(argc==2 && !std::strcmp(argv[1],"--async-disk")) {
        TestAsyncDiskDecode();puts("PASS: async disk decode and queued layers");return 0;
    }
    if(argc==2 && !std::strcmp(argv[1],"--packed-trim")) {
        TestPackedCudaTrim();puts("PASS: packed GPU physical trim");return 0;
    }
    if(argc==2 && !std::strcmp(argv[1],"--async-no-ram")) {
        TestAsyncWithoutRam();puts("PASS: async GPU cache without DDR");return 0;
    }
    TestPrefillMemoryPressure();
    TestPromotionPreflight();
    TestPackedCudaTrim();
    TestAsyncReleasesPrefillWorkspace();
    TestAsyncWithoutRam();
    TestAsyncDiskDecode();
    TestMixedDiskPlacement();
    for (int variant : {4,5,6,7}) {
        Fixture actual(variant,2560,640);
        SetMoeCpuCacheBytes(actual.bytes*8);SetMoeCudaCacheBytes(actual.bytes*4);
        for (int rows : {40,512})
            Compare(actual.Run(true,rows,0,0,3,false,FLOAT32),actual.Run(false,rows,0,-1,3,false,FLOAT32));
    }
    for (int variant : {4,7}) for (int experts : {12,64}) {
        // Cover a >2048-route work split and more groups than the three
        // upload-ring slots. Small fixtures never exercised either lifetime.
        Fixture actual(variant,2560,768,experts);
        SetMoeCpuCacheBytes(actual.bytes*8);SetMoeCudaCacheBytes(actual.bytes*4);
        Compare(actual.Run(true,512,0,0,10,false,FLOAT32),actual.Run(false,512,0,-1,10,false,FLOAT32));
        Compare(actual.Run(true,512,1,0,10,false,FLOAT32,true),actual.Run(false,512,1,-1,10,false,FLOAT32));
    }
    {
        Fixture preload(4);
        // Keep ordinary canonical mode enabled while the tiny GPU budget is
        // too small for a record, as during a RAM-only startup preload.
        SetMoeCpuCacheBytes(preload.bytes*8);SetMoeCudaCacheBytes(1);
        const auto before=GetDiskMoeCacheStats();
        const auto added=PreloadDiskMoeCache({preload.disk});
        const auto after=GetDiskMoeCacheStats();
        Check(added>0 && after.cpuBytes-before.cpuBytes==added,"ordinary RAM preload");
        Check(after.cpuBytes<=GetMoeCpuCacheBytes() && !after.cudaBytes &&
            !after.cpuCudaOverlapBytes,"RAM preload budgets");
        Compare(preload.Run(true,1,0,0,3,false,FLOAT32),preload.Run(false,1,0,-1,3,false,FLOAT32));
        const auto reused=GetDiskMoeCacheStats();
        Check(reused.cpuHits-after.cpuHits==3 && reused.diskBytes==after.diskBytes,
            "preloaded ordinary experts reread checkpoint");
        SetMoeCudaCacheBytes(preload.bytes*4);
        for (int pass=0;pass<3;++pass)
            Compare(preload.Run(true,1,0,0,3,true,FLOAT32),preload.Run(false,1,0,-1,3,false,FLOAT32));
        const auto promoted=GetDiskMoeCacheStats();
        Check(promoted.cudaBytes>0 && !promoted.cpuCudaOverlapBytes,
            "preloaded canonical RAM experts could not enter GPU cache");
    }
    for(int variant=0;variant<5;++variant){
        Fixture f(variant);

        SetMoeCpuCacheBytes(f.bytes*5);SetMoeCudaCacheBytes(f.bytes*3);
        auto start=GetDiskMoeCacheStats();
        for(int device=0;device<std::min(2,FastllmCudaGetDeviceCount());++device)
            for(int rows:{1,1,3,9,40,1,260,1}){
                auto before=GetDiskMoeCacheStats();
                auto want=f.Run(false,rows,0,-1);Compare(f.Run(true,rows,0,device),want);
                auto stats=GetDiskMoeCacheStats();Check(stats.cpuBytes<=GetMoeCpuCacheBytes(),"RAM budget");
                Check(stats.cudaBytes<=GetMoeCudaCacheBytes()*(device+1),"GPU budget");
                Check(stats.cpuCudaOverlapBytes==0,"exclusive RAM/GPU tiers");
                Check(stats.cpuHits-before.cpuHits+stats.cudaHits-before.cudaHits+
                    stats.misses-before.misses==uint64_t(rows*3),"each route counted once");
            }
        auto warm=GetDiskMoeCacheStats();Check(warm.cudaHits>start.cudaHits && warm.cpuHits>start.cpuHits,"mixed cache hits");
        Check(warm.uploads>start.uploads,"GGUF upload");
        for (int rows : {40,260}) {
            const auto before=GetDiskMoeCacheStats();
            Compare(f.Run(true,rows,1,0,3,false,FLOAT32),f.Run(false,rows,1,-1,3,false,FLOAT32));
            const auto after=GetDiskMoeCacheStats();
            Check(after.cpuHits-before.cpuHits+after.cudaHits-before.cudaHits+
                after.misses-before.misses==uint64_t(rows*3),"streamed prefill skipped routes");
            Check(after.cpuBytes<=GetMoeCpuCacheBytes() && after.cudaBytes<=GetMoeCudaCacheBytes() &&
                !after.cpuCudaOverlapBytes,"streamed prefill cache budgets");
        }
        for (int pass=0;pass<12;++pass)
            Compare(f.Run(true,1,pass%4,0,8,true),f.Run(false,1,pass%4,-1,8));
        for (DataType type : {FLOAT32,BFLOAT16})
            for (int offset : {0,1,3})
                Compare(f.Run(true,1,offset,0,8,true,type),f.Run(false,1,offset,-1,8,false,type));
        {
            Data x(FLOAT32,{1,f.hidden}),ids(INT32,{1,1}),scores(FLOAT32,{1,1}),y(FLOAT32,{1,f.hidden});
            x.Allocate(0.f);ids.Allocate(true);scores.Allocate(1.f);y.Allocate(7.f);
            int callbacks=0;DiskMergeMOE op;
            Check(!op.RunGpuGGUFDecode(x,ids,scores,y,f.memory,[&]{++callbacks;}),"accepted non-disk experts");
            Check(!callbacks && y.dataDevice==CPU && reinterpret_cast<float*>(y.cpuData)[0]==7.f,"rejected handoff modified output");
        }
        Check(GetDiskMoeCacheStats().decodeOutputH2DCopies>warm.decodeOutputH2DCopies,"pinned ordinary output upload");
        for(int offset=1;offset<8;++offset) Compare(f.Run(true,3,offset,0),f.Run(false,3,offset,-1));
        SetMoeCpuCacheBytes(0);Compare(f.Run(true,1,0,0),f.Run(false,1,0,-1));
        Check(GetDiskMoeCacheStats().cpuBytes==0,"zero RAM budget");
        SetMoeCudaCacheBytes(0);
        Compare(f.Run(true,1,0,0),f.Run(false,1,0,-1));
        SetMoeCpuCacheBytes(f.bytes*8);
        for(int rows:{1,5,17}) Compare(f.Run(true,rows,1,0),f.Run(false,rows,1,-1));
        SetMoeCudaCacheBytes(f.bytes*8);
        for(int repeat=0;repeat<5;++repeat)
            for(int topk:{8,21}) Compare(f.Run(true,5,0,0,topk),f.Run(false,5,0,-1,topk));

    }
    auto s=GetDiskMoeCacheStats();Check(s.cpuBytes==0 && s.cudaBytes==0,"unload");
    puts("PASS: disk Qwen GGUF cache; IQ2 pairs, CPU/GPU, arbitrary rows, eviction, budgets, unload");return 0;
}catch(const std::exception&e){fprintf(stderr,"FAIL: %s\n",e.what());return 1;}}
