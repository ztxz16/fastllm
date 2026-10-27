#include "fastllm.h"
#include "executor.h"
#include "utils.h"
#include "gguf.h"
#include "devices/disk/diskdevice.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <unistd.h>
using namespace fastllm;
namespace fastllm { void RegisterNumas(Data *, std::string); }
static void Check(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
static std::string currentCase;
struct Fixture {
    static constexpr int hidden=512, inter=256, experts=12;
    std::string path;
    std::vector<std::unique_ptr<Data>> owned;
    std::vector<Data *> memory, disk;
    uint64_t bytes=0;
    Fixture(int variant) : memory(2*(experts+1)), disk(memory.size()) {
        char name[]="/tmp/fastllm-disk-gguf-XXXXXX";
        int fd=mkstemp(name); Check(fd>=0,"tempfile"); path=name;
        FILE *file=fdopen(fd,"wb"); std::mt19937 rng(712+variant);
        for(int e=0;e<experts;++e) for(int part=0;part<2;++part) {
            const auto type=part ? (variant%2 ? GGML_TYPE_IQ4_XS : GGML_TYPE_IQ3_XXS) :
                                  (variant<2 ? GGML_TYPE_IQ2_XXS : GGML_TYPE_IQ2_S);
            auto dims=part ? std::vector<int>{hidden,inter} : std::vector<int>{2*inter,hidden};
            auto w=std::make_unique<Data>(DATA_GGUF_FORMAT,type,dims);
            w->disableGGUFRepack=true; w->isGGUFData=true; w->Allocate(false);
            for(size_t i=0;i<w->GetBytes();++i) w->cpuData[i]=rng()>>24;
            for(size_t i=0;i<w->GetBytes();i+=ggml_type_size(type)) {
                uint16_t scale=float_to_half(std::ldexp(1.f,-12+int(rng()%3)));
                memcpy(w->cpuData+i,&scale,2);
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
        for(size_t i=2;i<memory.size();++i) {
            memory[i]->disableGGUFRepack=false;
            RegisterNumas(memory[i],i%2 ? "linearColumn" : "linearSwiglu");
        }
    }
    ~Fixture(){owned.clear();unlink(path.c_str());}
    void CheckBatchedCuda(int device) {
        FastllmCudaSetDevice(device);
        Data records(INT8,{int(bytes*experts)});records.Allocate(false);
        FILE *file=fopen(path.c_str(),"rb");Check(file!=nullptr,"batch records");
        for(int e=0;e<experts;++e)for(int part=0;part<2;++part){
            const auto &p=disk[2+2*e+part]->diskWeightParts[0];
            Check(fseek(file,p.fileOffset,SEEK_SET)==0,"batch seek");
            auto *dst=records.cpuData+e*bytes+(part?disk[2]->GetBytes():0);
            Check(fread(dst,1,p.bytes,file)==p.bytes,"batch read");
        }
        fclose(file);records.ToDevice(CUDA,{device},true);
        for(int rows:{1,3,9,33})for(int topk:{1,3,16}){
            Data input(BFLOAT16,{rows,hidden}),slots(INT32,{rows,topk}),scores(FLOAT32,{rows,topk});
            input.Allocate(false);slots.Allocate(false);scores.Allocate(false);
            for(int i=0;i<rows*hidden;++i)reinterpret_cast<uint16_t*>(input.cpuData)[i]=
                Float32ToBFloat16RNEBits(std::sin(i*.17f)*2);
            for(int i=0;i<rows*topk;++i){
                reinterpret_cast<int32_t*>(slots.cpuData)[i]=i%7==0?-1:i%experts;
                reinterpret_cast<float*>(scores.cpuData)[i]=i%3==0?0:i%3==1?.375f:-.125f;
            }
            input.ToDevice(CUDA,{device},true);slots.ToDevice(CUDA,{device},true);scores.ToDevice(CUDA,{device},true);
            Data workspace(FLOAT32,{rows,hidden+topk*inter}),out(FLOAT32,{rows,topk,hidden}),activation;
            workspace.ToDevice(CUDA,{device},false);workspace.Allocate(false);
            out.ToDevice(CUDA,{device},false);out.Allocate(false);
            FastllmCudaMoeGGUFCacheView view{static_cast<const uint8_t*>(records.cudaData),
                static_cast<const int32_t*>(slots.cudaData),bytes,disk[2]->GetBytes(),
                disk[2]->ggmlType,disk[3]->ggmlType,hidden,inter,workspace.cudaData,workspace.GetBytes()};
            Check(FastllmCudaMoeGlm5GGUFCacheCompute(input,activation,view,
                static_cast<const float*>(scores.cudaData),topk,.125f,static_cast<float*>(out.cudaData)),"batch compute");
            out.ToDevice(CPU);std::vector<uint8_t> expected(out.cpuData,out.cpuData+out.GetBytes());
            out.ToDevice(CUDA,{device},true);
            for(int row=0;row<rows;++row){
                Data x(BFLOAT16,{1,hidden},CUDA,static_cast<uint8_t*>(input.cudaData)+size_t(row)*hidden*2);
                x.cudaDataBorrowed=true;x.dataDeviceIds={device};
                auto one=view;one.routeSlots+=row*topk;
                Check(FastllmCudaMoeGlm5GGUFCacheCompute(x,activation,one,
                    static_cast<const float*>(scores.cudaData)+row*topk,topk,.125f,
                    static_cast<float*>(out.cudaData)+size_t(row)*topk*hidden),"row compute");
            }
            out.ToDevice(CPU);Check(memcmp(expected.data(),out.cpuData,expected.size())==0,
                "batched GGUF output differs from individual rows");
        }
    }
    std::vector<float> Run(bool lazy,int rows,int offset,int device,int topk=3) {
        currentCase = "gate=" + std::to_string(disk[2]->ggmlType) + " down=" + std::to_string(disk[3]->ggmlType) +
            " rows=" + std::to_string(rows) + " topk=" + std::to_string(topk) + " offset=" + std::to_string(offset) +
            " device=" + std::to_string(device);
        Data x(BFLOAT16,{rows,hidden}), y(BFLOAT16,{rows,hidden});
        Data ids(INT32PARAM,{rows,topk}), scores(FLOAT32,{rows,topk}),a,b,c;
        x.Allocate(false);ids.Allocate(false);scores.Allocate(false);
        for(int r=0;r<rows;++r) {
            for(int j=0;j<hidden;++j) ((uint16_t*)x.cpuData)[r*hidden+j]=Float32ToBFloat16RNEBits(std::sin((r*hidden+j)*.71f)*3);
            for(int k=0;k<topk;++k) {
                ((int32_t*)ids.cpuData)[r*topk+k]=(offset+(k==2 && r%3==0?0:k*2)+r%2)%experts;
                ((float*)scores.cpuData)[r*topk+k]=k==0?0:k==1?.375f:-.125f;
            }
        }
        auto &table=lazy?disk:memory;std::vector<Data*> biases(table.size());
        DataDict data={{"input",&x},{"output",&y},{"index",&ids},{"score",&scores},
            {"weights",(Data*)table.data()},{"biass",(Data*)biases.data()},{"w1",&a},{"w2",&b},{"w3",&c}};
        IntDict ip={{"weights___batch",int(table.size())},{"biass___batch",int(table.size())},
            {"deepSeekV4Mode",1},{"activationQuantBlock",128}};
        FloatDict fp={{"sharedScale",0},{"swigluLimit",.125f}};
        if(lazy) {
            if(device>=0){FastllmCudaSetDevice(device);x.ToDevice(CUDA,{device},true);x.ToDevice(CPU);}
            DiskMergeMOE op;op.Run("MergeMOE",data,fp,ip);
        } else {
            // GLM decode arithmetic per row remains the reference for verify/prefill.
            y.Allocate(false);
            for(int r=0;r<rows;++r){
                Data xr(BFLOAT16,{1,hidden},CPU,x.cpuData+size_t(r)*hidden*2),yr(BFLOAT16,{1,hidden});
                Data ir(INT32PARAM,{1,topk},CPU,ids.cpuData+r*topk*4),sr(FLOAT32,{1,topk},CPU,scores.cpuData+r*topk*4);
                auto one=data;one["input"]=&xr;one["output"]=&yr;one["index"]=&ir;one["score"]=&sr;
                Data curInput, curOutput;
                one["curInput"]=&curInput;one["curOutput"]=&curOutput;
                static_cast<Executor *>(GetExecutor())->RunOnDevice("numa","MergeMOE",one,fp,ip);
                memcpy(y.cpuData+size_t(r)*hidden*2,yr.cpuData,hidden*2);
            }
        }
        y.ToDevice(CPU);std::vector<float> out(rows*hidden);
        for(size_t j=0;j<out.size();++j) out[j]=BFloat16BitsToFloat32(((uint16_t*)y.cpuData)[j]);
        return out;
    }
};
static void Compare(const std::vector<float>&a,const std::vector<float>&b) {
    double err=0,norm=0;
    for(size_t i=0;i<a.size();++i){Check(std::isfinite(a[i]),"finite");err+=(a[i]-b[i])*(a[i]-b[i]);norm+=b[i]*b[i];}
    double rel=std::sqrt(err/std::max(norm,1e-20));
    if(rel>.008){fprintf(stderr,"%s relative L2 %.8g\n",currentCase.c_str(),rel);throw std::runtime_error("GGUF disk numerics");}
}
int main(){try{
    if(FastllmCudaGetDeviceCount()<1){puts("FASTLLM_TEST_SKIP_NO_DEVICE");return 77;}
    SetThreads(8);
    for(int variant=0;variant<4;++variant){
        Fixture f(variant);
        for(int device=0;device<std::min(2,FastllmCudaGetDeviceCount());++device)f.CheckBatchedCuda(device);
        SetMoeCpuCacheBytes(f.bytes*5);SetMoeCudaCacheBytes(f.bytes*3);
        auto start=GetDiskMoeCacheStats();
        for(int device=0;device<std::min(2,FastllmCudaGetDeviceCount());++device)
            for(int rows:{1,1,3,9,40,1,260,1}){
                auto want=f.Run(false,rows,0,-1);Compare(f.Run(true,rows,0,device),want);
                auto stats=GetDiskMoeCacheStats();Check(stats.cpuBytes<=GetMoeCpuCacheBytes(),"RAM budget");
                Check(stats.cudaBytes<=GetMoeCudaCacheBytes()*(device+1),"GPU budget");
            }
        auto warm=GetDiskMoeCacheStats();Check(warm.cudaHits>start.cudaHits && warm.cpuHits>start.cpuHits,"mixed cache hits");
        Check(warm.uploads>start.uploads,"GGUF upload");
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
        if(FastllmCudaGetDeviceCount()>=2){
            SetMoeCpuCacheBytes(0);SetMoeCudaCacheBytes(0);TrimDiskMoeCache();
            SetMoeCpuCacheBytes(f.bytes*3);SetMoeCudaCacheBytes(f.bytes*4);
            DiskMoeCudaDeviceScope devices({0,1});
            for(int repeat=0;repeat<24;++repeat){
                int offset=repeat%12;
                Compare(f.Run(true,3,offset,0,8),f.Run(false,3,offset,-1,8));
            }
            auto before=GetDiskMoeCacheStats();
            Check(before.cudaBytes>GetMoeCudaCacheBytes(),"TP uses both GPU caches");
            for(int rows:{1,3,9,40}){
                Compare(f.Run(true,rows,0,0,21),f.Run(false,rows,0,-1,21));
                auto after=GetDiskMoeCacheStats();
                Check(after.cudaBytes<=2*GetMoeCudaCacheBytes(),"TP GPU budgets");
                Check(after.cpuBytes<=GetMoeCpuCacheBytes(),"TP shared RAM budget");
                Check((after.cudaHits-before.cudaHits)+(after.cpuHits-before.cpuHits)+
                    (after.misses-before.misses)==uint64_t(rows*21),"TP counts each route once");
                before=after;
            }
            // Cross multiple global heat-decay epochs while exercising GPU
            // admissions and removals. Expected values use the NUMA backend.
            std::vector<std::vector<float>> reference;
            for(int offset=0;offset<12;++offset) reference.push_back(f.Run(false,1,offset,-1,8));
            for(int repeat=0;repeat<900;++repeat)
                Compare(f.Run(true,1,repeat%12,0,8),reference[repeat%12]);
        }
    }
    auto s=GetDiskMoeCacheStats();Check(s.cpuBytes==0 && s.cudaBytes==0,"unload");
    puts("PASS: disk GLM GGUF cache; four IQ pairs, CPU/GPU, arbitrary rows, eviction, budgets, unload");return 0;
}catch(const std::exception&e){fprintf(stderr,"FAIL: %s\n",e.what());return 1;}}
